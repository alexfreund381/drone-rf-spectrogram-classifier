"""Lazy PyTorch dataset over rows of the splits file.

Labels come from the manifest's ``binary_label`` column only. The model never receives the
filename, tag, SNR or box count; those stay in ``self.frame`` for analysis.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import AugmentConfig, augment, compute_mean_std, load_gray_uint8, to_tensor


def resolve_image_path(image_dir: str | Path, stem: str) -> Path:
    """Images live either directly in ``image_dir`` or in ``image_dir/images``."""
    root = Path(image_dir)
    direct = root / f"{stem}.png"
    return direct if direct.exists() else root / "images" / f"{stem}.png"


def filter_rows(
    splits: pd.DataFrame,
    split: str,
    snr_levels: list[int] | None = None,
    image_dir: str | Path | None = None,
    require_files: bool = False,
) -> pd.DataFrame:
    """Rows of one split with a usable binary label, optionally restricted to SNR levels and
    to images that exist on disk."""
    df = splits[(splits.split == split) & splits.binary_label.isin(["0", "1"])]
    if snr_levels is not None:
        df = df[df.snr_db.isin(snr_levels)]
    if require_files and image_dir is not None:
        exists = [resolve_image_path(image_dir, s).exists() for s in df.filename_stem]
        df = df[exists]
    return df.reset_index(drop=True)


class SpectrogramDataset(Dataset):
    """Returns ``(image [1,H,W] float32 in [0,1], label int64, row_index)``."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: str | Path,
        size: int = 224,
        augment_cfg: AugmentConfig | None = None,
        seed: int = 0,
        preload: bool = True,
    ):
        if not frame.binary_label.isin(["0", "1"]).all():
            raise ValueError("dataset rows must have binary_label 0 or 1 (no quarantined rows)")
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.size = size
        self.augment_cfg = augment_cfg
        self.labels = self.frame.binary_label.astype(int).to_numpy()
        self._rng = torch.Generator().manual_seed(seed)
        self._cache: np.ndarray | None = None
        if preload:
            self.preload()

    def _load(self, i: int) -> np.ndarray:
        return load_gray_uint8(resolve_image_path(self.image_dir, self.frame.filename_stem[i]), self.size)

    def preload(self) -> None:
        """Decode and resize every image once into RAM (N x size x size bytes)."""
        if self._cache is None and len(self.frame):
            with ThreadPoolExecutor(max_workers=8) as pool:
                self._cache = np.stack(list(pool.map(self._load, range(len(self.frame)))))

    def arrays(self) -> np.ndarray:
        """All images as uint8 ``[N, H, W]`` (preloads if needed)."""
        self.preload()
        assert self._cache is not None
        return self._cache

    def mean_std(self) -> tuple[float, float]:
        """Pixel mean/std of THIS dataset. Only ever call on the training dataset."""
        return compute_mean_std(self.arrays())

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        arr = self._cache[i] if self._cache is not None else self._load(i)
        x = to_tensor(arr)
        if self.augment_cfg is not None:
            x = augment(x, self.augment_cfg, self._rng)
        return x, int(self.labels[i]), i
