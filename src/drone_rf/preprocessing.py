"""Image loading, resizing, train-only normalization statistics and augmentations.

Images leave this module as float32 tensors ``[1, H, W]`` scaled to [0, 1]. Standardization
(mean/std) happens inside the model so a checkpoint carries its own normalization.

Augmentations are deliberately limited to operations with an RF interpretation. We call the
image axes "x" and "y" because the dataset README does not say which one is time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_gray_uint8(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load an image as 8-bit grayscale ``[H, W]``; optionally resize to ``size x size``.

    Bilinear resampling with PIL's built-in antialiasing (``reducing_gap``) so thin bursts are
    averaged into neighbouring pixels rather than skipped.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None and im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR, reducing_gap=2.0)
        return np.asarray(im, dtype=np.uint8)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 ``[H, W]`` -> float32 ``[1, H, W]`` in [0, 1]."""
    return torch.from_numpy(np.array(arr, dtype=np.uint8)).float().div_(255.0).unsqueeze(0)


def compute_mean_std(arrays: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of pixel values in [0, 1]. Call with TRAINING images only."""
    x = arrays.astype(np.float64) / 255.0
    return float(x.mean()), float(max(x.std(), 1e-6))


@dataclass
class AugmentConfig:
    """Training-only augmentations. All default to off."""

    gain: float = 0.0  # +/- brightness offset as a fraction of full scale (receiver gain change)
    mask_x: int = 0  # max width in pixels of one blanked vertical band
    mask_y: int = 0  # max height in pixels of one blanked horizontal band
    shift: int = 0  # max circular shift in pixels along each axis
    pixel_noise: float = 0.0  # std of PIXEL-domain Gaussian noise (not RF AWGN)

    @property
    def enabled(self) -> bool:
        return any((self.gain, self.mask_x, self.mask_y, self.shift, self.pixel_noise))


def augment(x: torch.Tensor, cfg: AugmentConfig, rng: torch.Generator) -> torch.Tensor:
    """Apply the configured augmentations to one ``[1, H, W]`` tensor in [0, 1]."""
    if not cfg.enabled:
        return x
    x = x.clone()
    _, h, w = x.shape

    def randint(high: int) -> int:
        return int(torch.randint(0, high + 1, (1,), generator=rng).item())

    if cfg.shift:
        x = torch.roll(x, (randint(2 * cfg.shift) - cfg.shift, randint(2 * cfg.shift) - cfg.shift), (1, 2))
    if cfg.gain:
        x = x + (torch.rand(1, generator=rng).item() * 2 - 1) * cfg.gain
    if cfg.pixel_noise:
        x = x + torch.randn(x.shape, generator=rng) * cfg.pixel_noise
    fill = float(x.mean())
    if cfg.mask_x:
        width = randint(cfg.mask_x)
        start = randint(w - width)
        x[:, :, start : start + width] = fill
    if cfg.mask_y:
        height = randint(cfg.mask_y)
        start = randint(h - height)
        x[:, start : start + height, :] = fill
    return x.clamp_(0.0, 1.0)
