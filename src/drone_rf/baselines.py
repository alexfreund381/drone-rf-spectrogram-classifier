"""Reference baselines that use no learned image features.

* majority: always predict the most frequent training class.
* brightness: logistic regression on ONE number, the mean pixel value.
* energy: logistic regression on global intensity statistics plus coarse row/column energy
  profiles (39 numbers).

If these do about as well as the CNN, the classes can be separated by overall brightness or
noise-floor differences, which points at the recording-environment confound rather than
drone-signal structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import SpectrogramDataset, filter_rows
from .evaluate import full_report, save_report
from .manifest import read_manifest
from .splits import assert_group_safe, split_hash

PROFILE_BINS = 16
BASELINES = ("majority", "brightness", "energy")


def energy_features(arrays: np.ndarray) -> np.ndarray:
    """uint8 ``[N, H, W]`` -> float ``[N, 39]``: 7 global statistics + 16 row + 16 column means."""
    x = arrays.astype(np.float32) / 255.0
    n = len(x)
    flat = x.reshape(n, -1)
    stats = np.stack(
        [
            flat.mean(1),
            flat.std(1),
            np.percentile(flat, 5, axis=1),
            np.percentile(flat, 50, axis=1),
            np.percentile(flat, 95, axis=1),
            np.percentile(flat, 99, axis=1),
            flat.max(1),
        ],
        axis=1,
    )

    def profile(axis: int) -> np.ndarray:
        prof = x.mean(axis=axis)  # [N, L]
        edges = np.linspace(0, prof.shape[1], PROFILE_BINS + 1).astype(int)
        return np.stack([prof[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:], strict=True)], axis=1)

    return np.concatenate([stats, profile(2), profile(1)], axis=1)


class MajorityBaseline:
    def fit(self, features: np.ndarray, y: np.ndarray) -> MajorityBaseline:
        counts = np.bincount(y, minlength=2)
        self.label_ = int(counts.argmax())
        self.p_drone_ = float(counts[1] / max(counts.sum(), 1))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        p = np.full(len(features), float(self.label_))
        return np.stack([1 - p, p], axis=1)


def make_baseline(name: str):
    if name == "majority":
        return MajorityBaseline()
    if name in ("brightness", "energy"):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"unknown baseline: {name}")


def baseline_features(name: str, arrays: np.ndarray) -> np.ndarray:
    feats = energy_features(arrays)
    return feats[:, :1] if name == "brightness" else feats


def run_baselines(config: dict, split: str = "val", n_boot: int = 1000) -> dict[str, dict]:
    """Fit every baseline on the training split and evaluate on ``split``.

    Uses the same splits file, image directory, input size and SNR filters as the CNN config.
    Results go to ``<output_dir>/a0_<baseline>/metrics_<split>.json``.
    """
    from .train import merge_config

    cfg = merge_config(config)
    dcfg = cfg["data"]
    splits = read_manifest(dcfg["splits_csv"])
    assert_group_safe(splits)
    common = {"image_dir": dcfg["image_dir"], "require_files": dcfg["require_files"]}
    train_ds = SpectrogramDataset(filter_rows(splits, "train", dcfg["train_snr"], **common), dcfg["image_dir"], dcfg["size"])
    eval_ds = SpectrogramDataset(filter_rows(splits, split, dcfg["eval_snr"], **common), dcfg["image_dir"], dcfg["size"])
    results: dict[str, dict] = {}
    for name in BASELINES:
        model = make_baseline(name).fit(baseline_features(name, train_ds.arrays()), train_ds.labels)
        p = model.predict_proba(baseline_features(name, eval_ds.arrays()))[:, 1]
        pred: pd.DataFrame = eval_ds.frame.copy()
        pred["y_true"], pred["p_drone"], pred["y_pred"] = eval_ds.labels, p, (p >= 0.5).astype(int)
        report = full_report(pred, n_boot=n_boot)
        report.update({"split": split, "baseline": name, "split_hash": split_hash(splits)})
        run_dir = Path(cfg["output_dir"]) / f"a0_{name}"
        save_report(report, pred, run_dir, split)
        meta = {
            "run_name": f"a0_{name}",
            "model": {"name": f"baseline_{name}"},
            "pretrained_weights": "n/a",
            "split_hash": split_hash(splits),
            "train_images": len(train_ds),
            "train_snr": dcfg["train_snr"],
            "input_size": dcfg["size"],
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        results[name] = report
    return results
