"""Checks for the recording-environment confound.

Drone signals were recorded in an anechoic chamber and background in a university building, so
a classifier may separate the classes by overall image appearance instead of drone bursts.
Two measurements help judge that:

* ``intensity_by_class_and_snr``: do the classes differ in plain brightness at each SNR level?
* ``masked_box_eval``: paint over every annotated burst in drone images with noise-like pixels
  taken from the same image. If the model still answers "drone", it is not relying on the bursts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .annotations import parse_label_text
from .dataset import SpectrogramDataset, filter_rows
from .evaluate import predict
from .manifest import read_manifest


def intensity_by_class_and_snr(frame: pd.DataFrame, image_dir: str | Path, size: int = 224) -> pd.DataFrame:
    """Mean pixel value (0-255 scale) per binary class and SNR level, with the spread across images."""
    ds = SpectrogramDataset(frame, image_dir, size)
    arrays = ds.arrays().astype(np.float32)
    df = ds.frame[["snr_db", "binary_label", "class_tag"]].copy()
    df["image_mean"] = arrays.mean(axis=(1, 2))
    df["image_std"] = arrays.std(axis=(1, 2))
    out = (
        df.groupby(["snr_db", "binary_label"])
        .agg(
            n_images=("image_mean", "size"),
            mean_of_image_means=("image_mean", "mean"),
            std_of_image_means=("image_mean", "std"),
            mean_of_image_stds=("image_std", "mean"),
        )
        .reset_index()
    )
    return out


def mask_boxes(arr: np.ndarray, label_text: str, margin: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Replace every annotated box (plus a margin) with pixels drawn at random from the rest of
    the same image. Returns a copy; the input is unchanged."""
    rng = rng or np.random.default_rng(0)
    h, w = arr.shape
    hidden = np.zeros((h, w), dtype=bool)
    for b in parse_label_text(label_text).boxes:
        x0 = int(np.floor((b.x_center - b.width / 2) * w)) - margin
        x1 = int(np.ceil((b.x_center + b.width / 2) * w)) + margin
        y0 = int(np.floor((b.y_center - b.height / 2) * h)) - margin
        y1 = int(np.ceil((b.y_center + b.height / 2) * h)) + margin
        hidden[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)] = True
    out = arr.copy()
    visible = arr[~hidden]
    if hidden.any() and visible.size:
        out[hidden] = rng.choice(visible, size=int(hidden.sum()))
    return out


def masked_box_eval(
    checkpoint: str | Path,
    label_text_csv: str | Path,
    split: str = "val",
    snr_levels: list[int] | None = None,
    all_snr: bool = False,
    seed: int = 0,
) -> dict:
    """Drone recall on original images versus the same images with all annotated bursts hidden.

    Also reports how much of the image area was hidden, and the false-positive rate on
    background images as the reference for "what the model does when no burst is visible".
    """
    from .train import load_checkpoint, pick_device

    device = pick_device("auto")
    model, ckpt = load_checkpoint(checkpoint, device)
    dcfg = ckpt["config"]["data"]
    splits = read_manifest(dcfg["splits_csv"])
    snr = None if all_snr else snr_levels if snr_levels is not None else dcfg["eval_snr"]
    df = filter_rows(splits, split, snr, dcfg["image_dir"], dcfg["require_files"])
    texts = pd.read_csv(label_text_csv, keep_default_na=False).set_index("filename_stem").label_text
    drone = df[df.binary_label == "1"].reset_index(drop=True)
    background = df[df.binary_label == "0"].reset_index(drop=True)

    ds = SpectrogramDataset(drone, dcfg["image_dir"], dcfg["size"])
    original = predict(model, ds, 64, device)
    rng = np.random.default_rng(seed)
    arrays = ds.arrays()
    masked = np.stack([mask_boxes(a, texts[s], rng=rng) for a, s in zip(arrays, ds.frame.filename_stem, strict=True)])
    hidden_fraction = float((masked != arrays).mean())
    ds._cache = masked
    after = predict(model, ds, 64, device)
    bg = predict(model, SpectrogramDataset(background, dcfg["image_dir"], dcfg["size"]), 64, device)

    per_snr = (
        pd.DataFrame({"snr_db": original.snr_db, "orig": original.y_pred, "masked": after.y_pred})
        .groupby("snr_db")
        .agg(n_drone_images=("orig", "size"), recall_original=("orig", "mean"), recall_bursts_hidden=("masked", "mean"))
        .reset_index()
    )
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "snr_levels": "all" if snr is None else list(snr),
        "n_drone_images": int(len(drone)),
        "n_drone_groups": int(drone.split_group_id.nunique()),
        "n_background_images": int(len(background)),
        "approx_fraction_of_pixels_changed": hidden_fraction,
        "drone_recall_original": float(original.y_pred.mean()) if len(original) else float("nan"),
        "drone_recall_bursts_hidden": float(after.y_pred.mean()) if len(after) else float("nan"),
        "mean_p_drone_original": float(original.p_drone.mean()) if len(original) else float("nan"),
        "mean_p_drone_bursts_hidden": float(after.p_drone.mean()) if len(after) else float("nan"),
        "background_false_positive_rate": float(bg.y_pred.mean()) if len(bg) else float("nan"),
        "per_snr": per_snr.to_dict(orient="records"),
    }
