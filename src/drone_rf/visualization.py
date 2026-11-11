"""Figures generated from real run outputs. Every figure function takes a ``note`` that is
printed on the figure (use it for provenance, e.g. "SYNTHETIC" or a split hash).

Axes of spectrogram images are labelled "image x / image y" because the dataset README does
not state which axis is time.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .dataset import resolve_image_path  # noqa: E402
from .preprocessing import load_gray_uint8  # noqa: E402
from .sources import CLASS_NAMES  # noqa: E402


def _finish(fig, out: str | Path, note: str = "") -> Path:
    if note:
        fig.text(0.995, -0.01, note, ha="right", va="top", fontsize=7, color="gray")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(run_dir: str | Path, out: str | Path, note: str = "") -> Path:
    rows = [json.loads(line) for line in (Path(run_dir) / "epochs.jsonl").read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(df.epoch, df.train_loss, label="train loss")
    a.plot(df.epoch, df.val_loss, label="validation loss")
    a.set_xlabel("epoch"), a.set_ylabel("cross-entropy loss"), a.legend()
    b.plot(df.epoch, df.train_acc, label="train accuracy")
    b.plot(df.epoch, df.val_acc, label="validation accuracy")
    b.plot(df.epoch, df.val_macro_f1, label="validation macro-F1", linestyle="--")
    b.set_xlabel("epoch"), b.legend()
    fig.suptitle(Path(run_dir).name)
    return _finish(fig, out, note)


def plot_confusion(matrix, out: str | Path, title: str = "", note: str = "") -> Path:
    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES), ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    return _finish(fig, out, note)


def plot_metric_by_snr(
    reports: dict[str, dict], out: str | Path, metric: str = "accuracy", note: str = ""
) -> Path:
    """One line per run. The image count per SNR level is printed under the axis."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    counts = None
    for name, report in reports.items():
        df = pd.DataFrame(report["per_snr"]).sort_values("snr_db")
        ax.plot(df.snr_db, df[metric], marker="o", markersize=3, label=name)
        counts = df[["snr_db", "n"]]
    ax.set_xlabel("SNR of the AWGN variant (dB)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(alpha=0.3), ax.legend()
    if counts is not None and len(counts):
        ax.set_title(f"images per SNR level: {int(counts.n.min())}–{int(counts.n.max())}", fontsize=9)
    return _finish(fig, out, note)


def plot_gallery(
    frame: pd.DataFrame,
    image_dir: str | Path,
    out: str | Path,
    titles: list[str] | None = None,
    cols: int = 6,
    size: int | None = None,
    note: str = "",
) -> Path:
    """Grid of spectrograms for the rows in ``frame``."""
    n = len(frame)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, stem in enumerate(frame.filename_stem):
        ax = axes[k // cols][k % cols]
        ax.imshow(load_gray_uint8(resolve_image_path(image_dir, stem), size), cmap="gray", vmin=0, vmax=255)
        ax.set_title(titles[k] if titles else stem.replace("IQdata_", ""), fontsize=7)
    return _finish(fig, out, note)


def plot_error_gallery(pred: pd.DataFrame, image_dir: str | Path, out: str | Path, per_kind: int = 6, note: str = ""):
    """Most confident false positives and false negatives, plus a few true positives."""
    fp = pred[(pred.y_true == 0) & (pred.y_pred == 1)].sort_values("p_drone", ascending=False).head(per_kind)
    fn = pred[(pred.y_true == 1) & (pred.y_pred == 0)].sort_values("p_drone").head(per_kind)
    tp = pred[(pred.y_true == 1) & (pred.y_pred == 1)].sort_values("p_drone", ascending=False).head(per_kind)
    parts, titles = [], []
    for kind, part in (("FP", fp), ("FN", fn), ("TP", tp)):
        parts.append(part)
        titles += [
            f"{kind} s{r.source_sample_id} {r.class_tag}\nSNR {r.snr_db} dB  p={r.p_drone:.2f}"
            for r in part.itertuples()
        ]
    frame = pd.concat(parts)
    if not len(frame):
        return None
    return plot_gallery(frame, image_dir, out, titles, cols=per_kind, note=note)
