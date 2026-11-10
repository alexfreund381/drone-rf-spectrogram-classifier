"""Metrics and evaluation slices.

Class order is fixed everywhere: index 0 = background_rf (negative), 1 = drone_rf (positive).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .sources import CLASS_NAMES


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """TP/FP/FN/TN with drone_rf (1) as the positive class."""
    t, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tp": int(((t == 1) & (p == 1)).sum()),
        "fp": int(((t == 0) & (p == 1)).sum()),
        "fn": int(((t == 1) & (p == 0)).sum()),
        "tn": int(((t == 0) & (p == 0)).sum()),
    }


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def _f1(precision: float, recall: float) -> float:
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        return 0.0 if not (np.isnan(precision) and np.isnan(recall)) else float("nan")
    return 2 * precision * recall / (precision + recall)


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    """All headline metrics from hard predictions. Undefined ratios are NaN (never silently 0)."""
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    prec_d, rec_d = _div(tp, tp + fp), _div(tp, tp + fn)
    prec_b, rec_b = _div(tn, tn + fn), _div(tn, tn + fp)
    f1_d, f1_b = _f1(prec_d, rec_d), _f1(prec_b, rec_b)
    return {
        **c,
        "n": n,
        "support_drone": tp + fn,
        "support_background": tn + fp,
        "accuracy": _div(tp + tn, n),
        "balanced_accuracy": float(np.nanmean([rec_d, rec_b])) if n else float("nan"),
        "precision_drone": prec_d,
        "recall_drone": rec_d,
        "f1_drone": f1_d,
        "precision_background": prec_b,
        "recall_background": rec_b,
        "f1_background": f1_b,
        "macro_f1": float(np.nanmean([f1_d, f1_b])) if n else float("nan"),
        "false_positive_rate": _div(fp, fp + tn),
    }


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """2x2 matrix, rows = true, columns = predicted, order [background_rf, drone_rf]."""
    c = confusion_counts(y_true, y_pred)
    return np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])


def metrics_by(pred: pd.DataFrame, column: str) -> pd.DataFrame:
    """Metrics for each value of ``column`` (e.g. ``snr_db``). ``n`` is the slice's denominator."""
    rows = []
    for value, part in pred.groupby(column):
        rows.append({column: value, **binary_metrics(part.y_true, part.y_pred)})
    return pd.DataFrame(rows)


def recall_by_tag(pred: pd.DataFrame) -> pd.DataFrame:
    """Fraction classified correctly per source tag (recall for drone tags, specificity for labnoise)."""
    g = pred.assign(correct=(pred.y_true == pred.y_pred)).groupby("class_tag")
    return g.agg(
        n_images=("correct", "size"), n_groups=("split_group_id", "nunique"), correct_rate=("correct", "mean")
    ).reset_index()


def aggregate_by_group(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per split group: mean drone probability over its images, thresholded at 0.5.

    This counts each independent source once instead of once per SNR copy.
    """
    g = pred.groupby("split_group_id").agg(
        y_true=("y_true", "first"), p_drone=("p_drone", "mean"), n_images=("y_true", "size")
    )
    g["y_pred"] = (g.p_drone >= 0.5).astype(int)
    return g.reset_index()


def bootstrap_by_group(
    pred: pd.DataFrame, metric: str = "macro_f1", n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for an image-level metric, resampling GROUPS (not images) with replacement."""
    groups = pred.split_group_id.to_numpy()
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    yt, yp = pred.y_true.to_numpy(), pred.y_pred.to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        values.append(binary_metrics(yt[idx], yp[idx])[metric])
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


@torch.no_grad()
def predict(model: torch.nn.Module, dataset, batch_size: int = 64, device: str = "cpu") -> pd.DataFrame:
    """Run the model over a dataset. Returns the dataset's metadata plus y_true, p_drone, y_pred."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, idxs = [], []
    for x, _, i in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu())
        idxs.append(i)
    p = torch.cat(probs).numpy() if probs else np.zeros(0)
    order = torch.cat(idxs).numpy() if idxs else np.zeros(0, dtype=int)
    out = dataset.frame.iloc[order].copy()
    out["y_true"] = dataset.labels[order]
    out["p_drone"] = p
    out["y_pred"] = (p >= 0.5).astype(int)
    return out.reset_index(drop=True)


def full_report(pred: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Everything we report for one model on one split."""
    report: dict = {
        "class_order": list(CLASS_NAMES),
        "n_images": int(len(pred)),
        "n_groups": int(pred.split_group_id.nunique()),
        "per_image": binary_metrics(pred.y_true, pred.y_pred),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(pred.y_true, pred.y_pred).tolist(),
    }
    grouped = aggregate_by_group(pred)
    report["per_group"] = binary_metrics(grouped.y_true, grouped.y_pred)
    if n_boot and len(pred):
        for metric in ("accuracy", "macro_f1", "recall_drone", "false_positive_rate"):
            report[f"ci95_{metric}_group_bootstrap"] = bootstrap_by_group(pred, metric, n_boot)
    report["per_snr"] = metrics_by(pred, "snr_db").to_dict(orient="records")
    report["per_tag"] = recall_by_tag(pred).to_dict(orient="records")
    return report


def save_report(report: dict, pred: pd.DataFrame, out_dir: str | Path, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cols = ["filename_stem", "class_tag", "source_sample_id", "snr_db", "split_group_id", "split"]
    pred[[*cols, "y_true", "p_drone", "y_pred"]].to_csv(out / f"predictions_{name}.csv", index=False)
