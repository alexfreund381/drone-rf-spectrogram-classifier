"""Collect finished runs into one comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COMPARE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "recall_drone", "precision_drone", "false_positive_rate")


def compare_runs(runs_dir: str | Path, split: str = "val") -> pd.DataFrame:
    """One row per run directory that has ``meta.json`` and ``metrics_<split>.json``."""
    rows = []
    for run in sorted(Path(runs_dir).iterdir()):
        meta_path, metrics_path = run / "meta.json", run / f"metrics_{split}.json"
        if not (meta_path.exists() and metrics_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "run": meta.get("run_name", run.name),
            "model": meta.get("model", {}).get("name", ""),
            "pretrained_weights": meta.get("pretrained_weights", ""),
            "split": split,
            "n_images": metrics["n_images"],
            "n_groups": metrics["n_groups"],
            **{m: metrics["per_image"][m] for m in COMPARE_METRICS},
            "group_macro_f1": metrics["per_group"]["macro_f1"],
            "best_epoch": meta.get("best_epoch", ""),
            "wall_seconds": meta.get("wall_seconds", ""),
            "split_hash": meta.get("split_hash", ""),
        }
        ci = metrics.get("ci95_macro_f1_group_bootstrap")
        if ci:
            row["macro_f1_ci95_low"], row["macro_f1_ci95_high"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Small dependency-free Markdown table writer."""
    def fmt(v) -> str:
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(fmt(v) for v in row) + " |\n" for row in df.itertuples(index=False))
