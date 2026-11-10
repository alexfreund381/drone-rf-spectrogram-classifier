"""Training loop, early stopping, checkpoints and run logging.

One call to ``train_run(config)`` creates ``<output_dir>/<run_name>/`` containing:
  config.yaml, meta.json, epochs.jsonl (one line per epoch), best.pt, metrics_val.json,
  predictions_val.csv

The best checkpoint is the epoch with the highest validation macro-F1 (ties: lower validation
loss). The test split is never read here.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SpectrogramDataset, filter_rows
from .evaluate import binary_metrics, full_report, predict, save_report
from .manifest import read_manifest
from .models import build_model, count_parameters
from .preprocessing import AugmentConfig
from .splits import assert_group_safe, split_hash

DEFAULTS: dict = {
    "run_name": "run",
    "seed": 42,
    "output_dir": "runs",
    "device": "auto",
    "data": {
        "splits_csv": "data/manifest/splits.csv",
        "image_dir": "data/selected",
        "size": 224,
        "train_snr": None,
        "eval_snr": None,
        "require_files": True,
    },
    "model": {"name": "scratch_cnn"},
    "train": {
        "batch_size": 32,
        "epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 5,
        "scheduler": "none",
        "head_epochs": 0,
        "head_lr": 1e-3,
        "class_weights": False,
        "num_threads": None,
    },
    "augment": {},
}


def merge_config(user: dict | None) -> dict:
    """Overlay a user config on DEFAULTS (one level deep)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for key, value in (user or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def load_config(path: str | Path) -> dict:
    return merge_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float, int]:
    """One pass. Trains when ``optimizer`` is given. Returns (mean loss, accuracy, steps)."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, n, steps = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            n += len(y)
    return loss_sum / max(n, 1), correct / max(n, 1), steps


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, extra: dict) -> None:
    torch.save({"model_state": model.state_dict(), "config": cfg, **extra}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint. Never downloads pretrained weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = dict(ckpt["config"]["model"])
    was_pretrained = bool(model_cfg.get("pretrained", model_cfg.get("name") == "resnet18"))
    model_cfg["pretrained"] = False  # weights come from the checkpoint itself
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    if hasattr(model, "pretrained"):
        model.pretrained = was_pretrained
    return model.to(device).eval(), ckpt


def train_run(config: dict) -> Path:
    """Train one model as described by ``config``. Returns the run directory."""
    cfg = merge_config(config)
    tcfg, dcfg = cfg["train"], cfg["data"]
    set_seed(int(cfg["seed"]))
    if tcfg.get("num_threads"):
        torch.set_num_threads(int(tcfg["num_threads"]))
    device = pick_device(cfg["device"])
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    splits = read_manifest(dcfg["splits_csv"])
    assert_group_safe(splits)
    shash = split_hash(splits)
    common = {"image_dir": dcfg["image_dir"], "require_files": dcfg["require_files"]}
    train_df = filter_rows(splits, "train", dcfg["train_snr"], **common)
    val_df = filter_rows(splits, "val", dcfg["eval_snr"], **common)
    if not len(train_df) or not len(val_df):
        raise RuntimeError("no training or validation images found; check image_dir and splits_csv")
    aug = AugmentConfig(**cfg["augment"]) if cfg["augment"] else None
    train_ds = SpectrogramDataset(train_df, dcfg["image_dir"], dcfg["size"], aug, seed=cfg["seed"])
    val_ds = SpectrogramDataset(val_df, dcfg["image_dir"], dcfg["size"])
    in_mean, in_std = train_ds.mean_std()  # training images only

    model = build_model(cfg["model"], in_mean, in_std).to(device)
    counts = np.bincount(train_ds.labels, minlength=2)
    weight = None
    if tcfg["class_weights"]:
        weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"] * 2, shuffle=False)

    # stages: optional head-only stage (frozen backbone), then full training
    stages: list[tuple[str, int, float, bool]] = []
    if int(tcfg["head_epochs"]) > 0 and hasattr(model, "freeze_backbone"):
        stages.append(("head_only", int(tcfg["head_epochs"]), float(tcfg["head_lr"]), True))
    if int(tcfg["epochs"]) > 0:
        stages.append(("full", int(tcfg["epochs"]), float(tcfg["lr"]), False))

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log_path = run_dir / "epochs.jsonl"
    log_path.write_text("", encoding="utf-8")
    best = {"macro_f1": -1.0, "val_loss": float("inf"), "epoch": 0, "stage": ""}
    epoch, total_steps, start = 0, 0, time.time()
    for stage_name, n_epochs, lr, frozen in stages:
        if hasattr(model, "freeze_backbone"):
            model.freeze_backbone(frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(tcfg["weight_decay"]))
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
            if tcfg["scheduler"] == "cosine"
            else None
        )
        stale = 0
        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr_loss, tr_acc, steps = _run_epoch(model, train_loader, criterion, optimizer, device)
            total_steps += steps
            va_loss, _, _ = _run_epoch(model, val_loader, criterion, None, device)
            val_pred = predict(model, val_ds, tcfg["batch_size"] * 2, device)
            vm = binary_metrics(val_pred.y_true, val_pred.y_pred)
            row = {
                "epoch": epoch,
                "stage": stage_name,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": vm["accuracy"],
                "val_macro_f1": vm["macro_f1"],
                "val_recall_drone": vm["recall_drone"],
                "val_precision_drone": vm["precision_drone"],
                "val_false_positive_rate": vm["false_positive_rate"],
                "trainable_params": count_parameters(model)["trainable"],
                "epoch_seconds": time.time() - t0,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[{cfg['run_name']}] epoch {epoch} ({stage_name}) train_loss={tr_loss:.4f} "
                f"val_loss={va_loss:.4f} val_macro_f1={vm['macro_f1']:.4f} {row['epoch_seconds']:.0f}s",
                flush=True,
            )
            if scheduler:
                scheduler.step()
            improved = (vm["macro_f1"], -va_loss) > (best["macro_f1"], -best["val_loss"])
            if improved:
                best = {"macro_f1": vm["macro_f1"], "val_loss": va_loss, "epoch": epoch, "stage": stage_name}
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    cfg,
                    {
                        "epoch": epoch,
                        "val_macro_f1": vm["macro_f1"],
                        "in_mean": in_mean,
                        "in_std": in_std,
                        "split_hash": shash,
                        "git_commit": git_commit(),
                    },
                )
                stale = 0
            else:
                stale += 1
                if int(tcfg["patience"]) > 0 and stale >= int(tcfg["patience"]):
                    print(f"[{cfg['run_name']}] early stop in stage {stage_name} at epoch {epoch}", flush=True)
                    break

    wall = time.time() - start
    model, _ = load_checkpoint(run_dir / "best.pt", device)
    val_pred = predict(model, val_ds, tcfg["batch_size"] * 2, device)
    save_report(full_report(val_pred, n_boot=200), val_pred, run_dir, "val")
    meta = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "pretrained_weights": getattr(model, "weights_name", "none (random init)")
        if getattr(model, "pretrained", False)
        else "none (random init)",
        "parameters": count_parameters(model)["total"],
        "seed": cfg["seed"],
        "split_hash": shash,
        "git_commit": git_commit(),
        "device": device,
        "hardware": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
        },
        "train_images": int(len(train_ds)),
        "train_groups": int(train_df.split_group_id.nunique()),
        "train_class_counts": {"background_rf": int(counts[0]), "drone_rf": int(counts[1])},
        "val_images": int(len(val_ds)),
        "val_groups": int(val_df.split_group_id.nunique()),
        "train_snr": dcfg["train_snr"],
        "eval_snr": dcfg["eval_snr"],
        "input_size": dcfg["size"],
        "input_mean_std_from_train": [in_mean, in_std],
        "augment": cfg["augment"],
        "epochs_run": epoch,
        "optimizer_steps": total_steps,
        "best_epoch": best["epoch"],
        "best_stage": best["stage"],
        "best_val_macro_f1": best["macro_f1"],
        "wall_seconds": wall,
        "checkpoint": str(run_dir / "best.pt"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def evaluate_checkpoint(
    checkpoint: str | Path,
    split: str = "val",
    snr_levels: list[int] | None = None,
    n_boot: int = 1000,
    name: str | None = None,
    all_snr: bool = False,
) -> dict:
    """Evaluate a saved checkpoint on one split using the data paths stored in its config.

    Every evaluation on the test split is appended to ``test_evaluations.log`` in the run
    directory, so "the test set was used once" can be checked.
    """
    ckpt_path = Path(checkpoint)
    device = pick_device("auto")
    model, ckpt = load_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    dcfg = cfg["data"]
    splits = read_manifest(dcfg["splits_csv"])
    assert_group_safe(splits)
    if split_hash(splits) != ckpt["split_hash"]:
        raise RuntimeError("splits file changed since training (hash mismatch); refusing to evaluate")
    snr = None if all_snr else snr_levels if snr_levels is not None else dcfg["eval_snr"]
    df = filter_rows(splits, split, snr, dcfg["image_dir"], dcfg["require_files"])
    ds = SpectrogramDataset(df, dcfg["image_dir"], dcfg["size"])
    pred = predict(model, ds, 64, device)
    report = full_report(pred, n_boot=n_boot)
    report.update({"split": split, "checkpoint": str(ckpt_path), "split_hash": ckpt["split_hash"]})
    save_report(report, pred, ckpt_path.parent, name or split)
    if split == "test":
        with open(ckpt_path.parent / "test_evaluations.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} commit={git_commit()} n={len(pred)}\n")
    return report
