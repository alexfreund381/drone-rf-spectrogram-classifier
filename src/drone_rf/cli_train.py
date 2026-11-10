"""Training/evaluation subcommands (registered by ``cli.py``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_train(args: argparse.Namespace) -> int:
    from .train import load_config, train_run

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    run_dir = train_run(cfg)
    print(f"run directory: {run_dir}")
    print((run_dir / "meta.json").read_text(encoding="utf-8"))
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from .baselines import run_baselines
    from .train import load_config

    results = run_baselines(load_config(args.config), args.split, args.n_boot)
    for name, report in results.items():
        m = report["per_image"]
        print(f"{name:10s} {args.split}: accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} n={m['n']}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .train import evaluate_checkpoint

    report = evaluate_checkpoint(args.checkpoint, args.split, args.snr, args.n_boot, args.name)
    print(json.dumps({k: report[k] for k in ("split", "n_images", "n_groups", "per_image", "per_group")}, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .experiments import compare_runs, to_markdown

    table = compare_runs(args.runs, args.split)
    if not len(table):
        print("no finished runs found")
        return 1
    print(table.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        out.with_suffix(".md").write_text(to_markdown(table), encoding="utf-8")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    import pandas as pd

    from .visualization import plot_confusion, plot_error_gallery, plot_metric_by_snr, plot_training_curves

    out = Path(args.out)
    reports = {}
    for run in sorted(Path(args.runs).iterdir()):
        metrics = run / f"metrics_{args.split}.json"
        if not metrics.exists():
            continue
        report = json.loads(metrics.read_text(encoding="utf-8"))
        reports[run.name] = report
        note = f"{args.note} {run.name} split={args.split} hash={report.get('split_hash', '')}".strip()
        plot_confusion(
            report["confusion_matrix_rows_true_cols_pred"], out / f"{run.name}_confusion_{args.split}.png", run.name, note
        )
        if (run / "epochs.jsonl").exists():
            plot_training_curves(run, out / f"{run.name}_curves.png", note)
        pred_path = run / f"predictions_{args.split}.csv"
        if args.image_dir and pred_path.exists():
            plot_error_gallery(
                pd.read_csv(pred_path), args.image_dir, out / f"{run.name}_errors_{args.split}.png", note=note
            )
    if reports:
        for metric in ("accuracy", "recall_drone", "false_positive_rate"):
            plot_metric_by_snr(reports, out / f"snr_{metric}_{args.split}.png", metric, f"{args.note} split={args.split}")
    print(f"figures for {len(reports)} runs written to {out}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    from .infer import predict_image

    print(json.dumps(predict_image(args.checkpoint, args.image), indent=2))
    return 0


def cmd_model_summary(args: argparse.Namespace) -> int:
    from .models import ResNet18Classifier, ScratchCNN, count_parameters

    cnn = ScratchCNN()
    print("ScratchCNN (random init)")
    for name, shape in cnn.describe(args.size):
        print(f"  {name:16s} {shape}")
    print(f"  parameters: {count_parameters(cnn)}")
    resnet = ResNet18Classifier(pretrained=False)
    print(f"ResNet18Classifier (architecture only, weights not loaded here)\n  parameters: {count_parameters(resnet)}")
    resnet.freeze_backbone(True)
    print(f"  with frozen backbone: {count_parameters(resnet)}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    s = sub.add_parser("train", help="train one model from a YAML config")
    s.add_argument("--config", required=True)
    s.add_argument("--run-name", help="override run_name from the config")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("baselines", help="fit majority/brightness/energy baselines (run A0)")
    s.add_argument("--config", required=True, help="a training config; its data section is reused")
    s.add_argument("--split", default="val", choices=["val", "test"])
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(func=cmd_baselines)

    s = sub.add_parser("evaluate", help="evaluate a checkpoint on a split")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--split", default="val", choices=["train", "val", "test"])
    s.add_argument("--snr", type=int, nargs="+", help="restrict to these SNR levels")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--name", help="suffix of the output files (default: split name)")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("compare", help="table of all finished runs")
    s.add_argument("--runs", default="runs")
    s.add_argument("--split", default="val")
    s.add_argument("--out", help="write CSV (and .md) here")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("figures", help="confusion matrices, curves, SNR plots and error galleries")
    s.add_argument("--runs", default="runs")
    s.add_argument("--split", default="val")
    s.add_argument("--out", default="artifacts/figures")
    s.add_argument("--image-dir", help="needed for error galleries")
    s.add_argument("--note", default="", help="text stamped on every figure, e.g. SYNTHETIC")
    s.set_defaults(func=cmd_figures)

    s = sub.add_parser("infer", help="classify one prepared spectrogram PNG")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--image", required=True)
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("model-summary", help="print layer shapes and parameter counts")
    s.add_argument("--size", type=int, default=224)
    s.set_defaults(func=cmd_model_summary)
