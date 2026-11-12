"""End-to-end CPU smoke tests on SYNTHETIC fixtures. They verify the software, not the science."""

import json

import numpy as np
import pytest
import torch

from drone_rf.baselines import MajorityBaseline, energy_features, run_baselines
from drone_rf.cli import main
from drone_rf.dataset import SpectrogramDataset, filter_rows
from drone_rf.evaluate import predict
from drone_rf.experiments import compare_runs
from drone_rf.infer import predict_image
from drone_rf.splits import split_hash
from drone_rf.train import _run_epoch, evaluate_checkpoint, load_checkpoint, train_run


@pytest.fixture(scope="module")
def workdir(tmp_path_factory, splits, fixture_dir):
    root = tmp_path_factory.mktemp("work")
    splits.to_csv(root / "splits.csv", index=False)
    return root


def _cfg(workdir, fixture_dir, **over):
    cfg = {
        "run_name": "smoke_cnn",
        "seed": 0,
        "output_dir": str(workdir / "runs"),
        "data": {"splits_csv": str(workdir / "splits.csv"), "image_dir": str(fixture_dir), "size": 32},
        "model": {"name": "scratch_cnn", "widths": [8, 16], "dropout": 0.0},
        "train": {"batch_size": 16, "epochs": 6, "lr": 0.01, "patience": 0, "num_threads": 2},
    }
    cfg.update(over)
    return cfg


@pytest.fixture(scope="module")
def cnn_run(workdir, fixture_dir):
    return train_run(_cfg(workdir, fixture_dir))


def test_run_directory_contents_and_log_fields(cnn_run, splits):
    for name in ("config.yaml", "meta.json", "epochs.jsonl", "best.pt", "metrics_val.json", "predictions_val.csv"):
        assert (cnn_run / name).exists(), name
    rows = [json.loads(line) for line in (cnn_run / "epochs.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(np.isfinite(r["train_loss"]) and np.isfinite(r["val_loss"]) for r in rows)
    meta = json.loads((cnn_run / "meta.json").read_text())
    for key in (
        "seed", "split_hash", "git_commit", "hardware", "optimizer_steps", "wall_seconds", "best_epoch",
        "train_class_counts", "parameters", "input_mean_std_from_train", "pretrained_weights",
    ):
        assert key in meta, key
    assert meta["split_hash"] == split_hash(splits)
    assert meta["pretrained_weights"] == "none (random init)"
    assert not (cnn_run / "metrics_test.json").exists()  # training never touches the test split


def test_loss_decreases_on_learnable_toy_data(cnn_run):
    rows = [json.loads(line) for line in (cnn_run / "epochs.jsonl").read_text().splitlines()]
    assert rows[-1]["train_loss"] < rows[0]["train_loss"]


def test_checkpoint_roundtrip_gives_identical_predictions(cnn_run, splits, fixture_dir):
    ds = SpectrogramDataset(filter_rows(splits, "val"), fixture_dir, size=32)
    a, ckpt = load_checkpoint(cnn_run / "best.pt")
    b, _ = load_checkpoint(cnn_run / "best.pt")
    pa, pb = predict(a, ds), predict(b, ds)
    assert np.array_equal(pa.p_drone.to_numpy(), pb.p_drone.to_numpy())
    saved = json.loads((cnn_run / "metrics_val.json").read_text())
    assert saved["per_image"]["tp"] == int(((pa.y_true == 1) & (pa.y_pred == 1)).sum())
    assert ckpt["in_mean"] == pytest.approx(float(a.in_mean))


def test_overfit_tiny_fixed_batch(splits, fixture_dir):
    """A healthy pipeline can memorise 16 images. Failure here means a bug, not a hard dataset."""
    torch.manual_seed(0)
    df = filter_rows(splits, "train")
    df = df.groupby("binary_label").head(8).reset_index(drop=True)
    ds = SpectrogramDataset(df, fixture_dir, size=32)
    from drone_rf.models import ScratchCNN

    model = ScratchCNN(widths=(8, 16), dropout=0.0, in_mean=ds.mean_std()[0], in_std=ds.mean_std()[1])
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(60):
        loss, acc, _ = _run_epoch(model, loader, crit, opt, "cpu")
    assert acc == 1.0 and loss < 0.2


def test_evaluate_on_test_split_is_logged(cnn_run):
    report = evaluate_checkpoint(cnn_run / "best.pt", "test", n_boot=20)
    assert report["split"] == "test" and report["n_images"] > 0
    assert (cnn_run / "metrics_test.json").exists()
    assert len((cnn_run / "test_evaluations.log").read_text().splitlines()) == 1


def test_evaluate_refuses_changed_split(cnn_run, workdir, manifest):
    from drone_rf.splits import assign_splits

    original = (workdir / "splits.csv").read_bytes()
    try:
        assign_splits(manifest, seed=999).to_csv(workdir / "splits.csv", index=False)
        with pytest.raises(RuntimeError, match="hash mismatch"):
            evaluate_checkpoint(cnn_run / "best.pt", "val", n_boot=0)
    finally:
        (workdir / "splits.csv").write_bytes(original)


def test_resnet_two_stage_training_offline(workdir, fixture_dir):
    cfg = _cfg(
        workdir,
        fixture_dir,
        run_name="smoke_resnet",
        model={"name": "resnet18", "pretrained": False},
        train={"batch_size": 16, "epochs": 1, "head_epochs": 1, "lr": 1e-4, "patience": 0, "num_threads": 2},
    )
    run = train_run(cfg)
    rows = [json.loads(line) for line in (run / "epochs.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["head_only", "full"]
    assert rows[0]["trainable_params"] == 1026 and rows[1]["trainable_params"] > 11_000_000
    meta = json.loads((run / "meta.json").read_text())
    assert meta["pretrained_weights"] == "none (random init)"  # never mislabel random weights


def test_baselines_and_compare(cnn_run, workdir, fixture_dir):
    results = run_baselines(_cfg(workdir, fixture_dir), "val", n_boot=20)
    assert set(results) == {"majority", "brightness", "energy"}
    maj = results["majority"]["per_image"]
    assert maj["tp"] + maj["fp"] in (0, maj["n"])  # predicts one class only
    table = compare_runs(workdir / "runs", "val")
    assert {"a0_majority", "a0_energy", "smoke_cnn"} <= set(table.run)
    assert table.split_hash.nunique() == 1


def test_majority_baseline_uses_training_majority():
    m = MajorityBaseline().fit(np.zeros((5, 1)), np.array([1, 1, 1, 0, 0]))
    assert m.predict_proba(np.zeros((3, 1)))[:, 1].tolist() == [1.0, 1.0, 1.0]


def test_energy_features_shape_and_content():
    arrays = np.zeros((2, 32, 32), dtype=np.uint8)
    arrays[1, :, 16:] = 255
    f = energy_features(arrays)
    assert f.shape == (2, 39) and np.isfinite(f).all()
    assert f[0, 0] == 0.0 and f[1, 0] == pytest.approx(0.5)


def test_infer_single_image(cnn_run, fixture_dir):
    image = next((fixture_dir / "images").glob("*target0*.png"))
    out = predict_image(cnn_run / "best.pt", image)
    assert set(out["scores_uncalibrated"]) == {"background_rf", "drone_rf"}
    assert sum(out["scores_uncalibrated"].values()) == pytest.approx(1.0, abs=1e-5)


def test_cli_help_and_end_to_end(tmp_path, capsys):
    for cmd in (
        "source-info", "make-synthetic-fixture", "index-labels", "audit", "split", "extract", "md5", "fetch-regions",
        "train", "baselines", "evaluate", "compare", "figures", "infer", "model-summary", "confound-check",
    ):
        if cmd == "source-info":
            assert main([cmd]) == 0
            continue
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--help"])
        assert exc.value.code == 0
    data, tar = tmp_path / "fx", tmp_path / "fx.tar"
    assert main(["make-synthetic-fixture", "--output", str(data), "--samples-per-tag", "3", "--tar", str(tar)]) == 0
    manifest, summary = tmp_path / "manifest.csv", tmp_path / "audit"
    assert main(["audit", "--archive", str(tar), "--out", str(manifest), "--summary-dir", str(summary)]) == 0
    assert main(["index-labels", "--archive", str(tar), "--out", str(tmp_path / "m2.csv"),
                 "--summary-dir", str(tmp_path / "audit2"), "--chunk-mb", "1"]) == 0
    splits = tmp_path / "splits.csv"
    assert main(["split", "--manifest", str(manifest), "--out", str(splits), "--seed", "1"]) == 0
    assert main(["extract", "--archive", str(tar), "--splits", str(splits), "--output", str(tmp_path / "sel")]) == 0
    assert len(list((tmp_path / "sel" / "images").glob("*.png"))) == 36 * 3
    assert json.loads((summary / "summary.json").read_text())["n_images"] == 108
    capsys.readouterr()


def test_meta_reports_the_weights_the_run_started_from(workdir, fixture_dir, monkeypatch):
    """Regression test: a pretrained run was once logged as 'random' because the label was read
    after the model had been rebuilt from its checkpoint."""
    import drone_rf.train as train_module
    from drone_rf.models import ResNet18Classifier

    def fake_build(cfg, in_mean=0.5, in_std=0.25):
        model = ResNet18Classifier(pretrained=False)  # no download in tests
        if cfg.get("pretrained"):
            model.pretrained, model.weights_name = True, "FAKE_PRETRAINED_FOR_TEST"
        return model

    monkeypatch.setattr(train_module, "build_model", fake_build)
    cfg = _cfg(
        workdir,
        fixture_dir,
        run_name="smoke_resnet_labelled_pretrained",
        model={"name": "resnet18", "pretrained": True},
        train={"batch_size": 16, "epochs": 1, "head_epochs": 0, "lr": 1e-4, "patience": 0, "num_threads": 2},
    )
    run = train_run(cfg)
    assert json.loads((run / "meta.json").read_text())["pretrained_weights"] == "FAKE_PRETRAINED_FOR_TEST"
    model, ckpt = train_module.load_checkpoint(run / "best.pt")
    assert ckpt["initial_weights"] == "FAKE_PRETRAINED_FOR_TEST"
    assert model.pretrained and model.weights_name == "FAKE_PRETRAINED_FOR_TEST"
