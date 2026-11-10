import numpy as np
import pandas as pd

from drone_rf.confounds import intensity_by_class_and_snr, mask_boxes, masked_box_eval
from drone_rf.dataset import filter_rows
from drone_rf.train import train_run


def test_mask_boxes_hides_only_the_box_and_copies():
    arr = np.full((100, 100), 50, dtype=np.uint8)
    arr[40:60, 20:30] = 250  # bright burst: x 0.2-0.3, y 0.4-0.6
    out = mask_boxes(arr, "0 0.25 0.5 0.1 0.2\n", margin=1)
    assert arr[50, 25] == 250  # input untouched
    assert out.max() == 50  # burst replaced by background pixels from the same image
    assert out.shape == arr.shape and out.dtype == arr.dtype


def test_mask_boxes_without_boxes_is_identity():
    arr = np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8)
    assert np.array_equal(mask_boxes(arr, ""), arr)


def test_intensity_table_has_one_row_per_class_and_snr(splits, fixture_dir):
    table = intensity_by_class_and_snr(filter_rows(splits, "train"), fixture_dir, size=32)
    assert len(table) == 2 * 3
    assert table.n_images.sum() == (splits.split == "train").sum()
    # the synthetic generator makes low-SNR images noisier
    noisy = table[table.snr_db == -30].mean_of_image_stds.mean()
    clean = table[table.snr_db == 20].mean_of_image_stds.mean()
    assert noisy > clean


def test_masked_box_eval_runs_and_reports(tmp_path, splits, fixture_dir):
    splits.to_csv(tmp_path / "splits.csv", index=False)
    texts = [
        {"filename_stem": p.stem, "label_text": p.read_bytes().decode()}
        for p in (fixture_dir / "labels").glob("*.txt")
    ]
    pd.DataFrame(texts).to_csv(tmp_path / "label_text.csv", index=False)
    run = train_run(
        {
            "run_name": "c",
            "seed": 0,
            "output_dir": str(tmp_path / "runs"),
            "data": {"splits_csv": str(tmp_path / "splits.csv"), "image_dir": str(fixture_dir), "size": 32},
            "model": {"name": "scratch_cnn", "widths": [8, 16], "dropout": 0.0},
            "train": {"batch_size": 16, "epochs": 8, "lr": 0.01, "patience": 0, "num_threads": 2},
        }
    )
    out = masked_box_eval(run / "best.pt", tmp_path / "label_text.csv", "val")
    assert out["n_drone_images"] > 0 and out["n_background_images"] > 0
    assert 0.0 < out["approx_fraction_of_pixels_changed"] < 0.6
    assert 0.0 <= out["drone_recall_bursts_hidden"] <= 1.0
    assert len(out["per_snr"]) == 3
    # SYNTHETIC fixtures differ ONLY by their rectangles, so hiding them must not raise confidence
    assert out["mean_p_drone_bursts_hidden"] <= out["mean_p_drone_original"] + 1e-6
