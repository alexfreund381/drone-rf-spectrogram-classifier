import numpy as np
import pytest
import torch

from drone_rf.dataset import SpectrogramDataset, filter_rows
from drone_rf.manifest import QUARANTINE, build_manifest_from_dir
from drone_rf.preprocessing import AugmentConfig, augment, compute_mean_std, load_gray_uint8, to_tensor
from drone_rf.splits import assign_splits


def test_tensor_shape_dtype_range(splits, fixture_dir):
    ds = SpectrogramDataset(filter_rows(splits, "train"), fixture_dir, size=32)
    x, y, i = ds[0]
    assert x.shape == (1, 32, 32) and x.dtype == torch.float32
    assert torch.isfinite(x).all() and 0.0 <= x.min() and x.max() <= 1.0
    assert y in (0, 1) and i == 0


def test_resize_and_native_size(fixture_dir):
    path = next((fixture_dir / "images").glob("*.png"))
    assert load_gray_uint8(path).shape == (48, 48)
    assert load_gray_uint8(path, 24).shape == (24, 24)
    assert to_tensor(load_gray_uint8(path, 24)).shape == (1, 24, 24)


def test_labels_come_from_manifest_not_filename(splits, fixture_dir):
    df = filter_rows(splits, "train").copy()
    flipped = df.assign(binary_label=df.binary_label.map({"0": "1", "1": "0"}))
    a = SpectrogramDataset(df, fixture_dir, size=16)
    b = SpectrogramDataset(flipped, fixture_dir, size=16)
    assert (a.labels == 1 - b.labels).all()


def test_quarantined_rows_rejected_and_filtered(edge_dir):
    m, _ = build_manifest_from_dir(edge_dir)
    s = assign_splits(m, seed=1)
    with pytest.raises(ValueError):
        SpectrogramDataset(m[m.binary_label == QUARANTINE], edge_dir, size=16)
    for split in ("train", "val", "test"):
        assert set(filter_rows(s, split).binary_label) <= {"0", "1"}


def test_filter_by_snr_and_existing_files(splits, fixture_dir, tmp_path):
    df = filter_rows(splits, "train", snr_levels=[20])
    assert set(df.snr_db) == {20}
    none = filter_rows(splits, "train", image_dir=tmp_path, require_files=True)
    assert len(none) == 0
    assert len(filter_rows(splits, "train", image_dir=fixture_dir, require_files=True)) == (splits.split == "train").sum()


def test_eval_dataset_is_deterministic(splits, fixture_dir):
    ds = SpectrogramDataset(filter_rows(splits, "val"), fixture_dir, size=32)
    assert torch.equal(ds[3][0], ds[3][0])
    again = SpectrogramDataset(filter_rows(splits, "val"), fixture_dir, size=32, preload=False)
    assert torch.equal(ds[3][0], again[3][0])


def test_mean_std_uses_only_the_given_images():
    arrays = np.zeros((2, 4, 4), dtype=np.uint8)
    arrays[1] = 255
    mean, std = compute_mean_std(arrays)
    assert mean == pytest.approx(0.5) and std == pytest.approx(0.5)
    assert compute_mean_std(arrays[:1]) == (0.0, 1e-6)


def test_train_stats_do_not_depend_on_val_or_test(splits, fixture_dir):
    train = SpectrogramDataset(filter_rows(splits, "train"), fixture_dir, size=16)
    changed = splits[splits.split == "train"]  # same train rows, val/test removed entirely
    alone = SpectrogramDataset(filter_rows(changed, "train"), fixture_dir, size=16)
    assert train.mean_std() == alone.mean_std()


def test_augmentation_off_is_identity_and_on_is_bounded():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 32, 32)
    assert torch.equal(augment(x, AugmentConfig(), g), x)
    cfg = AugmentConfig(gain=0.1, mask_x=8, mask_y=8, shift=4, pixel_noise=0.05)
    y = augment(x, cfg, g)
    assert y.shape == x.shape and not torch.equal(x, y)
    assert 0.0 <= y.min() and y.max() <= 1.0 and torch.isfinite(y).all()


def test_augmentation_only_when_configured(splits, fixture_dir):
    df = filter_rows(splits, "train")
    plain = SpectrogramDataset(df, fixture_dir, size=32)
    aug = SpectrogramDataset(df, fixture_dir, size=32, augment_cfg=AugmentConfig(gain=0.2))
    assert torch.equal(plain[0][0], plain[0][0])
    assert not torch.equal(aug[0][0], plain[0][0])
