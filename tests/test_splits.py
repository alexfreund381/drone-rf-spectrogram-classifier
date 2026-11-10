import pandas as pd
import pytest

from drone_rf.manifest import QUARANTINE, build_manifest_from_dir
from drone_rf.splits import (
    assert_group_safe,
    assign_splits,
    group_id,
    select_subset,
    split_hash,
    split_summary,
)


def test_no_group_in_two_splits(splits):
    assert (splits.groupby("split_group_id").split.nunique() == 1).all()


def test_no_sample_snr_variants_straddle_splits(splits):
    per_sample = splits.groupby(["class_tag", "source_sample_id"]).split.nunique()
    assert (per_sample == 1).all()
    # every sample keeps all three SNR variants together
    assert (splits.groupby(["class_tag", "source_sample_id"]).size() == 3).all()


def test_every_split_has_both_classes_and_all_tags(splits):
    for split in ("train", "val", "test"):
        part = splits[splits.split == split]
        assert set(part.binary_label) == {"0", "1"}
    assert set(splits[splits.split == "train"].class_tag) == set(splits.class_tag)


def test_same_seed_same_split_and_hash(manifest):
    a, b = assign_splits(manifest, seed=7), assign_splits(manifest, seed=7)
    assert a.equals(b) and split_hash(a) == split_hash(b)


def test_row_order_does_not_change_assignment(manifest):
    a = assign_splits(manifest, seed=7)
    b = assign_splits(manifest.sample(frac=1.0, random_state=1), seed=7)
    assert split_hash(a) == split_hash(b)


def test_different_seed_different_split(manifest):
    assert split_hash(assign_splits(manifest, seed=1)) != split_hash(assign_splits(manifest, seed=2))


def test_fractions_roughly_respected(manifest):
    s = assign_splits(manifest, seed=3)
    groups = s.drop_duplicates("split_group_id").split.value_counts()
    assert groups["train"] > groups["val"] and groups["train"] > groups["test"]
    assert groups["val"] >= 1 and groups["test"] >= 1


def test_quarantined_rows_get_no_split(edge_dir):
    m, _ = build_manifest_from_dir(edge_dir)
    s = assign_splits(m, seed=42)
    assert (s[s.binary_label == QUARANTINE].split == "").all()
    assert (s[s.binary_label != QUARANTINE].split != "").all()


def test_block_grouping_keeps_neighbours_together(manifest):
    s = assign_splits(manifest, seed=42, block_size=2)
    lab = s[s.class_tag == "labnoise"].drop_duplicates("source_sample_id")
    blocks = lab.groupby(lab.source_sample_id // 2).split.nunique()
    assert (blocks == 1).all()
    assert group_id("target1", 7, 4) == group_id("target1", 5, 4) != group_id("target1", 8, 4)


def test_group_id_includes_tag():
    assert group_id("target0", 5) != group_id("target1", 5)


def test_leak_detector_catches_leak(splits):
    bad = splits.copy()
    idx = bad[bad.split == "train"].index[0]
    bad.loc[idx, "split"] = "test"
    with pytest.raises(AssertionError):
        assert_group_safe(bad)


def test_leak_detector_catches_split_on_quarantined():
    df = pd.DataFrame(
        {
            "filename_stem": ["a"],
            "class_tag": ["target0"],
            "source_sample_id": [1],
            "binary_label": [QUARANTINE],
            "split_group_id": ["g"],
            "split": ["train"],
        }
    )
    with pytest.raises(AssertionError):
        assert_group_safe(df)


def test_subset_selection_is_balanced_group_safe_and_snr_filtered(splits):
    quota = {"train": 6, "val": 2, "test": 2}
    sub = select_subset(splits, snr_levels=[20], max_groups_per_class=quota, seed=0)
    assert set(sub.snr_db) == {20}
    assert_group_safe(sub)
    summary = split_summary(sub).set_index(["split", "binary_label"]).groups
    assert summary[("train", "0")] == summary[("train", "1")] == 6
    assert summary[("val", "0")] == summary[("val", "1")] == 2
    # round-robin over tags: all six drone tags are represented in train
    assert sub[(sub.split == "train") & (sub.binary_label == "1")].class_tag.nunique() == 6
    again = select_subset(splits, snr_levels=[20], max_groups_per_class=quota, seed=0)
    assert sub.equals(again)
