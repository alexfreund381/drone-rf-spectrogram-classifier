"""Group-safe deterministic train/val/test splits.

A *group* is the set of images that must stay together. At minimum it is one source sample
(all of its SNR variants). With ``block_size > 1`` neighbouring sample IDs within a class tag
are also kept together, as a conservative stand-in for unknown recording-level provenance.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .manifest import QUARANTINE

SPLITS = ("train", "val", "test")


def group_id(class_tag: str, source_sample_id: int, block_size: int = 1) -> str:
    """Split-group key. Includes the tag so a sample ID reused across tags cannot collide."""
    if block_size <= 1:
        return f"{class_tag}_s{int(source_sample_id)}"
    return f"{class_tag}_b{int(source_sample_id) // block_size}x{block_size}"


def _tag_seed(seed: int, tag: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


def assign_splits(
    manifest: pd.DataFrame,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    block_size: int = 1,
) -> pd.DataFrame:
    """Add ``split_group_id`` and ``split`` columns.

    Groups are shuffled per class tag (stratification by tag) with a seed derived from
    ``seed`` and the tag, then cut into train/val/test. Every row of a group gets the group's
    split, except quarantined rows, whose ``split`` is left empty so they can never be loaded.
    A group counts only if it has at least one usable (non-quarantined) image.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    df = manifest.copy()
    df["split_group_id"] = [
        group_id(t, s, block_size) for t, s in zip(df.class_tag, df.source_sample_id, strict=True)
    ]
    usable = df[df.binary_label != QUARANTINE]
    assignment: dict[str, str] = {}
    for tag in sorted(usable.class_tag.unique()):
        groups = sorted(usable.loc[usable.class_tag == tag, "split_group_id"].unique())
        rng = np.random.default_rng(_tag_seed(seed, tag))
        order = rng.permutation(len(groups))
        n = len(groups)
        n_test = int(round(n * fractions[2]))
        n_val = int(round(n * fractions[1]))
        if n >= 3:  # every split gets at least one group when possible
            n_test, n_val = max(n_test, 1), max(n_val, 1)
        n_train = n - n_val - n_test
        for rank, idx in enumerate(order):
            split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            assignment[groups[idx]] = split
    df["split"] = df.split_group_id.map(assignment).fillna("")
    df.loc[df.binary_label == QUARANTINE, "split"] = ""
    assert_group_safe(df)
    return df


def assert_group_safe(df: pd.DataFrame) -> None:
    """Raise AssertionError if any group or any (tag, sample) appears in more than one split."""
    used = df[df.split != ""]
    per_group = used.groupby("split_group_id")["split"].nunique()
    leaking = per_group[per_group > 1]
    if len(leaking):
        raise AssertionError(f"{len(leaking)} groups appear in more than one split, e.g. {leaking.index[0]}")
    per_sample = used.groupby(["class_tag", "source_sample_id"])["split"].nunique()
    leaking = per_sample[per_sample > 1]
    if len(leaking):
        raise AssertionError(f"{len(leaking)} source samples straddle splits, e.g. {leaking.index[0]}")
    bad = set(used.split.unique()) - set(SPLITS)
    if bad:
        raise AssertionError(f"unknown split names: {bad}")
    if (df.loc[df.binary_label == QUARANTINE, "split"] != "").any():
        raise AssertionError("quarantined rows must not have a split")


def split_hash(df: pd.DataFrame) -> str:
    """Content hash of the split assignment (order-independent). Recorded by every run."""
    used = df[df.split != ""]
    rows = zip(used.filename_stem, used.binary_label, used.split, strict=True)
    lines = sorted(f"{s},{b},{p}" for s, b, p in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def select_subset(
    df: pd.DataFrame,
    snr_levels: list[int] | None = None,
    max_groups_per_class: dict[str, int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Choose a class-balanced subset of *groups* per split, then keep the requested SNR levels.

    Selection uses only manifest metadata (never image content). ``max_groups_per_class`` maps
    split name -> number of groups to keep for EACH binary class in that split. Within the drone
    class, groups are taken round-robin over transmitter tags so no tag dominates.
    """
    used = df[df.split != ""]
    if snr_levels is not None:
        used = used[used.snr_db.isin(snr_levels)]
    if not max_groups_per_class:
        return used.copy()
    keep: set[str] = set()
    rng = np.random.default_rng(seed)
    for split in SPLITS:
        quota = max_groups_per_class.get(split)
        part = used[used.split == split]
        for label in ("0", "1"):
            cls = part[part.binary_label == label]
            by_tag = {
                tag: list(rng.permutation(sorted(g.split_group_id.unique())))
                for tag, g in sorted(cls.groupby("class_tag"), key=lambda kv: kv[0])
            }
            chosen: list[str] = []
            while by_tag and (quota is None or len(chosen) < quota):
                for tag in list(by_tag):
                    if quota is not None and len(chosen) >= quota:
                        break
                    chosen.append(by_tag[tag].pop())
                    if not by_tag[tag]:
                        del by_tag[tag]
            keep.update(chosen)
    return used[used.split_group_id.isin(keep)].copy()


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Groups, source samples and images per split and binary class."""
    used = df[df.split != ""]
    return (
        used.groupby(["split", "binary_label"])
        .agg(
            groups=("split_group_id", "nunique"),
            source_samples=("source_sample_id", "nunique"),
            images=("filename_stem", "size"),
        )
        .reset_index()
    )
