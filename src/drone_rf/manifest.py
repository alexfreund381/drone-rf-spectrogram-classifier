"""Build the manifest: one row per image with provenance, box summary and binary label.

The binary label is derived from the filename tag AND the label-file contents. Anything
ambiguous is quarantined with a reason instead of being guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .annotations import LabelParse, parse_label_text
from .sources import BACKGROUND_TAG, RADIO_SYSTEMS, TAG_TO_CLASS_ID

STEM_RE = re.compile(r"^IQdata_sample(\d+)_(target[0-5]|labnoise)_snr(-?\d+)db$")

QUARANTINE = "quarantine"

MANIFEST_COLUMNS = [
    "filename_stem",
    "image_member",
    "label_member",
    "class_tag",
    "radio_system",
    "source_sample_id",
    "original_recording_id",
    "snr_db",
    "n_valid_boxes",
    "box_class_ids",
    "binary_label",
    "quarantine_reason",
]


@dataclass(frozen=True)
class ParsedName:
    source_sample_id: int
    class_tag: str
    snr_db: int


def parse_stem(stem: str) -> ParsedName | None:
    """Parse ``IQdata_sample<N>_<tag>_snr<S>db``. Returns None if the name does not match."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    return ParsedName(int(m.group(1)), m.group(2), int(m.group(3)))


def derive_binary_label(class_tag: str, label: LabelParse | None) -> tuple[str, str]:
    """Apply the labeling rules. Returns ``(binary_label, quarantine_reason)``.

    ``binary_label`` is "1", "0" or "quarantine"; the reason is "" unless quarantined.
    ``label`` is None when the label file is missing.
    """
    if label is None:
        return QUARANTINE, "missing_label"
    if label.error:
        return QUARANTINE, f"invalid_label:{label.error}"
    n = len(label.boxes)
    if class_tag == BACKGROUND_TAG:
        return ("0", "") if n == 0 else (QUARANTINE, "background_has_boxes")
    if n == 0:
        return QUARANTINE, "target_zero_boxes"
    if set(label.class_ids) != {TAG_TO_CLASS_ID[class_tag]}:
        return QUARANTINE, "class_mismatch"
    return "1", ""


def build_manifest(
    images: dict[str, str] | None,
    labels: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the manifest from image and label listings.

    Args:
        images: ``{stem: member_path}`` for every image, or None when images were not listed
            (label-index mode). In that mode the image path is left empty and no
            ``missing_image`` check is possible.
        labels: ``{stem: (member_path, file_text)}`` for every label file.

    Returns:
        ``(manifest, problems)``. ``problems`` lists entries that could not become manifest
        rows: unparseable names and labels without an image.
    """
    rows: list[dict] = []
    problems: list[dict] = []
    stems = set(labels) if images is None else set(images) | set(labels)
    for stem in sorted(stems):
        parsed = parse_stem(stem)
        if parsed is None:
            problems.append({"filename_stem": stem, "problem": "unparseable_name"})
            continue
        if images is not None and stem not in images:
            problems.append({"filename_stem": stem, "problem": "missing_image"})
            continue
        label_member, label_parse = "", None
        if stem in labels:
            label_member, text = labels[stem]
            label_parse = parse_label_text(text)
        binary, reason = derive_binary_label(parsed.class_tag, label_parse)
        boxes = label_parse.boxes if label_parse and not label_parse.error else []
        rows.append(
            {
                "filename_stem": stem,
                "image_member": images[stem] if images is not None else "",
                "label_member": label_member,
                "class_tag": parsed.class_tag,
                "radio_system": RADIO_SYSTEMS[parsed.class_tag],
                "source_sample_id": parsed.source_sample_id,
                "original_recording_id": "",
                "snr_db": parsed.snr_db,
                "n_valid_boxes": len(boxes),
                "box_class_ids": ";".join(str(b.class_id) for b in boxes),
                "binary_label": binary,
                "quarantine_reason": reason,
            }
        )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return manifest, pd.DataFrame(problems, columns=["filename_stem", "problem"])


def build_manifest_from_dir(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the manifest from an extracted dataset directory with ``images/`` and ``labels/``."""
    root = Path(dataset_dir)
    images = {p.stem: p.relative_to(root).as_posix() for p in sorted((root / "images").glob("*.png"))}
    labels = {
        p.stem: (p.relative_to(root).as_posix(), p.read_text(encoding="utf-8", errors="replace"))
        for p in sorted((root / "labels").glob("*.txt"))
    }
    return build_manifest(images, labels)


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest/splits CSV keeping label columns as strings."""
    return pd.read_csv(
        path,
        dtype={"binary_label": str, "box_class_ids": str, "quarantine_reason": str, "split": str},
        keep_default_na=False,
    )


def summarize(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Count tables for the dataset audit. Keys are table names."""
    m = manifest
    out: dict[str, pd.DataFrame] = {}
    out["images_by_tag_and_label"] = (
        m.groupby(["class_tag", "binary_label"]).size().unstack(fill_value=0).reset_index()
    )
    by_snr = m.groupby(["snr_db", "class_tag"]).size().unstack(fill_value=0)
    out["images_by_tag_and_snr"] = by_snr.reset_index()
    out["quarantine_by_reason_and_tag"] = (
        m[m.binary_label == QUARANTINE]
        .groupby(["quarantine_reason", "class_tag"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    per_sample = (
        m.groupby(["class_tag", "source_sample_id"])
        .agg(
            n_images=("snr_db", "size"),
            n_snr_levels=("snr_db", "nunique"),
            n_usable=("binary_label", lambda s: int((s != QUARANTINE).sum())),
            min_boxes=("n_valid_boxes", "min"),
            max_boxes=("n_valid_boxes", "max"),
        )
        .reset_index()
    )
    out["per_sample"] = per_sample
    out["samples_by_tag"] = (
        per_sample.groupby("class_tag")
        .agg(
            n_source_samples=("source_sample_id", "size"),
            min_sample_id=("source_sample_id", "min"),
            max_sample_id=("source_sample_id", "max"),
            samples_fully_usable=(
                "n_usable",
                lambda s: int((s == per_sample.loc[s.index, "n_images"]).sum()),
            ),
            samples_fully_quarantined=("n_usable", lambda s: int((s == 0).sum())),
            samples_partly_quarantined=(
                "n_usable",
                lambda s: int(((s > 0) & (s < per_sample.loc[s.index, "n_images"])).sum()),
            ),
        )
        .reset_index()
    )
    ids_per_tag = m.groupby("source_sample_id")["class_tag"].nunique()
    out["sample_ids_under_multiple_tags"] = (
        ids_per_tag[ids_per_tag > 1].rename("n_tags").reset_index()
    )
    return out
