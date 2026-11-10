"""Generate SYNTHETIC fake spectrograms for software tests.

These images are random noise with bright rectangles. They follow the real dataset's file
naming and label format so the pipeline can be tested without the 101 GB archive. Any metric
computed on them says nothing about real RF classification performance.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from .sources import ARCHIVE_ROOT, TAG_TO_CLASS_ID

DEFAULT_SNRS = (-30, 0, 20)


def _render(rng: np.random.Generator, size: int, boxes: list[tuple[float, float, float, float]], snr: int):
    """Noise floor plus one bright rectangle per box; lower SNR -> stronger noise."""
    noise_sigma = 12.0 + (20 - snr) * 0.8
    img = rng.normal(60.0, noise_sigma, (size, size))
    for xc, yc, w, h in boxes:
        x0, x1 = int((xc - w / 2) * size), int((xc + w / 2) * size)
        y0, y1 = int((yc - h / 2) * size), int((yc + h / 2) * size)
        img[max(y0, 0) : max(y1, y0 + 1), max(x0, 0) : max(x1, x0 + 1)] += 110.0
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fixture(
    output: str | Path,
    samples_per_tag: int = 6,
    snrs: tuple[int, ...] = DEFAULT_SNRS,
    size: int = 64,
    seed: int = 0,
    with_edge_cases: bool = False,
) -> Path:
    """Write a synthetic dataset directory (``images/``, ``labels/``, ``data.txt``).

    Sample IDs are unique across tags and contiguous per tag, like the real data appears to be.
    Box positions are fixed per source sample across SNR levels.

    With ``with_edge_cases=True`` extra deliberately broken entries are added (see
    ``EDGE_CASE_STEMS``) so quarantine logic can be tested.
    """
    out = Path(output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tags = [*TAG_TO_CLASS_ID, "labnoise"]
    sample_id = 0
    for tag in tags:
        # labnoise gets as many samples as all drone tags together, to keep classes balanced
        n = samples_per_tag * 6 if tag == "labnoise" else samples_per_tag
        for _ in range(n):
            boxes = []
            if tag != "labnoise":
                for _ in range(int(rng.integers(1, 4))):
                    w, h = rng.uniform(0.08, 0.3), rng.uniform(0.08, 0.3)
                    boxes.append((rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2), w, h))
            for snr in snrs:
                stem = f"IQdata_sample{sample_id}_{tag}_snr{snr}db"
                Image.fromarray(_render(rng, size, boxes, snr), mode="L").save(out / "images" / f"{stem}.png")
                cls = TAG_TO_CLASS_ID.get(tag)
                text = "".join(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes)
                (out / "labels" / f"{stem}.txt").write_bytes(text.encode("utf-8"))
            sample_id += 1
    names = "\n".join(f"  {i}: {n}" for n, i in TAG_TO_CLASS_ID.items())
    (out / "data.txt").write_text(f"nc: 6\nnames:\n{names}\n", encoding="utf-8")
    if with_edge_cases:
        _add_edge_cases(out, rng, size, first_id=sample_id)
    return out


# stem suffix -> (label text or None for "no label file", write image?)
EDGE_CASES: dict[str, tuple[str | None, bool]] = {
    "9000_target0_snr0db": ("", True),  # target with empty label -> target_zero_boxes
    "9001_labnoise_snr0db": ("0 0.5 0.5 0.1 0.1\n", True),  # background_has_boxes
    "9002_target1_snr0db": ("3 0.5 0.5 0.1 0.1\n", True),  # class_mismatch
    "9003_target2_snr0db": ("2 0.5 0.5 0.1 0.1\n4 0.2 0.2 0.1 0.1\n", True),  # mixed -> class_mismatch
    "9004_target3_snr0db": ("3 1.5 0.5 0.1 0.1\n", True),  # coord_out_of_range
    "9005_target4_snr0db": ("4 0.5 0.5 0.0 0.1\n", True),  # non_positive_size
    "9006_target5_snr0db": ("5 0.5 0.5 0.1\n", True),  # bad_column_count
    "9007_target0_snr0db": ("0 abc 0.5 0.1 0.1\n", True),  # non_numeric
    "9008_target0_snr0db": (None, True),  # image without label -> missing_label
    "9009_target0_snr0db": ("0 0.5 0.5 0.1 0.1\n", False),  # label without image -> missing_image
    "9010_target1_snr0db": ("7 0.5 0.5 0.1 0.1\n", True),  # bad_class_id
}
UNPARSEABLE_STEM = "holiday_photo_final"


def _add_edge_cases(out: Path, rng: np.random.Generator, size: int, first_id: int) -> None:
    for suffix, (text, has_image) in EDGE_CASES.items():
        stem = f"IQdata_sample{suffix}"
        if has_image:
            Image.fromarray(_render(rng, size, [], 0), mode="L").save(out / "images" / f"{stem}.png")
        if text is not None:
            (out / "labels" / f"{stem}.txt").write_bytes(text.encode("utf-8"))
    Image.fromarray(_render(rng, size, [], 0), mode="L").save(out / "images" / f"{UNPARSEABLE_STEM}.png")
    (out / "labels" / f"{UNPARSEABLE_STEM}.txt").write_bytes(b"")


def make_fixture_tar(dataset_dir: str | Path, tar_path: str | Path, shuffle_seed: int = 0) -> Path:
    """Pack a fixture directory into an uncompressed TAR laid out like the real archive:
    root folder, all images (shuffled), then all labels (shuffled)."""
    root = Path(dataset_dir)
    rng = np.random.default_rng(shuffle_seed)
    images = sorted((root / "images").glob("*.png"))
    labels = sorted((root / "labels").glob("*.txt"))
    rng.shuffle(images)
    rng.shuffle(labels)
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as tf:
        for d in (ARCHIVE_ROOT, f"{ARCHIVE_ROOT}/images"):
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for p in images:
            tf.add(p, arcname=f"{ARCHIVE_ROOT}/images/{p.name}")
        info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/labels")
        info.type = tarfile.DIRTYPE
        tf.addfile(info)
        for p in labels:
            tf.add(p, arcname=f"{ARCHIVE_ROOT}/labels/{p.name}")
        if (root / "data.txt").exists():
            tf.add(root / "data.txt", arcname=f"{ARCHIVE_ROOT}/data.txt")
    return Path(tar_path)


def make_unsafe_tar(tar_path: str | Path) -> Path:
    """A TAR with hostile members, for testing safe extraction."""
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as tf:

        def add(name: str, data: bytes = b"x", type_=tarfile.REGTYPE, linkname: str = "") -> None:
            info = tarfile.TarInfo(name)
            info.type = type_
            info.linkname = linkname
            info.size = len(data) if type_ == tarfile.REGTYPE else 0
            tf.addfile(info, io.BytesIO(data) if type_ == tarfile.REGTYPE else None)

        add("root/images/ok.png", b"fine")
        add("root/../../escape.png")
        add("/abs/path.png")
        add("root/images/link.png", type_=tarfile.SYMTYPE, linkname="/etc/passwd")
        add("root/images/hard.png", type_=tarfile.LNKTYPE, linkname="root/images/ok.png")
        add("root/images/dup.png", b"first")
        add("root/images/dup.png", b"second")
    return Path(tar_path)
