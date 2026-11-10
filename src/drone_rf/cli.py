"""Command-line interface: ``python -m drone_rf.cli <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _write_summary(manifest: pd.DataFrame, problems: pd.DataFrame, out_dir: Path, extra: dict) -> None:
    from .manifest import summarize

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = summarize(manifest)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    problems.to_csv(out_dir / "problems.csv", index=False)
    info = {
        "n_images": int(len(manifest)),
        "n_label_values": manifest.binary_label.value_counts().to_dict(),
        "n_source_samples": int(manifest.groupby(["class_tag", "source_sample_id"]).ngroups),
        "snr_levels": sorted(int(s) for s in manifest.snr_db.unique()),
        "n_problems": int(len(problems)),
        **extra,
    }
    (out_dir / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


def cmd_source_info(args: argparse.Namespace) -> int:
    from .sources import source_info

    print(json.dumps(source_info(), indent=2))
    return 0


def cmd_make_synthetic_fixture(args: argparse.Namespace) -> int:
    from .synthetic import make_fixture, make_fixture_tar

    out = make_fixture(
        args.output, args.samples_per_tag, tuple(args.snrs), args.size, args.seed, args.edge_cases
    )
    print(f"SYNTHETIC fixture written to {out}")
    if args.tar:
        print(f"SYNTHETIC fixture TAR written to {make_fixture_tar(out, args.tar)}")
    return 0


def cmd_index_labels(args: argparse.Namespace) -> int:
    """Fetch only the label region at the end of the archive and build the manifest from it."""
    from .archive import FileRangeReader, HttpRangeReader, index_labels_from_tail
    from .manifest import build_manifest
    from .sources import ARCHIVE_URL

    if args.archive:
        reader = FileRangeReader(args.archive)
        source = str(args.archive)
    else:
        source = args.url or ARCHIVE_URL
        reader = HttpRangeReader(source)
    idx = index_labels_from_tail(reader, args.chunk_mb * 1024 * 1024, progress=lambda s: print(s, flush=True))
    manifest, problems = build_manifest(None, idx.labels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    if args.boxes_out:
        rows = [{"filename_stem": stem, "label_text": text} for stem, (_, text) in sorted(idx.labels.items())]
        pd.DataFrame(rows).to_csv(args.boxes_out, index=False)
    _write_summary(
        manifest,
        problems,
        Path(args.summary_dir),
        {
            "mode": "label_index_from_tail",
            "source": source,
            "archive_size_bytes": idx.archive_size,
            "label_region_start": idx.region_start,
            "label_region_bytes": idx.archive_size - idx.region_start,
            "bytes_fetched": idx.bytes_fetched,
            "n_label_files": len(idx.labels),
            "other_members_in_region": idx.other_members[:20],
            "n_duplicate_members": len(idx.duplicate_members),
        },
    )
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Build the manifest from a local TAR or an extracted directory (images AND labels)."""
    from .archive import read_tar_listing
    from .manifest import build_manifest, build_manifest_from_dir

    extra: dict = {}
    if args.dataset_dir:
        manifest, problems = build_manifest_from_dir(args.dataset_dir)
        extra = {"mode": "dataset_dir", "source": str(args.dataset_dir)}
    else:
        images, labels, dups = read_tar_listing(args.archive)
        manifest, problems = build_manifest(images, labels)
        extra = {"mode": "local_tar", "source": str(args.archive), "n_duplicate_members": len(dups)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    _write_summary(manifest, problems, Path(args.summary_dir), extra)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    from .manifest import read_manifest
    from .splits import assign_splits, split_hash, split_summary

    df = assign_splits(read_manifest(args.manifest), args.seed, tuple(args.fractions), args.block_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(split_summary(df).to_string(index=False))
    print(f"split hash: {split_hash(df)}  (seed={args.seed}, block_size={args.block_size})")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .archive import extract_members
    from .manifest import read_manifest
    from .sources import ARCHIVE_ROOT

    df = read_manifest(args.splits)
    df = df[df.split != ""]
    wanted: set[str] = set()
    for stem, img in zip(df.filename_stem, df.image_member, strict=True):
        wanted.add(img or f"{ARCHIVE_ROOT}/images/{stem}.png")
    report = extract_members(args.archive, wanted, args.output)
    print(
        f"written={len(report.written)} existing={len(report.skipped_existing)} "
        f"rejected={len(report.rejected)} not_found={len(report.not_found)}"
    )
    for member, reason in report.rejected[:20]:
        print(f"  rejected {member}: {reason}")
    return 1 if report.rejected or report.not_found else 0


def cmd_fetch_regions(args: argparse.Namespace) -> int:
    """Download a few contiguous byte regions of the archive and keep the complete images in them.

    The archive's member order is shuffled, so a contiguous region holds an arbitrary mix of
    samples, classes and SNR levels. Region positions depend only on the arguments, never on
    image content. This is a partial download: no archive-level MD5 check is possible, so the
    SHA-256 of every saved image is logged instead.
    """
    import hashlib
    from pathlib import PurePosixPath

    from .archive import BLOCK, FileRangeReader, HttpRangeReader, fetch_region_members, unsafe_reason
    from .manifest import parse_stem, read_manifest
    from .sources import ARCHIVE_URL

    reader = FileRangeReader(args.archive) if args.archive else HttpRangeReader(args.url or ARCHIVE_URL)
    end = args.image_region_end or reader.size
    region = args.region_mb * 1024 * 1024
    allowed = None
    if args.splits:
        df = read_manifest(args.splits)
        allowed = set(df.loc[df.split != "", "filename_stem"])
    out = Path(args.output) / "images"
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output) / "fetch_log.csv"
    log_rows = []
    for i in range(args.regions):
        centre = int((i + args.phase) / args.regions * end)
        start = max(0, min(centre - region // 2, end - region))
        start -= start % BLOCK
        kept = skipped = 0
        for header, data in fetch_region_members(reader, start, region, lambda h: h.name.endswith(".png")):
            stem = PurePosixPath(header.name).stem
            if unsafe_reason(header.name) or parse_stem(stem) is None or (allowed is not None and stem not in allowed):
                skipped += 1
                continue
            target = out / f"{stem}.png"
            if not target.exists():
                target.write_bytes(data)
            log_rows.append(
                {
                    "filename_stem": stem,
                    "member": header.name,
                    "archive_offset": header.offset,
                    "size": header.size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "region_start": start,
                    "region_bytes": region,
                }
            )
            kept += 1
        print(f"region {i + 1}/{args.regions} at byte {start}: kept {kept} images, skipped {skipped}", flush=True)
    new = pd.DataFrame(log_rows)
    if log_path.exists():
        new = pd.concat([pd.read_csv(log_path), new]).drop_duplicates("filename_stem")
    new.to_csv(log_path, index=False)
    print(f"{len(new)} images logged in {log_path}; {reader.bytes_read / 1e6:.0f} MB downloaded this run")
    return 0


def cmd_md5(args: argparse.Namespace) -> int:
    from .archive import md5_file
    from .sources import ARCHIVE_MD5

    digest = md5_file(args.archive)
    print(f"md5 {digest}  expected {ARCHIVE_MD5}  match={digest == ARCHIVE_MD5}")
    return 0 if digest == ARCHIVE_MD5 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drone_rf", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("source-info", help="print official dataset URLs, checksum and tag table")
    s.set_defaults(func=cmd_source_info)

    s = sub.add_parser("make-synthetic-fixture", help="generate SYNTHETIC fake spectrograms for tests")
    s.add_argument("--output", required=True)
    s.add_argument("--samples-per-tag", type=int, default=6)
    s.add_argument("--snrs", type=int, nargs="+", default=[-30, 0, 20])
    s.add_argument("--size", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--edge-cases", action="store_true", help="add deliberately broken entries")
    s.add_argument("--tar", help="also pack the fixture into this TAR file")
    s.set_defaults(func=cmd_make_synthetic_fixture)

    s = sub.add_parser("index-labels", help="build the manifest from the label region at the end of the TAR")
    s.add_argument("--url", help="archive URL (default: official Zenodo URL)")
    s.add_argument("--archive", help="use a local TAR instead of a URL")
    s.add_argument("--out", default="data/manifest/manifest.csv")
    s.add_argument("--boxes-out", help="also save raw label text per image to this CSV")
    s.add_argument("--summary-dir", default="artifacts/audit")
    s.add_argument("--chunk-mb", type=int, default=16)
    s.set_defaults(func=cmd_index_labels)

    s = sub.add_parser("audit", help="build the manifest from a local TAR or extracted directory")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--archive")
    g.add_argument("--dataset-dir")
    s.add_argument("--out", default="data/manifest/manifest.csv")
    s.add_argument("--summary-dir", default="artifacts/audit")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("split", help="assign group-safe train/val/test splits")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--fractions", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    s.add_argument("--block-size", type=int, default=1, help="keep N neighbouring sample IDs together")
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("extract", help="safely extract only the images named in a splits file")
    s.add_argument("--archive", required=True)
    s.add_argument("--splits", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("fetch-regions", help="download a partial image sample by byte range (no full download)")
    s.add_argument("--url", help="archive URL (default: official Zenodo URL)")
    s.add_argument("--archive", help="use a local TAR instead of a URL")
    s.add_argument("--output", default="data/selected")
    s.add_argument("--regions", type=int, default=8, help="number of evenly spaced regions")
    s.add_argument("--region-mb", type=int, default=256, help="size of each region in MiB")
    s.add_argument("--phase", type=float, default=0.5, help="0..1 position of each region inside its slot")
    s.add_argument("--image-region-end", type=int, help="byte offset where images end (label_region_start)")
    s.add_argument("--splits", help="keep only images with a split in this file (skips quarantined)")
    s.set_defaults(func=cmd_fetch_regions)

    s = sub.add_parser("md5", help="verify a fully downloaded archive against Zenodo's MD5")
    s.add_argument("--archive", required=True)
    s.set_defaults(func=cmd_md5)

    from . import cli_train

    cli_train.register(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
