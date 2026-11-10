# Data pipeline: sources, labels, manifest, archive tools

**Code:** `src/drone_rf/{sources,annotations,manifest,archive,synthetic}.py`
**Tests:** `tests/test_annotation.py`, `tests/test_manifest.py`, `tests/test_archive.py`
**Owner:** workstream A (unassigned)

## Purpose

Turn a 101 GB archive we do not want to download blindly into a small, trustworthy table (the
*manifest*) with one row per image: which recording it came from, its noise level, how many boxes its
label file has, and the binary label we derived, or the reason it was set aside.

## Inputs and outputs

| | Format | Example |
|---|---|---|
| Input | Zenodo TAR (remote URL or local file), or an extracted folder with `images/` and `labels/` | `…/images/IQdata_sample3693_target1_snr-2db.png` |
| Output | `data/manifest/manifest.csv`, one row per image | columns in PROJECT_PLAN.md §4.7 |
| Output | Count tables in `artifacts/audit/` | `samples_by_tag.csv` |
| Output | Extracted images in `data/selected/images/` | |

## How it works

**1. Filename parsing (`manifest.parse_stem`).** A name must match
`IQdata_sample<N>_<target0-5|labnoise>_snr<signed int>db`. Anything else is reported as
`unparseable_name` instead of being forced into the pattern.

**2. Label parsing (`annotations.parse_label_text`).** Each row must be
`class_id x_center y_center width height`. A file is invalid if any row has the wrong number of
columns, non-numeric text, a class ID outside 0–5, a centre outside [0, 1], or a non-positive size.
An empty file is valid and means zero boxes. One bad row invalidates the whole file.

**3. Labeling rules (`manifest.derive_binary_label`).**

| Tag | Label file | Result |
|---|---|---|
| `target*` | ≥1 box, all with the tag's class ID | `1` (drone_rf) |
| `labnoise` | empty | `0` (background_rf) |
| `target*` | empty | quarantine `target_zero_boxes` |
| `labnoise` | has boxes | quarantine `background_has_boxes` |
| `target*` | other class IDs present | quarantine `class_mismatch` |
| any | invalid | quarantine `invalid_label:<reason>` |
| any | no label file | quarantine `missing_label` |

**4. Getting the labels without the images (`archive.index_labels_from_tail`).** A TAR file is a
sequence of 512-byte headers, each followed by that file's data. In this archive all images come
first and all labels sit at the end. The tool requests the last 16 MiB, then the 16 MiB before that,
and so on, until a chunk contains an image header. The label region starts right after that image.
It then walks the region header by header and reads every label file. Each header's checksum is
verified, so random data cannot be mistaken for a header.

**5. Partial image sample (`archive.fetch_region_members`, CLI `fetch-regions`).** Downloads a few
evenly spaced byte ranges of the image region and keeps every image that lies completely inside a
range. Because the archive order is shuffled, each range holds a mix of samples, classes and SNR
levels. Range positions are fixed by the command arguments and do not depend on image content.

**6. Safe extraction from a local TAR (`archive.extract_members`).** Extracts only requested members.
Rejects absolute paths, `..`, symlinks/hardlinks/devices, oversized members, and the second copy of a
duplicate name; never overwrites; never calls `tarfile.extractall`.

**7. Synthetic fixtures (`synthetic.make_fixture`).** Random-noise images with bright rectangles and
matching label files, named like the real data, plus deliberately broken cases. Used by tests only.

## How to run it

```
python -m drone_rf.cli source-info
python -m drone_rf.cli index-labels --out data/manifest/manifest.csv --boxes-out data/manifest/label_text.csv --summary-dir artifacts/audit
python -m drone_rf.cli fetch-regions --output data/selected --regions 12 --region-mb 256 --image-region-end 100792909312 --splits data/manifest/splits.csv
```

Real output of `index-labels` on 2026-11-10 (trimmed):

```
"n_images": 333164, "n_source_samples": 12814, "n_problems": 0,
"label_region_start": 100792909312, "label_region_bytes": 284563968, "bytes_fetched": 285212672,
"n_label_files": 333164, "other_members_in_region": [".../data.txt"], "n_duplicate_members": 0
```

With a fully downloaded archive: `md5`, `audit --archive <tar>`, `extract --archive <tar> --splits … --output …`.

## Design decisions

| Decision | Alternatives considered | Why we chose this |
|---|---|---|
| Read labels by range request from the archive tail | Download all 101 GB first | 285 MB gave exact counts for every image; needed to judge feasibility before committing to a big download |
| Label from tag **and** label file | Tag only; label file only | Tag only would call 310 box-less drone samples "drone" without evidence. File only would make every empty file "background", including those 310 |
| Quarantine instead of dropping or guessing | Treat zero-box drone samples as negatives | They come from the drone recording environment; using them as negatives would inject wrong labels |
| One bad row invalidates the file | Keep the good rows | A half-parsed annotation is not trustworthy; in practice no file was invalid |
| Own 512-byte header parser | Python `tarfile` over HTTP | `tarfile` needs a seekable file; our parser is tested to agree with `tarfile` on a fixture archive |
| Fixture generator writes bytes, not text | `write_text` | On Windows `write_text` turns `\n` into `\r\n`, so fixtures differed from real label files |

## Failure modes and how they are handled

| Failure | Handling |
|---|---|
| Network error during a range request | Up to 5 retries with back-off, then a clear error |
| Short read | Treated as an error and retried |
| Corrupt header while walking | `ValueError` with the byte offset |
| Image cut off at a region edge | Skipped; only complete images are kept |
| Unexpected file name | Listed in `problems.csv` |

## Tests

- Real file names from the archive parse to the right sample ID, tag and signed SNR; six kinds of unexpected names return nothing.
- Each labeling rule and each quarantine reason has a case.
- Our header walk equals Python `tarfile` on a fixture TAR; a corrupted header is rejected.
- The tail index returns exactly the label files of the fixture for chunk sizes 512 B, 4 KiB and 1 MiB, without reading the image region.
- Region fetch started mid-member resynchronises and returns byte-identical images.
- Hostile TAR: traversal, absolute path, symlink, hardlink, and duplicate are each rejected with the right reason.

## Known limitations

- Label-index mode does not list image members, so "every label has an image" is not checked directly.
- A partial download cannot be verified against Zenodo's MD5; `fetch_log.csv` records each image's SHA-256 instead.
- The contiguous-region sample is only as random as the archive's member order.
