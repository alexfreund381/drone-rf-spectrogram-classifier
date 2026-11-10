# Download and Storage

How to get the data, what each route costs, and what has actually been downloaded so far.

The dataset is one uncompressed TAR of 101,077,473,280 bytes (101.1 GB) on Zenodo. Nothing from it is
stored in git; everything goes under `data/`, which is git-ignored.

## What has been downloaded (as of 2026-11-10)

| What | Bytes | Command | Result |
|---|---|---|---|
| All label files (archive tail) | 285 MB | `index-labels` | `data/manifest/manifest.csv`, counts in `artifacts/audit/` |
| Pilot image sample (12 regions × 256 MiB) | 3.2 GB | `fetch-regions` | `data/selected/images/`, log in `data/selected/fetch_log.csv` |
| Full archive | — | — | **not downloaded** |

Measured download speed from Zenodo on this connection: about 3 MB/s per connection. At that rate the
full archive would take roughly 9–10 hours.

## Route 1 — Labels only (285 MB, a few minutes)

Enough to rebuild the manifest, the audit tables and the frozen split.

```
python -m drone_rf.cli index-labels --out data/manifest/manifest.csv --boxes-out data/manifest/label_text.csv --summary-dir artifacts/audit
python -m drone_rf.cli split --manifest data/manifest/manifest.csv --out data/manifest/splits.csv --seed 42 --block-size 20
```

The second command must print split hash `ddfcbce03785a3af`.

## Route 2 — Partial image sample (what we use now)

```
python -m drone_rf.cli fetch-regions --output data/selected --regions 12 --region-mb 256 --image-region-end 100792909312 --splits data/manifest/splits.csv
```

- Downloads 12 evenly spaced 256 MiB byte ranges of the image region and keeps every complete image
  inside them (about 850 per range). Quarantined images are skipped.
- The archive's member order is shuffled, so the sample mixes all tags and SNR levels. Its tag and
  SNR mix is compared with the full label index in DATASET_AUDIT.md §6.
- Region positions come from the arguments only. To add **different** images later, rerun with
  another `--phase` (for example `--phase 0.25`); existing files are never overwritten.
- Integrity: a partial download cannot be compared with Zenodo's archive MD5. Each saved image's
  SHA-256, archive offset and size are logged in `fetch_log.csv`, and every header checksum is
  verified while walking.
- To reproduce exactly the same sample, use exactly the same arguments.

## Route 3 — Full archive (101 GB)

Needs about 101 GB free plus space for extracted images. This machine has 331 GB free.

```
curl -L -C - -o data/raw/LocalisingDroneRFEmissionsInSpectrograms.tar "https://zenodo.org/records/21428081/files/LocalisingDroneRFEmissionsInSpectrograms.tar?download=1"
python -m drone_rf.cli md5 --archive data/raw/LocalisingDroneRFEmissionsInSpectrograms.tar
python -m drone_rf.cli audit --archive data/raw/LocalisingDroneRFEmissionsInSpectrograms.tar --out data/manifest/manifest_full.csv --summary-dir artifacts/audit_full
python -m drone_rf.cli extract --archive data/raw/LocalisingDroneRFEmissionsInSpectrograms.tar --splits <subset.csv> --output data/selected
```

- `curl -C -` resumes an interrupted download.
- `md5` must report Zenodo's value `0e9e7f813de8d3816e6293bf37c3a8c6`. Hashing reads all 101 GB.
- `extract` writes only the images named in the given splits file and refuses unsafe archive members.
- These commands are implemented and tested on a fixture archive. **They have not been run on the
  real archive.**

## Which route when

| Need | Route |
|---|---|
| Class counts, approval email, split | 1 |
| Train and compare models on a CPU | 2 (thousands of images) |
| Exactly chosen images (for example one fixed SNR for every sample), archive checksum | 3 |

## Storage

| Item | Size |
|---|---|
| One PNG | about 300 KB |
| 10,000 images on disk | about 3 GB |
| 10,000 images in RAM at 224×224 | about 0.5 GB |
| Manifest + label text | about 90 MB |
