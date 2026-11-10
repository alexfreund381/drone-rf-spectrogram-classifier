# Dataset Audit

What we measured about the dataset, how, and what is still unchecked. Evidence labels are defined in
[PROJECT_PLAN.md](PROJECT_PLAN.md).

**Source of every number in sections 1–4:** `artifacts/audit/*.csv` and `artifacts/audit/summary.json`,
produced on 2026-11-10 by

```
python -m drone_rf.cli index-labels --out data/manifest/manifest.csv --boxes-out data/manifest/label_text.csv --summary-dir artifacts/audit
```

This command downloaded only the **label region** at the end of the archive (285 MB of 101.1 GB) with
HTTP range requests and parsed all label files. It did not download the images. Statements about
image *files* therefore rest on the labels plus the README's claim that every image has a label;
section 5 lists what the image sample has confirmed so far.

## 1. Totals (VERIFIED from label files)

| | Count |
|---|---|
| Label files | 333,164 |
| Source samples (distinct sample IDs) | 12,814 |
| SNR levels | 26 (−30 to +20 dB, step 2) |
| Label files per source sample | exactly 26 for every sample (12,814 × 26 = 333,164) |
| Unparseable filenames | 0 |
| Duplicate archive members in the label region | 0 |
| Invalid label rows (bad format, class ID, or coordinates) | 0 |
| Other files in the label region | `data.txt` |

Label region: bytes 100,792,909,312 to 101,077,473,280 (284,563,968 bytes). The image region before
it is 100.79 GB, which is 302.5 KB per label file on average, matching the 291–316 KB PNG sizes seen
in the archive head. That supports one image per label file (CLAIMED by the README; not counted
directly).

No separate noise-free ("pristine") files exist in the label set: every file name has an `snr…db`
part and the levels are exactly the 26 listed.

## 2. Independent source samples per class (VERIFIED)

This is the table that answers the assignment's "at least 100 observations per class".

| Tag | Radio system | Sample ID range | Source samples | Usable | Quarantined |
|---|---|---|---|---|---|
| `target0` | DJI | 0–1279 | 1,280 | 1,065 | 215 |
| `target1` | FutabaT14 | 1280–4751 | 3,472 | 3,434 | 38 |
| `target2` | FutabaT7 | 4752–5552 | 801 | 801 | 0 |
| `target3` | Graupner | 5553–6353 | 801 | 770 | 31 |
| `target5` | Taranis | 6354–8016 | 1,663 | 1,637 | 26 |
| `target4` | Turnigy | 8017–8871 | 855 | 855 | 0 |
| `labnoise` | Wi-Fi/Bluetooth background | 8872–12813 | 3,942 | 3,942 | 0 |

| Binary class | Usable source samples | Usable images (× 26 SNR) |
|---|---|---|
| `drone_rf` (1) | **8,562** | 222,612 |
| `background_rf` (0) | **3,942** | 102,492 |
| quarantined | 310 | 8,060 |
| **total** | 12,814 | 333,164 |

Both classes have far more than 100 independent source samples, and the usable total of 12,504
exceeds the preferred ~5,000. The class ratio is about 2.2 drone : 1 background. Transmitter families
are unbalanced: FutabaT14 alone is 40% of the usable drone samples.

"Independent" needs a caveat, see section 4.

## 3. Label quality (VERIFIED)

- **Sample IDs are unique across tags** and run contiguously from 0 to 12,813, one block per tag
  (`artifacts/audit/sample_ids_under_multiple_tags.csv` is empty).
- **Labels are identical across the 26 SNR versions of a sample.** No sample has label text that
  differs between SNR levels. The boxes describe where the bursts are in the recording, not whether
  they are visible at that noise level. Consequence: at low SNR an image is labelled `drone_rf` even
  if the burst is buried in noise. That is the correct label for "is a drone signal present", and it
  is why accuracy must fall toward chance at very low SNR.
- **310 drone-tagged samples have zero boxes** (all 26 versions each, 8,060 images): 215 DJI,
  38 FutabaT14, 31 Graupner, 26 Taranis. We cannot tell whether the recording segment contains no
  burst or the automatic annotator missed it, so they are quarantined (`target_zero_boxes`). 16.8% of
  DJI samples are affected; the other families are at 0–3.9%.
- No `labnoise` label contains a box. No label has a class ID that disagrees with its tag.
- Boxes per usable drone sample range from 1 to 13 (`artifacts/audit/per_sample.csv`).

## 4. Are neighbouring samples independent? (evidence says no)

The number of boxes in sample *N* and sample *N+1* is correlated within several tags (DJI −0.31,
FutabaT7 −0.20, Graupner −0.19), while randomly shuffled pairs from the same tag show about 0. A
negative correlation between neighbours is what consecutive cuts of one continuous recording with
periodic bursts would produce. We read this as evidence that **adjacent sample IDs are consecutive
segments of longer recordings**, not unrelated recordings. The dataset gives no recording or session
identifier, so the true number of independent recording sessions is **UNKNOWN** and is certainly
smaller than 12,814.

What we do about it: the split keeps blocks of 20 neighbouring IDs together
([SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md)). In the approval email and the report we say "12,504 usable
source segments" and state this limitation.

## 5. Image files

Images checked so far come from the partial image sample described in
[DOWNLOAD_AND_STORAGE.md](DOWNLOAD_AND_STORAGE.md). Results are recorded in section 6 once measured.

| Property | Status |
|---|---|
| 640×640, 8-bit grayscale PNG | see section 6 |
| One image per label file for all 333,164 | CLAIMED (README); consistent with archive size |
| Full archive MD5 | not checked (archive not downloaded) |
| Axis orientation, sample rate, bandwidth | UNKNOWN (not in README) |

## 6. Image sample checks

*Filled in after the pilot image sample is fetched.*

## 7. What this audit does not establish

- That the automatic boxes are correct. They are energy-based annotations, not human labels.
- That the drone and background recordings are comparable. They were made in different places
  (CLAIMED: anechoic chamber vs. university building); see `CONFOUNDS_AND_LIMITATIONS.md`.
- The number of independent recording sessions.
