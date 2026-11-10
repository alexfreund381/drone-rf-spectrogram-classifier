# Split Protocol

How the data is divided into training, validation and test sets, and how we know nothing leaks.

## The frozen split

| | |
|---|---|
| Command | `python -m drone_rf.cli split --manifest data/manifest/manifest.csv --out data/manifest/splits.csv --seed 42 --block-size 20` |
| Seed | 42 |
| Group key | class tag + block of 20 consecutive sample IDs (`<tag>_b<id // 20>x20`) |
| Fractions | 70% / 15% / 15% of groups, per class tag |
| **Split hash** | **`ddfcbce03785a3af`** |
| Frozen on | 2026-11-10, before any image was downloaded |
| Committed copy | `artifacts/splits_by_sample.csv` (one row per source sample) |

The full per-image file (`data/manifest/splits.csv`, 333,164 rows) is too large for git. Because
labels are identical across SNR versions, the per-sample table determines it completely; rebuilding
with the command above must reproduce the hash.

## Sizes (VERIFIED, from `artifacts/splits_by_sample.csv`)

| Split | Class | Groups | Source samples | Images (all 26 SNR) |
|---|---|---|---|---|
| train | background_rf | 138 | 2,760 | 71,760 |
| train | drone_rf | 312 | 5,944 | 154,544 |
| val | background_rf | 30 | 600 | 15,600 |
| val | drone_rf | 68 | 1,323 | 34,398 |
| test | background_rf | 30 | 582 | 15,132 |
| test | drone_rf | 68 | 1,295 | 33,670 |

Source samples per tag:

| Split | labnoise | DJI | FutabaT14 | FutabaT7 | Graupner | Turnigy | Taranis |
|---|---|---|---|---|---|---|---|
| train | 2,760 | 733 | 2,401 | 568 | 534 | 583 | 1,125 |
| val | 600 | 169 | 518 | 120 | 120 | 140 | 256 |
| test | 582 | 163 | 515 | 113 | 116 | 132 | 256 |

Experiments use subsets of these rows (particular SNR levels, or the images actually on disk). A
subset never moves a sample to another split.

## Rules

1. **All 26 SNR versions of a source sample are in the same split.** They are one recording with
   different amounts of added noise.
2. **Blocks of 20 neighbouring sample IDs are in the same split.** The audit found that neighbouring
   IDs are statistically related (DATASET_AUDIT.md §4), which suggests they are consecutive pieces of
   longer recordings. Keeping neighbours together stops the model from being tested on a segment
   recorded a moment after one it trained on.
3. **Stratified by class tag.** Each of the seven tags is shuffled and cut separately, so every split
   has every transmitter family in about the same proportion.
4. **Quarantined images have no split** and cannot be loaded by the dataset class.
5. **The split was made from label metadata only**, before any image had been downloaded.
6. **The test split is used once**, after model selection. Every test evaluation is appended to
   `test_evaluations.log` in the run directory.
7. **Never re-randomize.** `evaluate` refuses to run if the splits file's hash differs from the one
   stored in the checkpoint.

## How leakage is prevented in code

| Check | Where |
|---|---|
| No group in more than one split | `splits.assert_group_safe`, called by `assign_splits`, `train_run`, `evaluate_checkpoint`, `run_baselines` |
| No (tag, sample) in more than one split | same function |
| Quarantined rows carry no split | same function |
| Detector actually fires on a planted leak | `tests/test_splits.py::test_leak_detector_catches_leak` |
| Same seed gives the same file and hash; row order does not matter | `tests/test_splits.py` |
| Normalization statistics come from training images only | `train.train_run` (`train_ds.mean_std()`), `tests/test_dataset.py` |
| Training never reads the test split | `tests/test_train_smoke.py::test_run_directory_contents_and_log_fields` |

## Why block size 20

The dataset has no recording/session identifier, so any block size is a judgement call. Larger blocks
give more protection but fewer groups. With 20, the smallest tags (FutabaT7 and Graupner, 801 samples
each) still have about 40 groups, enough for a 70/15/15 cut with several groups in validation and
test.

## Remaining risk (stated openly)

- At each block boundary, one pair of neighbouring samples can still sit in different splits.
- If a whole tag was captured in one long session, every split shares that session's conditions. No
  split of this dataset can fix that.
- A possible follow-up experiment: compare validation scores under `--block-size 1` and
  `--block-size 20`. A large gap would measure how much neighbour leakage inflates results.
