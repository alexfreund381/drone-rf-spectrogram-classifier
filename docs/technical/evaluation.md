# Evaluation, comparison and figures

**Code:** `src/drone_rf/{evaluate,experiments,visualization,infer}.py`
**Tests:** `tests/test_evaluation.py`, `tests/test_train_smoke.py`
**Owner:** workstream D (unassigned)

## Purpose

Turn model predictions into numbers and pictures we can defend: correct formulas, a fixed class order,
separate counts for images and independent groups, results per SNR level, and uncertainty estimates
that respect the fact that SNR copies are not independent.

## Inputs and outputs

| | Format | Example |
|---|---|---|
| Input | Checkpoint | `runs/a1_scratch_cnn/best.pt` |
| Output | `metrics_<name>.json`, `predictions_<name>.csv` in the run directory | |
| Output | Comparison table | `artifacts/results/compare_val.csv` and `.md` |
| Output | Figures | `artifacts/figures/*.png` |

## How it works

**Class order is fixed:** index 0 = `background_rf` (negative), index 1 = `drone_rf` (positive).
The confusion matrix has rows = true class and columns = predicted class, in that order:
`[[TN, FP], [FN, TP]]`.

| Metric | Formula | Plain meaning |
|---|---|---|
| accuracy | (TP + TN) / N | share of all images classified correctly |
| recall (drone) | TP / (TP + FN) | of the drone images, how many were found |
| precision (drone) | TP / (TP + FP) | of the "drone" calls, how many were right |
| false-positive rate | FP / (FP + TN) | of the background images, how many raised a false alarm |
| F1 (per class) | harmonic mean of precision and recall | |
| macro-F1 | mean of the two classes' F1 | treats both classes equally |
| balanced accuracy | mean of the two classes' recall | |

A ratio with a zero denominator is reported as NaN, not 0.

**Per image and per group.** `per_image` counts every PNG. `per_group` averages the drone probability
over all images of a split group and thresholds at 0.5, so each group of related recordings counts
once.

**Per SNR.** `per_snr` gives all metrics for each SNR level, with `n` and the class supports for that
level.

**Per tag.** `per_tag` gives the share classified correctly for each transmitter family and for the
background.

**Confidence intervals.** `bootstrap_by_group` resamples *groups* with replacement 1,000 times and
recomputes the metric. Resampling images would treat SNR copies and neighbouring segments as
independent and give intervals that are too narrow.

**Test-set discipline.** `evaluate --split test` appends a line to `test_evaluations.log` in the run
directory, and refuses to run if the splits file hash differs from the checkpoint's.

## How to run it

```
python -m drone_rf.cli evaluate --checkpoint runs/a1_scratch_cnn/best.pt --split val --all-snr --name val_all_snr
python -m drone_rf.cli compare --runs runs --split val --out artifacts/results/compare_val.csv
python -m drone_rf.cli figures --runs runs --split val --image-dir data/selected --out artifacts/figures
python -m drone_rf.cli infer --checkpoint runs/a1_scratch_cnn/best.pt --image <spectrogram.png>
```

## Design decisions

| Decision | Alternatives considered | Why we chose this |
|---|---|---|
| Report images and groups separately | One pooled number | 26 SNR copies of a sample are not 26 observations |
| Bootstrap over groups | Bootstrap over images; none | Images within a group are correlated |
| Threshold fixed at 0.5 | Tune the threshold | A tuned threshold would have to be chosen on validation data; we keep it simple and the same for all models |
| NaN for undefined ratios | Report 0 | 0 would look like a real, terrible score |
| Own metric code with hand-checked tests | Only scikit-learn | We must be able to explain every formula; tests check them against a hand-computed example |
| `infer` reports "uncalibrated scores" | Call them probabilities | Softmax outputs are not calibrated probabilities |

## Tests

- TP/FP/FN/TN and every formula checked against a 10-image example worked by hand.
- Confusion matrix order; NaN for undefined ratios.
- Per-SNR denominators; group aggregation counts each group once.
- Bootstrap: zero-width interval when all images are one group; reproducible with a seed.

## Known limitations

- Group-level results depend on which SNR levels are present for each group.
- With few groups per tag in a split, per-tag numbers are noisy.
- No calibration analysis yet.
