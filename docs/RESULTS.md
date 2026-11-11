# Results

**State on 2026-11-11: validation results for the baselines (A0) and the scratch CNN (A1) only.**
The pretrained ResNet18 (B1) and the robustness runs have not been trained. **The test split has not
been touched.** Everything here may change the choice of models; nothing here is a final result.

All numbers are from real images (the 10,407-image pilot sample), frozen split `ddfcbce03785a3af`.

| Source file | Contents |
|---|---|
| `artifacts/results/compare_val_snr10to20.csv` | Table 1 |
| `artifacts/results/compare_val_all_snr.csv` | Table 2 |
| `artifacts/results/masked_bursts_a1_scratch_cnn_val_*.json` | Hidden-burst test |
| `artifacts/results/intensity_by_class_and_snr_train.csv` | Brightness by class and SNR |
| `artifacts/figures/val_snr10to20/`, `artifacts/figures/val_all_snr/` | Curves, confusion matrices, SNR plots |

Per-epoch logs and predictions are in `runs/` (git-ignored; reproducible with the commands in the
README).

## Runs

| Run | Model | Trained on | Details |
|---|---|---|---|
| A0 | Majority class; brightness (1 number); energy (39 numbers) + logistic regression | 1,616 training images at SNR 10–20 dB | No tuning |
| A1 | Scratch CNN, 388,834 parameters, random initialization | same 1,616 images (531 background, 1,085 drone) | AdamW, learning rate 0.001, cosine schedule, batch 32, no augmentation, seed 42. Stopped early after 18 epochs; best epoch 13. 43 minutes on CPU |

Training took about 143 seconds per epoch (about 11 images/s), slower than the 25–33 images/s
benchmarked the day before. The cause was not investigated.

## Table 1 — Validation, SNR 10–20 dB (the condition the models were trained on)

395 images (138 background, 257 drone) from 98 groups.

| Run | Accuracy | Balanced accuracy | Macro-F1 (95% interval) | Drone recall | Drone precision | False-positive rate |
|---|---|---|---|---|---|---|
| A0 majority | 0.651 | 0.500 | 0.394 (0.352–0.430) | 1.000 | 0.651 | 1.000 |
| A0 brightness | 0.825 | 0.767 | 0.787 (0.700–0.866) | 0.961 | 0.807 | 0.428 |
| **A0 energy** | **1.000** | 1.000 | 1.000 (1.000–1.000) | 1.000 | 1.000 | 0.000 |
| A1 scratch CNN | 0.997 | 0.998 | 0.997 (0.992–1.000) | 0.996 | 1.000 | 0.000 |

Intervals are group bootstraps (resampling the 98 groups). A1's confusion matrix, rows = true,
columns = predicted, order [background, drone]: `[[138, 0], [1, 256]]`.

**Reading.** At high SNR the task is solved by 39 simple brightness numbers and a linear classifier.
The CNN matches that but does not beat it (one error against zero, on 395 images). A near-perfect
score here says nothing about whether a model recognises drone bursts.

## Table 2 — Validation, all 26 SNR levels (models still trained on SNR 10–20 dB only)

1,671 images (528 background, 1,143 drone) from 98 groups. This tests the models far outside the
noise levels they were trained on.

| Run | Accuracy | Balanced accuracy | Macro-F1 (95% interval) | Drone recall | Drone precision | False-positive rate |
|---|---|---|---|---|---|---|
| A0 majority | 0.684 | 0.500 | 0.406 (0.371–0.436) | 1.000 | 0.684 | 1.000 |
| A0 brightness | 0.480 | 0.555 | 0.480 (0.426–0.523) | 0.352 | 0.759 | 0.242 |
| A0 energy | 0.671 | 0.759 | 0.670 (0.609–0.717) | 0.519 | 1.000 | 0.000 |
| A1 scratch CNN | 0.713 | 0.773 | 0.708 (0.649–0.749) | 0.610 | 0.954 | 0.064 |

A1 confusion matrix: `[[494, 34], [446, 697]]`. The CNN misses 446 of 1,143 drone images.

## Accuracy by SNR level (validation, 52–84 images per level)

Figure: `artifacts/figures/val_all_snr/snr_accuracy_val.png`.

| SNR (dB) | Images | A1 drone recall | A1 false-positive rate | Energy baseline drone recall |
|---|---|---|---|---|
| −30 to −16 | 499 | 0.00–0.02 | 0.00–0.07 | 0.00 |
| −14 | 56 | 0.17 | 0.15 | 0.00 |
| −12 | 59 | 0.19 | 0.09 | 0.00 |
| −10 | 55 | 0.41 | 0.17 | 0.05 |
| −8 | 70 | 0.77 | 0.14 | 0.13 |
| −6 | 63 | 0.96 | 0.24 | 0.44 |
| −4 | 73 | 1.00 | 0.10 | 0.72 |
| −2 | 73 | 1.00 | 0.33 | 0.87 |
| 0 | 75 | 1.00 | 0.09 | 1.00 |
| 2 | 55 | 1.00 | 0.21 | 1.00 |
| 4 to 20 | 593 | 0.97–1.00 | 0.00 | 0.98–1.00 |

The energy baseline's false-positive rate is 0 at every level.

**Reading.**
- From 0 dB upward both the CNN and the energy baseline find essentially every drone image.
- Between about −10 and −2 dB the CNN finds clearly more drone images than the energy baseline
  (for example 0.77 against 0.13 at −8 dB). This is the only range where the CNN shows an advantage.
  It pays for it with false alarms on background images (0.09–0.33) that the baseline does not make.
- At −16 dB and below both call nearly everything "background". Their accuracy there (about 0.25–0.33)
  is simply the share of background images, and is worse than always guessing "drone".
- Each SNR level has only 52–84 images from different recordings, so individual points are noisy.
  The dips at −2 dB and +2 dB may be noise.

## By transmitter family (A1, validation, all SNR levels)

| Tag | Images | Groups | Correct |
|---|---|---|---|
| labnoise (background) | 528 | 30 | 0.936 |
| target4 Turnigy | 111 | 7 | 0.757 |
| target5 Taranis | 247 | 13 | 0.664 |
| target2 FutabaT7 | 93 | 6 | 0.602 |
| target3 Graupner | 102 | 6 | 0.588 |
| target1 FutabaT14 | 435 | 26 | 0.568 |
| target0 DJI | 155 | 10 | 0.555 |

These pool all SNR levels, so they mostly reflect the low-SNR failures. Some families have only 6–7
validation groups; differences between families should not be read into yet.

## Hidden-burst test (A1)

Every annotated burst in each validation drone image was replaced by pixels drawn from the rest of the
same image (about 1.4% of pixels changed), and the model was asked again.

| Condition | Drone images | Recall, original | Recall, bursts hidden |
|---|---|---|---|
| SNR 10–20 dB | 257 | 0.996 | 0.988 |
| All SNR levels | 1,143 | 0.610 | 0.505 |

By SNR level:

| SNR (dB) | Drone images | Recall, original | Recall, bursts hidden |
|---|---|---|---|
| −10 | 37 | 0.41 | 0.03 |
| −8 | 48 | 0.77 | 0.06 |
| −6 | 46 | 0.96 | 0.30 |
| −4 | 53 | 1.00 | 0.72 |
| −2 | 52 | 1.00 | 0.87 |
| 0 | 53 | 1.00 | 0.93 |
| 2 to 20 | 430 | 0.97–1.00 | 0.97–1.00 |

**Reading.** At SNR 10–20 dB, hiding the drone bursts barely changes the answer: the CNN still says
"drone" for 98.8% of drone images. At high SNR it is therefore **not** relying on the bursts; it is
using something else about the image. Between −10 and −4 dB, hiding the bursts removes most of the
detections, so in that range the bursts are what it responds to.

Caveats: the boxes are automatic and may not cover all of a burst's energy; the replaced pixels are
not a perfect imitation of the background; and the model was never trained on images below 10 dB.

## Brightness by class and SNR (training images)

Mean pixel value on a 0–255 scale (`intensity_by_class_and_snr_train.csv`):

| SNR (dB) | Background | Drone | Gap |
|---|---|---|---|
| +20 | 160.9 | 128.5 | 32.4 |
| +10 | 156.1 | 139.2 | 16.9 |
| 0 | 163.4 | 153.2 | 10.2 |
| −10 | 179.3 | 169.9 | 9.4 |
| −20 | 191.0 | 189.2 | 1.8 |
| −30 | 201.2 | 204.9 | −3.7 |

Background images are brighter than drone images at high SNR, and the gap closes as noise is added.
This is the "something else" available to a classifier at high SNR, and it disappears exactly where
the models stop working.

## What these results do and do not show

- They show that on these recordings, at SNR of 0 dB and above, drone and background images are
  separable by simple global image statistics. That is consistent with the recording-environment
  confound (CONFOUNDS_AND_LIMITATIONS.md) and is **not** evidence of drone-signal recognition.
- They show that a CNN trained only on clean images fails on noisy ones. Whether training on all SNR
  levels fixes that is run C1, not yet done.
- They suggest the informative comparison between models is in the −12 to 0 dB range.
- They are validation results on a sample with few images per SNR level. They are not test results.
- Nothing here applies to drone signals recorded in a real radio environment.

## Not yet run

B1 (pretrained ResNet18), A2 (augmentation), C1 and C1b (training on all SNR levels), any input-size
ablation, the error gallery, and the single final test-set evaluation.
