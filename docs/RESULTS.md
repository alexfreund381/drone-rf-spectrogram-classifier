# Results

**State on 2026-11-12: validation results for the baselines (A0), the scratch CNN (A1) and the
pretrained ResNet18 (B1), all trained on clean images (SNR 10–20 dB) only.** Models trained on all SNR
levels are not finished. **The test split has not been touched.** Everything here may change the choice of models; nothing here is a final result.

All numbers are from real images (the 10,407-image pilot sample), frozen split `ddfcbce03785a3af`.

| Source file | Contents |
|---|---|
| `artifacts/results/compare_val_snr10to20.csv` | Table 1 |
| `artifacts/results/compare_val_all_snr.csv` | Table 2 |
| `artifacts/results/masked_bursts_<run>_val_*.json` | Hidden-burst test |
| `artifacts/results/intensity_by_class_and_snr_train.csv` | Brightness by class and SNR |
| `artifacts/figures/val_snr10to20/`, `artifacts/figures/val_all_snr/` | Curves, confusion matrices, SNR plots |

Per-epoch logs and predictions are in `runs/` (git-ignored; reproducible with the commands in the
README).

## Runs

| Run | Model | Trained on | Details |
|---|---|---|---|
| A0 | Majority class; brightness (1 number); energy (39 numbers) + logistic regression | 1,616 training images at SNR 10–20 dB | No tuning |
| A1 | Scratch CNN, 388,834 parameters, random initialization | same 1,616 images (531 background, 1,085 drone) | AdamW, learning rate 0.001, cosine schedule, batch 32, no augmentation, seed 42. Stopped early after 18 epochs; best epoch 13. 43 minutes on CPU |

| B1 | ResNet18 with ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`), 11,177,538 parameters | same 1,616 images | Stage 1: backbone frozen, 3 epochs, learning rate 0.001 (1,026 trainable parameters). Stage 2: all layers, learning rate 0.0001, stopped early after 6 epochs. Best epoch 5 overall (the second fine-tuning epoch). 24 minutes on CPU |

A1 took about 143 seconds per epoch (about 11 images/s), slower than the 25–33 images/s benchmarked
beforehand; the laptop was on battery power, which may explain it. B1 took about 35 seconds per
head-only epoch and about 4 minutes per fine-tuning epoch.

**Choosing the fine-tuning length (B1).** Validation macro-F1 by epoch: 0.969, 0.975, 0.989 with the
backbone frozen, then 1.000 after the first fine-tuning epoch and 1.000 / 0.997 / 1.000 / 0.997 / 1.000
after that. One or two fine-tuning epochs are enough on this data; more only adds noise. The saved
checkpoint is epoch 5 (lowest validation loss among the perfect epochs).

**Correction.** B1's `meta.json` originally recorded its weights as "random" because of a labelling
bug: the label was read after the model had been rebuilt from its checkpoint. The run did use ImageNet
weights (its config says `pretrained: true`, and the frozen-backbone stage reached 0.969 with only the
1,026-parameter head training, which random features would not do). The bug is fixed, a regression
test was added, and the file was corrected with a note.

## Table 1 — Validation, SNR 10–20 dB (the condition the models were trained on)

395 images (138 background, 257 drone) from 98 groups.

| Run | Accuracy | Balanced accuracy | Macro-F1 (95% interval) | Drone recall | Drone precision | False-positive rate |
|---|---|---|---|---|---|---|
| A0 majority | 0.651 | 0.500 | 0.394 (0.352–0.430) | 1.000 | 0.651 | 1.000 |
| A0 brightness | 0.825 | 0.767 | 0.787 (0.700–0.866) | 0.961 | 0.807 | 0.428 |
| **A0 energy** | **1.000** | 1.000 | 1.000 (1.000–1.000) | 1.000 | 1.000 | 0.000 |
| A1 scratch CNN | 0.997 | 0.998 | 0.997 (0.992–1.000) | 0.996 | 1.000 | 0.000 |
| B1 pretrained ResNet18 | 1.000 | 1.000 | 1.000 (1.000–1.000) | 1.000 | 1.000 | 0.000 |

Intervals are group bootstraps (resampling the 98 groups). Confusion matrices, rows = true,
columns = predicted, order [background, drone]: A1 `[[138, 0], [1, 256]]`, B1 `[[138, 0], [0, 257]]`.

**Reading.** At high SNR the task is solved by 39 simple brightness numbers and a linear classifier.
Both networks match that; neither can beat it. The scratch CNN makes one error and ResNet18 none,
a difference of a single image that means nothing. A near-perfect score here says nothing about
whether a model recognises drone bursts, and this condition cannot rank the two networks.

## Table 2 — Validation, all 26 SNR levels (models still trained on SNR 10–20 dB only)

1,671 images (528 background, 1,143 drone) from 98 groups. This tests the models far outside the
noise levels they were trained on.

| Run | Accuracy | Balanced accuracy | Macro-F1 (95% interval) | Drone recall | Drone precision | False-positive rate |
|---|---|---|---|---|---|---|
| A0 majority | 0.684 | 0.500 | 0.406 (0.371–0.436) | 1.000 | 0.684 | 1.000 |
| A0 brightness | 0.480 | 0.555 | 0.480 (0.426–0.523) | 0.352 | 0.759 | 0.242 |
| A0 energy | 0.671 | 0.759 | 0.670 (0.609–0.717) | 0.519 | 1.000 | 0.000 |
| A1 scratch CNN | 0.713 | 0.773 | 0.708 (0.649–0.749) | 0.610 | 0.953 | 0.064 |
| B1 pretrained ResNet18 | 0.675 | 0.739 | 0.672 (0.614–0.712) | 0.565 | 0.934 | 0.087 |

Confusion matrices: A1 `[[494, 34], [446, 697]]` (misses 446 of 1,143 drone images);
B1 `[[482, 46], [497, 646]]` (misses 497).

**Reading.** Trained on clean images only, neither network copes with noise, and the pretrained
ResNet18 is no better than the scratch CNN here; its numbers are slightly lower, but the intervals
overlap, so we do not claim a difference. Both are close to the energy baseline and barely above
always guessing "drone".

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

The same levels for B1 (pretrained ResNet18), with A1 repeated for comparison:

| SNR (dB) | Images | A1 drone recall | B1 drone recall | A1 false-positive rate | B1 false-positive rate |
|---|---|---|---|---|---|
| −30 to −12 | 614 | 0.00–0.19 | 0.00 | 0.00–0.15 | 0.00–0.18 |
| −10 | 55 | 0.41 | 0.16 | 0.17 | 0.33 |
| −8 | 70 | 0.77 | 0.50 | 0.14 | 0.36 |
| −6 | 63 | 0.96 | 0.72 | 0.24 | 0.29 |
| −4 | 73 | 1.00 | 0.94 | 0.10 | 0.25 |
| −2 | 73 | 1.00 | 0.96 | 0.33 | 0.33 |
| 0 | 75 | 1.00 | 1.00 | 0.09 | 0.05 |
| 2 | 55 | 1.00 | 1.00 | 0.21 | 0.13 |
| 4 to 20 | 593 | 0.97–1.00 | 1.00 | 0.00 | 0.00–0.05 |

Between −10 and −4 dB the scratch CNN finds more drone images than ResNet18 at every level, with
fewer or equal false alarms. Each level has 55–73 images, so this is a consistent pattern across five
neighbouring levels rather than a proven difference.

**Reading.**
- From 0 dB upward both the CNN and the energy baseline find essentially every drone image.
- Between about −10 and −2 dB the CNN finds clearly more drone images than the energy baseline
  (for example 0.77 against 0.13 at −8 dB). This is the only range where the CNN shows an advantage.
  It pays for it with false alarms on background images (0.09–0.33) that the baseline does not make.
- At −16 dB and below both call nearly everything "background". Their accuracy there (about 0.25–0.33)
  is simply the share of background images, and is worse than always guessing "drone".
- Each SNR level has only 52–84 images from different recordings, so individual points are noisy.
  The dips at −2 dB and +2 dB may be noise.

## By transmitter family (validation, all SNR levels)

Share classified correctly. B1: labnoise 0.913, Turnigy 0.685, Taranis 0.607, FutabaT14 0.545,
Graupner 0.539, FutabaT7 0.527, DJI 0.510. A1:

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

## Hidden-burst test (A1 and B1)

Every annotated burst in each validation drone image was replaced by pixels drawn from the rest of the
same image (about 1.4% of pixels changed), and the model was asked again.

| Model | Condition | Drone images | Recall, original | Recall, bursts hidden |
|---|---|---|---|---|
| A1 scratch CNN | SNR 10–20 dB | 257 | 0.996 | 0.988 |
| A1 scratch CNN | All SNR levels | 1,143 | 0.610 | 0.505 |
| B1 ResNet18 | SNR 10–20 dB | 257 | 1.000 | 0.992 |
| B1 ResNet18 | All SNR levels | 1,143 | 0.565 | 0.530 |

A1 by SNR level:

| SNR (dB) | Drone images | Recall, original | Recall, bursts hidden |
|---|---|---|---|
| −10 | 37 | 0.41 | 0.03 |
| −8 | 48 | 0.77 | 0.06 |
| −6 | 46 | 0.96 | 0.30 |
| −4 | 53 | 1.00 | 0.72 |
| −2 | 52 | 1.00 | 0.87 |
| 0 | 53 | 1.00 | 0.93 |
| 2 to 20 | 430 | 0.97–1.00 | 0.97–1.00 |

B1 by SNR level: recall with bursts hidden is 0.05 at −10 dB (0.16 unmasked), 0.31 at −8 dB (0.50),
0.54 at −6 dB (0.72), 0.81 at −4 dB (0.94), and 0.90–1.00 from −2 dB upward.

**Reading.** At SNR 10–20 dB, hiding the drone bursts barely changes the answer for either network:
the scratch CNN still says "drone" for 98.8% of drone images and ResNet18 for 99.2%. At high SNR
neither is relying on the bursts; they use something else about the image. Between −10 and −4 dB,
hiding the bursts removes most of the scratch CNN's detections, so in that range the bursts are what
it responds to. ResNet18 loses less when the bursts are hidden (0.565 to 0.530 overall, against 0.610
to 0.505), which suggests it leans on the bursts less than the scratch CNN does.

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
- They show that networks trained only on clean images fail on noisy ones, pretrained or not. Whether
  training on all SNR levels fixes that is runs C1 and C1b, not yet done.
- They do not show that ImageNet pretraining helps on this task. On clean images there is no room to
  help; on noisy images the pretrained model was not better.
- They suggest the informative comparison between models is in the −12 to 0 dB range.
- They are validation results on a sample with few images per SNR level. They are not test results.
- Nothing here applies to drone signals recorded in a real radio environment.

## Not yet run

C1 and C1b (training on all SNR levels), baselines trained on all SNR levels, A2 (augmentation), a
randomly initialised ResNet18 (to separate architecture from pretraining), any input-size ablation,
the error gallery, and the single final test-set evaluation.
