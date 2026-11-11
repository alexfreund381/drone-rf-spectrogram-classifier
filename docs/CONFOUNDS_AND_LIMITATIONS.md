# Confounds and Limitations

What could make our results look better than they are, how we check for it, and what the project
cannot show regardless.

**State on 2026-11-11:** the checks have been run for the baselines and the scratch CNN (run A1) on
validation data. Findings are in section 8. They have not been run for ResNet18 or for models trained
on all SNR levels, because those models do not exist yet.

## 1. The recording-environment confound

**The problem.** According to the dataset README (CLAIMED), the six drone systems were recorded in an
anechoic chamber and the Wi-Fi/Bluetooth background in a public university building. Every drone
image therefore comes from one environment and every background image from another. A classifier can
score well by learning "chamber vs. building" (noise floor, overall brightness, density of other
signals) without learning what a drone signal looks like.

**What we have seen so far.**
- Two images viewed by eye at +20 dB: the drone image is a smooth background with a few thin bursts;
  the background image is filled with many narrow bursts. The classes look different as a whole, not
  only where the annotated bursts are.
- Measured over the 7,195 training images of the sample: background images are brighter than drone
  images at high SNR (160.9 against 128.5 at +20 dB), and the gap closes as noise is added (section 8).

**How we will measure it.**

| Check | Command | What a worrying result looks like |
|---|---|---|
| Brightness baseline: logistic regression on the mean pixel value | `baselines` (run A0) | Close to the CNN's score: the classes separate by brightness alone |
| Energy baseline: 39 global intensity and profile numbers | `baselines` (run A0) | Close to the CNN's score: no learned shapes are needed |
| Mean brightness per class and SNR level | `confound-check` → `intensity_by_class_and_snr_train.csv` | A consistent gap between classes at the same SNR |
| Hidden-burst test: every annotated burst in a drone image is replaced by pixels drawn from the rest of the same image, then the model is asked again | `confound-check` → `masked_bursts_*.json` | Drone recall stays high with the bursts hidden: the model is not using the bursts |
| Accuracy against SNR | `evaluate --all-snr`, `figures` | Accuracy stays high at −30 dB, where bursts should be invisible |

**What we cannot do.** We cannot remove the confound. The dataset has no drone signal recorded in the
building and no background recorded in the chamber. Whatever the checks show, the honest summary is
"separates these drone recordings from these background recordings".

## 2. Labels are automatic

The boxes come from the authors' energy-based segmentation, not from people (CLAIMED). Effects:

- 310 drone-tagged samples have no box at all and are quarantined (VERIFIED). DJI loses 16.8% of its
  samples this way, so DJI is under-represented relative to the raw data.
- Some quarantined samples probably do contain a drone signal the annotator missed, and some labelled
  samples may contain boxes around something else. We have not measured either rate.
- Labels are the same at every SNR level (VERIFIED). At very low SNR an image is labelled `drone_rf`
  although the burst may be invisible. Low accuracy there is expected and is not a model failure.

## 3. Sample independence

- 26 SNR versions of a sample are the same recording. They are always in one split.
- Neighbouring sample IDs appear to be consecutive slices of longer recordings (VERIFIED correlation
  of box counts between neighbours). Blocks of 20 neighbours share a split. Leakage at block edges
  and through shared recording sessions remains possible.
- The dataset has no session identifier, so the number of truly independent recordings is UNKNOWN.
- Consequence: confidence intervals are computed by resampling groups, and are still probably too
  narrow.

## 4. Class and family imbalance

- About 2.2 drone source samples per background sample. We report macro-F1, balanced accuracy and
  per-class recall, not accuracy alone.
- FutabaT14 is 40% of the usable drone samples; FutabaT7 and Graupner are 9% each. A model can score
  well overall while being weak on a small family, so we report results per family.

## 5. Preprocessing choices that may cost accuracy

- Images are resized from 640×640 to 224×224. Bursts only a few pixels wide become fainter. A larger
  input size is a planned ablation.
- ResNet18 receives the grayscale image copied into three channels with ImageNet normalization.
  ImageNet photos and spectrograms are very different kinds of image.

## 6. Limits of the image sample

- We train and evaluate on a 10,407-image sample (3.2 GB of 101 GB), not the full dataset.
- Each source sample appears at only one or two arbitrary SNR levels, so we cannot follow the *same*
  recording across all 26 levels. Per-SNR results compare different recordings at each level, with
  only 50–84 validation or test images per level.
- The full-archive checksum has not been verified.

## 7. What the project does not show

- That a drone can be detected in a real radio environment.
- That drone types outside these six systems would be recognised.
- Anything about where a transmitter is.
- That the model reacts to drone signal structure rather than recording conditions, unless the checks
  in section 1 support it.
- Comparability with the source paper: it reports object-detection mAP; we report image-level
  classification metrics.

## 8. Findings so far (validation data, baselines and scratch CNN A1)

Numbers and source files are in [RESULTS.md](RESULTS.md).

1. **The confound is real and large at high SNR.** A linear classifier on 39 global brightness numbers
   scores 1.000 accuracy on the 395 validation images at SNR 10–20 dB. No learned shape is needed.
2. **The classes differ in plain brightness.** At +20 dB background images average 160.9 and drone
   images 128.5 on a 0–255 scale. The gap shrinks as noise is added and is gone by about −20 dB.
   (An earlier note in this document, based on four images, had the direction the other way round;
   the table over 7,195 training images supersedes it.)
3. **At high SNR the scratch CNN does not use the drone bursts.** With every annotated burst painted
   over, it still calls 98.8% of drone images "drone" (99.6% unmasked).
4. **In the middle SNR range it does use them.** Between −10 and −4 dB, hiding the bursts cuts drone
   recall sharply (for example from 0.77 to 0.06 at −8 dB), and in that range the CNN finds more drone
   images than the brightness baseline.
5. **Below about −16 dB nothing works.** Both the CNN and the baseline call almost everything
   "background". This model was trained only on SNR 10–20 dB, so this is also an out-of-distribution
   failure, not only a physical limit.

**What this means for the project.** High-SNR accuracy is not a meaningful headline number on this
dataset. The comparison between the scratch CNN and ResNet18 should be made on models trained across
all SNR levels and judged mainly in the range where brightness alone fails, together with the
hidden-burst test.

**Not yet checked:** ResNet18; models trained on all SNR levels; whether the brightness gap comes from
the recording environment, the receiver settings, or the image scaling used by the dataset authors
(we cannot tell from the images alone).
