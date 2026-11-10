# Presentation Notes

Slide-by-slide source material. Format and rules are in
[../DOCUMENTATION_GUIDE.md](../DOCUMENTATION_GUIDE.md) §4. Build the deck from this file.

**State on 2026-11-10:** dataset counts and model sizes are filled in from the audit. No experiment
has been run, so every results slide is still a placeholder. Speakers and time limits are unassigned because the presentation length has not been
given to us.

| Section | Slides | Speaker |
|---|---|---|
| 1. Problem and data | 1–5 | TBD (workstream A) |
| 2. Making the experiment trustworthy | 6–8 | TBD (workstream B) |
| 3. Models and training | 9–12 | TBD (workstream C) |
| 4. Results, limits, demo | 13–18 | TBD (workstream D) |

---

## Section 1 — Problem and data

## Slide 1: Can a CNN tell drone radio signals from Wi-Fi by looking at a spectrogram?

**The one thing the audience should remember:** We classify images of radio activity, not photos of
drones.

**On the slide:**
- Title, team names
- One drone spectrogram next to one background spectrogram

**Visual:** TO CREATE: side-by-side pair from the extracted subset (`visualization.py` gallery).

**Say:** Drones and their controllers communicate by radio. We asked whether a neural network can
recognize that radio traffic in an image and tell it apart from everyday Wi-Fi and Bluetooth. We built
one network from scratch and compared it with a pretrained ResNet18.

**Likely question:** Is this radar? → `docs/qa/CONCEPTS_QA.md` "Is this radar?"

**Evidence:** n/a

## Slide 2: A spectrogram turns a radio recording into a picture

**The one thing the audience should remember:** Brightness is signal power at a given time and
frequency.

**On the slide:**
- Recording → short slices → frequencies in each slice → image
- Bright = strong signal (log scale)

**Visual:** TO CREATE: annotated spectrogram. Do not label the axes until orientation is confirmed
from the paper.

**Say:** Like sheet music shows which notes sound when, a spectrogram shows which radio frequencies
are active when. Each drone system leaves its own pattern of short bursts.

**Likely question:** What does one pixel mean? → Q&A "What is a spectrogram…"

**Evidence:** CLAIMED (dataset README); axes UNKNOWN

## Slide 3: The data comes from six drone radio systems and one background class

**The one thing the audience should remember:** Seven source tags become two classes: drone RF and
background RF.

**On the slide:**
- DJI, FutabaT14, FutabaT7, Graupner, Turnigy, Taranis → `drone_rf`
- Wi-Fi/Bluetooth → `background_rf`
- Source: Zenodo, CC BY 4.0, Glüge et al. 2026

**Visual:** TO CREATE: one example image per tag.

**Say:** The dataset was published with a paper on object detection. We reuse its images and automatic
box annotations to define a yes/no label per image.

**Likely question:** Why not object detection? → Q&A "The dataset has bounding boxes…"

**Numbers for the slide (VERIFIED, `docs/DATASET_AUDIT.md` §2):** usable source samples — DJI 1,065,
FutabaT14 3,434, FutabaT7 801, Graupner 770, Turnigy 855, Taranis 1,637 (8,562 drone in total);
background 3,942.

**Evidence:** tag meanings CLAIMED (README); counts VERIFIED (`artifacts/audit/samples_by_tag.csv`)

## Slide 4: 101 GB sounds huge, but it is the same recordings 26 times

**The one thing the audience should remember:** Count independent recordings, not files.

**On the slide:**
- Each source sample × 26 noise levels (−30 to +20 dB)
- 333,164 images = 12,814 source samples × 26 noise levels
- Usable: 8,562 drone and 3,942 background source samples

**Visual:** TO CREATE: one sample shown at five SNR levels.

**Say:** Every recording appears at 26 noise levels. That is useful for studying robustness, but it is
not 26 times more evidence. All our counts separate images from independent samples.

**Likely question:** Are the noise copies useless? → Q&A "The dataset is 101 GB…"

**Evidence:** VERIFIED, `docs/DATASET_AUDIT.md` §1–2

## Slide 5: Labels are automatic, so we set uncertain images aside

**The one thing the audience should remember:** We do not trust a filename alone; ambiguous images are
quarantined and counted.

**On the slide:**
- Drone tag + ≥1 box → drone
- Background tag + no boxes → background
- Anything else → quarantine with a reason

**Visual:** TO CREATE: labeling-rule table; quarantine counts by reason.

**Say:** The boxes were produced by an automatic procedure. Some drone-tagged files have no boxes. We
do not guess; we set them aside and report how many.

**Likely question:** How many did you exclude? → 310 of 8,872 drone-tagged source samples (8,060
images); 215 of them are DJI.

**Evidence:** VERIFIED, `artifacts/audit/quarantine_by_reason_and_tag.csv`

*Handoff:* "With labels defined, the next risk is fooling ourselves in how we split the data."

---

## Section 2 — Making the experiment trustworthy

## Slide 6: Noise copies of one recording must stay in one split

**The one thing the audience should remember:** A random split would leak test recordings into
training.

**On the slide:**
- Wrong: shuffle images → same recording in train and test
- Right: all 26 noise versions, and blocks of 20 neighbouring samples, go to one split
- 646 groups: 450 train / 98 validation / 98 test
- Automated test fails on any overlap

**Visual:** TO CREATE: leakage diagram (wrong vs. right).

**Say:** If a recording's clean version is in training and its noisy version is in the test set, the
test is not a real test. We split by recording group, freeze the split, and never change it.

**Likely question:** Is grouping by sample enough? → Q&A "Why can't we just shuffle…"

**Evidence:** VERIFIED, `docs/SPLIT_PROTOCOL.md` (split hash `ddfcbce03785a3af`)

## Slide 7: A simple baseline tells us whether the CNN is doing anything clever

**The one thing the audience should remember:** Every CNN number is shown next to a trivial baseline.

**On the slide:**
- Majority class
- Brightness/energy features + logistic regression

**Visual:** `PLACEHOLDER` baseline vs. CNN bar chart.

**Say:** Before trusting a neural network's score, we check what a rule based only on overall
brightness achieves.

**Evidence:** `PLACEHOLDER — run A0`

## Slide 8: The two classes were recorded in different places, and that limits our claims

**The one thing the audience should remember:** Drone = anechoic chamber, background = university
building; the model may partly learn the room.

**On the slide:**
- The confound
- How we probed it
- What we found: `PLACEHOLDER`

**Visual:** `PLACEHOLDER` class-wise intensity statistics.

**Say:** This is the main limitation of the dataset and we want to state it ourselves. Here is how we
measured how much it matters.

**Likely question:** Then what is the result worth? → Q&A "Isn't your model just learning which room…"

**Evidence:** `PLACEHOLDER — CONFOUNDS_AND_LIMITATIONS.md`

*Handoff:* "With a split we trust, here are the two models."

---

## Section 3 — Models and training

## Slide 9: A CNN slides small filters over the image to find local patterns

**The one thing the audience should remember:** Early layers find edges and blobs; deeper layers
combine them into burst patterns.

**Visual:** TO CREATE: convolution/pooling schematic.

**Evidence:** n/a (concept)

## Slide 10: Our scratch CNN — four convolution stages, trained from random weights

**The one thing the audience should remember:** We designed and trained this network ourselves with no
pretrained weights.

**On the slide:**
- 224×224 → 112 → 56 → 28 → 14, channels 32 → 64 → 128 → 256, then average pooling and 2 outputs
- 388,834 parameters (ResNet18 has 11,177,538)

**Evidence:** VERIFIED, output of `python -m drone_rf.cli model-summary`

## Slide 11: ResNet18 reuses ImageNet knowledge through transfer learning

**The one thing the audience should remember:** We start from weights learned on photos, replace the
last layer, train the head, then fine-tune.

**On the slide:**
- Skip connections in one picture
- Freeze → train head → unfreeze at lower learning rate
- Grayscale copied to three channels

**Likely question:** Why would photo features help on spectrograms? → Q&A "Why two models…"

**Evidence:** n/a (concept); fine-tuning epochs `PLACEHOLDER`

## Slide 12: How we trained and chose settings without touching the test set

**The one thing the audience should remember:** All choices were made on validation data; the test set
was used once.

**Visual:** `PLACEHOLDER` training/validation curves.

**Evidence:** `PLACEHOLDER — runs/`

*Handoff:* "Here is what happened on recordings the models had never seen."

---

## Section 4 — Results, limits, demo

## Slide 13: Headline comparison — baseline vs. scratch CNN vs. ResNet18

**Evidence:** `PLACEHOLDER — no result yet`

## Slide 14: Where each model is wrong — confusion matrices

**Evidence:** `PLACEHOLDER — no result yet`

## Slide 15: Performance as noise increases

**The one thing the audience should remember:** `PLACEHOLDER` (which model degrades more gracefully).

**Visual:** `PLACEHOLDER` accuracy/recall vs. SNR with per-point denominators.

**Evidence:** `PLACEHOLDER — no result yet`

## Slide 16: Failure cases

**Visual:** `PLACEHOLDER` gallery of false positives and false negatives with source IDs.

**Evidence:** `PLACEHOLDER — no result yet`

## Slide 17: What this does and does not show

**On the slide:**
- Shows: separation of these recorded systems from this background, and the model comparison under
  added noise
- Does not show: real-world drone detection, unknown drone types, location of anything
- Labels are automatic; classes come from different environments

**Evidence:** limitations are known now; findings `PLACEHOLDER`

## Slide 18: Live demo and questions

**On the slide:** one command classifying a prepared spectrogram.

**Evidence:** `PLACEHOLDER — infer.py`
