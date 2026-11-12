# Conceptual Q&A

Plain-language questions and answers about the project. Format and rules are in
[../DOCUMENTATION_GUIDE.md](../DOCUMENTATION_GUIDE.md) §3. Results quoted here are validation results for models trained on clean images only; the test set has
not been used.

All entries below are **drafts** and need review by a teammate.

---

## Part 1 — The problem

### Q: What is this project about, in one breath?

**Short answer:** We teach a neural network to look at a picture of radio activity and say whether it
contains a signal from a known drone remote-control system or only Wi-Fi/Bluetooth background.

**Explanation:** Drones and their controllers talk to each other by radio. A radio receiver can record
that traffic, and the recording can be drawn as an image called a spectrogram. Different radio systems
leave different visual patterns. Our classifier looks at one such image and outputs "drone RF present"
or "background only".

**In our project:** Binary image classification on the Zenodo dataset *Drone RF Emissions in
Spectrograms*. See PROJECT_PLAN.md §1.

**Follow-up you might get:** *Can it find where the drone is?* No. It only says whether a drone-type
signal is in the image. It gives no direction, distance or location.

*Status: draft · Evidence: n/a*

### Q: Is this radar?

**Short answer:** No. Radar sends out its own signal and listens for the echo. We only listen to
signals the drone system itself transmits.

**Explanation:** Radar is active: transmit, wait, measure the reflection. Our data is passive: a
receiver recorded the radio link between controller and drone. The physics, the data and the
conclusions are different, so the two must not be mixed.

**In our project:** RF emissions only. An earlier radar-based idea was dropped.

**Follow-up you might get:** *Why does that matter for the results?* A radar system could see a drone
that is radio-silent. Our approach cannot; it depends on the drone system transmitting.

*Status: draft · Evidence: CLAIMED (dataset README)*

### Q: What is a spectrogram, and what does one pixel mean?

**Short answer:** A spectrogram is a picture of how much radio energy there is at each frequency at
each moment. A brighter pixel means more signal power at that time and frequency.

**Explanation:** Think of sheet music: one direction is time, the other is pitch, and marks show which
notes sound when. A spectrogram does the same for radio. The recording is cut into short slices, a
Fourier transform works out which frequencies are present in each slice, and the results are placed
side by side. Brightness is on a logarithmic (decibel) scale so that weak and strong signals are both
visible in one image.

**In our project:** Each image is a 640×640 grayscale PNG (CLAIMED by the README; we measure it after
extraction). The README does not say which image axis is time and which is frequency, so we do not
label axes until the paper confirms it.

**Follow-up you might get:** *So is this really a computer-vision problem?* The input is an image and
the model is a CNN, but the pixels come from signal processing, not a camera. We are asking the
professor to confirm this counts.

*Status: draft · Evidence: CLAIMED / UNKNOWN (axes)*

### Q: What are I/Q samples?

**Short answer:** They are the raw numbers a radio receiver records: two values per instant that
together capture both the strength and the phase of the radio wave.

**Explanation:** A single number per instant would tell you how strong the wave is but not where it is
in its cycle. Recording two components, "in-phase" (I) and "quadrature" (Q), keeps both. That complete
description is what allows the Fourier transform to compute a spectrogram.

**In our project:** We never touch I/Q data. The authors converted it to spectrogram images, and added
noise at the I/Q stage. We work only with the finished PNGs.

*Status: draft · Evidence: CLAIMED*

---

## Part 2 — The data

### Q: The dataset is 101 GB. Doesn't that mean we have a huge number of examples?

**Short answer:** No. Most of that size is the same recordings repeated 26 times with different
amounts of artificial noise.

**Explanation:** Imagine photographing 100 different dogs, then making 26 copies of each photo with
increasing amounts of fog. You have 2,600 files but still only 100 dogs. For judging whether a model
generalizes, the number of *different* originals is what counts.

**In our project:** 333,164 images come from 12,814 source samples, each at exactly 26 noise levels
(VERIFIED, DATASET_AUDIT.md). All our tables report "independent source samples" separately from
"images".

**Follow-up you might get:** *Are the noise copies useless, then?* No. They are valuable for training
a noise-robust model and for testing how performance falls as noise rises. They just do not count as
extra independent evidence.

*Status: draft · Evidence: VERIFIED*

### Q: Why can't we just shuffle all the images and split them 70/15/15?

**Short answer:** Because noise copies of the same recording would land in both training and test,
and the model would be tested on things it has effectively already seen.

**Explanation:** This is called data leakage. If the +20 dB version of a recording is in training and
the +18 dB version is in the test set, the test is close to an open-book exam with the answer key.
The score would look excellent and mean very little. The fix is to split by *group*: decide for each
source sample which split it belongs to, and send every version of it there.

**In our project:** `splits.py` assigns a split per group, an automated test fails if any group
appears in two splits, and the split file is frozen with a hash. See PROJECT_PLAN.md §4.8.

**Follow-up you might get:** *Is grouping by sample enough?* Possibly not. If several samples were cut
from one long recording, neighbouring samples are also related. We may not be able to recover that
from the filenames, so we keep neighbouring IDs together and state the remaining risk.

*Status: draft · Evidence: design decision*

### Q: What is SNR, and what does "noise was added at −30 dB to +20 dB" mean?

**Short answer:** SNR is signal-to-noise ratio: how strong the signal is compared with the noise.
High SNR means a clean signal; a negative SNR in decibels means the noise is stronger than the signal.

**Explanation:** Decibels are a logarithmic scale. Every +10 dB is ten times more signal power
relative to noise. At +20 dB the signal has 100 times the noise power; at 0 dB they are equal; at
−30 dB the noise is 1,000 times stronger. AWGN (additive white Gaussian noise) is the standard model
of random receiver noise: it is added on top of the signal, it is spread evenly over all frequencies,
and its values follow a bell curve.

**In our project:** The authors made 26 versions of each sample, from −30 to +20 dB in 2 dB steps, by
adding noise to the raw I/Q data before making the image (CLAIMED). We use these to plot accuracy
against SNR.

**Follow-up you might get:** *Is adding random noise to the pixels the same thing?* No. Their noise
went through the same signal processing as the signal. Noise added to a finished image behaves
differently, so if we try it we call it "pixel noise".

*Status: draft · Evidence: CLAIMED*

### Q: The dataset has bounding boxes. Why aren't we doing object detection?

**Short answer:** The assignment asks for an image classifier built from scratch and a pretrained
comparison. We use the boxes only to decide the yes/no label for each image.

**Explanation:** The authors trained a YOLO detector that draws a rectangle around each signal burst.
We ask a simpler question: is there any drone burst in this image at all? An image with at least one
box is "drone"; a background recording with no boxes is "background".

**In our project:** Labeling rules are in PROJECT_PLAN.md §4.6. Because our task differs, the paper's
detection score (mAP) cannot be compared with our accuracy or F1.

**Follow-up you might get:** *Are the boxes reliable?* They were produced automatically by an
energy-based procedure, not drawn by people. We already saw drone-tagged files with no boxes at all
(VERIFIED in a small probe), so we treat those as uncertain and set them aside.

*Status: draft · Evidence: VERIFIED (empty drone-tagged labels exist), CLAIMED (annotation method)*

### Q: What does "quarantine" mean and why not just delete bad files?

**Short answer:** Quarantine means we set an image aside with a written reason instead of using it or
quietly discarding it.

**Explanation:** If a drone-tagged image has no boxes, we do not know whether the drone signal is
absent, too weak to see, or missed by the automatic annotator. Calling it "drone" or "background"
would both be guesses. Keeping a list with reasons lets us report how many there are and check whether
they cluster in one class or at low SNR.

**In our project:** The manifest has `binary_label = quarantine` and a `quarantine_reason` column.

*Status: draft · Evidence: design decision*

---

## Part 3 — The hard questions

### Q: Isn't your model just learning which room the recording was made in?

**Short answer:** It might be, and we cannot fully rule it out with this dataset. We measure how
likely it is and report it.

**Explanation:** Drone signals were recorded in an anechoic chamber (a room built to have no echoes or
outside signals). Background was recorded in a university building. So every drone image comes from
one environment and every background image from another. A model could separate them using the
general look of the recording rather than the drone signal. This is called a confound.

**In our project:** We tested this, and at high SNR the answer is largely yes. A baseline using only
39 brightness numbers classified every clean validation image correctly, and when we painted over
every drone burst the CNN still said "drone" for 98.8% of drone images. Background images are simply
brighter than drone images at high SNR. Between about −10 and −4 dB the picture changes: there the
CNN finds more drone images than the brightness baseline, and hiding the bursts removes most of its
detections. When we trained the CNN on images at all noise levels instead, it reached 0.962 accuracy against
0.911 for the brightness baseline trained the same way, and hiding the bursts cut its drone detections
by more than half. So a model trained that way does respond to the drone signal, although most of the
separation is still available from brightness-type statistics. (Validation data; `docs/RESULTS.md`.)

**Follow-up you might get:** *Then what is your result worth?* It shows how well these methods
separate these recordings and how the two models compare under noise. It does not show that the system
would work on drone signals recorded in a real, busy radio environment.

*Status: draft · Evidence: CLAIMED (recording locations); VERIFIED on validation data for run A1*

### Q: Why build a simple baseline when we have neural networks?

**Short answer:** A baseline tells us whether the neural network is doing anything clever.

**Explanation:** If always guessing the majority class already gives 70% accuracy, a CNN at 72% is
not impressive. If a three-number brightness rule reaches 99%, a CNN at 99% has not proved it learned
signal patterns. Baselines put the headline number in context.

**In our project:** Run A0. On clean validation images (SNR 10–20 dB): always guessing "drone" gives
0.651 accuracy, one brightness number gives 0.825, 39 brightness numbers give 1.000, and our scratch
CNN gives 0.997. Without the baseline we would have presented 0.997 as a success; with it, we know
the clean-image task is trivial on this dataset (`docs/RESULTS.md`).

*Status: draft · Evidence: n/a*

### Q: Why two models, and what do "from scratch" and "pretrained" mean?

**Short answer:** "From scratch" starts with random weights and learns only from our data.
"Pretrained" starts with weights already learned on a large photo collection (ImageNet) and adapts
them. The assignment requires both, and comparing them is our research question.

**Explanation:** A pretrained network already knows how to find edges, textures and shapes in photos.
Reusing that is called transfer learning and usually helps when data is limited. Spectrograms are not
photos, though, so whether it helps here is something we have to measure rather than assume.

**In our project:** Model A is our own four-stage CNN. Model B is ResNet18 with ImageNet weights, with
the grayscale image copied into three channels. Both use the same split and evaluation.

**What we found so far (validation only):** on clean images both score essentially 100%, as does a
trivial brightness baseline, so that comparison is uninformative. On noisy images, after training on
clean images only, ResNet18 was not better than the scratch CNN (0.675 against 0.713 accuracy, with
overlapping uncertainty). So far we have no evidence that ImageNet pretraining helps here
(`docs/RESULTS.md`).

**Follow-up you might get:** *If ResNet18 wins, is that because of pretraining or because it is a
bigger network?* Fair point; the comparison changes two things at once. An optional run of ResNet18
with random weights would separate the two effects.

*Status: draft · Evidence: VERIFIED on validation data (runs A1, B1)*

### Q: How did you use AI tools, and do you understand the code?

**Short answer:** AI assistance is permitted for this assignment. We used it to help plan, write and
document code, and each of us is responsible for understanding and explaining what we submit.

**In our project:** Every pull request requires that the author can explain the change, and every Q&A
entry is reviewed by a teammate who did not write that part. See DOCUMENTATION_GUIDE.md §6–7.

*Status: draft*

---

## Part 4 — How the models work

### Q: What does a convolution layer actually do?

**Short answer:** It slides a small grid of numbers (a filter) over the image and, at each position,
measures how well the patch under it matches the filter's pattern.

**Explanation:** Think of a stencil shaped like a short vertical line. Slide it across the picture;
wherever a vertical line sits under it, the response is large. A layer has many such filters, and the
network learns their numbers during training instead of us designing them. Early layers end up with
simple detectors (edges, blobs); later layers combine those into larger patterns such as "several
thin bursts in a row".

**In our project:** `ScratchCNN` has four stages. Each uses 3×3 filters: 32 in the first stage, then
64, 128 and 256 (`src/drone_rf/models.py`).

**Follow-up you might get:** *Why 3×3?* Two stacked 3×3 layers see a 5×5 area with fewer weights than
one 5×5 layer, and it is the standard choice.

*Status: draft*

### Q: What is pooling and why do we halve the image four times?

**Short answer:** Max pooling keeps the strongest response in each 2×2 block, shrinking the image by
half per side. It makes the network cheaper and less sensitive to exactly where a pattern is.

**Explanation:** After detecting "there is a burst edge around here", the exact pixel rarely matters.
Pooling throws away that precision. Each halving also lets the next layer's 3×3 filters cover a
larger part of the original image.

**In our project:** 224 → 112 → 56 → 28 → 14 pixels per side. A final *global average pooling* turns
the 14×14 map of 256 features into 256 numbers, so a burst contributes wherever it appears.

**Follow-up you might get:** *Could pooling erase very thin bursts?* Max pooling keeps the strongest
value, so thin bright bursts survive. The larger risk is the earlier resize from 640 to 224 pixels,
which averages a burst only a few pixels wide with its surroundings. Testing a larger input size is a
planned ablation.

*Status: draft*

### Q: What are batch normalization and dropout for?

**Short answer:** Batch normalization keeps the numbers flowing through the network in a steady
range so training is stable. Dropout randomly switches off some features during training so the
network cannot rely on any single one.

**In our project:** Batch norm follows every convolution. Dropout (30%) sits just before the final
layer. Both behave differently in training and evaluation, which is why the code calls
`model.train()` and `model.eval()`. When the ResNet backbone is frozen we also keep its batch-norm
statistics fixed.

*Status: draft*

### Q: How does the network learn? What are the loss, gradients and AdamW?

**Short answer:** The loss is a number saying how wrong the predictions are. Gradients say how to
nudge each weight to reduce it. The optimizer (AdamW) applies those nudges, a small step at a time.

**Explanation:** *Cross-entropy loss* is small when the network gives high probability to the correct
class and grows quickly when it is confidently wrong. *Backpropagation* computes, for every weight,
whether increasing it would raise or lower the loss. *AdamW* adjusts each weight using a running
average of its recent gradients and also shrinks weights slightly each step (*weight decay*), which
discourages overly complicated solutions. The *learning rate* sets the step size.

**In our project:** Cross-entropy on two outputs, AdamW, learning rate 0.001 for the scratch CNN and
0.0001 when fine-tuning ResNet18 (`configs/`).

**Follow-up you might get:** *Why a smaller learning rate for ResNet18?* Its weights already encode
useful features; large steps would destroy them.

*Status: draft*

### Q: What is overfitting and how do we guard against it?

**Short answer:** Overfitting is memorising the training images instead of learning what generalises.
We watch performance on validation images the model never trains on, and keep the version that did
best there.

**In our project:** After every epoch we measure validation macro-F1. The best epoch's weights are
saved as `best.pt`; training stops early if there is no improvement for several epochs. The
training curves (train vs. validation loss) show the gap.

**Follow-up you might get:** *Then isn't the validation score also optimistic?* Slightly, yes, because
we chose the best epoch using it. That is exactly why a separate test set exists.

*Status: draft*

### Q: What is special about ResNet18?

**Short answer:** It has "skip connections": each block adds its input back to its output, which
makes deep networks much easier to train.

**Explanation:** In a plain deep network, the learning signal weakens as it travels back through many
layers. A skip connection gives it a shortcut. Each block then only has to learn a *correction* to
what it received (the "residual"), which is where the name comes from. ResNet18 has 18 layers with
weights and about 11 million parameters.

**In our project:** `ResNet18Classifier` wraps torchvision's ResNet18 (11,177,538 parameters; our
scratch CNN has 388,834).

*Status: draft*

### Q: What do "freezing" and "fine-tuning" mean?

**Short answer:** Freezing means holding the pretrained layers fixed and training only the new final
layer. Fine-tuning means then letting all layers adjust, gently.

**Explanation:** The new final layer starts with random numbers. If everything trained at once, its
large early errors would push big, damaging changes back into the pretrained layers. So we first
train only the head, then unfreeze and continue with a ten times smaller learning rate.

**In our project:** With the backbone frozen only 1,026 of 11,177,538 parameters train. Stage lengths
are in `configs/resnet18.yaml`; the epoch that performs best on validation data is kept.

**Follow-up you might get:** *ImageNet has photos of dogs and cars. Why would it help here?* The early
layers detect generic edges and textures, which also exist in spectrograms. Whether that is enough to
help is something we measure rather than assume.

*Status: draft*

---

## Part 5 — Measuring results

### Q: Why not just report accuracy?

**Short answer:** Accuracy hides which kind of mistake is being made, and it flatters a model when
one class is more common.

**Explanation:** About two thirds of our images are drone images. A model that always says "drone"
would be about 68% accurate and completely useless. So we also report *recall* (of the real drone
images, how many did we catch), *precision* (of the images we called drone, how many were),
the *false-positive rate* (how often background raises a false alarm), and *macro-F1*, which averages
the F1 score of both classes so neither dominates.

**In our project:** `src/drone_rf/evaluate.py`; formulas are checked against a hand-worked example in
`tests/test_evaluation.py`.

**Follow-up you might get:** *Which error is worse?* It depends on the use. Missing a drone (low
recall) and constant false alarms (high false-positive rate) cost different things, which is why we
report both and do not merge them into one number.

*Status: draft*

### Q: How do you read the confusion matrix?

**Short answer:** Rows are the truth, columns are the prediction. The diagonal is correct; the other
two cells are the two kinds of error.

**In our project:** Order is always `[background_rf, drone_rf]`, so the matrix is
`[[true negatives, false positives], [false negatives, true positives]]`.

*Status: draft*

### Q: Why do you report results "per group" as well as "per image"?

**Short answer:** Because many images come from the same recording, and counting each one separately
overstates how much evidence we have.

**Explanation:** If one recording is hard, all of its noise copies tend to be wrong together. Per
image that looks like many errors; really it is one difficult recording. Per-group results average
the model's score over a group's images and count the group once. For the same reason our confidence
intervals come from resampling whole groups (a *group bootstrap*), not individual images.

**In our project:** `aggregate_by_group` and `bootstrap_by_group` in `evaluate.py`.

*Status: draft*

### Q: Why keep blocks of 20 neighbouring samples together in the split?

**Short answer:** Neighbouring sample numbers appear to be consecutive slices of the same longer
recording, so they are not independent of each other.

**Explanation:** In the label data, the number of bursts in sample N and in sample N+1 is correlated
for several transmitters, while random pairs are not (VERIFIED, DATASET_AUDIT.md §4). That is the
signature of one continuous recording cut into pieces. Testing on the slice right after a training
slice would be easier than testing on a genuinely new recording.

**In our project:** `--block-size 20` in the split command. The dataset has no recording ID, so this
is a precaution, not a guarantee.

**Follow-up you might get:** *Why 20 and not 100?* Bigger blocks protect more but leave fewer groups.
With 20, the smallest transmitter classes still have about 40 groups to divide.

*Status: draft · Evidence: VERIFIED (correlation), judgement call (block size)*

### Q: A drone image at −30 dB looks like pure noise. Why is it still labelled "drone"?

**Short answer:** Because the label says what was transmitted, not what is visible. The drone signal
is in the recording; the added noise just buries it.

**Explanation:** The annotation boxes are identical across all 26 noise versions of a sample
(VERIFIED). So at very low SNR we are asking the model to find something that may be physically
undetectable in the image, and accuracy should fall toward a coin flip. If it does not, the model is
using something other than the bursts, which is one of our confound checks.

**In our project:** See the SNR curves and the hidden-burst test in
`docs/CONFOUNDS_AND_LIMITATIONS.md`.

*Status: draft · Evidence: VERIFIED (labels identical across SNR)*

### Q: Your first CNN scored 99.7%. Why did you say that was not a good result?

**Short answer:** Because a rule using 39 brightness numbers scored 100% on the same images, and the
same CNN scored only 71% once noisy images were included.

**Explanation:** A high score only means something if the test is hard. On clean images of this
dataset, drone and background recordings differ in overall brightness, so almost anything can separate
them. The 99.7% told us about the dataset, not about the network. When we evaluated the same network on
images with added noise, where the brightness difference fades, it missed 39% of the drone images.

**In our project:** Run A1 against run A0 in `docs/RESULTS.md`, Tables 1 and 2.

**Follow-up you might get:** *How did you fix it?* We trained the same network on images at all noise
levels (run C1). Validation accuracy across all noise levels rose from 0.713 to 0.962.

*Status: draft · Evidence: VERIFIED on validation data*

### Q: What mattered more, the choice of network or the training data?

**Short answer:** So far, the training data. Changing the network from our scratch CNN to a pretrained
ResNet18 changed little; changing the training images from clean-only to all noise levels changed a
lot.

**Explanation:** With clean-only training, the scratch CNN and the pretrained ResNet18 scored 0.713
and 0.675 on validation images at all noise levels, a difference smaller than the uncertainty. With
the scratch CNN unchanged but trained on all noise levels, the score became 0.962. A model can only
learn to handle conditions it has seen.

**In our project:** `docs/RESULTS.md`, Tables 2 and 3.

**Follow-up you might get:** *Is that conclusion final?* No. We have not yet trained ResNet18 on all
noise levels, so we do not know whether pretraining helps under the better training condition, and
none of this has been checked on the test set.

*Status: draft · Evidence: VERIFIED on validation data; comparison incomplete (C1b not run)*
