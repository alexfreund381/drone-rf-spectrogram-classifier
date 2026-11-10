# Conceptual Q&A

Plain-language questions and answers about the project. Format and rules are in
[../DOCUMENTATION_GUIDE.md](../DOCUMENTATION_GUIDE.md) §3. This is the starter set, written before any
code or results exist; entries about models, training and results are added as those parts are built.

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

**In our project:** We estimate roughly 330,000 images from roughly 12,000–13,000 source samples
(ESTIMATE from a small probe; exact counts come from the audit). All our tables report "independent
source samples" separately from "images".

**Follow-up you might get:** *Are the noise copies useless, then?* No. They are valuable for training
a noise-robust model and for testing how performance falls as noise rises. They just do not count as
extra independent evidence.

*Status: draft · Evidence: ESTIMATE*

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

**In our project:** We compare the CNN against a very simple baseline that only uses overall
brightness and energy profiles. If the simple baseline does as well, the task is easy for the wrong
reason. We also look at accuracy across SNR and, where possible, at what happens when the drone bursts
are masked out. See PROJECT_PLAN.md §4.10.

**Follow-up you might get:** *Then what is your result worth?* It shows how well these methods
separate these recordings and how the two models compare under noise. It does not show that the system
would work on drone signals recorded in a real, busy radio environment.

*Status: draft · Evidence: CLAIMED (recording locations); analysis not yet run*

### Q: Why build a simple baseline when we have neural networks?

**Short answer:** A baseline tells us whether the neural network is doing anything clever.

**Explanation:** If always guessing the majority class already gives 70% accuracy, a CNN at 72% is
not impressive. If a three-number brightness rule reaches 99%, a CNN at 99% has not proved it learned
signal patterns. Baselines put the headline number in context.

**In our project:** Run A0: majority-class and energy baselines on the same split as the CNNs.

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

**Follow-up you might get:** *If ResNet18 wins, is that because of pretraining or because it is a
bigger network?* Fair point; the comparison changes two things at once. An optional run of ResNet18
with random weights would separate the two effects.

*Status: draft · Evidence: n/a (no runs yet)*

### Q: How did you use AI tools, and do you understand the code?

**Short answer:** AI assistance is permitted for this assignment. We used it to help plan, write and
document code, and each of us is responsible for understanding and explaining what we submit.

**In our project:** Every pull request requires that the author can explain the change, and every Q&A
entry is reviewed by a teammate who did not write that part. See DOCUMENTATION_GUIDE.md §6–7.

*Status: draft*

---

## To be written as the project progresses

Convolution, kernels and pooling · batch normalization and dropout · cross-entropy loss · gradients
and AdamW · overfitting and early stopping · normalization of inputs · ResNet18 skip connections ·
freezing vs. fine-tuning · accuracy vs. precision, recall and F1 · reading a confusion matrix ·
bootstrap confidence intervals by group · what our measured results show and do not show.
