# Documentation Guide

How this team writes documentation. It applies to people and to AI assistants (Claude Code reads
`CLAUDE.md`, which points here).

We keep **three kinds of documentation**, because we have three different readers:

| Track | Reader | Question it answers | Lives in |
|---|---|---|---|
| 1. Technical | A teammate changing or running the code | "How does this work and how do I run it?" | `docs/technical/`, plus the audit/protocol/results docs |
| 2. Conceptual Q&A | A teammate preparing to explain the project, or facing the professor's questions | "What is this, and why did we do it that way?" | `docs/qa/` |
| 3. Presentation notes | Whoever builds and delivers the slides | "What do I show and say?" | `docs/presentation/` |

The same topic usually appears in all three, at different depths. Example for the split:
technical = the grouping key, the function, the tests; Q&A = "Why can't we just split the images
randomly?"; presentation = one slide with a leakage diagram and three sentences to say.

---

## 1. Rules for all three tracks

1. **Never state a number we did not measure.** Counts, accuracies, timings and sizes come from a file
   in `artifacts/` or `runs/`, and the doc names that file.
2. **Use the evidence labels** from PROJECT_PLAN.md: VERIFIED, CLAIMED, ESTIMATE, UNKNOWN, PENDING,
   SYNTHETIC. Anything from generated fixtures is SYNTHETIC and is never presented as a result.
3. **Use the project's fixed vocabulary** (§5). One name per thing.
4. **Say what the project does not show.** RF emissions, not radar. These recordings, not the real
   world. Presence classification, not detection or localisation.
5. **Write so a teammate who missed the meeting can follow.** Define a term the first time it appears.
6. **Update docs in the same pull request as the code** (§6).
7. **You must be able to explain what you commit.** If an AI wrote a paragraph you do not understand,
   ask it to explain, then rewrite it in your own words.

---

## 2. Track 1 — Technical documentation

**One page per module or pipeline stage** in `docs/technical/<name>.md`, plus docstrings in code.

### Template

```markdown
# <Module or stage name>

**Code:** `src/drone_rf/<file>.py`   **Tests:** `tests/<file>.py`   **Owner:** <name>

## Purpose
Two or three sentences: what problem this solves in the pipeline.

## Inputs and outputs
| | Format | Example |
|---|---|---|
| Input | | |
| Output | | |

## How it works
Step-by-step description of the logic. Include the rules and edge cases, not a line-by-line
paraphrase of the code.

## How to run it
Exact command(s), copied from a terminal where they worked, with expected output.

## Design decisions
| Decision | Alternatives considered | Why we chose this |
|---|---|---|

## Failure modes and how they are handled

## Tests
What each test proves, in one line each.

## Known limitations
```

### Rules

- Commands must be ones that actually ran. Paste real output, trimmed.
- The "Design decisions" table is required. It is the raw material for the Q&A and for defending
  our choices.
- Docstrings: every public function gets one sentence on what it does, its arguments, what it
  returns, and what it raises. Comments explain *why*, not *what*.
- Results documents (`DATASET_AUDIT.md`, `RESULTS.md`, `EXPERIMENTS.md`) are generated from or
  checked against machine-readable outputs. Each table names its source file, the git commit, and
  the split hash.

---

## 3. Track 2 — Conceptual Q&A

Plain-language questions and answers. This is our study guide and our defense preparation. Files in
`docs/qa/`, grouped by topic. `CONCEPTS_QA.md` is the starting file; split it by topic when it grows
(`QA_DATA.md`, `QA_MODELS.md`, `QA_EVALUATION.md`, …).

### Entry pattern

```markdown
### Q: <A question a curious classmate or the professor would actually ask>

**Short answer:** One or two sentences, no jargon that has not been defined.

**Explanation:** A paragraph or two. Use an everyday analogy where it helps, and say where the
analogy stops being accurate.

**In our project:** What we specifically did or will do, and where to find it
(`file`, command, or doc section).

**Follow-up you might get:** One harder question, with a brief answer.

*Status: draft | reviewed by <name>* · *Evidence: VERIFIED / CLAIMED / …*
```

### Rules

- **Write the question first, in the asker's words.** "Why can't we shuffle all the images and split
  them?" is better than "Describe group-aware splitting."
- **Short answer must stand alone.** Someone should be able to say it out loud in ten seconds.
- **Concept before mechanism.** Explain the idea, then the detail. Formulas are allowed only after
  the idea is clear in words.
- **Always include "In our project".** A general textbook answer is not enough; the professor will
  ask what *we* did.
- **Include the uncomfortable questions.** "Isn't your model just learning the room the recording
  was made in?" belongs here with an honest answer.
- **Three kinds of question to cover for every component:** *What is it? Why did we choose it over
  the alternative? What could go wrong, or what does it not prove?*
- Each entry is reviewed by a teammate who did **not** write that code. If they cannot explain it
  back, rewrite it.

### Topics the Q&A must cover by the end

RF and I/Q samples · Fourier transform and spectrograms · log power · AWGN and SNR · what a YOLO box
is and how we turn boxes into an image label · why noise copies are not independent · train/val/test
· group leakage · normalization · convolution, kernels, pooling, batch norm, dropout · cross-entropy
· gradients and AdamW · overfitting and early stopping · training from scratch vs. pretrained ·
ResNet18 and skip connections · freezing vs. fine-tuning · why ImageNet weights might or might not
help on spectrograms · accuracy vs. precision/recall/F1 · confusion matrix · why a baseline matters ·
the chamber-vs-building confound · what our results do not prove · how we used AI assistance.

---

## 4. Track 3 — Presentation notes

Source material for slides, not the slides themselves. One file,
`docs/presentation/PRESENTATION_NOTES.md`, with one block per planned slide. Whoever builds the deck
copies from here, so the deck and the repo never disagree.

### Slide block pattern

```markdown
## Slide <n>: <Title that states the point, not the topic>

**Speaker:** <name>   **Time:** <seconds>

**The one thing the audience should remember:**
A single sentence.

**On the slide:**
- At most three or four short bullets, or one figure.

**Visual:** `artifacts/figures/<file>.png` (how it is generated), or "TO CREATE: <description>".

**Say (speaker notes):**
Three to six sentences in spoken language.

**Likely question:** <question> → see `docs/qa/<file>.md#<anchor>`

**Evidence:** <source file / run id>, or `PLACEHOLDER — no result yet`
```

### Rules

- **Titles are claims.** "Noise copies of one recording must stay in one split", not "Data splitting".
- **One idea per slide.** If the "one thing" sentence needs "and", make two slides.
- **Figures are generated by code** in `visualization.py` from real logs. No hand-drawn result charts.
- **No fake numbers, ever.** Until a run exists the slide says `PLACEHOLDER — no result yet`.
  Figures made from fixtures are stamped SYNTHETIC and do not go in the final deck.
- **Each slide links to a Q&A entry** so the speaker is ready for the obvious question.
- **Each speaker section has a handoff sentence** to the next speaker.
- Keep a running "limitations" slide. Being upfront about the confound is a strength in the defense.

---

## 5. Fixed vocabulary

| Use | Meaning | Avoid |
|---|---|---|
| RF emission | Radio signal transmitted by a controller or drone | "radar signal", "radar return" |
| spectrogram | Image of signal power over time and frequency | "photo", "picture of a drone" |
| source sample | One original recording segment, identified by `<N>` in the filename | "image" when you mean the recording |
| SNR variant / noise variant | One of the 26 noise-added versions of a source sample | "sample", "observation" |
| image | One PNG file (a specific source sample at a specific SNR) | |
| group | The set of images that must stay in one split | |
| class tag | `target0`…`target5`, `labnoise` from the filename | "label" |
| binary label | `drone_rf` (1) or `background_rf` (0), derived by our rules | |
| quarantine | Excluded with a recorded reason | "deleted", "dropped" |
| box / annotation | An automatically generated YOLO rectangle around one burst | "ground truth" without the word "automatic" |
| scratch CNN | Our own network, random initialization | "custom pretrained" |
| pretrained ResNet18 | ResNet18 with ImageNet weights | plain "ResNet" when the weights are random |
| validation set | Used for choosing models and settings | "test set" |
| test set | Used once at the end | |
| presence classification | Saying whether drone RF is in the image | "drone detection" |

---

## 6. Definition of done for a pull request

- [ ] Tests added or updated and passing (`pytest -q`, `ruff check .`)
- [ ] Technical page for the touched module created or updated, including the design-decision table
- [ ] At least one Q&A entry added or updated for any new concept or decision
- [ ] Presentation notes updated if a slide's content, figure or number is affected
- [ ] PROJECT_PLAN.md status tables updated if a phase, fact label or open question changed
- [ ] No unmeasured numbers; evidence labels correct
- [ ] Author can explain every line of the change without the AI

---

## 7. Using Claude Code (or another AI) to write documentation

AI use is allowed in this course, with the condition that we understand and can present everything.
These prompts keep the three tracks consistent. Run them after the code and tests for a module pass.

**Technical page**
```
Write docs/technical/<module>.md for src/drone_rf/<module>.py following the Track 1 template in
docs/DOCUMENTATION_GUIDE.md. Run the commands you document and paste real output. Fill the design
decisions table with the alternatives that were actually considered. Do not state any number you
did not measure in this session.
```

**Q&A entries**
```
Read src/drone_rf/<module>.py and docs/technical/<module>.md. Add Q&A entries to docs/qa/ using the
Track 2 pattern in docs/DOCUMENTATION_GUIDE.md. Cover: what it is, why we chose it over the
alternative, and what could go wrong or what it does not prove. Write for a classmate who has taken
Intro to AI but knows no signal processing. Include one uncomfortable question a professor might ask.
```

**Presentation notes**
```
Update docs/presentation/PRESENTATION_NOTES.md for <topic> using the Track 3 slide block pattern in
docs/DOCUMENTATION_GUIDE.md. Titles must be claims. Use only figures that exist in artifacts/figures
and numbers that exist in runs/ or artifacts/; otherwise write PLACEHOLDER. Link each slide to a Q&A
entry.
```

**Self-check (for the human, before committing)**
```
Quiz me on docs/qa/<file>.md. Ask one question at a time, wait for my answer, then tell me what I
got wrong or left out. Do not show me the written answer first.
```

**Consistency check (before the presentation)**
```
Compare every number and claim in docs/presentation/ and docs/qa/ against docs/RESULTS.md,
docs/DATASET_AUDIT.md and the files in artifacts/ and runs/. List every mismatch and every claim
with no source. Do not fix anything; report only.
```
