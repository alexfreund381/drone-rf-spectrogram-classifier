# Drone RF Spectrogram Classifier

Intro to AI team project: classify grayscale radio-frequency spectrograms as **drone RF present** or
**Wi-Fi/Bluetooth background only**, using (A) a CNN trained from scratch and (B) an
ImageNet-pretrained ResNet18, and compare them across signal-to-noise ratios.

> **Status (2026-11-10).** The pipeline is implemented and tested, the whole dataset's labels have
> been audited, the train/validation/test split is frozen, and a 10,407-image sample is ready for
> training. **No model has been trained on real data yet, so there are no results.** The professor
> has approved the dataset and task.

This project is about passive RF emissions, not radar. It makes no claim about detecting drones in
the real world.

## Start here

| Read | For |
|---|---|
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Status, architecture, testing strategy, experiments, team workstreams |
| [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md) | Measured class counts and label-quality findings |
| [docs/SPLIT_PROTOCOL.md](docs/SPLIT_PROTOCOL.md) | The frozen split and how leakage is prevented |
| [docs/DOWNLOAD_AND_STORAGE.md](docs/DOWNLOAD_AND_STORAGE.md) | How to get the data |
| [docs/CONFOUNDS_AND_LIMITATIONS.md](docs/CONFOUNDS_AND_LIMITATIONS.md) | What the results can and cannot show |
| [docs/technical/](docs/technical/) | How each part of the code works |
| [docs/qa/CONCEPTS_QA.md](docs/qa/CONCEPTS_QA.md) | Plain-language questions and answers |
| [docs/presentation/PRESENTATION_NOTES.md](docs/presentation/PRESENTATION_NOTES.md) | Slide-by-slide presentation source |
| [docs/DOCUMENTATION_GUIDE.md](docs/DOCUMENTATION_GUIDE.md) | How we write documentation |

## Dataset

*Drone RF Emissions in Spectrograms* — Kraft, Glüge, Schüpbach, Nyfeler (2026), Zenodo,
https://doi.org/10.5281/zenodo.21428081, CC BY 4.0. One 101.1 GB TAR archive. The data is **not**
stored in this repository.

Measured from the label files (details in the audit):

| | Source samples | Images (26 noise levels each) |
|---|---|---|
| `drone_rf` | 8,562 | 222,612 |
| `background_rf` | 3,942 | 102,492 |
| quarantined (drone-tagged, no annotated signal) | 310 | 8,060 |

Images are 640×640 8-bit grayscale PNGs (checked on the 10,407-image sample).

## Install

Python 3.11 or newer. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate            # Windows;  on macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Use the default PyTorch index instead if you have an NVIDIA GPU.

Check the install (uses generated fake images only, no download):

```
ruff check .
pytest -q
```

## Get the data

```
python -m drone_rf.cli index-labels --out data/manifest/manifest.csv --boxes-out data/manifest/label_text.csv --summary-dir artifacts/audit
python -m drone_rf.cli split --manifest data/manifest/manifest.csv --out data/manifest/splits.csv --seed 42 --block-size 20
python -m drone_rf.cli fetch-regions --output data/selected --regions 12 --region-mb 256 --image-region-end 100792909312 --splits data/manifest/splits.csv
```

The first command downloads 285 MB (labels only), the third 3.2 GB (10,407 images, about an hour at
3 MB/s). The `split` command must print hash `ddfcbce03785a3af`.

## Train and evaluate

```
python -m drone_rf.cli baselines --config configs/binary_cnn.yaml --split val
python -m drone_rf.cli train --config configs/binary_cnn.yaml
python -m drone_rf.cli train --config configs/resnet18.yaml
python -m drone_rf.cli evaluate --checkpoint runs/a1_scratch_cnn/best.pt --split val --all-snr --name val_all_snr
python -m drone_rf.cli confound-check --checkpoint runs/a1_scratch_cnn/best.pt --split val
python -m drone_rf.cli compare --runs runs --split val
python -m drone_rf.cli figures --runs runs --split val --image-dir data/selected
```

These commands are tested on synthetic fixtures. They have not yet been run on the real images.
Training runs on CPU; measured speed on the development laptop is about 25–30 images/s for the scratch
CNN and about 20 images/s for ResNet18.

All commands: `python -m drone_rf.cli --help`.

## Rules for contributors

- Never commit data, archives or checkpoints (`data/`, `runs/` are git-ignored).
- Never change the frozen split or use the test set to choose anything.
- Never write a number in a document that does not come from a file in `artifacts/` or `runs/`.
- Results from synthetic fixtures are software tests, not findings.
- Update the technical page, the Q&A and the presentation notes with every code change.
