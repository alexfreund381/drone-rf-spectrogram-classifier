# Drone RF Spectrogram Classifier

Intro to AI team project: classify grayscale radio-frequency spectrograms as **drone RF present** or
**Wi-Fi/Bluetooth background only**, using (A) a CNN trained from scratch and (B) an
ImageNet-pretrained ResNet18, and compare them across signal-to-noise ratios.

> **Status (2026-11-10): planning stage.** The repository contains the project plan and documentation
> framework. No code, no downloaded dataset, no trained model and no results exist yet. Professor
> approved the dataset and task.

This project is about passive RF emissions, not radar, and makes no claim about real-world drone
detection.

## Start here

| Read | For |
|---|---|
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Architecture, dataset, testing strategy, experiments, phases, team workstreams |
| [docs/DOCUMENTATION_GUIDE.md](docs/DOCUMENTATION_GUIDE.md) | How we write technical docs, conceptual Q&A and presentation notes |
| [docs/qa/CONCEPTS_QA.md](docs/qa/CONCEPTS_QA.md) | Plain-language questions and answers about the project |
| [docs/presentation/PRESENTATION_NOTES.md](docs/presentation/PRESENTATION_NOTES.md) | Slide-by-slide presentation source |

## Dataset

*Drone RF Emissions in Spectrograms* — Kraft, Glüge, Schüpbach, Nyfeler (2026), Zenodo,
https://doi.org/10.5281/zenodo.21428081, CC BY 4.0. One 101.1 GB TAR archive. The data is **not**
stored in this repository. What we have and have not verified is listed in
[docs/PROJECT_PLAN.md §4](docs/PROJECT_PLAN.md).

## Publishing this repository to GitHub

The repository currently exists only locally. To create the private GitHub remote:

```
winget install --id GitHub.cli
gh auth login
gh repo create drone-rf-spectrogram-classifier --private --source . --remote origin --push
```

Then invite teammates under *Settings → Collaborators* on GitHub.
