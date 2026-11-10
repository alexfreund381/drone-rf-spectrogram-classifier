# Project instructions for Claude Code

University Intro to AI team project: binary classification of RF spectrograms (`drone_rf` vs.
`background_rf`) with a scratch CNN and a pretrained ResNet18. Read `docs/PROJECT_PLAN.md` before
doing anything; it holds the architecture, dataset facts, testing strategy and current status.

## Non-negotiable rules

- **RF, not radar.** No radar loaders, range–Doppler, or bird/human classes.
- **Never invent numbers.** Counts, accuracies, plots and run outcomes come only from real files in
  `artifacts/` or `runs/`. Use the evidence labels (VERIFIED / CLAIMED / ESTIMATE / UNKNOWN / PENDING
  / SYNTHETIC) defined in `docs/PROJECT_PLAN.md`.
- **Synthetic fixtures prove the code runs, nothing else.** Never report fixture metrics as results.
- **Group-safe splits.** All SNR variants of a source sample stay in one split. Never rerandomize the
  frozen split or use the test set for tuning.
- The professor has approved the dataset and task (reported by the team on 2026-11-10).
- **Do not download or extract the full 101 GB archive** without the user agreeing. Build the label
  index by range request first.
- **Never commit dataset images, archives, checkpoints or secrets.** No `tar.extractall`.
- The binary task is the deliverable. Do not replace it with YOLO/detection.
- The network sees pixels only; filename, tag, SNR and box counts are metadata for analysis.

## Documentation is part of every change

Follow `docs/DOCUMENTATION_GUIDE.md`. Any code change also updates:
1. the technical page in `docs/technical/`,
2. the conceptual Q&A in `docs/qa/` (plain language, question-and-answer pattern),
3. `docs/presentation/PRESENTATION_NOTES.md` if a slide, figure or number is affected,
4. the status tables in `docs/PROJECT_PLAN.md`.

Write for teammates who must explain this work to a professor: define terms, give the reason for each
decision, and state what the result does not prove.

## Conventions

- Python package in `src/drone_rf/`, CLI via `python -m drone_rf.cli`.
- `pytest -q` and `ruff check .` must pass before committing.
- Tests use generated fixtures only; network or real-data tests are marked `integration`.
- Code must run on CPU and CUDA; this development machine has no CUDA GPU.
