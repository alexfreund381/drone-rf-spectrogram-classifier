"""Classify one already-prepared spectrogram image with a saved checkpoint.

This works on spectrogram PNGs made the same way as the dataset's. It does not work on live
radio equipment or on ordinary photos, and the scores are uncalibrated softmax outputs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .preprocessing import load_gray_uint8, to_tensor
from .sources import CLASS_NAMES
from .train import load_checkpoint


@torch.no_grad()
def predict_image(checkpoint: str | Path, image: str | Path) -> dict:
    model, ckpt = load_checkpoint(checkpoint)
    size = int(ckpt["config"]["data"]["size"])
    x = to_tensor(load_gray_uint8(image, size)).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0].tolist()
    return {
        "image": str(image),
        "checkpoint": str(checkpoint),
        "model": ckpt["config"]["model"]["name"],
        "trained_at_commit": ckpt.get("git_commit", "unknown"),
        "split_hash": ckpt.get("split_hash", "unknown"),
        "scores_uncalibrated": dict(zip(CLASS_NAMES, probs, strict=True)),
        "prediction": CLASS_NAMES[int(probs[1] >= 0.5)],
    }
