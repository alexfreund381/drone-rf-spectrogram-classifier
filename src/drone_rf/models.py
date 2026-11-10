"""The two required models. Both take ``[B, 1, H, W]`` images in [0, 1] and return 2 logits.

Each model standardizes its own input, so a checkpoint is self-contained:
  * ScratchCNN uses the training set's pixel mean/std (stored as buffers).
  * ResNet18Classifier copies the grayscale channel three times and applies ImageNet mean/std,
    because that is what the pretrained weights expect.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ScratchCNN(nn.Module):
    """Model A: small CNN, random initialization, no pretrained weights anywhere.

    ``len(widths)`` stages of Conv3x3 -> BatchNorm -> ReLU -> MaxPool2, then global average
    pooling, dropout and a linear layer to 2 logits.
    """

    def __init__(
        self,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
        in_mean: float = 0.5,
        in_std: float = 0.25,
        num_classes: int = 2,
    ):
        super().__init__()
        self.register_buffer("in_mean", torch.tensor(float(in_mean)))
        self.register_buffer("in_std", torch.tensor(float(in_std)))
        stages: list[nn.Module] = []
        prev = 1
        for w in widths:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(prev, w, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(w),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            prev = w
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.in_mean) / self.in_std
        for stage in self.stages:
            x = stage(x)
        return self.head(self.dropout(self.pool(x).flatten(1)))

    @torch.no_grad()
    def describe(self, size: int = 224) -> list[tuple[str, tuple[int, ...]]]:
        """Tensor shape after each stage for one ``size x size`` image."""
        was_training = self.training
        self.eval()
        x = torch.zeros(1, 1, size, size)
        shapes = [("input", tuple(x.shape))]
        x = (x - self.in_mean) / self.in_std
        for i, stage in enumerate(self.stages, 1):
            x = stage(x)
            shapes.append((f"stage{i}", tuple(x.shape)))
        x = self.pool(x).flatten(1)
        shapes.append(("global_avg_pool", tuple(x.shape)))
        shapes.append(("logits", tuple(self.head(x).shape)))
        self.train(was_training)
        return shapes


class ResNet18Classifier(nn.Module):
    """Model B: torchvision ResNet18 with a new 2-class head.

    ``pretrained=True`` loads ImageNet weights (downloads them on first use).
    ``pretrained=False`` gives random weights and must never be reported as "pretrained".
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.pretrained = bool(pretrained)
        self.weights_name = str(weights) if weights is not None else "random"
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._backbone_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.backbone(x)

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Stage 1: train only the new head. Stage 2: ``freeze_backbone(False)``."""
        for name, p in self.backbone.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = not frozen
        self._backbone_frozen = frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._backbone_frozen:
            # keep BatchNorm running statistics of the frozen backbone fixed
            for name, module in self.backbone.named_children():
                if name != "fc":
                    module.eval()
        return self


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_model(cfg: dict, in_mean: float = 0.5, in_std: float = 0.25) -> nn.Module:
    """Build a model from the ``model`` section of a config."""
    name = cfg.get("name", "scratch_cnn")
    if name == "scratch_cnn":
        return ScratchCNN(
            widths=tuple(cfg.get("widths", (32, 64, 128, 256))),
            dropout=float(cfg.get("dropout", 0.3)),
            in_mean=in_mean,
            in_std=in_std,
        )
    if name == "resnet18":
        return ResNet18Classifier(pretrained=bool(cfg.get("pretrained", True)))
    raise ValueError(f"unknown model name: {name}")
