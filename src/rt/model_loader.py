"""Load any trained checkpoint into a ready-to-run classifier.

The checkpoint written by `src/train.py` is self-describing: it carries the
backbone name, the class list, and the input spec (size / mean / std /
crop_mode). So loading a *different* model is nothing more than pointing at a
different `best.pt` -- no code or config changes. That is what makes the demo
modular (CLAUDE.md: config-driven, swap models freely).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.models.build import build_model


@dataclass(frozen=True)
class LoadedModel:
    """A checkpoint restored into an eval-ready model plus its metadata."""

    model: nn.Module
    classes: list[str]
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_mode: str
    backbone: str
    val_acc: float | None
    test_acc: float | None
    path: Path

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def describe(self) -> str:
        acc = "" if self.test_acc is None else f"  test_acc={self.test_acc:.3f}"
        return (
            f"{self.path.name}: {self.backbone} | {self.num_classes} classes | "
            f"{self.size}px | crop={self.crop_mode}{acc}"
        )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> LoadedModel:
    """Rebuild the model described by `path` and load its weights.

    Reads `backbone`, `classes`, and `input` back out of the checkpoint, builds
    the matching timm model (freeze state is irrelevant at inference), loads the
    saved weights, and puts it in eval mode on `device`.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    classes = list(ckpt["classes"])
    inp = ckpt["input"]
    backbone = ckpt["backbone"]

    # pretrained=False: we immediately overwrite with the trained weights, so
    # there is no need to pull the ImageNet download again.
    model = build_model(backbone, len(classes), freeze_backbone=False, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    return LoadedModel(
        model=model,
        classes=classes,
        size=int(inp["size"]),
        mean=tuple(inp["mean"]),
        std=tuple(inp["std"]),
        crop_mode=inp.get("crop_mode", "full_frame"),
        backbone=backbone,
        val_acc=ckpt.get("val_acc"),
        test_acc=ckpt.get("test_acc"),
        path=path,
    )
