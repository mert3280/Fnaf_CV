"""Model construction: a pretrained timm backbone with a fresh classification
head, plus freeze control. This is the "full intended model" (AD-06/AD-07):
MobileNetV3 by default, head swapped to the working-subset size, backbone
frozen for the Phase-2 baseline (AD-08 Stage A).
"""

from __future__ import annotations

import timm
import torch.nn as nn
from timm.data import resolve_data_config


def build_model(
    name: str = "mobilenetv3_large_100",
    num_classes: int = 8,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    unfreeze_blocks: int = 0,
    head_init: str = "timm_default",
) -> nn.Module:
    """Create a timm backbone with a fresh `num_classes` head.

    timm swaps the head automatically for `num_classes`; the new head is
    randomly initialized (verify via param_summary). With freeze_backbone=True
    every parameter except the classifier has requires_grad=False.

    `unfreeze_blocks > 0` then re-enables the last N backbone block-stages
    (Strategy-2.1 fix #5 / AD-08 Stage B). Default 0 = fully frozen (current
    behavior, unchanged). *How many* blocks to unfreeze and *when* is Ted's
    modeling call -- this only provides the mechanism.

    `head_init` selects the head initialization; the default keeps timm's, so
    this signature change cannot move any existing number. See init_classifier().
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    init_classifier(model, head_init)
    if freeze_backbone:
        freeze_all_but_head(model)
        if unfreeze_blocks > 0:
            unfreeze_last_n_blocks(model, unfreeze_blocks)
    return model


HEAD_INIT_MODES = ("timm_default", "pytorch_linear")


def init_classifier(model: nn.Module, mode: str = "timm_default") -> str:
    """Re-initialize the classification head. Returns the mode applied.

    WHY THIS KNOB EXISTS (measured 2026-07-26, Week-4 tuning). timm's
    EfficientNet-family `_init_weights` initializes a `Linear` head as
    `U(-r, r)` with `r = 1/sqrt(fan_out)` -- where `fan_out` is the number of
    **output classes**, not the 1280 input features. For ImageNet's 1000-class
    head that is a sane r=0.032; for our **2-class** head it is r=0.707, ~25x
    PyTorch's own `Linear` default of `1/sqrt(1280)=0.028`. Over 1280-d pooled
    features that starts the probe at logits of +/-27 and cross-entropy **2.70**
    instead of ln(2)=0.69, so the first ~30 epochs are spent shrinking the init
    rather than learning. This is exactly the "val still climbing at ep40" note in
    DOCS/models/fistvsrest_frozen_mnv3_large/README.md.

      * `timm_default`   -- leave timm's init untouched. THE DEFAULT, so every
                            previously committed run stays bit-reproducible.
      * `pytorch_linear` -- `reset_parameters()` on the head: PyTorch's standard
                            `kaiming_uniform` bounded by 1/sqrt(in_features).

    Whether `pytorch_linear` becomes the project default is Ted's modeling call
    (see the Week-4 report §1); this only exposes both, measurably.
    """
    if mode not in HEAD_INIT_MODES:
        raise ValueError(f"head_init must be one of {HEAD_INIT_MODES}, got {mode!r}")
    if mode == "pytorch_linear":
        for m in model.get_classifier().modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    return mode


def freeze_all_but_head(model: nn.Module) -> None:
    """Freeze the whole backbone; leave only the classifier trainable (AD-08 A)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, n: int) -> int:
    """Re-enable grads on the last `n` backbone block-stages (AD-08 Stage B).

    Assumes a timm `EfficientNet`-family backbone with a `.blocks` Sequential
    (MobileNetV3 = 7 stages, EfficientNet-B0 = 7) -- the AD-07 default + fallback.
    Progressive unfreezing then means training the head plus the top `n` stages,
    the layers most worth adapting to hand crops. Returns how many stages were
    unfrozen. Raises if the backbone has no `.blocks` so a wrong assumption fails
    loudly rather than silently unfreezing nothing.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            f"{type(model).__name__} has no `.blocks`; progressive unfreeze here "
            f"targets timm EfficientNet-family backbones (mobilenetv3/efficientnet)."
        )
    stages = model.blocks
    n = max(0, min(n, len(stages)))
    for stage in stages[len(stages) - n:]:
        for p in stage.parameters():
            p.requires_grad = True
    return n


def param_summary(model: nn.Module) -> dict:
    """Trainable vs. frozen parameter counts (assert only the head trains)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / total,
    }


def get_data_config(model: nn.Module) -> tuple[int, tuple, tuple]:
    """The backbone's expected (input_size, mean, std) -- authoritative for
    preprocessing (AD-06). Returns (size, mean, std)."""
    cfg = resolve_data_config({}, model=model)
    return cfg["input_size"][-1], tuple(cfg["mean"]), tuple(cfg["std"])


def pooled_features(model: nn.Module, x):
    """Pre-classifier pooled feature vector (the head's input). Used to cache
    frozen-backbone features so the head can be trained as a fast linear probe
    -- mathematically identical to training the frozen model end-to-end."""
    return model.forward_head(model.forward_features(x), pre_logits=True)
