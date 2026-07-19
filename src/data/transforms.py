"""Preprocessing + augmentation transforms.

Two pipelines built from configs/data.yaml:
  * train: random-resized-crop -> label-aware augmentation (AD-09) -> normalize
  * eval : resize -> center-crop -> normalize        (val AND test AND runtime)

CRITICAL (AD A.3): the *eval* pipeline here must match the real-time
preprocessing in src/rt/preprocess.py byte-for-byte -- same size, interpolation,
and normalization. Train accuracy is meaningless live if this drifts.

torchvision is imported lazily so the annotation/splitting modules stay
importable in environments without it. timm is only touched by
resolve_backbone_config().
"""

from __future__ import annotations

from .config import AugmentCfg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_backbone_config(backbone: str, pretrained: bool = True):
    """Ask timm for a backbone's expected input size + normalization stats.

    Returns (size:int, mean:tuple, std:tuple). This is the authoritative source
    at train time -- a pretrained backbone expects the exact stats it was
    trained with, so these OVERRIDE the config defaults (AD-06).
    """
    import timm
    from timm.data import resolve_data_config

    model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    cfg = resolve_data_config({}, model=model)
    size = cfg["input_size"][-1]  # (C, H, W) -> W
    return size, tuple(cfg["mean"]), tuple(cfg["std"])


def build_transforms(
    train: bool,
    size: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    augment: AugmentCfg | None = None,
):
    """Build a torchvision transform Compose for the train or eval pipeline."""
    from torchvision import transforms as T

    if train:
        aug = augment or AugmentCfg()
        ops = [
            T.RandomResizedCrop(size, scale=tuple(aug.rrc_scale)),
        ]
        if aug.hflip:  # label-safe for the 8-class subset only (AD-09)
            ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(
            T.RandomAffine(
                degrees=aug.rotation_deg,
                translate=(aug.translate, aug.translate),
            )
        )
        # Strategy-2.1 fix #3: perspective warp for the webcam-close-hand gap.
        # Off unless aug.perspective > 0, so the baseline/bbox runs are unchanged.
        if getattr(aug, "perspective", 0.0) > 0.0:
            ops.append(T.RandomPerspective(distortion_scale=aug.perspective, p=0.5))
        ops.append(
            T.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation,
                hue=aug.hue,
            )
        )
        # Strategy-2.1 fix #3: mild blur for fixed-focus webcam softness. Off by
        # default; only added when aug.blur_sigma > 0.
        if getattr(aug, "blur_sigma", 0.0) > 0.0:
            ops.append(T.GaussianBlur(kernel_size=5, sigma=(0.1, aug.blur_sigma)))
        ops += [T.ToTensor(), T.Normalize(mean, std)]
        return T.Compose(ops)

    # eval / val / test / runtime: deterministic, no augmentation.
    resize = int(round(size * 256 / 224))  # standard resize-then-center-crop ratio
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
