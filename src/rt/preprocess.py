"""Runtime preprocessing -- the live-loop half of the AD-A.3 contract.

CRITICAL: this must match the *eval* pipeline in `src/data/transforms.py`
byte-for-byte (same resize ratio, interpolation, center-crop, normalization).
We guarantee that not by re-implementing it but by calling the very same
`build_transforms(train=False, ...)`. If training's eval transform changes,
this changes with it.

The one thing training gets that the webcam does not is a hand bounding box.
Training in `crop_mode="bbox"` crops tightly to the annotated hand before the
transform; live, there is no annotation. We approximate it with a fixed on-screen
ROI box the user places their hand in (see `RoiCropper`). `crop_mode="full_frame"`
models take the whole frame, exactly as they were trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from src.data.transforms import build_transforms


class Preprocessor:
    """Turns a BGR webcam frame (OpenCV's native format) into a normalized,
    batched tensor ready for the model. Built straight from a LoadedModel's
    input spec so it always matches the checkpoint it serves."""

    def __init__(self, size: int, mean, std):
        # train=False -> the deterministic Resize->CenterCrop->Normalize pipeline.
        self.transform = build_transforms(False, size, mean, std)

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # OpenCV frames are BGR; PIL / training expect RGB.
        rgb = frame_bgr[:, :, ::-1]
        img = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
        return self.transform(img).unsqueeze(0)  # (1, C, H, W)


@dataclass
class RoiCropper:
    """A centered square region-of-interest, expressed as a fraction of the
    frame's shorter side. Stands in for the training-time hand bbox when a
    `crop_mode="bbox"` model is run live: the user aligns their hand inside the
    drawn box. For `full_frame` models the cropper is bypassed entirely."""

    frac: float = 0.6  # side length as a fraction of min(h, w)

    def box(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Pixel (x1, y1, x2, y2) of the ROI square for an (h, w) frame."""
        side = int(min(h, w) * self.frac)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        return x1, y1, x1 + side, y1 + side

    def crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.box(h, w)
        return frame_bgr[y1:y2, x1:x2]
