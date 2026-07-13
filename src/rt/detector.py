"""Stage 1 of the two-stage pipeline (AD-04): find the hand, hand a crop to the
classifier.

An off-the-shelf **MediaPipe Hand Landmarker** locates the hand each frame; we
take the axis-aligned bounding box of its 21 landmarks and pad/clamp it with the
*same* geometry training used (`padded_bbox_pixels`), so the crop the live
classifier sees matches the HaGRID bbox crop it trained on. MediaPipe is only the
*detector* -- the transfer-learned CNN is still the ML deliverable (AD-05).

MediaPipe 0.10.x ships the **Tasks** API (`HandLandmarker`), which needs a model
bundle downloaded once:

    models/mediapipe/hand_landmarker.task
    (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

Caveat carried from AD-04: MediaPipe's landmark box and HaGRID's annotated box
may not be *equally tight*. `pad` is the knob that reconciles them -- it defaults
to the training value (0.15) but is exposed for tuning against the live feed.
That tightness match is a preprocessing/modeling call for the human, not the
detector's to decide.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.dataset import padded_bbox_pixels

DEFAULT_MODEL = Path("models/mediapipe/hand_landmarker.task")
DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandBox:
    """One detected hand for a frame."""

    bbox_norm: tuple[float, float, float, float]  # COCO [x, y, w, h] over landmark hull
    box_px: tuple[int, int, int, int]             # padded (l, t, r, b), == training geometry
    score: float                                  # detection/handedness confidence
    landmarks_px: np.ndarray                       # (21, 2) int, for the HUD

    def is_valid(self) -> bool:
        l, t, r, b = self.box_px
        return r - l >= 8 and b - t >= 8  # reject degenerate slivers


class HandDetector:
    """Thin wrapper over MediaPipe's Tasks `HandLandmarker` in VIDEO mode.

    `detect(frame_bgr)` -> the best `HandBox` or `None` (no hand). Timestamps for
    VIDEO mode are managed internally (must be monotonic), so callers just pass
    frames. `crop(frame, hb)` returns the padded hand crop, matching training.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        pad: float = 0.15,
        min_confidence: float = 0.5,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                f"Download it once with:\n"
                f"  python -c \"import urllib.request as u; "
                f"u.urlretrieve('{DOWNLOAD_URL}', r'{model_path}')\""
            )
        # Imported here so the rest of src/rt still imports without mediapipe.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.pad = pad
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.time()
        self._last_ts = -1

    def _next_timestamp_ms(self) -> int:
        # VIDEO mode requires strictly increasing millisecond timestamps.
        ts = int((time.time() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def detect(self, frame_bgr: np.ndarray) -> HandBox | None:
        """Locate the hand in an OpenCV BGR frame. Returns None if none found."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())

        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        xs = np.clip([p.x for p in lms], 0.0, 1.0)
        ys = np.clip([p.y for p in lms], 0.0, 1.0)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        bbox_norm = (x0, y0, x1 - x0, y1 - y0)

        box_px = padded_bbox_pixels(bbox_norm, w, h, self.pad)
        score = 1.0
        if result.handedness and result.handedness[0]:
            score = float(result.handedness[0][0].score)
        landmarks_px = np.stack([xs * w, ys * h], axis=1).astype(int)

        hb = HandBox(bbox_norm, box_px, score, landmarks_px)
        return hb if hb.is_valid() else None

    @staticmethod
    def crop(frame_bgr: np.ndarray, hb: HandBox) -> np.ndarray:
        """Padded hand crop -- identical geometry to the training bbox crop."""
        l, t, r, b = hb.box_px
        return frame_bgr[t:b, l:r]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
