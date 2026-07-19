"""Stage 1 -> mouse position: the absolute frame-to-screen cursor mapper (AD-19).

The detector already finds the hand every frame; this module turns that position
into a screen coordinate the input layer can move the mouse to. The chain, per
frame (Strategy 3 spec):

    palm-center anchor -> (mirror) -> control box -> clamp -> EMA -> dead-zone
                                                                  -> screen (x, y)

* **Anchor** = one point in UNCLIPPED normalized frame coords, measured from the
  frame's top-left; hand/box size never enters the mapping (Strategy 3.1, the
  distance-invariance fix). Default point: mean of the wrist + four finger-base
  landmarks (MediaPipe ids 0, 5, 9, 13, 17) -- fingertips travel a lot between
  palm and fist; the palm base barely moves, so the cursor doesn't lurch when
  the player clenches to click. `source="box"` uses the hull center instead.
* **Control box**: a central sub-rectangle of the frame maps to the FULL screen,
  so the hand reaches every screen edge without leaving the camera's comfortable
  FOV; positions outside the box clamp to its edge.
* **Adaptive box (Strategy 3.1.1, default)**: the box scales with the hand's
  apparent size (`palm_span`), so the *gain* is constant across distance -- the
  same physical arm movement moves the cursor the same amount whether the hand
  is close (big in frame, big box) or far (small in frame, small box). This is
  deliberate: position-invariance (3.1: a frame position = a screen position)
  and motion-invariance (same arm motion = same cursor travel) are mutually
  exclusive when distance changes; the live tests wanted the second.
  `box_mode="fixed"` restores the 3.1 behavior.
* **EMA + dead-zone**: `alpha` is the weight of the NEW sample (higher = more
  responsive, lower = steadier). The dead-zone holds the output still while the
  smoothed target stays within `deadzone` (normalized screen units) of it, so
  hand tremor doesn't buzz the cursor -- but the EMA keeps integrating, so a
  slow deliberate drift still gets through once it accumulates.
* **No hand -> freeze**: `update(None)` returns the last output unchanged. The
  EMA state is kept across dropouts, so on re-detection the cursor *glides* to
  the new position instead of teleporting (AD-19).

Mirroring: the demo/play loops flip the frame horizontally BEFORE detection, so
landmark x already increases to the player's right and no extra mirror is needed
(`mirror_x=False`, the default). If you feed an UN-mirrored frame, set
`mirror_x=True` -- the mapper and the preview must agree or the cursor moves
backwards (AD-19 consequence).

All numeric defaults are the Strategy-3 starting points -- live tuning of the
box / alpha / dead-zone is Ted's call (CLAUDE.md), which is why every one is a
constructor arg surfaced as a CLI flag by the callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Wrist + the four metacarpophalangeal (finger-base) joints -- the palm plate.
PALM_ANCHOR_LANDMARKS = (0, 5, 9, 13, 17)


def hand_anchor_norm(landmarks_norm: np.ndarray, source: str = "palm") -> tuple[float, float]:
    """The cursor's anchor: ONE point, measured from the frame's top-left in
    normalized coords. Hand/box size never enters (Strategy 3.1) -- the same
    physical pointing position gives the same anchor at any distance.

    Must be fed the detector's **unclipped** `landmarks_norm`: clipping the
    landmarks to the frame biases the anchor toward the frame interior exactly
    when the hand is close/large, which was the 3.0 distance bug. Values may
    fall outside [0, 1] when part of the hand is off-frame -- the mapper's
    control-box clamp handles that cleanly.

    `source` (Ted's live A/B, `--anchor`):
      * "palm" -- mean of the wrist + four finger-base landmarks. Stable across
        the palm->fist squeeze (fingertips move, the palm plate doesn't), so
        clicking doesn't lurch the cursor.
      * "box"  -- center of the landmark hull ("middle of the box"). Simplest,
        but the hull shrinks when fingers curl, so the anchor can dip slightly
        on the squeeze.
    """
    pts = landmarks_norm.astype(float)
    if source == "palm":
        p = pts[list(PALM_ANCHOR_LANDMARKS)]
        return float(p[:, 0].mean()), float(p[:, 1].mean())
    if source == "box":
        return (
            float((pts[:, 0].min() + pts[:, 0].max()) / 2),
            float((pts[:, 1].min() + pts[:, 1].max()) / 2),
        )
    raise ValueError(f"unknown anchor source: {source!r} (palm | box)")


def palm_span(landmarks_norm: np.ndarray) -> float:
    """Apparent hand size: wrist -> middle-finger-base distance in normalized
    frame units. Drives the adaptive control box (3.1.1). Chosen because it is
    stable through the palm->fist squeeze (the hull is not: fingers curl)."""
    d = landmarks_norm[9].astype(float) - landmarks_norm[0].astype(float)
    return float(np.hypot(d[0], d[1]))


@dataclass
class CursorMapper:
    """Absolute mapper: normalized frame position -> integer screen pixel.

    Stateful: keeps the EMA target and the last emitted position so it can
    freeze on dropouts and dead-zone out tremor. One instance per session.
    """

    screen_w: int
    screen_h: int
    box_w: float = 0.60      # control-box width, fraction of the frame (fixed mode)
    box_h: float = 0.55      # control-box height, fraction of the frame (fixed mode)
    box_cx: float = 0.50     # control-box center x (fraction of the frame)
    box_cy: float = 0.50     # control-box center y
    box_mode: str = "adaptive"  # adaptive (3.1.1: constant gain) | fixed (3.1)
    box_gain: float = 4.0    # adaptive: box width = gain * palm_span (Ted tunes)
    alpha: float = 0.35      # EMA weight of the NEW sample (1.0 = no smoothing)
    deadzone: float = 0.005  # min normalized move before the output budges
    mirror_x: bool = False   # True only when fed an un-mirrored frame

    _ema: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _out: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _scale: float | None = field(default=None, init=False, repr=False)
    _eff: tuple[float, float] | None = field(default=None, init=False, repr=False)

    def _effective_box(self, hand_scale: float | None) -> tuple[float, float]:
        """Box (w, h) for this frame. Adaptive mode sizes it from the smoothed
        hand scale so cursor gain is distance-invariant; box_h/box_w fixes the
        aspect. Falls back to the fixed dims without a usable scale."""
        if self.box_mode != "adaptive" or hand_scale is None or hand_scale <= 0:
            self._eff = (self.box_w, self.box_h)
            return self._eff
        # Light EMA on the scale so box size (and thus the cursor) doesn't
        # jitter with per-frame landmark noise.
        self._scale = (hand_scale if self._scale is None
                       else self._scale + 0.3 * (hand_scale - self._scale))
        w = min(1.0, max(0.15, self.box_gain * self._scale))
        h = min(1.0, max(0.15, w * (self.box_h / self.box_w)))
        self._eff = (w, h)
        return self._eff

    def update(
        self,
        anchor_norm: tuple[float, float] | None,
        hand_scale: float | None = None,
    ) -> tuple[int, int] | None:
        """Feed one frame's anchor (or None when no hand) plus the hand's
        apparent size (`palm_span`, adaptive mode); get the screen pixel the
        cursor should be at (or None if no hand has ever been seen)."""
        if anchor_norm is None:
            return self._out  # freeze: hold position through the dropout

        x, y = anchor_norm
        if self.mirror_x:
            x = 1.0 - x
        box_w, box_h = self._effective_box(hand_scale)

        # control box -> normalized screen position, clamped to [0, 1]
        nx = (x - (self.box_cx - box_w / 2)) / box_w
        ny = (y - (self.box_cy - box_h / 2)) / box_h
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))

        if self._ema is None:  # first-ever detection: seed directly
            self._ema = (nx, ny)
        else:  # includes re-detection after a dropout -> glide, never teleport
            ex, ey = self._ema
            self._ema = (ex + self.alpha * (nx - ex), ey + self.alpha * (ny - ey))

        ex, ey = self._ema
        target = (int(round(ex * (self.screen_w - 1))), int(round(ey * (self.screen_h - 1))))
        if self._out is not None:
            dx = abs(target[0] - self._out[0]) / self.screen_w
            dy = abs(target[1] - self._out[1]) / self.screen_h
            if max(dx, dy) < self.deadzone:
                return self._out  # inside the dead-zone: hold still
        self._out = target
        return self._out

    def box_px(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Control-box (x1, y1, x2, y2) in frame pixels, for HUD drawing.
        Reflects the current *effective* box (adaptive mode resizes live)."""
        w, h = self._eff if self._eff is not None else (self.box_w, self.box_h)
        x1 = int((self.box_cx - w / 2) * frame_w)
        y1 = int((self.box_cy - h / 2) * frame_h)
        x2 = int((self.box_cx + w / 2) * frame_w)
        y2 = int((self.box_cy + h / 2) * frame_h)
        return x1, y1, x2, y2

    def reset(self) -> None:
        """Forget all state (e.g. when the camera or mirror setting changes)."""
        self._ema = None
        self._out = None
        self._scale = None
        self._eff = None
