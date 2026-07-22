"""Stage 2 -> mouse button: the debounced, edge-triggered click FSM (AD-20).

Decided semantics (Ted, 2026-07-14): **single click per fist**. Each confirmed
palm->fist transition fires exactly one click; holding the fist does nothing
more until the hand returns to palm and re-arms.

    DISARMED ── K confident palm frames ──▶ ARMED ── K confident fist frames ──▶ fire once
        ▲                                     │                                   │
        └── ambiguity SUSTAINED past `grace` ─┘◀──────────────────────────────────┘
            (brief low-conf / dropout frames        (re-arm requires palm again)
             are tolerated, not a disarm -- 3.1.1)

Why this shape (AD-20 rationale):
* **K confident frames** above a confidence threshold = one noisy frame can
  never click (AD-12's principle, simplified for two labels / one action).
* **Re-arm on palm only** = double-fires are structurally impossible, not merely
  debounced -- after firing, the FSM won't fire again until it has *seen* K
  confident palm frames.
* **Grace window (3.1.1)**: ambiguous frames (low confidence or a brief
  detection dropout) no longer hard-disarm -- they just don't count, and ARMED
  survives up to `grace` of them in a row. The 3.0 version disarmed on the
  FIRST ambiguous frame, which made a *slow* palm->fist physically unclickable:
  the in-between poses read as low-confidence, the FSM disarmed mid-squeeze,
  and the finished fist then had nothing to fire from (Ted's live finding,
  2026-07-15). Firing still requires K confident fist frames -- the grace only
  stops uncertainty from *cancelling* an armed squeeze.
* **Sustained absence still disarms**: more than `grace` consecutive ambiguous
  frames while armed drops to DISARMED, so a hand that actually left can never
  come back as a surprise click.
* **Cooldown** (~0.3 s) is a pure backstop against classifier flicker mid-
  transition; the palm re-arm is the real guard.

Works with any binary no-click / click checkpoint: pass the no-click class as
`palm_label` and the click class as `fist_label`. For the palm/fist model
(AD-18) that is "palm"/"fist"; for the fist-vs-rest model (Strategy 3.2 / AD-21)
it is "not_fist"/"fist" -- the callers read both out of the checkpoint's class
list. Any other label (e.g. from the legacy 8-class models) counts as "neither"
and disarms.

K, the confidence threshold, and the cooldown are Ted's live-tuning calls
(CLAUDE.md); defaults here are the Strategy-3 starting points, surfaced as CLI
flags by the callers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

DISARMED = "DISARMED"
ARMED = "ARMED"


@dataclass
class ClickFSM:
    """Feed one (label, confidence) per frame; `update()` returns True on the
    single frame a click should fire."""

    k: int = 3                   # confident frames to confirm a pose
    conf_threshold: float = 0.70  # min softmax confidence for a frame to count
    cooldown_s: float = 0.30     # min seconds between fires (backstop)
    grace: int = 10              # ambiguous frames tolerated before state decays (3.1.1)
    palm_label: str = "palm"
    fist_label: str = "fist"

    state: str = field(default=DISARMED, init=False)
    _palm_streak: int = field(default=0, init=False, repr=False)
    _fist_streak: int = field(default=0, init=False, repr=False)
    _ambig: int = field(default=0, init=False, repr=False)
    _last_fire: float = field(default=-1e9, init=False, repr=False)

    def update(self, label: str | None, confidence: float, now: float | None = None) -> bool:
        """Advance one frame. `label=None` means no hand this frame."""
        if now is None:
            now = time.monotonic()
        confident = label is not None and confidence >= self.conf_threshold
        is_palm = confident and label == self.palm_label
        is_fist = confident and label == self.fist_label

        if not (is_palm or is_fist):
            # Ambiguous frame (no hand / low confidence / other label): don't
            # count it, don't reset anything -- unless it persists past `grace`.
            self._ambig += 1
            if self._ambig > self.grace:
                self.state = DISARMED
                self._palm_streak = 0
                self._fist_streak = 0
            return False
        self._ambig = 0

        if self.state == DISARMED:
            if is_fist:
                self._palm_streak = 0  # actual contrary evidence resets
            else:
                self._palm_streak += 1
                if self._palm_streak >= self.k:
                    self.state = ARMED
                    self._fist_streak = 0
            return False

        # ARMED
        if is_palm:
            self._fist_streak = 0  # holding palm keeps us armed
            return False
        self._fist_streak += 1
        if self._fist_streak >= self.k and (now - self._last_fire) >= self.cooldown_s:
            self._last_fire = now
            self.state = DISARMED  # re-arm requires K confident palms again
            self._palm_streak = 0
            self._fist_streak = 0
            return True
        return False

    def describe(self) -> str:
        """One-liner for the HUD: state plus the live streak counters."""
        s = f"{self.state} palm:{self._palm_streak}/{self.k} fist:{self._fist_streak}/{self.k}"
        if self._ambig:
            s += f" ?{self._ambig}/{self.grace}"
        return s

    def reset(self) -> None:
        self.state = DISARMED
        self._palm_streak = 0
        self._fist_streak = 0
        self._ambig = 0
