"""Simulated mouse input for the real game (AD-14, amended by AD-17).

Wraps `pydirectinput` (SendInput -- the known-good path for DirectX games like
FNAF) behind a small class the play loop drives every frame:

    move_to(x, y)   -- cursor movement (new with the cursor pivot: every frame)
    click()         -- one left click at the current cursor position
    kill()          -- the mandatory kill-switch: irreversibly disables output

Design notes carried over from the day-1 de-risk work
(`keyboard_mock_controller.py`, kept as the record of what was learned):

* **DPI awareness**: on a scaled display Windows reports a smaller "virtual"
  resolution, so moveTo lands in the wrong spot. `SetProcessDPIAware()` runs
  before any movement, making all coordinates true pixels -- `screen_size()`
  reports the same true-pixel space.
* `pydirectinput.PAUSE = 0` -- the default 0.1 s sleep per call would cap the
  loop at ~10 FPS. We pace ourselves; the frame loop is the clock.
* `FAILSAFE = False` -- the corner-slam failsafe would constantly trigger with
  a cursor the *player's hand* is steering; ESC (wired by the caller) is the
  kill-switch instead.
* **dry_run**: full pipeline, no OS input -- for testing the loop with the
  game not running (or not focused). Movement/clicks print instead.

**Click hold time (Strategy 3.2.3).** `pydirectinput.click()` issues ONE
`SendInput` with `dwFlags = MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP` -- press
and release in the *same input event*, hold time **0 ms**. FNAF 1 is a Clickteam
Fusion / DirectX title that samples mouse state once per game frame (~16 ms at
60 FPS), so a pulse that begins and ends between two samples can be observed as
"button was up, and is still up" -- the click never happened. That is the
non-deterministic missed-click symptom from the live tests, and it is
independent of both frame rate and the model.

So a click here is `mouseDown()` -> **hold** -> `mouseUp()`, with the hold
(default 60 ms ~ 3-4 game frames) spanning several frames of *our* loop without
blocking it: `click()` presses and arms a deadline, and the caller's per-frame
`tick()` releases when it expires. Two consequences the caller must respect:

* **The cursor is frozen while the button is down** (`is_pressed`) -- a cursor
  that drifts mid-press turns a click into a drag off the button, which on a
  target the size of a FNAF door button reads as "the click didn't work".
* **`kill()` releases the button first.** The kill-switch must never leave the
  physical mouse button stuck down.

`hold_s=0` restores the old single-`SendInput` behaviour for an A/B.

The kill-switch contract (CLAUDE.md): `kill()` flips one flag checked by every
output call; nothing re-enables it for the life of the process.
"""

from __future__ import annotations

import sys
import time


def _make_dpi_aware() -> None:
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # non-fatal: worst case is scaled coordinates
            pass


def screen_size() -> tuple[int, int]:
    """True-pixel primary screen size (DPI-aware)."""
    _make_dpi_aware()
    if sys.platform == "win32":
        import ctypes

        u32 = ctypes.windll.user32
        return int(u32.GetSystemMetrics(0)), int(u32.GetSystemMetrics(1))
    # Non-Windows dev box fallback: a common size; only the HUD preview uses it.
    return 1920, 1080


class InputSim:
    """Cursor movement + clicks via pydirectinput, behind the kill-switch."""

    def __init__(self, dry_run: bool = False, hold_s: float = 0.06):
        self.dry_run = dry_run
        self.hold_s = hold_s
        self.enabled = True
        self.moves = 0
        self.clicks = 0
        self._release_at: float | None = None  # button-down deadline (3.2.3)
        self._last_xy: tuple[int, int] | None = None
        _make_dpi_aware()
        if dry_run:
            self._pdi = None
            return
        try:
            import pydirectinput
        except ImportError:
            raise SystemExit(
                "Missing dependency: pydirectinput. Install with:\n"
                "    pip install pydirectinput\n"
                "(or run with --dry-run to test the loop without it)"
            )
        pydirectinput.PAUSE = 0.0
        pydirectinput.FAILSAFE = False
        self._pdi = pydirectinput

    @property
    def is_pressed(self) -> bool:
        """True while a click's button-down is still held (3.2.3)."""
        return self._release_at is not None

    def move_to(self, x: int, y: int) -> None:
        if not self.enabled:
            return
        # Hold the cursor still for the duration of a press: moving between
        # button-down and button-up is a drag, not a click, and on a small
        # target the game reads it as a miss.
        if self.is_pressed:
            return
        # The mapper's dead-zone already decided this pixel shouldn't move;
        # re-sending it is one wasted SendInput per frame.
        if (x, y) == self._last_xy:
            return
        self._last_xy = (x, y)
        self.moves += 1
        if self.dry_run:
            return  # position is visible on the caller's HUD; stay quiet
        self._pdi.moveTo(x, y)

    def click(self) -> None:
        """Press the left button at the current cursor position.

        With `hold_s > 0` this only sends button-**down** and arms a release
        deadline -- the caller's per-frame `tick()` sends button-up once the
        hold has elapsed, so the frame loop is never blocked by the hold.
        """
        if not self.enabled or self.is_pressed:
            return
        self.clicks += 1
        if self.dry_run:
            print(f"[dry-run] CLICK #{self.clicks} (hold {self.hold_s * 1000:.0f} ms)")
        if self.hold_s <= 0:  # legacy 0 ms pulse, kept for the A/B
            if not self.dry_run:
                self._pdi.click()
            return
        if not self.dry_run:
            self._pdi.mouseDown()
        self._release_at = time.monotonic() + self.hold_s

    def tick(self, now: float | None = None) -> None:
        """Call once per frame: releases a held button when its hold expires."""
        if self._release_at is None:
            return
        if (now or time.monotonic()) >= self._release_at:
            self._release()

    def _release(self) -> None:
        """Send button-up if one is outstanding. Safe to call any time."""
        if self._release_at is None:
            return
        self._release_at = None
        if not self.dry_run:
            self._pdi.mouseUp()

    def kill(self) -> None:
        """Kill-switch: permanently disable all simulated input (CLAUDE.md).
        Idempotent -- both the global ESC hotkey and the in-window key check
        may call this for the same press; only the first should print/count."""
        if not self.enabled:
            return
        # Release before disabling: a kill mid-press must not leave the physical
        # mouse button stuck down (every output call below is a no-op after).
        self._release()
        self.enabled = False
        print(f"\n[KILL-SWITCH] simulated input disabled "
              f"({self.moves} moves, {self.clicks} clicks sent).")
