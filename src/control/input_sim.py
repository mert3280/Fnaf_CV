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

The kill-switch contract (CLAUDE.md): `kill()` flips one flag checked by every
output call; nothing re-enables it for the life of the process.
"""

from __future__ import annotations

import sys


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

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.enabled = True
        self.moves = 0
        self.clicks = 0
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

    def move_to(self, x: int, y: int) -> None:
        if not self.enabled:
            return
        self.moves += 1
        if self.dry_run:
            return  # position is visible on the caller's HUD; stay quiet
        self._pdi.moveTo(x, y)

    def click(self) -> None:
        """One left click at the current cursor position."""
        if not self.enabled:
            return
        self.clicks += 1
        if self.dry_run:
            print(f"[dry-run] CLICK #{self.clicks}")
            return
        self._pdi.click()

    def kill(self) -> None:
        """Kill-switch: permanently disable all simulated input (CLAUDE.md).
        Idempotent -- both the global ESC hotkey and the in-window key check
        may call this for the same press; only the first should print/count."""
        if not self.enabled:
            return
        self.enabled = False
        print(f"\n[KILL-SWITCH] simulated input disabled "
              f"({self.moves} moves, {self.clicks} clicks sent).")
