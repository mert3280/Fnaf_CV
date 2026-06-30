"""Day-1 input de-risk: drive real FNAF with keyboard-mocked gestures.

Purpose
-------
Prove the *riskiest* half of the Phase-4 pipeline works before any ML exists:

    [keyboard key]  ->  gesture name  ->  mode-aware controller  ->  pydirectinput  ->  FNAF

The keyboard layer is a stand-in for the future classifier. It feeds the SAME
`Controller.handle_gesture(name)` entry point the real webcam loop will call, so
this is the controller skeleton, not a throwaway.

Everything is hard-coded in this file: the key->gesture map and the button
pixel coordinates (COORDS below). Edit COORDS to match your screen, then run.

Usage
----- RUN IN TERMINAL 
    pip install pydirectinput keyboard
    # If clicks don't register in-game, see the TROUBLESHOOTING notes at bottom.
    python -m src.control.keyboard_mock_controller

Then: launch FNAF, get into an actual night, click the FNAF window so it has
focus, and press the keys. ESC is the kill-switch — it stops all input.

The full key/gesture -> action table is in CONTROLS below (printed at startup).
"""

from __future__ import annotations

import sys
import time

try:
    import pydirectinput
    import keyboard
except ImportError as e:  # pragma: no cover - dependency hint
    sys.exit(
        f"Missing dependency: {e.name}. Install with:\n"
        "    pip install pydirectinput keyboard"
    )

# pydirectinput sleeps 0.1s between calls by default; we manage timing ourselves.
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False  # we provide our own ESC kill-switch

# HiDPI fix: on a scaled display (e.g. 2560x1600 at 150%) the OS reports a
# smaller "virtual" resolution, so moveTo lands in the wrong spot and every
# click/hover misses. Declaring the process DPI-aware makes coordinates true
# pixels. Must run before any cursor movement.
if sys.platform == "win32":
    import ctypes

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # pragma: no cover - non-fatal if it fails
        pass

# ---------------------------------------------------------------------------
# HARD-CODED BUTTON COORDINATES — edit these to match YOUR screen (true pixels;
# the process is DPI-aware above). Read a coord by hovering the button and
# running:  python -c "import pyautogui,time; time.sleep(3); print(pyautogui.position())"
# Doors/lights/cameras are CLICK targets. The monitor is a HOVER target: in
# FNAF 1 the camera flips up/down when the cursor touches the bottom CAMERA bar.
# ---------------------------------------------------------------------------
COORDS = {
    "left_door":   (220, 650),   # nudged ~30px outward (toward left edge)
    "right_door":  (1700, 650),  # nudged ~30px outward (toward right edge)
    "left_light":  (105, 650),   # nudged ~45px outward
    "right_light": (1815, 650),  # nudged ~45px outward
    "monitor_bar": (1280, 1580),  # bottom-center "CAMERA" tab — HOVER to flip
    "cam_prev":    (1500, 950),
    "cam_next":    (1650, 950),
    "neutral":     (1280, 800),   # park spot away from any hot-zone
}

# How long (seconds) to hold the cursor on the camera bar so the flip triggers.
MONITOR_HOVER_SEC = 0.3

# CONTROLS REFERENCE — what each key / gesture does in game. Printed at startup.
# Each KEY mocks a GESTURE; the same gesture means different things per MODE.
# Rebind freely: edit KEY_TO_GESTURE (keys) and Controller._office / _camera
# (gesture -> action). Mapping follows phase-1 §4.
CONTROLS = """\
============================ CONTROLS (start in OFFICE) ============================
 OFFICE mode (monitor DOWN):         CAMERA mode (monitor UP):
   a   like     toggle LEFT door       a   like     previous camera
   d   fist     toggle RIGHT door      x   dislike  next camera
   q   one      tap LEFT light         d   fist     lower monitor -> OFFICE
   e   two_up   tap RIGHT light        s   mute     idle / no-op
   w   palm     raise monitor -> CAMERA
   s   mute     idle / no-op
 ESC = kill-switch: disable all input and quit.
==================================================================================="""

# Each physical key maps to one gesture name. The controller decides what each
# gesture does based on the current mode (Office vs Camera).
KEY_TO_GESTURE = {
    "a": "like",
    "d": "fist",
    "q": "one",
    "e": "two_up",
    "w": "palm",
    "s": "mute",
    "x": "dislike",
    "space": "ok",
}


class Controller:
    """Minimal mode-aware state machine: gesture (+mode) -> simulated input."""

    def __init__(self) -> None:
        self.mode = "office"  # "office" | "camera"
        self.enabled = True   # kill-switch flips this off

    def _click(self, name: str) -> None:
        if not self.enabled:
            return
        x, y = COORDS[name]
        pydirectinput.moveTo(x, y)
        pydirectinput.click()
        print(f"  click {name:<12} ({x}, {y})")

    def _monitor(self) -> None:
        """Flip the camera monitor by HOVERING the bottom CAMERA bar (no click).

        Afterward we move the cursor to a neutral spot so the *next* flip is a
        fresh hover-enter on the bar (the game toggles on enter, not while held).
        """
        if not self.enabled:
            return
        bx, by = COORDS["monitor_bar"]
        pydirectinput.moveTo(bx, by)
        time.sleep(MONITOR_HOVER_SEC)  # let the flip animation trigger
        nx, ny = COORDS["neutral"]
        pydirectinput.moveTo(nx, ny)   # leave the bar so the next hover re-fires
        print(f"  monitor hover ({bx}, {by})")

    def handle_gesture(self, gesture: str) -> None:
        """Entry point — the real webcam loop will call this too."""
        if not self.enabled or gesture == "mute":
            return
        (self._office if self.mode == "office" else self._camera)(gesture)

    def _office(self, g: str) -> None:
        if g == "like":
            self._click("left_door")
        elif g == "fist":
            self._click("right_door")
        elif g == "one":
            self._click("left_light")
        elif g == "two_up":
            self._click("right_light")
        elif g == "palm":
            self._monitor()
            self.mode = "camera"
            print("  -> mode: CAMERA")
        else:
            print(f"  (office: '{g}' unmapped)")

    def _camera(self, g: str) -> None:
        if g == "like":
            self._click("cam_prev")
        elif g == "dislike":
            self._click("cam_next")
        elif g == "fist":
            self._monitor()
            self.mode = "office"
            print("  -> mode: OFFICE")
        else:
            print(f"  (camera: '{g}' unmapped)")

    def kill(self) -> None:
        self.enabled = False
        print("\n[KILL-SWITCH] simulated input disabled.")


def main() -> None:
    ctrl = Controller()

    print(CONTROLS)
    print("\nFocus the FNAF window, then press keys.\n")

    def on_key(gesture: str, key: str) -> None:
        print(f"[{key}] gesture={gesture} mode={ctrl.mode}")
        ctrl.handle_gesture(gesture)

    for key, gesture in KEY_TO_GESTURE.items():
        keyboard.add_hotkey(key, on_key, args=(gesture, key))

    keyboard.wait("esc")  # blocks until ESC
    ctrl.kill()
    time.sleep(0.1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# TROUBLESHOOTING — "it didn't work in-game"
#
# 1. HiDPI / display scaling: now handled — SetProcessDPIAware() runs at import,
#    so COORDS are TRUE pixels. But the placeholder COORDS are still guesses; you
#    must read your real button pixels (see the COORDS comment) for clicks to land.
#
# 2. Window focus: the game MUST be the foreground window when a key fires.
#    Click into FNAF first; the keyboard hotkeys still fire globally.
#
# 3. Admin elevation mismatch: if FNAF runs elevated and this script doesn't,
#    Windows silently blocks injected input. Run the terminal AS ADMINISTRATOR.
#
# 4. Fullscreen-exclusive DirectX is the hardest case for injection. Try
#    windowed / borderless FNAF first.
#
# 5. Sanity-check the move alone: comment out .click() and watch whether the
#    cursor even jumps to the right button. If the cursor lands wrong, it's a
#    coordinate/DPI problem (#1), not an injection problem.
# ---------------------------------------------------------------------------
