"""Threaded webcam capture -- fix #1 of Strategy 3.2.3.

`cv2.VideoCapture.read()` is **synchronous**: it blocks until the camera's next
frame is ready. Measured on this box (640x480 YUY2 via DSHOW) that is **33.6 ms**
of the ~90 ms frame budget spent doing nothing, and it gets worse than "wasted
time" when the loop is slower than the camera -- the driver queues frames, so
`read()` hands back a **stale** one and the hand we classify is where the hand
*was*, ~100 ms ago. That lag is what turns a quick squeeze into a missed click.

`ThreadedCapture` moves the grab onto its own thread and keeps exactly **one**
frame -- the newest. The consumer never waits on the camera (the wait overlaps
with MediaPipe + the classifier) and never sees a stale frame: frames the loop
was too slow to take are *dropped*, counted, and reported, which is the honest
behaviour for a control loop. `dropped` on the HUD is a direct readout of "the
loop is slower than the camera."

`read()` blocks until a frame **newer than the last one you took** arrives, so a
loop that outruns the camera idles instead of re-processing the same frame twice
(which would inflate the FPS counter while doing nothing new).

Drop-in for the `cap.read()` / `cap.release()` pair; `--no-threaded-capture` on
`play.py` reverts to the plain synchronous `cv2.VideoCapture`.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class ThreadedCapture:
    """Background grabber that keeps only the most recent frame."""

    def __init__(
        self,
        index: int = 0,
        api: int = cv2.CAP_DSHOW,  # DSHOW: fast open on Windows
        width: int | None = None,
        height: int | None = None,
    ):
        self._cap = cv2.VideoCapture(index, api)
        if not self._cap.isOpened():
            raise SystemExit(f"could not open camera {index}")
        # Ask the driver for a 1-deep queue too. Not all backends honour it (DSHOW
        # commonly reports -1), which is exactly why the latest-frame slot below
        # is the real guarantee rather than an optimization on top of it.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._lock = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0          # frames produced
        self._taken = 0        # frames actually consumed
        self._failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    # -- producer ---------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                if not ok:
                    self._failed = True
                    self._lock.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._lock.notify_all()

    # -- consumer ---------------------------------------------------------
    def read(self, timeout: float = 2.0) -> tuple[bool, np.ndarray | None]:
        """Newest unseen frame. Blocks until one arrives (or `timeout`)."""
        with self._lock:
            got = self._lock.wait_for(
                lambda: self._failed or self._seq > self._taken, timeout=timeout
            )
            if self._failed or not got or self._frame is None:
                return False, None
            self._taken = self._seq
            return True, self._frame

    @property
    def dropped(self) -> int:
        """Frames the camera produced that the loop never took. A live readout
        of how far behind the camera the pipeline is running."""
        with self._lock:
            return max(0, self._seq - self._taken)

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()

    def __enter__(self) -> "ThreadedCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class SyncCapture:
    """Plain `cv2.VideoCapture` behind the same tiny interface, so the loop can
    A/B threaded vs. synchronous capture without branching (`--no-threaded-capture`)."""

    def __init__(self, index: int = 0, api: int = cv2.CAP_DSHOW):
        self._cap = cv2.VideoCapture(index, api)
        if not self._cap.isOpened():
            raise SystemExit(f"could not open camera {index}")

    def read(self, timeout: float = 2.0) -> tuple[bool, np.ndarray | None]:
        return self._cap.read()

    @property
    def dropped(self) -> int:
        return 0  # unknowable here: stale frames arrive silently

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "SyncCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def open_camera(index: int = 0, threaded: bool = True):
    """`ThreadedCapture` (3.2.3 default) or the synchronous fallback."""
    return ThreadedCapture(index) if threaded else SyncCapture(index)
