"""The hand-tracking control loop as a start/stop background thread.

This is `src.control.play`'s per-frame pipeline, lifted into a class the web
server can own:

    webcam -> MediaPipe detect -+-> palm anchor -> CursorMapper -> move REAL cursor
                                +-> crop -> CNN (palm/fist) -> ClickFSM -> REAL click

The dashboard needs the tracker as an object it can `start()` when the page
loads and `stop()` when the user hits *Finish* -- so the survey can be filled
with the normal mouse. `stop()` is the kill-switch (CLAUDE.md): it flips
`InputSim`'s one-way flag and joins the loop. All tuning knobs default to the
Strategy-3 starting points (Ted's live calls); the server exposes them as CLI
flags so a run can reuse whatever he tuned in `play.py`.

Preview is OFF by default: the browser is the thing being looked at, and a
cv2 window created off the main thread is flaky on some setups. ESC still kills
input globally via the `keyboard` hook, exactly as in `play.py`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackerConfig:
    """Everything the loop needs; mirrors the `src.control.play` flags."""

    checkpoint: str
    camera: int = 0
    device: str = "cpu"
    dry_run: bool = False          # run the pipeline but send NO real OS input
    mirror: bool = True
    # detector
    pad: float = 0.35  # recalibrated hull-vs-annotated-bbox gap (Strategy 3.2.4 / AD-25)
    hand_model: str | None = None
    detect_confidence: float = 0.3
    presence_confidence: float = 0.3
    tracking_confidence: float = 0.3
    # cursor mapper (AD-19)
    anchor: str = "palm"
    box_mode: str = "adaptive"
    box_gain: float = 4.0
    box_w: float = 0.60
    box_h: float = 0.55
    cursor_alpha: float = 0.35
    deadzone: float = 0.005
    # click FSM (AD-20)
    fsm_k: int = 3
    fsm_conf: float = 0.70
    fsm_grace: int = 10
    cooldown: float = 0.30
    # runtime / latency (Strategy 3.2.3) -- mirrors src.control.play's flags
    backend: str = "auto"
    torch_threads: int = 4
    threaded_capture: bool = True
    click_hold: float = 0.06


@dataclass
class TrackerStatus:
    """A snapshot the server hands to the browser (`/api/status`)."""

    running: bool = False
    input_enabled: bool = False
    dry_run: bool = False
    hand_seen: bool = False
    label: str | None = None
    confidence: float = 0.0
    fsm: str = ""
    fps: float = 0.0
    clicks: int = 0
    error: str | None = None


class HandTracker:
    """Owns the webcam + models + InputSim and runs the control loop in a thread."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status = TrackerStatus(dry_run=cfg.dry_run)
        self._sim = None  # created in the loop thread
        self._hook_installed = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        # a fresh run gets a fresh InputSim; forget the killed one so status /
        # the loop condition don't read a stale disabled sim before _run rebuilds it.
        self._sim = None
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        """Kill-switch + shutdown. Disables input immediately, then joins the
        loop so the webcam/detector are released before we return."""
        self._stop.set()
        sim = self._sim
        if sim is not None:
            sim.kill()  # one-way: no simulated input can fire after this
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=join_timeout)

    def status(self) -> TrackerStatus:
        with self._lock:
            # cheap copy so the caller never sees a half-written update
            return TrackerStatus(**vars(self._status))

    # -- the loop ----------------------------------------------------------
    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._status, k, v)

    def _run(self) -> None:
        # Imports live here so `--no-tracker` can serve the UI on a box without
        # a camera / torch / mediapipe installed.
        try:
            import cv2

            from src.control.click_fsm import ClickFSM
            from src.control.input_sim import InputSim, screen_size
            from src.rt.backends import make_runner
            from src.rt.capture import open_camera
            from src.rt.cursor import CursorMapper, hand_anchor_norm, palm_span
            from src.rt.detector import DEFAULT_MODEL, HandDetector
            from src.rt.model_loader import load_checkpoint
            from src.rt.preprocess import Preprocessor
        except Exception as e:  # missing dep -> report, don't crash the server
            self._set(error=f"tracker deps unavailable: {type(e).__name__}: {e}")
            return

        cfg = self.cfg
        try:
            lm = load_checkpoint(cfg.checkpoint, device=cfg.device)
            if "fist" not in lm.classes or lm.num_classes != 2:
                raise ValueError(
                    f"checkpoint classes {lm.classes} are not a binary click model "
                    "(need exactly 2 classes incl. 'fist')."
                )
            noclick_label = next(c for c in lm.classes if c != "fist")
            pre = Preprocessor(lm.size, lm.mean, lm.std)
            # 3.2.3: same tensor in, same probabilities out -- only faster. The
            # ONNX path is parity-checked against torch at load or it isn't used.
            predict, backend = make_runner(lm, cfg.backend, cfg.device, cfg.torch_threads)
            detector = HandDetector(
                cfg.hand_model or DEFAULT_MODEL, pad=cfg.pad,
                min_detection_confidence=cfg.detect_confidence,
                min_presence_confidence=cfg.presence_confidence,
                min_tracking_confidence=cfg.tracking_confidence,
            )
            screen = screen_size()
            mapper = CursorMapper(
                screen_w=screen[0], screen_h=screen[1],
                box_w=cfg.box_w, box_h=cfg.box_h,
                box_mode=cfg.box_mode, box_gain=cfg.box_gain,
                alpha=cfg.cursor_alpha, deadzone=cfg.deadzone,
                mirror_x=not cfg.mirror,
            )
            fsm = ClickFSM(k=cfg.fsm_k, conf_threshold=cfg.fsm_conf,
                           cooldown_s=cfg.cooldown, grace=cfg.fsm_grace,
                           palm_label=noclick_label, fist_label="fist")
            self._sim = InputSim(dry_run=cfg.dry_run, hold_s=cfg.click_hold)
            self._ensure_kill_hook()

            try:
                cap = open_camera(cfg.camera, threaded=cfg.threaded_capture)
            except SystemExit as e:  # open_camera raises this on a CLI; catch it here
                raise RuntimeError(str(e)) from e
        except Exception as e:
            self._set(error=f"tracker init failed: {type(e).__name__}: {e}")
            return

        self._set(running=True, input_enabled=True, error=None)
        print(f"[tracker] runtime: backend {backend}  "
              f"capture {'threaded' if cfg.threaded_capture else 'sync'}  "
              f"click-hold {cfg.click_hold * 1000:.0f} ms")

        last, fps = time.time(), 0.0
        try:
            while not self._stop.is_set() and self._sim.enabled:
                ok, frame = cap.read()
                if not ok:
                    self._set(error="frame grab failed")
                    break
                if cfg.mirror:
                    frame = cv2.flip(frame, 1)

                label, conf, hand_seen = None, 0.0, False
                hb = detector.detect(frame)
                if hb is not None:
                    hand_seen = True
                    cursor_px = mapper.update(
                        hand_anchor_norm(hb.landmarks_norm, cfg.anchor),
                        hand_scale=palm_span(hb.landmarks_norm),
                    )
                    model_in = detector.crop(frame, hb) if lm.crop_mode == "bbox" else frame
                    if model_in.size > 0:
                        probs = predict(pre(model_in))
                        idx = int(probs.argmax())
                        label, conf = lm.classes[idx], float(probs[idx])
                else:
                    cursor_px = mapper.update(None)  # freeze on dropout

                # Release a held click whose hold has expired (3.2.3), before the
                # move: the cursor stays frozen for exactly the press's duration.
                self._sim.tick()
                if cursor_px is not None:
                    self._sim.move_to(*cursor_px)
                if fsm.update(label, conf):
                    self._sim.click()

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
                last = now
                self._set(hand_seen=hand_seen, label=label, confidence=conf,
                          fsm=fsm.describe(), fps=fps, clicks=self._sim.clicks,
                          input_enabled=self._sim.enabled)
        finally:
            cap.release()
            detector.close()
            if self._sim is not None and self._sim.enabled:
                self._sim.kill()
            self._set(running=False, input_enabled=False)

    def _kill_switch(self) -> None:
        """ESC target: kill the current run's input and stop the loop. Bound once
        but always reads `self._sim`, so it works across start/stop cycles."""
        sim = self._sim
        if sim is not None:
            sim.kill()
        self._stop.set()

    def _ensure_kill_hook(self) -> None:
        """Global ESC -> kill-switch, active even without the browser focused.
        Installed once (a restart reuses it) so hooks never stack up."""
        if self._hook_installed:
            return
        try:
            import keyboard
        except ImportError:
            return
        try:
            keyboard.add_hotkey("esc", self._kill_switch)
            self._hook_installed = True
        except Exception:
            pass
