"""Play FNAF hands-free: the full Strategy-3 control loop (AD-17).

Composes every piece built so far into the real thing:

    webcam -> MediaPipe detect -+-> palm anchor -> CursorMapper -> move cursor
                                +-> crop -> CNN (palm/fist) -> ClickFSM -> click

Run it **dry first** (full pipeline, no OS input) to sanity-check tracking and
click detection, then live against the game:

    python -m src.control.play --checkpoint models/palmfist_frozen_mnv3_large/best.pt --dry-run
    python -m src.control.play --checkpoint models/palmfist_frozen_mnv3_large/best.pt

Day-1 registration check (plan.md Step 5): with FNAF focused, confirm (1) the
cursor MOVES smoothly in-game, (2) a fist CLICK lands on a door button. If
moves/clicks don't register: windowed/borderless FNAF, run terminal as admin --
see the troubleshooting notes in keyboard_mock_controller.py.

**ESC is the global kill-switch** (works even while FNAF has focus, via the
`keyboard` hook): it permanently disables simulated input and exits. `q` in the
preview window quits too. All tuning knobs (control box, smoothing, debounce K,
confidence, cooldown) are CLI flags -- their values are Ted's live calls; the
defaults are the Strategy-3 starting points.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.control.click_fsm import ClickFSM
from src.control.input_sim import InputSim, screen_size
from src.control.strategies import FSM_PRESETS, resolve_fsm
from src.rt.backends import make_runner
from src.rt.capture import open_camera
from src.rt.cursor import CursorMapper, hand_anchor_norm, palm_span
from src.rt.model_loader import load_checkpoint
from src.rt.preprocess import Preprocessor

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to a best.pt (palm/fist model)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full pipeline but send NO real input (test mode)")
    ap.add_argument("--no-preview", action="store_true", help="disable the HUD window")
    ap.add_argument("--no-mirror", action="store_true",
                    help="do not mirror the webcam (mapper compensates)")
    # detector (same flags/defaults as webcam_demo)
    ap.add_argument("--pad", type=float, default=0.35,
                    help="live hand-bbox padding on the MediaPipe hull "
                         "(0.35 recalibrates for the hull-vs-annotated-bbox gap, "
                         "Strategy 3.2.4 / AD-25; training itself still pads at 0.15)")
    ap.add_argument("--hand-model", default=None)
    ap.add_argument("--detect-confidence", type=float, default=0.3)
    ap.add_argument("--presence-confidence", type=float, default=0.3)
    ap.add_argument("--tracking-confidence", type=float, default=0.3)
    # cursor mapper (AD-19) -- Ted's tuning surface
    ap.add_argument("--anchor", choices=("palm", "box"), default="palm",
                    help="cursor anchor: palm plate or landmark-hull center (3.1)")
    ap.add_argument("--box-mode", choices=("adaptive", "fixed"), default="adaptive",
                    help="adaptive (3.1.1): constant gain across distance; fixed: 3.1 box")
    ap.add_argument("--box-gain", type=float, default=4.0,
                    help="adaptive box: width = gain x palm span")
    ap.add_argument("--box-w", type=float, default=0.60, help="control-box width (frame frac)")
    ap.add_argument("--box-h", type=float, default=0.55, help="control-box height (frame frac)")
    ap.add_argument("--cursor-alpha", type=float, default=0.35,
                    help="EMA weight of the new sample (higher = more responsive)")
    ap.add_argument("--deadzone", type=float, default=0.005,
                    help="min normalized move before the cursor budges")
    # click FSM (AD-20) -- Ted's tuning surface
    ap.add_argument("--strategy", choices=list(FSM_PRESETS), default=None,
                    help="named click-FSM preset (sets conf + K): "
                         + "; ".join(f"{k} = {v.blurb}" for k, v in FSM_PRESETS.items())
                         + ". Explicit --fsm-conf / --fsm-k override it.")
    ap.add_argument("--fsm-k", type=int, default=None,
                    help="consecutive confident frames to confirm (default 3, or the --strategy preset)")
    ap.add_argument("--fsm-conf", type=float, default=None,
                    help="min confidence per frame (default 0.70, or the --strategy preset)")
    ap.add_argument("--fsm-grace", type=int, default=10,
                    help="ambiguous frames tolerated before disarm (3.1.1)")
    ap.add_argument("--cooldown", type=float, default=0.30, help="min seconds between clicks")
    # runtime / latency (Strategy 3.2.3) -- these change how fast the loop runs
    # and how the click is delivered, NOT what the model or the FSM decide.
    ap.add_argument("--backend", choices=("auto", "onnx", "torch"), default="auto",
                    help="stage-2 inference backend: onnx is 3-4x faster on this CPU and "
                         "parity-checked at load; torch reverts (default: auto)")
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="torch intra-op threads (measured: 4 beats the 16-thread default)")
    ap.add_argument("--no-threaded-capture", action="store_true",
                    help="grab frames synchronously (pre-3.2.3 behaviour: same FPS, but "
                         "~67 ms staler frames -- the driver serves from a 2-frame backlog)")
    ap.add_argument("--click-hold", type=float, default=0.06,
                    help="seconds to hold the mouse button down per click; 0 = the old "
                         "single-SendInput pulse a DirectX game can miss")
    ap.add_argument("--hud-every", type=int, default=1,
                    help="render the preview every Nth frame (2 saves a few ms)")
    return ap.parse_args()


def warn_if_on_battery() -> None:
    """On this laptop, battery power drops the CPU to its 1.4 GHz base clock and
    costs ~2.1x on every stage -- a bigger term than anything Strategy 3.2.3
    changed (9-11 FPS on battery vs. 28 on AC). Cheap to check, easy to forget."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        class SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        st = SPS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st))
        if st.ACLineStatus == 0:
            print("[warn] running ON BATTERY -- the CPU drops to its base clock and the "
                  "loop runs at roughly half speed (3.2.3). Plug in for the live test.")
    except Exception:  # never let a diagnostic break the loop
        pass


def hook_kill_switch(sim: InputSim) -> bool:
    """Global ESC -> kill-switch, active even while FNAF has focus. Returns
    whether the hook was installed (needs the `keyboard` package)."""
    try:
        import keyboard
    except ImportError:
        print("[warn] `keyboard` package missing -- ESC kill-switch only works "
              "with the preview window focused. pip install keyboard")
        return False
    keyboard.add_hotkey("esc", sim.kill)
    return True


def draw_hud(frame, mapper, hand_box, cursor_px, screen, label, conf, fsm, sim, fps, clicked,
             dropped=0, backend=""):
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = mapper.box_px(w, h)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 90, 90), 1)  # control box
    if hand_box is not None:
        l, t, r, b = hand_box
        cv2.rectangle(frame, (l, t), (r, b), (255, 200, 0), 2)
    if cursor_px is not None:  # virtual cursor, scaled screen -> frame
        cx = int(cursor_px[0] / (screen[0] - 1) * (w - 1))
        cy = int(cursor_px[1] / (screen[1] - 1) * (h - 1))
        color = (0, 0, 255) if clicked else (0, 255, 255)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 22, 2)

    mode = "DRY-RUN" if sim.dry_run else ("LIVE" if sim.enabled else "KILLED")
    top = f"[{mode}]  {label or '-- no hand --'}"
    if label:
        top += f" {conf*100:.0f}%"
    cv2.putText(frame, top, (10, 24), FONT, 0.6, (0, 230, 0) if sim.enabled else (0, 0, 255),
                2, cv2.LINE_AA)
    cv2.putText(frame, fsm.describe(), (10, 48), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    # `drop` = frames the camera produced that the loop was too slow to take:
    # a direct readout of "the pipeline is behind the camera" (3.2.3).
    foot = (f"{fps:4.1f}FPS {backend} drop{dropped}  cursor={cursor_px}  "
            f"clicks={sim.clicks}  ESC=kill  q=quit")
    cv2.putText(frame, foot, (10, h - 12), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if clicked:
        cv2.putText(frame, "CLICK", (w // 2 - 40, 40), FONT, 1.0, (0, 0, 255), 3, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    lm = load_checkpoint(args.checkpoint, device=args.device)
    print("loaded", lm.describe())
    # Binary click model: one "fist" (click) class + one no-click class. That
    # covers both palm/fist (AD-18) and not_fist/fist (Strategy 3.2 / AD-21);
    # the no-click label is whichever of the two isn't "fist".
    if "fist" not in lm.classes or lm.num_classes != 2:
        raise SystemExit(
            f"checkpoint classes {lm.classes} are not a binary click model "
            "(need exactly 2 classes incl. 'fist') -- wrong model."
        )
    noclick_label = next(c for c in lm.classes if c != "fist")
    print(f"click model: no-click='{noclick_label}'  click='fist'")
    if lm.crop_mode != "bbox":
        print("[warn] full_frame checkpoint: classifier gets the whole frame; the "
              "detector still runs for the cursor.")
    pre = Preprocessor(lm.size, lm.mean, lm.std)
    # 3.2.3: same tensor in, same probabilities out -- only faster. The ONNX
    # path is parity-checked against torch at load or it isn't used.
    predict, backend = make_runner(lm, args.backend, args.device, args.torch_threads)

    # The cursor NEEDS the detector (unlike the demo there is no ROI fallback).
    from src.rt.detector import DEFAULT_MODEL, HandDetector
    detector = HandDetector(
        args.hand_model or DEFAULT_MODEL, pad=args.pad,
        min_detection_confidence=args.detect_confidence,
        min_presence_confidence=args.presence_confidence,
        min_tracking_confidence=args.tracking_confidence,
    )

    screen = screen_size()
    mapper = CursorMapper(
        screen_w=screen[0], screen_h=screen[1],
        box_w=args.box_w, box_h=args.box_h,
        box_mode=args.box_mode, box_gain=args.box_gain,
        alpha=args.cursor_alpha, deadzone=args.deadzone,
        mirror_x=args.no_mirror,  # un-mirrored frame -> mapper does the mirror
    )
    # --strategy sets (conf, K); explicit --fsm-conf/--fsm-k still override it.
    fsm_conf, fsm_k = resolve_fsm(args.strategy, conf=args.fsm_conf, k=args.fsm_k)
    fsm = ClickFSM(k=fsm_k, conf_threshold=fsm_conf,
                   cooldown_s=args.cooldown, grace=args.fsm_grace,
                   palm_label=noclick_label, fist_label="fist")
    sim = InputSim(dry_run=args.dry_run, hold_s=args.click_hold)
    hook_kill_switch(sim)

    box_txt = (f"box adaptive gain={args.box_gain}" if args.box_mode == "adaptive"
               else f"box {args.box_w:.2f}x{args.box_h:.2f}")
    strat_txt = f"strategy {args.strategy}  " if args.strategy else ""
    print(f"screen {screen[0]}x{screen[1]}  {box_txt}  "
          f"alpha {args.cursor_alpha}  {strat_txt}K={fsm_k}  conf>={fsm_conf}  "
          f"grace {args.fsm_grace}  cooldown {args.cooldown}s  "
          f"{'DRY-RUN' if args.dry_run else 'LIVE'}")
    warn_if_on_battery()
    print(f"runtime: backend {backend}  "
          f"capture {'threaded' if not args.no_threaded_capture else 'sync'}  "
          f"click-hold {args.click_hold * 1000:.0f} ms")
    if not args.dry_run:
        print("Focus the FNAF window. ESC = kill-switch.")

    cap = open_camera(args.camera, threaded=not args.no_threaded_capture)
    win = "FNAF hands-free control"
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    mirror = not args.no_mirror
    last, fps = time.time(), 0.0
    click_flash = 0
    frames = 0

    try:
        while sim.enabled:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed"); break
            frames += 1
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            label, conf, hand_box, clicked = None, 0.0, None, False
            hb = detector.detect(frame)
            if hb is not None:
                hand_box = hb.box_px
                cursor_px = mapper.update(hand_anchor_norm(hb.landmarks_norm, args.anchor),
                                          hand_scale=palm_span(hb.landmarks_norm))
                model_in = detector.crop(frame, hb) if lm.crop_mode == "bbox" else frame
                if model_in.size > 0:
                    probs = predict(pre(model_in))
                    idx = int(probs.argmax())
                    label, conf = lm.classes[idx], float(probs[idx])
            else:
                cursor_px = mapper.update(None)  # freeze

            # Release a held click whose hold has expired (3.2.3), before the
            # move: the cursor stays frozen for exactly the press's duration.
            sim.tick()
            if cursor_px is not None:
                sim.move_to(*cursor_px)
            if fsm.update(label, conf):
                sim.click()
                clicked, click_flash = True, 6
                print(f"CLICK at {cursor_px}")

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            if not args.no_preview and frames % max(1, args.hud_every) == 0:
                click_flash = max(0, click_flash - 1)
                draw_hud(frame, mapper, hand_box, cursor_px, screen,
                         label, conf, fsm, sim, fps, clicked or click_flash > 0,
                         dropped=cap.dropped, backend=backend)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    sim.kill()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        if sim.enabled:
            sim.kill()


if __name__ == "__main__":
    main()
