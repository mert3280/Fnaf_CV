"""Live webcam gesture demo -- a simple UI to eyeball a trained model.

Feeds your webcam through a checkpoint and overlays the model's top-k guesses in
real time. Point `--checkpoint` at any `best.pt` to swap models; everything the
model needs (backbone, classes, input spec) is read from the file itself.

Two-stage pipeline (AD-04): a **MediaPipe hand detector** (`src/rt/detector.py`)
finds and crops the hand, then the CNN classifies the crop -- exactly how a
`crop_mode="bbox"` model was trained. The crop source is auto-selected from the
checkpoint:

  * bbox model       -> MediaPipe detector crop   (the real hand box)
  * full_frame model -> whole frame               (no detector)

Strategy-2.1 robustness aids for the live distance gap (DOCS/build/strategies):
  * a **framing hint** warns when the hand is held too close to the lens (the
    out-of-distribution case vs HaGRID's at-a-distance hands) -- fix #1;
  * **live pad tuning** with `[` / `]` to reconcile the MediaPipe box tightness
    with the training box, read straight off the HUD -- fix #2;
  * **detection-confidence** flags to tune how eagerly MediaPipe locks on -- fix #6.

    python -m src.rt.webcam_demo --checkpoint models/palmfist_frozen_mnv3_large/best.pt

Strategy-3 preview (AD-17): when the detector is on, the HUD also shows the
**cursor-control pipeline without touching the real mouse** -- the control box
(AD-19), a crosshair where the cursor *would* be (the frame stands in for the
screen), and the click FSM's state (AD-20) with a CLICK flash on a confirmed
palm->fist squeeze. Tune the mapper/FSM flags here safely, then carry the same
values to `src.control.play`, which drives real input.

Keys:  q/Esc quit | m mirror | d detector on/off | r ROI-fallback | [ / ] pad -/+

Note: diagnostic viewer only -- no game input is simulated here (that is
`src/control/play.py`, which has the kill-switch), so there is nothing to
kill-switch.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from src.control.click_fsm import ClickFSM
from src.rt.cursor import CursorMapper, hand_anchor_norm, palm_span
from src.rt.model_loader import load_checkpoint
from src.rt.preprocess import Preprocessor, RoiCropper

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to a best.pt checkpoint")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--topk", type=int, default=5, help="how many predictions to show")
    ap.add_argument("--roi", type=float, default=0.6,
                    help="fallback ROI square side as a fraction of the short side")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="hand-bbox padding for the detector crop (match training: 0.15)")
    ap.add_argument("--pad-step", type=float, default=0.05,
                    help="how much the [ and ] keys change pad live")
    ap.add_argument("--near-frac", type=float, default=0.4,
                    help="hand-area fraction above which the 'too close' hint fires (fix #1)")
    ap.add_argument("--hand-model", default=None,
                    help="path to hand_landmarker.task (default models/mediapipe/...)")
    ap.add_argument("--detect-confidence", type=float, default=0.3,
                    help="MediaPipe min_hand_detection_confidence (fix #6; 3.1.1 default 0.3)")
    ap.add_argument("--presence-confidence", type=float, default=0.3,
                    help="MediaPipe min_hand_presence_confidence (fix #6; 3.1.1 default 0.3)")
    ap.add_argument("--tracking-confidence", type=float, default=0.3,
                    help="MediaPipe min_tracking_confidence (fix #6; 3.1.1 default 0.3)")
    ap.add_argument("--detector", dest="detector", action="store_true", default=None,
                    help="force the MediaPipe detector on")
    ap.add_argument("--no-detector", dest="detector", action="store_false",
                    help="force the detector off (use ROI / full frame)")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="EMA factor on probabilities, 0=off .. <1 (higher=steadier)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    # Strategy-3 preview knobs -- same flags/defaults as src.control.play, so
    # values tuned here transfer directly. All of them are Ted's live calls.
    ap.add_argument("--box-mode", choices=("adaptive", "fixed"), default="adaptive",
                    help="adaptive (3.1.1): box scales with hand size so the same arm "
                         "motion moves the cursor the same at any distance; "
                         "fixed: the 3.1 static box (--box-w/--box-h)")
    ap.add_argument("--box-gain", type=float, default=4.0,
                    help="adaptive box: width = gain x palm span (higher = less arm travel)")
    ap.add_argument("--box-w", type=float, default=0.60,
                    help="cursor control-box width as a frame fraction (fixed mode / aspect)")
    ap.add_argument("--box-h", type=float, default=0.55,
                    help="cursor control-box height as a frame fraction (fixed mode / aspect)")
    ap.add_argument("--anchor", choices=("palm", "box"), default="palm",
                    help="cursor anchor point: palm plate (stable on squeeze) or "
                         "landmark-hull center (3.1; both are size-invariant)")
    ap.add_argument("--cursor-alpha", type=float, default=0.35,
                    help="cursor EMA weight of the new sample (higher = more responsive)")
    ap.add_argument("--deadzone", type=float, default=0.005,
                    help="min normalized move before the virtual cursor budges")
    ap.add_argument("--fsm-k", type=int, default=3,
                    help="click FSM: consecutive confident frames to confirm a pose (AD-20)")
    ap.add_argument("--fsm-conf", type=float, default=0.70,
                    help="click FSM: min per-frame confidence")
    ap.add_argument("--fsm-grace", type=int, default=10,
                    help="click FSM: ambiguous frames tolerated before disarm (3.1.1)")
    ap.add_argument("--click-cooldown", type=float, default=0.30,
                    help="click FSM: min seconds between fires")
    return ap.parse_args()


@torch.no_grad()
def predict(model, tensor, device) -> np.ndarray:
    """Forward pass -> softmax probabilities as a 1-D numpy array."""
    logits = model(tensor.to(device))
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def try_make_detector(args, want: bool):
    """Build the MediaPipe detector, or return (None, reason) if unavailable."""
    if not want:
        return None, "disabled"
    try:
        from src.rt.detector import DEFAULT_MODEL, HandDetector
        path = args.hand_model or DEFAULT_MODEL
        det = HandDetector(
            path,
            pad=args.pad,
            min_detection_confidence=args.detect_confidence,
            min_presence_confidence=args.presence_confidence,
            min_tracking_confidence=args.tracking_confidence,
        )
        return det, "ok"
    except Exception as e:  # missing mediapipe or model bundle -> graceful fallback
        return None, f"{type(e).__name__}: {e}"


def framing_hint(area_frac: float, near_frac: float):
    """Fix #1: turn the detected hand size into a distance hint. Big fraction =
    hand held up close to the lens = out-of-distribution vs HaGRID. Returns
    (text, bgr_color) or None when framing looks fine."""
    if area_frac >= near_frac:
        return "hand too close - move it back toward your body", (0, 165, 255)
    if area_frac <= near_frac * 0.15:
        return "hand very small - move a little closer", (0, 200, 255)
    return None


def draw_hand(frame, landmarks_px, box_px):
    """Cyan hand box + landmark dots for the detector path."""
    l, t, r, b = box_px
    cv2.rectangle(frame, (l, t), (r, b), (255, 200, 0), 2)
    for x, y in landmarks_px:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 200, 0), -1)


def draw_overlay(frame, classes, probs, order, fps, header, crop_box, idle_msg, hint, footer):
    """Crop region + translucent panel + top-k bars (or an idle banner), plus a
    framing hint and a status footer."""
    h, w = frame.shape[:2]

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

    n_rows = 0 if idle_msg else len(order)
    panel_w, panel_h = 300, 34 + 26 * max(n_rows, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, header, (10, 22), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if idle_msg:
        cv2.putText(frame, idle_msg, (10, 52), FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        bar_x, bar_max = 130, panel_w - 130 - 12
        for i, idx in enumerate(order):
            y = 52 + i * 26
            p = float(probs[idx])
            color = (0, 230, 0) if i == 0 else (170, 170, 170)
            cv2.putText(frame, f"{classes[idx][:11]:<11}", (10, y), FONT, 0.5, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (bar_x, y - 11), (bar_x + int(bar_max * p), y - 1), color, -1)
            cv2.putText(frame, f"{p*100:4.0f}%", (bar_x + bar_max - 34, y),
                        FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if hint is not None:
        text, color = hint
        cv2.putText(frame, text, (10, h - 40), FONT, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(frame, footer, (10, h - 12), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def warn_mismatch(crop_mode: str, using_detector: bool, using_roi: bool):
    """One-line train/serve preprocessing sanity check (AD A.3)."""
    if crop_mode == "bbox" and not using_detector:
        src = "a centered ROI" if using_roi else "the whole frame"
        print(f"[warn] bbox-trained model but detector OFF -> feeding it {src}; "
              f"predictions will be unreliable (train/serve mismatch, AD A.3).")
    if crop_mode == "full_frame" and using_detector:
        print("[warn] full_frame-trained model but detector ON -> feeding it hand "
              "crops it never saw; use --no-detector for this checkpoint.")


def main() -> None:
    args = parse_args()

    lm = load_checkpoint(args.checkpoint, device=args.device)
    print("loaded", lm.describe())
    pre = Preprocessor(lm.size, lm.mean, lm.std)
    cropper = RoiCropper(frac=args.roi)

    # Crop source: default follows the checkpoint's training mode (AD-04). A bbox
    # model wants the detector; a full_frame model wants the raw frame.
    want_detector = (lm.crop_mode == "bbox") if args.detector is None else args.detector
    detector, why = try_make_detector(args, want_detector)
    use_detector = detector is not None
    if want_detector and not use_detector:
        print(f"[warn] detector requested but unavailable ({why}); falling back to ROI.")
    use_roi = (not use_detector) and (lm.crop_mode == "bbox")
    mirror = not args.no_mirror
    topk = min(args.topk, lm.num_classes)
    warn_mismatch(lm.crop_mode, use_detector, use_roi)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DSHOW: fast open on Windows
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    win = "FNAF gesture demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    ema: np.ndarray | None = None
    a = float(args.smooth)
    last, fps = time.time(), 0.0

    # Strategy-3 preview: the frame stands in for the screen, so the crosshair
    # shows exactly where play.py would put the real cursor. Sized lazily from
    # the first frame; no OS input is ever sent from the demo.
    mapper: CursorMapper | None = None
    # No-click class for the FSM preview: for a binary click model it's whichever
    # class isn't "fist" -- "palm" (AD-18) or "not_fist" (Strategy 3.2 / AD-21).
    # For legacy multi-class checkpoints we leave the "palm" default (the click
    # preview isn't meaningful there anyway).
    noclick_label = (next(c for c in lm.classes if c != "fist")
                     if "fist" in lm.classes and lm.num_classes == 2 else "palm")
    fsm = ClickFSM(k=args.fsm_k, conf_threshold=args.fsm_conf,
                   cooldown_s=args.click_cooldown, grace=args.fsm_grace,
                   palm_label=noclick_label, fist_label="fist")
    cursor_px: tuple[int, int] | None = None
    click_flash = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed"); break
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            crop_box = None       # region drawn + fed to the model
            landmarks = None
            model_in = None
            idle_msg = None
            hint = None
            source = "full"

            if mapper is None:
                mapper = CursorMapper(
                    screen_w=w, screen_h=h, box_w=args.box_w, box_h=args.box_h,
                    box_mode=args.box_mode, box_gain=args.box_gain,
                    alpha=args.cursor_alpha, deadzone=args.deadzone,
                )

            if use_detector:
                source = "detector"
                hb = detector.detect(frame)
                if hb is None:
                    idle_msg = "-- no hand --"
                    ema = None    # don't smooth across a detection gap
                    cursor_px = mapper.update(None)  # AD-19: freeze, never teleport
                else:
                    crop_box, landmarks = hb.box_px, hb.landmarks_px
                    model_in = detector.crop(frame, hb)
                    hint = framing_hint(hb.area_frac, args.near_frac)  # fix #1
                    cursor_px = mapper.update(hand_anchor_norm(hb.landmarks_norm, args.anchor),
                                              hand_scale=palm_span(hb.landmarks_norm))
            elif use_roi:
                source = "roi"
                crop_box = cropper.box(h, w)
                model_in = cropper.crop(frame)
            else:
                model_in = frame  # full_frame

            order, shown, raw_probs = [], None, None
            if model_in is not None and model_in.size > 0:
                probs = predict(lm.model, pre(model_in), args.device)
                raw_probs = probs
                if 0.0 < a < 1.0:
                    ema = probs if ema is None else a * ema + (1 - a) * probs
                    shown = ema
                else:
                    shown = probs
                order = np.argsort(shown)[::-1][:topk]

            # Click FSM preview (AD-20): fed the RAW top-1 (the FSM's K-frame
            # debounce is the smoothing; the display EMA would double-smooth).
            top_label, top_conf = None, 0.0
            if raw_probs is not None:
                i = int(raw_probs.argmax())
                top_label, top_conf = lm.classes[i], float(raw_probs[i])
            if fsm.update(top_label, top_conf):
                click_flash = 6
                print(f"[fsm] CLICK (virtual) at {cursor_px}")

            if landmarks is not None:
                draw_hand(frame, landmarks, crop_box)

            if use_detector and mapper is not None:  # cursor preview (AD-19)
                bx1, by1, bx2, by2 = mapper.box_px(w, h)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 90, 90), 1)
                if cursor_px is not None:
                    color = (0, 0, 255) if click_flash else (0, 255, 255)
                    cv2.drawMarker(frame, cursor_px, color, cv2.MARKER_CROSS, 22, 2)
                if click_flash:
                    cv2.putText(frame, "CLICK", (w // 2 - 40, 40), FONT, 1.0,
                                (0, 0, 255), 3, cv2.LINE_AA)
            click_flash = max(0, click_flash - 1)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            header = f"{lm.path.name}  [{lm.crop_mode} | src={source}]"
            pad_txt = f"pad {detector.pad:.2f}" if use_detector else "pad --"
            footer = f"{fps:4.1f}FPS  {pad_txt}  {fsm.describe()}  q m d r [ ]"
            draw_overlay(frame, lm.classes, shown, order, fps, header,
                         crop_box, idle_msg, hint, footer)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
                if mapper is not None:  # geometry flipped: forget cursor state
                    mapper.reset()
                fsm.reset()
                cursor_px = None
            if key == ord("d") and detector is not None:
                use_detector = not use_detector
                use_roi = (not use_detector) and (lm.crop_mode == "bbox")
                ema = None
                if mapper is not None:
                    mapper.reset()
                fsm.reset()
                cursor_px = None
                warn_mismatch(lm.crop_mode, use_detector, use_roi)
            if key == ord("r") and not use_detector:
                use_roi = not use_roi
                ema = None
            if use_detector and key in (ord("["), ord("]")):  # fix #2: live pad tune
                step = args.pad_step if key == ord("]") else -args.pad_step
                detector.pad = round(min(1.0, max(0.0, detector.pad + step)), 3)
                ema = None
                print(f"[pad] {detector.pad:.2f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
