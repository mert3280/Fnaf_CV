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

The old fixed centered-ROI box remains as a manual fallback (`r`) for when the
detector is unavailable.

    python -m src.rt.webcam_demo --checkpoint models/bbox_frozen_mnv3_large/best.pt

Keys:  q/Esc quit | m mirror | d detector on/off | r ROI-fallback on/off

Note: diagnostic viewer only -- no game input is simulated here (that lands with
the control loop in Phase 4/5), so there is nothing to kill-switch.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

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
    ap.add_argument("--hand-model", default=None,
                    help="path to hand_landmarker.task (default models/mediapipe/...)")
    ap.add_argument("--detector", dest="detector", action="store_true", default=None,
                    help="force the MediaPipe detector on")
    ap.add_argument("--no-detector", dest="detector", action="store_false",
                    help="force the detector off (use ROI / full frame)")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="EMA factor on probabilities, 0=off .. <1 (higher=steadier)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
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
        return HandDetector(path, pad=args.pad), "ok"
    except Exception as e:  # missing mediapipe or model bundle -> graceful fallback
        return None, f"{type(e).__name__}: {e}"


def draw_hand(frame, landmarks_px, box_px):
    """Cyan hand box + landmark dots for the detector path."""
    l, t, r, b = box_px
    cv2.rectangle(frame, (l, t), (r, b), (255, 200, 0), 2)
    for x, y in landmarks_px:
        cv2.circle(frame, (int(x), int(y)), 2, (255, 200, 0), -1)


def draw_overlay(frame, classes, probs, order, fps, header, crop_box, idle_msg):
    """Draw the crop region, a translucent panel, and the top-k probability bars
    (or an idle banner when there's nothing to classify)."""
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

    cv2.putText(frame, f"{fps:4.1f} FPS   q=quit m=mirror d=detector r=roi",
                (10, h - 12), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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
            source = "full"

            if use_detector:
                source = "detector"
                hb = detector.detect(frame)
                if hb is None:
                    idle_msg = "-- no hand --"
                    ema = None    # don't smooth across a detection gap
                else:
                    crop_box, landmarks = hb.box_px, hb.landmarks_px
                    model_in = detector.crop(frame, hb)
            elif use_roi:
                source = "roi"
                crop_box = cropper.box(h, w)
                model_in = cropper.crop(frame)
            else:
                model_in = frame  # full_frame

            order, shown = [], None
            if model_in is not None and model_in.size > 0:
                probs = predict(lm.model, pre(model_in), args.device)
                if 0.0 < a < 1.0:
                    ema = probs if ema is None else a * ema + (1 - a) * probs
                    shown = ema
                else:
                    shown = probs
                order = np.argsort(shown)[::-1][:topk]

            if landmarks is not None:
                draw_hand(frame, landmarks, crop_box)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            header = f"{lm.path.name}  [{lm.crop_mode} | src={source}]"
            draw_overlay(frame, lm.classes, shown, order, fps, header, crop_box, idle_msg)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("d") and detector is not None:
                use_detector = not use_detector
                use_roi = (not use_detector) and (lm.crop_mode == "bbox")
                ema = None
                warn_mismatch(lm.crop_mode, use_detector, use_roi)
            if key == ord("r") and not use_detector:
                use_roi = not use_roi
                ema = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
