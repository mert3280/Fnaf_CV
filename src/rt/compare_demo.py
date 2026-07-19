"""Side-by-side live comparison of several checkpoints on one webcam.

Why this exists: a webcam can only be opened by **one** process at a time
(Windows/DSHOW locks it), so you can't just launch three `webcam_demo`s. Even if
you could, each would grab a *different* frame -- not a fair comparison. This
runs every model in **one** process, on the **same** frame each tick, and tiles
their outputs so you can watch all three react to the identical hand at once.

Each model is fed according to its own `crop_mode` (read from the checkpoint):
  * bbox model       -> the shared MediaPipe detector crop (the real hand box)
  * full_frame model -> the whole frame (no crop)
One detector runs for the whole frame and is shared by every bbox model, so the
crop is identical across them (only the classifier differs).

Default: the three strategy models (baseline / bbox / palmfist). Override with
`--checkpoints a/best.pt b/best.pt ...` (2 or more).

    python -m src.rt.compare_demo
    python -m src.rt.compare_demo --checkpoints models/bbox_frozen_mnv3_large/best.pt \
                                                models/palmfist_frozen_mnv3_large/best.pt

Keys:  q/Esc quit | m mirror | d detector on/off

Diagnostic viewer only -- no game input is simulated (that's src/control/play.py).
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.rt.model_loader import LoadedModel, load_checkpoint
from src.rt.preprocess import Preprocessor, RoiCropper
from src.rt.webcam_demo import predict, try_make_detector

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 440  # each model's panel width in pixels

DEFAULT_CHECKPOINTS = [
    "models/baseline_mnv3_large/best.pt",
    "models/bbox_frozen_mnv3_large/best.pt",
    "models/palmfist_frozen_mnv3_large/best.pt",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS,
                    help="two or more best.pt paths to compare (default: the 3 strategy models)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--topk", type=int, default=4, help="predictions shown per panel")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="hand-bbox padding for the detector crop (match training: 0.15)")
    ap.add_argument("--roi", type=float, default=0.6,
                    help="fallback ROI square side (fraction of short side) if detector is off")
    ap.add_argument("--near-frac", type=float, default=0.4, help="unused hint threshold (kept for parity)")
    ap.add_argument("--hand-model", default=None, help="path to hand_landmarker.task")
    ap.add_argument("--detect-confidence", type=float, default=0.3)
    ap.add_argument("--presence-confidence", type=float, default=0.3)
    ap.add_argument("--tracking-confidence", type=float, default=0.3)
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the webcam")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="per-model EMA on probabilities, 0=off .. <1 (higher=steadier)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    return ap.parse_args()


def draw_panel(frame_bgr, lm: LoadedModel, probs, order, source: str,
               crop_box, idle_msg, header_color) -> np.ndarray:
    """Render one model's view: a scaled copy of the frame with its crop box,
    a name header, and its top-k probability bars (or an idle banner)."""
    h, w = frame_bgr.shape[:2]
    scale = PANEL_W / w
    disp = cv2.resize(frame_bgr, (PANEL_W, int(round(h * scale))))
    ph, pw = disp.shape[:2]

    if crop_box is not None:
        x1, y1, x2, y2 = (int(v * scale) for v in crop_box)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 0), 2)

    # Header band: model folder name + crop mode + test acc.
    name = lm.path.parent.name
    acc = "" if lm.test_acc is None else f"  test {lm.test_acc*100:.1f}%"
    cv2.rectangle(disp, (0, 0), (pw, 46), (0, 0, 0), -1)
    cv2.putText(disp, name[:32], (8, 19), FONT, 0.5, header_color, 1, cv2.LINE_AA)
    cv2.putText(disp, f"[{lm.crop_mode} | src={source}]{acc}", (8, 39),
                FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # Prediction panel at the bottom.
    if idle_msg:
        cv2.putText(disp, idle_msg, (8, ph - 14), FONT, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        return disp

    n = len(order)
    band_h = 22 + 24 * n
    y0 = ph - band_h
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, y0), (pw, ph), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)

    bar_x, bar_max = 118, pw - 118 - 46
    for i, idx in enumerate(order):
        y = y0 + 26 + i * 24
        p = float(probs[idx])
        color = (0, 230, 0) if i == 0 else (170, 170, 170)
        cv2.putText(disp, f"{lm.classes[idx][:11]:<11}", (8, y), FONT, 0.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(disp, (bar_x, y - 11), (bar_x + int(bar_max * p), y - 1), color, -1)
        cv2.putText(disp, f"{p*100:3.0f}%", (pw - 44, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("give at least two --checkpoints to compare")

    models = [load_checkpoint(p, device=args.device) for p in args.checkpoints]
    pres = [Preprocessor(m.size, m.mean, m.std) for m in models]
    emas: list[np.ndarray | None] = [None] * len(models)
    for m in models:
        print("loaded", m.describe())

    # One detector shared by every bbox model; skip it if nothing needs a crop.
    need_detector = any(m.crop_mode == "bbox" for m in models)
    detector, why = try_make_detector(args, need_detector)
    use_detector = detector is not None
    if need_detector and not use_detector:
        print(f"[warn] detector unavailable ({why}); bbox models fall back to a centered ROI.")
    cropper = RoiCropper(frac=args.roi)
    mirror = not args.no_mirror
    a = float(args.smooth)

    header_colors = [(0, 255, 255), (255, 200, 0), (180, 120, 255),
                     (0, 200, 120), (200, 200, 0), (120, 180, 255)]

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    win = "model comparison"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed"); break
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detect once for the whole frame; every bbox model reuses this crop.
            hb = detector.detect(frame) if use_detector else None
            det_crop = detector.crop(frame, hb) if (use_detector and hb is not None) else None
            det_box = hb.box_px if hb is not None else None
            roi_crop = cropper.crop(frame)
            roi_box = cropper.box(h, w)

            panels = []
            for i, (lm, pre) in enumerate(zip(models, pres)):
                idle_msg = None
                crop_box = None
                if lm.crop_mode == "bbox":
                    if use_detector:
                        source = "detector"
                        model_in, crop_box = det_crop, det_box
                        if model_in is None:
                            idle_msg = "-- no hand --"
                    else:
                        source, model_in, crop_box = "roi", roi_crop, roi_box
                else:  # full_frame
                    source, model_in = "full", frame

                order, shown = [], None
                if model_in is not None and model_in.size > 0:
                    probs = predict(lm.model, pre(model_in), args.device)
                    if 0.0 < a < 1.0:
                        emas[i] = probs if emas[i] is None else a * emas[i] + (1 - a) * probs
                        shown = emas[i]
                    else:
                        shown = probs
                    order = list(np.argsort(shown)[::-1][:min(args.topk, lm.num_classes)])
                else:
                    emas[i] = None  # don't smooth across a detection gap

                color = header_colors[i % len(header_colors)]
                panels.append(draw_panel(frame, lm, shown, order, source,
                                         crop_box, idle_msg, color))

            # Equalize heights (all frames share a size, so this is a no-op guard)
            hmax = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 0,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0)) for p in panels]
            grid = cv2.hconcat(panels)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            det_txt = "det ON" if use_detector else "det OFF"
            cv2.putText(grid, f"{fps:4.1f} FPS  {det_txt}  q quit  m mirror  d detector",
                        (8, grid.shape[0] - 8), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("d") and detector is not None:
                use_detector = not use_detector
                emas = [None] * len(models)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
