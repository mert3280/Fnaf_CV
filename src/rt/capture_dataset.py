"""Self-capture tool -- record your own hand crops for a personal fine-tune set.

Strategy-2.1 fix #4. The single most direct close of the live distance/domain gap
is a few hundred frames of *your* hands on *your* webcam at the distances you
actually use, fine-tuned on top of the HaGRID model. This tool collects them.

It runs the **same MediaPipe detector + pad** as the live demo, so every saved
sample is the exact crop the classifier will see at inference -- no train/serve
drift. Crops are written in an **ImageFolder** layout:

    DATA/selfcapture/<class>/<uuid>.jpg

which a small fine-tune loader can consume directly (the images are already hand
crops, so fine-tune them as `crop_mode: full_frame`). This tool only *collects*
data -- whether/how to fine-tune (epochs, LR, freeze) is Ted's modeling call.

    python -m src.rt.capture_dataset --classes like dislike fist one two_up palm ok mute

Keys:  1..8 pick class | SPACE capture one | c continuous on/off | m mirror | q quit
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_CLASSES = ["like", "dislike", "fist", "one", "two_up", "palm", "ok", "mute"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="DATA/selfcapture", help="output root (ImageFolder layout)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="class labels to collect")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pad", type=float, default=0.15, help="crop pad (match training / live)")
    ap.add_argument("--hand-model", default=None, help="path to hand_landmarker.task")
    ap.add_argument("--no-mirror", action="store_true")
    return ap.parse_args()


def counts(root: Path, classes) -> dict[str, int]:
    return {c: len(list((root / c).glob("*.jpg"))) if (root / c).exists() else 0 for c in classes}


def main() -> None:
    args = parse_args()
    from src.rt.detector import DEFAULT_MODEL, HandDetector

    classes = args.classes[:9]  # digit keys 1..9
    root = Path(args.out)
    for c in classes:
        (root / c).mkdir(parents=True, exist_ok=True)
    n = counts(root, classes)

    detector = HandDetector(args.hand_model or DEFAULT_MODEL, pad=args.pad)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    win = "self-capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    sel = 0                # selected class index
    mirror = not args.no_mirror
    continuous = False
    saved_flash = 0.0      # wall-clock time of the last save, for a brief flash

    print(f"capturing into {root}/  classes={classes}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed"); break
            if mirror:
                frame = cv2.flip(frame, 1)

            hb = detector.detect(frame)
            crop = detector.crop(frame, hb) if hb is not None else None
            if hb is not None:
                l, t, r, b = hb.box_px
                cv2.rectangle(frame, (l, t), (r, b), (255, 200, 0), 2)

            # save (SPACE once, or every frame in continuous mode) when a hand is present
            key = cv2.waitKey(1) & 0xFF
            want_save = (key == ord(" ")) or (continuous and hb is not None)
            if want_save and crop is not None and crop.size > 0:
                label = classes[sel]
                cv2.imwrite(str(root / label / f"{uuid.uuid4().hex}.jpg"), crop)
                n[label] += 1
                saved_flash = time.time()

            # HUD
            h, w = frame.shape[:2]
            cv2.putText(frame, f"class[{sel+1}]: {classes[sel]}   total {n[classes[sel]]}",
                        (10, 26), FONT, 0.7, (0, 230, 0), 2, cv2.LINE_AA)
            mode = "CONTINUOUS" if continuous else "single (SPACE)"
            hand = "hand OK" if hb is not None else "-- no hand --"
            cv2.putText(frame, f"{mode}   {hand}", (10, 52), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "1-%d class | SPACE save | c continuous | m mirror | q quit" % len(classes),
                        (10, h - 12), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if time.time() - saved_flash < 0.15:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 230, 0), 6)
            cv2.imshow(win, frame)

            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("c"):
                continuous = not continuous
            if ord("1") <= key <= ord(str(len(classes))):
                sel = key - ord("1")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("final counts:", counts(root, classes))


if __name__ == "__main__":
    main()
