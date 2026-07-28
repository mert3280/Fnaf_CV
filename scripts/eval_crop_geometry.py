"""Measure the AD-A.3 **crop-source** contract: does the box the live detector
produces actually match the box the classifier trained on?

A.3 requires that "the live MediaPipe bbox must approximate the HaGRID training
bbox (same relative tightness and `bbox_pad`), or the classifier is fed a
distribution it never trained on." That clause has been asserted since Strategy 2
and flagged as unmeasured in every model record since. This script measures it.

Two independent passes, either runnable alone:

  --stage geometry  (fast, no model, no MediaPipe)
      HaGRID's annotation JSONs carry BOTH the annotated bbox and the 21
      landmarks, so the landmark-hull box the live detector *would* build is
      recoverable for every downloaded image. Reports the linear scale
      hull/annotation per gesture -- the "definition gap" -- over the full
      population, and the live `pad` that would equalize the two.

  --stage classifier  (slow: runs MediaPipe + the model over the test split)
      Feeds the SAME held-out images to a checkpoint twice -- once cropped from
      the annotated bbox (what it trained and was scored on), once from a real
      MediaPipe landmark hull (what it gets live) -- across a sweep of live
      `pad` values. Reports fist click-rate and false-click rate at the FSM's
      confidence gate, which is what the click actually depends on.

    python scripts/eval_crop_geometry.py --stage geometry
    python scripts/eval_crop_geometry.py --stage classifier \
        --checkpoints models/week4_tuned_fistvsrest/best.pt --pads 0.15 0.25 0.35 0.45

Neither stage decides anything: the live crop tightness is a preprocessing call
for the human (CLAUDE.md), and this only supplies the number it should be made on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import DataConfig  # noqa: E402
from src.data.dataset import _split, padded_bbox_pixels  # noqa: E402

DEFAULT_GESTURES = ["fist", "palm", "one", "two_up", "like", "dislike", "mute", "ok"]
TRAIN_PAD = 0.15  # what `configs/data_fistvsrest.yaml` crops the ANNOTATED box with


def equalizing_pad(lin_scale: float, train_pad: float = TRAIN_PAD) -> float:
    """The live `pad` on a landmark-hull box that reproduces the training crop.

    Training grows the annotated box to `ann * (1 + 2*train_pad)`. The hull is
    `lin_scale * ann`, so matching extents needs
    `hull * (1 + 2*p) == ann * (1 + 2*train_pad)`.
    """
    return ((1 + 2 * train_pad) / lin_scale - 1) / 2


# --------------------------------------------------------------------------
# Stage 1 -- geometry, straight out of the annotations (no model, no MediaPipe)
# --------------------------------------------------------------------------
def stage_geometry(cfg: DataConfig, gestures: list[str]) -> pd.DataFrame:
    rows = []
    for g in gestures:
        for _images, ann in cfg.sources:
            path = Path(ann) / f"{g}.json"
            if not path.exists():
                continue
            for uuid, rec in json.loads(path.read_text()).items():
                if g not in rec["labels"]:
                    continue
                i = rec["labels"].index(g)
                if i >= len(rec.get("landmarks", [])):
                    continue
                bw, bh = rec["bboxes"][i][2], rec["bboxes"][i][3]
                lm = np.asarray(rec["landmarks"][i], dtype=float)
                if lm.size == 0 or not np.isfinite(lm).all() or bw <= 0 or bh <= 0:
                    continue
                xs, ys = np.clip(lm[:, 0], 0, 1), np.clip(lm[:, 1], 0, 1)
                hw, hh = xs.max() - xs.min(), ys.max() - ys.min()
                if hw <= 0 or hh <= 0:
                    continue
                rows.append(dict(gesture=g, uuid=uuid,
                                 lin=float(np.sqrt((hw * hh) / (bw * bh)))))
    df = pd.DataFrame(rows).drop_duplicates(subset=["gesture", "uuid"])
    if df.empty:
        raise SystemExit("no annotations with landmarks found -- check --data-config sources")

    out = df.groupby("gesture").agg(n=("lin", "size"), lin_hull_over_ann=("lin", "median"))
    out["equalizing_live_pad"] = out["lin_hull_over_ann"].map(equalizing_pad)
    out = out.round(3).sort_values("lin_hull_over_ann")

    print("\n=== definition gap: landmark hull vs annotated bbox (HaGRID's own data) ===")
    print("lin < 1 => the hull the live detector builds is TIGHTER than the box")
    print(f"training cropped ({TRAIN_PAD:+.2f} pad on the annotated box)\n")
    print(out.to_string())
    med = float(df.lin.median())
    print(f"\noverall median hull/ann linear scale: {med:.3f}")
    print(f"single live pad that equalizes on average: {equalizing_pad(med):.2f}")
    print("\nThe spread across gestures is the part a single pad cannot fix: a fist's "
          "landmarks sit\ninside its own silhouette, an open palm's reach the fingertips.")
    return out


# --------------------------------------------------------------------------
# Stage 2 -- what the geometry does to the click decision
# --------------------------------------------------------------------------
def mediapipe_hulls(paths: dict[str, Path], model_path: Path, cache: Path | None):
    """{uuid: normalized COCO hull} from the real detector, cached to JSON."""
    if cache and cache.exists():
        hulls = json.loads(cache.read_text())
        if all(u in hulls for u in paths):
            print(f"[cache] {len(hulls)} MediaPipe hulls from {cache}")
            return hulls
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    from PIL import Image

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3))
    hulls = {}
    for n, (uuid, p) in enumerate(paths.items(), 1):
        arr = np.ascontiguousarray(np.array(Image.open(p).convert("RGB")))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        if not res.hand_landmarks:
            continue
        raw = np.array([[q.x, q.y] for q in res.hand_landmarks[0]])
        xs, ys = np.clip(raw[:, 0], 0, 1), np.clip(raw[:, 1], 0, 1)
        hulls[uuid] = [float(xs.min()), float(ys.min()),
                       float(xs.max() - xs.min()), float(ys.max() - ys.min())]
        if n % 50 == 0:
            print(f"  detected {n}/{len(paths)}", flush=True)
    det.close()
    print(f"[detect] {len(hulls)}/{len(paths)} images gave a hand")
    if cache:
        cache.write_text(json.dumps(hulls))
    return hulls


def stage_classifier(cfg: DataConfig, args) -> pd.DataFrame:
    import torch
    from PIL import Image

    from src.data.transforms import build_transforms
    from src.rt.model_loader import load_checkpoint

    index = _split(cfg).test
    if args.limit:
        index = pd.concat([g.head(args.limit)
                           for _, g in index.groupby("source_label", sort=False)])
    print(f"held-out test images: {len(index)}  {index['label'].value_counts().to_dict()}")

    hulls = mediapipe_hulls({r.uuid: Path(r.path) for r in index.itertuples()},
                            Path(args.hand_model), Path(args.cache) if args.cache else None)
    index = index[index["uuid"].isin(hulls)]

    # Decode each image once; every (model, pad) cell reuses the pixels.
    images = {r.uuid: Image.open(r.path).convert("RGB") for r in index.itertuples()}

    rows = []
    for ckpt in args.checkpoints:
        lm = load_checkpoint(ckpt, device=args.device)
        name = Path(ckpt).parent.name
        print(f"\n[model] {name}: {lm.describe()}")
        tf = build_transforms(False, lm.size, lm.mean, lm.std)
        fist_i = lm.classes.index("fist")

        for pad in list(args.pads) + ["annotated"]:
            p_fist, p_neg = [], []
            for r in index.itertuples():
                img = images[r.uuid]
                W, H = img.size
                box = r.bbox if pad == "annotated" else hulls[r.uuid]
                use_pad = cfg.input.bbox_pad if pad == "annotated" else pad
                crop = img.crop(padded_bbox_pixels(box, W, H, use_pad))
                with torch.no_grad():
                    prob = float(torch.softmax(
                        lm.model(tf(crop).unsqueeze(0).to(args.device)), 1)[0, fist_i])
                (p_fist if r.label == "fist" else p_neg).append(prob)
            f, n = np.array(p_fist), np.array(p_neg)
            rows.append(dict(
                model=name, crop=pad, n_fist=len(f), n_not_fist=len(n),
                fist_p50=round(float(np.median(f)), 3),
                fist_p10=round(float(np.quantile(f, 0.10)), 3),
                fist_click_pct=round(100 * float((f >= args.conf_gate).mean()), 1),
                false_click_pct=round(100 * float((n >= args.conf_gate).mean()), 1),
            ))
            print(f"  {str(pad):>9}  fist click {rows[-1]['fist_click_pct']:5.1f}%  "
                  f"p50 {rows[-1]['fist_p50']:.3f}  p10 {rows[-1]['fist_p10']:.3f}  "
                  f"false click {rows[-1]['false_click_pct']:5.1f}%", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n=== fist click-rate @ conf-gate {args.conf_gate} (want HIGH) ===")
    print(df.pivot(index="model", columns="crop", values="fist_click_pct").to_string())
    print("\n=== false-click rate on not_fist (want LOW) ===")
    print(df.pivot(index="model", columns="crop", values="false_click_pct").to_string())
    print("\n=== 10th-percentile p(fist) on real fists -- the marginal squeeze ===")
    print(df.pivot(index="model", columns="crop", values="fist_p10").to_string())
    print("\n'annotated' is the training/offline-report crop; the numeric columns are "
          "the live\nMediaPipe crop at that pad. The gap between them is the "
          "train/serve geometry gap.")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["geometry", "classifier", "both"], default="both")
    ap.add_argument("--data-config", default="configs/data_fistvsrest.yaml")
    ap.add_argument("--gestures", nargs="+", default=DEFAULT_GESTURES)
    ap.add_argument("--checkpoints", nargs="+",
                    default=["models/week4_tuned_fistvsrest/best.pt"])
    ap.add_argument("--pads", nargs="+", type=float, default=[0.15, 0.25, 0.35, 0.45])
    ap.add_argument("--conf-gate", type=float, default=0.70,
                    help="FSM confidence gate -- a fist below this does not click (AD-20)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap test images per source gesture (0 = all)")
    ap.add_argument("--hand-model", default="models/mediapipe/hand_landmarker.task")
    ap.add_argument("--cache", default="", help="JSON path to cache MediaPipe hulls in")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="", help="write the classifier stage's table as CSV")
    args = ap.parse_args()

    cfg = DataConfig.from_yaml(args.data_config)
    if args.stage in ("geometry", "both"):
        stage_geometry(cfg, args.gestures)
    if args.stage in ("classifier", "both"):
        df = stage_classifier(cfg, args)
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
