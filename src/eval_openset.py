"""Open-set false-click evaluation for the binary click classifiers.

The honest question for Strategy 3.2 (AD-21) is NOT "what's the test accuracy" --
on a balanced binary task that's a near-useless headline. It's:

    when the hand does a gesture the model was NEVER trained on, does it stay
    OFF (no click), or does it fire a false `fist`?

This script runs one or more checkpoints over whole gesture folders and reports,
per gesture, how often each model predicts `fist` -- both as raw argmax and as
"would actually click" (argmax==fist AND confidence >= the FSM gate, AD-20). For a
NON-fist gesture, lower is better (fewer accidental clicks); for `fist` itself,
higher is better (real clicks retained).

`ok` is the key row: it is held out of BOTH models' training (palm/fist never saw
it; fist-vs-rest deliberately excluded it), so it's a fair unseen-pose A/B.

    python -m src.eval_openset \
        --checkpoints models/palmfist_frozen_mnv3_large/best.pt \
                      models/fistvsrest_frozen_mnv3_large/best.pt \
        --gestures fist palm one two_up like dislike mute ok
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.config import DataConfig
from src.data.dataset import HagridDataset
from src.data.hagrid_annotations import build_index
from src.data.transforms import build_transforms
from src.rt.model_loader import load_checkpoint


def index_gesture(cfg: DataConfig, gesture: str):
    """Index every downloaded image of one gesture across all sources, with its
    own hand bbox. Ungrouped (label == folder), so it works for any folder,
    trained-on or not."""
    frames = []
    for images, ann in cfg.sources:
        if not (Path(images) / gesture).is_dir():
            continue
        frames.append(build_index(images, ann, [gesture], {gesture: 0}))
    if not frames:
        raise FileNotFoundError(f"no downloaded images for gesture '{gesture}'")
    import pandas as pd
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="uuid")


@torch.no_grad()
def fist_stats(lm, index, pad: float, conf_gate: float, device: str):
    """Run the model over one gesture's crops -> (n, fist_rate, click_rate,
    mean_fist_prob). `fist_rate` = argmax is fist; `click_rate` = argmax is fist
    AND prob >= conf_gate (what the FSM would actually act on)."""
    fist_idx = lm.classes.index("fist")
    tf = build_transforms(False, lm.size, lm.mean, lm.std)
    ds = HagridDataset(index, tf, crop_mode=lm.crop_mode, bbox_pad=pad)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    n = fired = clicked = 0
    prob_sum = 0.0
    for x, _ in loader:
        probs = torch.softmax(lm.model(x.to(device)), dim=1).cpu()
        p_fist = probs[:, fist_idx]
        pred_fist = probs.argmax(1) == fist_idx
        n += len(x)
        fired += int(pred_fist.sum())
        clicked += int((pred_fist & (p_fist >= conf_gate)).sum())
        prob_sum += float(p_fist.sum())
    return n, fired / n, clicked / n, prob_sum / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one or more best.pt paths (binary click models)")
    ap.add_argument("--gestures", nargs="+",
                    default=["fist", "palm", "one", "two_up", "like", "dislike", "mute", "ok"],
                    help="gesture folders to probe (default: the downloaded set)")
    ap.add_argument("--data-config", default="configs/data_fistvsrest.yaml",
                    help="only its `sources:` are used, to locate the image folders")
    ap.add_argument("--conf-gate", type=float, default=0.70,
                    help="FSM confidence threshold: a fist below this wouldn't click (AD-20)")
    ap.add_argument("--pad", type=float, default=0.15, help="bbox padding (match training)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per gesture (0 = all); a seeded sample is plenty for a rate")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = DataConfig.from_yaml(args.data_config)
    models = [load_checkpoint(p, device=args.device) for p in args.checkpoints]
    names = [Path(p).parent.name for p in args.checkpoints]
    for m, nm in zip(models, names):
        print(f"loaded {nm}: classes={m.classes} crop={m.crop_mode} size={m.size}")
    print(f"\nconf-gate = {args.conf_gate} (a 'click' = argmax fist AND prob >= gate)\n")

    # Pre-index each gesture once (shared across models). Optionally cap to a
    # seeded sample -- a few hundred crops give a stable rate far faster on CPU.
    indices = {}
    for g in args.gestures:
        idx = index_gesture(cfg, g)
        if args.limit and len(idx) > args.limit:
            idx = idx.sample(n=args.limit, random_state=42).reset_index(drop=True)
        indices[g] = idx

    # Header
    col = 22
    head = f"{'gesture':<10}{'n':>6}   " + "".join(f"{nm[:col]:>{col+2}}" for nm in names)
    print(head)
    print(f"{'':<10}{'':>6}   " + "".join(f"{'fist% / click% / meanp':>{col+2}}" for _ in names))
    print("-" * len(head))
    for g in args.gestures:
        idx = indices[g]
        row = f"{g:<10}{len(idx):>6}   "
        goal = "  (want HIGH)" if g == "fist" else ""
        cells = []
        for m in models:
            n, fr, cr, pm = fist_stats(m, idx, args.pad, args.conf_gate, args.device)
            cells.append(f"{fr*100:5.1f} / {cr*100:5.1f} / {pm:4.2f}")
        print(row + "".join(f"{c:>{col+2}}" for c in cells) + goal)

    print("\nRead: for every NON-fist gesture, lower fist%/click% = fewer accidental "
          "clicks.\n`ok` is unseen by both models -- the fair open-set A/B. `fist` "
          "row is the control (clicks retained).")


if __name__ == "__main__":
    main()
