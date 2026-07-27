"""Final evaluation of ONE registered/tuned checkpoint on the held-out test split.

This is the AD-10 "report the test number once" step, plus the two things the
headline accuracy does not tell you about a *click trigger*:

  1. **Click-gate sweep.** The FSM only clicks when `p(fist) >= gate` (AD-20), so
     the operating point -- not the argmax -- is what the player feels. For every
     candidate gate we report clicks retained (fist recall) vs false clicks
     (not_fist scored as a click). That curve is how the gate gets chosen.
  2. **Sample predictions with confidences** on images the model never trained
     on, including its actual mistakes and an *unseen gesture class* (`ok`), which
     is the open-set case the negative class was designed for (AD-21).

Writes metrics JSON + figures next to the report.

    python -m src.eval_final --checkpoint models/week4_tuned_fistvsrest/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.config import DataConfig
from src.data.dataset import HagridDataset, build_datasets, padded_bbox_pixels
from src.data.transforms import build_transforms
from src.eval_openset import index_gesture
from src.plotting import PALETTE, apply_style, save, seq_color, text_on
from src.rt.model_loader import load_checkpoint

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# inference over a split
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_split(lm, dataset, device: str, batch_size: int = 64):
    """-> (y_true, y_pred, probs) for a HagridDataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(lm.model(x.to(device)), dim=1).cpu())
        ys.append(y)
    probs = torch.cat(ps).numpy()
    return torch.cat(ys).numpy(), probs.argmax(1), probs


def gate_sweep(y_true, probs, click_idx: int, gates) -> list[dict]:
    """At each confidence gate: clicks retained on the click class, false clicks
    on the no-click class. Both in the units the player experiences."""
    p_click = probs[:, click_idx]
    is_click_class = y_true == click_idx
    rows = []
    for g in gates:
        fires = (probs.argmax(1) == click_idx) & (p_click >= g)
        rows.append({
            "gate": round(float(g), 2),
            "clicks_retained": round(float(fires[is_click_class].mean()), 4),
            "false_click_rate": round(float(fires[~is_click_class].mean()), 4),
            "missed_clicks": int((~fires[is_click_class]).sum()),
            "false_clicks": int(fires[~is_click_class].sum()),
        })
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_confusion(cm, classes, out: Path, title: str):
    import matplotlib.pyplot as plt

    n = len(classes)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    vmax = cm.max()
    for i in range(n):
        for j in range(n):
            c = seq_color(cm[i, j], vmax)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=c,
                                       edgecolor=PALETTE["surface"], linewidth=2))
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color=text_on(c), fontsize=13, fontweight="bold")
    ax.set_xticks(range(n), classes)
    ax.set_yticks(range(n), classes)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return save(fig, out)


def fig_gate_sweep(rows, out: Path, chosen: float):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    gates = [r["gate"] for r in rows]
    retained = [r["clicks_retained"] * 100 for r in rows]
    false_r = [r["false_click_rate"] * 100 for r in rows]
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.plot(gates, retained, color=PALETTE["series"][0], marker="o", markersize=4,
            label="true clicks retained (fist recall)")
    ax.plot(gates, false_r, color=PALETTE["series"][1], marker="s", markersize=4,
            label="false clicks (not_fist fires)")
    ax.axvline(chosen, color=PALETTE["ink_muted"], linestyle="--", linewidth=1)
    ax.annotate(f"FSM gate {chosen:g}", (chosen, 55), xytext=(4, 0),
                textcoords="offset points", color=PALETTE["ink_secondary"], fontsize=8)
    # direct-label the ends so identity never rests on color alone
    ax.annotate(f"{retained[-1]:.0f}%", (gates[-1], retained[-1]), xytext=(4, 0),
                textcoords="offset points", color=PALETTE["ink_secondary"], fontsize=8)
    ax.annotate(f"{false_r[-1]:.0f}%", (gates[-1], false_r[-1]), xytext=(4, 0),
                textcoords="offset points", color=PALETTE["ink_secondary"], fontsize=8)
    ax.set_xlabel("click confidence gate  p(fist) >=")
    ax.set_ylabel("% of test images")
    ax.set_ylim(-3, 103)
    ax.set_title("Click operating point on the held-out test split", loc="left")
    ax.legend(loc="center left")
    return save(fig, out)


def fig_samples(samples, out: Path, title: str):
    """Grid of real crops with predicted class + confidence; errors marked."""
    import matplotlib.pyplot as plt

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.42 * rows))
    for ax, s in zip(np.array(axes).ravel(), samples):
        ax.imshow(s["image"])
        ok = s.get("correct")
        color = (PALETTE["ink_secondary"] if ok is None
                 else (PALETTE["good"] if ok else PALETTE["critical"]))
        mark = "" if ok is None else ("OK  " if ok else "MISS  ")
        ax.set_title(f"{mark}{s['pred']}  {s['conf']:.3f}", fontsize=8, color=color,
                     loc="left", pad=3, fontweight="bold")
        ax.set_xlabel(s["caption"], fontsize=7.5, color=PALETTE["ink_muted"])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(1.6)
    for ax in np.array(axes).ravel()[len(samples):]:
        ax.axis("off")
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, fontweight="bold",
                 color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, out)


# --------------------------------------------------------------------------- #
# sample collection
# --------------------------------------------------------------------------- #
def crop_pil(row, pad: float):
    from PIL import Image

    img = Image.open(row["path"]).convert("RGB")
    if row["bbox"] is None:
        return img
    w, h = img.size
    return img.crop(padded_bbox_pixels(row["bbox"], w, h, pad))


def collect_test_samples(lm, index, y_true, y_pred, probs, pad, n_correct=6, n_wrong=2):
    """A few correct + (up to) a few genuine mistakes, with real confidences."""
    wrong = np.flatnonzero(y_true != y_pred)[:n_wrong]
    correct = np.flatnonzero(y_true == y_pred)
    rng = np.random.default_rng(42)
    picks = list(rng.choice(correct, size=min(n_correct, len(correct)), replace=False))
    picks += list(wrong)
    out = []
    for i in picks:
        row = index.iloc[int(i)]
        out.append({
            "image": crop_pil(row, pad),
            "pred": lm.classes[int(y_pred[i])],
            "conf": float(probs[i].max()),
            "correct": bool(y_true[i] == y_pred[i]),
            "caption": f"true {lm.classes[int(y_true[i])]} · {row['source_label']}",
        })
    return out


def failure_modes(index, y_true, y_pred, probs, pad: float) -> dict:
    """Are the test errors *systematic*? Compares crop resolution, brightness, and
    the model's own confidence on the images it got wrong vs. right.

    Motivation: the sample-prediction figure shows a heavily-pixelated crop and a
    dark/backlit one among the misses, and both are conditions the live webcam
    reproduces (a hand far from the lens is a small crop; a dim room is a dark
    one). This exists to check whether that visual impression survives contact with
    all the errors -- for the current model it does NOT (see the Week-4 report
    section 2: resolution and luma are indistinguishable between errors and correct
    predictions; only confidence separates them). Keep running it after retrains:
    the answer is a property of the model, not a fact about the dataset.
    """
    from PIL import Image

    stats = {"wrong": [], "right": []}
    for i in range(len(y_true)):
        row = index.iloc[i]
        img = Image.open(row["path"]).convert("L")
        w, h = img.size
        if row["bbox"] is not None:
            img = img.crop(padded_bbox_pixels(row["bbox"], w, h, pad))
        side = min(img.size)          # the crop's short side, in source pixels
        arr = np.asarray(img, dtype=np.float32)
        bucket = "right" if y_true[i] == y_pred[i] else "wrong"
        stats[bucket].append({"short_side_px": side, "mean_luma": float(arr.mean()),
                              "confidence": float(probs[i].max())})

    def agg(rows):
        if not rows:
            # An empty bucket is a *result* (e.g. zero test errors), so say so
            # explicitly rather than emitting {} for a reader to interpret.
            return {"n": 0, "note": "no images in this bucket"}
        return {
            "n": len(rows),
            "median_short_side_px": round(float(np.median([r["short_side_px"] for r in rows])), 1),
            "pct_under_224px": round(100.0 * float(np.mean(
                [r["short_side_px"] < 224 for r in rows])), 1),
            "median_mean_luma": round(float(np.median([r["mean_luma"] for r in rows])), 1),
            "median_confidence": round(float(np.median([r["confidence"] for r in rows])), 4),
        }

    out = {"errors": agg(stats["wrong"]), "correct": agg(stats["right"])}
    print("\nfailure-mode analysis (test errors vs correct):")
    for k, v in out.items():
        print(f"  {k:<9} " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
    return out


@torch.no_grad()
def collect_openset_samples(lm, cfg, gesture: str, device: str, k: int = 4):
    """Unseen-gesture crops (never in ANY split) with what the model says."""
    idx = index_gesture(cfg, gesture).sample(n=k, random_state=7).reset_index(drop=True)
    tf = build_transforms(False, lm.size, lm.mean, lm.std)
    ds = HagridDataset(idx, tf, crop_mode=lm.crop_mode, bbox_pad=cfg.input.bbox_pad)
    probs = torch.softmax(lm.model(torch.stack([ds[i][0] for i in range(len(ds))]).to(device)),
                          dim=1).cpu().numpy()
    click_idx = lm.classes.index("fist") if "fist" in lm.classes else 1
    out = []
    for i in range(len(idx)):
        p = probs[i]
        out.append({
            "image": crop_pil(idx.iloc[i], cfg.input.bbox_pad),
            "pred": lm.classes[int(p.argmax())],
            "conf": float(p.max()),
            "correct": None,  # no ground-truth label in this label space
            "caption": f"unseen '{gesture}' · p(click) {p[click_idx]:.3f}",
        })
    return out, probs[:, click_idx]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/week4_tuned_fistvsrest/best.pt")
    ap.add_argument("--data-config", default="configs/data_fistvsrest.yaml")
    ap.add_argument("--out-dir", default="DOCS/class-related/week4/figures")
    ap.add_argument("--metrics-out", default="DOCS/class-related/week4/final-metrics.json")
    ap.add_argument("--openset-gesture", default="ok",
                    help="a gesture held out of ALL training (open-set probe)")
    ap.add_argument("--click-gate", type=float, default=0.70, help="AD-20 FSM gate")
    ap.add_argument("--label", default="week4-tuned", help="tag used in figure titles")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    apply_style()
    out_dir = REPO / args.out_dir
    cfg = DataConfig.from_yaml(REPO / args.data_config)
    lm = load_checkpoint(REPO / args.checkpoint, device=args.device)
    print(f"model: {lm.describe()}")

    datasets, splits, _ = build_datasets(cfg, backbone=lm.backbone,
                                         write_split_manifest=False, augment_train=False)
    y_true, y_pred, probs = predict_split(lm, datasets["test"], args.device)
    classes = lm.classes
    click_idx = classes.index("fist") if "fist" in classes else len(classes) - 1

    acc = float((y_true == y_pred).mean())
    rep = classification_report(y_true, y_pred, target_names=classes, digits=4,
                                zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print(f"\nTEST accuracy {acc:.4f}  (n={len(y_true)}, random={1/len(classes):.3f})")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4,
                                zero_division=0))

    gates = np.round(np.arange(0.50, 0.96, 0.05), 2)
    sweep = gate_sweep(y_true, probs, click_idx, gates)
    print("gate sweep (clicks retained / false-click rate):")
    for r in sweep:
        print(f"  {r['gate']:.2f}  {r['clicks_retained']*100:5.1f}%  "
              f"{r['false_click_rate']*100:5.1f}%")

    # --- figures ---------------------------------------------------------- #
    print("\nfigures:")
    fig_confusion(cm, classes, out_dir / "01_confusion_matrix.png",
                  f"Test confusion matrix — {args.label}")
    fig_gate_sweep(sweep, out_dir / "02_click_gate_sweep.png", args.click_gate)
    samples = collect_test_samples(lm, splits.test, y_true, y_pred, probs,
                                   cfg.input.bbox_pad)
    fig_samples(samples, out_dir / "03_sample_predictions.png",
                "Held-out test crops — prediction + confidence (MISS = model error)")
    modes = failure_modes(splits.test, y_true, y_pred, probs, cfg.input.bbox_pad)
    os_samples, os_pclick = collect_openset_samples(lm, cfg, args.openset_gesture,
                                                    args.device)
    fig_samples(os_samples, out_dir / "04_openset_predictions.png",
                f"Unseen gesture '{args.openset_gesture}' — never in any split "
                f"(want: no click)")

    metrics = {
        "checkpoint": args.checkpoint,
        "label": args.label,
        "classes": classes,
        "n_test": int(len(y_true)),
        "test_accuracy": round(acc, 4),
        "random_baseline": round(1 / len(classes), 4),
        "val_accuracy_from_checkpoint": lm.val_acc,
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
        "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
        "per_class": {c: {k: round(v, 4) for k, v in rep[c].items()} for c in classes},
        "confusion_matrix": {"labels": classes, "rows_are_true": cm.tolist()},
        "click_gate_sweep": sweep,
        "click_gate_default": args.click_gate,
        "failure_modes": modes,
        "openset_probe": {
            "gesture": args.openset_gesture,
            "n": len(os_pclick),
            "mean_p_click": round(float(np.mean(os_pclick)), 4),
            "max_p_click": round(float(np.max(os_pclick)), 4),
            "note": "sampled crops for the figure, not the full-folder rate "
                    "(use src/eval_openset.py for that)",
        },
    }
    path = REPO / args.metrics_out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
