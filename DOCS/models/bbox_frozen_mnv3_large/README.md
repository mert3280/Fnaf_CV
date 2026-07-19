# Model — `bbox_frozen_mnv3_large`

**The current live model: a frozen-backbone, head-only MobileNetV3 trained on
*hand-cropped* HaGRID images.** Same recipe as the [baseline](../baseline_mnv3_large/README.md)
in every respect *except* the input — it sees the hand crop instead of the whole
frame. That single change took held-out test accuracy **53.0% → 91.7%**, and it is
the empirical basis for the two-stage detect-and-crop pipeline
([AD-04](../../build/architecture-and-decisions.md)). This is the checkpoint the
webcam demo loads by default.

> **Status:** current / active. Paired live with the MediaPipe hand cropper
> (`src/rt/detector.py`). Strategy history:
> [strategies/2-two-stage-mediapipe-crop](../../build/strategies/2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md).

---

## 1. What it is

| Property | Value |
|---|---|
| Backbone | `mobilenetv3_large_100` (timm, ImageNet-pretrained) — AD-07 |
| Head | fresh 8-class linear classifier (timm `num_classes=8`) |
| Freeze state | **backbone frozen**, head-only (AD-08 Stage A) |
| Input | **`bbox`** crop (+15% pad), 224 px, ImageNet norm |
| Augmentation | none (kept off so the crop is the *only* change vs. the baseline) |
| Classes | `like, dislike, fist, one, two_up, palm, ok, mute` (order = `label_idx`) |
| Config | [`configs/bbox_frozen.yaml`](../../../configs/bbox_frozen.yaml) + [`configs/data.yaml`](../../../configs/data.yaml) (`crop_mode: bbox`) |
| Checkpoint | `models/bbox_frozen_mnv3_large/best.pt` (**git-ignored**) |

Only difference from the baseline: `crop_mode: bbox`. Everything else (backbone,
seed, epochs, optimizer, split, freeze policy) is identical — a **controlled A/B**
so the accuracy delta is attributable to cropping alone.

---

## 2. How it was trained

Linear-probe fast path (frozen backbone + no aug ⇒ deterministic features cached
once, head trained on them; identical math, ~50× faster on CPU).

- AdamW, lr 1e-3, weight decay 1e-4, 40 epochs, batch 64, seed 42.
- Best-by-val-accuracy checkpoint saved (full model: frozen backbone + trained head).
- Shared 70/15/15 by-user split (AD-16): train 5919 / val 1273 / test 1402.

---

## 3. Results (honest, by-user test split)

Full per-class report + confusion matrix: **[results.md](results.md)** (regenerated
directly from the checkpoint).

| Metric | Baseline (full_frame) | **This model (bbox)** | Δ |
|---|---|---|---|
| Best val accuracy | 0.5695 | **0.9403** | +0.371 |
| **Held-out test accuracy** | 0.5300 | **0.9165** | **+0.387** |

**Per-class read.** Strong across the board (F1 0.82–0.97). The remaining
confusions are the *finger-count* pairs — `one`↔`two_up` (17+14 off-diagonal;
`one` weakest at F1 0.815) and `ok`→`palm` (12) — the same weak pairs as the
baseline, but far reduced. These are class-similarity errors, not input-quality
errors, so cropping can't fix them further; augmentation or unfreezing might.

> ⚠️ **These are offline numbers on HaGRID's own annotated crops.** Live, the crop
> comes from MediaPipe, which is not equally tight — on detected HaGRID stills
> classification drops to ~71%, and the live webcam test showed a distance
> sensitivity (accurate hand-near-body, worse hand-near-lens). The full analysis
> and fixes are in the strategy page:
> [strategies/2-two-stage-mediapipe-crop](../../build/strategies/2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md).

---

## 4. Reproduce / run

**Retrain** (`configs/data.yaml` already ships `crop_mode: bbox`):

```bash
python -m src.train --config configs/bbox_frozen.yaml
# writes models/bbox_frozen_mnv3_large/best.pt
```

**Regenerate this report** from the existing checkpoint (no retrain):

```bash
python -m src.eval --checkpoint models/bbox_frozen_mnv3_large/best.pt   # (planned src/eval.py)
```

**Run live** (bbox model ⇒ MediaPipe detector auto-enabled):

```bash
python -m src.rt.webcam_demo --checkpoint models/bbox_frozen_mnv3_large/best.pt
```

The demo reads `crop_mode` from the checkpoint and defaults the crop source to the
detector, so no flags are needed. Use `[` / `]` to tune the crop pad live.
