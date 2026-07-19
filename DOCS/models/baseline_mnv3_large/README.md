# Baseline model — `baseline_mnv3_large`

**The Phase-2 baseline: a frozen-backbone, head-only MobileNetV3 trained on
*full-frame* HaGRID images.** This is the project's first trained classifier and
the reference point every later model is measured against. It is deliberately the
*simplest* thing that could work — no augmentation, no unfreezing, whole-frame
input — so its weaknesses would be diagnostic rather than mysterious. They were:
this model is the reason the project adopted the two-stage detect-and-crop
pipeline ([AD-04](../../build/architecture-and-decisions.md#ad-04--two-stage-pipeline-detect-and-crop-the-hand-then-classify)).

> **Status:** superseded as the live model by the bbox-crop counterpart
> (`bbox_frozen_mnv3_large`, **91.7% test** vs this model's 53.0%). Kept as the
> honest baseline and as the `full_frame` fallback in the AD-04 switching note.
> Do **not** delete — it anchors the Phase-3 A/B.

---

## 1. What it is

| Property | Value |
|---|---|
| Backbone | `mobilenetv3_large_100` (timm, ImageNet-pretrained) — [AD-07](../../build/architecture-and-decisions.md#ad-07--mobilenetv3-backbone) |
| Head | fresh 8-class linear classifier (timm `num_classes=8`) |
| Freeze state | **backbone frozen**, head-only ([AD-08](../../build/architecture-and-decisions.md#ad-08--frozen-backbone-baseline-then-progressive-unfreezing) Stage A) |
| Input | **`full_frame`** (whole image), 224 px, ImageNet norm |
| Augmentation | none (baseline is un-augmented; AD-09 adds it in Phase 3) |
| Classes | `like, dislike, fist, one, two_up, palm, ok, mute` (order = `label_idx`) |
| Config | [`configs/baseline.yaml`](../../../configs/baseline.yaml) + [`configs/data.yaml`](../../../configs/data.yaml) with `crop_mode: full_frame` |
| Checkpoint | `models/baseline_mnv3_large/best.pt` (**git-ignored**) |

### Saved config snapshot (`config.snapshot.json`, verbatim)

```json
{
  "model":  { "backbone": "mobilenetv3_large_100", "num_classes": 8,
              "freeze_backbone": true, "pretrained": true },
  "train":  { "epochs": 40, "lr": 0.001, "weight_decay": 0.0001,
              "optimizer": "adamw", "batch_size": 64, "seed": 42,
              "augment_train": false, "precompute_features": true },
  "out_dir": "models/baseline_mnv3_large"
}
```

> **Note — `full_frame` vs the repo default.** `configs/data.yaml` now ships
> `crop_mode: bbox` (the AD-04 default). This baseline was trained with
> `crop_mode: full_frame`. To reproduce it exactly you must set
> `input.crop_mode: full_frame` in `data.yaml` (see §5).

---

## 2. How it was trained

**Linear-probe fast path.** Because the backbone is frozen and there's no
augmentation, the backbone's features are deterministic — so `src/train.py`
caches them in one pass and trains only the head on the cached tensors. This is
*mathematically identical* to training the frozen model end-to-end, ~50× faster
on CPU. (`train.precompute_features: true`.)

- **Optimizer:** AdamW, lr 1e-3, weight decay 1e-4, 40 epochs, batch 64, seed 42.
- **Selection:** best-by-val-accuracy checkpoint saved (the full model:
  frozen backbone + trained head).
- **Split:** the shared 70/15/15 by-user split ([AD-16](../../build/architecture-and-decisions.md#ad-16--701515-split-grouped-by-user_id-across-all-images)) —
  train 5919 / val 1273 / test 1402. Identical to every other run, so results
  are directly comparable.

---

## 3. Results (honest, by-user test split)

Full auto-generated report (per-class P/R/F1 + confusion matrix) is preserved
next to this file: **[results.md](results.md)**. Headline:

| Metric | Value | Notes |
|---|---|---|
| Best val accuracy | **0.5695** | model-selection metric |
| Held-out **test** accuracy | **0.5300** | random floor = 0.1250 (1/8) |
| Train accuracy | ≈ 0.85 | train/val gap ⇒ some overfitting, but not the main story |

**The main story is the *input*, not overfitting.** A frozen ImageNet backbone
was trained to recognize whole objects/scenes; on a full HaGRID frame the hand
occupies <5% of the pixels (median 1.8% — Viz-4), so most of the receptive field
is spent on background. See [data-preparation.md §4](../../build/data/data-preparation.md#4-cropping-ad-04)
for the side-by-side visual.

### Per-class read (from [results.md](results.md))
- **Best:** `mute` (F1 0.80) — visually distinct.
- **Weakest / most confused:** the finger-count pairs `one`↔`two_up` (F1 0.37 /
  0.44; 43+44 off-diagonal) and `ok`↔`palm` (F1 0.41 / 0.48). These same pairs
  remain the hardest even after cropping — a class-similarity signal, not just an
  input-quality one.

---

## 4. Why it was superseded (→ AD-04)

This baseline **motivated** the two-stage pipeline. Training the *identical*
recipe (same backbone/seed/epochs/split, still frozen, still no aug) on **bbox
crops** instead of full frames took test accuracy **53.0% → 91.7%** — a +38.7 pt
jump attributable to cropping alone. That controlled A/B is the evidence base for
AD-04. The bbox counterpart is now the live model:

- **This model** (`full_frame`, 53.0%) → live via the raw-frame fallback
  (`runtime.use_detector: false`).
- **`bbox_frozen_mnv3_large`** (`bbox`, 91.7%, [bbox results.md](../bbox_frozen_mnv3_large/results.md))
  → live via the **MediaPipe hand cropper** (`src/rt/detector.py`).

Preprocessing contract: a `full_frame` checkpoint must be fed **raw frames** live;
feeding it MediaPipe crops (or vice-versa) breaks the train/serve match
([A.3](../../build/architecture-and-decisions.md#a3-data-flow--contracts)).

---

## 5. Reproduce / run

**Retrain** (must force full-frame first — see the note in §1):

```bash
# set input.crop_mode: full_frame in configs/data.yaml, then:
python -m src.train --config configs/baseline.yaml
# writes models/baseline_mnv3_large/best.pt and regenerates DOCS/results.md
```

**Run live in the webcam demo** (full-frame model ⇒ detector off, raw frame):

```bash
python -m src.rt.webcam_demo --checkpoint models/baseline_mnv3_large/best.pt --no-detector
```

The demo reads `crop_mode` from the checkpoint and defaults the crop source to
match, so `--no-detector` is the natural pairing for this model.
