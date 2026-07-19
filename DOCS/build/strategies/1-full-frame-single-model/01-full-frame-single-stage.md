# Strategy 1 — Full-frame single-stage classification

**Status: retired** (kept as the baseline and the `full_frame` fallback in the
[AD-04 switching note](../architecture-and-decisions.md#ad-04--how-to-switch-between-full-frame-and-two-stage-crop)).
Superseded by [Strategy 2](02-two-stage-mediapipe-crop.md).

## The idea

The simplest thing that could work: take the whole webcam frame, resize/normalize
it, and hand it straight to the classifier. **One model, one pass, no detector.**

```
Webcam frame ──▶ resize + center-crop + normalize ──▶ CNN ──▶ gesture
```

## Why we tried it first

- Fewest moving parts — no second component, no extra dependency or latency.
- It's the honest baseline: establish what a frozen pretrained backbone can do on
  raw frames before adding machinery, so any later gain is attributable.

## How it was built

- Model: `mobilenetv3_large_100`, ImageNet-pretrained, **backbone frozen,
  head-only** (AD-08 Stage A), no augmentation.
- Data: HaGRID, `crop_mode: full_frame`, 8-class working subset, 70/15/15 by-user
  split.
- Full write-up + reproduce steps: [models/baseline_mnv3_large/](../models/baseline_mnv3_large/README.md).

## Results (honest, by-user test split)

| Metric | Value |
|---|---|
| Best val accuracy | 0.5695 |
| **Held-out test accuracy** | **0.5300** (random = 0.125) |
| Train accuracy | ≈ 0.85 |

Per-class report + confusion matrix: [models/baseline_mnv3_large/results.md](../models/baseline_mnv3_large/results.md).
Best class `mute` (F1 0.80); worst the finger-count pairs `one`↔`two_up`,
`ok`↔`palm`.

## Why it failed (and it's not what you'd guess)

Not overfitting, not a bad backbone — the **input** is the problem. HaGRID photos
are shot at conversational distance, so the hand is a small patch (median **1.8%**
of the frame, Viz-4). After resize→center-crop the gesture is a ~40×50 px blob in
a 224×224 tensor; the rest is walls, torso, background. A frozen ImageNet backbone
was trained to recognize whole scenes, not to find a small hand shape buried in
95%+ background — so most of its pretrained features are spent on the wrong pixels.
Visual proof: [data-preparation.md §4](../data-preparation.md#4-cropping-ad-04).

## What it taught us

The fix is to **change the input, not the model**: crop to the hand so it fills
the tensor. That insight *is* [Strategy 2](02-two-stage-mediapipe-crop.md). Same
backbone, same freeze policy, same everything — just crops instead of full frames —
took test accuracy from **53% → 92%**.

## When this strategy is still the right call

- As a **fallback** if MediaPipe's latency ever blows the ≥15 FPS budget, or on a
  machine where the detector won't install — run a `full_frame`-trained checkpoint
  with `--no-detector` (accepting the accuracy ceiling).
- As the **control arm** of the crop A/B — never delete it; it's the "nothing
  else changed" reference that makes the +39 pts attributable to cropping.
