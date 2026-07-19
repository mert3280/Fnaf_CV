# Strategy 2 — Two-stage detect-then-classify (MediaPipe crop → CNN)

**Status: superseded by [Strategy 2.1](02.1-two-stage-robustness-fixes.md)** — the
pipeline below is still the active pipeline; 2.1 is its hardening against the live
distance gap found in the test at the bottom of this page. Read this page for the
approach and results; read 2.1 for the fixes.

This is the two-stage approach ([AD-04](../../architecture-and-decisions.md#ad-04--two-stage-pipeline-detect-and-crop-the-hand-then-classify)).
It replaced [Strategy 1](../1-full-frame-single-model/01-full-frame-single-stage.md)
after the full-frame baseline showed the input — not the model — was the bottleneck.

## The idea

Split perception into two jobs and use the right tool for each:

```
Webcam frame ─▶ MediaPipe HandLandmarker ─▶ hand bbox ─▶ crop (+pad) ─▶ CNN ─▶ gesture
                (off-the-shelf detector)                 (== training geometry)   (our transfer-learned model)
```

1. **Stage 1 — detect (borrowed).** An off-the-shelf MediaPipe hand detector
   finds the hand and gives 21 landmarks; we take their bounding box. No training —
   Google already solved "where is the hand."
2. **Stage 2 — classify (ours).** The *same* transfer-learned CNN from Strategy 1
   classifies the **crop**. This is the ML deliverable ([AD-05](../../architecture-and-decisions.md#ad-05--transfer-learning-with-a-pretrained-cnn-not-landmark-only)) —
   MediaPipe is only the cropper, not the classifier.

The classifier's *input distribution* changed from "whole cluttered frame" to
"tight hand crop." That is the entire fix.

## Why this over the alternatives

- **vs. full-frame (Strategy 1):** cropping puts the hand at full resolution
  instead of a ~40 px blob in 95% background — the measured +39 pts below.
- **vs. a landmark-only classifier:** would hand the whole vision problem to
  Google and leave a trivial classifier — low learning value (AD-05).
- **vs. one YOLO-style detect+classify model:** one pass, but it swaps
  transfer-learned classification for an object-detection objective, superseding
  the course's learning goal (AD-04 rationale). Rejected unless goals change.

## How it's built

| Piece | File | Role |
|---|---|---|
| Detector | [`src/rt/detector.py`](../../../../src/rt/detector.py) | MediaPipe Tasks `HandLandmarker` (VIDEO mode) → `HandBox` (padded pixel box + landmarks). Returns `None` when no hand. |
| **Shared crop geometry** | [`src/data/dataset.py`](../../../../src/data/dataset.py) `padded_bbox_pixels()` | One function computes the padded/clamped box. **Training and the live cropper both call it**, so the runtime crop is byte-identical to training (AD A.3). Verified identical to the old inline formula over 1000 random cases. |
| Preprocess | [`src/rt/preprocess.py`](../../../../src/rt/preprocess.py) | Reuses training's exact `build_transforms(train=False, …)` — same resize/crop/normalize. |
| Classifier loader | [`src/rt/model_loader.py`](../../../../src/rt/model_loader.py) | Rebuilds any checkpoint from its own metadata; crop source auto-selected from the saved `crop_mode`. |
| Live UI | [`src/rt/webcam_demo.py`](../../../../src/rt/webcam_demo.py) | Webcam loop, detector crop, HUD (hand box, top-k bars, FPS), idle on no-hand, mismatch warning. |

- **Detector model bundle:** `models/mediapipe/hand_landmarker.task` (git-ignored;
  the detector prints the one-line download command if missing).
- **Env note:** MediaPipe 0.10.35 on Python 3.13 ships **only the Tasks API**
  (no legacy `mp.solutions.hands`) — hence `HandLandmarker`.

## Results

### The classifier itself (crop-trained) — strong
Same backbone / seed / epochs / split as the baseline, still frozen + no aug,
only the crop changed:

| Metric | Full-frame (Strat. 1) | **Two-stage crop (Strat. 2)** | Δ |
|---|---|---|---|
| Best val acc | 0.5695 | **0.9403** | +0.371 |
| **Held-out test acc** | 0.5300 | **0.9165** | **+0.387** |

The +39 pts came from cropping alone — the evidence base for AD-04. Full report:
[bbox results.md](../../../models/bbox_frozen_mnv3_large/results.md).

### The *live* crop (MediaPipe box) — the honest gap
That 0.917 uses HaGRID's **own annotated** box. Live, the box comes from
**MediaPipe** instead, and the two aren't identical. Measured on 240 labeled
HaGRID stills:

| Measure | Value | Meaning |
|---|---|---|
| Hand detected | **~48%** of frames | MediaPipe misses HaGRID's far/blurred/side hands |
| Classify \| detected (MediaPipe crop) | **~71%** | vs 91.7% on the annotated crop — a ~20 pt drop |
| End-to-end (detect **and** classify) | ~34% | on this *hard* still-image proxy |

**Read this as a pessimistic lower bound, not the live experience.** HaGRID stills
are a *harder* distribution for MediaPipe than the real use case (a hand held up to
a webcam). The live test below bears that out.

## Live findings (webcam test, 2026-07-13)

Ran `python -m src.rt.webcam_demo --checkpoint models/bbox_frozen_mnv3_large/best.pt`.
Clean run, detector auto-enabled, no mismatch warning — the two-stage path worked
end-to-end on a real camera.

**What worked:** it read gestures **well** overall. MediaPipe locked onto the hand
reliably; the classifier tracked the poses.

**The key observation — distance sensitivity:**
> Accuracy was **much higher with the hand held closer to the body** (farther from
> the camera) and **dropped when the hand was held out close to the webcam**.

**Why this happens (analysis).** This is a *train/serve domain gap* on top of the
crop-tightness gap:
- HaGRID photos are taken at **conversational distance** — the hand sits well back
  from the camera, at a natural size and perspective, body visible behind it. A
  hand held **near your body** reproduces that geometry, so the MediaPipe crop
  looks like a HaGRID crop → the classifier is on home turf.
- A hand held **close to a wide webcam lens** adds distortion the model never
  trained on: **perspective foreshortening** (fingers/palm splay, lens is close),
  a different apparent aspect ratio, and often **soft focus** (fixed-focus webcams
  focus at a distance). The crop is in-distribution *shape* but out-of-distribution
  *perspective/scale/focus* → confidence and accuracy fall.

So the live behavior is consistent with the ~20 pt crop gap measured above, plus a
distance/perspective component that only shows up on a live camera. It is **not** a
bug in the pipeline — it's the classifier honestly reporting it was trained on
at-a-distance hands.

## Potential fixes

> **These are now built.** Every fix below has been implemented as code in
> **[Strategy 2.1](02.1-two-stage-robustness-fixes.md)** — with the exact files,
> commands, and which are live-now vs. opt-in-retrain. This section is the
> original diagnosis; 2.1 is the implementation.

Ordered cheapest → most involved. **The modeling choices here are Ted's to make**
(freeze schedule, augmentation, data collection, tightness) — these are options
with trade-offs, not a decision.

1. **Usage / framing (free, immediate).** Hold the hand at a natural distance
   (roughly near the body / mid-reach), front-on to the camera. Matches training;
   biggest instant win. Could be stated as a demo guideline.
2. **Tune `--pad` (config lever).** MediaPipe's landmark box is tighter than
   HaGRID's annotated box; `--pad` (default 0.15, matching training) reconciles
   them. A quick 0.15 / 0.25 / 0.35 sweep on the live feed shows which framing the
   classifier likes. *Caveat:* changing live pad without retraining widens the
   train/serve gap — the principled version is to pick a pad and **retrain the crop
   at that pad** so both halves match.
3. **Perspective / scale / lighting augmentation (Phase 3, AD-09).** Add
   perspective-warp, stronger random-resized-crop, and brightness/blur jitter so
   the classifier learns to tolerate webcam-close hands. Directly targets the
   distance gap; no new data needed.
4. **Small self-captured fine-tune (Phase 5 / AD-04 open question).** A few hundred
   frames of *your* hands on *your* webcam at the distances you'll actually use,
   fine-tuned on top of the HaGRID model. The most direct close of the domain gap;
   this was already flagged as an open question in AD-04.
5. **Progressive unfreezing (AD-08 Stage B).** Unfreeze the top block(s) with a
   discriminative LR so the backbone can adapt features to hand crops, not just the
   head. More capacity to absorb the webcam distribution; watch for overfitting.
6. **Detection robustness.** If "-- no hand --" appears too often live, lower
   `min_hand_detection_confidence` in `detector.py`; if it grabs the wrong thing,
   raise it. Temporal smoothing (already in the demo; AD-12) hides brief dropouts.

## How to run

```bash
# current strategy — detector auto-enabled for the bbox checkpoint
python -m src.rt.webcam_demo --checkpoint models/bbox_frozen_mnv3_large/best.pt

# probe the tightness lever
python -m src.rt.webcam_demo --checkpoint models/bbox_frozen_mnv3_large/best.pt --pad 0.25

# contrast: Strategy 1 (full-frame, no detector)
python -m src.rt.webcam_demo --checkpoint models/baseline_mnv3_large/best.pt --no-detector
```

Keys: `q`/`Esc` quit · `m` mirror · `d` detector on/off · `r` ROI fallback.

## Open questions carried forward

- **Does MediaPipe's per-frame cost fit the ≥15 FPS budget?** (AD-04 / Phase-4
  Day-1 measurement — read the FPS in the HUD.)
- **Which fix(es) to pursue** for the distance gap — augmentation vs. self-capture
  vs. unfreezing — and in what order. Ted's call; measure on **val**, report on
  **test** once (AD-10).
- **Pad + tightness:** is there a single pad that matches MediaPipe's box to
  HaGRID's, or is a retrain-at-pad needed?
