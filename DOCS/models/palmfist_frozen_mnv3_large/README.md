# palmfist_frozen_mnv3_large — the binary click classifier (Strategy 3)

The first model of the **cursor-control pivot** ([AD-17/AD-18](../../build/architecture-and-decisions.md#scope-pivot--cursor-control-2026-07-14)):
`palm` = no click, `fist` = click. Stage 2 of the two-stage pipeline — MediaPipe
tracks the hand (cursor + crop), this model reads the crop and feeds the click
FSM ([Strategy 3](../../build/strategies/3-cursor-and-click/03-cursor-and-click.md)).

## Headline numbers (2026-07-15)

| Metric | Value |
|---|---|
| Best val accuracy | **0.9639** |
| **Held-out test accuracy** | **0.9815** (random = 0.50) |
| palm F1 / fist F1 (test) | 0.9812 / 0.9818 |
| Test errors | 6 of 325 (4 palm→fist, 2 fist→palm) |

Full per-class report + confusion matrix: [results.md](results.md).

## Recipe — deliberately identical to the 8-class bbox run

Same backbone, seed, epochs, freeze policy, and `crop_mode: bbox` as
[`bbox_frozen_mnv3_large`](../bbox_frozen_mnv3_large/README.md); the **only**
changes are the class list (`configs/data.yaml → classes: [palm, fist]`, AD-18)
and `num_classes: 2`. That keeps the numbers comparable across the pivot:

| | 8-class (`bbox_frozen`) | **binary (`palmfist_frozen`)** |
|---|---|---|
| Task | 8 gestures | palm vs fist |
| Random floor | 0.125 | 0.50 |
| Test acc | 0.917 | **0.9815** |

- **Config:** [`configs/palmfist_frozen.yaml`](../../../configs/palmfist_frozen.yaml)
  (snapshot saved next to the checkpoint as `config.snapshot.json`).
- **Data:** 2,179 images (palm 1,082 / fist 1,097), split 1,494 / 360 / 325
  (70/15/15 grouped by `user_id`, seed 42 — AD-16; leak assertions pass).
- **Training:** Stage A (AD-08) — backbone frozen, head-only, linear-probe on
  cached features; 40 epochs, AdamW, lr 1e-3, no augmentation.
- `label_idx`: `palm=0, fist=1` — fixed by config order, head is bound to it.

## Reading the curves (for Ted)

Val accuracy was **still climbing at epoch 40** (0.9528 @ ep20 → 0.9639 @ ep39-40)
with train at 0.998 — no overfitting signal in the frozen regime. Levers if more
is wanted (all Ted's calls, AD-08/AD-09): more epochs, `unfreeze_blocks`,
perspective/blur augmentation, or the self-capture fine-tune. Whether 0.98 test
is "good enough to play" is decided by the **live transition reliability**, not
this offline number.

## Honest caveats

- 0.9815 is on HaGRID's **annotated** crops. Live, the crop comes from
  MediaPipe, and Strategy 2 measured a real gap on that path for the 8-class
  model (~20 pts on stills). The binary task should shrink the gap, but it is
  **unmeasured until the live test** — especially at the arm-extended distances
  cursor driving uses.
- 4 of the 6 test errors are palm misread as fist — i.e. **phantom clicks** are
  the likelier failure direction. The click FSM's K-frame debounce + confidence
  gate (AD-20) exists exactly for this; watch it live.

## How to run

```bash
# HUD-only preview: virtual cursor + click FSM, no real input
python -m src.rt.webcam_demo --checkpoint models/palmfist_frozen_mnv3_large/best.pt

# full control loop, dry run (no OS input), then live
python -m src.control.play --checkpoint models/palmfist_frozen_mnv3_large/best.pt --dry-run
python -m src.control.play --checkpoint models/palmfist_frozen_mnv3_large/best.pt
```
