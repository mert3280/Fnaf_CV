# Strategies

Each **strategy** is a distinct approach to the same problem — *turn a webcam
frame into a gesture label the game can act on* — tried, measured, and kept as a
record. This folder is the running history: what we tried, why, what the numbers
and the live feel actually were, and what we changed as a result.

It complements, not duplicates, the other docs:
- **Why** a strategy was chosen → the decision records in [architecture-and-decisions.md](../architecture-and-decisions.md) (AD-04 is the pivot here).
- **What** a trained checkpoint is → [models/](../models/README.md).
- **How** the data is prepped → [data-preparation.md](../data-preparation.md).

This folder answers a different question: *of the approaches we've tried, which
worked, how well, and where did each break?*

---

## Strategies so far

| # | Strategy | Pipeline | Test acc | Live verdict | Status |
|---|---|---|---|---|---|
| 1 | [Full-frame single-stage](01-full-frame-single-stage.md) | frame → classify whole frame | **0.530** | too weak to use | **retired** (baseline / fallback) |
| 2 | [Two-stage detect-then-classify](02-two-stage-mediapipe-crop.md) | frame → **MediaPipe hand crop** → classify | **0.917**¹ | reads gestures well; distance-sensitive | **current** |

¹ 0.917 is on HaGRID's *own* annotated crops. Fed the **live MediaPipe** crop it
is ~0.71 on detected HaGRID stills — the train/serve gap is documented in the
strategy-2 page.

## The arc (1 → 2)

Strategy 1 classified the **whole webcam frame**. The Phase-2 baseline settled it:
53% test (random 12.5%) — not because the model or training was wrong, but
because the *input* was: a HaGRID hand occupies <5% of the frame, so a frozen
ImageNet backbone spent its receptive field on background.

Strategy 2 fixes the input. An off-the-shelf **MediaPipe hand detector** finds and
crops the hand first, then the *same* transfer-learned CNN classifies the crop.
Training the identical recipe on crops instead of full frames took test accuracy
**53% → 92%** — the +39 pts came from cropping alone. This is the two-stage
detect-then-classify pipeline ([AD-04](../architecture-and-decisions.md#ad-04--two-stage-pipeline-detect-and-crop-the-hand-then-classify)),
and it is the current approach.

Open thread carried into strategy 2: MediaPipe's live crop is not a perfect
stand-in for HaGRID's annotated box (tightness + a webcam-distance domain gap),
which is exactly what the live test surfaced. See
[02-two-stage-mediapipe-crop.md](02-two-stage-mediapipe-crop.md) §Live findings.
