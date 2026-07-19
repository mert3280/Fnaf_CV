# Strategies

Each **strategy** is a distinct approach to the same problem — *turn a webcam
frame into a gesture label the game can act on* — tried, measured, and kept as a
record. This folder is the running history: what we tried, why, what the numbers
and the live feel actually were, and what we changed as a result.

It complements, not duplicates, the other docs:
- **Why** a strategy was chosen → the decision records in [architecture-and-decisions.md](../architecture-and-decisions.md) (AD-04 is the pivot here).
- **What** a trained checkpoint is → [models/](../../models/README.md).
- **How** the data is prepped → [data-preparation.md](../data/data-preparation.md).

This folder answers a different question: *of the approaches we've tried, which
worked, how well, and where did each break?*

---

## Strategies so far

| # | Strategy | Pipeline | Test acc | Live verdict | Status |
|---|---|---|---|---|---|
| 1 | [Full-frame single-stage](1-full-frame-single-model/01-full-frame-single-stage.md) | frame → classify whole frame | **0.530** | too weak to use | **retired** (baseline / fallback) |
| 2 | [Two-stage detect-then-classify](2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md) | frame → **MediaPipe hand crop** → classify | **0.917**¹ | reads gestures well; distance-sensitive | superseded by 2.1 |
| 2.1 | [Two-stage, hardened](2-two-stage-mediapipe-crop/02.1-two-stage-robustness-fixes.md) | same pipeline **+ 6 robustness fixes** | — ² | closes the live distance gap | folded into 3 |
| 3 | [Cursor + click](3-cursor-and-click/03-cursor-and-click.md) | MediaPipe **tracks hand → moves cursor**; crop → **binary palm/fist** → click | **0.9815** ³ | works; cursor was distance-sensitive | folded into 3.1 |
| 3.1 | [Distance-invariant cursor](3-cursor-and-click/03.1-distance-invariant-cursor.md) | same + **size-invariant anchor calibration** (unclipped landmarks) | — ⁴ | position fixed; gain + clicks still bad live | folded into 3.1.1 |
| 3.1.1 | [Live robustness fixes](3-cursor-and-click/03.1.1-live-robustness-fixes.md) | same + **adaptive box (constant gain)**, **click grace window**, looser detection | — ⁴ | fixes slow-click + distance gain; re-test pending | **current** |

¹ 0.917 is on HaGRID's *own* annotated crops. Fed the **live MediaPipe** crop it
is ~0.71 on detected HaGRID stills — the train/serve gap is documented in the
strategy-2 page.
² 2.1 is an *incremental hardening* of 2, not a new model. Its runtime fixes are
live now; its training/data fixes are opt-in levers Ted retrains with, so there's
no single new headline number — see the 2.1 page.
³ Binary task: random = 50% (vs 12.5% for the 8-class rows), so compare per-class
F1 (palm 0.981 / fist 0.982), not raw accuracy across rows. On HaGRID's annotated
crops, like the others — the live MediaPipe-crop number is pending the live test.
Full report: [palmfist_frozen_mnv3_large](../../models/palmfist_frozen_mnv3_large/README.md).
⁴ 3.1 changes the cursor *calibration*, not the model — same checkpoint, same
number; its verdict is the live pointing feel across distances.

## The arc (1 → 2 → 2.1 → 3 → 3.1)

**1 → 2.** Strategy 1 classified the **whole webcam frame** and hit 53% test —
not a model failure but an *input* one: a HaGRID hand is <5% of the frame, so a
frozen backbone spent its receptive field on background. Strategy 2 fixed the
input with an off-the-shelf **MediaPipe hand crop** before the *same* CNN; test
accuracy went **53% → 92%** from cropping alone (the AD-04 evidence).

**2 → 2.1.** The live test showed strategy 2 reads gestures well but is
**distance-sensitive**: accurate with the hand near the body (matches HaGRID's
at-a-distance data), worse with the hand close to the lens (perspective/scale/focus
it never trained on). Strategy 2.1 keeps the pipeline and adds **six fixes** for
that gap — framing hint, live pad tuning, detection-confidence, perspective/blur
augmentation, progressive unfreezing, and a self-capture fine-tune tool.

**2.1 → 3 (the scope pivot, AD-17).** Strategy 3 keeps the entire two-stage
pipeline and 2.1's fixes, but changes what the outputs *mean*: MediaPipe's hand
position now **drives the mouse cursor** (absolute mapping), and the classifier
shrinks from 8 gestures to a **binary palm/fist** — palm = no click, fist =
click. One universal point-and-click interaction replaces the gesture→action
vocabulary and its Office/Camera mode controller, and the classifier's live
reliability problem collapses to the two most distinct poses in the set. The
gesture-vocabulary plan is preserved in [DOCS/legacy/](../../legacy/README.md).

**3 → 3.1.** The first live test showed the *cursor* was distance-sensitive:
the 3.0 anchor came from frame-clipped landmarks, so a close/large hand (parts
off-frame) had its anchor dragged toward the frame interior — same pointing
spot, different cursor position depending on hand size. 3.1 recalibrates the
anchor to a **pure position from the frame's top-left**, computed from
**unclipped** landmarks: hand/box size can no longer enter the mapping, and a
`--anchor palm|box` flag lets Ted A/B the anchor point live.

**3.1 → 3.1.1.** The 3.1 live test surfaced three faults: touchy detection
(strict 0.5 thresholds), clicks landing ~50% with **slow squeezes never
clicking** (the FSM hard-disarmed on the first ambiguous mid-squeeze frame —
structural, not tuning), and a residual distance error (position was fixed,
but **gain** still scaled with distance). 3.1.1 lowers detection defaults to
0.3, gives the FSM a **grace window** (ambiguity pauses instead of disarming;
safety properties preserved), and makes the control box **scale with hand
size** so the same arm motion moves the cursor the same amount at any
distance (`--box-mode fixed` reverts).
