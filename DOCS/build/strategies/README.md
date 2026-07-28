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
| 3.1.1 | [Live robustness fixes](3-cursor-and-click/03.1.1-live-robustness-fixes.md) | same + **adaptive box (constant gain)**, **click grace window**, looser detection | — ⁴ | fixes slow-click + distance gain; re-test pending | folded into 3.2 |
| 3.2 | [Fist vs. not-fist](3-cursor-and-click/03.2-fist-vs-rest.md) | same pipeline, classifier retrained so **negative class = `not_fist`** (6 diverse gestures) → unknown poses read as no-click | 0.9358 ⁵ | slashes false clicks (pointing hand 76%→9%; unseen `ok` 6.8%→0.8%); live re-test pending | **current** |
| 3.2.1 | [Snappier click FSM](3-cursor-and-click/03.2.1-snappier-fsm.md) | **same 3.2 model**, click FSM retuned: conf 0.70→**0.80**, K 3→**2** (stricter gate, faster confirm) | — ⁶ | snappier clicks; false-click cost meant to wash (0.80 gate offsets K=2); live A/B pending | tuning variant |
| 3.2.2 | [Tuned champion checkpoint](3-cursor-and-click/03.2.2-tuned-champion-checkpoint.md) | same pipeline, **retrained checkpoint**: head-init fix (AD-22) + TPE search + 2-block unfreeze; registered in MLflow (AD-24), orchestrated by a task DAG (AD-23) | **1.0000** ⁷ | offline benchmark saturated; open-set `ok` false-click rate unchanged (0.8%→0.8%); live A/B pending | tuning variant |
| 3.2.3 | [Latency + click delivery](3-cursor-and-click/03.2.3-latency-and-click-delivery.md) | same model & FSM values, **runtime re-engineered**: ONNX Runtime for stage 2, threaded capture, capped torch threads, and a click with real hold time | — ⁸ | **21.0 → 28.3 FPS** on AC (camera-capped), −67 ms input lag, 0 → 60 ms click hold; **plug the laptop in** — on battery it's 9–11 FPS regardless; live test pending | **current runtime** |
| 3.2.4 | [Crop geometry + live `pad`](3-cursor-and-click/03.2.4-crop-geometry-and-pad.md) | same model, same FSM, same runtime — **measures the crop the classifier actually gets live** and recalibrates `--pad` default 0.15 → **0.35** (accepted, AD-25) | — ⁹ | live crop was **1.3× tighter** than training's, worst on `fist`; at the old `pad 0.15` a real fist's p10 confidence was **0.464** (under both gates), at the new default **0.35** it's **0.932** and fist click-rate 81.5 → 94.4% with false clicks 0.7 → 0.0%; live A/B against the real game still pending | **measurement + accepted default change** |

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
⁵ 3.2 is a *retrain*, not a pipeline change. Balanced binary ⇒ raw accuracy
(0.9358) is the wrong headline (random = 50%, and it's lower than palm/fist's
0.9815 *by design* — a harder negative). The metric that counts is the
**open-set false-click rate**: on non-fist poses the palm/fist model false-clicks
41–81% of the time (a *pointing* hand 76%), fist-vs-rest 0–9%; on the unseen `ok`
gesture 6.8% → **0.8%**. Cost: `fist` click-rate 99.2% → 96.0%. Full A/B in the
[model record](../../models/fistvsrest_frozen_mnv3_large/README.md). Live
re-test at arm's length still pending.
⁶ 3.2.1 is a *tuning* variant, not a retrain — the 3.2 checkpoint runs unchanged;
only two FSM knobs move (`--fsm-conf` 0.70→0.80, `--fsm-k` 3→2). The two shifts
pull opposite ways on false clicks by design (a stricter per-frame gate offsets a
looser confirm), so there's no new offline headline — the decision is the live
3.2-vs-3.2.1 feel. Defined in [`src/control/strategies.py`](../../../src/control/strategies.py);
selectable via `play.py --strategy 3.2.1` and the dashboard's Click-strategy picker.
⁷ 3.2.2 is a *retrain* of 3.2, not a reframe — same `not_fist`/`fist` task, but a
fixed head init (AD-22) dominates the gain (+5.19 of the total +7.79 val points),
with tuning and a 2-block unfreeze on top. 327/327 test means the offline
benchmark is **saturated**, not that the model is flawless: it's independently
verified as leakage-free, but the open-set `ok` false-click rate — the metric 3.2
exists to move — didn't budge (0.8%→0.8%). Orthogonal to 3.2.1's FSM knobs; either
preset can point at this checkpoint. Registered as `fnaf-click-classifier` v1
@`champion` in MLflow (AD-24); full record in
[week4_tuned_fistvsrest](../../models/week4_tuned_fistvsrest/README.md).
⁸ 3.2.3 is neither a retrain nor a tuning variant — it changes **no model and no
FSM value**, so it has no offline number by construction. Its metrics are the
frame budget and click delivery, measured end-to-end on the real loop by
[`scripts/bench_pipeline.py`](../../../scripts/bench_pipeline.py) (2026-07-27,
HUD on, medians over 8 s/config): **45.7 ms → 32.6 ms per frame, 21.0 → 28.3
FPS**, which is the 30 FPS camera's ceiling. The classifier stage carries it
(20.7 → **6.2 ms** on the already-exported, parity-checked `model.onnx`);
capping torch's thread pool at 4 is worth another 14.7 → 21.0 FPS on the torch
path, because a 16-thread pool fighting MediaPipe's own threads also inflates
*preprocessing* from 2.6 ms to 138+ ms. Two corrections to the pre-measurement
guesses are recorded in the page: threaded capture turned out to be a **latency**
fix (a synchronous read returns in 0.2 ms — from a **2-frame, ~67 ms** driver
backlog, i.e. a stale hand), not a throughput one; and `pydirectinput.click()`
really does send button-down and button-up in a **single** `SendInput` (0 ms
hold), which a 60 Hz DirectX game can sample right past — now a ~60 ms held press
with the cursor frozen for its duration. A finding that outweighs all of it: this is a **laptop**, and on
**battery** the CPU drops to its 1.4 GHz base clock, which costs ~2.1× on every
stage — 9–11 FPS *with* every fix applied, i.e. worse than the unfixed loop on
AC. Power state is the largest single term in the budget. Implemented and
unit-tested; **live test against the game still pending**, like everything else
in Step 5.
⁹ 3.2.4 changes **no model and no FSM value** — it is a measurement of a
contract (§A.3) that had been asserted since Strategy 2 and never checked, and
Ted accepted its recommendation: the live `--pad` default moved 0.15 → 0.35 in
`play.py`, `webcam_demo.py`, and the dashboard tracker (training's `bbox_pad`
config is untouched). Training crops HaGRID's **annotated hand bbox**; the live
loop crops MediaPipe's **21-landmark hull**. Both used to add the same
`pad = 0.15`, which is why the geometry *looked* matched — but a hull is not a
bbox. Measured over the full downloaded population
(27k–29k images/gesture, from HaGRID's own landmarks, no webcam needed): the hull
is **0.768×** the annotated box linearly, and the spread is anatomical —
**`fist` 0.705** (curled fingers sit *inside* the silhouette) vs. `palm` 0.849
(landmarks reach the fingertips). So the most-distorted class at inference is the
one class that has to be crisp to fire a click. MediaPipe itself is exonerated:
its detected hull matches HaGRID's ground-truth hull at **0.995–1.054×**, so the
gap is a *definition* mismatch, not detection error. At the click gate this costs
the tail, not the average — median p(fist) 0.927 vs 0.984, but **p10 0.464 vs
0.932**, and `K` consecutive frames over the gate is exactly what a bottom-decile
frame breaks. The "overfit" hypothesis is ruled out specifically: val = test =
1.0000 with no variance gap, the loss *reverses* when only the crop changes, and
the tuned champion is the **most** robust of the five checkpoints under live
geometry (81.5% fist click / 0.7% false, vs. the 3.2 model it replaced at 77.8% /
**10.7%**). Reproduce with
[`scripts/eval_crop_geometry.py`](../../../scripts/eval_crop_geometry.py).

## The arc (1 → 2 → 2.1 → 3 → 3.1 → 3.1.1 → 3.2 → 3.2.1 → 3.2.2 → 3.2.3 → 3.2.4)

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

**3.1.1 → 3.2 (AD-21).** With the cursor solid, attention moved to the *other*
click failure: the 2-class palm/fist model must map **every** hand shape onto
palm or fist, so an unrecognised pose (point, "ok", a slow squeeze's mid-frames)
can read as `fist` and fire an **accidental click**. 3.2 keeps the whole
pipeline but **retrains the classifier** so class 0 becomes `not_fist` — trained
on six diverse non-fist gestures — making "unknown → no-click" a learned default
rather than an FSM band-aid. `fist` stays the one crisp positive. `ok` is held
out unseen to measure the real metric: does an untrained pose read as `not_fist`?
The palm/fist model stays runnable for the A/B.

**3.2 → 3.2.1.** With the *model* handling false clicks, the remaining question is
**feel**: 3.2's FSM (conf 0.70, K 3) is deliberately steady, and a live tester
wanted clicks to fire faster. 3.2.1 keeps the 3.2 checkpoint untouched and moves
only the two FSM knobs — tighten the per-frame gate to **0.80** while dropping the
confirm to **K = 2**. The stricter gate is there to buy back the false-click
safety the looser confirm gives up (a wash by construction), so the net is a
snappier click without leaning harder on the FSM to reject unknown poses — that's
still the model's job (AD-21). It's a runtime preset, not a retrain: pick it with
`play.py --strategy 3.2.1` or the dashboard's Click-strategy picker; the live
3.2-vs-3.2.1 A/B decides whether it stays.

**3.2 → 3.2.2 (Week 4, AD-22/23/24).** Orthogonal to 3.2.1 — this axis retrains
the *checkpoint* 3.2's FSM presets point at, rather than the FSM. Building the
Week-4 tuner surfaced a bug, not just a hyperparameter win: timm's head init
scales with class count, so the 2-class head was starting ~25× wider than
PyTorch's own default and spending ~30 of its 40 epochs undoing that alone
(AD-22). Fixing it plus a 60-trial search plus unfreezing 2 backbone blocks
takes val 0.9221→1.0000, registered in MLflow as the `champion` alias (AD-24)
and reproducible end-to-end via a small task DAG (AD-23). The honest read: the
offline benchmark is now saturated (verified leakage-free, but no longer able to
separate candidates), and the metric with actual headroom — false clicks on the
unseen `ok` gesture — didn't move. Whether this checkpoint becomes the default
is, like every strategy since 3.1, a live-test call.

**3.2.2 → 3.2.3.** With the model saturated offline, the live test's remaining
complaints turned out not to be model problems at all: the loop ran at 21 FPS
with 67 ms of stale-frame lag, and some clicks never reached the game. 3.2.3 is
the first strategy on the **runtime** axis, and the first whose findings mostly
came from *measuring instead of reasoning* — two of the three fixes did something
other than what was predicted for them. (a) Every FSM gate counts *frames*, so a
46 ms frame plus a 67 ms stale sample leaves a natural ~150 ms squeeze about one
frame of margin; no knob recovers a pose the loop never sampled. (b)
`pydirectinput.click()` presses and releases in one `SendInput` — a 0 ms hold a
60 Hz DirectX title can miss outright, a *delivery* bug independent of both FPS
and the model. The fixes are all plumbing: the already-exported ONNX model (the
classifier stage 20.7 → 6.2 ms), a torch thread cap that turned out to matter
more than expected (a 16-thread pool starves MediaPipe *and* torchvision
preprocessing), newest-frame-only threaded capture (which buys latency, not
throughput), and a ~60 ms held click with the cursor frozen for its duration.
**No** FSM value was touched, so the still-pending 3.2-vs-3.2.1 A/B is finally
judged through a sampling interval close to what those presets were designed for.
