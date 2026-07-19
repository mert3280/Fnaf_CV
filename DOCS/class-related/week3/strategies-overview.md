# Strategies Overview — 3 approaches, 6 iterations

> A one-page map of every approach we've tried to turn a webcam frame into a
> game action, why each is *genuinely distinct* from the last, and what the
> evidence said. Written for the Week 3 briefing ([W3P1.md](W3P1.md)) — it
> answers the three things the audience is told to challenge: are the
> experiments really different, is the model choice justified by the evidence
> (not just the top number), and can a non-technical stakeholder follow why.
>
> Source of truth (deeper detail): [DOCS/build/strategies/](../../build/strategies/README.md).
> Decisions behind each pivot: [architecture-and-decisions.md](../../build/architecture-and-decisions.md) (`AD-NN`).

## The problem in one line

Play *Five Nights at Freddy's* hands-free: a webcam watches one hand, the hand
**moves the mouse**, and **open-hand vs. closed-fist** decides whether to click.
Every experiment below is a different answer to *"how do we read that hand
reliably enough to trust it with the real game?"*

## The three strategies at a glance

| # | Strategy | What's different about it | Headline result | Verdict | Status |
|---|---|---|---|---|---|
| **1** | Full-frame single-stage | Classify the **whole frame**, one model, no detector | **53%** test acc (8-class) | Too weak — but told us *why* | Retired → baseline/fallback |
| **2** | Two-stage: detect → crop → classify | Add MediaPipe to **crop to the hand** first | **92%** test acc | Reads gestures well; distance-sensitive live | Superseded by 2.1 |
| **3** | Cursor + binary click | Hand **drives the cursor**; classify **palm/fist only** | **98%** test acc (binary) | Works offline; live-tuned across 3.1 / 3.1.1 | **Current** |

Iterations (2.1, 3.1, 3.1.1) are **hardening passes on the same strategy**, not
new strategies — each one closes a specific gap a *live* test exposed. They're
listed separately below so the distinction is honest.

---

## Strategy 1 → 2 → 3: why each is a real experiment, not a tweak

The three strategies change **different variables**, which is what makes them
distinct rather than a hyperparameter sweep:

- **1 → 2 changed the *input*.** Same backbone, same freeze policy, same data,
  same everything — the *only* change was cropping to the hand instead of feeding
  the whole frame. That isolation is deliberate: it makes the **+39-point jump
  (53% → 92%) attributable to cropping alone**, which is the entire evidence base
  for going two-stage ([AD-04](../../build/architecture-and-decisions.md#ad-04)).
- **2 → 3 changed the *task*.** Same two-stage pipeline, but the outputs *mean*
  something new: MediaPipe's hand position now moves the cursor, and the
  classifier drops from **8 gestures to a binary palm/fist**. This is a product
  pivot ([AD-17](../../build/architecture-and-decisions.md#ad-17)), not just a
  metric chase — it deletes the whole gesture→action mapping and concentrates all
  model risk on the two most visually distinct poses in the dataset.

### The 1 → 2 finding (the one worth highlighting)

Strategy 1 scored 53% — and the important part is **why it failed**. It wasn't
overfitting or a bad model: a HaGRID hand is a **median 1.8% of the frame**, so
after resizing, the gesture is a ~40 px blob in a sea of background. A frozen
ImageNet backbone spent its attention on walls and torso. The fix was to **change
the input, not the model** — crop to the hand. Same model, cropped input, **53% →
92%**. This is the headline experiment: a clean, isolated variable with a large,
explainable effect.

---

## The iterations — each closes a *live* gap the offline numbers hid

The offline test accuracy looked great (92%, then 98%), but every strategy
behaved worse on a real webcam than on the dataset. The iterations are the honest
record of chasing that train/serve gap:

| Iter | Live problem it found | Fix | Type |
|---|---|---|---|
| **2.1** | Accurate with hand **near body**, degrades **close to lens** (perspective/scale/focus the model never saw) | 6 robustness levers: framing hint, live pad, detection-confidence, perspective/blur aug, progressive unfreeze, self-capture tool | Runtime + opt-in retrain |
| **3.1** | Cursor **drifted with hand distance** — anchor came from frame-clipped landmarks, so a close/large hand biased its position | Recompute anchor from **unclipped** landmarks → size-invariant pointing | Calibration only |
| **3.1.1** | Detection touchy; **slow squeezes never clicked**; cursor *gain* still scaled with distance | Detection 0.5 → 0.3; FSM **grace window** (ambiguity pauses, doesn't disarm); **adaptive control box** (constant gain at any distance) | Runtime |

The **slow-squeeze bug (3.1.1)** is the most instructive: it was **structural,
not a tuning problem**. The click state machine disarmed on the first ambiguous
mid-squeeze frame, and re-arming required seeing *palms* again — but the hand was
already a fist, so a slow clench could *never* fire, by construction. Fast
squeezes worked only because they crossed the ambiguous zone within one frame.
No threshold tweak fixes that; it needed a redesign (the grace window).

---

## The models — every trained checkpoint, and the detector that feeds them

Three CNNs were trained, one per strategy. All are `mobilenetv3_large_100`,
ImageNet-pretrained, **frozen backbone + head-only**, same seed / epochs / split
— so any difference between rows is the *input* or the *task*, nothing else. Full
per-model docs: [DOCS/models/](../../models/README.md).

| Model | Strategy | Classes | Input crop | Val acc | **Test acc** | Random floor | Status |
|---|---|---|---|---|---|---|---|
| `baseline_mnv3_large` | 1 | 8 | full frame | 0.570 | **0.530** | 0.125 | retired / fallback |
| `bbox_frozen_mnv3_large` | 2 | 8 | MediaPipe crop | 0.940 | **0.917** | 0.125 | pre-pivot A/B record |
| `palmfist_frozen_mnv3_large` | 3 | **2** | MediaPipe crop | 0.964 | **0.9815** | 0.50 | **current** |

**Current model, read honestly** — `palmfist_frozen_mnv3_large`:
- Per-class F1: **palm 0.981 / fist 0.982**; only **6 errors in 325** test images.
- The errors lean one way: **4 of 6 are palm misread as fist** → a *phantom
  click* is the likelier failure than a missed one. The click FSM's K-frame
  debounce + confidence gate exists exactly to absorb that.
- Val was **still climbing at epoch 40** (train 0.998, no overfitting signal in
  the frozen regime) — headroom is left on the table deliberately.

### MediaPipe HandLandmarker — the detector (borrowed, *not* trained)

Stage 1 is Google's off-the-shelf **MediaPipe Tasks `HandLandmarker`** (VIDEO
mode) — it finds the hand, returns **21 landmarks**, and we take their bounding
box for the crop *and* the palm anchor for the cursor. **We train none of it**;
that's the point of the two-stage split (AD-04/AD-05) — Google already solved
"where is the hand," so our ML effort goes entirely into the classifier.

Because it's the front of the pipeline, its **detection rate caps everything** —
a hand it never finds is a frame the CNN never sees. Measured on **240 labeled
HaGRID stills** (the honest, *pessimistic* proxy):

| MediaPipe metric | Value | What it means |
|---|---|---|
| **Hand detected** | **~48%** of frames | misses HaGRID's far / blurred / side-on hands |
| Classify \| detected (live MP crop) | **~71%** | vs **91.7%** on HaGRID's own crop → a **~20 pt train/serve gap** |
| **End-to-end** (detect *and* classify) | **~34%** | the compounded worst case on this hard proxy |

**Why this is a floor, not the live experience:** HaGRID stills are a *harder*
distribution for MediaPipe than the real use case — a hand deliberately held up
to a webcam is far easier to detect than a candid photo at conversational
distance. Live, MediaPipe locked on reliably; the 48% is the dataset being
adversarial, not the detector being weak.

**Detector tuning levers** (all live, no retrain — Strategy 2.1 fix 6 / 3.1.1
fix 1): three confidence gates (`detection`, `presence`, `tracking`), **defaulted
0.5 → 0.3** in 3.1.1 for a stickier lock and faster re-acquire after motion blur.
Raise `--detect-confidence` back toward 0.5 if it ever grabs a non-hand. Its
per-frame cost is also the main risk to the **≥15 FPS budget** — an open Week-4
measurement.

## Model selection — justified by evidence, not just the top metric

The chosen model is **`palmfist_frozen_mnv3_large`**: MobileNetV3-Large,
ImageNet-pretrained, frozen backbone + a fresh 2-class head, fed MediaPipe hand
crops. Why this one, in stakeholder terms:

1. **Small on purpose.** MobileNetV3 is chosen for **real-time speed** (≥15 FPS
   with MediaPipe in the loop), not because it topped a leaderboard — a bigger,
   more accurate model that can't keep up with a live game is the wrong choice
   here.
2. **The 98% is read carefully, not celebrated.** Binary random is **50%**, not
   12.5% — so the honest metric is **per-class F1 (palm 0.981 / fist 0.982)**,
   not raw accuracy compared against the 8-class rows. We compare like-for-like.
3. **The offline number is not the number that matters.** 98% is on HaGRID's
   *own* clean crops. The metric that decides "good enough to play" is
   **palm↔fist reliability at real arm's-length distance** — the exact regime
   where Strategy 2 degraded. That's why the work moved to live iterations (3.1,
   3.1.1) instead of stopping at the test score.
4. **Risk was engineered down, not just measured.** Picking the two most distinct
   poses means a misread costs **one click**, not a wrong in-game action — the
   product design lowers the stakes of any model error.

**Honest trade-off:** pointing precision now depends on hand steadiness and
smoothing, and holding an arm up is more tiring than flashing a pose. Both are
tunable (control-box size, smoothing) and both are open live questions.

---

## Where this stands heading into Week 4

- **Done:** all three strategies built and measured; binary model trained (98%
  test); full runtime stack implemented and offline-verified (39 checks) — cursor
  mapper, click FSM, input layer, play loop, kill-switch.
- **Current head:** Strategy **3.1.1**, defaults-on, live re-test pending.
- **Next (Week 4):** the verdict that counts — **in-game FNAF test**. Open
  questions: can the smoothed cursor hit FNAF's smallest targets; does palm↔fist
  read cleanly with the arm extended; arm fatigue over a real session; and does
  the whole loop hold ≥15 FPS.

## The non-technical version

We tried the simplest thing first (look at the whole picture) — it barely worked,
and it *told us why*: the hand is tiny in the frame. So we **zoomed in on the
hand** first, and accuracy jumped from a coin-flip-ish 53% to 92%. Then we
**simplified the job** to just "hand open or closed" and let the hand steer the
mouse — that's 98% in the lab. The lab score isn't the finish line: on a real
webcam the hand looks different up close, so the last three rounds of work were
about making it feel right **live** — steady cursor, reliable clicks, same feel
whether your hand is near or far. Week 4 is the real test: playing the actual
game.
