# Plan v2 — Hands-free FNAF via cursor control

> **Status 2026-07-26:** Steps 1–2 **done and now tuned** (AD-21 fist-vs-rest is the
> deployed model; Week-4 tuning + MLflow registry + serving endpoint landed — see the
> [Week-4 update](#week-4-update-2026-07-26--on-track-tuning--pipeline-done-step-5-still-the-gate)).
> Steps 3–4 **code-complete and offline-verified** (`src/rt/cursor.py`,
> `src/control/click_fsm.py`, HUD preview in `webcam_demo`). Step 5 **built,
> not yet run against the game** (`src/control/play.py` + `input_sim.py`; the
> in-game registration test and all live tuning are Ted's) — **this is the
> outstanding gate.** Step 6 not started.

**Effective 2026-07-14** (the [AD-17](architecture-and-decisions.md#ad-17--pivot-to-cursor-control-motion-tracked-cursor--binary-click-classifier)
scope pivot). This replaces the phase plans now archived in
[../legacy/phases/](../legacy/README.md); the design itself is specified
in [Strategy 3](strategies/3-cursor-and-click/03-cursor-and-click.md).

## Week-4 update (2026-07-26) — on track; tuning + pipeline done, Step 5 still the gate

**On track, roadmap unchanged.** Week 4 executed the tuning levers Step 2 had been
holding open and built the orchestration/serving layer around them. **Step 5
(drive the real game) is still the next build milestone and is still unstarted —
that has not moved since 2026-07-15, and it remains the project's real risk.**

What landed (full detail: [Week-4 report](../class-related/week4/tuning-orchestration-report.md)):

- **Tuning** ([`src/tune.py`](../../src/tune.py), [`configs/tune_fistvsrest.yaml`](../../configs/tune_fistvsrest.yaml)):
  a 60-trial TPE search on cached frozen features plus an end-to-end
  `unfreeze × augmentation` grid, every trial an MLflow run, seed-repeated at the
  top, test read once at the end. **Tuning target was the *deployed* model
  (`fistvsrest`, AD-21)** — not Week 3's named candidate (`palmfist`), because
  AD-21 replaced the negative class after that report; `palmfist` stays as the A/B.
- **The week's actual finding (AD-22):** the deployed run's *"val still climbing at
  ep40"* was **timm's 2-class head initialization**, not an epoch shortage —
  `r = 1/sqrt(out_features)` gives a binary head ±0.707 weights, ±27 logits, CE
  2.70 at init. Fixing only the init, same recipe and seed: **val 0.9221 → 0.9740**.
  Hyper-parameter search then added ~1 point on top. `head_init` is an opt-in knob
  with the old default preserved; **making it the project default is my call
  (AD-22 is Proposed).**
- **Orchestration (AD-23):** [`src/pipeline/`](../../src/pipeline/dag.py) — a local
  task graph with freshness/`cached` skips, retries, upstream-failure propagation,
  and JSON run records. **Event-triggered (`--if-data-changed`), not scheduled**, and
  the [pipeline diagram](../class-related/week4/pipeline-diagram.png) is *rendered
  from the task definitions* so it can't drift.
- **Deployment (AD-24):** MLflow **Model Registry** + `champion` alias, served by
  [`src/serve/app.py`](../../src/serve/app.py) over a pyfunc artifact that carries
  its own preprocessing; ONNX exported with a torch-parity check (AD-11 satisfied).

**What this does NOT change:** every number above is still offline, on HaGRID's
annotated crops. The live/arm's-length gap is untouched and unmeasured, so the
Step-2 follow-up (self-capture fine-tune at cursor distances) and Step 5's live
tuning are exactly where they were. The tuned model is a better starting point for
that work, not a substitute for it.

## Week-3 update (2026-07-19) — on track; tracking consolidated, tuning next

**On track — no roadmap change.** Steps 1–2 (data trim + binary retrain) and the
offline halves of Steps 3–4 are done; Step 5 (drive the real game) remains the next
build milestone and is unchanged. This week was **experiment consolidation and
reporting**, not new modeling:

- Adopted **MLflow** as the experiment store: the three already-run experiments
  (`baseline` / `bbox_frozen` / `palmfist_frozen`) are now logged and comparable via
  [`scripts/mlflow_log_runs.py`](../../scripts/mlflow_log_runs.py) +
  [`scripts/mlflow_export_comparison.py`](../../scripts/mlflow_export_comparison.py)
  (store `./mlruns`, git-ignored; exported comparison committed under
  [`class-related/week3/`](../class-related/week3/ml-experimentation-report.md)).
- Wrote the Week-3 [ML experimentation report](../class-related/week3/ml-experimentation-report.md)
  (feature engineering, experiment design, 3-run results, model selection).

**Confirmed candidate for Week-4 tuning:** `palmfist_frozen_mnv3_large`. Tuning
work is already scoped inside **Step 2 / Step 5** below and does not change them —
Week 4 executes those levers (AD-08 Stage-B unfreeze, AD-09 augmentation, and a
self-capture fine-tune at cursor distances) explicitly to close the **train/serve
gap** (0.9815 on HaGRID crops vs the lower, still-unmeasured live/arm's-length
number). No new steps needed; the plan already anticipated this.

## The system in one line

MediaPipe hand tracking **moves the mouse cursor** (absolute mapping); a
transfer-learned MobileNetV3 classifies the hand crop as **palm (no click) or
fist (click)**; a debounced FSM fires **one click per palm→fist squeeze** into
the real game via `pydirectinput`.

## What's already banked (carries forward unchanged)

- **Data pipeline** (`src/data/`): index → user-grouped 70/15/15 split →
  bbox-crop → backbone-driven transforms. Class list is one config line.
- **Images on disk**: `palm` (1,082) and `fist` (1,097) are already downloaded —
  the pivot needs **no new acquisition**.
- **Training stack** (`src/train.py`, `src/models/build.py`): config-driven,
  checkpoint self-describing, unfreeze knob ready (AD-08).
- **Two-stage runtime** (`src/rt/`): MediaPipe detector, shared crop contract,
  webcam demo with the six Strategy-2.1 robustness fixes, self-capture tool.
- **Evidence base**: strategies 1 → 2 → 2.1 measured and documented; the 8-class
  checkpoints still run for comparison.

## The workflow (remaining build, in order)

### Step 1 — Trim the data to `palm` + `fist` (AD-18)
- `configs/data.yaml → classes: [palm, fist]`; rerun `python -m src.data.dataset`
  to verify the recomputed split (~2,179 images, user-grouped, no leakage).
- Regenerate the split manifest next to the future checkpoint.

### Step 2 — Retrain MobileNetV3 as the binary click classifier (AD-18)
- Same recipe as `bbox_frozen_mnv3_large`: pretrained backbone, fresh 2-class
  head, `crop_mode: bbox`, frozen first (AD-08 Stage A).
- **Ted's calls, in the loop:** whether/when to unfreeze (Stage B), whether to
  enable the 2.1 perspective/blur augmentation, what test number is "good
  enough to play." Evaluate on val, report test **once** (AD-10).
- Likely follow-up: **self-capture fine-tune at cursor distances**
  (`src/rt/capture_dataset.py`) — arm-extended hands are exactly the regime
  HaGRID lacks and cursor driving lives in.
- Document the run under `DOCS/models/` like the previous two.

### Step 3 — Stage 1 becomes the cursor (AD-19)
- New `src/rt/cursor.py`: palm-center anchor (landmarks 0/5/9/13/17) → mirror →
  control box (~60%×55%, config) → EMA (`alpha ≈ 0.35`) + dead-zone → screen
  coords; freeze on no-hand, re-seed on re-detection.
- Prove it in the webcam demo HUD first (draw the target point), **before**
  wiring real cursor movement.

### Step 4 — Stage 2 becomes the click (AD-20)
- New `src/control/click_fsm.py`: ARMED → K confident fist frames → one click →
  re-arm on K confident palm frames; DISARM on dropout; ~0.3 s cooldown.
- HUD shows FSM state; **K, thresholds, cooldown are Ted's live-tuning calls.**

### Step 5 — Drive the real game (AD-14, amended)
- `pydirectinput` **movement + click** registration test in FNAF on day 1 of
  this step — movement smoothness is a new unknown on top of the old click
  check. Global **kill-switch** before anything else runs.
- Then: live Night-1 attempts, tune control box / smoothing / K against real
  buttons, log failures honestly.

### Step 6 — Robustness & demo
- Lighting/background/fatigue passes, calibration notes, recorded hands-free
  Night 1, final results write-up, retrospective + AI-usage log wrap-up.

## Success criteria (unchanged in spirit from the proposal)

1. **Headline demo:** complete FNAF Night 1 entirely hands-free, on video.
2. **Honest numbers:** binary test accuracy reported once, *plus* the live
   metrics that actually matter now — click transition reliability and cursor
   pointing precision at play distance.
3. **Learning objective intact:** the transfer-learning story (freeze →
   unfreeze decisions, curve reading, train/serve gap management) is fully
   exercised by the retrain in Step 2 — same mechanics, simpler label space.

## Risks specific to v2

| Risk | Mitigation path |
|---|---|
| Cursor can't hit small FNAF controls | Tune control box / smoothing → One-Euro filter → relative-mapping fallback (AD-19) |
| palm↔fist unreliable at arm's length | 2.1 aug levers → self-capture fine-tune at cursor distances |
| Arm fatigue during a full night | Control-box sizing (small hand motion = full screen travel); rest via no-hand freeze |
| `pydirectinput` movement not smooth in FNAF | Test day 1 of Step 5; fallback to raw SendInput move calls if needed |
