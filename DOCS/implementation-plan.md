# Implementation Plan — Gesture-Controlled FNAF (Living)

> **Append-only.** This is the project's living 5-week execution plan. When the
> plan changes I **add a dated entry to the change log at the bottom** rather than
> rewriting history. The detailed, per-phase mechanics live in
> [phases/](phases/); the week↔requirement mapping and milestones live in
> [schedule.md](schedule.md); the "why" behind each choice is
> [architecture-and-decisions.md](architecture-and-decisions.md) (`AD-NN`).

**End goal (one line):** play the original **Five Nights at Freddy's 1**
hands-free — a webcam-fed transfer-learned image classifier recognizes 8 static
hand gestures and drives the real game via simulated mouse input.

---

## Finalized Core Requirements (measurable)

Re-affirmed after Week-2 data validation. Full status table + rationale for the
two revisions in [week2/data-understanding-report.md](week2/data-understanding-report.md) §5.

| # | Requirement | Measure of done |
|---|---|---|
| R1 | Classify 8 static gestures (`like, dislike, fist, one, two_up, palm, ok, mute`) | model outputs a class per RGB frame |
| R2 | Two-stage detect-and-crop preprocessing (bbox +15% → 224²), identical train↔runtime | preprocessing parity test passes (AD-04, AD A.3) |
| R3 | Honest eval: 70/15/15 split grouped by `user_id` | `assert_disjoint` — 0 shared users across splits (AD-10) |
| R4 | Transfer learning: pretrained `timm` backbone, fresh head, freeze → progressive unfreeze | trainable-param assertions per stage (AD-08) |
| R5 | **Test accuracy ≥ 90%** on the 8-class by-user test set | one-time held-out eval, per-class P/R/F1 |
| R6 | Real-time loop **≥ 15 FPS** with smoothing + debounce + MediaPipe crop | measured FPS + stable HUD |
| R7 | Drive real FNAF 1 via `pydirectinput`; global kill-switch; input verified early | in-game action fires; kill-switch cuts input instantly |
| R8 | Full hands-free **Night 1** playthrough | completed on video |

---

## Five-week plan (execution order)

Each week = one phase with a hard exit gate the next week depends on. Detailed
plan behind each link; summarized here so the whole arc is on one page.

| Wk | Phase | Build | Exit gate |
|---|---|---|---|
| 1 | [Setup & Data](phases/phase-1-setup-and-data.md) | repo/env, HaGRID parser, downloader, `HagridDataset`, EDA | batch loads/renders; class list, input size, normalization locked |
| 2 | [Baseline Model](phases/phase-2-baseline-model.md) | frozen-backbone `timm` + fresh head, `train.py`, TensorBoard, confusion matrix | reproducible baseline number *(done: 53% test full-frame, [results.md](results.md))* |
| 3 | [Fine-Tuning](phases/phase-3-training-finetuning.md) | full-frame-vs-bbox A/B, progressive unfreeze + discriminative LRs, aug, HP sweep, **one-time test eval**, ONNX/TorchScript export | **≥90% test acc**, exported artifact matches PyTorch, latency measured |
| 4 | [Real-Time & Control](phases/phase-4-realtime-and-control.md) | OpenCV loop → MediaPipe crop → model → smoothing/debounce → `pydirectinput` → FNAF, HUD, kill-switch | each gesture reliably fires its in-game action |
| 5 | [Integration & Demo](phases/phase-5-integration-demo.md) | calibration, robustness passes, hands-free Night 1, docs + demo video, merge to `main` | full night beaten hands-free on video |

**Human-owned decisions (not delegated to AI — see [Claude.md](Claude.md)):** the
freeze/unfreeze schedule and reading training curves; under- vs. over-fitting
diagnosis; the gesture→FNAF action map and debounce/cooldown tuning; final
evaluation honesty.

**Top risks & fallbacks** (full register in [schedule.md](schedule.md)): 716 GB
dataset → never download full, use range-request subset (**done**); small hand in
frame → MediaPipe crop (**now core, R2**); DirectX ignores synthetic clicks →
`pydirectinput`, test Day-1 of Phase 4; latency → smaller backbone + ONNX;
gesture chatter → debounce + confidence threshold + `mute` idle.

---

## Change log (append-only)

| Date | Change |
|---|---|
| 2026-06-30 | Plan created from proposal: solo, transfer learning w/ custom head + freezing, drive real FNAF 1 via simulated input, static poses only. Five phases defined. |
| 2026-07-06 | Data acquisition finalized: retired "scheduled ingestion" (static dataset) for a deterministic range-request downloader; retired subsample-as-test for a **70/15/15 split grouped by `user_id`** (AD-16). |
| 2026-07-08 | Data-prep pipeline built (`src/data/*`, `configs/data.yaml`): index → grouped split → crop/normalize/augment → loaders. |
| 2026-07-10 | Phase-2 baseline recorded: `mobilenetv3_large_100` frozen, full-frame → **57% val / 53% test** ([results.md](results.md)). |
| 2026-07-12 | **Week-2 finalization.** Data validated (8594 usable images, 100% readable, EDA in [week2/](week2/)). Revised **R2**: primary preprocessing moved full-frame → **two-stage detect-and-crop** (AD-04), driven by the EDA (median hand = 1.8% of frame) + the 53% full-frame baseline; **MediaPipe crop promoted from contingency to a planned Phase-3/4 component**. Overfit-single-batch test passed (loss → 0), pipeline validated. |
