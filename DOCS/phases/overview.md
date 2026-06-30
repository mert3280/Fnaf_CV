# Phase Overview — FNAF Hands-Free (5 Weeks)

> One-page map of the five phases. Each links to its full plan. The arc: **prove the data → baseline model → fine-tune & export → drive the real game → harden & demo.** Every phase ends with a hard exit gate that the next phase depends on.

| # | Week | Phase | Core deliverable | Exit gate |
|---|---|---|---|---|
| [1](phase-1-setup-and-data.md) | 1 | Setup & Data Validation | Reproducible repo + working `DataLoader` over the class subset | Verified batch loads/renders; class list, input size & normalization locked |
| [2](phase-2-baseline-model.md) | 2 | Baseline Model (frozen backbone) | Trained frozen-backbone checkpoint + metrics | Reproducible baseline number + confusion matrix |
| [3](phase-3-training-finetuning.md) | 3 | Full Training & Fine-Tuning | Fine-tuned model, honest test eval, exported artifact | ≥90% test acc, one exported file, measured per-frame latency |
| [4](phase-4-realtime-and-control.md) | 4 | Real-Time Inference & Control | Webcam → gesture → FNAF input loop | Each gesture reliably fires its in-game action |
| [5](phase-5-integration-demo.md) | 5 | Integration, Robustness & Demo | Hands-free Night 1 (recorded) + docs merged | Full night completed on video; `DOCS/` merged to `main` |

---

## Phase 1 — Setup & Data Validation
**Goal:** Prove the HaGRID data is real, understood, and loadable — no modeling yet; this de-risks the dataset.
- **Core work:** repo/env scaffold (CUDA on the RTX 4060 verified); annotation parser for HaGRID JSON (UUID → bbox + 21 landmarks); image acquisition (subsample now, 512px planned); `HagridDataset` + transforms; EDA notebook.
- **Key decision:** finalize the **working class subset** — `like, dislike, fist, one, two_up, palm, ok, mute` — and the draft **gesture → FNAF action map** (Office vs. Camera mode).
- **Done when:** one notebook cell renders an augmented, correctly-labeled batch with bbox overlays.

## Phase 2 — Baseline Model (Frozen-Backbone Transfer Learning)
**Goal:** First end-to-end transfer-learning run — pretrained `timm` backbone, fresh head, **backbone frozen**, head-only training. This is the baseline every later gain is measured against.
- **Core work:** `build_model` (swap head via `num_classes`, freeze body, assert trainable-param count); config-driven `train.py` (AdamW, CrossEntropy, checkpoint best-by-val, TensorBoard, seeds); eval = accuracy/loss curves + confusion matrix + per-class precision/recall.
- **Human-owned:** the freezing decision and reading the curves (frozen features good enough, or fine-tune?).
- **Done when:** frozen model trains to completion, beats random widely, metrics logged & reproducible from saved config.

## Phase 3 — Full Training & Fine-Tuning
**Goal:** Turn the baseline into a **deployable** model via staged fine-tuning, then export it.
- **Core work:** **progressive unfreezing** (head → top blocks → optional full backbone) with **discriminative LRs** (`lr/10`–`lr/100` on backbone) + scheduler/early stopping; label-aware augmentation (careful flip — it changes `*_inverted` and left/right meaning); HP sweep picking best by **val**; **test split evaluated once** (per-class P/R/F1 + confusion matrix); export to **TorchScript/ONNX** with a parity smoke test + latency benchmark.
- **Done when:** test acc ≥ **90%** on the subset, exported model matches PyTorch on a fixed sample, latency within the real-time budget.

## Phase 4 — Real-Time Inference & Game Control
**Goal:** Connect a live webcam to the model and drive **actual FNAF**.
- **Pipeline:** Webcam (OpenCV) → preprocess (must match Phase 3 exactly) → model → **smoothing/debounce** → gesture event → mode-aware controller → simulated mouse input → FNAF, with a debug HUD.
- **Stabilization (critical):** N-frame majority vote, confidence threshold, per-action cooldown, explicit `mute` idle, edge- vs. level-triggered actions.
- **Controller:** small state machine, `Office`/`Camera` modes (`palm` raises monitor, `fist` lowers it).
- **Highest risk — de-risk Day 1:** FNAF is DirectX; **test `pydirectinput` click registration in-game on Day 1**, calibrate button coordinates in `configs/fnaf_layout.yaml`, and wire a global **kill-switch**.
- **Done when:** a gesture reliably triggers the intended door/light/camera action inside the running game.

## Phase 5 — Integration, Robustness & Demo
**Goal:** Make it survivable, robust, and presentable — and beat Night 1 hands-free.
- **Core work:** startup **calibration** (webcam/lighting/ROI/thresholds → `configs/session.yaml`); robustness passes (lighting, clutter/other people, idle handling, latency under game load, debounce tuning) — possibly a small **fine-tune on self-captured frames** to close the train/serve gap.
- **Headline metric:** complete **Night 1 entirely hands-free**, logging failure incidents and iterating.
- **Ship:** `README`, architecture diagram, finalized `results.md` with real-world reliability, known limitations/future work, **demo video**, final AI-usage entry, branch → PR → merge to `main`.
- **Plus:** a written retrospective (what worked, learning goals hit, honest reliability numbers).
