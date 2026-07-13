# Architecture & Decisions

**Project:** Gesture-Controlled *Five Nights at Freddy's*
**Author:** Ted Roper · **Status:** Planning · **Last updated:** 2026-06-30

This document records **how the system is built** (Part A — Architecture) and **why each major choice was made** (Part B — Decision Records). It is the single source of truth for technical direction; the per-phase build steps live in [phases/](phases/).

> **Conventions:** Decision records (ADRs) use *Context → Decision → Rationale → Alternatives → Consequences*. Each is numbered `AD-NN` and can be referenced from code/PRs. "Status" of an AD is **Accepted** unless marked *Proposed* (will be confirmed during the phase that implements it) or *Superseded*.

---

# Part A — Architecture

## A.1 System overview

The system is a **real-time perception → decision → actuation loop**. A webcam streams frames; a **hand detector (MediaPipe) finds and crops the hand** in each frame; a CNN classifies the gesture in that crop; a smoothing layer converts noisy per-frame predictions into stable discrete *gesture events*; a mode-aware controller maps events to FNAF actions; an input-simulation layer issues the corresponding mouse clicks to the real game. This **two-stage detect-then-classify** design is AD-04.

```
                         ┌──────────────────────── OFFLINE (Phases 1–3) ────────────────────────┐
   HaGRID annotations ─▶ Data pipeline ─▶ timm backbone + custom head ─▶ fine-tune ─▶ export
   (DATA/ JSON: bbox)    (parse/crop-to-  (transfer learning)            (TorchScript/ONNX)  │
                          hand bbox)                                                         │ model artifact
                         ┌──────────────────────── ONLINE  (Phases 4–5) ────────────────────┼──────────┐
                         ▼                                                                   ▼          │
   Webcam ─▶ Capture ─▶ Detect+Crop ─▶ Preprocess ─▶ Inference ─▶ Smoothing/Debounce ─▶ Controller ─▶ Input ─▶ FNAF
   (OpenCV)  (frame)    (MediaPipe      (resize/norm) (exported    (N-frame vote,        (Office/Cam   sim   (Steam,
                         hand bbox)                    model)       cooldown, idle)       state machine)(pydirectinput)
                                                          │
                                                          ▼
                                                   Debug HUD overlay
```

## A.2 Components

### Offline (training) pipeline
| Component | Module (planned) | Responsibility |
|---|---|---|
| Annotation parser | `src/data/hagrid_annotations.py` | HaGRID JSON → records `{uuid, label, bbox, landmarks}`; validation. |
| Dataset | `src/data/dataset.py` | `(image_tensor, label_idx)`; full-frame or bbox-crop mode; transforms. |
| Model builder | `src/models/build.py` | Load `timm` backbone, swap head, set freeze state. |
| Trainer | `src/train.py` | Config-driven loop, checkpointing, TensorBoard, seeds. |
| Evaluator | `src/eval.py` | Test-set metrics, per-class report, confusion matrix. |
| Exporter | `src/export.py` | TorchScript/ONNX export + parity smoke test. |

### Online (runtime) pipeline
| Component | Module (planned) | Responsibility |
|---|---|---|
| Capture | `src/rt/capture.py` | OpenCV webcam loop, mirroring, frame timing. |
| Detector | `src/rt/detector.py` | **MediaPipe Hands → hand bbox per frame; crop (+`bbox_pad`) before classify (AD-04).** No-hand → idle. Toggle via `runtime.use_detector`. |
| Preprocess | `src/rt/preprocess.py` | Match training transforms **exactly** (size, norm, crop policy); operates on the detector crop. |
| Inference | `src/rt/infer.py` | Run exported model; return `(label, confidence)`. |
| Smoothing | `src/rt/smoothing.py` | N-frame majority vote, confidence gate, per-action cooldown, idle. |
| Controller | `src/control/controller.py` | Office/Camera **state machine**; gesture event → action. |
| Input | `src/control/input_sim.py` | `pydirectinput` clicks at calibrated coords; kill-switch. |
| HUD | `src/rt/hud.py` | Overlay: raw/smoothed gesture, confidence, mode, last action. |
| Config | `configs/*.yaml` | Model, runtime thresholds, `fnaf_layout.yaml` button coords. |

## A.3 Data flow & contracts

- **Training sample:** `image (H×W×3 uint8)` → **crop to the hand bbox (+15% pad, AD-04)** → `label ∈ working_subset` → transform → `tensor (3×S×S float)`, normalized with the backbone's stats. (`crop_mode: full_frame` skips the crop — see the AD-04 switching note.)
- **Model I/O:** input `tensor (B×3×S×S)` → logits `(B×N)` → softmax → `(label, confidence)`.
- **Gesture event:** emitted only when the smoothing layer confirms a gesture held ≥ N frames above the confidence threshold and the action is off cooldown. Events are **edge-triggered** (door/camera toggles) except lights, which may be **hold-to-activate**.
- **Action:** controller maps `(mode, gesture_event)` → one FNAF action → one or more coordinate clicks via the input layer.

The **critical contract**: preprocessing in `src/rt/preprocess.py` must be **byte-for-byte equivalent** to training transforms (size, interpolation, normalization, crop policy). Drift here is the most common cause of "great test accuracy, useless live." With the two-stage pipeline this extends to the **crop source**: the live MediaPipe bbox must approximate the HaGRID training bbox (same relative tightness and `bbox_pad`), or the classifier is fed a distribution it never trained on. The runtime loads the mode from the checkpoint's saved config (A.5) and **refuses to run if the runtime `crop_mode`/detector setting doesn't match** what the model trained with.

## A.4 Runtime model (threads / timing)
- Single capture→**detect**→infer→act loop, target **≥ 15 FPS** end-to-end (the MediaPipe detector is part of this budget — measure its per-frame share in Phase 4, Day 1).
- If inference stalls the loop, split capture (producer) and inference (consumer) across two threads with a 1-frame queue (drop stale frames — we want *latest*, not *all*).
- Per-action **cooldowns** (≈0.5–1 s) prevent door flapping independent of frame rate.

## A.5 Configuration & reproducibility
- All training runs are **config-driven** (YAML/argparse); the config is saved next to each checkpoint.
- Seeds fixed; environment captured in `DOCS/env.md`.
- Runtime thresholds (confidence, vote window, cooldowns) and `fnaf_layout.yaml` are config, not code, so they can be tuned per session without edits.

## A.6 Repository layout (target)
```
Fnaf_CV/
├── DATA/                  # HaGRID annotations (images downloaded separately, git-ignored)
├── DOCS/                  # proposal, schedule, this file, phases/, AI usage
├── src/
│   ├── data/  models/  rt/  control/
│   ├── train.py  eval.py  export.py
├── configs/               # model + runtime + fnaf_layout.yaml
├── notebooks/             # EDA, experiments
├── models/                # checkpoints + exported artifacts (git-ignored)
├── CLAUDE.md  README.md  requirements.txt
```

---

# Part B — Decision Records

## Data

### AD-01 — Use HaGRID as the gesture dataset · *Accepted*
- **Context:** Need a labeled static-hand-gesture dataset; `DATA/` already contains HaGRID annotations (18 classes, bbox + 21 landmarks).
- **Decision:** Build on **HaGRID**.
- **Rationale:** Already on disk; large and diverse (552k FullHD images, many subjects); class names map cleanly to control actions; well-documented (WACV 2024).
- **Alternatives:** Collect a custom dataset (too slow for 5 weeks); MediaPipe's built-in recognizer (less learning value — see AD-05); other gesture sets (no advantage, not local).
- **Consequences:** Bound to HaGRID's gesture vocabulary; must download images separately (AD-02); a train/serve domain gap vs. my webcam to be closed in Phase 5.

### AD-02 — Subsample for iteration, 512px for final training · *Accepted*
- **Context:** Full HaGRID is **716 GB** (1920px) — infeasible to store/train on the local machine. `DATA/` holds only annotations.
- **Decision:** Use the **subsample** (matches `ann_subsample`) for fast iteration; use the **512px** variant for final training. Download **only the working-subset classes** (AD-03), per-class, not all 18. Never download the full 1920px set.
- **Rationale:** Keeps iteration cycles short; 512px is plenty of resolution for a hand-pose classifier; restricting to 8 of 18 classes at 512px is ~12 GB, well within local storage. Avoids TB-scale storage entirely.
- **Alternatives:** Full dataset (storage/compute blowout); subsample only (may under-train); all 18 classes at 512px (~26 GB, unnecessary for an 8-class control set).
- **Consequences:** Final numbers reported on the 512px subset, documented as such. Download size/paths recorded in `DOCS/data.md`.

### AD-02b — All training runs on the local machine (no cloud) · *Accepted*
- **Context:** Need to decide where fine-tuning runs. The local machine is a Windows 11 laptop with an **NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)**, Core Ultra 7 165H, 32 GB RAM.
- **Decision:** **All training, evaluation, and the live demo run locally on the RTX 4060.** No Colab, no cloud GPU, no cloud storage.
- **Rationale:** 8 GB VRAM comfortably fine-tunes a small `timm` backbone (`mobilenetv3`/`efficientnet_b0`) on an 8-class 512px subset; keeping everything local removes upload/sync friction, cost, and a train/serve environment split. The webcam + game + model all live on one machine anyway, so local is the natural fit.
- **Alternatives:** Colab/cloud GPU fallback (rejected — adds cost, data-transfer overhead, and a second environment to keep in sync for no benefit at this scale).
- **Consequences:** Batch size / input resolution must respect the 8 GB VRAM budget (use mixed precision and gradient accumulation if needed). CUDA/driver/torch versions recorded in `DOCS/env.md`.

### AD-03 — Reduced working class subset (~8 of 18) · *Accepted*
- **Context:** 18 classes include near-duplicates (`peace`/`peace_inverted`, `two_up`/`two_up_inverted`) that confuse a live classifier and exceed the control vocabulary FNAF needs.
- **Decision:** Train/control on a **well-separated subset**: `like, dislike, fist, one, two_up, palm, ok, mute`.
- **Rationale:** Live reliability scales with inter-class separability, not class count; fewer classes → fewer real-time misfires; FNAF needs only a handful of actions.
- **Alternatives:** All 18 (lower live reliability, no benefit); 3–4 classes (not enough actions for doors+lights+camera modes).
- **Consequences:** Confusion matrix in Phase 2 may prompt swapping a class. Final subset is locked at end of Phase 1.

### AD-04 — Two-stage pipeline: detect-and-crop the hand, then classify · *Accepted (supersedes the earlier full-frame-first stance)*
- **Context:** Classify the whole webcam frame, or detect the hand and crop to it first? HaGRID ships a bbox per image offline; a live webcam has none. The Phase-2 frozen baseline settled the direction: `full_frame` `mobilenetv3_large_100` reached only **57% val / 53% test** (random 12.5%), and [data-preparation.md §4](data-preparation.md#4-cropping-ad-04) shows why — the hand occupies <5% of a typical HaGRID frame, so a frozen ImageNet backbone spends its receptive field on background.
- **Decision:** Adopt a **two-stage detect-then-classify pipeline** as the primary approach.
  - **Offline (train):** train the classifier on **bbox crops** — `crop_mode: bbox`, +15% pad — using HaGRID's annotated bbox.
  - **Online (live):** a **hand detector, MediaPipe Hands**, produces a bbox each frame; we crop to it (same pad), then run the `timm` classifier on the crop.
  - The classifier — the ML learning objective — is unchanged; only its **input** changes from full frame to hand crop.
- **Rationale:** Cropping tightens the input distribution so the small backbone sees the hand at full resolution instead of a ~40px blob in a cluttered room — the highest-leverage fix for both the weak baseline and live robustness to background/position/distance. Using an **off-the-shelf** detector (MediaPipe) means **no second model to train**, so the transfer-learning deliverable stays fully intact (AD-05).
- **Alternatives:** *Full-frame classification* — the earlier baseline; simpler, no detector dependency, but demonstrably weak here (kept behind the flag for the Phase-3 A/B and as a fallback). *Single detection-classifier* (YOLO-style, one model outputs box + class) — one pass, but it replaces transfer-learned classification with an object-detection objective, superseding AD-05's learning goal — rejected unless the learning goals change. *Landmark-only classifier* (AD-05) — low learning value.
- **Consequences:** The runtime gains a **MediaPipe dependency and its per-frame latency**, which must fit the ≥15 FPS budget (Phase 4, measured Day 1). Train and runtime crop policy must be **byte-identical** (same pad, same square-vs-aspect handling) or live accuracy collapses — this is now the single most important preprocessing contract (§A.3). The choice is **confirmed by a controlled A/B in Phase 3** (train `full_frame` vs `bbox`, same seed/epochs/split, compare val accuracy + confusion matrix); if `bbox` does not clearly win, revert via the switching note below. Because everything sits behind one `crop_mode` flag plus a runtime detector toggle, reverting is a **config change, not a rewrite**.

#### AD-04 · How to switch between full-frame and two-stage crop

The whole reason this lives behind a flag is that the approach is reversible without touching model code. Switching is **three coordinated settings**, all config:

| Layer | Full-frame (single-stage) | Two-stage crop (**current default**) |
|---|---|---|
| Training input | `configs/data.yaml → input.crop_mode: full_frame` | `input.crop_mode: bbox` (+ `bbox_pad: 0.15`) |
| Runtime preprocess | `src/rt/preprocess.py`: resize the raw frame | MediaPipe → crop bbox (+`bbox_pad`) → resize |
| Runtime toggle | `configs/*.yaml → runtime.use_detector: false` | `runtime.use_detector: true` |

**The rule that makes it safe:** the runtime `crop_mode`/detector setting **must match the loaded checkpoint's training mode** — a crop-trained model fed full frames (or vice-versa) sees an unfamiliar distribution and misbehaves. Each checkpoint saves its training config next to it (§A.5); the runtime reads that and refuses to start in a mismatched mode, so you can't silently pair the wrong two halves.

**To run the A/B (Phase 3):** train two checkpoints identical except `crop_mode`, same seed/epochs/split; compare val accuracy + confusion matrix. Adopt the winner project-wide by flipping the three settings above.

**Going to a *single* model with no detector at all** means either accepting `full_frame`'s accuracy ceiling, or moving to a detection-classifier (YOLO-style) — but the latter is a different training objective (object detection, not transfer-learned classification) and would supersede AD-05, so it's out of scope unless the course's learning goals change.

### AD-16 — 70/15/15 split grouped by user_id (across all images) · *Accepted*
- **Context:** The data needs a train/val/test split. An earlier design kept the subsample images (781, ~9%) as a fixed held-out test set and split only the `train_val` pool. A 70/15/15 target can't be met that way (test is only ~9%), so all 8594 downloaded images are pooled and split together.
- **Decision:** One **70/15/15** split drawn across **all images**, **grouped by `user_id`** (`GroupShuffleSplit`, two-stage) so no subject appears in more than one split. Seeded (`seed 42`), computed at load time — nothing moves on disk. Configurable via `configs/data.yaml → split.{train,val,test,group_by_user}`.
- **Rationale:** HaGRID repeats the same person across many images; a per-image split would leak a subject's hands from train into test and inflate the score. Grouping by user makes **test a genuine unseen-subject estimate** (this is what AD-10 asks for). The 8594 images span 4124 users (median 1 img/user), so grouping barely perturbs class balance.
- **Alternatives:** Keep the fixed subsample test (can't hit 15% test); plain class-stratified per-image split (exact proportions but subject leakage — dishonest test); k-fold CV (heavier, redundant for a 5-week scope).
- **Consequences:** Realized sample proportions are **approximate** (measured train 68.9 / val 14.8 / test 16.3) because whole users stay together — deliberately *not* seed-tuned to fake exactness. The fixed subsample-as-test design (formerly under AD-03) is retired; `download_*.py` now only governs acquisition. Test is reported once (AD-10). Per-class balance verified healthy (val 139–183, test 161–188).

## Model & approach

### AD-05 — Transfer learning with a pretrained CNN (not landmark-only) · *Accepted*
- **Context:** Two viable routes: (a) train an **image classifier** via transfer learning; (b) feed **MediaPipe 21-landmarks** to a tiny classifier.
- **Decision:** **Image classifier via transfer learning** (timm/HF) is the core ML deliverable.
- **Rationale:** This is an "ML Projects" course — the stated learning goal is *end-to-end transfer learning* (freezing, head-swapping, staged fine-tuning). Landmark-only delegates the hard vision to Google's pretrained model and yields a trivial classifier with little learning value.
- **Alternatives:** Landmark + small classifier (faster/robust but low learning value); hybrid MediaPipe-crop → CNN — **now the adopted primary pipeline (AD-04)**.
- **Consequences:** More training work and a real train/serve gap to manage — which is exactly the intended learning. MediaPipe is used as the **runtime hand cropper** (AD-04), **not** the classifier — the transfer-learned CNN remains the ML deliverable.

### AD-06 — `timm` + Hugging Face for backbones · *Accepted*
- **Context:** Need pretrained backbones and an idiomatic fine-tuning workflow.
- **Decision:** Use **`timm`** (with HF Hub), following the referenced timm/HF tutorial.
- **Rationale:** One-line head-swapping via `num_classes`, exposes correct input size/normalization via `data_config`, huge model zoo, clean freeze control. Matches the project's reference tutorial.
- **Alternatives:** `torchvision.models` (smaller zoo, clunkier head-swap); raw HF `transformers` ViT (heavier than needed); build from scratch (no — defeats transfer-learning goal).
- **Consequences:** A `timm` dependency; backbone choice is a config value (AD-07).

### AD-07 — MobileNetV3 backbone · *Accepted*
- **Context:** The model must run in real time alongside the game.
- **Decision:** Use **`mobilenetv3`** as the default backbone (timm). Keep **`resnet18`** as a comparison baseline and **`efficientnet_b0`** as a fallback if the Phase 3 sweep shows MobileNetV3's accuracy ceiling is too low for the latency it buys.
- **Rationale:** Latency budget (≥15 FPS while FNAF runs) favors small nets; a hand-pose classifier over ~8 classes doesn't need a large backbone. MobileNetV3 is the lightest of the candidates and is designed for exactly this real-time-on-modest-hardware case.
- **Alternatives:** `efficientnet_b0` (slightly higher accuracy ceiling, more compute — kept as fallback); ResNet50/ViT (higher ceiling, worse latency — rejected for a twitch game).
- **Consequences:** Backbone is a config value, so swapping to the fallback is a one-line change. Possible small accuracy ceiling traded for responsiveness — the right trade here. Phase 3 sweep now **validates** this choice rather than deciding it.

## Training process

### AD-08 — Frozen backbone baseline, then progressive unfreezing · *Accepted*
- **Context:** Core learning objective: understand freezing/fine-tuning mechanics.
- **Decision:** **Stage A** head-only (frozen backbone) → **Stage B** unfreeze top block(s) with discriminative LR → **Stage C** optional full unfreeze with small LR + warmup.
- **Rationale:** Establishes how much pretrained features already carry, then adapts deeper layers only as needed; discriminative LRs protect general low-level features.
- **Alternatives:** Full fine-tune from the start (over-fitting risk on a subset, less insight); head-only forever (may under-fit).
- **Consequences:** Multiple staged runs to log/compare; each stage tracked separately in TensorBoard.

### AD-09 — Label-aware augmentation (careful with flips) · *Accepted*
- **Context:** Webcam conditions vary (lighting, angle); augmentation improves robustness. But horizontal flip changes the meaning of some gestures.
- **Decision:** Apply color/brightness/contrast jitter, small rotation, random resized crop, slight translation. **Horizontal flip only where label-safe**, or remap labels on flip (`peace`↔`peace_inverted`, `two_up`↔`two_up_inverted`, left/right semantics).
- **Rationale:** Brightness/color aug directly closes the Phase-5 lighting gap; naive flipping would inject wrong labels.
- **Alternatives:** No augmentation (poor live robustness); aggressive aug incl. blind flips (label noise).
- **Consequences:** Augmentation config documents flip policy explicitly.

### AD-10 — Evaluate once on HaGRID's by-user test split · *Accepted*
- **Context:** HaGRID is split by `user_id` so no subject leaks across train/test.
- **Decision:** Tune on **val**, report on the **test split once**; report overall + **per-class** precision/recall/F1 + confusion matrix.
- **Rationale:** Honest generalization estimate without subject leakage; per-class detail reveals which gestures threaten control reliability.
- **Alternatives:** Random split (subject leakage, inflated numbers); repeated test peeking (overfitting to test).
- **Consequences:** Target **≥90%** on the subset (gate, see schedule); failures reported, not hidden.

## Real-time & control

### AD-11 — Export model (TorchScript/ONNX) for inference · *Proposed (Phase 3)*
- **Context:** The runtime loop needs fast, dependency-light inference.
- **Decision:** Export the best model to **TorchScript and/or ONNX**, run via ONNX Runtime / TorchScript, with a **parity smoke test** vs. the PyTorch model. Fallback: plain PyTorch `eval()`.
- **Rationale:** Lower latency and a cleaner runtime; parity test guards against export drift.
- **Alternatives:** Ship raw PyTorch (simpler, possibly slower) — retained as fallback.
- **Consequences:** Extra export step; must verify identical predictions before trusting the artifact.

### AD-12 — Temporal smoothing + debounce + explicit idle · *Accepted*
- **Context:** Per-frame classification "chatters"; a twitch horror game punishes both lag and false inputs.
- **Decision:** **N-frame majority vote** + **confidence threshold** + **per-action cooldown** + explicit **`mute` idle** gesture. Actions edge-triggered; lights may be hold-to-activate.
- **Rationale:** Converts noisy frames into stable intentional events; idle makes "no command" a real signal instead of a misfire; cooldown stops door flapping.
- **Alternatives:** Act on every frame (unusable chatter); a temporal model/LSTM (out of scope — static poses only).
- **Consequences:** A few hundred ms of intentional latency traded for stability; thresholds are tunable config.

### AD-13 — Mode-aware controller (Office vs. Camera) · *Accepted*
- **Context:** FNAF has two interaction contexts (office with doors/lights; camera monitor). Mapping every action to a distinct gesture would need too many classes.
- **Decision:** A **state machine** with **Office** and **Camera** modes; `palm` raises the monitor, `fist` (in Camera) lowers it. Same gesture means different things per mode.
- **Rationale:** Keeps per-state vocabulary small → higher live reliability; mirrors how the game itself is modal.
- **Alternatives:** Flat one-gesture-per-action map (needs more, more-confusable classes); cursor-emulation free-control (imprecise, slow).
- **Consequences:** Players learn two small maps; mode must be shown on the HUD to avoid confusion.

### AD-14 — Drive the real game via `pydirectinput`; test registration early · *Accepted*
- **Context:** FNAF is a DirectX game on Windows. `pyautogui` synthetic clicks are often ignored by DirectX input.
- **Decision:** Drive the **real Steam game** using **`pydirectinput`** (SendInput), clicking **calibrated button coordinates** from `fnaf_layout.yaml`. **Test in-game registration on Phase-4 Day 1**, before building the full controller. Include a global **kill-switch**.
- **Rationale:** Most compelling demo; `pydirectinput` is the known-good path for DirectX; front-loading the test de-risks the project's biggest integration unknown.
- **Alternatives:** Build a Pygame FNAF clone (no game ownership needed, but extra build scope and a less impressive demo); `pyautogui` (unreliable in DirectX).
- **Consequences:** Tied to a known window position/resolution; coordinates re-calibrated per setup; safety kill-switch mandatory whenever simulation runs.

## Scope

### AD-15 — Static poses only, FNAF 1 only, solo · *Accepted*
- **Context:** Five-week solo project; must be achievable while still exercising the full pipeline.
- **Decision:** **Static gestures only** (no temporal models), **FNAF 1 only**, single player.
- **Rationale:** Static poses fit the image-classifier approach and a reliable 5-week demo; dynamic gestures (LSTM/sliding window) and multi-game support are large additions with little extra learning on the core objectives.
- **Alternatives:** Add dynamic gestures / later nights / a second game — all deferred to "future work."
- **Consequences:** Clear, defensible boundary; listed under Out of Scope in [schedule.md](schedule.md).

---

## Open questions (to resolve as phases land)
- **AD-04** resolved: **two-stage detect-and-crop** adopted as primary (the 57% `full_frame` Phase-2 baseline motivated it); the Phase-3 A/B confirms `bbox` vs `full_frame` before locking it. Open sub-question: does MediaPipe's latency fit the ≥15 FPS budget (Phase 4, Day 1)?
- **AD-07** backbone decided: **MobileNetV3** (Accepted). Phase 3 sweep validates it vs. `resnet18` baseline and `efficientnet_b0` fallback.
- **AD-11** TorchScript vs. ONNX — pick whichever hits the latency budget with clean parity.
- Whether a **small self-captured fine-tune** (my hands, my room) is needed to close the train/serve gap (Phase 5).

## Change log
| Date | Change |
|---|---|
| 2026-06-30 | Initial architecture + AD-01…AD-15 recorded at planning stage. |
| 2026-06-30 | AD-07 accepted: **MobileNetV3** selected as the backbone (resnet18 baseline, efficientnet_b0 fallback). |
| 2026-06-30 | AD-02b accepted: **all training/eval/demo run locally on the RTX 4060 (8 GB) — no cloud/Colab.** Removed Colab fallback from proposal.md and schedule.md; AD-02 scoped downloads to the 8 working classes at 512px (~12 GB). |
| 2026-07-08 | AD-16 accepted: **70/15/15 split grouped by `user_id`** across all images (no subject leakage); retired the fixed subsample-as-test design. Data-prep pipeline built under `src/data/` + `configs/data.yaml`; documented in [data-preparation.md](data-preparation.md). |
| 2026-07-12 | **AD-04 accepted as a two-stage detect-then-classify pipeline** (MediaPipe hand crop → `timm` classifier), superseding the earlier full-frame-first stance — motivated by the 57% `full_frame` Phase-2 baseline. Default `crop_mode` flipped to `bbox` in `configs/data.yaml`; added the AD-04 switching how-to; updated Part A (system overview, components, contracts, runtime budget), AD-05, and phases 1/3/4 + overview. Full-frame kept behind the flag for the Phase-3 A/B and as a fallback. |
