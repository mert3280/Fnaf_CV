# Architecture & Decisions

**Project:** Hands-Free *Five Nights at Freddy's* (motion-tracked cursor + palm/fist click)
**Author:** Ted Roper · **Status:** Building · **Last updated:** 2026-07-14

This document records **how the system is built** (Part A — Architecture) and **why each major choice was made** (Part B — Decision Records). It is the single source of truth for technical direction; the current roadmap lives in [plan.md](plan.md) (the original phase plans are preserved in [../legacy/phases/](../legacy/phases/)).

> **2026-07-14 scope pivot (AD-17):** control changed from a *gesture vocabulary* (8 gestures → mapped FNAF actions) to a *cursor*: MediaPipe hand tracking moves the mouse pointer; a binary palm/fist classifier fires clicks. Part A below reflects the new design; superseded decision records are kept and marked.

> **Conventions:** Decision records (ADRs) use *Context → Decision → Rationale → Alternatives → Consequences*. Each is numbered `AD-NN` and can be referenced from code/PRs. "Status" of an AD is **Accepted** unless marked *Proposed* (will be confirmed during the phase that implements it) or *Superseded*.

---

# Part A — Architecture

## A.1 System overview

The system is a **real-time perception → decision → actuation loop**. A webcam streams frames; a **hand detector (MediaPipe) finds the hand** in each frame and does two jobs with it: its **position drives the mouse cursor** (absolute frame→screen mapping, AD-19), and its **bbox crops the hand** for a CNN that makes a **binary palm/fist** call (AD-18); a debounced click state machine turns confirmed palm→fist transitions into **single mouse clicks** (AD-20); an input-simulation layer issues the cursor moves and clicks to the real game. The **two-stage detect-then-classify** backbone of this design is AD-04; the cursor-control pivot is AD-17.

```
                         ┌──────────────────────────── OFFLINE ─────────────────────────────────┐
   HaGRID annotations ─▶ Data pipeline ─▶ timm backbone + 2-class head ─▶ fine-tune ─▶ export
   (palm + fist only,    (parse/crop-to-  (transfer learning)             (TorchScript/ONNX)  │
    AD-18)                hand bbox)                                                          │ model artifact
                         ┌──────────────────────────── ONLINE ──────────────────────────────┼──────────┐
                         ▼                                                                   ▼          │
   Webcam ─▶ Capture ─▶ Detect (MediaPipe) ─┬▶ hand position ─▶ Cursor mapper ─────────────▶ Input ─▶ FNAF
   (OpenCV)  (frame)                        │                  (mirror, control box,   sim: move  (Steam,
                                            │                   EMA smoothing)          + click   pydirect-
                                            └▶ crop ─▶ Preprocess ─▶ Inference ─▶ Click FSM ──▶┘   input)
                                                       (resize/norm)  (palm/fist)  (K-frame debounce,
                                                          │                        edge-trigger, cooldown)
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
| Inference | `src/rt/infer.py` | Run exported model; return `(label, confidence)` over `{palm, fist}`. |
| Cursor mapper | `src/rt/cursor.py` *(planned)* | Palm-center anchor → mirror → control box → EMA smoothing + dead-zone → screen coords; **freezes on no-hand** (AD-19). |
| Click FSM | `src/control/click_fsm.py` *(planned)* | K-frame debounce, **edge-triggered single click** on palm→fist, cooldown, re-arm on confirmed palm (AD-20). |
| Input | `src/control/input_sim.py` | `pydirectinput` cursor moves + clicks at the live cursor position; kill-switch. |
| HUD | `src/rt/hud.py` | Overlay: raw/debounced label, confidence, FSM state, cursor target, last click. |
| Config | `configs/*.yaml` | Model, runtime thresholds (confidence, K, cooldown), cursor-mapping params (control box, alpha, dead-zone). |

## A.3 Data flow & contracts

- **Training sample:** `image (H×W×3 uint8)` → **crop to the hand bbox (+15% pad, AD-04)** → `label ∈ {palm, fist}` (AD-18) → transform → `tensor (3×S×S float)`, normalized with the backbone's stats. (`crop_mode: full_frame` skips the crop — see the AD-04 switching note.)
- **Model I/O:** input `tensor (B×3×S×S)` → logits `(B×2)` → softmax → `(label, confidence)`.
- **Cursor sample (every frame):** detector hand position → palm-center anchor → mirror → control-box map → EMA + dead-zone → screen `(x, y)` → input layer moves the cursor. **No hand → cursor holds position** (never teleports) and the click FSM disarms (AD-19).
- **Click event:** emitted only when the click FSM confirms **K consecutive fist frames** above the confidence threshold from an armed (confirmed-palm) state, off cooldown. **Edge-triggered, exactly one click per palm→fist transition**; re-arming requires K confirmed palm frames (AD-20).
- **Action:** a click event → one `pydirectinput` click at the **current cursor position**. No gesture→action mapping and no per-button coordinates — the game's own UI does the interpreting.

The **critical contract**: preprocessing in `src/rt/preprocess.py` must be **byte-for-byte equivalent** to training transforms (size, interpolation, normalization, crop policy). Drift here is the most common cause of "great test accuracy, useless live." With the two-stage pipeline this extends to the **crop source**: the live MediaPipe bbox must approximate the HaGRID training bbox (same relative tightness and `bbox_pad`), or the classifier is fed a distribution it never trained on. The runtime loads the mode from the checkpoint's saved config (A.5) and **refuses to run if the runtime `crop_mode`/detector setting doesn't match** what the model trained with.

## A.4 Runtime model (threads / timing)
- Single capture→**detect**→infer→act loop, target **≥ 15 FPS** end-to-end (the MediaPipe detector is part of this budget — measure its per-frame share in Phase 4, Day 1).
- If inference stalls the loop, split capture (producer) and inference (consumer) across two threads with a 1-frame queue (drop stale frames — we want *latest*, not *all*).
- A post-click **cooldown** (≈0.3 s, config) backs up the FSM's edge-triggering so classifier flicker mid-transition can never double-click, independent of frame rate.
- The **cursor mapper runs every frame regardless of classifier state** — pointing must stay fluid even while a click is debouncing.

## A.5 Configuration & reproducibility
- All training runs are **config-driven** (YAML/argparse); the config is saved next to each checkpoint.
- Seeds fixed; environment captured in `DOCS/env.md`.
- Runtime thresholds (confidence, debounce K, cooldown) and cursor-mapping parameters (control box, smoothing alpha, dead-zone) are config, not code, so they can be tuned per session without edits. (`fnaf_layout.yaml` button coordinates are retired with AD-13 — clicks land wherever the cursor is.)

## A.6 Repository layout (target)
```
Fnaf_CV/
├── DATA/                  # HaGRID annotations (images downloaded separately, git-ignored)
├── DOCS/                  # build/ (this file, plan, strategies, data), models/, AI/, legacy/, class-related/
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

### AD-03 — Reduced working class subset (~8 of 18) · *Superseded by AD-18 (2026-07-14)*

> The 8-class subset below was the working set through Phases 1–2 and the Strategy-1/2 models. The cursor-control pivot (AD-17) trims it to **`palm` + `fist` only** — see AD-18. Kept for the record.
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

### AD-12 — Temporal smoothing + debounce + explicit idle · *Accepted, mechanism simplified by AD-20 (2026-07-14)*

> The principle (never act on a single noisy frame) survives the pivot unchanged; with only two labels and one action, the N-frame vote + `mute` idle collapses into the simpler K-consecutive-frame click FSM of AD-20. "No hand detected" replaces `mute` as the explicit idle signal.
- **Context:** Per-frame classification "chatters"; a twitch horror game punishes both lag and false inputs.
- **Decision:** **N-frame majority vote** + **confidence threshold** + **per-action cooldown** + explicit **`mute` idle** gesture. Actions edge-triggered; lights may be hold-to-activate.
- **Rationale:** Converts noisy frames into stable intentional events; idle makes "no command" a real signal instead of a misfire; cooldown stops door flapping.
- **Alternatives:** Act on every frame (unusable chatter); a temporal model/LSTM (out of scope — static poses only).
- **Consequences:** A few hundred ms of intentional latency traded for stability; thresholds are tunable config.

### AD-13 — Mode-aware controller (Office vs. Camera) · *Superseded by AD-17 (2026-07-14, never built)*

> Retired before implementation. A cursor needs no modes — the game's own UI disambiguates every click — so the state machine, the per-mode gesture maps, and the HUD mode indicator all disappear. Note the irony recorded in AD-13's own alternatives row: "cursor-emulation free-control" was originally rejected as imprecise; MediaPipe's live tracking in Strategy 2 proved steadier than assumed, which is part of the AD-17 rationale.
- **Context:** FNAF has two interaction contexts (office with doors/lights; camera monitor). Mapping every action to a distinct gesture would need too many classes.
- **Decision:** A **state machine** with **Office** and **Camera** modes; `palm` raises the monitor, `fist` (in Camera) lowers it. Same gesture means different things per mode.
- **Rationale:** Keeps per-state vocabulary small → higher live reliability; mirrors how the game itself is modal.
- **Alternatives:** Flat one-gesture-per-action map (needs more, more-confusable classes); cursor-emulation free-control (imprecise, slow).
- **Consequences:** Players learn two small maps; mode must be shown on the HUD to avoid confusion.

### AD-14 — Drive the real game via `pydirectinput`; test registration early · *Accepted, amended by AD-17 (2026-07-14)*

> Amendment: the input layer now also issues **cursor movement** every frame, not just clicks — and clicks land at the live cursor position instead of pre-calibrated `fnaf_layout.yaml` coordinates (that file is retired with AD-13). The early-registration test gains a second check: does FNAF track `pydirectinput` *movement* smoothly, not just clicks? The kill-switch requirement is unchanged and now also freezes cursor motion.
- **Context:** FNAF is a DirectX game on Windows. `pyautogui` synthetic clicks are often ignored by DirectX input.
- **Decision:** Drive the **real Steam game** using **`pydirectinput`** (SendInput), clicking **calibrated button coordinates** from `fnaf_layout.yaml`. **Test in-game registration on Phase-4 Day 1**, before building the full controller. Include a global **kill-switch**.
- **Rationale:** Most compelling demo; `pydirectinput` is the known-good path for DirectX; front-loading the test de-risks the project's biggest integration unknown.
- **Alternatives:** Build a Pygame FNAF clone (no game ownership needed, but extra build scope and a less impressive demo); `pyautogui` (unreliable in DirectX).
- **Consequences:** Tied to a known window position/resolution; coordinates re-calibrated per setup; safety kill-switch mandatory whenever simulation runs.

## Scope

### AD-15 — Static poses only, FNAF 1 only, solo · *Accepted, clarified by AD-17*

> Still holds after the pivot: the **classifier** sees only static poses (palm, fist). Tracking the hand's *position* over time for the cursor is geometry from MediaPipe's per-frame output — it involves no temporal model, so the "no dynamic gestures / no LSTM" boundary is intact.
- **Context:** Five-week solo project; must be achievable while still exercising the full pipeline.
- **Decision:** **Static gestures only** (no temporal models), **FNAF 1 only**, single player.
- **Rationale:** Static poses fit the image-classifier approach and a reliable 5-week demo; dynamic gestures (LSTM/sliding window) and multi-game support are large additions with little extra learning on the core objectives.
- **Alternatives:** Add dynamic gestures / later nights / a second game — all deferred to "future work."
- **Consequences:** Clear, defensible boundary; originally listed under Out of Scope in the (now legacy) [schedule.md](../legacy/phases/schedule.md).

## Scope pivot — cursor control (2026-07-14)

### AD-17 — Pivot to cursor control: motion-tracked cursor + binary click classifier · *Accepted*
- **Context:** The original design (AD-03/AD-13) mapped 8 gestures to discrete FNAF actions through a mode-aware controller. Strategy 2's live tests showed the classifier reads gestures well but loses accuracy to inter-class confusion and hand-distance effects — and every misread would be a *wrong in-game action*. Meanwhile FNAF is entirely mouse-driven: every control is a hover or a click.
- **Decision:** Replace the gesture vocabulary with **direct cursor control**. Stage 1 (MediaPipe, already in the pipeline as the cropper) additionally emits the hand position, which **moves the mouse cursor** (AD-19). Stage 2 becomes a **binary palm/fist classifier** — palm = no click, **fist = click** (AD-18, AD-20). The player points at what they want and squeezes to click.
- **Rationale:** (1) One universal interaction covers *all* of FNAF with no mapping, no modes, no memorization. (2) Model risk concentrates on the two most visually distinct poses in HaGRID — most confusion pairs vanish, and a misread costs one click, not a wrong action. (3) The transfer-learning deliverable (AD-05) is untouched: same backbone, same freeze/unfreeze mechanics, only `num_classes` changes. (4) Stage 1's tracking output was already computed per frame — the cursor costs no new model or latency.
- **Alternatives:** Stay the course with 2.1's hardening (kept as fallback — the 8-class checkpoints still run); landmark-heuristic click detection, e.g. fingers-curled ratio (no learning value, AD-05); dwell-to-click, hover N seconds (no classifier at all, and slow for a twitch game).
- **Consequences:** AD-03 → superseded by AD-18; AD-13 retired unbuilt; AD-12's mechanism simplifies into the click FSM; AD-14 gains cursor movement; `fnaf_layout.yaml` calibration is no longer needed. New risks — pointing precision, arm fatigue, palm↔fist reliability at arm's-length — are tracked in [Strategy 3](strategies/3-cursor-and-click/03-cursor-and-click.md). The old plan is preserved in [DOCS/legacy/](../legacy/README.md).

### AD-18 — Trim training classes to `palm` + `fist` · *Accepted*
- **Context:** The click decision needs exactly two states: no-click (open hand) and click (closed hand). The 8-class set exists only to feed the retired gesture→action map.
- **Decision:** `configs/data.yaml → classes: [palm, fist]` (`palm=0, fist=1` — order fixes `label_idx`; the head must be retrained). Retrain the same recipe as `bbox_frozen_mnv3_large` with a fresh 2-class head, frozen backbone first (AD-08 unchanged), then apply Strategy-2.1 levers as needed.
- **Rationale:** `palm` (all fingers extended) vs `fist` (all curled) is the maximal-separation pair in the set. Both classes are already downloaded (palm 1,082 / fist 1,097 ≈ 2,179 images) — no new acquisition. ~2.2k images over 2 classes is a comfortable transfer-learning regime for a frozen MobileNetV3.
- **Alternatives:** Keep 8 classes and use only palm/fist live (wastes capacity on distinctions that no longer matter, and the softmax spreads probability over dead classes); add a third "neither" class (worth revisiting if palm/fist confidence proves poorly calibrated on non-gesture hands — the FSM's confidence gate covers this for now).
- **Consequences:** The split (AD-16) recomputes over ~2,179 images — still grouped by user, still 70/15/15. Binary random is 50%, so headline accuracy will read inflated vs. the 8-class era; the honest live metric is **transition reliability at cursor distances** (Strategy 3). AD-09's flip-safety holds trivially (palm/fist are both mirror-symmetric in meaning). Old 8-class checkpoints remain runnable via their saved configs.

### AD-19 — Absolute frame-to-screen cursor mapping · *Accepted, amended 2026-07-15 (Strategy 3.1)*

> Amendment from the first live test: the anchor is computed from **unclipped** landmark coordinates and is a **pure position measured from the frame's top-left** — hand/box size must never enter the mapping. (The 3.0 anchor used frame-clipped landmarks, which biased it inward for close/large hands — a distance-dependent cursor error.) The anchor *point* is selectable live: palm plate (default, stable through the click squeeze) or landmark-hull center — see [Strategy 3.1](strategies/3-cursor-and-click/03.1-distance-invariant-cursor.md).
>
> Second amendment (3.1.1, same day): the second live test showed position-invariance wasn't the invariance that matters — cursor **gain** still scaled with distance. Default mapping is now the **adaptive control box** (box width = gain × palm span): the same physical arm motion moves the cursor the same amount at any distance, trading away fixed frame-position→screen-position correspondence (the two invariances are mutually exclusive). `--box-mode fixed` restores the pure absolute box — see [Strategy 3.1.1](strategies/3-cursor-and-click/03.1.1-live-robustness-fixes.md).
- **Context:** The hand position must become a screen coordinate. Two families: **absolute** (frame position maps to screen position, like a touchscreen) and **relative/joystick** (offset from a home zone sets cursor velocity).
- **Decision:** **Absolute mapping**: palm-center anchor (mean of MediaPipe landmarks `0, 5, 9, 13, 17` — stable across palm↔fist, unlike fingertips or the bbox center) → mirrored x → a central **control box** (~60%×55% of frame, config) mapped to the full screen, clamped at edges → **EMA smoothing** (`alpha ≈ 0.35`) with a small dead-zone. **No hand → cursor freezes** in place; on re-detection the smoothed position is re-seeded so the cursor never teleports.
- **Rationale:** Absolute is the intuitive "point at the thing" model, needs no home-position calibration each session, and FNAF's targets are large enough that touchscreen-style pointing should suffice. The control box keeps the whole screen reachable inside the camera's comfortable FOV and keeps frame-edge detection jitter off the cursor.
- **Alternatives:** Relative/joystick (steadier for tiny targets, but slower to traverse and harder to point intuitively) — **kept as the documented fallback** if absolute pointing can't hit FNAF's smallest controls after tuning; One-Euro filter instead of EMA (adopt if EMA's lag/jitter trade-off disappoints — drop-in swap).
- **Consequences:** Pointing precision is now a first-class live metric. Control-box size, alpha, and dead-zone are session-tunable config (§A.5). The demo's mirror setting and the mapper's mirror **must agree** or the cursor moves backwards.

### AD-20 — Single click per fist, edge-triggered · *Accepted, amended 2026-07-15 (Strategy 3.1.1)*

> Amendment: ambiguous frames (low confidence / brief dropout) no longer hard-disarm — the FSM tolerates up to a **grace window** (default 10 frames) of them, because hard-disarm made a *slow* palm→fist structurally unclickable (the in-between poses read as ambiguous and cancelled the armed squeeze). Firing still requires K confident fist frames and re-arm still requires K confident palms; sustained absence past grace still disarms — see [Strategy 3.1.1](strategies/3-cursor-and-click/03.1.1-live-robustness-fixes.md).
- **Context:** The binary label stream must become mouse-button events. Candidate semantics: hold-while-fist (button down while fist held — supports drag) vs. single-click-per-fist.
- **Decision:** **Edge-triggered single click.** A three-state FSM: ARMED → (K consecutive fist frames ≥ confidence threshold) → fire **one** click at the current cursor position → re-arm only after K consecutive confident palm frames. No-hand or low confidence from any state → DISARMED (no click possible until a confirmed palm). Post-click cooldown (~0.3 s) as a backstop. Starting points: `K = 3`, tuned live by Ted.
- **Rationale:** FNAF has no drag or hold interaction — every control is a discrete click — so hold-while-fist buys nothing and adds a failure mode (classifier flicker mid-hold chattering the button). Edge-triggering plus palm-re-arm makes double-fires structurally impossible rather than merely debounced.
- **Alternatives:** Hold-while-fist (revisit only if a future target needs drag); click-on-release, fire on fist→palm (feels laggy — the click lands when you let go).
- **Consequences:** Click latency is deliberately `K/FPS` (~0.1–0.2 s) — the AD-12 trade of latency for stability, in simpler form. The FSM state belongs on the HUD (ARMED/FIRED/DISARMED) so misfires can be diagnosed by eye.

### AD-21 — Reframe the click classifier as `fist` vs. `not_fist` · *Accepted (2026-07-20, Strategy 3.2)*
- **Context:** The AD-18 model has exactly two outputs, trained *only* on `palm` and `fist` crops. Live, when the hand does anything else — pointing, an "ok" sign, a relaxed half-curl, the in-between poses of a slow squeeze — the softmax is **forced** onto palm or fist. The model never saw those poses, so it can land confidently on `fist` and fire an **accidental click**. The classifier has no way to say "this isn't a fist."
- **Decision:** Keep the head binary but redefine the negative class as **`not_fist` (idx 0) = no-click**, trained on a **diverse set of non-fist gestures** (`palm, one, two_up, like, dislike, mute`), against **`fist` (idx 1) = click**. "If it isn't a confident fist, it's no-click" is now learned by the model, not left undefined. Mechanised by `label_groups` in the data config (many gesture folders → one target label; the bbox is still the folder's own hand) plus `balance: true` (downsample `not_fist` to the `fist` count, drawn evenly across its six sources, seeded). Same recipe as AD-18 otherwise (`configs/fistvsrest_frozen.yaml`: mobilenetv3_large_100, frozen linear-probe, seed 42) so it's a clean A/B.
- **Rationale:** The failure mode we actually care about is a *false click*, and it comes from out-of-distribution poses, not palm↔fist confusion. Teaching the negative class to span many hand shapes makes "unknown → no-click" the model's default, structurally reducing false clicks instead of relying only on the FSM's confidence gate. `fist` is the one pose that must be crisp; everything else collapsing to no-click is exactly the interaction we want. Costs no new download (all six negatives + `fist` were already on disk) and no runtime model change (the checkpoint is self-describing; the FSM already takes the no-click label as a parameter).
- **Alternatives:** Keep palm/fist and lean on the FSM confidence threshold alone (what AD-18 does — the "add a third class" note in AD-18 is essentially this decision, now taken); a dedicated `no_gesture`/"neither" third class (HaGRID has `no_gesture` bboxes, but a 2-class fist/not-fist is simpler and the negative diversity already covers it); an explicit open-set/OOD score (heavier, and unnecessary if the negative class is broad enough).
- **Consequences:** New split recomputes over ~2,194 balanced images (still grouped by user, still 70/15/15). `ok` is **held out of training** as an unseen gesture for the honest open-set check — *does an untrained pose read as `not_fist`?* — the real metric for "no accidental clicks", reported alongside per-class P/R/F1. The negative label is `not_fist`, so `play.py`/`webcam_demo.py` derive the FSM's no-click label from the checkpoint (works for both palm/fist and not_fist/fist). AD-18's palm/fist model and config stay intact and reproducible for the A/B. Data layer gained `label_groups` + `balance` (backward-compatible: absent ⇒ the old folder-is-label behavior). Full write-up: [Strategy 3.2](strategies/3-cursor-and-click/03.2-fist-vs-rest.md).

---

## Open questions (to resolve as work lands)
- **AD-19 precision:** can the smoothed absolute cursor reliably hit FNAF's smallest controls (camera-flip strip, door buttons)? If not, escalate: bigger control box → stronger smoothing/One-Euro → relative-mapping fallback.
- **AD-18 live reliability:** does palm↔fist read cleanly with the arm extended toward the camera — the distance regime where Strategy 2 degraded? Mitigation path: 2.1 levers + self-capture at cursor distances.
- **Latency budget:** does MediaPipe + cursor mapper + click FSM fit ≥15 FPS end-to-end (carried from AD-04; measure on day 1 of runtime work)?
- **AD-11** TorchScript vs. ONNX — pick whichever hits the latency budget with clean parity.
- Whether a **small self-captured fine-tune** (my hands, my room) is needed to close the train/serve gap — more likely *yes* now, since cursor use lives exactly in the arm-extended regime HaGRID lacks.

## Change log
| Date | Change |
|---|---|
| 2026-06-30 | Initial architecture + AD-01…AD-15 recorded at planning stage. |
| 2026-06-30 | AD-07 accepted: **MobileNetV3** selected as the backbone (resnet18 baseline, efficientnet_b0 fallback). |
| 2026-06-30 | AD-02b accepted: **all training/eval/demo run locally on the RTX 4060 (8 GB) — no cloud/Colab.** Removed Colab fallback from proposal.md and schedule.md; AD-02 scoped downloads to the 8 working classes at 512px (~12 GB). |
| 2026-07-08 | AD-16 accepted: **70/15/15 split grouped by `user_id`** across all images (no subject leakage); retired the fixed subsample-as-test design. Data-prep pipeline built under `src/data/` + `configs/data.yaml`; documented in [data-preparation.md](data-preparation.md). |
| 2026-07-12 | **AD-04 accepted as a two-stage detect-then-classify pipeline** (MediaPipe hand crop → `timm` classifier), superseding the earlier full-frame-first stance — motivated by the 57% `full_frame` Phase-2 baseline. Default `crop_mode` flipped to `bbox` in `configs/data.yaml`; added the AD-04 switching how-to; updated Part A (system overview, components, contracts, runtime budget), AD-05, and phases 1/3/4 + overview. Full-frame kept behind the flag for the Phase-3 A/B and as a fallback. |
| 2026-07-14 | **Scope pivot to cursor control — AD-17…AD-20 accepted** (motion-tracked cursor via MediaPipe, classes trimmed to `palm`/`fist`, absolute mapping, edge-triggered single click). AD-03 superseded by AD-18; AD-13 retired unbuilt; AD-12 mechanism simplified; AD-14 amended (cursor movement, no `fnaf_layout.yaml`); AD-15 clarified. Part A rewritten around the cursor pipeline. Old plan docs (proposal, phases) moved to [DOCS/legacy/](../legacy/README.md); new roadmap in [plan.md](plan.md); design spec in [Strategy 3](strategies/3-cursor-and-click/03-cursor-and-click.md). |
| 2026-07-15 | **Strategy 3 built end-to-end**: classes trimmed (AD-18), binary model trained (**0.9815 test** — [record](../models/palmfist_frozen_mnv3_large/README.md)), cursor mapper / click FSM / input layer / play loop implemented. **AD-19 amended after the first live test (Strategy 3.1)**: anchor recalibrated to a size-invariant position from unclipped landmarks — the clipped-landmark anchor was distance-biased. |
| 2026-07-15 | **Strategy 3.1.1 after the second live test**: AD-19 amended again — default mapping is the **adaptive control box** (constant physical gain across distance); AD-20 amended — click FSM gains a **grace window** (slow squeezes were structurally unclickable under hard-disarm); MediaPipe detection defaults lowered 0.5 → 0.3. Revert levers: `--box-mode fixed`, `--fsm-grace 0`, `--detect-confidence 0.5`. |
| 2026-07-20 | **AD-21 accepted — Strategy 3.2 (`fist` vs. `not_fist`)**: the binary click classifier's negative class is redefined from `palm` to **`not_fist`**, trained on six diverse non-fist gestures so unknown poses resolve to no-click by construction (fewer accidental clicks). Data layer gained `label_groups` + `balance` (backward-compatible); new `configs/{data_fistvsrest,fistvsrest_frozen}.yaml`; `ok` held out for an open-set check; `play.py`/`webcam_demo.py` derive the FSM no-click label from the checkpoint. AD-18's palm/fist model kept intact for the A/B. |
