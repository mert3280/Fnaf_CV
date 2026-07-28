# Daily Updates

> Merged 2026-07-10 from the repo-root `daily-update.md` (technical log) and this file (reflection/class journal) into one dated log.

## 2026-07-01

- Merged all branches (`Docs`, `data`, `playController`) into `main`; branches kept.
- Created `.gitignore` with `DATA/` and untracked all annotation JSON files from git.
- Purged `DATA/` from entire git history using `git-filter-repo` to reduce repo size.
- Force-pushed cleaned history to `origin/main`.
- Added daily-update logging rule to `CLAUDE.md`.
### 1-1 check in
- We discussed. the plan and overview of the project, data, model, and interfacing with the game
- Feed back I recieved. it was a pretty solid plan, there are a few things to consider about playing the game and potential data biases. also we taked about how the stakeholder could be many people, specifically any one with dirty hands who cant use a keyboard right then and there.
- Action items. I would like to add a "blank" class where the model wont do anything if the screen is read as blank.
- Reflection. I think this will be a very cool project and its awesome that its something I really enjoy and greatly look forward to working on.

## 2026-07-02
### presentation reflection
- I presented my project to the class and I think I did really well. I was able to take all the questions and give quick and accurate answers which felt really good. I think this project will be really fun and I feel like I have a very good understanding of it as of now.

## 2026-07-06

**Focus:** Built the training/validation image set and locked in the train/val/test split.

### Done
- Added `src/data/download_train_val.py` — pulls 1000 images/class from `ann_train_val` via HTTP range requests (`remotezip`), parallelized across 8 connections/class, resume-aware.
- **Defined the split:** the already-downloaded subsample (781 imgs, `DATA/images/subsample/`) is now the held-out **test set**; the new pull (7813 imgs, `DATA/images/train_val/`) is the **train + val** pool.
- **Prevented a test leak:** 92–96% of each class's subsample UUIDs are drawn from `train_val`, so the script excludes all `ann_subsample` UUIDs from the candidate pool. Selection is deterministic (`--seed 42`).
- **Duplicate verification:** script MD5-hashes every train_val + test image after download; result was **0 test leaks, 0 intra-set duplicates** across all 8 classes.
- Downloaded 7813/8000 (97.7%); 187 UUIDs missing-in-zip (same annotation/release version drift already seen with the subsample). Per-class 923–997.
- Updated `DOCS/data.md`: new "Train/val/test split" section (exclusion rationale + 3 dedup guarantees), updated variant/path tables, per-class counts, change log.

### Files changed
- `src/data/download_train_val.py` (new)
- `DOCS/data.md`
- `DATA/images/train_val/` (git-ignored images)

### Decisions
- Test images kept in existing `DATA/images/subsample/` folder (not renamed to `test/`) to avoid breaking `download_subsample.py` resume logic; docs make the "subsample = test set" mapping explicit.
- Reused the document-defined subsample as the test split (fixed/reproducible) rather than doing a random hold-out.

## 2026-07-08

**Focus:** Built the offline data-prep pipeline (disk → normalized `(tensor, label_idx)` batches) and documented it.

### Done
- Added `src/data/` pipeline, config-driven from new `configs/data.yaml`:
  - `config.py` — typed `DataConfig` loader.
  - `hagrid_annotations.py` — parse/validate HaGRID JSON, `build_index()` joins on-disk images ⨝ annotations into one DataFrame (`uuid, label, label_idx, path, bbox, has_ann`).
  - `splits.py` — **70/15/15 split grouped by `user_id`** (`GroupShuffleSplit`, two-stage) across all images; asserts no subject/UUID leaks any split; `write_manifest()`.
  - `transforms.py` — train (RRC + label-aware aug, AD-09) / eval (resize+center-crop) pipelines; timm `data_config` drives size+normalization when a backbone is given.
  - `dataset.py` — `HagridDataset` (full_frame or bbox-crop, AD-04) + `build_dataloaders()` factory; `build_full_index()` pools both image folders.
- **Chose split method (asked Ted):** picked **70/15/15 grouped by user** over a plain random split — the images span 4124 users and grouping keeps a person out of two splits (honest test, AD-10).
- **Ran it:** validated annotations (6 benign `no_gesture`-only records, none downloaded), indexed all 8594 images, split → train 5919 / val 1273 / test 1402, confirmed **0 user overlap** across splits + determinism; `HagridDataset` loads real images in both crop modes. (torchvision/timm transform path is standard API — not exercised in this CPU-only env.)
- Wrote `DOCS/data-preparation.md` (full spec: index, split %, crop, normalize, augment, config ref, reproducibility); updated `DOCS/data.md`; added **AD-16** for the split.

### Files changed
- `configs/data.yaml` (new); `src/data/{__init__,config,hagrid_annotations,splits,transforms,dataset}.py` (new)
- `DOCS/data-preparation.md` (new); `DOCS/data.md`; `DOCS/architecture-and-decisions.md`

### Decisions
- **70/15/15 grouped by `user_id` (AD-16, Accepted)** — retired the earlier fixed subsample-as-test design; both folders now pooled and re-split. Realized 68.9/14.8/16.3 (approximate because whole users stay together; deliberately not seed-tuned). Test set is the honest generalization number (AD-10).
- Index drives off files-on-disk (not annotations), since downloads are a subset; images with no matching record are kept for full-frame (bbox-crop falls back to full frame).
- Normalization is backbone-authoritative (timm `data_config`) with ImageNet config defaults as fallback.

### Baseline model (Phase 2, AD-08 Stage A)
- Installed the intended stack (`torchvision 0.25`, `timm 1.0.27`, tensorboard) and populated the empty `requirements.txt`.
- Added `src/models/build.py` (timm backbone + fresh head + freeze control + `pooled_features`) and `src/train.py` (config-driven; frozen-feature linear-probe fast path **and** standard end-to-end loop), config `configs/baseline.yaml`.
- Trained the **full intended model** — `mobilenetv3_large_100`, pretrained, **backbone frozen** (trainable 10,248 / 4.21M params = 0.24%), head-only, no augmentation, full-frame.
- **Result (honest, by-user test):** best val **56.95%**, held-out test **53.0%** (random 12.5% → ~4.2×). Report + confusion matrix in `DOCS/results.md`; checkpoint `models/baseline_mnv3_large/best.pt` (git-ignored).
- **Observations (for Ted to interpret — not decided):** clear train/val gap (train ~85% vs val ~57%); strong classes `mute`/`dislike`, weak `one`↔`two_up` and `palm`↔`ok` confusions. Likely levers = **bbox crop (AD-04)** since the hand is tiny in a 1920px full frame, plus **fine-tuning/unfreezing (Phase 3)** and augmentation (AD-09).
- **CPU-env notes:** this box is CPU-only; JPEG decode (~13 img/s) dominated. `num_workers>0` was *slower* (multiproc IPC/oversubscription on Windows) so the baseline used `num_workers=0`; the frozen+no-aug case caches backbone features once then trains the head in seconds. On the RTX 4060 the same code runs the standard loop fast.

### Files changed (baseline)
- `requirements.txt`; `configs/baseline.yaml` (new); `src/models/{__init__,build}.py` (new); `src/train.py` (new); `src/data/dataset.py` (added `augment_train` toggle); `DOCS/results.md` (new, auto-generated)

## 2026-07-10

**Focus:** Repo hygiene — got everything from the 07-06/07-08 work actually committed and merged into `main`, and fixed a gitignore bug that had been silently hiding source files.

### Done
- Discovered `src/data/*.py` (`dataset.py`, `transforms.py`, `splits.py`, `hagrid_annotations.py`, `config.py`, both download scripts, `__init__.py`) had **never been tracked in git** — Windows' `core.ignorecase=true` made the old unanchored `DATA/` gitignore rule match `src/data/` too. Anchored the rule to `/DATA/`, added an anchored `/models/` rule (weights only, not `src/models/`), plus `__pycache__/` and `*.pyc`; untracked stale compiled `.pyc` files.
- Committed the backlog: `src/train.py`, `src/models/`, `src/data/`, `configs/`, `requirements.txt`, docs (`data-preparation.md`, `results.md`, `audience-notes-week2.md`, augmentation/preprocessing example images), and updates to `architecture-and-decisions.md` / `data.md` / `audience-notes-week1.md`.
- Left `models/baseline_mnv3_large/best.pt` (17 MB checkpoint) out of git, per the "weights git-ignored" convention.
- Merged `Docs` into `main` (fast-forward, local only — not yet pushed to `origin/main`).
- Merged this file and the repo-root `daily-update.md` into one log here; repo-root file removed. Updated `CLAUDE.md`'s daily-update pointer to this path.

### Files changed
- `.gitignore`; repo-root `daily-update.md` (removed); `DOCS/class-related/daily-update.md` (this file, merged); `CLAUDE.md` (pointer update)

### Decisions
- Kept the daily log at `DOCS/class-related/daily-update.md` going forward (one file, chronological) instead of two logs with overlapping purposes.

## 2026-07-12

**Focus:** Design decision — pivoted the implementation plan from single-stage full-frame classification to a **two-stage detect-and-crop pipeline** (MediaPipe hand crop → `timm` classifier), and documented how to switch approaches.

### Done
- Discussed one-model vs. two-model architectures; chose **two-stage** (off-the-shelf MediaPipe detector for the crop, my transfer-learned classifier unchanged) so the ML learning objective stays intact while gaining webcam robustness. Motivated by the 57% `full_frame` Phase-2 baseline (hand is <5% of the frame).
- **Rewrote AD-04** in `architecture-and-decisions.md` from "full-frame first, crop fallback (Proposed)" to "**two-stage detect-then-classify (Accepted)**", superseding the old stance. Added an explicit **"How to switch between full-frame and two-stage crop"** section (three coordinated config settings + the checkpoint/runtime mode-match rule).
- Propagated the change through Part A (system overview, component table `+detector`, data-flow contracts, runtime FPS budget), AD-05 (MediaPipe = runtime cropper, not classifier), the open-questions and change-log.
- Updated phases: Phase 1 (`crop_mode: bbox` now primary), Phase 2 (baseline outcome note), Phase 3 (added the `full_frame`-vs-`bbox` **A/B confirmation** §0 + record crop mode in the export sidecar), Phase 4 (MediaPipe crop is the primary path; measure detector latency Day 1), and `overview.md`.
- Updated `data-preparation.md` §4 + config table (default now `bbox`).
- Flipped the code/config default: `configs/data.yaml` and `src/data/config.py` `crop_mode` → `bbox`. Verified the config still parses (`crop_mode = bbox`); the bbox crop path already exists in `dataset.py`.

### Files changed
- `DOCS/architecture-and-decisions.md`, `DOCS/data-preparation.md`, `DOCS/phases/{overview,phase-1-setup-and-data,phase-2-baseline-model,phase-3-training-finetuning,phase-4-realtime-and-control}.md`
- `configs/data.yaml`, `src/data/config.py`

### Decisions
- **Two-stage crop adopted as the primary approach**, but honesty preserved: the `full_frame`-vs-`bbox` A/B in Phase 3 still *confirms* it (not treated as already-proven), and the full-frame path stays behind the `crop_mode` flag as a one-config-change fallback.
- AI (Claude) drafted the doc rewrite and config edits; I own the architecture decision itself (per `DOCS/Claude.md` — full-frame-vs-crop is a human-owned call).

### Week-2 deliverables — data-understanding submission
**Focus:** Produced the Week-2 "Data Understanding" deliverable set required by `DOCS/assignments/data_understanding.md`.

#### Done
- Built and **executed** `DOCS/week2/eda-notebook.ipynb` against the real `src/data` pipeline over all 8594 on-disk images — 5 visualizations (class balance, bbox overlays, resolution/aspect, relative hand area, overfit-single-batch curve) with written interpretations, figures saved to `DOCS/week2/figures/`.
- **Overfit-a-single-batch test passed:** `mobilenetv3_large_100`, 5 samples, loss → 0.0000 / 100% batch acc — pipeline validated end-to-end.
- Wrote `DOCS/week2/data-understanding-report.md` (rubric sections 1–5) with real numbers: **8594 usable images / 100% readable**, 4124 subjects, median hand = **1.8% of frame** (87.9% under 5%), 206 UUIDs (~2.3%) lost upstream to annotation/release version drift.
- Documented the **honest deviation from Week 1**: no Airflow/scheduled ingestion (HaGRID is static) — replaced by the deterministic range-request downloader.
- Created `DOCS/implementation-plan.md` (append-only living plan with finalized measurable R1–R8 + change log); added Week-2 entries to `DOCS/AI-usage.md` and a cadence note to `DOCS/Claude.md`.

#### Files changed
- `DOCS/week2/` (new: report, executed notebook, `figures/`), `DOCS/implementation-plan.md` (new), `DOCS/AI-usage.md`, `DOCS/Claude.md`

#### Decisions / notes for me to review
- The report **carries forward** existing human-owned decisions (8-class subset, ≥90% test target, by-user split, crop adoption) rather than inventing new requirements — I should re-read §5 and confirm the requirement wording is mine.
- Installed `nbconvert`/`ipykernel` locally to execute the notebook headlessly (dev tooling only, not a project dependency).

## 2026-07-13

**Focus:** Ran the Phase-3 §0 A/B — trained the **bbox-crop** frozen head-only model as the controlled counterpart to the 53% full-frame baseline (crop is the only variable).

### Done
- Added `configs/bbox_frozen.yaml` — identical to `baseline.yaml` (mobilenetv3_large_100, frozen backbone, head-only, no aug, seed 42, 40 epochs, linear-probe fast path) except `crop_mode: bbox` (via `data.yaml`); separate `out_dir` so the full-frame checkpoint is preserved.
- Preserved the prior full-frame `results.md` as `DOCS/results_full_frame.md` before the run (train.py overwrites `DOCS/results.md` each time).
- Trained on the CPU box (feature-caching pass ~19 img/s over 8594 imgs, then head trains in seconds). Split identical to baseline (train 5919 / val 1273 / test 1402, same by-user split).

### Result (honest, by-user test)
- **Test 91.65% / best val 94.03%** — vs full-frame **53.0% / 57.0%**. **+38.7 pts test** from cropping alone, backbone still frozen, no augmentation. Clears the ≥90% target at Stage A.
- Confirms AD-04 (two-stage detect-and-crop) empirically — consistent with Viz-4 (median hand = 1.8% of frame). Checkpoint `models/bbox_frozen_mnv3_large/best.pt` (git-ignored).

### Observations (for Ted to interpret — not decided)
- Remaining confusions are the finger-count pairs: `one`↔`two_up` (17 + 14 off-diagonal; `one` weakest at 0.815 F1) and `ok`→`palm` (12) — same weak pairs as full-frame, much reduced.
- Train ~99.7% vs val ~94% → some overfitting headroom; augmentation (§2) / unfreezing (§1 Stage B) are the levers, but already >90% frozen. **My call whether to pursue them.**
- Honesty (§4): future unfreeze/aug sweeps select on **val**; treat this as the frozen-Stage-A test data point, don't re-tune against test.

### Files changed
- `configs/bbox_frozen.yaml` (new); `DOCS/results.md` (regenerated, bbox); `DOCS/results_full_frame.md` (new, preserved full-frame — later moved to `DOCS/models/baseline_mnv3_large/results.md`); `models/bbox_frozen_mnv3_large/best.pt` (git-ignored)

### Note
- Auto-generated `results.md` header still reads "Baseline … (frozen backbone, head-only)" and omits `crop_mode` — `write_report` is hardcoded and doesn't distinguish runs; consider stamping crop mode / config name so reports self-identify.

### Also today — live webcam demo UI (AI-assisted scaffolding)
- Added `src/rt/` real-time inference package to eyeball trained models live and swap them freely:
  - `model_loader.py` — `load_checkpoint()` rebuilds any `best.pt` purely from its own metadata (backbone, classes, input size/mean/std/crop_mode). Switching models = point `--checkpoint` elsewhere; no code/config edits.
  - `preprocess.py` — `Preprocessor` reuses the *same* `build_transforms(train=False, …)` as training so runtime preprocessing matches byte-for-byte (AD-A.3). `RoiCropper` = centered square ROI standing in for the training bbox when a `crop_mode="bbox"` model runs live (no annotation at runtime).
  - `webcam_demo.py` — OpenCV capture loop with overlay (ROI box, top-k probability bars, FPS, live keys). `q/Esc` quit, `m` mirror, `r` toggle ROI. bbox models auto-enable ROI; full_frame use the whole frame.
- Installed `opencv-python` (was already in `requirements.txt`, missing from this env).
- Verified offline end-to-end (load → preprocess synthetic frame → predict) on **both** checkpoints and rendered the overlay to an image; the only untested surface is the literal camera grab (no camera/GUI in this environment).
- Run: `python -m src.rt.webcam_demo --checkpoint models/bbox_frozen_mnv3_large/best.pt`
- **For Ted:** ROI framing is a preprocessing-matching decision for bbox models — the fixed center square is a stand-in, not the eventual Phase-4 hand detector (MediaPipe / AD-05). Tune `--roi` to how tightly training cropped, or wire in a real detector later.

### Also today (2) — baseline docs + the real two-stage MediaPipe detector (AD-04)
**Documented the baseline model in its own subfolder** (`DOCS/models/`):
- `DOCS/models/baseline_mnv3_large/README.md` — full record of the Phase-2 baseline (full_frame frozen head-only mnv3, 53.0% test): config snapshot, training method (linear-probe), results, per-class read, why it was superseded by bbox, reproduce/run commands.
- Moved `DOCS/results_full_frame.md` → `DOCS/models/baseline_mnv3_large/results.md` (frozen auto-gen report); added `DOCS/models/README.md` index (baseline vs bbox, the AD-04 A/B).

**Replaced the fixed ROI stand-in with the actual two-stage detect-then-classify pipeline (AD-04):**
- `src/data/dataset.py` — extracted the crop geometry into module fn `padded_bbox_pixels(bbox, w, h, pad)`; `_crop_to_bbox` now calls it. **One source of truth** so the live crop is byte-identical to training (AD A.3). Verified byte-identical to the old inline formula over 1000 random cases.
- `src/rt/detector.py` (new) — `HandDetector` wraps **MediaPipe Tasks `HandLandmarker`** (VIDEO mode): 21 landmarks → landmark-hull bbox → `padded_bbox_pixels` (same pad) → crop. `detect()`→`HandBox|None`, manages VIDEO timestamps internally, degenerate-box guard.
- `src/rt/webcam_demo.py` — detector is now the **primary crop source**, auto-selected from the checkpoint's `crop_mode` (bbox→detector, full_frame→raw frame). No hand → **idle** (never classify a bad crop, per Phase-4). Keys: `d` toggle detector, `r` ROI fallback, `m` mirror, `q` quit. Draws the detected hand box + landmarks; prints a train/serve **mismatch warning** if detector/crop_mode disagree (AD A.3).
- Env: installed `mediapipe` (0.10.35, Python 3.13 → **Tasks API only**, no legacy `mp.solutions`). Downloaded the model bundle to `models/mediapipe/hand_landmarker.task` (git-ignored; detector prints the download command if missing).

**Honest finding for Ted to interpret (NOT decided — preprocessing/tightness is your call):**
- On HaGRID stills: MediaPipe detects a hand in **~48%** of frames; when detected, the bbox model classifies the **MediaPipe crop at 71%** vs **91.7%** on HaGRID's *own* annotated crop. That ~20-pt drop is the **MediaPipe-box-vs-HaGRID-box tightness mismatch** AD-04 flagged. **Caveat:** HaGRID's far/blurry/side hands are a *pessimistic* proxy for the real use case (a hand held up close to the webcam) — expect both detection and classification to be markedly higher live. Levers if the gap bites: `--pad` (currently 0.15 to match training), possibly re-cropping training bboxes to MediaPipe-style tightness, or a small self-captured fine-tune. Happy to run a `--pad` sweep on request.

### Files changed (this session)
- New: `src/rt/{__init__,model_loader,preprocess,detector,webcam_demo}.py`; `DOCS/models/README.md`; `DOCS/models/baseline_mnv3_large/README.md`
- Changed: `src/data/dataset.py` (extract `padded_bbox_pixels`); moved `results_full_frame.md` into `DOCS/models/baseline_mnv3_large/results.md`
- Env/assets: `opencv-python`, `mediapipe` installed; `models/mediapipe/hand_landmarker.task` (git-ignored)

### Also today (3) — first LIVE webcam test of the two-stage strategy + strategies folder
- **Live-ran** `webcam_demo` on the bbox model with the MediaPipe detector. Clean run (exit 0), detector auto-enabled, **no train/serve mismatch warning** — two-stage pipeline confirmed end-to-end on a real camera.
- **Live finding (Ted):** read gestures **well** overall; **notably more accurate with the hand held closer to the body (farther from cam)** and **worse when held close to the webcam.** Analysis: HaGRID hands are shot at conversational distance, so a hand near the body matches that distribution; a hand close to a wide webcam lens adds perspective foreshortening + odd aspect + soft focus the model never trained on. Consistent with the ~20-pt MediaPipe-crop gap measured earlier, plus a distance/perspective domain gap that only shows live. Not a pipeline bug — the classifier honestly reporting its at-a-distance training.
- **Created `DOCS/strategies/`** — running history of approaches tried, with results:
  - `README.md` (index + the 1→2 arc), `01-full-frame-single-stage.md` (Strategy 1, 53%, retired/fallback), `02-two-stage-mediapipe-crop.md` (Strategy 2, current — full build, results incl. the 92% annotated vs ~71% MediaPipe-crop gap, the live distance finding, and **6 ranked potential fixes** flagged as Ted's modeling calls: framing guideline, `--pad` tune / retrain-at-pad, perspective/scale aug (AD-09), self-captured fine-tune (AD-04 open Q), progressive unfreeze (AD-08 B), detection-confidence tuning).
- New: `DOCS/strategies/{README,01-full-frame-single-stage,02-two-stage-mediapipe-crop}.md`

### Also today (4) — bbox model doc, Strategy 2.1, and all 6 fixes implemented in code
_(Note: DOCS was reorganized into `build/`, `AI/`, `class-related/`, `models/` between sessions; strategies became per-approach folders. Fixed reorg-broken relative links in every doc touched below.)_

**Documented the current (bbox) model** — `DOCS/models/bbox_frozen_mnv3_large/`:
- **Regenerated the honest report from the checkpoint** (the transient root `DOCS/results.md` was removed in the reorg and never committed): VAL 0.9403 / **TEST 0.9165**, per-class F1 0.82–0.97, confusion matrix → `results.md`. `README.md` = full record (config, method, baseline A/B, weak finger-count pairs, live-gap caveat). Updated `DOCS/models/README.md` index to point at it.

**Strategy 2.1 — hardening of Strategy 2 against the live distance gap:**
- New `DOCS/build/strategies/2-two-stage-mediapipe-crop/02.1-two-stage-robustness-fixes.md` — explains 2.1 = same pipeline + 6 fixes; documents **each fix + how it's implemented + exact command**, tagged 🟢 live-now / 🟡 needs-retrain / 🔵 needs-data. Updated strategies `README.md` (added 2.1 row + 1→2→2.1 arc) and marked the Strategy-2 page superseded.

**Implemented all 6 fixes in code (defaults preserve current behavior; modeling decisions left to Ted per CLAUDE.md):**
- **#1 Framing hint (live):** `HandBox.area_frac` + `framing_hint()` warn "hand too close" when it fills too much of the frame. `--near-frac` (0.40). [`detector.py`, `webcam_demo.py`]
- **#2 Live pad tune (live):** `[` / `]` keys step `detector.pad` live, shown in HUD; `--pad`, `--pad-step`. [`webcam_demo.py`]
- **#3 Perspective/blur aug (knob, default 0=off):** `AugmentCfg.perspective` + `blur_sigma` → `RandomPerspective`/`GaussianBlur` added to the *train* pipeline only when >0. [`config.py`, `transforms.py`]
- **#4 Self-capture (tool):** new `src/rt/capture_dataset.py` records detector crops (same pad) into ImageFolder `DATA/selfcapture/<class>/`; keys 1-8/SPACE/c. Fine-tune recipe documented; the run is Ted's.
- **#5 Progressive unfreeze (mechanism, default 0):** `build_model(..., unfreeze_blocks=N)` + `unfreeze_last_n_blocks()` re-enable the last N of 7 backbone stages; `train.py` reads it and disables the feature-cache when N>0. [`build.py`, `train.py`]
- **#6 Detection-confidence (live):** detector exposes the 3 MediaPipe thresholds; `--detect/-presence/-tracking-confidence`. [`detector.py`, `webcam_demo.py`]
- **Verified offline:** unfreeze2 → 51.98% trainable (vs 0.24% frozen); aug knobs add ops only when >0 and default config still 0.0; `area_frac`/`framing_hint` correct; new HUD end-to-end (detector crop→predict→draw) OK on the real checkpoint; all changed modules `py_compile` clean; all doc links resolve. **Not run:** the retrain-dependent fixes (#3, #5) and the self-capture fine-tune (#4) — those await Ted's training decisions.
- New: `DOCS/models/bbox_frozen_mnv3_large/{README,results}.md`; `DOCS/build/strategies/2-two-stage-mediapipe-crop/02.1-two-stage-robustness-fixes.md`; `src/rt/capture_dataset.py`
- Changed: `src/rt/{detector,webcam_demo}.py`; `src/data/{config,transforms}.py`; `src/models/build.py`; `src/train.py`; `DOCS/models/README.md`; `DOCS/build/strategies/README.md` + `2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md`; `DOCS/models/baseline_mnv3_large/README.md` (link fixes)

## 2026-07-14

**Focus:** THE SCOPE PIVOT — control changes from an 8-gesture vocabulary to a motion-tracked cursor. Docs-only session (code changes follow next).

### The decision (Ted's, recorded as AD-17…AD-20)
- **AD-17 — cursor control:** MediaPipe (stage 1, already the cropper) now also drives the **mouse cursor** from the hand's position; the classifier (stage 2) shrinks to **binary palm/fist** — palm = no click, **fist = click**. One point-and-click interaction covers all of FNAF; no gesture→action map, no Office/Camera mode controller (AD-13 retired unbuilt).
- **AD-18 — data trim:** `classes: [palm, fist]` (~2,179 images, already downloaded — no new acquisition); retrain MobileNetV3 with a fresh 2-class head, same recipe as `bbox_frozen_mnv3_large`.
- **AD-19 — absolute mapping** (Ted chose over relative/joystick): palm-center anchor → mirror → control box (~60%×55%) → EMA + dead-zone; cursor freezes on no-hand. Relative mapping documented as fallback.
- **AD-20 — single click per fist** (Ted chose over hold-while-fist): edge-triggered FSM, K-consecutive-frame debounce, re-arm on confirmed palm, cooldown backstop — double-fires structurally impossible.

### Done (all documentation)
- **Created `DOCS/legacy/`** and `git mv`'d the superseded plan there (history preserved): `proposal.md` + the entire `phases/` tree (8 files). Wrote `legacy/README.md` — why the pivot, what moved, what still carries forward (Phases 1–2 outputs all bank).
- **New [Strategy 3](../build/strategies/3-cursor-and-click/03-cursor-and-click.md)** — full design spec: pipeline diagram, cursor-mapping chain, click-FSM states, data/retrain workflow, what carries over from 2.1 (everything), open questions (pointing precision, arm's-length palm/fist reliability, fatigue, FPS budget). Strategies `README.md`: added row 3, arc now 1→2→2.1→3, 2.1 marked "folded into 3".
- **New [plan.md](../build/plan.md)** (replaces legacy phases): 6-step roadmap — trim data → binary retrain → cursor mapper → click FSM → game integration (move+click registration test day 1) → robustness/demo; v2-specific risk table.
- **`architecture-and-decisions.md`:** Part A rewritten around the cursor pipeline (new diagram, cursor-mapper + click-FSM components, per-frame cursor contract); AD-17…AD-20 added; AD-03 superseded, AD-13 retired, AD-12 simplified-into-FSM note, AD-14 amended (cursor movement; `fnaf_layout.yaml` retired), AD-15 clarified (still static-poses — cursor motion is geometry, not a temporal model); open questions + changelog updated.
- **Pivot banners/notes:** `data-preparation.md` (8-class tables = pre-pivot era), `models/README.md` (both checkpoints = 8-class era; binary model doc lands on retrain). Updated root `CLAUDE.md` (one-liner, palm/fist facts, defer-list now cursor/FSM tuning, fixed stale pointers) and `DOCS/AI/Claude.md` (gesture→action wording → cursor/click, fixed legacy links).

### Decisions & notes
- Strategies 1–2.1 stay in place as the measured history (that folder's whole point); only *plan* docs moved to legacy.
- Learning objective explicitly intact: stage 2 is the same transfer-learning exercise, `num_classes` 8→2.
- **Next session (code):** `configs/data.yaml` trim + split verify, binary retrain, then `src/rt/cursor.py` + `src/control/click_fsm.py` per plan.md.
- AI use: Claude drafted this entire doc restructure from Ted's four decisions (legacy scope, click semantics, mapping choice, docs-first) — log in `AI-usage.md` week 3.

### Files changed
- New: `DOCS/legacy/README.md`; `DOCS/build/strategies/3-cursor-and-click/03-cursor-and-click.md`; `DOCS/build/plan.md`
- Moved: `DOCS/build/proposal.md` → `DOCS/legacy/`; `DOCS/build/phases/` → `DOCS/legacy/phases/`
- Changed: `DOCS/build/architecture-and-decisions.md`; `DOCS/build/strategies/README.md`; `DOCS/build/data/data-preparation.md`; `DOCS/models/README.md`; `DOCS/AI/Claude.md`; root `CLAUDE.md`

## 2026-07-15

**Focus:** Built out Strategy 3 end-to-end — data trim, binary retrain, cursor mapper, click FSM, input layer, play loop (plan.md Steps 1–5 code-complete; live/game testing is next and is Ted's).

### Done
- **Step 1 — data trim (AD-18):** `configs/data.yaml → classes: [palm, fist]`; verified split over 2,179 images: train 1,494 / val 360 / test 325, user-grouped, leak assertions pass.
- **Step 2 — binary retrain (AD-18):** new `configs/palmfist_frozen.yaml` (identical recipe to `bbox_frozen`, only classes/num_classes change) → `models/palmfist_frozen_mnv3_large/best.pt`. **VAL 0.9639 / TEST 0.9815** (random 0.50); palm F1 0.9812 / fist F1 0.9818; 6 test errors (4 are palm→fist, i.e. phantom-click direction — the FSM debounce is the guard). Ran on CPU via the linear-probe path (~3 min). *For Ted:* val was still climbing at epoch 40 with no overfit signal — more epochs / unfreeze / aug are your levers if the live test wants more. Model documented in `DOCS/models/palmfist_frozen_mnv3_large/`.
- **Step 3 — cursor mapper (AD-19):** new `src/rt/cursor.py` — palm-center anchor (landmarks 0/5/9/13/17), control box → clamp → EMA → dead-zone → screen px; freezes on no-hand, glides (never teleports) on re-detection. `webcam_demo` now previews the whole thing with **no real input**: control box, virtual-cursor crosshair, click-FSM state + CLICK flash; same flags/defaults as `play` so tuned values transfer.
- **Step 4 — click FSM (AD-20):** new `src/control/click_fsm.py` — DISARMED/ARMED, K consecutive confident frames, edge-triggered single fire, re-arm on confirmed palm only, cooldown backstop; dropout can never click; legacy 8-class labels safely disarm.
- **Step 5 — input + play loop (AD-14 amended):** new `src/control/input_sim.py` (`pydirectinput` move+click, DPI-aware, PAUSE=0, `--dry-run`, kill-switch) and `src/control/play.py` (detector → cursor → classifier → FSM → input; global **ESC kill-switch** via `keyboard` hook; HUD). Built, **not yet run against FNAF** — the day-1 movement+click registration test is Ted's next step (dry-run first).
- **Verified offline:** 27-check suite passed (mapper clamp/freeze/glide/dead-zone/mirror, FSM single-fire/re-arm/dropout/cooldown, checkpoint reload → 2-way softmax); `py_compile` clean on all touched modules; fixed a real bug the smoke test caught (`play.py --help` crashed on cp1252 consoles — Unicode arrows in the docstring).
- **Docs:** model record `DOCS/models/palmfist_frozen_mnv3_large/{README,results}.md`; models index + strategies README rows updated (strategy 3 → built+trained, 0.9815¹); strategy-3 page status + built-pieces table; plan.md status banner.

### Decisions & notes
- All tuning values (control box 0.60×0.55, alpha 0.35, dead-zone 0.005, K=3, conf 0.70, cooldown 0.3 s) are **starting-point defaults surfaced as CLI flags** — untouched decisions, Ted tunes live.
- 0.9815 is on HaGRID's annotated crops; the MediaPipe-crop and arm-extended-distance numbers are unmeasured until the live test.
- **Next (Ted):** (1) `webcam_demo` with the new checkpoint — eyeball palm/fist + virtual cursor; (2) `play --dry-run`; (3) day-1 in-game registration test (windowed FNAF, admin terminal if needed).
- AI use: Claude implemented Steps 1–5 (config, training run, all four new modules, demo integration, verification) per AD-17…AD-20's already-locked decisions — log in `AI-usage.md` week 3.

### Files changed
- New: `configs/palmfist_frozen.yaml`; `src/rt/cursor.py`; `src/control/{click_fsm,input_sim,play}.py`; `models/palmfist_frozen_mnv3_large/` (weights git-ignored); `DOCS/models/palmfist_frozen_mnv3_large/{README,results}.md`
- Changed: `configs/data.yaml` (classes → palm/fist); `src/rt/webcam_demo.py` (Strategy-3 preview); `DOCS/models/README.md`; `DOCS/build/strategies/README.md`; `DOCS/build/strategies/3-cursor-and-click/03-cursor-and-click.md`; `DOCS/build/plan.md`; `DOCS/results.md` (regenerated by train.py)

### Also today — first live test finding → Strategy 3.1 (distance-invariant cursor)
- **Live finding (Ted):** the pipeline works but the cursor had **too much of a distance issue** — behavior changed with hand distance. Requirement: calibration must care only about the anchor point's position from the frame's top-left, never hand/box size.
- **Root cause found (a real bug, not tuning):** the 3.0 anchor was computed from the detector's **frame-clipped** landmarks. A close/large hand has fingers/wrist off-frame; clipping drags those points to the frame edge and **biases the anchor toward the frame interior** — same pointing spot, different cursor position, worse the closer the hand.
- **Fix (Strategy 3.1):** `HandBox` now carries **unclipped** normalized landmarks (`landmarks_norm`); the anchor (`hand_anchor_norm`, replaces `palm_anchor_norm`) is a pure position from top-left computed from them — size-invariant by construction; off-frame anchors are handled by the control-box clamp instead of clip bias. The classifier crop still uses clipped values (a crop can't leave the frame). New `--anchor palm|box` flag on demo + play (palm plate = stable through the squeeze; box = hull middle, dips slightly on fist) — Ted's live A/B.
- **Verified:** suite extended to 32 checks — palm/box anchors bit-identical across a 9× hand-size change; unclipped anchor keeps tracking with a third of the hand off-frame (clipped version visibly drags inward); all prior checks still pass. (Also fixed a bad first version of the size-invariance test itself — uncentered template.)
- **Not fixed by 3.1 (documented):** classifier accuracy at unusual distances (that's the Strategy-2 domain gap — 2.1 levers / self-capture remain the path) and far-distance reach (shrink `--box-w/--box-h` to raise gain — tuning, not a bug).
- Docs: new `DOCS/build/strategies/3-cursor-and-click/03.1-distance-invariant-cursor.md`; strategies README (3.1 row, arc → 3 → 3.1); strategy-3 page marked hardened-by-3.1; AD-19 amendment + changelog in `architecture-and-decisions.md`.
- Files — Changed: `src/rt/{detector,cursor,webcam_demo}.py`; `src/control/play.py`. New: the 03.1 strategy doc.

### Also today — second live test → Strategy 3.1.1 (detection, click, distance gain)
- **Live findings (Ted):** (1) detection touchy — hands missed, fast motion breaks tracking, slow re-acquire; (2) clicks ~50%, and a **slow** palm→fist never clicks; (3) distance error still present after 3.1.
- **Fix 1 — detection:** MediaPipe confidence defaults lowered **0.5 → 0.3** (all three gates, detector + both CLIs). Stickier lock, eager re-detect; raise `--detect-confidence` if false grabs appear.
- **Fix 2 — click FSM grace (the slow-fist bug was structural):** 3.0 hard-disarmed on the first ambiguous frame; a slow squeeze's in-between poses are ambiguous, so it disarmed mid-squeeze and the finished fist had nothing to fire from — slow clicks were impossible *by construction*. Now ambiguous frames just don't count; only ambiguity sustained past `--fsm-grace` (default 10 frames) disarms. Firing still needs K confident fists; re-arm still needs K palms; sustained absence still disarms — safety intact. HUD shows a `?n/grace` counter.
- **Fix 3 — adaptive control box (the real distance invariance):** 3.1 fixed *position*; the residual error was *gain* (same arm motion = huge cursor travel up close, tiny far away). Default `--box-mode adaptive`: box width = `--box-gain` (4.0) × palm span (wrist→middle-knuckle — stable through the squeeze), so the same physical motion moves the cursor the same amount at any distance (verified ±1 px at 4× size difference). Documented trade-off: position- and motion-invariance are mutually exclusive; `--box-mode fixed` reverts to 3.1.
- **Verified:** suite now **39 checks**, all passing — slow squeeze fires exactly once, one dropout mid-squeeze no longer cancels, sustained absence can't fire, constant gain across distance, fixed mode ignores scale.
- Docs: new `03.1.1-live-robustness-fixes.md`; strategies README (3.1.1 row + arc); 3.1 page marked patched; AD-19 second amendment + AD-20 amendment + changelog.
- Files — Changed: `src/rt/{detector,cursor,webcam_demo}.py`; `src/control/{click_fsm,play}.py`. New: the 03.1.1 strategy doc.

## 2026-07-19

### Week-3 deliverable (W3A1) — ML experimentation report + MLflow tracking
- **Goal:** meet the Week-3 assignment deliverables (experiment tracking, evaluation, model selection, report).
- **MLflow gap closed:** the project had tracked runs via self-describing checkpoints + frozen `results.md`, not MLflow. Decision (Ted): **retroactively log the three real experiments into MLflow** rather than re-train or skip it. New `scripts/mlflow_log_runs.py` logs `baseline`/`bbox_frozen`/`palmfist_frozen` with params from each `config.snapshot.json` and metrics transcribed verbatim from the committed `DOCS/models/<run>/results.md` (nothing invented); `scripts/mlflow_export_comparison.py` exports the run-comparison CSV + accuracy chart. Store `./mlruns` is git-ignored; the exported artifacts are committed under `week3/`.
- **Report:** wrote `class-related/week3/ml-experimentation-report.md` (rubric §1 feature engineering, §2 experiment design, §3 three-run results, §4 model selection). Candidate for Week-4 tuning confirmed: `palmfist_frozen_mnv3_large`, with the honest train/serve caveat (0.9815 on HaGRID crops ≠ live/arm's-length accuracy).
- **Docs updated in place:** `build/plan.md` (Week-3 update — on track, no roadmap change), `AI/Claude.md` (MLflow added to tools + Week-3 cadence note), `AI/AI-usage.md` (Week-3 entry).
- **AI use:** Claude wrote the MLflow glue scripts and drafted the report/updates from existing decision records and real recorded numbers; the model-selection call and evaluation-honesty framing are Ted's. Log in `AI-usage.md` week 3.

### Files changed
- New: `scripts/mlflow_log_runs.py`; `scripts/mlflow_export_comparison.py`; `DOCS/class-related/week3/{ml-experimentation-report.md, mlflow-run-comparison.csv, mlflow-run-comparison.png}`
- Changed: `DOCS/build/plan.md`; `DOCS/AI/Claude.md`; `DOCS/AI/AI-usage.md`; `.gitignore` (add `/mlruns/`)

## 2026-07-20

### Strategy 3.2 (AD-21) — reframe the click classifier as `fist` vs. `not_fist`
- **Motivation (Ted's ask):** the AD-18 model only knows `palm`/`fist`; a 2-way softmax must map *any* other hand shape onto one of them, so an unrecognised pose (point, "ok", a slow squeeze's mid-frames) can read as `fist` and fire an **accidental click**. Fix: make "if it isn't a confident fist, it's no-click" something the **model learns**, not undefined behaviour.
- **Decisions (Ted delegated these — "decide yourself"):** negative class named **`not_fist`** (honest confusion matrix; zero runtime cost); trained on six diverse non-fist gestures (`palm, one, two_up, like, dislike, mute`); **`ok` held out of training** as an unseen gesture for the open-set false-click check; **balance by seeded downsampling** of `not_fist` to the `fist` count (keeps the linear-probe fast path, stays A/B-comparable); frozen linear-probe recipe **identical to `palmfist_frozen`**.
- **Data layer (backward-compatible):** `configs/data.py` gained `label_groups` (many folders → one target label; bbox still the folder's own hand) and `balance` (downsample majority to minority, drawn evenly across sources, seeded). `build_index` now takes `source_to_label` and records a `source_label` column; `build_full_index` applies the balance step. Absent config ⇒ old folder-is-label behavior — verified the palm/fist `data.yaml` still indexes identically (2179 rows, palm 1082 / fist 1097).
- **Verified (data smoke test):** grouped+balanced index = `not_fist 1097 / fist 1097`, negatives drawn ~183 evenly across all six gestures, `ok` correctly excluded, user-grouped 70/15/15 split intact (train 1559 / val 308 / test 327).
- **Runtime:** `play.py` now accepts any binary click model (2 classes incl. `fist`) and derives the FSM no-click label from the checkpoint; `webcam_demo.py` does the same for its click preview (legacy multi-class keeps the `palm` default); `click_fsm.py` docstring updated (logic unchanged — `palm_label`/`fist_label` were already the seam). Preprocessing, crop contract, cursor, kill-switch untouched — a 3.2 checkpoint is a drop-in for a 3.0 one.
- **Trained (frozen linear-probe, seed 42, CPU):** best val **0.9221**, test **0.9358** (F1 not_fist 0.9307 / fist 0.9402; 21/327 errors). Lower than palm/fist's 0.9815 *by design* — `not_fist` is a much harder negative.
- **Open-set eval built + run (`src/eval_openset.py`) — the metric that matters:** for each gesture folder, how often each model fires `fist` (a click); `ok` is held out of *both* models so it's a fair unseen-pose A/B. Result (250-img sample, conf-gate 0.70) — **click% on non-fist poses, palm/fist → fist-vs-rest:** one (pointing!) 75.6 → **9.2**, two_up 81.2 → **2.4**, dislike 74.4 → **3.2**, mute 60.4 → **0.0**, like 41.2 → **2.8**; **`ok` (unseen by both) 6.8 → 0.8**. Cost: `fist` click-rate 99.2 → 96.0 (~3% of real fists read as not_fist). **Conclusion: the reframe massively cuts accidental clicks on ordinary hand shapes, and generalises to an unseen gesture, for a small true-click cost** — exactly what AD-21 predicted.
- **Not done here (Ted's call):** the **live** re-test at arm's-length distances (the verdict that ultimately counts) + whether the ~3% fist-recall cost is acceptable or worth unfreezing/more negatives.
- **AI use:** Claude designed, implemented, trained, and evaluated the reframe end-to-end (data layer, configs, runtime wiring, training run, open-set eval, docs) after Ted delegated the ML decisions for this task; the live verdict remains Ted's. Log in `AI-usage.md` week 3.

### Files changed
- New: `configs/data_fistvsrest.yaml`; `configs/fistvsrest_frozen.yaml`; `src/eval_openset.py`; `models/fistvsrest_frozen_mnv3_large/` (weights git-ignored); `DOCS/models/fistvsrest_frozen_mnv3_large/{README,results}.md`; `DOCS/build/strategies/3-cursor-and-click/03.2-fist-vs-rest.md`
- Changed: `src/data/{config,hagrid_annotations,dataset}.py`; `src/control/{click_fsm,play}.py`; `src/rt/webcam_demo.py`; `DOCS/build/architecture-and-decisions.md` (AD-21 + changelog); `DOCS/build/strategies/README.md`; `DOCS/models/README.md`; `DOCS/results.md` (regenerated by train.py)

## 2026-07-21

### Interactive hand-tracking evaluation dashboard (new `src/eval_dashboard/`)
- **Goal (Ted's ask):** a browser "obstacle course" that measures how well the hand controller performs, then collects a 1–5 survey — a way to quantify the controller Ted has been tuning, not just eyeball it.
- **Architecture:** a Flask app that *owns* the Strategy-3 control loop. `tracker.py` lifts `play.py`'s per-frame pipeline (detector → cursor mapper → CNN → click FSM → `InputSim`) into a start/stop background thread; the browser is driven by the **real** cursor + real clicks it emits, so every obstacle exercises the actual controller. `server.py` serves the SPA and exposes `/api/status`, `/api/finish` (kill-switch: stops the tracker so the survey uses the normal mouse), `/api/results` (writes the doc). All tracker flags mirror `play.py` so a tuned run carries over.
- **Three obstacles (front-end `static/`):** (1) **Precision** — buttons 1→5 pop at random spots inside a fixed box (no scrolling); timed click-1→click-5 + per-button splits. (2) **Rapid clicks** — a DOWN button pressed 7× with a filling depth gauge; timed. (3) **Tracer** — trace an SVG sine line start→green dot; per-frame deviation → 0–100 accuracy score + path-coverage gate so you can't skip to the end. Then a **Finish** button stops the tracker and a 5-question survey (the exact prompts Ted specified) is answered with the mouse.
- **Results storage:** `results.py::save_results` writes a human-readable report — a real `.docx` when `python-docx` is installed, else a `.txt` fallback — plus a `.json` sibling for analysis, to `eval_results/`.
- **UX:** white background, colorful buttons, hidden OS arrow replaced by a custom purple pointer dot (hand drives it), a live tracker-status chip (label/conf/fps/clicks). ESC remains the global kill-switch (CLAUDE.md).
- **Verified:** `save_results` end-to-end (report formatting correct); server boots and all four routes respond (`--no-tracker --no-browser` smoke test). **Not yet run live** with the camera + `palmfist`/`fistvsrest` checkpoint driving real input — that's Ted's next step (`python -m src.eval_dashboard.server --checkpoint models/palmfist_frozen_mnv3_large/best.pt`, F11 fullscreen).
- **AI use:** Claude designed and implemented the dashboard end-to-end (thread wrapper, Flask API, front-end obstacle logic, doc writer) at Ted's request; obstacle scoring thresholds (reach/end tolerance, accuracy DEV_MAX, coverage gate) are starting points for Ted to tune. Log in `AI-usage.md`.

### Files changed
- New: `src/eval_dashboard/{__init__,tracker,results,server}.py`; `src/eval_dashboard/static/{index.html,style.css,app.js}`
- Changed: `DOCS/class-related/daily-update.md`

### Tracer obstacle clarity pass (dashboard Obstacle 3)
- **Goal (Ted's ask):** make the tracer easier to understand — clearly mark the start, and once tracing begins draw a line showing where the hand actually went.
- **Changes (front-end only):** (1) the target path now **previews** on entry (was hidden until Start was clicked), with a **labeled `START`** (yellow) and **`FINISH`** (green) caption on each dot and a **pulsing halo** around the start dot to draw the eye; the pulse stops when tracing begins. (2) A live **pink breadcrumb `<polyline>`** records the cursor path from the moment Start is clicked, so the tester sees their actual route vs. the target line; it clears on each (re)start and persists through the finish summary. Banner title updated to name the yellow start dot. No scoring/threshold logic changed — coverage/accuracy/timing untouched.
- **Verified:** `node --check app.js` passes. Not yet eyeballed live in the browser — Ted's next run.
- **AI use:** Claude implemented the UX changes at Ted's request. Log in `AI-usage.md`.
- Changed: `src/eval_dashboard/static/{index.html,style.css,app.js}`

### Controller & calibration reference doc (new)
- **Goal (Ted's ask):** a doc explaining how a model prediction becomes a real cursor move + click — the end-to-end runtime control layer and its calibration knobs, in one place.
- **New doc** `DOCS/build/strategies/3-cursor-and-click/controller-and-calibration.md`: walks the per-frame chain (detect/mirror/crop → **stage 1** cursor mapping: anchor → control box → clamp → EMA → dead-zone → screen pixel → `InputSim.move_to`; **stage 2** CNN softmax → `ClickFSM` confidence gate + K-frame confirm + grace → `InputSim.click`), a mermaid overview of both branches, the DirectX/DPI/kill-switch input facts, and two **calibration tables** (cursor knobs + click knobs) with raise-it/lower-it guidance keyed to `play.py` CLI flags. Grounded in the actual code (`cursor.py`, `click_fsm.py`, `input_sim.py`, `play.py`); no new decisions — defers cursor/FSM tuning to Ted per CLAUDE.md. Includes the plan.md Step-5 day-1 registration check.
- **Changed:** added a pointer to the new doc near the top of `03-cursor-and-click.md`.
- **AI use:** Claude wrote the reference doc from the existing runtime modules at Ted's request. Log in `AI-usage.md`.
- New: `DOCS/build/strategies/3-cursor-and-click/controller-and-calibration.md`
- Changed: `DOCS/build/strategies/3-cursor-and-click/03-cursor-and-click.md`; `DOCS/class-related/daily-update.md`

### Dashboard Overview tab + run history (previous-trials view)
- **Goal (Ted's ask):** an Overview tab on the start screen showing metrics/results from previous trials — score overviews plus the model and strategy each run used.
- **Front-end:** the welcome card now has **Start / Overview** tabs. Overview shows (1) an aggregate **stat-tile row** — trials recorded, best precision time (+avg), best trace accuracy (+avg), average survey rating — and (2) a scrollable **per-trial table** (newest first): when, model, strategy, precision/rapid times, trace accuracy, survey avg. Built with the existing design tokens (stat tiles + recessive table, tabular-nums, sticky header); numbers stay on ink colors, no color-coded series (followed the dataviz skill's form guidance — headline scores are stat tiles, detail is a table).
- **Back-end:** new `GET /api/history` returns compact summaries of every `eval_results/*.json` (new `results.summarize_run`), newest first, tolerant of older runs missing fields. `POST /api/results` now **stamps a `meta` block** (model = checkpoint dir name, derived strategy label — `3.2 · fist-vs-rest` / `3.0 · palm/fist` / generic — device) onto each saved run so the Overview can attribute scores; older runs (no meta) render as "—". Strategy/model derived server-side from `--checkpoint` (`_derive_meta`); `--no-tracker` runs label as a mouse UI test.
- **Verified:** `summarize_run` over the 3 existing runs; `/api/history` returns them; a test POST confirmed the `meta` block is written (`palmfist_frozen_mnv3_large`, `3.0 · palm/fist`, cpu) then cleaned up. Not yet eyeballed live — Ted's browser check.
- **AI use:** Claude implemented the tab, history API, and run-meta stamping at Ted's request. Log in `AI-usage.md`.
- Changed: `src/eval_dashboard/{server,results}.py`; `src/eval_dashboard/static/{index.html,style.css,app.js}`

### Backfilled model/strategy meta on the 3 existing eval runs
- The three `eval_results/eval_20260721_*.json` trials predate the dashboard's `meta` block, so the Overview tab (`summarize_run`) showed them with null model/strategy. All three were run with the current model, so hand-added a `meta` block to each (`model` fistvsrest_frozen_mnv3_large, `strategy` "3.2 · fist-vs-rest", `checkpoint`, `device` cpu) — matching exactly what `server._derive_meta` stamps on new runs. Verified all three parse via `summarize_run`.
- Changed: `eval_results/eval_20260721_140459.json`, `eval_results/eval_20260721_141942.json`, `eval_results/eval_20260721_142253.json`

### Strategy 3.2.1 — snappier click FSM as a selectable preset
- **Goal (Ted's ask):** create a Strategy 3.2.1 = the 3.2 model with the click FSM retuned to **conf 0.80 / K 2**, document it, make it testable, and let the eval dashboard run **either 3.2 or 3.2.1**. (Ted directed the values; per CLAUDE.md this is his tuning call — I implemented, didn't decide.)
- **Single source of truth:** new `src/control/strategies.py` defines the FSM presets (`3.2` = 0.70/3, `3.2.1` = 0.80/2) plus `resolve_fsm()` (explicit `--fsm-conf`/`--fsm-k` override the preset, preset overrides the AD-20 base defaults). Both `play.py` and the dashboard import it so the numbers can't drift.
- **`play.py`:** added `--strategy {3.2,3.2.1}`; `--fsm-k`/`--fsm-conf` now default to `None` and resolve through the preset (explicit flags still win). Startup line prints the chosen strategy. `python -m src.control.play --checkpoint models/fistvsrest_frozen_mnv3_large/best.pt --strategy 3.2.1 --dry-run` runs the snappy variant.
- **Dashboard:** `_strategy_fits()` gates the presets to fist-vs-rest checkpoints; new `GET /api/strategies` (applicable/current/options); `POST /api/start` accepts `{strategy}`, applies the preset to the tracker cfg *before* the loop reads it and stamps the label onto `_run_meta` (so the Overview **Strategy** column shows `3.2.1 · fist-vs-rest (snappy)`). New `--strategy` server flag sets the default pick. Front-end: a **Click-strategy picker** (segmented 3.2 / 3.2.1) on the Start panel, shown only when applicable, POSTed on start and locked once the tracker runs.
- **Design note:** the two knobs pull opposite ways on false clicks *by design* — the stricter 0.80 gate buys back the safety the looser K=2 gives up; net is snappier clicks without leaning harder on the FSM to reject unknown poses (still the model's job, AD-21). No FSM logic changed.
- **Verified:** `resolve_fsm` unit checks (None→0.70/3, 3.2→0.70/3, 3.2.1→0.80/2, overrides win); `play.py`/`server.py` parse; `server` imports; `_strategy_fits` True for fistvsrest / False for palmfist & no-tracker; `/api/strategies` returns the options; `play.py --help` renders in PowerShell. **Not yet run live** — the 3.2-vs-3.2.1 A/B (obstacle course + arm's-length feel) is Ted's next run and the verdict that counts.
- **AI use:** Claude implemented the preset module, CLI/dashboard wiring, UI, and docs at Ted's request (Ted set the 0.80/2 values). Log in `AI-usage.md`.
- New: `src/control/strategies.py`; `DOCS/build/strategies/3-cursor-and-click/03.2.1-snappier-fsm.md`
- Changed: `src/control/play.py`; `src/eval_dashboard/server.py`; `src/eval_dashboard/static/{index.html,app.js,style.css}`; `DOCS/build/strategies/README.md`; `DOCS/build/architecture-and-decisions.md`

### Dashboard Overview tab — filter + sort controls
- **Goal (Ted's ask):** on the Overview tab, filter past trials by any metric (e.g. `accuracy > 50`, `precision < 5s`) and by strategy/model, plus a sort-by control.
- **Front-end only** (`static/{index.html,app.js,style.css}`), no server/API change — all filtering/sorting runs client-side over the runs `/api/history` already returns. Added a control bar above the per-trial table:
  - **Strategy** and **Model** dropdowns, auto-populated from the unique values present in the loaded runs (selection preserved across reloads; syncs out if a value disappears).
  - **Composable metric filters:** pick a metric (Precision / Rapid seconds, Trace accuracy, Survey) + operator (`> ≥ < ≤ =`) + value → **Add filter**; each becomes a removable chip. Filters AND together. Time metrics are compared in **seconds** (the units shown in the table), so `precision < 5` means 5 s not 5 ms. Enter in the value box adds the filter; **Clear all** resets everything.
  - **Sort by** any metric or Date, with a High→Low / Low→High direction toggle. Runs missing a value sink to the bottom.
- The **aggregate stat tiles now reflect the filtered set** (count tile reads "N of M trials shown" when a filter is active); an empty filter result shows "No trials match these filters" while keeping the controls visible to adjust. Built with existing design tokens (purple accents, recessive selects, pill chips) — consistent with the rest of the card.
- **Verified:** `node --check app.js` passes. Logic exercised by reading through the filter/sort/render path; **not yet eyeballed live** — Ted's browser check on the Overview tab.
- **AI use:** Claude implemented the filter/sort UI and client-side logic at Ted's request. Log in `AI-usage.md`.
- Changed: `src/eval_dashboard/static/{index.html,app.js,style.css}`

## 2026-07-22

### Git housekeeping — split the accumulated working-tree changes into topic branches
- **Goal (Ted's ask):** the 2026-07-20/21 work (Strategy 3.2, Strategy 3.2.1, the eval dashboard) had all landed as one large uncommitted working-tree diff on `main`. Split it into the three real bodies of work, each on its own branch, then merge each into `main` — instead of one undifferentiated commit.
- **Grouping:** `strategy-3.2-fist-vs-rest` (AD-21: `label_groups`/`balance` in the data layer, the fist-vs-rest configs/model docs, `eval_openset.py`, the play.py/webcam_demo.py/click_fsm.py binary-click-model generalization); `strategy-3.2.1-snappier-fsm` (AD-20 amendment: `src/control/strategies.py` presets, `play.py --strategy`); `eval-dashboard` (`src/eval_dashboard/`, `eval_results/`, the controller-and-calibration doc, `code-map.md`, the `flask`/`python-docx` requirements). Merge order `A → B → C` — verified `src/eval_dashboard/server.py` imports `src.control.strategies`, so the dashboard branch genuinely depends on the FSM-presets branch merging first.
- **Shared docs** (`architecture-and-decisions.md`, `strategies/README.md`, `daily-update.md`) had hunks belonging to more than one branch (e.g. AD-20's amendment vs. AD-21 in the same file). Split each by hand along its existing section/paragraph boundaries so every branch's diff against `main` was exactly its own slice — verified each intermediate slice against the final working-tree content before committing.
- **Verified:** every intermediate `play.py` cut parsed (`ast.parse`); each branch's diff against `main` matched its intended slice exactly; final `main` after all three merges is byte-identical to the original pre-split working tree (`git status` clean, no residual diff).
- **AI use:** Claude did the branch/commit split and merges end-to-end at Ted's request. Log in `AI-usage.md`.
- Branches created and merged into `main`: `strategy-3.2-fist-vs-rest`, `strategy-3.2.1-snappier-fsm`, `eval-dashboard`.

## 2026-07-26

### Week 4 — hyperparameter tuning, pipeline orchestration, and model deployment (M4A1)
- **Goal (Ted's ask):** complete the Week-4 assignment criteria end-to-end — tune, register, evaluate, orchestrate, deploy, document — and merge to `main`.
- **Tuning target correction:** Week 3 named `palmfist_frozen` as the candidate, but AD-21 (2026-07-20) replaced the negative class afterwards, so the model actually in the loop — and the one tuned — is **`fistvsrest_frozen`** (val 0.9221 / test 0.9358). Its own record set the agenda: *"val still climbing at ep40 … more epochs / unfreeze / augmentation remain Ted's call."*
- **Two-arm search** (`src/tune.py`, `configs/tune_fistvsrest.yaml`), sized to a benchmarked CPU budget (~22 img/s fwd, ~15 img/s fwd+bwd → ~95 s/epoch): **Arm A** = 60-trial Optuna **TPE** over lr/wd/optimizer/batch/label-smoothing trained on *cached frozen features* (~1-2 s a trial, identical math to the frozen end-to-end loop); **Arm B** = 5-cell end-to-end grid over `unfreeze_blocks` × augmentation (~20-29 min a cell, 1 h 56 m total). Every trial is its own MLflow run under `fnaf-week4-tuning`; selection on **val only**, test read **once** (AD-10).
- **The week's real finding (AD-22, Proposed — Ted's call):** the harness re-runs the deployed recipe as a **control** and fired its drift warning — the old recipe scored 0.9740 in the new harness vs the committed 0.9221. Cause: timm's EfficientNet-family head init uses `r = 1/sqrt(out_features)`, so a **2-class** head starts at ±0.707 weights → **±27 logits → CE 2.70** instead of ln(2)=0.69. The predecessor spent ~30 of its 40 epochs undoing its own init — that *is* the "still climbing" note. Ablation (same seed, same features, init the only change): **0.9221 → 0.9740, best epoch 38 → 16**. Shipped as an opt-in `head_init` knob with `timm_default` **preserved as the default** so every committed run stays reproducible.
- **Attribution, which is the point:** +5.19 init fix · +0.87 TPE search · +1.41 unfreezing 2 blocks · +0.32 augmentation (noise). Calling the total a "tuning win" would have been the most misleading thing in the report.
- **Champion registered:** Arm B `unfreeze2_aug` → `fnaf-click-classifier` **v1**, alias `champion`, **val 1.0000 / test 1.0000 (327/327)**.
- **Attacked the perfect score before publishing it** (new `scripts/audit_split_leakage.py`): an independent eval path reproduces 327/327 *and* reproduces the old model's 0.9358 exactly; **0** duplicate files across splits (2,194 unique MD5s); test→train nearest-neighbour cosine max **0.845** against a train→train control of **0.891** — test images are no closer to train than train is to itself. **Not leakage.** Also recorded that `group_by_user` is thinner than AD-16 implies here: 1,569 user_ids for 2,194 images, **median 1 image/user**. Conclusion for the report: the offline benchmark is **saturated**, not conquered — and on the only metric with headroom (`ok`, unseen by every model) tuning changed **nothing**: 0.8% before, 0.8% after.
- **Orchestration (AD-23):** `src/pipeline/` — a ~200-line DAG engine (topological order, per-task freshness → `cached`, retries, upstream-failure propagation, JSON run records) + 9 training tasks + a documented per-frame inference graph. **Event-triggered** (`--if-data-changed` compares the split-manifest hash against the registered model's), explicitly not cron. Full run verified: expensive stages `cached`, so regenerating evaluation + serving checks costs **76 s** instead of 2 h.
- **Deployment (AD-24):** MLflow **pyfunc** artifact carrying its own preprocessing (so a served model can't drift from AD-A.3) returning a *click decision*; Flask endpoint serving `models:/fnaf-click-classifier@champion`; **ONNX** export with torch parity **7.15e-07** and a verified dynamic batch axis (AD-11 satisfied). Latency p50 **40.8 ms** round-trip — which is why the live loop keeps calling the model in-process against a ~33 ms frame budget.
- **Verified:** deployed-model reproduction exact (0.9221 val / 0.9358 test / per-class F1s); leakage audit; DAG run green (evaluate → export_onnx → serve_smoke, the last loading from the *registry* and returning `fist conf 0.974 click=True`); ONNX parity + dynamic batch; endpoint transcript captured live incl. a 422 on malformed input; `ruff` clean on all new modules. **Not yet run live** — every number is on HaGRID's *annotated* crops; the MediaPipe/arm's-length gap is untouched, and Step 5 (drive the real game) is still the outstanding gate.
- **Fixed along the way:** `capture_endpoint_transcript.py` deadlocked because the server subprocess wrote to an undrained `PIPE` (Flask's per-request logging filled the ~64 KB buffer) — now redirected to a file. ONNX export needed `external_data=False` to stay a single file, plus `onnxscript` (added to `requirements.txt`, along with the never-declared `mlflow` and the new `optuna`/`requests`).
- **AI use:** Claude built the tuning harness, orchestration, serving layer, evidence scripts, and report/doc drafts at Ted's request; Ted owns the reading of the results (init-vs-tuning attribution, the saturation call, AD-22 left Proposed). Log in `AI-usage.md`.
- New: `src/tune.py`, `src/eval_final.py`, `src/plotting.py`, `src/pipeline/{__init__,dag,tasks,run}.py`, `src/serve/{__init__,pyfunc_model,app}.py`, `configs/tune_fistvsrest.yaml`, `scripts/{mlflow_export_tuning,capture_endpoint_transcript,render_pipeline_diagram,audit_split_leakage}.py`, `DOCS/class-related/week4/*`, `DOCS/models/week4_tuned_fistvsrest/README.md`
- Changed: `src/models/build.py` (`head_init`/`init_classifier`), `src/train.py` (`head_init` passthrough, default unchanged), `requirements.txt`, `.gitignore`, `DOCS/build/{architecture-and-decisions,plan,code-map}.md`, `DOCS/AI/{Claude,AI-usage}.md`, `DOCS/models/README.md`

## 2026-07-27

### Strategy 3.2.3 — fixed the live FPS and missed-click complaints (measured, implemented)
- **Goal (Ted's ask):** "FPS is really bad and slow as well as some clicks don't get registered" — create Strategy 3.2.3 to fix or improve both, then implement it.
- **Result, measured end-to-end on the real loop** (`scripts/bench_pipeline.py`, HUD on, medians over 8 s/config, `week4_tuned_fistvsrest`): **45.7 ms → 32.6 ms per frame, 21.0 → 28.3 FPS** — the 30 FPS camera is now the ceiling — plus **−67 ms** of input lag and a click that is actually held for 60 ms instead of 0.
- **Diagnosis of the missed clicks — two independent causes.** (1) Every FSM gate counts *frames*, so a 46 ms frame plus a stale sample left a natural ~150 ms squeeze roughly one frame of margin; no `conf`/`K` value recovers a pose that was never sampled. (2) Verified in `pydirectinput`'s source: `click()` issues a **single** `SendInput` with `dwFlags = MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP` — a **0 ms** hold. FNAF (Clickteam/DirectX) samples mouse state once per game frame and can observe "up, still up", which explains clicks vanishing while `sim.clicks` increments. Independent of both FPS and the model.
- **Three of my pre-measurement claims did not survive the benchmark, and the doc records the corrections:**
  - Threaded capture is a **latency** fix, not a throughput one. Predicted −33 ms/frame; actual ≈ 0. When the loop is slower than the camera a synchronous `read()` returns in **0.2 ms** — it is serving from the driver's backlog. Measured that backlog directly (run a slow loop, then count frames that return instantly): **2 frames ≈ 67 ms of stale-frame lag**, bounded but permanent. That lag is the part that eats clicks, so the fix stayed — for the right reason.
  - `torch.set_num_threads(4)` is not the −6 ms micro-optimization I called it: with torch's 16-thread default the same loop measures **14.7 FPS** instead of 21.0, and *preprocessing* (torchvision ops on one 224 px tensor) inflates from 2.6 ms to **138–161 ms**. A 16-thread pool fighting MediaPipe's XNNPACK threads for 16 logical cores loses badly — and since preprocessing is torchvision, it hurt the ONNX path too. The cap is applied for **both** backends.
  - The original per-stage table was measured stage-by-stage in isolation and added up to a wrong total (~90 ms / 11 FPS). The real serial loop is 45.7 ms / 21 FPS, because the 33.6 ms camera read only costs that in a read-only loop. Replaced with an end-to-end table from `bench_pipeline.py`.
- **Also caught:** an early version of the bench reported every config after the first as progressively slower — its own artifact (each ONNX session and torch runner holds a thread pool, and measuring four configs in one process oversubscribes the CPU). Each config now runs in a subprocess. Separately, `torch` here is a **CPU-only build** (`2.10.0+cpu`), so `play.py --device cuda` is a dead flag; left that way deliberately, since ONNX-on-CPU already hits the camera ceiling.
- **Implemented** (all revertible with `--backend torch --no-threaded-capture --click-hold 0`): `src/rt/backends.py` (ONNX vs. torch, chosen at load and **parity-checked** on probabilities — 1.2e-07 against a 1e-3 tolerance — with fallback to torch on a missing file, missing `onnxruntime`, or a failed check); `src/rt/capture.py` (background grabber keeping only the newest frame, `dropped` counter on the HUD); `src/control/input_sim.py` (`mouseDown` → non-blocking 60 ms deadline → `mouseUp` via a new `tick()`, cursor frozen while pressed, redundant `move_to` skipped, and `kill()` now releases the button **before** disabling output so a mid-press kill can't leave it stuck down); `src/control/play.py` (wiring + `--backend/--torch-threads/--no-threaded-capture/--click-hold/--hud-every`); `scripts/bench_pipeline.py`.
- **Verified:** unit tests for the click sequence (down/hold/up), the press-suppresses-movement rule, kill-mid-press releasing, and the `--click-hold 0` legacy path; both backends return **identical** probabilities (0.8428/0.1572) and ONNX falls back cleanly when `model.onnx` is absent; `play.py` smoke-tested on both the new defaults and the full revert path. **Not tested against the running game** — Step 5 is still open, so whether the 60 ms hold actually fixes FNAF's missed clicks is unverified.
- **Deliberately changed no FSM value** (`K`, conf, grace, cooldown) — those are Ted's live calls per CLAUDE.md, and the pending 3.2-vs-3.2.1 A/B is only now being judged through a sampling interval close to what those presets assumed. Left open for Ted: the `--click-hold` duration, whether `K`/`grace`/`cooldown` move to milliseconds, whether the presets want re-tuning at the higher frame rate, and the detection-downscale lever (worthless at the camera ceiling).
- **Biggest finding of the day, found by accident:** late in the session every measurement went ~2.1x slower at once — every stage, uniformly, CPU otherwise idle (9% load). Verified at that moment: this box is a **laptop** (Core Ultra 7 165H), it was **discharging**, and the CPU was pinned at its **1.4 GHz base clock**. Model forward in isolation, same code/day: torch 23.9 → 51.6 ms, ONNX 5.3 → 11.2 ms; whole loop **28.3 → 9–11 FPS**, i.e. **on battery, with every 3.2.3 fix applied, the loop is slower than the unfixed loop was on AC.** Power state is the single largest term in the frame budget and plausibly a big part of the original "FPS is really bad" complaint. The earlier runs' power state wasn't recorded, so the AC/battery labels are the reading that fits, not a controlled A/B. Acted on it: `play.py` now prints a **[warn] running ON BATTERY** line at startup, and `bench_pipeline.py` stamps AC/battery + CPU clock onto every report so no future number is uncomparable.
- **Measurement caveat, recorded in the doc:** absolute FPS varies with desktop load (Spotify/Discord/Chrome/VS Code plus Ted's eval-dashboard server, PID 24964, were running throughout — left alone since it's his). A contended re-run measured the same four configs at 7.1 / 5.4 / 23.4 / 9.5 FPS. Stable across every run is the ordering and the gap size: ONNX wins every time, and torch degrades *worse* under contention (model stage 48–104 ms vs. ONNX's 6.5 ms). Anything sharing the CPU — the dashboard included — competes with `play.py`.
- **AI use:** Claude did the benchmarking, the `pydirectinput` source read, the implementation, and the doc drafts at Ted's request; all tuning/adoption calls left open. Log in `AI-usage.md`.
- New: `src/rt/backends.py`, `src/rt/capture.py`, `scripts/bench_pipeline.py`, `DOCS/build/strategies/3-cursor-and-click/03.2.3-latency-and-click-delivery.md`
- Changed: `src/control/play.py`, `src/control/input_sim.py`, `DOCS/build/strategies/README.md`, `DOCS/build/code-map.md`

## 2026-07-28

### Ported Strategy 3.2.3 into the eval dashboard's tracker; found and fixed a stuck-button bug
- **Goal (Ted's ask):** boot the eval dashboard and confirm Strategy 3.2.3 works with the new model picker (`src/eval_dashboard/model_registry.py`, `/api/models`, already in progress on disk).
- **Found a real bug before it shipped:** `src/eval_dashboard/tracker.py` still built `InputSim` and called `sim.click()` every frame but never called the new `sim.tick()` — which is what sends the deferred `mouseUp()`. Confirmed in isolation (`InputSim(dry_run=True, hold_s=0.06)`): without `tick()`, `is_pressed` stays `True` forever after the first click, and `is_pressed` is what `move_to()`/`click()` both check first — so the **first click in the dashboard would have frozen the cursor and blocked every click after it** for the rest of the run. `play.py` already called `sim.tick()` per frame; the dashboard's tracker was the one caller strategy 3.2.3's doc flagged as "the obvious next port" and not yet done.
- **Fixed:** ported all of 3.2.3 into `tracker.py`, mirroring `play.py` — `sim.tick()` added to the per-frame loop; `InputSim(hold_s=cfg.click_hold)` instead of the old no-arg call; swapped raw `cv2.VideoCapture`/manual torch softmax for `src.rt.capture.open_camera` (threaded capture) and `src.rt.backends.make_runner` (ONNX-with-parity-check, torch fallback); added `backend`/`torch_threads`/`threaded_capture`/`click_hold` to `TrackerConfig`. `open_camera` raises `SystemExit` on a failed camera open (fine for a CLI, not for the tracker thread's existing try/except Exception contract) — caught explicitly and re-raised as `RuntimeError` so a bad camera index still reports through `/api/status` instead of dying silently in the background thread.
- Added matching `--backend/--torch-threads/--no-threaded-capture/--click-hold` flags to `src/eval_dashboard/server.py`'s CLI, same names/defaults as `play.py`, threaded into the `TrackerConfig` it builds.
- **Verified live, not just read:** booted the dashboard (`week4_tuned_fistvsrest`, the 3.2.2 tuned-champion checkpoint), confirmed `/api/models` lists all five checkpoints with correct `fistvsrest` flags and `DOCS/models/README.md` metadata, POSTed `/api/start` with strategy 3.2.1, and watched `/api/status` go `running: true` at ~30 FPS against the real camera (dry-run, no OS input sent). Separately confirmed `make_runner` picks ONNX for this checkpoint on this machine (`parity max|torch-onnx| = 2.68e-07`). Re-ran the `tick()` unit check standalone to confirm the fix: with `tick()` called, press → hold → release → re-click all work; without it, the button never comes back up. Stopped the tracker via `/api/finish` and killed the test server afterward; no stray `eval_results/` files were written by the dry-run test.
- **What's not verified:** live gameplay against FNAF itself (Step 5 is still open, same caveat as 3.2.3's original entry) — this session only confirms the dashboard's tracker now *runs* the 3.2.3 code path correctly, not that click feel through the dashboard matches `play.py`'s.
- **No FSM values touched** — same boundary as 3.2.3 (Ted's live calls per CLAUDE.md).
- **AI use:** Claude found the stuck-button bug, did the port, and verified it by booting the real server against the real camera, at Ted's request.
- Changed: `src/eval_dashboard/tracker.py`, `src/eval_dashboard/server.py`
