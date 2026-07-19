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
