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
