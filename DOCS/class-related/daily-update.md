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
