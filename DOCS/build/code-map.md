# Code Map

A map of every code file under `src/` (plus `configs/`), grouped by subsystem.
The pipeline flows left-to-right: **data prep → training → real-time inference →
game control → evaluation**.

**Big picture:** two-stage design (AD-04) — MediaPipe *detects/tracks* the hand
(→ cursor position + crop), a transfer-learned CNN *classifies* palm/fist. The
recurring theme is the **train/serve preprocessing contract**: `padded_bbox_pixels`
and `build_transforms` are deliberately shared between offline training and the
live loop so "great test accuracy, useless live" can't happen (AD-A.3).

```
webcam ─► MediaPipe detect ─┬─► palm anchor ─► CursorMapper ─► move cursor
                            └─► crop ─► CNN (palm/fist) ─► ClickFSM ─► click
```

## `src/data/` — HaGRID data pipeline
Import-safe where possible (no torch/timm in the annotation/split modules).

| File | Role |
|---|---|
| [config.py](../../src/data/config.py) | Typed loader for `configs/data.yaml` (`DataConfig`, `SplitCfg`, `AugmentCfg`). Single documented home for defaults; import-safe. |
| [hagrid_annotations.py](../../src/data/hagrid_annotations.py) | Parses HaGRID annotation JSONs (normalized COCO bbox + 21 landmarks) into one pandas DataFrame joined to images on disk. `build_index()`. No pixels read. |
| [splits.py](../../src/data/splits.py) | 70/15/15 train/val/test split **grouped by `user_id`** so no subject leaks across splits (AD-10). Deterministic from (index, seed); writes an auditable manifest. |
| [transforms.py](../../src/data/transforms.py) | Builds train (RRC + label-aware aug) and eval (resize→center-crop→normalize) pipelines. The eval pipeline is the byte-for-byte contract with `rt/preprocess.py` (AD-A.3). |
| [dataset.py](../../src/data/dataset.py) | `HagridDataset` → `(image_tensor, label_idx)`, plus `build_dataloaders()` wiring config→index→split→transforms→loaders. `padded_bbox_pixels()` is the **single source of truth for crop geometry**, shared with the live detector. |
| [download_subsample.py](../../src/data/download_subsample.py) | Fetches the ~100-img/class subsample (the held-out TEST set) from HaGRID ZIPs via HTTP range requests. |
| [download_train_val.py](../../src/data/download_train_val.py) | Fetches the train/val pool, **excluding** every subsample UUID (plus MD5 verify) to prevent test leakage. |

## `src/models/` — model construction

| File | Role |
|---|---|
| [build.py](../../src/models/build.py) | Builds a pretrained `timm` backbone (MobileNetV3 default) with a fresh N-class head + freeze control (`freeze_all_but_head`, `unfreeze_last_n_blocks`). Also `get_data_config`, `param_summary`, `pooled_features`. |

## `src/train.py` — training
Config-driven training loop. Two paths: fast **linear-probe** (cache frozen-backbone
features, train head on cached tensors) and standard end-to-end (for
unfrozen/augmented phases). Saves the best-by-val-acc checkpoint (self-describing:
backbone + classes + input spec) and writes an honest metrics report to
`DOCS/results.md`.

## `src/rt/` — real-time inference (webcam → prediction)

| File | Role |
|---|---|
| [detector.py](../../src/rt/detector.py) | Stage 1: MediaPipe Hand Landmarker finds the hand, pads/clamps its box with the **same geometry as training** to produce the crop. |
| [preprocess.py](../../src/rt/preprocess.py) | Runtime preprocessing — calls the same `build_transforms(train=False)` so it can't drift from training. `RoiCropper` for full-frame models. |
| [cursor.py](../../src/rt/cursor.py) | Stage 1→mouse position (AD-19): palm-anchor → control box → clamp → EMA → dead-zone → screen (x,y). Adaptive box scales with hand size for distance-invariant gain. |
| [model_loader.py](../../src/rt/model_loader.py) | Loads any self-describing `best.pt` into a ready classifier — swap models by pointing at a different file. |
| [webcam_demo.py](../../src/rt/webcam_demo.py) | Live single-model demo with top-k overlay, framing hints, live pad tuning `[`/`]`, cursor preview (no real mouse). |
| [compare_demo.py](../../src/rt/compare_demo.py) | Runs several checkpoints in one process on the **same frame** (webcam is single-owner) and tiles their outputs for fair comparison. |
| [capture_dataset.py](../../src/rt/capture_dataset.py) | Records your own hand crops (same detector+pad) into ImageFolder layout for personal fine-tuning. |

## `src/control/` — game control (prediction → clicks)

| File | Role |
|---|---|
| [click_fsm.py](../../src/control/click_fsm.py) | Stage 2→button: debounced, edge-triggered click FSM (AD-20). One click per palm→fist transition; re-arm requires palm; grace window tolerates brief ambiguous frames. |
| [input_sim.py](../../src/control/input_sim.py) | `pydirectinput` SendInput wrapper: `move_to`, `click`, `kill`. DPI-aware, PAUSE=0, failsafe off, dry-run mode. |
| [strategies.py](../../src/control/strategies.py) | Named FSM tuning presets (3.2: conf≥0.70/K=3; 3.2.1: conf≥0.80/K=2). Single source of truth shared by play + dashboard. |
| [play.py](../../src/control/play.py) | **The full Strategy-3 control loop** composing everything: webcam→detect→cursor+CNN→FSM→click. ESC global kill-switch; `--dry-run`. |
| [keyboard_mock_controller.py](../../src/control/keyboard_mock_controller.py) | Day-1 de-risk: drive FNAF with keyboard-mocked gestures through the same `Controller` entry point, before any ML. Kept as record + skeleton. |

## `src/eval_dashboard/` — usability evaluation

| File | Role |
|---|---|
| [server.py](../../src/eval_dashboard/server.py) | Web app serving an obstacle course (precision / rapid-click / tracer); owns the tracker driving real cursor+clicks. `/api/finish` (kill-switch), `/api/results`. |
| [tracker.py](../../src/eval_dashboard/tracker.py) | `play.py`'s per-frame pipeline lifted into a start/stop background thread the server owns. `stop()` is the kill-switch. |
| [results.py](../../src/eval_dashboard/results.py) | Persists one run (obstacle timings + 5-question survey) to `.docx` (or `.txt` fallback) plus a `.json` sibling. |

## `src/eval_openset.py` — open-set false-click eval
Open-set false-click evaluation (AD-21): runs checkpoints over whole gesture
folders and reports how often each fires a false `fist` on unseen poses (`ok` is
the key A/B row, held out of both models' training).

## Configs (`configs/*.yaml`)
`data.yaml` (pipeline defaults) plus per-experiment training configs: `baseline`,
`bbox_frozen`, `palmfist_frozen`, `data_fistvsrest`, `fistvsrest_frozen`.
