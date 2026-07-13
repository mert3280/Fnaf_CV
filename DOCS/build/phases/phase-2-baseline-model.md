# Phase 2 — Baseline Model: Frozen-Backbone Transfer Learning (Week 2)

**Phase goal:** Get the first end-to-end transfer-learning run on the board — a pretrained backbone with a **new, custom classification head**, backbone **frozen**, trained on a subset. This establishes the baseline every later improvement is measured against.

> This phase is the heart of the project's learning objective (transfer learning). I drive the layer/freezing decisions; AI assists only with scaffolding and debugging (see [../Claude.md](../Claude.md)).

> **Outcome (2026-07-12):** the first baseline ran `full_frame` (`mobilenetv3_large_100`, frozen) → **57% val / 53% test** ([../results.md](../results.md)). That result — the hand occupies <5% of a typical frame — motivated adopting the **two-stage detect-and-crop pipeline** ([AD-04](../architecture-and-decisions.md)) going into Phase 3. Re-run the frozen baseline with `crop_mode: bbox` as the head of the Phase-3 A/B.

---

## 1. Approach: take a general model, swap the head, freeze the body

Following the timm/Hugging Face transfer-learning workflow:

1. **Load a pretrained backbone** from `timm` (ImageNet weights). Candidate backbones, smallest-first for real-time later:
   - `mobilenetv3_small_100` / `mobilenetv3_large_100` (fast — favored for Phase 4 latency)
   - `efficientnet_b0`
   - `resnet18` (simple, well-understood baseline)
2. **Replace the classification head** with a fresh `nn.Linear` sized to the working subset (~7–8 classes from Phase 1). Use `timm.create_model(name, pretrained=True, num_classes=N)` which swaps the head automatically; verify the new head is randomly initialized.
3. **Freeze the backbone:** set `requires_grad=False` on all params except the head. Confirm via a parameter count (trainable vs. frozen) printed at startup.
4. **Train only the head** — this is fast and establishes how much signal the pretrained features already carry.

## 2. Tasks

### 2.1 Model
- [ ] `src/models/build.py`: `build_model(name, num_classes, freeze_backbone=True)`; return model + the correct input size / normalization from `timm`'s `data_config`.
- [ ] Print trainable-parameter summary (assert only head is trainable when frozen).

### 2.2 Training loop
- [ ] `src/train.py`: config-driven (YAML/argparse) — backbone, LR, batch size, epochs, optimizer (AdamW), scheduler.
- [ ] Standard loop: forward → `CrossEntropyLoss` → backward → step; train/val each epoch.
- [ ] **Checkpointing:** save best-by-val-accuracy to `models/`.
- [ ] **Logging:** TensorBoard scalars (loss, acc, LR); log a few prediction images per epoch.
- [ ] Set seeds; record config with each run for reproducibility.

### 2.3 Evaluation
- [ ] Validation **accuracy + loss curves**.
- [ ] **Confusion matrix** on val — identify which gestures get confused (informs the Phase 1 subset; e.g. drop a class that collides badly).
- [ ] Per-class precision/recall.

## 3. What I am specifically learning here
- *Where* the head is in a `timm` model and how `num_classes` swaps it.
- The difference between **freezing** (no grad) and just using a low LR.
- Reading early curves: is the frozen-feature baseline already strong, or do features need fine-tuning (→ Phase 3)?

## 4. Deliverables
- `build_model` + config-driven `train.py`.
- A trained baseline checkpoint + TensorBoard logs.
- Baseline metrics table (val acc, per-class, confusion matrix) in `DOCS/results.md`.

## 5. Definition of done
A frozen-backbone model trains to completion, beats random by a wide margin, and its metrics are logged and reproducible from a saved config.

## 6. Behind-schedule signal
No training run completes, or no baseline accuracy is recorded by end of Week 2.

## 7. Exit criteria → Phase 3
A reproducible baseline number + a confusion matrix telling me which classes are weak and whether frozen features suffice or fine-tuning is needed.
