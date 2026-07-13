# Phase 3 — Full Training & Fine-Tuning (Week 3)

**Phase goal:** Turn the frozen baseline into a **deployable** model through staged fine-tuning, augmentation, and honest test-set evaluation — then export it for real-time use.

---

## 0. Confirm the input approach — `full_frame` vs `bbox` crop (AD-04)

The plan adopts the **two-stage detect-and-crop** pipeline ([AD-04](../architecture-and-decisions.md)); this A/B is the controlled check that confirms it before the rest of the phase builds on it.

- [ ] Train two checkpoints **identical except `input.crop_mode`** (`full_frame` vs `bbox`), same seed / epochs / split / backbone.
- [ ] Compare **val accuracy + confusion matrix**; expect `bbox` to clearly beat the 57% `full_frame` baseline. Record both in [../results.md](../results.md).
- [ ] Adopt the winner project-wide. If `bbox` wins (expected), all downstream fine-tuning uses crops and the runtime needs the MediaPipe cropper (Phase 4). If not, flip back per the AD-04 switching note — one config change.

## 1. Progressive unfreezing strategy

The core fine-tuning technique. Unfreeze in stages, lowest learning rate on the earliest (most general) layers:

1. **Stage A** — head only (this is the Phase 2 baseline).
2. **Stage B** — unfreeze the **top backbone block(s)**; train head + top blocks with a **lower LR** on the backbone (discriminative learning rates).
3. **Stage C** — optionally unfreeze the **full backbone** with a small global LR + warmup.

- [ ] Implement layer-group LR ("discriminative LR"): backbone groups get `lr/10`–`lr/100` vs. the head.
- [ ] Use a scheduler (cosine or step) + early stopping on val.
- [ ] Track each stage separately in TensorBoard; compare against the Phase 2 baseline.

## 2. Augmentation (label-aware)

- [ ] Color jitter, brightness/contrast (robustness to webcam lighting — directly helps Phase 5).
- [ ] Small rotation, random resized crop, slight translation.
- [ ] **Horizontal flip with care:** flipping changes the meaning of some gestures (`peace` vs `peace_inverted`, `two_up` vs `two_up_inverted`, left/right semantics). Either exclude flip for the affected classes or remap labels on flip. Document the decision.

## 3. Hyperparameter tuning
- [ ] Sweep: learning rate, batch size, backbone choice, input resolution.
- [ ] Keep a results table; pick the best by **val** accuracy, *then* report **test** once.
- [ ] Watch for over-fitting (val loss rising while train falls) → more augmentation/regularization/dropout or stop earlier.

## 4. Honest evaluation (held-out test set)
- [ ] Evaluate the chosen model **once** on the HaGRID **test split** (`DATA/ann_test`, split by user-id — no subject leakage).
- [ ] Report overall accuracy, **per-class** precision/recall/F1, and a confusion matrix.
- [ ] Note failure modes (which gestures still confuse) and how they affect the control mapping.

## 5. Export for inference
- [ ] Export best model to **TorchScript** and/or **ONNX**.
- [ ] **Parity smoke test:** exported model predictions must match the PyTorch model on a fixed batch (within tolerance).
- [ ] **Record the `crop_mode` + `bbox_pad` in the exported artifact's sidecar config** so Phase 4 loads the matching preprocessing and refuses a mismatch (AD-04 / §A.3).
- [ ] Benchmark single-frame latency (CPU and GPU) — feeds the Phase 4 FPS budget. Note: the runtime also pays the **MediaPipe detector** cost per frame (measured in Phase 4), not just this classifier.

## 6. Deliverables
- Fine-tuned model beating the Phase 2 baseline.
- `DOCS/results.md`: baseline vs. fine-tuned, per-class report, confusion matrices, latency.
- Exported model artifact + parity test.

## 7. Definition of done
Test accuracy ≥ **90%** on the working subset; exported model matches PyTorch predictions on a fixed sample; single-frame latency within the real-time budget.

## 8. Behind-schedule signal
Test accuracy stuck below the usable threshold with no diagnosed cause, or the export path is blocked with no fallback (fallback = run plain PyTorch `eval()` in Phase 4).

## 9. Exit criteria → Phase 4
A single exported model file + documented input pre-processing (size, normalization, **hand-crop policy + `bbox_pad`**, recorded in the artifact's sidecar) + measured per-frame latency. The `full_frame`-vs-`bbox` A/B (§0) is decided and its result logged.
