# 5-Week Schedule — Gesture-Controlled FNAF

This is the high-level plan mapping core requirements onto five weeks. Each week corresponds to a detailed phase document in [phases/](phases/). Weeks 1–2 cover setup and **data validation**; technical milestones begin in Week 2.

> **Legend** — Each week lists: **Goal**, **Core deliverables**, **Definition of done (DoD)**, and a **"Behind schedule" signal** (the early-warning trip-wire that tells me to escalate, cut scope, or ask for help).

---

## Week 1 — Setup & Data Validation → [Phase 1](phases/phase-1-setup-and-data.md)

- **Goal:** Stand up the project and *prove the data is real and loadable*.
- **Core deliverables:**
  - Repo scaffold, `environment.yml`/`requirements.txt`, reproducible env.
  - HaGRID annotation parser (bboxes + labels + 21 landmarks JSON → records).
  - Image acquisition: download the matching HaGRID **subsample** (fast iteration) and plan for **512px** (final training); document sizes/paths.
  - A working PyTorch `Dataset` + `DataLoader` returning `(image_tensor, label)`.
  - EDA notebook: class balance, example crops with bbox overlays, image size/quality checks.
  - **Finalized gesture → FNAF action map** (design artifact).
- **DoD:** I can run one command/notebook cell and see a labeled, augmented batch rendered to screen.
- **Behind-schedule signal:** By end of week I still cannot load and visualize a single `(image, label)` batch, or I have annotations but no matching images.

## Week 2 — Baseline Model (Frozen Backbone) → [Phase 2](phases/phase-2-baseline-model.md)

- **Goal:** First end-to-end transfer-learning run — get a number on the board.
- **Core deliverables:**
  - `timm` pretrained backbone (e.g. `resnet18`/`mobilenetv3`/`efficientnet_b0`) with a **new classification head** sized to the chosen class subset.
  - **Backbone frozen**; train only the head on a subset.
  - Training loop with checkpointing, TensorBoard logging, train/val accuracy + loss curves.
  - Baseline **confusion matrix** on validation.
- **DoD:** A baseline model trains to completion and beats random by a wide margin; metrics are logged and reproducible.
- **Behind-schedule signal:** No training loop completes a full run, or no baseline accuracy is recorded by end of week.

## Week 3 — Full Training & Fine-Tuning → [Phase 3](phases/phase-3-training-finetuning.md)

- **Goal:** Turn the baseline into a deployable model via staged fine-tuning.
- **Core deliverables:**
  - **Progressive unfreezing** (head → top blocks → full), discriminative learning rates.
  - Augmentation pipeline (flips with label-aware handling, color jitter, rotation, random crop).
  - Hyperparameter sweep (LR, batch size, backbone choice, image size).
  - **Held-out test-set** evaluation (HaGRID test split, by-user) + per-class report.
  - Best model **exported** (TorchScript/ONNX) with an inference smoke test.
- **DoD:** Test accuracy ≥ **90%** on the chosen subset; exported model produces identical predictions to the PyTorch model on a fixed sample.
- **Behind-schedule signal:** Test accuracy stuck below the usable threshold with no diagnosed cause, or export path blocked with no fallback.

## Week 4 — Real-Time Inference & Game Control → [Phase 4](phases/phase-4-realtime-and-control.md)

- **Goal:** Connect a live webcam to actual FNAF inputs.
- **Core deliverables:**
  - OpenCV webcam loop → preprocess → model inference at interactive FPS (target ≥ 15 FPS).
  - **Temporal smoothing + debouncing** (e.g. N-frame majority vote, cooldown per action) to prevent gesture chatter.
  - Gesture → action **controller** with a mode model (Office mode vs. Camera mode).
  - **Input simulation** layer (`pydirectinput`) firing clicks at FNAF's button coordinates; verified registration in-game.
  - On-screen HUD (current gesture, confidence, active mode) for debugging.
- **DoD:** Making a gesture in front of the webcam reliably triggers the intended door/light/camera action inside the running game.
- **Behind-schedule signal:** Gesture recognition works on the HUD but inputs do not reliably register in the actual DirectX game window.

## Week 5 — Integration, Robustness & Demo → [Phase 5](phases/phase-5-integration-demo.md)

- **Goal:** Make it survivable, robust, and presentable.
- **Core deliverables:**
  - Calibration step (lighting/ROI/threshold tuning per session).
  - Robustness passes: lighting variation, background clutter, idle/"no-gesture" handling.
  - **Full hands-free Night 1 playthrough.**
  - Final documentation, architecture diagram, known-limitations list, and a recorded **demo video**.
  - Merge `main` with complete `DOCS/`.
- **DoD:** A full Night 1 of FNAF completed using only hand gestures, captured on video; docs complete.
- **Behind-schedule signal:** Cannot complete a full hands-free night, or recurring misclassifications make play infeasible and remain unresolved.

---

## Risk register & contingencies

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Full HaGRID (716 GB) too large to store locally | High | Never download the full set; use the subsample for iteration + 512px variant (working-subset classes only, ~12 GB) for final training. |
| Local 8 GB VRAM (RTX 4060) insufficient for chosen batch/model | Low | Use a small backbone (mobilenetv3/efficientnet_b0), mixed precision, smaller batch + gradient accumulation, lower input resolution. All training stays local — no cloud. |
| Full-frame classification unreliable (hand too small in frame) | Medium | Insert MediaPipe hand-crop as preprocessing (already in tech stack). |
| `pyautogui` clicks ignored by DirectX FNAF | Medium | Switch to `pydirectinput` (SendInput); test registration in Week 4, not Week 5. |
| Real-time latency too high for FNAF's tension | Medium | Smaller backbone (mobilenet), ONNX Runtime, reduce input resolution. |
| Gesture chatter / false positives | Medium | Debounce + confidence threshold + explicit idle gesture (`mute`). |
| Scope creep (dynamic gestures, FNAF 2+) | Low | Explicitly out of scope — static poses, FNAF 1 only. |

## Out of scope (this project)

- Dynamic / motion-based gestures (temporal models).
- Games beyond FNAF 1.
- Multi-hand or two-player control.
- Mobile/edge deployment.
