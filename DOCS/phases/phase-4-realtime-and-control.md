# Phase 4 — Real-Time Inference & Game Control (Week 4)

**Phase goal:** Connect a live webcam to the trained model and drive **actual FNAF inputs**. This is where an offline classifier becomes an interactive system — the project's second core learning objective.

---

## 1. Real-time pipeline

```
Webcam (OpenCV) → preprocess → model → smoothing/debounce → gesture event
       → mode-aware controller → simulated mouse input → FNAF
                       ↘ HUD overlay (gesture, confidence, mode)
```

### 1.1 Capture & preprocess
- [ ] OpenCV `VideoCapture` loop; mirror the frame for natural interaction.
- [ ] Match Phase 3 preprocessing exactly (size, normalization, crop vs. full-frame).
- [ ] **Optional MediaPipe crop:** if full-frame accuracy is poor live, detect the hand and crop before classifying (already in the stack). Decide based on live behavior.
- [ ] Target **≥ 15 FPS** end-to-end; use the exported model (ONNX Runtime / TorchScript) and a small backbone.

### 1.2 Temporal smoothing & debounce (critical for usability)
A raw per-frame classifier "chatters." Stabilize it:
- [ ] **N-frame majority vote** (e.g. gesture must hold across the last 5–7 frames).
- [ ] **Confidence threshold** — ignore low-confidence frames.
- [ ] **Per-action cooldown** — after firing a door toggle, suppress repeats for ~0.5–1 s.
- [ ] **Explicit idle gesture (`mute`)** as the rest state so "no command" is a first-class signal, not a misfire.
- [ ] Treat actions as **edge-triggered** (gesture *enters* a state) vs. level-triggered, except for lights which may be **hold-to-activate**.

## 2. Gesture → action controller
- [ ] `src/control/controller.py`: a small **state machine** with `Office` and `Camera` modes (mapping from [phase-1](phase-1-setup-and-data.md) §4).
- [ ] Map debounced gesture events → high-level actions (toggle left door, raise monitor, next cam, …).
- [ ] Mode transitions: `palm` raises the monitor (→ Camera), `fist` in Camera lowers it (→ Office).

## 3. Input simulation (de-risk EARLY in the week)
FNAF is DirectX; synthetic input is the highest-risk integration point.
- [ ] **Test `pydirectinput` click registration in FNAF on Day 1 of this phase**, before building the full controller. If clicks don't register, escalate immediately (window focus, admin rights, SendInput vs. event injection).
- [ ] Calibrate **screen coordinates** for each FNAF button (left/right door, left/right light, camera panel hot-zone, individual camera buttons) — store in `configs/fnaf_layout.yaml` keyed to a known resolution/window position.
- [ ] Map actions → coordinate clicks (and any held inputs for lights).
- [ ] Safety: a global **kill-switch** key to disable simulated input instantly.

## 4. Debug HUD
- [ ] Overlay window showing current raw gesture, smoothed gesture, confidence, active mode, and last action fired. Essential for tuning thresholds and for the demo.

## 5. Deliverables
- Real-time webcam → gesture → action loop at interactive FPS.
- Mode-aware controller + `fnaf_layout.yaml` coordinate map.
- Verified in-game input registration + kill-switch.
- Debug HUD.

## 6. Definition of done
A gesture in front of the webcam reliably triggers the intended door/light/camera action **inside the running FNAF window**.

## 7. Behind-schedule signal
Recognition shows correctly on the HUD but inputs do **not** reliably register in the actual game (the reason to test input simulation on Day 1, not at week's end).

## 8. Exit criteria → Phase 5
Each mapped gesture reliably produces its in-game action in isolation; thresholds/cooldowns set to sane defaults; ready for full-night playtesting and robustness work.
