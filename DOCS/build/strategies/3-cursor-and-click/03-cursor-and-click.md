# Strategy 3 — Motion-tracked cursor + binary click classifier

**Status: hardened by [Strategy 3.1](03.1-distance-invariant-cursor.md)** — the
first live test (2026-07-15) found a distance-dependent cursor bias; 3.1 fixes
the anchor calibration (size-invariant, unclipped landmarks). Read this page
for the approach; read 3.1 for the calibration fix.

**Built and trained 2026-07-15; in-game test pending.**
The binary model scored **0.9815 test** (palm F1 0.981 / fist 0.982 —
[full report](../../../models/palmfist_frozen_mnv3_large/README.md)); the cursor
mapper (`src/rt/cursor.py`), click FSM (`src/control/click_fsm.py`), input layer
(`src/control/input_sim.py`), and play loop (`src/control/play.py`) are
implemented and offline-verified. Preview everything without touching the real
mouse via `webcam_demo`; drive the game via `play` (dry-run first).
This is the **cursor-control pivot** ([AD-17](../../architecture-and-decisions.md#ad-17--pivot-to-cursor-control-motion-tracked-cursor--binary-click-classifier)).
It keeps the two-stage pipeline of [Strategy 2](../2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md)
/ [2.1](../2-two-stage-mediapipe-crop/02.1-two-stage-robustness-fixes.md) intact
and changes what each stage's output *means*: stage 1 now also **moves the mouse
cursor**, and stage 2 shrinks from an 8-class gesture vocabulary to a **binary
palm/fist** decision — palm = no click, **fist = click**.

## The idea

The old plan asked the player to memorize a gesture vocabulary and asked the
model to tell 8 hand poses apart live. The new plan replaces both with the one
interaction every mouse-driven game already understands — *point and click*:

```
                 ┌───────────────────────── stage 1: MediaPipe (borrowed) ─────────────────────────┐
Webcam frame ─▶  HandLandmarker ──▶ hand position ──▶ absolute mapping + smoothing ──▶ CURSOR MOVE
                        │
                        └─▶ hand bbox ──▶ crop (+pad, == training geometry)
                                               │
                 ┌─────────────────────────────┼──── stage 2: our CNN (transfer-learned) ──────────┐
                                               ▼
                                    MobileNetV3 (binary head) ──▶ palm / fist ──▶ click FSM ──▶ CLICK
```

1. **Stage 1 — detect & track (borrowed, does more work now).** MediaPipe finds
   the hand every frame, exactly as in Strategy 2. Its bbox still feeds the
   classifier crop — and its hand position now **drives the cursor** via an
   absolute frame→screen mapping (spec below).
2. **Stage 2 — classify (ours, simpler task).** The same transfer-learned
   MobileNetV3 pipeline, retrained on HaGRID trimmed to **two classes**:
   `palm` (no click) and `fist` (click). A confirmed palm→fist transition fires
   **one** mouse click at the current cursor position.

The transfer-learning deliverable (AD-05) is unchanged — same backbone, same
freeze→unfreeze mechanics, same crop contract. Only `num_classes` and the class
list change.

## Why this over the gesture vocabulary

- **Covers all of FNAF with zero mapping.** Doors, lights, the monitor, and
  camera-panning are all cursor interactions. No gesture→action table, no
  Office/Camera mode state machine (retires AD-13), nothing to memorize.
- **Concentrates model risk on the easiest possible task.** The live tests of
  Strategy 2 showed real accuracy loss from inter-class confusion and distance.
  `palm` vs `fist` are the two most visually distinct poses in HaGRID (open hand
  vs closed hand); with a binary task nearly every confusion pair disappears,
  and a misread costs one click, not a wrong in-game action.
- **Uses what stage 1 already computes for free.** MediaPipe was already in the
  loop as the cropper; the hand position that drives the cursor is the same
  landmark output. No new model, no new latency.

**The trade-off, honestly:** pointing precision now depends on hand steadiness
and the smoothing filter, and holding an arm up to steer a cursor is more
fatiguing than flashing a static pose. Both are measurable in live testing and
both have tuning levers (control-box size, smoothing strength).

## Cursor control spec (stage 1 → mouse position)

**Decided (Ted, 2026-07-14): absolute mapping** — the hand's position in the
camera frame maps directly to a screen coordinate, like a touchscreen
([AD-19](../../architecture-and-decisions.md#ad-19--absolute-frame-to-screen-cursor-mapping)).
The relative/joystick alternative (hand offset sets cursor *velocity*) is
documented in AD-19 and kept as the fallback if absolute pointing proves too
jittery for FNAF's smaller buttons.

The mapping chain, per frame:

1. **Anchor point.** Use a stable palm-center anchor — the mean of the wrist and
   the four finger-base landmarks (MediaPipe indices `0, 5, 9, 13, 17`) — *not*
   the bbox center. Fingertips move a lot between palm and fist; the palm base
   barely moves, so the cursor doesn't lurch when the player clenches to click.
2. **Mirror.** Flip x so moving the hand right moves the cursor right (the demo
   already mirrors its preview; the mapping must match it).
3. **Control box.** Map a central sub-rectangle of the frame — not the full
   frame — to the full screen, e.g. the middle **~60% horizontally / ~55%
   vertically** (tunable config). The hand reaches every screen edge without
   leaving the camera's comfortable FOV, and frame-edge detection jitter never
   touches the cursor. Positions outside the box clamp to its edge.
4. **Smoothing.** Exponential moving average on the normalized position
   (starting point `alpha ≈ 0.35`, tunable), plus a small **dead-zone** (ignore
   movements under ~0.5% of the screen) so hand tremor doesn't buzz the cursor.
   If EMA proves laggy-vs-jittery at the extremes, upgrade to a One-Euro filter
   — same interface, adaptive alpha.
5. **No hand → cursor freezes.** Detection dropout must not teleport the
   cursor; it stays where it was, and the click FSM disarms until the hand
   returns.

All numbers above are **starting points, not decisions** — they get tuned live
against FNAF's actual button sizes, and the tuned values land in the runtime
config, not code.

## Click spec (stage 2 → mouse button)

**Decided (Ted, 2026-07-14): single click per fist**
([AD-20](../../architecture-and-decisions.md#ad-20--single-click-per-fist-edge-triggered)).
Each *confirmed* palm→fist transition fires exactly **one** click; holding the
fist does nothing more until the hand returns to palm and re-arms. (FNAF needs
no drag or hold — every control is a discrete click or a hover.)

A small three-state FSM, fed one `(label, confidence)` per frame:

```
            K consecutive fist frames ≥ conf_thresh
   ARMED ─────────────────────────────────────────▶ FIRED (click once, at cursor)
     ▲                                                │
     │        K consecutive palm frames ≥ conf_thresh │
     └────────────────────────────────────────────────┘
   (no hand / low confidence from any state → DISARMED; re-arm requires
    confirmed palm — so a dropout can never fire a click)
```

- **Debounce = K consecutive frames** (start `K = 3`, ~0.1–0.2 s at 15–30 FPS)
  above a confidence threshold — one noisy frame never clicks. This inherits
  AD-12's smoothing idea with a simpler mechanism, since there are only two
  labels and one action.
- **Cooldown** (~300 ms, config) after each click as a backstop against
  classifier flicker mid-transition.
- **Re-arm on palm only.** After firing, the FSM will not fire again until it
  has seen K confident palm frames. Fist-hold, dropout, or low confidence
  cannot double-click.
- **Kill-switch** (global hotkey) instantly disables both cursor movement and
  clicking — unchanged, mandatory whenever input simulation runs (AD-14).

Debounce/cooldown magnitudes are **Ted's to tune** during live testing, per the
project guardrails.

## Data & retraining workflow (the palm/fist trim)

The pivot needs **no new downloads** — `palm` and `fist` are already on disk
from the 8-class pull. The workflow
([AD-18](../../architecture-and-decisions.md#ad-18--trim-training-classes-to-palm--fist)):

1. **Trim the class list**: `configs/data.yaml → classes: [palm, fist]`
   (order fixes `label_idx`: `palm=0, fist=1`). The index/split machinery
   re-derives everything from that one change — ~2,179 images
   (palm 1,082 / fist 1,097), split 70/15/15 grouped by user as before (AD-16).
2. **Retrain the same recipe** as `bbox_frozen_mnv3_large`: MobileNetV3-Large,
   ImageNet-pretrained, fresh **2-class head**, `crop_mode: bbox`, frozen
   backbone first (AD-08 Stage A). All Strategy-2.1 levers (perspective/blur
   aug, pad retrain, progressive unfreezing, self-capture fine-tune) still
   apply unchanged — they were built class-count-agnostic.
3. **Expectations**: binary random is 50%; the 8-class model already scored
   0.917, and this task is strictly easier. The live metric that matters is
   **palm↔fist transition reliability at real cursor-use distances**, not test
   accuracy alone — evaluate on val, report test once (AD-10). *What number is
   "good enough to play" is Ted's evaluation call.*
4. **Self-capture** (`src/rt/capture_dataset.py`) gets *more* valuable here:
   only two classes to capture, at exactly the arm-extended distances cursor
   driving uses — the distance gap Strategy 2 diagnosed is now the primary
   domain to cover.

## What carries over from Strategy 2 / 2.1 unchanged

- The two-stage pipeline, the MediaPipe detector, and the **shared crop
  contract** (`padded_bbox_pixels()`, training ≡ runtime — architecture §A.3).
- Checkpoint self-description: the runtime rebuilds the model from the saved
  config, so the 2-class checkpoint slots into the same loader; old 8-class
  checkpoints still run for comparison.
- All six 2.1 robustness fixes (framing hint, live pad, aug knobs, capture
  tool, unfreeze knob, detection-confidence flags).

**New runtime pieces (built 2026-07-15):**

| Piece | File | Notes |
|---|---|---|
| Cursor mapper | [`src/rt/cursor.py`](../../../../src/rt/cursor.py) | anchor → mirror → control box → EMA + dead-zone; freeze on no-hand, glide on re-detect |
| Click FSM | [`src/control/click_fsm.py`](../../../../src/control/click_fsm.py) | ARMED/DISARMED, K-frame debounce, palm re-arm, cooldown |
| Input layer | [`src/control/input_sim.py`](../../../../src/control/input_sim.py) | `pydirectinput` move+click, DPI-aware, `--dry-run`, kill-switch |
| Play loop | [`src/control/play.py`](../../../../src/control/play.py) | composes all of it; global **ESC kill-switch**; HUD preview |
| Demo preview | [`src/rt/webcam_demo.py`](../../../../src/rt/webcam_demo.py) | control box + virtual-cursor crosshair + FSM state, **no real input** |

All tuning values (box, alpha, dead-zone, K, confidence, cooldown) are CLI flags
shared between demo and play, so values tuned safely in the demo transfer
directly. The old mode-aware controller design is retired before ever being
built.

## Open questions

- **Pointing precision vs. FNAF's smallest target** (the camera-flip strip /
  door buttons): can the smoothed absolute cursor hit them reliably? If not:
  bigger control box, stronger smoothing, or fall back to relative mapping
  (AD-19 alternative).
- **Fatigue**: how long can an arm comfortably steer? Affects demo pacing.
- **Transition robustness at cursor distance**: does palm↔fist read cleanly
  with the arm extended toward the camera — the exact regime where Strategy 2
  degraded? (The 2.1 levers + self-capture are the mitigation path.)
- **Does the cursor mapper + FSM fit the ≥15 FPS budget** alongside MediaPipe?
  (Carried from AD-04; measure on day 1 of runtime work.)
