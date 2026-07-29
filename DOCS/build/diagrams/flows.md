# Flow diagrams

Four diagrams that zoom into pieces of the [system diagram](README.md) that
are single boxes there but have real internal logic worth seeing on their
own: the click state machine, the cursor mapper, one frame's trip through
the real-time loop, and the offline training/export pipeline. Kept as plain
Mermaid-in-Markdown (GitHub renders it natively) rather than a styled HTML
page like the system diagram — four pages would multiply the manual
mmd-to-HTML sync the [README](README.md) already flags as a cost, for
diagrams that change with the code more often than the overall architecture
does.

Source: [`src/control/click_fsm.py`](../../../src/control/click_fsm.py),
[`src/rt/cursor.py`](../../../src/rt/cursor.py),
[`src/rt/capture.py`](../../../src/rt/capture.py), [`src/train.py`](../../../src/train.py).
If a diagram and the code disagree, the code is right — update the diagram.

## Click FSM — palm/fist → single click (AD-20)

Debounced, edge-triggered: a click fires on the one frame a confirmed
palm→fist transition completes, never on a held fist. Re-arming requires
seeing palm again, which makes double-fires structurally impossible rather
than just debounced. The `grace` window (Strategy 3.1.1) exists because
disarming on the *first* ambiguous frame made a slow squeeze physically
unclickable — the in-between poses read as low-confidence and cancelled the
armed state before the finished fist could fire.

```mermaid
stateDiagram-v2
    [*] --> DISARMED

    DISARMED --> DISARMED: fist frame (contrary evidence resets palm streak)
    DISARMED --> ARMED: palm streak reaches K confident frames

    ARMED --> ARMED: palm frame (resets fist streak, stays armed)
    ARMED --> DISARMED: fist streak reaches K + cooldown elapsed — CLICK fires
    ARMED --> DISARMED: ambiguous frames exceed grace in a row

    note right of ARMED
        ambiguous frames (no hand, or confidence below conf_threshold)
        are tolerated up to `grace` in a row without resetting either
        streak -- only sustained ambiguity disarms
    end note
```

K, `conf_threshold`, `cooldown_s`, and `grace` are live-tuning calls (CLAUDE.md) — see `describe()` in `click_fsm.py` for the HUD readout used to tune them.

## Cursor mapping — hand position → screen pixel (AD-19)

Runs every frame in parallel with the classify path. Anchor and box size are
computed independently of each other so a click squeeze (which moves
fingertips a lot) doesn't lurch the cursor, and so the same physical arm
motion moves the cursor the same amount regardless of distance from the
camera (Strategy 3.1.1's motion-invariance goal).

```mermaid
flowchart TD
    LAND["21 hand landmarks (normalized, UNCLIPPED)<br/>from detector.py"]
    NOHAND{"hand detected<br/>this frame?"}
    FREEZE["freeze: return last output unchanged<br/>(EMA state kept -- re-detect glides, never teleports)"]
    ANCHOR["hand_anchor_norm()<br/>source=palm (default): mean of wrist + 4 MCP joints<br/>source=box: landmark-hull center"]
    MIRROR{"mirror_x?<br/>(only if frame wasn't<br/>pre-mirrored)"}
    FLIP["x = 1 - x"]
    SPAN["palm_span()<br/>wrist to middle-MCP distance<br/>(hand's apparent size)"]
    BOXMODE{"box_mode"}
    ADAPTBOX["adaptive control box<br/>EMA-smoothed scale x box_gain<br/>clamped to [0.15, 1.0] -- constant<br/>cursor gain at any distance"]
    FIXEDBOX["fixed control box<br/>box_w x box_h, centered on box_cx/box_cy"]
    NORM["map into box -> normalized [0,1]<br/>screen position, clamp to box edges"]
    EMASTEP{"first detection<br/>ever?"}
    SEED["seed EMA = (nx, ny) directly"]
    BLEND["ema += alpha x (new - ema)"]
    DEAD{"movement from last<br/>output < deadzone?"}
    HOLD["hold previous screen (x, y)<br/>(absorbs hand tremor)"]
    OUT["screen (x, y) pixel<br/>-> input_sim.move()"]

    LAND --> NOHAND
    NOHAND -- no --> FREEZE
    NOHAND -- yes --> ANCHOR
    ANCHOR --> MIRROR
    MIRROR -- yes --> FLIP --> SPAN
    MIRROR -- no --> SPAN
    SPAN --> BOXMODE
    BOXMODE -- adaptive --> ADAPTBOX --> NORM
    BOXMODE -- fixed --> FIXEDBOX --> NORM
    NORM --> EMASTEP
    EMASTEP -- yes --> SEED --> OUT
    EMASTEP -- no --> BLEND --> DEAD
    DEAD -- yes --> HOLD
    DEAD -- no --> OUT
```

## One frame, real-time loop (Strategy 3.2.3)

`ThreadedCapture` runs the blocking camera read (~33.6 ms measured, DSHOW
640x480) on its own thread and keeps exactly one slot: the newest frame. The
main loop never waits on the camera longer than it takes a fresher frame to
arrive, and it never processes a stale one — frames it was too slow to take
are dropped and counted (`dropped` on the HUD). Cursor and classify are
independent per-frame paths that both come off the same detection.

```mermaid
sequenceDiagram
    participant Cam as Webcam
    participant CT as capture thread<br/>(ThreadedCapture._run)
    participant ML as main loop<br/>(webcam_demo.py / play.py)
    participant DET as detector.py<br/>(MediaPipe Hands)
    participant CUR as cursor.py
    participant INF as backends.py<br/>(ONNX / PyTorch)
    participant FSM as click_fsm.py
    participant IN as input_sim.py<br/>(pydirectinput)
    participant GAME as FNAF 1 (DirectX)

    loop every camera frame, background thread
        Cam ->> CT: cv2.VideoCapture.read() (blocks ~33.6ms)
        CT ->> CT: store as newest frame, seq += 1, notify
    end

    loop every loop iteration, main thread
        ML ->> CT: read() -- blocks until seq > last taken
        CT -->> ML: newest unseen frame (stale frames dropped + counted)
        ML ->> DET: process(frame)
        DET -->> ML: 21 landmarks + hull crop (or None)
        par cursor path, every frame
            ML ->> CUR: update(anchor, palm_span)
            CUR -->> ML: screen (x, y)
            ML ->> IN: move(x, y)
        and classify path, every frame
            ML ->> INF: predict(hull crop -- same preprocess as training)
            INF -->> ML: (label, confidence)
            ML ->> FSM: update(label, confidence)
            FSM -->> ML: fire? (True on the single click frame)
            opt fire == True
                ML ->> IN: click() -- press, ~60ms hold, release
            end
        end
        IN ->> GAME: SendInput (move / click)
    end
```

## Offline training and export (AD-08, AD-11, AD-16)

Progressive unfreezing is the core learning objective (AD-08): establish how
much the pretrained backbone already carries before adapting deeper layers.
Stage C is optional and is Ted's call, made by watching the val curve, not
an automatic step.

```mermaid
flowchart TD
    HAGRID["HaGRID annotations (JSON)<br/>palm + fist only, ~2,179 images -- AD-18"]
    SPLIT["split by user_id -- AD-16<br/>70 / 15 / 15 train / val / test<br/>no user appears in more than one split"]
    CROP["crop to bbox + 15% pad"]
    XFORM["build_transforms()<br/>train: augmented -- val/test: deterministic<br/>same function reused live, contract SS A.3"]
    LOADER["DataLoader"]

    HAGRID --> SPLIT --> CROP --> XFORM --> LOADER

    STAGEA["Stage A -- head-only<br/>backbone frozen, fresh 2-class head trained"]
    STAGEB["Stage B -- unfreeze top block(s)<br/>discriminative LR: low on backbone, high on head"]
    STAGEC["Stage C -- optional full unfreeze<br/>small LR + warmup"]
    LOADER --> STAGEA --> STAGEB -.->|"val plateaued? (Ted's call)"| STAGEC

    CKPT["best-val checkpoint<br/>self-describing: backbone + classes + input spec travel with the weights"]
    STAGEB --> CKPT
    STAGEC -.-> CKPT

    EXPORT["export ONNX + TorchScript"]
    PARITY{"parity check -- AD-11<br/>export output matches PyTorch output?"}
    FAIL["fix export config, re-export"]
    REGISTRY[("MLflow Registry<br/>fnaf-click-classifier -- champion alias")]

    CKPT --> EXPORT --> PARITY
    PARITY -- no --> FAIL --> EXPORT
    PARITY -- yes --> REGISTRY
```
