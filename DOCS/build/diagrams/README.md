# Diagrams

Sensing-to-actuation system diagram: offline training pipeline, the Week-4
orchestration/deployment layer, and the online real-time loop that plays the
game. Reflects AD-25 (2026-07-28) — see
[architecture-and-decisions.md](../architecture-and-decisions.md) for the full
decision record.

- [architecture-diagram.html](architecture-diagram.html) — standalone designed
  page (legend, the §A.3 train/serve contract, and the key-decisions table).
  Open directly in a browser, no server needed.
- [architecture-diagram.mmd](architecture-diagram.mmd) — raw Mermaid source
  used by the page above; edit this and re-paste into the `<pre class="mermaid">`
  block in the HTML to keep both in sync.
- [flows.md](flows.md) — four zoomed-in flow diagrams for pieces that are a
  single box above but have real internal logic: the click FSM (AD-20), the
  cursor mapper (AD-19), one frame's trip through the real-time loop
  (Strategy 3.2.3's threaded capture), and the offline training/export
  pipeline (AD-08, AD-11, AD-16). Plain Mermaid-in-Markdown, no HTML page —
  see the top of that file for why.

Rendered directly below from the same source (GitHub renders Mermaid natively
in Markdown):

```mermaid
flowchart TD
  subgraph OFFLINE["OFFLINE · training — src/data, src/models, train.py"]
    direction TB
    HAGRID["HaGRID annotations<br/>palm + fist only · AD-18"]
    DATAPIPE["data pipeline<br/>parse → split by user_id (AD-16)<br/>crop to bbox +15% pad → transform"]
    BACKBONE["timm backbone<br/>MobileNetV3 + fresh 2-class head"]
    TRAINSTEP["train.py<br/>freeze → progressive unfreeze · AD-08"]
    EXPORT["export<br/>ONNX + TorchScript, parity-checked · AD-11"]
    HAGRID --> DATAPIPE --> BACKBONE --> TRAINSTEP --> EXPORT
  end

  subgraph ORCH["ORCHESTRATION & DEPLOYMENT · Week 4 — src/pipeline, src/serve"]
    direction TB
    DAG["pipeline/dag.py<br/>topo order · freshness cache · retries · AD-23"]
    TASKS["pipeline/tasks.py<br/>ingest → split → tune → register → serve-check"]
    TUNEBOX["tune.py<br/>Optuna TPE, cached-feature arm + control run · AD-22"]
    REGISTRY[("MLflow Registry<br/>fnaf-click-classifier · champion alias · AD-24")]
    SERVE["serve/app.py<br/>Flask /predict — validation only, not in RT loop"]
    DAG --> TASKS --> TUNEBOX --> REGISTRY
    REGISTRY --> SERVE
  end

  EXPORT --> REGISTRY

  subgraph ONLINE["ONLINE · real-time loop — 28.3 FPS measured — src/rt, src/control"]
    direction TB
    WEBCAM["webcam"]
    CAPTURE["capture.py<br/>threaded, newest-frame-only grabber"]
    DETECT["detector.py<br/>MediaPipe Hands → 21-landmark hull<br/>pad 0.35 · AD-25"]
    CURSOR["cursor.py<br/>palm anchor → adaptive control box<br/>EMA + dead-zone · AD-19"]
    PRE["preprocess.py<br/>same build_transforms as training · §A.3"]
    INFER["backends.py<br/>ONNX (default) / PyTorch<br/>fist vs. not_fist · AD-21"]
    FSM["click_fsm.py<br/>K-frame debounce, edge-triggered<br/>re-arm on palm · AD-20"]
    INPUT["input_sim.py<br/>pydirectinput SendInput<br/>move every frame + ~60ms click hold · AD-14"]
    FNAF[("FNAF 1<br/>real Steam game, DirectX")]
    WEBCAM --> CAPTURE --> DETECT
    DETECT -->|hand position, every frame| CURSOR
    DETECT -->|hull crop| PRE --> INFER --> FSM
    CURSOR --> INPUT
    FSM --> INPUT --> FNAF
  end

  REGISTRY -. loads champion .-> INFER

  classDef offline fill:#E4EEEE,stroke:#1F6F6F,stroke-width:1.5px,color:#123B3B
  classDef online fill:#FBEEDF,stroke:#B5651D,stroke-width:1.5px,color:#5C3A12
  classDef orch fill:#E9E9F3,stroke:#5B5F97,stroke-width:1.5px,color:#2C2E52
  classDef store fill:#1A2129,stroke:#1A2129,stroke-width:1.5px,color:#F5F3EE

  class HAGRID,DATAPIPE,BACKBONE,TRAINSTEP,EXPORT offline
  class WEBCAM,CAPTURE,DETECT,CURSOR,PRE,INFER,FSM,INPUT online
  class DAG,TASKS,TUNEBOX,SERVE orch
  class REGISTRY,FNAF store
```
