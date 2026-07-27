"""HTTP inference endpoint for the registered click classifier.

Serves whatever version the MLflow registry says is `champion` (or an explicit
`models:/name/N` URI), so deploying a newly tuned model is a registry alias move
plus a restart -- no code edit, no checkpoint path baked into the server.

    python -m src.serve.app                                   # champion alias
    python -m src.serve.app --model-uri models:/fnaf-click-classifier/1
    python -m src.serve.app --checkpoint models/week4_tuned_fistvsrest/best.pt

Routes
    GET  /health   liveness + which model version is loaded
    GET  /model    the served model's metadata (classes, input spec, gate)
    POST /predict  {"image_b64": "..."} | {"images": ["...", ...]}
                   or a multipart file upload under `file`
                   -> per-image predicted_class / confidence / p_click / click

The image must be an **already-cropped hand** (AD-04: MediaPipe or the HaGRID
box crops upstream) -- the same contract as the pyfunc artifact.

NOT the real-time path. The live game loop calls the model in-process
(src/control/play.py): a fist must be classified inside a ~33 ms frame budget and
the model alone costs ~35-40 ms on this CPU, so an HTTP round-trip per frame is
not affordable. This endpoint exists for validation, batch scoring, and to give
the model a versioned network-callable surface. See the Week-4 report §4.
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

from flask import Flask, jsonify, request

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_URI = "models:/fnaf-click-classifier@champion"

app = Flask(__name__)
STATE: dict = {}


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_from_registry(model_uri: str) -> dict:
    """Load the pyfunc flavour of a registered model version."""
    import mlflow

    mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
    model = mlflow.pyfunc.load_model(model_uri)
    meta = model.metadata.metadata or {}
    version = "unknown"
    try:  # resolve the alias/URI to a concrete registry version for /health
        client = mlflow.MlflowClient()
        name = model_uri.split("models:/")[-1].split("@")[0].split("/")[0]
        if "@" in model_uri:
            version = client.get_model_version_by_alias(
                name, model_uri.split("@")[-1]).version
        elif model_uri.split("/")[-1].isdigit():
            version = model_uri.split("/")[-1]
    except Exception as exc:  # a served model with an unresolvable version is
        app.logger.warning("could not resolve registry version: %s", exc)
    return {"predict": lambda df: model.predict(df), "source": model_uri,
            "version": version, "classes": meta.get("classes"),
            "click_gate": meta.get("click_gate"), "backend": "mlflow-pyfunc"}


def load_from_checkpoint(path: str) -> dict:
    """Fallback: serve a checkpoint directly, without the registry.

    Uses the SAME pyfunc class as the registered artifact, so a response here is
    identical to a response from the registry-served model.
    """
    from mlflow.pyfunc import PythonModelContext

    from src.serve.pyfunc_model import CLICK_GATE, GestureClickClassifier

    pm = GestureClickClassifier()
    pm.load_context(PythonModelContext(artifacts={"checkpoint": str(path)},
                                       model_config={"click_gate": CLICK_GATE}))
    return {"predict": lambda df: pm.predict(None, df), "source": str(path),
            "version": "unregistered", "classes": pm.lm.classes,
            "click_gate": pm.gate, "backend": "checkpoint"}


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_source": STATE["source"],
                    "model_version": STATE["version"], "backend": STATE["backend"]})


@app.get("/model")
def model_info():
    return jsonify({k: STATE[k] for k in
                    ("source", "version", "classes", "click_gate", "backend")})


@app.post("/predict")
def predict():
    import pandas as pd

    t0 = time.perf_counter()
    images: list[str] = []
    if request.files.get("file"):  # multipart upload
        images = [base64.b64encode(request.files["file"].read()).decode()]
    else:
        payload = request.get_json(silent=True) or {}
        if isinstance(payload.get("images"), list):
            images = payload["images"]
        elif payload.get("image_b64"):
            images = [payload["image_b64"]]

    if not images:
        return jsonify({"error": "send JSON {'image_b64': <base64>} or "
                                 "{'images': [...]}, or a multipart `file`"}), 400
    try:
        out = STATE["predict"](pd.DataFrame({"image_b64": images}))
    except Exception as exc:  # bad base64 / not an image / wrong shape
        app.logger.exception("prediction failed")
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 422

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    preds = out.to_dict(orient="records")
    return jsonify({
        "predictions": preds,
        "model_version": STATE["version"],
        "click_gate": STATE["click_gate"],
        "latency_ms": latency_ms,
        "latency_ms_per_image": round(latency_ms / len(images), 2),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-uri", default=DEFAULT_MODEL_URI,
                    help=f"registry URI (default {DEFAULT_MODEL_URI})")
    ap.add_argument("--checkpoint", default=None,
                    help="serve a checkpoint directly instead of the registry")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    STATE.update(load_from_checkpoint(args.checkpoint) if args.checkpoint
                 else load_from_registry(args.model_uri))
    print(f"serving {STATE['source']} (version {STATE['version']}, "
          f"classes={STATE['classes']}) on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
