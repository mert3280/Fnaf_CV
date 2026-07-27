"""The deployable click classifier, wrapped as an MLflow pyfunc model.

Why a pyfunc wrapper instead of `mlflow.pytorch.log_model`? Because the thing a
caller has is an **image**, not a normalized 224x224 tensor. Preprocessing is
half the contract (AD A.3: the eval transform must match training byte-for-byte),
so it belongs *inside* the artifact -- that way the registered model version
carries its own preprocessing and cannot drift from the checkpoint it serves.

Input  (pandas DataFrame): column `image_b64` -- base64-encoded JPEG/PNG bytes of
        an **already-cropped hand** (the two-stage contract, AD-04: MediaPipe /
        the annotation box does the cropping upstream).
Output (pandas DataFrame): one row per image ->
        predicted_class, label_index, confidence, p_click, click

`click` applies the FSM's confidence gate (AD-20, default 0.70): argmax is the
click class AND its probability clears the gate. That mirrors what the live
controller would actually act on, so an endpoint response is comparable to
in-game behaviour rather than being just an argmax.

    python -m src.serve.pyfunc_model --checkpoint models/week4_tuned_fistvsrest/best.pt
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLICK_GATE = 0.70          # AD-20 FSM confidence gate
CLICK_CLASSES = ("fist",)  # the class that means "click"


class GestureClickClassifier(mlflow.pyfunc.PythonModel):
    """Checkpoint + its own preprocessing, as one deployable unit."""

    def load_context(self, context):
        import torch

        from src.data.transforms import build_transforms
        from src.rt.model_loader import load_checkpoint

        self._torch = torch
        self.lm = load_checkpoint(context.artifacts["checkpoint"], device="cpu")
        # train=False -> the exact deterministic eval pipeline used for val/test
        # AND for the live webcam loop. Never re-implemented here (AD A.3).
        self.transform = build_transforms(False, self.lm.size, self.lm.mean, self.lm.std)
        self.click_idx = next(
            (self.lm.classes.index(c) for c in CLICK_CLASSES if c in self.lm.classes), None
        )
        cfg = context.model_config or {}
        self.gate = float(cfg.get("click_gate", CLICK_GATE))

    # -- helpers ----------------------------------------------------------- #
    def _decode(self, b64: str):
        from PIL import Image

        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _batch(self, images):
        return self._torch.stack([self.transform(im) for im in images])

    # -- pyfunc entry point ------------------------------------------------ #
    def predict(self, context, model_input, params=None):
        torch = self._torch
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if isinstance(model_input, pd.DataFrame):
            col = "image_b64" if "image_b64" in model_input.columns else model_input.columns[0]
            b64s = model_input[col].tolist()
        else:  # list / ndarray of base64 strings
            b64s = list(model_input)

        gate = float((params or {}).get("click_gate", self.gate))
        x = self._batch([self._decode(b) for b in b64s])
        with torch.no_grad():
            probs = torch.softmax(self.lm.model(x), dim=1)
        conf, pred = probs.max(dim=1)

        rows = []
        for i in range(len(b64s)):
            idx = int(pred[i])
            p_click = float(probs[i, self.click_idx]) if self.click_idx is not None else float("nan")
            rows.append({
                "predicted_class": self.lm.classes[idx],
                "label_index": idx,
                "confidence": round(float(conf[i]), 6),
                "p_click": round(p_click, 6),
                "click": bool(idx == self.click_idx and p_click >= gate),
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# packaging / registration
# --------------------------------------------------------------------------- #
def example_input(n: int = 1) -> pd.DataFrame:
    """A tiny synthetic input example so the logged model carries a signature."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return pd.DataFrame({"image_b64": [b64] * n})


def log_and_register(checkpoint: str | Path, classes: list[str],
                     registered_model_name: str,
                     click_gate: float = CLICK_GATE) -> str:
    """Log this pyfunc (with the checkpoint as an artifact) inside the ACTIVE run
    and register it. Returns the new model-registry version as a string."""
    from mlflow.models import infer_signature

    ex = example_input()
    out = pd.DataFrame([{"predicted_class": classes[0], "label_index": 0,
                         "confidence": 0.0, "p_click": 0.0, "click": False}])
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=GestureClickClassifier(),
        artifacts={"checkpoint": str(checkpoint)},
        code_paths=[str(REPO / "src")],
        signature=infer_signature(ex, out),
        input_example=ex,
        model_config={"click_gate": click_gate},
        registered_model_name=registered_model_name,
        metadata={"classes": classes, "click_gate": click_gate,
                  "crop_contract": "caller supplies an already-cropped hand (AD-04)"},
    )
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    mine = [v for v in versions if v.source and info.model_uri.split("/")[-1] in str(v.source)]
    version = (mine or sorted(versions, key=lambda v: int(v.version)))[-1].version
    client.set_registered_model_alias(registered_model_name, "champion", version)
    client.set_model_version_tag(registered_model_name, version, "click_gate", str(click_gate))
    return str(version)


def main() -> None:
    """Register a checkpoint standalone (normally `src.tune --stage final` does it)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default="fnaf-click-classifier")
    ap.add_argument("--experiment", default="fnaf-week4-tuning")
    args = ap.parse_args()

    from src.rt.model_loader import load_checkpoint

    lm = load_checkpoint(args.checkpoint, device="cpu")
    mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"register_{Path(args.checkpoint).parent.name}"):
        mlflow.log_params({"checkpoint": args.checkpoint, "backbone": lm.backbone,
                           "classes": ",".join(lm.classes)})
        if lm.val_acc is not None:
            mlflow.log_metric("val_accuracy", lm.val_acc)
        if lm.test_acc is not None:
            mlflow.log_metric("test_accuracy", lm.test_acc)
        v = log_and_register(args.checkpoint, lm.classes, args.name)
    print(f"registered {args.name} version {v}")


if __name__ == "__main__":
    main()
