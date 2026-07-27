"""The pipeline's tasks and the two DAGs they form.

Every task is a thin wrapper over a module that already exists and can still be
run by hand -- the pipeline adds order, freshness, and an audit trail, it does not
become a second implementation of anything.

Expensive tasks (`tune_search`, `tune_structural`) run as SUBPROCESSES, the same
isolation an Airflow operator would give: a crash or an OOM inside a 2-hour
training run cannot take the orchestrator's own bookkeeping with it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.pipeline.dag import REPO, DAG, Context, PipelineError, Task

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run_module(module: str, *args: str) -> None:
    """`python -m module args...` from the repo root; raise on nonzero exit."""
    cmd = [sys.executable, "-u", "-m", module, *args]
    print(f"    $ {' '.join(cmd[2:])}")
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
    if tail:
        print("\n".join(f"      {ln}" for ln in tail.splitlines()))
    if proc.returncode != 0:
        err = "\n".join((proc.stderr or "").strip().splitlines()[-8:])
        raise PipelineError(f"`{module}` exited {proc.returncode}\n{err}")


def _state(ctx: Context, name: str) -> dict | None:
    p = ctx.out_dir / "tuning" / name
    return json.loads(p.read_text()) if p.exists() else None


def _tune_cfg(ctx: Context) -> dict:
    return yaml.safe_load(ctx.tune_config.read_text())


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
def ingest_check(ctx: Context) -> dict:
    """Fail early and specifically if the inputs aren't on disk."""
    from src.data.config import DataConfig

    cfg = DataConfig.from_yaml(ctx.data_config)
    counts, missing = {}, []
    for images, ann in cfg.sources:
        if not (REPO / ann).is_dir():
            missing.append(f"annotations dir {ann} (see DOCS/build/data/data.md)")
        for gesture in cfg.source_classes:
            d = REPO / images / gesture
            if not d.is_dir():
                missing.append(f"images dir {images / gesture} "
                               f"(python -m src.data.download_train_val --classes {gesture})")
            else:
                counts[f"{Path(images).name}/{gesture}"] = len(list(d.glob("*.jpg")))
    if missing:
        raise PipelineError("missing inputs:\n      - " + "\n      - ".join(missing))
    total = sum(counts.values())
    print(f"      {total} images across {len(counts)} folders")
    return {"image_counts": counts, "total_images": total}


def index_and_split(ctx: Context) -> dict:
    """Index -> user-grouped 70/15/15 split -> write the manifest (the data version).

    The manifest hash IS the data version every downstream artifact is tagged
    with, so a run's numbers can always be traced to an exact dataset state.
    """
    from src.data.config import DataConfig
    from src.data.dataset import build_datasets
    from src.tune import dataset_fingerprint

    cfg = DataConfig.from_yaml(ctx.data_config)
    _, splits, _ = build_datasets(cfg, backbone=None, write_split_manifest=True,
                                  augment_train=False)
    # build_datasets() already asserts UUID/user disjointness (src/data/splits.py).
    version = dataset_fingerprint(REPO / cfg.manifest)
    sizes = {k: int(len(getattr(splits, k))) for k in ("train", "val", "test")}
    print(f"      split {sizes} · data_version {version}")
    return {"data_version": version, "sizes": sizes,
            "manifest": str(Path(cfg.manifest).as_posix())}


def cache_features(ctx: Context) -> dict:
    """One frozen-backbone pass -> a reusable feature cache (Arm A's fast path)."""
    import torch

    from src.data.config import DataConfig
    from src.tune import dataset_fingerprint, feature_cache_key, load_or_build_features

    cfg = _tune_cfg(ctx)
    data_cfg = DataConfig.from_yaml(ctx.data_config)
    data_cfg.loader.num_workers = 0
    fp = dataset_fingerprint(REPO / data_cfg.manifest)
    key = feature_cache_key(cfg, data_cfg, fp)
    feats = load_or_build_features(cfg, data_cfg, "cpu", ctx.out_dir / "cache", key)
    shapes = {k: list(v[0].shape) for k, v in feats.items()}
    del feats
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"cache_key": key, "shapes": shapes,
            "path": f"{ctx.out_dir.name}/cache/feats_{key}.pt"}


def tune_search(ctx: Context) -> dict:
    """Arm A: the TPE hyper-parameter search (+ baseline ablation, seed repeats)."""
    _run_module("src.tune", "--stage", "a", "--config",
                str(ctx.tune_config.relative_to(REPO).as_posix()))
    st = _state(ctx, "arm_a.json")
    if not st:
        raise PipelineError("arm_a.json was not written")
    return {"trials": len(st["trials"]), "best": st["best"]["name"],
            "best_val_acc": st["best"]["val_acc_mean"],
            "baseline_timm_init": st["baseline"]["timm_init"]["val_acc"],
            "baseline_fixed_init": st["baseline"]["fixed_init"]["val_acc"]}


def tune_structural(ctx: Context) -> dict:
    """Arm B: the end-to-end unfreeze/augmentation grid (hours on CPU)."""
    _run_module("src.tune", "--stage", "b", "--config",
                str(ctx.tune_config.relative_to(REPO).as_posix()))
    st = _state(ctx, "arm_b.json")
    if not st:
        raise PipelineError("arm_b.json was not written")
    return {"runs": {r["run"]: r["val_acc"] for r in st["runs"]}}


def select_and_register(ctx: Context) -> dict:
    """Pick the best-by-val config, read the test split ONCE, register the model."""
    _run_module("src.tune", "--stage", "final", "--config",
                str(ctx.tune_config.relative_to(REPO).as_posix()))
    st = _state(ctx, "final.json")
    if not st:
        raise PipelineError("final.json was not written")
    return {"winner": f"{st['winner']['arm']}/{st['winner']['name']}",
            "val_acc": st["val_acc"], "test_acc": st["test_acc"],
            "model_version": st["model_version"], "run_id": st["run_id"],
            "data_version": st["data_version"]}


def evaluate(ctx: Context) -> dict:
    """Final metrics + report figures for the registered checkpoint."""
    _run_module("src.eval_final", "--checkpoint",
                str((ctx.out_dir / "best.pt").relative_to(REPO).as_posix()),
                "--data-config", str(ctx.data_config.relative_to(REPO).as_posix()))
    p = ctx.week_dir / "final-metrics.json"
    if not p.exists():
        raise PipelineError(f"{p} was not written")
    m = json.loads(p.read_text())
    return {"test_accuracy": m["test_accuracy"], "macro_f1": m["macro_f1"],
            "figures": sorted(f.name for f in (ctx.week_dir / "figures").glob("*.png"))}


def export_onnx(ctx: Context) -> dict:
    """Export the served model to ONNX and verify numerical parity (AD-11).

    A second serving artifact for the dependency-light runtime path; the parity
    check is what makes it safe to swap in (a silent export mismatch would look
    exactly like a model regression).
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    from src.rt.model_loader import load_checkpoint

    ckpt = ctx.out_dir / "best.pt"
    if not ckpt.exists():
        raise PipelineError(f"no checkpoint at {ckpt} -- run select_and_register first")
    lm = load_checkpoint(ckpt, device="cpu")
    path = ctx.out_dir / "model.onnx"
    dummy = torch.randn(1, 3, lm.size, lm.size)
    # `external_data=False` keeps the weights INSIDE the .onnx (the default splits
    # them into a sidecar .onnx.data, which is easy to lose when copying a model
    # around). stdout is made utf-8-tolerant because torch's exporter logs emoji
    # and this box's console is cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    torch.onnx.export(lm.model, (dummy,), str(path), input_names=["input"],
                      output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      verbose=False, external_data=False)
    with torch.no_grad():
        torch_out = lm.model(dummy).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    if max_diff > 1e-3:
        raise PipelineError(f"ONNX parity check FAILED: max|torch-onnx| = {max_diff:.2e}")
    # a dynamic batch axis is part of the contract -- check it, don't assume it
    batch = sess.run(["logits"], {"input": torch.randn(4, 3, lm.size, lm.size).numpy()})[0]
    if batch.shape != (4, len(lm.classes)):
        raise PipelineError(f"ONNX batch axis is not dynamic: got {batch.shape}")
    print(f"      parity ok (max abs diff {max_diff:.2e}), "
          f"{path.stat().st_size/1e6:.1f} MB, dynamic batch ok")
    return {"path": str(path.relative_to(REPO).as_posix()),
            "max_abs_diff_vs_torch": max_diff,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "dynamic_batch": True}


def serve_smoke(ctx: Context) -> dict:
    """Prove the endpoint serves the registered model on a REAL crop.

    Uses Flask's test client (no port, no daemon) so the check is deterministic in
    CI/CLI, then asserts the response is a well-formed click decision.
    """
    import base64
    import io

    from PIL import Image

    from src.data.config import DataConfig
    from src.data.dataset import padded_bbox_pixels
    from src.data.hagrid_annotations import build_index
    from src.serve import app as serve_app

    cfg = DataConfig.from_yaml(ctx.data_config)
    images, ann = cfg.sources[0]
    idx = build_index(images, ann, ["fist"], {"fist": 0})
    row = idx.iloc[0]
    img = Image.open(row["path"]).convert("RGB")
    w, h = img.size
    img = img.crop(padded_bbox_pixels(row["bbox"], w, h, cfg.input.bbox_pad))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()

    uri = f"models:/{ctx.registered_model}@champion"
    try:
        serve_app.STATE.update(serve_app.load_from_registry(uri))
    except Exception as exc:
        raise PipelineError(f"could not load {uri} from the registry: {exc}") from exc

    client = serve_app.app.test_client()
    health = client.get("/health").get_json()
    resp = client.post("/predict", json={"image_b64": b64})
    if resp.status_code != 200:
        raise PipelineError(f"/predict returned {resp.status_code}: {resp.get_data(as_text=True)}")
    body = resp.get_json()
    pred = body["predictions"][0]
    if pred["predicted_class"] != "fist":
        raise PipelineError(f"served model called a real fist '{pred['predicted_class']}' "
                            f"({pred['confidence']:.3f}) -- serving path is suspect")
    print(f"      /predict -> {pred['predicted_class']} conf {pred['confidence']:.3f} "
          f"click={pred['click']} in {body['latency_ms']} ms")
    return {"health": health, "prediction": pred, "latency_ms": body["latency_ms"],
            "model_version": body["model_version"]}


# --------------------------------------------------------------------------- #
# freshness predicates (idempotency)
# --------------------------------------------------------------------------- #
def _fresh_manifest(ctx: Context) -> bool:
    from src.data.config import DataConfig

    return (REPO / DataConfig.from_yaml(ctx.data_config).manifest).exists()


def _fresh_cache(ctx: Context) -> bool:
    return any((ctx.out_dir / "cache").glob("feats_*.pt"))


def _fresh_search(ctx: Context) -> bool:
    return _state(ctx, "arm_a.json") is not None


def _fresh_structural(ctx: Context) -> bool:
    return _state(ctx, "arm_b.json") is not None


def _fresh_final(ctx: Context) -> bool:
    """Fresh only if the registered model was built from TODAY's data version."""
    from src.data.config import DataConfig
    from src.tune import dataset_fingerprint

    st = _state(ctx, "final.json")
    if not st:
        return False
    manifest = REPO / DataConfig.from_yaml(ctx.data_config).manifest
    return manifest.exists() and st.get("data_version") == dataset_fingerprint(manifest)


def _fresh_eval(ctx: Context) -> bool:
    m, ckpt = ctx.week_dir / "final-metrics.json", ctx.out_dir / "best.pt"
    return m.exists() and ckpt.exists() and m.stat().st_mtime >= ckpt.stat().st_mtime


def _fresh_onnx(ctx: Context) -> bool:
    o, ckpt = ctx.out_dir / "model.onnx", ctx.out_dir / "best.pt"
    return o.exists() and ckpt.exists() and o.stat().st_mtime >= ckpt.stat().st_mtime


# --------------------------------------------------------------------------- #
# the DAGs
# --------------------------------------------------------------------------- #
TRAINING_DAG = DAG(
    name="training",
    description="HaGRID images -> tuned, evaluated, registered, servable click classifier.",
    trigger=("event-based, not scheduled: run on a data event (new HaGRID pull or "
             "new self-capture session) or a recipe change. `--if-data-changed` "
             "makes that explicit -- it no-ops when the data fingerprint already "
             "matches the registered model's. No cron: the dataset is static, so a "
             "nightly retrain would burn ~2.5 h of CPU to reproduce the same weights."),
    tasks=[
        Task("ingest_check", "Verify annotation dirs and per-gesture image folders "
             "exist on disk; fail with the exact download command if not.",
             ingest_check, retries=0),
        Task("index_and_split", "Index every image, draw the user-grouped 70/15/15 "
             "split, assert no UUID/user leakage, write the split manifest whose "
             "hash is the data version.",
             index_and_split, depends_on=("ingest_check",), is_fresh=_fresh_manifest),
        Task("cache_features", "One frozen-backbone forward pass over all three "
             "splits -> cached 1280-d features (makes the head search ~1000x cheaper).",
             cache_features, depends_on=("index_and_split",), is_fresh=_fresh_cache),
        Task("tune_search", "Arm A: 60-trial TPE search over the head/optimizer "
             "hyper-parameters on cached features, plus the head-init ablation and "
             "top-3 seed repeats. Every trial is an MLflow run.",
             tune_search, depends_on=("cache_features",), is_fresh=_fresh_search,
             retries=1, expensive=True),
        Task("tune_structural", "Arm B: end-to-end grid over unfrozen blocks x "
             "augmentation (the levers the feature cache can't cover).",
             tune_structural, depends_on=("tune_search",), is_fresh=_fresh_structural,
             retries=1, expensive=True),
        Task("select_and_register", "Pick the best config by VAL across both arms, "
             "read the test split exactly once, save a self-describing checkpoint, "
             "log + register the pyfunc model and move the `champion` alias.",
             select_and_register, depends_on=("tune_structural",), is_fresh=_fresh_final),
        Task("evaluate", "Final test metrics, confusion matrix, click-gate sweep, "
             "and sample predictions with confidences (incl. an unseen gesture).",
             evaluate, depends_on=("select_and_register",), is_fresh=_fresh_eval),
        Task("export_onnx", "Export the champion to ONNX and verify torch/onnxruntime "
             "parity (AD-11) so the runtime has a dependency-light option.",
             export_onnx, depends_on=("select_and_register",), is_fresh=_fresh_onnx),
        Task("serve_smoke", "Load the champion from the registry through the HTTP "
             "endpoint and POST a real fist crop; fail the run if it isn't a click.",
             serve_smoke, depends_on=("select_and_register",), retries=1),
    ],
)

# The live loop is a real-time dataflow, not a batch job: it runs at ~15-30 Hz
# inside one process, so it is DOCUMENTED here (one source of truth for the
# diagram) rather than executed by this engine.
INFERENCE_DAG = DAG(
    name="inference",
    description=("The per-frame graph the game loop actually runs (src/control/play.py), "
                 "at webcam frame rate inside a single process."),
    trigger="continuous while the controller runs; one pass per webcam frame (~15-30 Hz)",
    executable=False,
    tasks=[
        Task("capture_frame", "Grab a BGR frame from the webcam (OpenCV)."),
        Task("detect_hand", "MediaPipe hand landmarks -> palm anchor + hand bbox; "
             "no-hand freezes the cursor and disarms the FSM.",
             depends_on=("capture_frame",)),
        Task("crop_and_preprocess", "Pad the bbox 15% and apply the SAME eval "
             "transform as training (AD A.3) -- shared code, not a re-implementation.",
             depends_on=("detect_hand",)),
        Task("classify_click", "Champion classifier -> p(fist). ~35-40 ms on this CPU, "
             "which is why this call is in-process and not an HTTP request.",
             depends_on=("crop_and_preprocess",)),
        Task("map_cursor", "Palm anchor -> mirrored control box -> EMA smoothing + "
             "dead-zone -> absolute screen coords (AD-19).",
             depends_on=("detect_hand",)),
        Task("click_fsm", "Debounced edge-trigger: K confident fist frames -> exactly "
             "one click -> re-arm on K not_fist frames, with cooldown (AD-20).",
             depends_on=("classify_click",)),
        Task("emit_input", "pydirectinput move/click into FNAF, behind the global "
             "kill-switch.", depends_on=("map_cursor", "click_fsm")),
    ],
)

DAGS = {d.name: d for d in (TRAINING_DAG, INFERENCE_DAG)}
