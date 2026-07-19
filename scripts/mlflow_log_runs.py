"""Retroactively log the three Week-3 experiments into MLflow.

Context (honest): the modeling was run earlier via `src/train.py`, whose
tracking is a self-describing checkpoint (`models/<run>/config.snapshot.json`)
plus a frozen markdown report (`DOCS/models/<run>/results.md`). MLflow was
adopted in Week 3 to consolidate those three *already-run* experiments into one
comparable experiment store and to export a side-by-side run comparison.

This script does **not** retrain anything. It transcribes:
  * hyper-parameters   <- the run's `config.snapshot.json` (+ the crop_mode and
                          split facts that live in configs/data.yaml at run time)
  * metrics            <- the run's frozen `DOCS/models/<run>/results.md`
                          (the real recorded val/test numbers -- nothing invented)

Run:
    python scripts/mlflow_log_runs.py
    mlflow ui --backend-store-uri ./mlruns      # then screenshot the comparison

Every number below is copied from the committed results docs; change a number
here only if you change it in the corresponding results.md (they must agree).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = "fnaf-gesture-classifier"

# Split is identical across all three runs (configs/data.yaml, AD-16):
# 70/15/15 over all images, grouped by user_id (no subject leakage), seed 42.
SPLIT = {"split_scheme": "70/15/15 by user_id (grouped, stratified)", "split_seed": 42}

# --------------------------------------------------------------------------- #
# The three experiments. `metrics` are the real recorded numbers, transcribed
# verbatim from each frozen DOCS/models/<run>/results.md. `crop_mode` and
# `num_classes` are the per-run facts (the shared config.snapshot.json doesn't
# capture crop_mode because it lived in data.yaml at run time).
# --------------------------------------------------------------------------- #
RUNS = [
    {
        "run_name": "baseline_mnv3_large",
        "snapshot": "models/baseline_mnv3_large/config.snapshot.json",
        "tags": {
            "era": "pre-pivot (8-class, AD-03)",
            "role": "single-stage full-frame baseline (AD-04 A/B control)",
            "crop_mode": "full_frame",
        },
        "params": {"crop_mode": "full_frame", "num_classes": 8, "stage": "A (frozen backbone, head-only)"},
        "metrics": {
            "val_accuracy": 0.5695,
            "test_accuracy": 0.5300,
            "random_baseline": 0.1250,
            "test_macro_f1": 0.5303,
            "test_weighted_f1": 0.5282,
            "test_f1_palm": 0.4818,   # click-relevant classes, tracked across all runs
            "test_f1_fist": 0.5876,
        },
    },
    {
        "run_name": "bbox_frozen_mnv3_large",
        "snapshot": "models/bbox_frozen_mnv3_large/config.snapshot.json",
        "tags": {
            "era": "pre-pivot (8-class, AD-03)",
            "role": "two-stage bbox-crop (AD-04 primary); the +15% pad crop",
            "crop_mode": "bbox",
        },
        "params": {"crop_mode": "bbox", "bbox_pad": 0.15, "num_classes": 8, "stage": "A (frozen backbone, head-only)"},
        "metrics": {
            "val_accuracy": 0.9403,
            "test_accuracy": 0.9165,
            "random_baseline": 0.1250,
            "test_macro_f1": 0.9156,
            "test_weighted_f1": 0.9165,
            "test_f1_palm": 0.9347,
            "test_f1_fist": 0.9443,
        },
    },
    {
        "run_name": "palmfist_frozen_mnv3_large",
        "snapshot": "models/palmfist_frozen_mnv3_large/config.snapshot.json",
        "tags": {
            "era": "current (cursor-control pivot, AD-17/AD-18)",
            "role": "binary click classifier: palm=no-click, fist=click",
            "crop_mode": "bbox",
        },
        "params": {"crop_mode": "bbox", "bbox_pad": 0.15, "num_classes": 2, "stage": "A (frozen backbone, head-only)"},
        "metrics": {
            "val_accuracy": 0.9639,
            "test_accuracy": 0.9815,
            "random_baseline": 0.5000,
            "test_macro_f1": 0.9815,
            "test_weighted_f1": 0.9815,
            "test_f1_palm": 0.9812,
            "test_f1_fist": 0.9818,
        },
    },
]


def flatten_snapshot(snap: dict) -> dict:
    """Pull the loggable hyper-parameters out of a config.snapshot.json."""
    m, t = snap["model"], snap["train"]
    return {
        "backbone": m["backbone"],
        "pretrained": m["pretrained"],
        "freeze_backbone": m["freeze_backbone"],
        "unfreeze_blocks": m.get("unfreeze_blocks", 0),
        "optimizer": t["optimizer"],
        "lr": t["lr"],
        "weight_decay": t["weight_decay"],
        "epochs": t["epochs"],
        "batch_size": t["batch_size"],
        "seed": t["seed"],
        "augment_train": t["augment_train"],
        "precompute_features": t["precompute_features"],
        "input_size": 224,
    }


def main() -> None:
    mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    for run in RUNS:
        snap = json.loads((REPO / run["snapshot"]).read_text())
        params = {**flatten_snapshot(snap), **SPLIT, **run["params"]}
        with mlflow.start_run(run_name=run["run_name"]):
            mlflow.set_tags({**run["tags"], "logged_by": "scripts/mlflow_log_runs.py"})
            mlflow.log_params(params)
            mlflow.log_metrics(run["metrics"])
            print(f"logged {run['run_name']}: "
                  f"val={run['metrics']['val_accuracy']:.4f} "
                  f"test={run['metrics']['test_accuracy']:.4f}")

    print(f"\nDone. Tracking store: {REPO / 'mlruns'}")
    print("View:  mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
