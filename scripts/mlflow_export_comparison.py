"""Export the MLflow run comparison for the Week-3 report.

Reads the ./mlruns store populated by scripts/mlflow_log_runs.py and writes two
committed artifacts into DOCS/class-related/week3/:
  * mlflow-run-comparison.csv  -- the exported side-by-side run table
  * mlflow-run-comparison.png  -- a grouped val/test accuracy bar chart

Run AFTER mlflow_log_runs.py:
    python scripts/mlflow_export_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "DOCS" / "class-related" / "week3"
EXPERIMENT = "fnaf-gesture-classifier"

# Logical progression of the investigation (not MLflow's start-time order).
ORDER = ["baseline_mnv3_large", "bbox_frozen_mnv3_large", "palmfist_frozen_mnv3_large"]
LABELS = ["baseline\n(8-cls, full-frame)", "bbox_frozen\n(8-cls, bbox crop)", "palmfist\n(2-cls, bbox crop)"]

BLUE, YELLOW = "#2a78d6", "#eda100"  # validated categorical pair (val vs test)
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e6e2"


def main() -> None:
    mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
    df = mlflow.search_runs(experiment_names=[EXPERIMENT])

    # order rows by the investigation arc
    df["order"] = df["tags.mlflow.runName"].map({n: i for i, n in enumerate(ORDER)})
    df = df.sort_values("order")

    cols = [
        "tags.mlflow.runName", "params.num_classes", "params.crop_mode",
        "params.backbone", "params.freeze_backbone", "params.epochs", "params.lr",
        "metrics.val_accuracy", "metrics.test_accuracy", "metrics.random_baseline",
        "metrics.test_macro_f1", "metrics.test_f1_palm", "metrics.test_f1_fist",
    ]
    table = df[cols].rename(columns=lambda c: c.split(".", 1)[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "mlflow-run-comparison.csv"
    table.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    # ---- grouped bar chart: val vs test accuracy per run --------------------
    val = df["metrics.val_accuracy"].to_numpy(dtype=float)
    test = df["metrics.test_accuracy"].to_numpy(dtype=float)
    x = range(len(ORDER))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    b1 = ax.bar([i - w / 2 for i in x], val, w, label="Validation accuracy", color=BLUE, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], test, w, label="Held-out test accuracy", color=YELLOW, zorder=3)

    # direct value labels (required: yellow is below the 3:1 surface-contrast floor)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, fontsize=9, color=INK)
    ax.set_ylabel("Accuracy", fontsize=10, color=MUTED)
    ax.set_title("FNAF gesture classifier — three experiments (MobileNetV3, frozen backbone)",
                 fontsize=11, color=INK, pad=12)
    ax.tick_params(colors=MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)

    fig.tight_layout()
    png_path = OUT / "mlflow-run-comparison.png"
    fig.savefig(png_path, facecolor=SURFACE)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
