"""Render the pipeline flowchart PNG straight from the DAG definitions.

The diagram is generated from `src/pipeline/tasks.py`, so it cannot drift from
the code it documents: add a task and re-run this, and the picture is correct.

    python scripts/render_pipeline_diagram.py

Layout is a deterministic layered graph (level = longest path from a root), which
is stable for DAGs this size -- no graphviz dependency, no hand-placed boxes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.plotting import PALETTE, apply_style, save  # noqa: E402
from src.pipeline.tasks import INFERENCE_DAG, TRAINING_DAG  # noqa: E402

OUT = REPO / "DOCS" / "class-related" / "week4" / "pipeline-diagram.png"

# Role accents: data (blue), compute/tune (orange), eval+serve (aqua). Every box
# is also text-labelled, so the color never carries meaning on its own.
ROLE = {
    "ingest_check": 0, "index_and_split": 0, "cache_features": 0,
    "tune_search": 1, "tune_structural": 1, "select_and_register": 1,
    "evaluate": 2, "export_onnx": 2, "serve_smoke": 2,
    "capture_frame": 0, "detect_hand": 0, "crop_and_preprocess": 0,
    "classify_click": 1, "map_cursor": 1, "click_fsm": 1, "emit_input": 2,
}
ROLE_LABEL = ["data / ingest", "model / compute", "evaluate / serve"]

# One short line per node -- the diagram has to be readable without the report.
CAPTION = {
    "ingest_check": "images + annotations\non disk?",
    "index_and_split": "user-grouped 70/15/15\n-> split manifest = data version",
    "cache_features": "1 frozen pass\n-> cached 1280-d features",
    "tune_search": "Arm A: 60-trial TPE\n(MLflow run per trial)",
    "tune_structural": "Arm B: unfreeze x aug\nend-to-end grid",
    "select_and_register": "best by VAL -> test ONCE\n-> registry + champion alias",
    "evaluate": "confusion matrix,\ngate sweep, samples",
    "export_onnx": "ONNX + torch parity\ncheck (AD-11)",
    "serve_smoke": "POST /predict\nwith a real crop",
    "capture_frame": "webcam BGR frame",
    "detect_hand": "MediaPipe landmarks\n-> anchor + bbox",
    "crop_and_preprocess": "pad 15% + the SAME\neval transform",
    "classify_click": "champion -> p(fist)\n~35-40 ms, in-process",
    "map_cursor": "control box + EMA\n-> screen coords",
    "click_fsm": "K frames -> 1 click\n+ cooldown",
    "emit_input": "pydirectinput\n(behind kill-switch)",
}


def levels(dag) -> dict[str, int]:
    """Longest-path depth per task (parents are always on an earlier row)."""
    depth: dict[str, int] = {}
    for task in dag.order():           # already topologically sorted
        depth[task.name] = (0 if not task.depends_on
                            else 1 + max(depth[d] for d in task.depends_on))
    return depth


def draw(ax, dag, title: str, subtitle: str, accent: str, xspan: float | None = None):
    import matplotlib.patches as mpatches

    depth = levels(dag)
    rows: dict[int, list[str]] = {}
    for name, d in depth.items():
        rows.setdefault(d, []).append(name)

    bw, bh = 2.5, 0.92          # box size in data units
    xgap, ygap = 0.42, 0.72
    pos: dict[str, tuple[float, float]] = {}
    max_w = max(len(v) * bw + (len(v) - 1) * xgap for v in rows.values())
    for d, names in rows.items():
        width = len(names) * bw + (len(names) - 1) * xgap
        x0 = (max_w - width) / 2
        for i, name in enumerate(names):
            pos[name] = (x0 + i * (bw + xgap) + bw / 2,
                         -d * (bh + ygap) - bh / 2)

    # edges first, so boxes sit on top of the arrow heads. An edge that skips a
    # level bows outward so it routes AROUND the boxes in between instead of
    # cutting through them.
    for task in dag.tasks:
        for dep in task.depends_on:
            x1, y1 = pos[dep]
            x2, y2 = pos[task.name]
            span = depth[task.name] - depth[dep]
            rad = 0.0 if span <= 1 else (0.28 if x1 >= x2 else -0.28)
            ax.annotate("", xy=(x2, y2 + bh / 2), xytext=(x1, y1 - bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color=PALETTE["axis"],
                                        linewidth=1.2, shrinkA=1, shrinkB=1,
                                        connectionstyle=f"arc3,rad={rad}"))
    for task in dag.tasks:
        x, y = pos[task.name]
        role = PALETTE["series"][ROLE.get(task.name, 0)]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=PALETTE["surface"], edgecolor=PALETTE["axis"], linewidth=1.0))
        ax.add_patch(mpatches.Rectangle((x - bw / 2, y - bh / 2), 0.075, bh,
                                        facecolor=role, edgecolor="none"))
        ax.text(x - bw / 2 + 0.17, y + 0.20, task.name.replace("_", " "),
                fontsize=8.5, fontweight="bold", color=PALETTE["ink"], va="center")
        ax.text(x - bw / 2 + 0.17, y - 0.16, CAPTION.get(task.name, ""),
                fontsize=6.8, color=PALETTE["ink_secondary"], va="center", linespacing=1.35)
        if task.expensive:
            ax.text(x + bw / 2 - 0.1, y + 0.20, "hrs", fontsize=6.5,
                    color=PALETTE["ink_muted"], ha="right", va="center", style="italic")

    # Title + subtitle live in AXES coordinates above the graph, with the graph's
    # top margin reserved for them, so they can never collide with the nodes.
    # A shared `xspan` keeps the box scale identical across panels.
    half = max(xspan or max_w, max_w) / 2
    centre = max_w / 2
    ax.set_xlim(centre - half - 0.4, centre + half + 0.4)
    bottom = -max(rows) * (bh + ygap) - bh - 0.4    # clear the last row's bottom edge
    top = bh / 2 + 0.4
    span = top - bottom
    header = 0.30 * span   # reserved band above the graph
    ax.set_ylim(bottom, top + header)
    ax.axis("off")
    ax.text(centre - half - 0.35, top + header * 0.80, title, color=PALETTE["ink"],
            fontsize=12.5, fontweight="bold", va="center")
    ax.text(centre - half - 0.35, top + header * 0.33, subtitle,
            color=PALETTE["ink_secondary"], fontsize=7.6, va="center", linespacing=1.6)
    ax.add_patch(mpatches.Rectangle((centre - half - 0.3, bottom + 0.1),
                                    2 * half + 0.6, span - 0.2, facecolor="none",
                                    edgecolor=accent, linewidth=1.4,
                                    linestyle=(0, (4, 3)), zorder=0))


def main() -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.6),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06})
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.02, right=0.99)
    # widest row across BOTH dags -> one box scale everywhere
    xspan = 3 * 2.5 + 2 * 0.42
    draw(axes[0], TRAINING_DAG,
         "Training pipeline  ('hrs' = hours of CPU)",
         "Trigger: EVENT-BASED (new data / recipe change), not scheduled.\n"
         "`python -m src.pipeline.run --dag training --if-data-changed`\n"
         "Fresh tasks report `cached` and are skipped; every task's status, duration\n"
         "and error lands in pipeline_runs/<dag>_<ts>.json.",
         PALETTE["series"][0], xspan=xspan)
    draw(axes[1], INFERENCE_DAG,
         "Serving path  (per webcam frame)",
         "Trigger: CONTINUOUS while the controller runs (~15-30 Hz, one process).\n"
         "`python -m src.control.play --checkpoint <champion>`\n"
         "The HTTP endpoint (src/serve/app.py) serves the same registered model for\n"
         "validation and batch use -- it is NOT in this loop (frame budget ~33 ms).",
         PALETTE["series"][1], xspan=xspan)

    handles = [mpatches.Patch(facecolor=PALETTE["series"][i], edgecolor="none",
                              label=ROLE_LABEL[i]) for i in range(3)]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.015))
    fig.text(0.012, 0.985, "FNAF hands-free — end-to-end ML pipeline, ingestion to serving",
             ha="left", va="top", fontsize=14.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.012, 0.958, "Generated from src/pipeline/tasks.py by "
                           "scripts/render_pipeline_diagram.py — the diagram cannot "
                           "drift from the code.",
             fontsize=8, color=PALETTE["ink_muted"], ha="left", va="top")
    save(fig, OUT)


if __name__ == "__main__":
    main()
