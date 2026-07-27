"""Export the Week-4 tuning experiment out of MLflow into committable evidence.

The MLflow store (`./mlruns`) is git-ignored, so the report can't point at it.
This reads the store back and writes, into DOCS/class-related/week4/:

  * `mlflow-tuning-runs.csv`      -- every run (baselines, 60 search trials, seed
                                    repeats, structural grid, final) with its
                                    params + metrics, straight from the store
  * `figures/10_tuning_ladder.png` -- what each lever was actually worth
  * `figures/11_search_surface.png`-- the search surface (val acc vs lr, by optimizer)
  * `figures/12_registered_model.png` -- the registry entry for the champion

These are *exported from the tracking store*, not screenshots of the UI. To see
the runs in the UI (and screenshot them if a literal screenshot is wanted):

    mlflow ui --backend-store-uri ./mlruns      # experiment: fnaf-week4-tuning

    python scripts/mlflow_export_tuning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from src.plotting import PALETTE, apply_style, save  # noqa: E402

EXPERIMENT = "fnaf-week4-tuning"
WEEK = REPO / "DOCS" / "class-related" / "week4"
FIGS = WEEK / "figures"
REGISTERED = "fnaf-click-classifier"


def load_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"experiment '{EXPERIMENT}' not found -- run `python -m src.tune` first")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
    if df.empty:
        raise SystemExit("no runs logged yet")
    return df


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run, only the columns a reader needs."""
    keep = {
        "tags.mlflow.runName": "run", "tags.arm": "arm", "tags.stage": "stage",
        "params.head_init": "head_init", "params.optimizer": "optimizer",
        "params.lr": "lr", "params.weight_decay": "weight_decay",
        "params.batch_size": "batch_size", "params.label_smoothing": "label_smoothing",
        "params.unfreeze_blocks": "unfreeze_blocks",
        "params.augment_train": "augment_train", "params.epochs": "epochs",
        "params.data_version": "data_version", "tags.of": "repeat_of",
        "metrics.val_accuracy": "val_accuracy", "metrics.test_accuracy": "test_accuracy",
        "metrics.best_epoch": "best_epoch", "metrics.seconds": "seconds",
        "metrics.train_accuracy_at_best": "train_accuracy_at_best",
    }
    out = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    for col in ("lr", "weight_decay", "label_smoothing", "val_accuracy",
                "test_accuracy", "best_epoch", "seconds", "train_accuracy_at_best"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    order = {"baseline_reference": 0, "tuning": 1, "seed_repeat": 2, "final": 3}
    out["_o"] = out["stage"].map(order).fillna(9)
    return (out.sort_values(["_o", "val_accuracy"], ascending=[True, False])
               .drop(columns="_o").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_ladder(rows: list[tuple[str, float, str]], out: Path):
    """What each lever bought, in val accuracy.

    A DOT plot, not bars: the interesting range is 0.92-0.99, and a bar chart whose
    baseline isn't zero overstates the differences by area. Dots encode position
    only, so a clipped axis is honest -- and the axis still starts low enough to
    show the deployed baseline sitting well below everything else.
    """
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    notes = [r[2] for r in rows]
    lo = min(vals) - 0.006
    fig, ax = plt.subplots(figsize=(7.6, 0.56 * len(rows) + 1.5))
    y = list(range(len(rows)))
    ax.hlines(y, lo, vals, color=PALETTE["grid"], linewidth=1.4, zorder=2)
    ax.scatter(vals, y, s=70, color=PALETTE["series"][0], zorder=3,
               edgecolor=PALETTE["surface"], linewidth=1.2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.35, -0.65)   # room for the note under the last row
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("best validation accuracy")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for i, (v, n) in enumerate(zip(vals, notes)):
        ax.text(v + 0.0022, i - 0.02, f"{v:.4f}", va="center", fontsize=8.5,
                fontweight="bold", color=PALETTE["ink"])
        if n:
            ax.text(v + 0.0022, i + 0.24, n, va="center", fontsize=7.2,
                    color=PALETTE["ink_muted"])
    ax.set_title("What each lever was worth (validation, n=308)", loc="left")
    return save(fig, out)


def fig_search(trials: pd.DataFrame, baselines: dict, best_name: str, out: Path):
    """The search surface: 60 trials, val acc vs learning rate, by optimizer."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    markers = {"adamw": "o", "adam": "s", "sgd": "^"}   # secondary encoding
    for i, (opt, grp) in enumerate(sorted(trials.groupby("optimizer"))):
        ax.scatter(grp["lr"], grp["val_accuracy"], s=42,
                   color=PALETTE["series"][i % 3], marker=markers.get(opt, "o"),
                   edgecolor=PALETTE["surface"], linewidth=1.0, zorder=3,
                   label=f"{opt} (n={len(grp)})")
    # Reference lines are chrome, labelled at the RIGHT edge so they can't collide
    # with the legend in the lower left.
    for label, (val, style) in baselines.items():
        ax.axhline(val, color=PALETTE["ink_muted"], linestyle=style, linewidth=1.2,
                   zorder=2)
        ax.annotate(f"{label}  {val:.4f}", (1.0, val), xycoords=("axes fraction", "data"),
                    xytext=(-3, 4), textcoords="offset points", fontsize=7.4,
                    ha="right", color=PALETTE["ink_secondary"])
    # The top of the leaderboard is a TIE, not a peak -- annotate the region rather
    # than crowning one trial, because single-seed rank among ties is noise.
    top = trials["val_accuracy"].max()
    tied = trials[trials["val_accuracy"] == top]
    note = (f"{len(tied)} trials tie at {top:.4f}\n"
            f"all {'/'.join(sorted(tied['optimizer'].unique()))}, "
            f"lr {tied['lr'].min():.1e}-{tied['lr'].max():.1e},\n"
            f"weight_decay >= {tied['weight_decay'].min():.1e}\n"
            f"best by 3-seed mean: {best_name}")
    ax.annotate(note, xy=(float(tied["lr"].median()), top), xytext=(0.53, 0.30),
                textcoords="axes fraction", fontsize=7.6, ha="left",
                color=PALETTE["ink"], linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=PALETTE["ink_muted"],
                                linewidth=0.9, shrinkB=5))
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Arm A search surface — 60 TPE trials on cached frozen features",
                 loc="left")
    lo = min(trials["val_accuracy"].min(), min(v for v, _ in baselines.values()))
    ax.set_ylim(lo - 0.008, trials["val_accuracy"].max() + 0.004)
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.03), ncols=1)
    return save(fig, out)


def fig_registry(info: list[tuple[str, str]], out: Path):
    """The registry entry as a readable card (the 'registered model' evidence)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(info) + 1.0))
    ax.axis("off")
    ax.set_title(f"MLflow Model Registry — {REGISTERED}", loc="left")
    for i, (k, v) in enumerate(info):
        y = 1 - (i + 0.5) / len(info)
        ax.text(0.0, y, k, fontsize=8.5, color=PALETTE["ink_secondary"], va="center")
        ax.text(0.34, y, v, fontsize=8.5, color=PALETTE["ink"], va="center",
                fontweight="bold" if i < 3 else "normal")
        ax.axhline(y - 0.5 / len(info), color=PALETTE["grid"], linewidth=0.8)
    return save(fig, out)


def registry_info() -> list[tuple[str, str]]:
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED}'")
    except Exception as exc:
        return [("registry", f"unavailable: {exc}")]
    if not versions:
        return [("registry", "no versions registered yet")]
    v = sorted(versions, key=lambda x: int(x.version))[-1]
    run = client.get_run(v.run_id)
    m, p = run.data.metrics, run.data.params
    aliases = ", ".join(v.aliases) if getattr(v, "aliases", None) else "-"
    return [
        ("model name", REGISTERED),
        ("version", str(v.version)),
        ("alias", aliases),
        ("val accuracy", f"{m.get('val_accuracy', float('nan')):.4f}"),
        ("test accuracy (read once)", f"{m.get('test_accuracy', float('nan')):.4f}"),
        ("selected from", p.get("selected_from", "-")),
        ("data version (split-manifest hash)", p.get("data_version", "-")),
        ("classes", p.get("classes", "-")),
        ("backbone", p.get("backbone", "-")),
        ("crop mode / input", f"{p.get('crop_mode', '-')} / {p.get('input_size', '-')}px"),
        ("flavor", "python_function (preprocessing inside the artifact)"),
        ("run id", v.run_id),
    ]


def main() -> None:
    apply_style()
    raw = load_runs()
    runs = tidy(raw)
    csv = WEEK / "mlflow-tuning-runs.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(csv, index=False)
    print(f"wrote {csv}  ({len(runs)} runs)")

    # Note: the two per-arm PARENT runs also carry stage="tuning" (they hold the
    # sweep's own params/summary metrics), so every selection below requires the
    # per-trial columns -- otherwise a parent sneaks in as an empty row.
    base = runs[runs["stage"] == "baseline_reference"].set_index("run")
    trials = runs[(runs["stage"] == "tuning") & (runs["arm"] == "A")].dropna(
        subset=["lr", "val_accuracy"])
    struct = runs[(runs["stage"] == "tuning") & (runs["arm"] == "B")].dropna(
        subset=["val_accuracy", "unfreeze_blocks"])
    final = runs[runs["stage"] == "final"]

    b_timm = float(base.loc["baseline_reference_timm_init", "val_accuracy"])
    b_fixed = float(base.loc["baseline_reference_fixed_init", "val_accuracy"])
    top = float(trials["val_accuracy"].max())
    n_tied = int((trials["val_accuracy"] == top).sum())

    # Selection used the 3-seed MEAN of the top configs, not the single-seed
    # leaderboard. Several configs tie on that mean too, so the authority for
    # "which one" is the FINAL run that actually registered a model -- re-deriving
    # it here with a different tiebreak would let this figure disagree with the
    # registry. Fall back to the seed-mean argmax only before `--stage final` runs.
    repeats = runs[runs["stage"] == "seed_repeat"].dropna(subset=["val_accuracy"])
    sel_name, sel_mean = None, None
    if not repeats.empty and "repeat_of" in repeats:
        means = repeats.groupby("repeat_of")["val_accuracy"].mean()
        top_mean = means.sort_values().index[-1]
        sel_name, sel_mean = top_mean, float(means.loc[top_mean])

    # Which run the pipeline ACTUALLY registered (may be an Arm-B run, which has no
    # seed repeats). The registry is the authority; nothing here re-derives it.
    registered = (raw["params.selected_from"].dropna().tolist()
                  if "params.selected_from" in raw else [])
    champion = registered[0].split("/")[-1] if registered else None

    print("\nfigures:")
    ladder = [
        ("deployed baseline\n(timm head init, 40 ep)", b_timm, "pre-tuning"),
        ("same recipe,\nhead init fixed", b_fixed, "init ablation"),
        (f"best Arm-A trial\n({n_tied} trials tie)", top, "+ TPE search, single seed"),
    ]
    if sel_name:
        ladder.append((f"best Arm-A config\n({sel_name}, 3-seed mean)", sel_mean,
                       "seed-mean, not single-seed"))
    for _, r in struct.sort_values("val_accuracy", ascending=False).iterrows():
        note = (f"unfreeze {r.get('unfreeze_blocks', '?')}, "
                f"aug {r.get('augment_train', '?')}")
        if champion and r["run"] == champion:
            note += "  <- REGISTERED"
        ladder.append((f"Arm B: {r['run']}", float(r["val_accuracy"]), note))
    fig_ladder(ladder, FIGS / "10_tuning_ladder.png")
    fig_search(trials, {"deployed baseline": (b_timm, (0, (5, 3))),
                        "init fixed": (b_fixed, (0, (2, 2)))},
               sel_name or "-", FIGS / "11_search_surface.png")
    fig_registry(registry_info(), FIGS / "12_registered_model.png")

    print(f"\nbaselines: timm {b_timm:.4f} | fixed {b_fixed:.4f}")
    print(f"Arm A top: {top:.4f} ({n_tied} trials tied); "
          f"best by seed-mean: {sel_name} {sel_mean}")
    print(f"registered champion: {champion or '(none yet)'}")
    print(f"trials >= init-fixed baseline: {int((trials['val_accuracy'] >= b_fixed).sum())}"
          f"/{len(trials)}; worst {trials['val_accuracy'].min():.4f}")
    if not struct.empty:
        print("Arm B:", {r["run"]: round(r["val_accuracy"], 4)
                         for _, r in struct.iterrows()})
    if not final.empty:
        f = final.iloc[0]
        print(f"final: val {f['val_accuracy']:.4f} test {f['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
