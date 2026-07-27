"""CLI / trigger surface for the pipeline.

    python -m src.pipeline.run --list                     # both DAGs, plain language
    python -m src.pipeline.run --dag training --dry-run   # order + what would run
    python -m src.pipeline.run --dag training             # fresh tasks are skipped
    python -m src.pipeline.run --dag training --force     # ignore freshness, redo all
    python -m src.pipeline.run --dag training --only evaluate export_onnx
    python -m src.pipeline.run --dag training --if-data-changed   # the data event

`--if-data-changed` is the event trigger: it compares the current split-manifest
fingerprint against the one recorded in the registered model's `final.json` and
exits 0 without touching anything when they match. That is what a file-watcher, a
post-capture hook, or a cron wrapper would call -- the decision to retrain lives
here, next to the data version, not in a scheduler's config.
"""

from __future__ import annotations

import argparse
import sys

from src.pipeline.dag import REPO, Context, PipelineError, describe, run_dag
from src.pipeline.tasks import DAGS


def data_changed(ctx: Context) -> tuple[bool, str]:
    """(should_run, why) by comparing data version vs the registered model's."""
    import json

    from src.data.config import DataConfig
    from src.tune import dataset_fingerprint

    manifest = REPO / DataConfig.from_yaml(ctx.data_config).manifest
    if not manifest.exists():
        return True, "no split manifest yet (first run)"
    current = dataset_fingerprint(manifest)
    final = ctx.out_dir / "tuning" / "final.json"
    if not final.exists():
        return True, f"data_version {current}; no registered model yet"
    recorded = json.loads(final.read_text()).get("data_version")
    if recorded != current:
        return True, f"data changed: registered {recorded} -> current {current}"
    return False, f"data_version {current} unchanged since the registered model"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dag", choices=sorted(DAGS), default="training")
    ap.add_argument("--list", action="store_true", help="describe the DAGs and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore freshness checks")
    ap.add_argument("--only", nargs="+", default=None, metavar="TASK")
    ap.add_argument("--skip", nargs="+", default=None, metavar="TASK")
    ap.add_argument("--if-data-changed", action="store_true",
                    help="no-op unless the data version differs from the registered model's")
    args = ap.parse_args()

    if args.list:
        for name in sorted(DAGS):
            print(describe(DAGS[name]))
            print()
        return 0

    ctx = Context(force=args.force,
                  trigger="event:data-changed" if args.if_data_changed else "manual")

    if args.if_data_changed:
        should, why = data_changed(ctx)
        print(f"data-change trigger: {why}")
        if not should:
            print("nothing to do (exit 0)")
            return 0

    try:
        record = run_dag(DAGS[args.dag], ctx, only=args.only, skip=args.skip,
                         dry_run=args.dry_run)
    except PipelineError as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2
    return 0 if record["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
