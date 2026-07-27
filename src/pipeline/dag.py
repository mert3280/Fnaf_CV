"""The DAG engine: dependency order, freshness, retries, and an audit record.

Deliberately small. What it does guarantee:

  * **Order** -- Kahn topological sort; a cycle or an unknown dependency is a
    load-time error, not a mid-run surprise.
  * **Idempotency** -- a task may declare `is_fresh(ctx)`. Fresh tasks are
    reported `cached` and skipped unless `--force`, so re-running the DAG after a
    crash costs only the work that is actually missing.
  * **Error containment** -- a task that raises is retried `retries` times, then
    marked `failed`; everything downstream is marked `skipped` (never silently
    "succeeded"), and the run record is written either way.
  * **Auditability** -- every run appends a JSON record: trigger, data version,
    per-task status / duration / attempts / error / outputs.

Not implemented, on purpose: parallel task execution (the expensive tasks are
CPU-bound on one box, so concurrency would only fight for the same cores) and
backfills (there is no time dimension to backfill -- the dataset is static).
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]


class PipelineError(RuntimeError):
    """A task failed for a reason the operator can act on (message is the fix)."""


@dataclass
class Context:
    """Everything a task needs, plus the outputs of the tasks before it."""

    tune_config: Path = REPO / "configs" / "tune_fistvsrest.yaml"
    data_config: Path = REPO / "configs" / "data_fistvsrest.yaml"
    out_dir: Path = REPO / "models" / "week4_tuned_fistvsrest"
    week_dir: Path = REPO / "DOCS" / "class-related" / "week4"
    registered_model: str = "fnaf-click-classifier"
    force: bool = False
    trigger: str = "manual"
    outputs: dict = field(default_factory=dict)

    def out(self, task: str, key: str, default=None):
        return (self.outputs.get(task) or {}).get(key, default)


@dataclass
class Task:
    name: str
    description: str                       # plain language (used by --list + diagram)
    fn: Callable[[Context], dict] | None = None   # None = documented-only node
    depends_on: tuple[str, ...] = ()
    retries: int = 0
    retry_wait_s: float = 3.0
    expensive: bool = False                # flagged in --list; never auto-skipped
    is_fresh: Callable[[Context], bool] | None = None


@dataclass
class DAG:
    name: str
    description: str
    trigger: str
    tasks: list[Task]
    executable: bool = True   # False = a documented graph (e.g. the live loop)

    @property
    def by_name(self) -> dict[str, Task]:
        return {t.name: t for t in self.tasks}

    def order(self) -> list[Task]:
        """Kahn topological sort; raises on unknown deps or cycles."""
        names = self.by_name
        indeg = {t.name: 0 for t in self.tasks}
        children: dict[str, list[str]] = {t.name: [] for t in self.tasks}
        for t in self.tasks:
            for d in t.depends_on:
                if d not in names:
                    raise PipelineError(f"task '{t.name}' depends on unknown task '{d}'")
                indeg[t.name] += 1
                children[d].append(t.name)
        ready = [n for n, d in indeg.items() if d == 0]
        out: list[str] = []
        while ready:
            n = ready.pop(0)
            out.append(n)
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(out) != len(self.tasks):
            cyc = sorted(set(indeg) - set(out))
            raise PipelineError(f"cycle in DAG '{self.name}' involving {cyc}")
        return [names[n] for n in out]


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_dag(dag: DAG, ctx: Context, only: list[str] | None = None,
            skip: list[str] | None = None, dry_run: bool = False) -> dict:
    """Execute `dag` in dependency order. Returns (and persists) the run record."""
    if not dag.executable:
        raise PipelineError(
            f"DAG '{dag.name}' is a documented graph, not an executable one "
            f"(it describes the live per-frame loop). Run `--list` to see it.")

    order = dag.order()
    skip = set(skip or [])
    only = set(only or [])
    record = {"dag": dag.name, "trigger": ctx.trigger, "started": _now(),
              "forced": ctx.force, "dry_run": dry_run, "tasks": []}
    failed: set[str] = set()

    print(f"\nDAG '{dag.name}'  trigger={ctx.trigger}  "
          f"tasks={len(order)}{'  [DRY RUN]' if dry_run else ''}")
    print(f"  {dag.description}\n" + "-" * 72)

    for task in order:
        blocked = sorted(set(task.depends_on) & failed)
        entry = {"task": task.name, "status": "pending", "attempts": 0,
                 "seconds": 0.0, "outputs": {}, "error": None}
        t0 = time.time()

        if blocked:
            entry["status"] = "skipped"
            entry["error"] = f"upstream failed: {', '.join(blocked)}"
            failed.add(task.name)          # propagate, so nothing downstream lies
        elif task.name in skip or (only and task.name not in only):
            entry["status"] = "skipped"
            entry["error"] = "excluded by --skip/--only"
        elif task.fn is None:
            entry["status"] = "documented"
        elif dry_run:
            entry["status"] = "would-run"
        elif task.is_fresh and not ctx.force and _safe_fresh(task, ctx):
            entry["status"] = "cached"
        else:
            entry["status"], entry["outputs"], entry["error"], entry["attempts"] = \
                _attempt(task, ctx)
            if entry["status"] == "failed":
                failed.add(task.name)
            else:
                ctx.outputs[task.name] = entry["outputs"]

        entry["seconds"] = round(time.time() - t0, 2)
        record["tasks"].append(entry)
        icon = {"succeeded": "ok", "cached": "--", "skipped": "..",
                "failed": "XX", "documented": "..", "would-run": "->"}[entry["status"]]
        line = f"[{icon}] {task.name:<22} {entry['status']:<10} {entry['seconds']:>7.2f}s"
        print(line + (f"  {entry['error']}" if entry["error"] else ""))

    record["finished"] = _now()
    record["status"] = "failed" if failed else "succeeded"
    record["failed_tasks"] = sorted(failed)
    path = _write_record(record)
    print("-" * 72)
    print(f"DAG '{dag.name}' {record['status'].upper()}  record -> "
          f"{path.relative_to(REPO)}")
    return record


def _safe_fresh(task: Task, ctx: Context) -> bool:
    """A broken freshness check must not decide the run; treat it as stale."""
    try:
        return bool(task.is_fresh(ctx))
    except Exception as exc:
        print(f"    (freshness check for '{task.name}' failed: {exc} -> running it)")
        return False


def _attempt(task: Task, ctx: Context) -> tuple[str, dict, str | None, int]:
    """Run a task with retries. -> (status, outputs, error, attempts)"""
    last = None
    for attempt in range(1, task.retries + 2):
        try:
            outputs = task.fn(ctx) or {}
            return "succeeded", outputs, None, attempt
        except PipelineError as exc:      # expected, actionable -> no traceback spam
            last = f"{type(exc).__name__}: {exc}"
            print(f"    {task.name} attempt {attempt} failed: {exc}")
        except Exception:
            last = traceback.format_exc(limit=6).strip().splitlines()[-1]
            print(f"    {task.name} attempt {attempt} raised:\n"
                  f"{traceback.format_exc(limit=6)}")
        if attempt <= task.retries:
            time.sleep(task.retry_wait_s)
    return "failed", {}, last, task.retries + 1


def _write_record(record: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPO / "pipeline_runs" / f"{record['dag']}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def describe(dag: DAG) -> str:
    """Plain-language listing of a DAG (also the source for the diagram)."""
    lines = [f"DAG: {dag.name}"
             f"{'' if dag.executable else '   [documented graph -- not executed]'}",
             f"  {dag.description}",
             f"  trigger: {dag.trigger}", ""]
    for t in dag.order():
        dep = f"  <- {', '.join(t.depends_on)}" if t.depends_on else ""
        tag = "  [expensive]" if t.expensive else ""
        lines.append(f"  {t.name}{dep}{tag}")
        lines.append(f"      {t.description}")
    return "\n".join(lines)
