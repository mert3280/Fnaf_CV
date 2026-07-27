"""Pipeline orchestration: the task graph that turns HaGRID images into a
registered, served click classifier.

`dag.py`   -- the engine (topological order, freshness/idempotency, retries,
              per-task error capture, JSON run records).
`tasks.py` -- the actual work, each task a thin wrapper over existing modules.
`run.py`   -- the CLI / trigger surface.

Deliberately ~200 lines instead of Airflow: this project trains on one local
Windows box (AD-02b) with a static dataset, so a scheduler daemon would be
infrastructure with no job to do. See the Week-4 report §3 for that trade-off and
for what would have to change to move these same task definitions onto Airflow.
"""
