"""Model packaging + serving: the MLflow pyfunc wrapper and the HTTP endpoint.

`pyfunc_model.py` is the *deployment artifact* (what gets registered and
versioned in MLflow); `app.py` is a thin Flask wrapper that exposes it over
HTTP for validation and batch use. The real-time game loop deliberately does
NOT go through HTTP -- see DOCS/class-related/week4/tuning-orchestration-report.md §4.
"""
