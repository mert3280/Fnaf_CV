"""What checkpoints exist and what's known about them, for the dashboard's
model picker + comparison panel.

Two sources, merged by directory name:
  - `models/<name>/best.pt` on disk -- what can actually be loaded.
  - the table in `DOCS/models/README.md` -- the honest test numbers, already
    written up per-model (see that file's own header for why: the checkpoints
    are git-ignored, so the doc is what survives).

Parsing the doc table (instead of hand-copying numbers here) means this file
can't drift from DOCS/models/README.md -- if a number there changes, the
dashboard picks it up on next launch.
"""

from __future__ import annotations

import re
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DOC = Path("DOCS/models/README.md")

# fist-vs-rest checkpoints are the only ones the 3.2/3.2.1 click-FSM presets
# fit (AD-21) -- see server._strategy_fits, which this mirrors.
_FISTVSREST_HINT = "fistvsrest"


def discover_checkpoints(models_dir: Path | str = MODELS_DIR) -> list[dict]:
    """Every `models/<name>/best.pt` on disk, sorted by name."""
    models_dir = Path(models_dir)
    out = []
    if not models_dir.exists():
        return out
    for sub in sorted(models_dir.iterdir()):
        ckpt = sub / "best.pt"
        if ckpt.is_file():
            out.append({
                "id": sub.name,
                "path": str(ckpt).replace("\\", "/"),
                "fistvsrest": _FISTVSREST_HINT in sub.name.lower(),
            })
    return out


def _parse_doc_table(doc_path: Path | str = MODELS_DOC) -> dict[str, dict]:
    """Pull the `| Model | Classes | Crop mode | Test acc | ... |` rows out of
    DOCS/models/README.md. Returns {model_id: {classes, crop_mode, test_acc, doc}}.
    Best-effort: an unreadable or reshaped doc just yields no metadata, it
    doesn't fail the dashboard.
    """
    doc_path = Path(doc_path)
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    out: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "`" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        name_m = re.search(r"`([\w.-]+)`", cells[0])
        if not name_m:
            continue
        doc_m = re.search(r"\[.*?\]\((.*?)\)", cells[-1]) if len(cells) > 4 else None
        classes = re.sub(r"\(AD-\d+\)", "", cells[1]) if len(cells) > 1 else None
        classes = re.sub(r"[*]", "", classes).strip() if classes else None
        out[name_m.group(1)] = {
            "classes": classes,
            "crop_mode": cells[2].strip("`") if len(cells) > 2 else None,
            "test_acc": re.search(r"[\d.]+", cells[3]).group(0) if len(cells) > 3 and re.search(r"[\d.]+", cells[3]) else None,
            "doc": doc_m.group(1) if doc_m else None,
        }
    return out


def list_models(models_dir: Path | str = MODELS_DIR,
                doc_path: Path | str = MODELS_DOC) -> list[dict]:
    """Checkpoints on disk, each enriched with whatever DOCS/models/README.md
    knows about it. Checkpoints with no doc row still appear (metadata fields
    just come back None) -- a missing writeup shouldn't hide a runnable model."""
    docs = _parse_doc_table(doc_path)
    models = discover_checkpoints(models_dir)
    for m in models:
        meta = docs.get(m["id"], {})
        m["classes"] = meta.get("classes")
        m["crop_mode"] = meta.get("crop_mode")
        m["test_acc"] = meta.get("test_acc")
        m["doc"] = meta.get("doc")
    return models
