"""HaGRID annotation parsing, validation, and unified on-disk indexing.

Turns the per-class HaGRID annotation JSONs (normalized COCO bbox + 21
landmarks per image UUID) into a single pandas DataFrame that joins each
annotation to the image actually present on disk. No pixels are read here, so
this module imports without torch / torchvision / timm.

Record schema (per UUID) in the HaGRID JSONs:
    {"bboxes": [[x, y, w, h], ...],   # normalized, top-left + w/h, in [0,1]
     "labels": ["fist", ...],         # parallel to bboxes; may include no_gesture
     "landmarks": [[[lx, ly], ...]],  # 21 normalized keypoints per hand
     "leading_hand": "left", "user_id": "..."}

A record for a class folder may contain a *second* hand labelled "no_gesture";
we always select the bbox whose label matches the folder's class.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Columns produced by build_index(). Documented so downstream code is stable.
# `label` is the TARGET class (what the head predicts); `source_label` is the
# gesture folder the image came from -- they differ only when label_groups
# collapses several folders into one class (Strategy 3.2 / AD-21).
INDEX_COLUMNS = ["uuid", "label", "label_idx", "path", "bbox", "user_id", "has_ann", "source_label"]


def load_class_annotations(ann_dir: str | Path, cls: str) -> dict[str, dict]:
    """Load the raw {uuid: record} dict for one class JSON."""
    json_path = Path(ann_dir) / f"{cls}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")
    return json.loads(json_path.read_text())


def bbox_for_label(record: dict, cls: str) -> list[float] | None:
    """Return the [x, y, w, h] bbox whose label == cls, or None if absent."""
    labels = record.get("labels", [])
    bboxes = record.get("bboxes", [])
    for lbl, box in zip(labels, bboxes):
        if lbl == cls:
            return list(box)
    return None


def validate_record(uuid: str, record: dict, cls: str) -> list[str]:
    """Return a list of human-readable problems with a record (empty == valid).

    Checks (Phase 1 §2.2): >=1 bbox, the class label is present, and every
    bbox coordinate is normalized into [0, 1].
    """
    problems: list[str] = []
    bboxes = record.get("bboxes", [])
    labels = record.get("labels", [])
    if not bboxes:
        problems.append(f"{uuid}: no bboxes")
    if cls not in labels:
        problems.append(f"{uuid}: class '{cls}' not in labels {labels}")
    for box in bboxes:
        if len(box) != 4:
            problems.append(f"{uuid}: bbox {box} is not length-4")
            continue
        if not all(0.0 <= v <= 1.0 for v in box):
            problems.append(f"{uuid}: bbox {box} outside [0,1]")
    return problems


def validate_class(ann_dir: str | Path, cls: str) -> list[str]:
    """Validate every record in a class JSON; returns all problems found."""
    raw = load_class_annotations(ann_dir, cls)
    problems: list[str] = []
    for uuid, record in raw.items():
        problems.extend(validate_record(uuid, record, cls))
    return problems


def build_index(
    images_dir: str | Path,
    ann_dir: str | Path,
    classes: list[str],
    class_to_idx: dict[str, int] | None = None,
    require_annotation: bool = False,
    source_to_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Scan on-disk images and join their HaGRID annotations into one index.

    We drive off the *files that actually exist* (downloads are a subset of the
    annotations, and a few annotated UUIDs never made it out of the source ZIP
    -- see DOCS/data.md), then attach each file's bbox/landmarks by UUID.

    Args:
        images_dir: dir laid out as <images_dir>/<folder>/<uuid>.jpg.
        ann_dir:    dir of <folder>.json HaGRID annotations.
        classes:    the TARGET labels (used only for the class_to_idx fallback).
        class_to_idx: TARGET label -> int; defaults to enumerate(classes).
        require_annotation: if True, drop images with no matching record
            (else keep them with bbox=None; full_frame training still works).
        source_to_label: image folder -> TARGET label. Defaults to the identity
            map over `classes` (folder name == label). Pass a many-to-one map to
            collapse several gesture folders into one class (Strategy 3.2): the
            bbox is still looked up by the *folder's own* gesture (we crop the
            hand that folder is about), only `label`/`label_idx` are the target.

    Returns a DataFrame with columns INDEX_COLUMNS, one row per image.
    """
    images_dir = Path(images_dir)
    class_to_idx = class_to_idx or {c: i for i, c in enumerate(classes)}
    source_to_label = source_to_label or {c: c for c in classes}

    rows: list[dict] = []
    for src in source_to_label:  # `src` is the image folder / annotated gesture
        label = source_to_label[src]  # the class the head is trained to predict
        cls_dir = images_dir / src
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing image folder: {cls_dir}")
        raw = load_class_annotations(ann_dir, src)
        for img_path in sorted(cls_dir.glob("*.jpg")):
            uuid = img_path.stem
            record = raw.get(uuid)
            bbox = bbox_for_label(record, src) if record else None
            if record is None and require_annotation:
                continue
            # user_id groups images by subject so a person can't leak across
            # splits; fall back to the uuid (its own group) if it's missing.
            user_id = record.get("user_id") if record else None
            rows.append(
                {
                    "uuid": uuid,
                    "label": label,
                    "label_idx": class_to_idx[label],
                    "path": str(img_path),
                    "bbox": bbox,
                    "user_id": user_id or f"u_{uuid}",
                    "has_ann": record is not None,
                    "source_label": src,
                }
            )

    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    if df.empty:
        raise RuntimeError(
            f"No images found under {images_dir} for classes {classes}. "
            "Have you run the download scripts (see DOCS/data.md)?"
        )
    return df
