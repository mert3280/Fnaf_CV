"""HaGRID data-preparation pipeline: config -> annotation index -> train/val/test
split -> transforms -> DataLoaders. See DOCS/data-preparation.md."""

from .config import DataConfig
from .hagrid_annotations import build_index, validate_class
from .splits import Splits, assert_disjoint, make_splits, write_manifest

__all__ = [
    "DataConfig",
    "build_index",
    "validate_class",
    "Splits",
    "make_splits",
    "assert_disjoint",
    "write_manifest",
]
