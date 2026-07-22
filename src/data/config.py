"""Typed loader for configs/data.yaml.

Keeps the rest of the data pipeline from reaching into a raw dict and lets the
defaults live in one documented place. Import-safe (no torch/timm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SplitCfg:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    group_by_user: bool = True
    stratify: bool = True

    def __post_init__(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


@dataclass
class InputCfg:
    size: int = 224
    crop_mode: str = "bbox"  # bbox (two-stage, default -- AD-04) | full_frame
    bbox_pad: float = 0.15


@dataclass
class AugmentCfg:
    hflip: bool = True
    rotation_deg: float = 12.0
    translate: float = 0.06
    rrc_scale: tuple[float, float] = (0.75, 1.0)
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.02
    # Strategy-2.1 fix #3 (default 0.0 = OFF, so existing runs are unchanged).
    # Target the live webcam-distance gap: perspective warp simulates a hand held
    # close to the lens; blur simulates a fixed-focus webcam's soft focus. Ted
    # sets the magnitudes -- this only exposes the knobs (AD-09 is his call).
    perspective: float = 0.0   # RandomPerspective distortion_scale (0..1)
    blur_sigma: float = 0.0    # max GaussianBlur sigma (0 = no blur)


@dataclass
class LoaderCfg:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class DataConfig:
    sources: list[tuple[Path, Path]]  # (images_dir, ann_dir) pairs to index
    manifest: Path
    classes: list[str]                # TARGET labels -> head size + label_idx
    # Optional label grouping (Strategy 3.2 / AD-21): map each target label to the
    # image folders that feed it, so many gesture folders can collapse into one
    # class -- e.g. `not_fist: [palm, one, two_up, ...]`. When None, every target
    # label is its own folder (folder name == label; the palm/fist scheme).
    label_groups: dict[str, list[str]] | None = None
    # Downsample the majority label(s) to the minority label's count, drawn evenly
    # across each label's source folders (seeded). Balances a grouped negative
    # class (6 folders) against a single-folder positive (`fist`) without an
    # extra loss weight -- see AD-21.
    balance: bool = False
    split: SplitCfg = field(default_factory=SplitCfg)
    input: InputCfg = field(default_factory=InputCfg)
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    loader: LoaderCfg = field(default_factory=LoaderCfg)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def source_classes(self) -> list[str]:
        """Flat, order-stable list of image folders to index. Without
        `label_groups` this is just `classes`; with it, every source folder
        across all groups (deduped)."""
        if not self.label_groups:
            return list(self.classes)
        seen: set[str] = set()
        out: list[str] = []
        for label in self.classes:
            for src in self.label_groups.get(label, [label]):
                if src not in seen:
                    seen.add(src)
                    out.append(src)
        return out

    @property
    def source_to_label(self) -> dict[str, str]:
        """Image folder gesture -> TARGET label. Identity map when ungrouped."""
        if not self.label_groups:
            return {c: c for c in self.classes}
        return {
            src: label
            for label in self.classes
            for src in self.label_groups.get(label, [label])
        }

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/data.yaml") -> "DataConfig":
        raw = yaml.safe_load(Path(path).read_text())
        norm = raw.get("normalize", {})
        sources = [(Path(s["images"]), Path(s["ann"])) for s in raw["sources"]]
        return cls(
            sources=sources,
            manifest=Path(raw["manifest"]),
            classes=list(raw["classes"]),
            label_groups=raw.get("label_groups"),
            balance=bool(raw.get("balance", False)),
            split=SplitCfg(**raw.get("split", {})),
            input=InputCfg(**raw.get("input", {})),
            normalize_mean=tuple(norm.get("mean", (0.485, 0.456, 0.406))),
            normalize_std=tuple(norm.get("std", (0.229, 0.224, 0.225))),
            augment=_augment_from(raw.get("augment", {})),
            loader=LoaderCfg(**raw.get("loader", {})),
        )


def _augment_from(d: dict) -> AugmentCfg:
    d = dict(d)
    if "rrc_scale" in d:
        d["rrc_scale"] = tuple(d["rrc_scale"])
    return AugmentCfg(**d)
