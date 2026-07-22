"""HagridDataset + the build_dataloaders() factory that wires the whole
data-prep pipeline together (config -> index -> split -> transforms -> loaders).

HagridDataset returns (image_tensor, label_idx). It only needs torch + PIL; the
transform is injected, so the heavy torchvision/timm deps live in transforms.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .hagrid_annotations import build_index
from .splits import Splits, make_splits, write_manifest
from .transforms import build_transforms, resolve_backbone_config


def padded_bbox_pixels(
    bbox, img_w: int, img_h: int, pad: float
) -> tuple[int, int, int, int]:
    """Normalized COCO `[x, y, w, h]` -> integer pixel `(left, top, right, bottom)`,
    grown by `pad` * bbox-size on each side and clamped to the image.

    THE single source of truth for the two-stage crop geometry (AD-04). Both the
    offline crop (`HagridDataset._crop_to_bbox`) and the live MediaPipe cropper
    (`src/rt/detector.py`) call this, so the runtime crop is byte-identical to
    training -- the AD A.3 contract that keeps "great test acc, useless live"
    from happening.
    """
    x, y, bw, bh = bbox
    pad_w, pad_h = bw * pad, bh * pad
    left = max(0.0, x - pad_w) * img_w
    top = max(0.0, y - pad_h) * img_h
    right = min(1.0, x + bw + pad_w) * img_w
    bottom = min(1.0, y + bh + pad_h) * img_h
    return int(left), int(top), int(right), int(bottom)


class HagridDataset(Dataset):
    """One split's images. crop_mode 'full_frame' uses the whole image; 'bbox'
    crops to the (padded) hand bbox before the transform (AD-04)."""

    def __init__(
        self,
        index,
        transform: Callable | None = None,
        crop_mode: str = "full_frame",
        bbox_pad: float = 0.15,
    ):
        self.index = index.reset_index(drop=True)
        self.transform = transform
        self.crop_mode = crop_mode
        self.bbox_pad = bbox_pad

    def __len__(self) -> int:
        return len(self.index)

    def _crop_to_bbox(self, img: Image.Image, bbox) -> Image.Image:
        """bbox is normalized COCO [x, y, w, h]; crop with padding, clamped.
        Geometry lives in `padded_bbox_pixels` so training and the live cropper
        stay byte-identical (AD A.3)."""
        if bbox is None:
            return img  # no annotation -> fall back to full frame
        w_img, h_img = img.size
        return img.crop(padded_bbox_pixels(bbox, w_img, h_img, self.bbox_pad))

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.crop_mode == "bbox":
            img = self._crop_to_bbox(img, row["bbox"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_idx"])


def build_full_index(cfg: DataConfig) -> pd.DataFrame:
    """Index every image across all configured sources into one DataFrame.

    Folder no longer implies split -- the 70/15/15 split is drawn across all of
    these by user_id (see splits.py). UUIDs are unique across sources (byte- and
    id-disjoint by construction, see DOCS/data.md), but we drop any accidental
    duplicate defensively so a UUID can never appear twice.

    With `cfg.label_groups`, several gesture folders collapse into one target
    label; with `cfg.balance`, the majority label is then downsampled to the
    minority count (drawn evenly across its source gestures, seeded) -- AD-21.
    """
    s2l = cfg.source_to_label
    frames = [
        build_index(images, ann, cfg.classes, cfg.class_to_idx, source_to_label=s2l)
        for images, ann in cfg.sources
    ]
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset="uuid").reset_index(drop=True)
    if cfg.balance:
        full = _balance_labels(full, cfg.split.seed)
    return full


def _balance_labels(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample every label to the smallest label's count.

    For a grouped label (several `source_label`s) the draw is spread as evenly as
    possible across its sources so the negative class stays diverse rather than
    over-representing one gesture; any shortfall from uneven source sizes is
    topped up from the remaining rows. Fully seeded => identical every run.
    """
    target = int(df["label"].value_counts().min())
    kept: list[pd.DataFrame] = []
    for _, grp in df.groupby("label", sort=False):
        if len(grp) <= target:
            kept.append(grp)
            continue
        srcs = list(grp["source_label"].unique())
        per = target // len(srcs)
        picks = [
            sub.sample(n=min(per, len(sub)), random_state=seed)
            for s in srcs
            for sub in [grp[grp["source_label"] == s]]
        ]
        drawn = pd.concat(picks) if picks else grp.iloc[:0]
        if len(drawn) < target:  # top up the remainder to hit `target` exactly
            remainder = grp.drop(drawn.index)
            drawn = pd.concat(
                [drawn, remainder.sample(n=target - len(drawn), random_state=seed)]
            )
        kept.append(drawn)
    return pd.concat(kept).sample(frac=1, random_state=seed).reset_index(drop=True)


def _split(cfg: DataConfig) -> Splits:
    s = cfg.split
    return make_splits(
        build_full_index(cfg),
        s.train, s.val, s.test, s.seed, s.group_by_user, s.stratify,
    )


def build_datasets(
    cfg: DataConfig,
    backbone: str | None = None,
    write_split_manifest: bool = False,
    augment_train: bool = True,
) -> tuple[dict[str, HagridDataset], Splits, tuple]:
    """Index -> split -> per-split HagridDataset with the right transforms.

    If `backbone` is given, input size + normalization come from timm's
    data_config for that backbone (authoritative); otherwise the config
    defaults are used. `augment_train=False` gives the train split the
    deterministic eval pipeline (used for the frozen Phase-2 baseline; AD-09
    introduces augmentation in Phase 3). Returns (datasets, splits, (size, mean, std)).
    """
    splits = _split(cfg)
    if write_split_manifest:
        write_manifest(splits, cfg.manifest)

    if backbone:
        size, mean, std = resolve_backbone_config(backbone)
    else:
        size = cfg.input.size
        mean, std = cfg.normalize_mean, cfg.normalize_std

    eval_tf = build_transforms(False, size, mean, std)
    train_tf = build_transforms(True, size, mean, std, cfg.augment) if augment_train else eval_tf

    datasets = {
        "train": HagridDataset(splits.train, train_tf, cfg.input.crop_mode, cfg.input.bbox_pad),
        "val": HagridDataset(splits.val, eval_tf, cfg.input.crop_mode, cfg.input.bbox_pad),
        "test": HagridDataset(splits.test, eval_tf, cfg.input.crop_mode, cfg.input.bbox_pad),
    }
    return datasets, splits, (size, mean, std)


def build_dataloaders(
    cfg: DataConfig,
    backbone: str | None = None,
    write_split_manifest: bool = False,
    augment_train: bool = True,
) -> tuple[dict[str, DataLoader], Splits, tuple]:
    """Full pipeline: returns ({train,val,test: DataLoader}, splits, (size,mean,std))."""
    datasets, splits, norm = build_datasets(cfg, backbone, write_split_manifest, augment_train)
    lc = cfg.loader
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=lc.batch_size, shuffle=True,
            num_workers=lc.num_workers, pin_memory=lc.pin_memory, drop_last=lc.drop_last,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=lc.batch_size, shuffle=False,
            num_workers=lc.num_workers, pin_memory=lc.pin_memory,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=lc.batch_size, shuffle=False,
            num_workers=lc.num_workers, pin_memory=lc.pin_memory,
        ),
    }
    return loaders, splits, norm


if __name__ == "__main__":
    # Smoke test: print the split summary from the real config.
    cfg = DataConfig.from_yaml(Path(__file__).resolve().parents[2] / "configs" / "data.yaml")
    print(_split(cfg).summary())
