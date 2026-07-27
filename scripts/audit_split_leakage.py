"""Audit the train/test split for leakage. Written because the Week-4 tuned model
scored **100% on the held-out test split** (327/327), and a perfect score is a
claim that has to be attacked before it is published.

Three independent checks:

  1. **Exact file duplicates** -- MD5 of every image, cross-referenced across
     splits. Catches the same file indexed twice under different UUIDs.
  2. **What `group_by_user` actually buys** -- HaGRID's `user_id` gives a median of
     ONE image per user here, so the grouped split is close to a per-image split.
     If the same physical person appears under several `user_id`s, grouping does not
     stop them straddling splits.
  3. **Near-duplicate detection in feature space** -- for every test image, the
     cosine similarity to its nearest TRAIN neighbour, using the frozen backbone's
     pooled features. Near-duplicates (same scene, same hand, adjacent frames) sit
     at very high similarity. The control is train->train nearest-neighbour
     similarity (excluding self): that says what "normal" closeness looks like in
     this dataset, so the test numbers can be read against something.

    python scripts/audit_split_leakage.py --data-config configs/data_fistvsrest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.config import DataConfig  # noqa: E402
from src.data.dataset import HagridDataset, build_full_index  # noqa: E402
from src.data.splits import make_splits  # noqa: E402
from src.data.transforms import build_transforms, resolve_backbone_config  # noqa: E402
from src.models.build import build_model, pooled_features  # noqa: E402


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def features(index, backbone: str, cfg: DataConfig, device: str = "cpu") -> np.ndarray:
    """L2-normalized pooled features, in the SAME row order as `index`."""
    size, mean, std = resolve_backbone_config(backbone)
    tf = build_transforms(False, size, mean, std)
    ds = HagridDataset(index, tf, crop_mode=cfg.input.crop_mode, bbox_pad=cfg.input.bbox_pad)
    model = build_model(backbone, 2, True, True).to(device).eval()
    out = []
    for x, _ in DataLoader(ds, batch_size=64, shuffle=False):   # shuffle=False!
        out.append(pooled_features(model, x.to(device)).cpu())
        print(f"    {sum(len(o) for o in out)}/{len(ds)}", end="\r")
    print()
    f = torch.cat(out).numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default="configs/data_fistvsrest.yaml")
    ap.add_argument("--backbone", default="mobilenetv3_large_100")
    ap.add_argument("--out", default="DOCS/class-related/week4/split-leakage-audit.json")
    args = ap.parse_args()

    cfg = DataConfig.from_yaml(REPO / args.data_config)
    index = build_full_index(cfg)
    s = cfg.split
    sp = make_splits(index, s.train, s.val, s.test, s.seed, s.group_by_user, s.stratify)
    report: dict = {"data_config": args.data_config,
                    "sizes": {k: int(len(getattr(sp, k))) for k in ("train", "val", "test")}}

    # -- 1. exact duplicates ------------------------------------------------- #
    print("1) hashing every image for exact duplicates...")
    digests: dict[str, list[tuple[str, str]]] = {}
    for split in ("train", "val", "test"):
        for _, row in getattr(sp, split).iterrows():
            digests.setdefault(md5(Path(row["path"])), []).append((split, row["uuid"]))
    dups = {d: v for d, v in digests.items() if len({s for s, _ in v}) > 1}
    print(f"   unique file hashes: {len(digests)} for {len(index)} indexed images")
    print(f"   files appearing in MORE THAN ONE split: {len(dups)}")
    report["exact_duplicates"] = {
        "unique_hashes": len(digests), "indexed_images": int(len(index)),
        "cross_split_duplicate_files": len(dups),
        "examples": [{"hash": d[:12], "occurrences": v} for d, v in list(dups.items())[:5]],
    }

    # -- 2. what user grouping buys ----------------------------------------- #
    counts = Counter(index["user_id"])
    per_user = np.array(sorted(counts.values()))
    report["user_grouping"] = {
        "images": int(len(index)), "unique_user_ids": len(counts),
        "median_images_per_user": float(np.median(per_user)),
        "mean_images_per_user": round(float(per_user.mean()), 3),
        "max_images_per_user": int(per_user.max()),
        "users_with_one_image": int((per_user == 1).sum()),
        "note": "a median of 1 image/user means the grouped split is close to a "
                "per-image split; grouping cannot separate the same PERSON if HaGRID "
                "gave them multiple user_ids",
    }
    print(f"2) user grouping: {len(counts)} user_ids for {len(index)} images "
          f"(median {np.median(per_user):.0f}/user)")

    # -- 3. near-duplicates in feature space -------------------------------- #
    print("3) computing frozen-backbone features (deterministic order)...")
    print("   train:")
    ftr = features(sp.train, args.backbone, cfg)
    print("   test:")
    fte = features(sp.test, args.backbone, cfg)

    sim_te = fte @ ftr.T                       # cosine (both L2-normalized)
    nn_te = sim_te.max(axis=1)
    nn_te_idx = sim_te.argmax(axis=1)

    sim_tr = ftr @ ftr.T
    np.fill_diagonal(sim_tr, -1.0)             # exclude self
    nn_tr = sim_tr.max(axis=1)

    def dist(a: np.ndarray) -> dict:
        return {"mean": round(float(a.mean()), 4), "p50": round(float(np.percentile(a, 50)), 4),
                "p95": round(float(np.percentile(a, 95)), 4), "max": round(float(a.max()), 4),
                "frac_over_0.95": round(float((a > 0.95).mean()), 4),
                "frac_over_0.98": round(float((a > 0.98).mean()), 4),
                "frac_over_0.99": round(float((a > 0.99).mean()), 4)}

    report["nearest_neighbour_cosine"] = {
        "test_to_train": dist(nn_te),
        "train_to_train_control": dist(nn_tr),
        "interpretation": "if test->train similarity looked much higher than the "
                          "train->train control, test images would be effectively "
                          "duplicated in training. Compare the two rows.",
    }
    print(f"   test->train  NN cosine: mean {nn_te.mean():.4f}  p95 "
          f"{np.percentile(nn_te, 95):.4f}  max {nn_te.max():.4f}  "
          f">0.98: {(nn_te > 0.98).mean()*100:.1f}%")
    print(f"   train->train NN cosine: mean {nn_tr.mean():.4f}  p95 "
          f"{np.percentile(nn_tr, 95):.4f}  max {nn_tr.max():.4f}  "
          f">0.98: {(nn_tr > 0.98).mean()*100:.1f}%")

    # the most suspicious pairs, for eyeballing
    order = np.argsort(-nn_te)[:10]
    report["most_similar_test_train_pairs"] = [
        {"cosine": round(float(nn_te[i]), 4),
         "test_uuid": sp.test.iloc[int(i)]["uuid"],
         "test_label": sp.test.iloc[int(i)]["label"],
         "test_user": str(sp.test.iloc[int(i)]["user_id"]),
         "train_uuid": sp.train.iloc[int(nn_te_idx[i])]["uuid"],
         "train_label": sp.train.iloc[int(nn_te_idx[i])]["label"],
         "train_user": str(sp.train.iloc[int(nn_te_idx[i])]["user_id"]),
         "same_label": bool(sp.test.iloc[int(i)]["label"]
                            == sp.train.iloc[int(nn_te_idx[i])]["label"])}
        for i in order
    ]
    print("   top-3 most similar test/train pairs:")
    for p in report["most_similar_test_train_pairs"][:3]:
        print(f"     cos {p['cosine']:.4f}  test {p['test_uuid'][:8]} ({p['test_label']}, "
              f"user {p['test_user'][:8]}) ~ train {p['train_uuid'][:8]} "
              f"({p['train_label']}, user {p['train_user'][:8]})")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
