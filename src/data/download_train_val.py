"""
Download the train/val image pool from HaGRID sbercloud ZIPs via HTTP range
requests (same mechanism as download_subsample.py).

Selects N images per class from DATA/ann_train_val/*.json, **excluding** every
UUID in DATA/ann_subsample/*.json. The subsample images (already on disk under
DATA/images/subsample/) are our held-out TEST set, and ~92-96 of each class's
100 subsample UUIDs are themselves drawn from the train_val split -- so without
this exclusion a naive pull would leak test images into training.

Guarantees against duplicates:
  1. Within a class: train_val UUIDs are unique JSON keys.
  2. train_val vs. test: all ann_subsample UUIDs are removed from the candidate
     pool (identity-level disjointness at the annotation level).
  3. Byte-level: run with --verify (default on) to MD5-hash every train_val and
     test image and report/remove any content-identical collisions.

Usage:
    python src/data/download_train_val.py
    python src/data/download_train_val.py --classes like fist --per-class 50
    python src/data/download_train_val.py --no-download --verify   # verify only
"""

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

BASE_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru"
    "/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset"
)

WORKING_CLASSES = ["like", "dislike", "fist", "one", "two_up", "palm", "ok", "mute"]


def load_uuids(ann_dir: Path, cls: str) -> list[str]:
    json_path = ann_dir / f"{cls}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")
    with open(json_path) as f:
        return list(json.load(f).keys())


def select_uuids(
    train_val: list[str], exclude: set[str], n: int, seed: int
) -> list[str]:
    """Deterministically pick up to n train_val UUIDs not in the exclude set."""
    candidates = sorted(set(train_val) - exclude)
    random.Random(seed).shuffle(candidates)
    return candidates[:n]


def download_class(
    cls: str, uuids: list[str], out_dir: Path, workers: int
) -> tuple[int, int, int]:
    """
    Returns (downloaded, skipped_present, missing_in_zip).

    Range requests are latency-bound, so we fan out across `workers` threads,
    each holding its own RemoteZip connection (RemoteZip is not thread-safe, so
    it must not be shared between threads).
    """
    cls_dir = out_dir / cls
    cls_dir.mkdir(parents=True, exist_ok=True)

    needed = [u for u in uuids if not (cls_dir / f"{u}.jpg").exists()]
    skipped = len(uuids) - len(needed)

    if not needed:
        return 0, skipped, 0

    url = f"{BASE_URL}/{cls}.zip"
    local = threading.local()
    counts = {"downloaded": 0, "missing": 0}
    lock = threading.Lock()

    def fetch(uuid: str) -> None:
        zf = getattr(local, "zf", None)
        if zf is None:
            zf = local.zf = RemoteZip(url)
        try:
            data = zf.read(f"{cls}/{uuid}.jpg")
            (cls_dir / f"{uuid}.jpg").write_bytes(data)
            with lock:
                counts["downloaded"] += 1
        except KeyError:
            tqdm.write(f"  WARNING: {uuid}.jpg not found in ZIP")
            with lock:
                counts["missing"] += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, u) for u in needed]
        for _ in tqdm(as_completed(futures), total=len(needed), desc=cls, unit="img"):
            pass

    return counts["downloaded"], skipped, counts["missing"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_no_duplicates(
    train_dir: Path, test_dir: Path, classes: list[str], remove_leaks: bool
) -> int:
    """
    Hash every test + train_val image. Report any content-identical pairs.
    A train_val image whose bytes match a test image is a leak; remove it if
    remove_leaks. Returns the number of leaks found.
    """
    print("\n=== Duplicate verification (MD5) ===")
    test_hashes: dict[str, Path] = {}
    for cls in classes:
        for p in (test_dir / cls).glob("*.jpg"):
            test_hashes[md5(p)] = p

    leaks = 0
    train_hashes: dict[str, Path] = {}
    for cls in classes:
        cls_leaks, cls_intra = 0, 0
        for p in sorted((train_dir / cls).glob("*.jpg")):
            digest = md5(p)
            if digest in test_hashes:
                cls_leaks += 1
                leaks += 1
                print(f"  LEAK  {cls}/{p.name} == test/{test_hashes[digest].name}")
                if remove_leaks:
                    p.unlink()
            elif digest in train_hashes:
                cls_intra += 1
                print(f"  DUP   {cls}/{p.name} == {train_hashes[digest].name}")
                if remove_leaks:
                    p.unlink()
            else:
                train_hashes[digest] = p
        flag = "" if (cls_leaks or cls_intra) else "  (clean)"
        print(f"  {cls:9s} leaks={cls_leaks} intra_dups={cls_intra}{flag}")

    verb = "removed" if remove_leaks else "found (not removed)"
    print(f"Total content-duplicates {verb}: {leaks} leaks")
    return leaks


def main():
    parser = argparse.ArgumentParser(description="Download HaGRID train/val image pool")
    parser.add_argument("--classes", nargs="+", default=WORKING_CLASSES)
    parser.add_argument("--ann", default="DATA/ann_train_val")
    parser.add_argument("--exclude-ann", default="DATA/ann_subsample",
                        help="Annotation dir whose UUIDs are the held-out test set")
    parser.add_argument("--out", default="DATA/images/train_val")
    parser.add_argument("--test-dir", default="DATA/images/subsample",
                        help="On-disk test images, used for byte-level verification")
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download connections per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading; only run verification")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the MD5 duplicate check")
    parser.add_argument("--keep-leaks", action="store_true",
                        help="Report byte-identical leaks but do not delete them")
    args = parser.parse_args()

    ann_dir = Path(args.ann)
    exclude_dir = Path(args.exclude_ann)
    out_dir = Path(args.out)

    print(f"Annotations:  {ann_dir}")
    print(f"Exclude (test): {exclude_dir}")
    print(f"Output:       {out_dir}")
    print(f"Per class:    {args.per_class}   seed={args.seed}")
    print(f"Classes:      {args.classes}\n")

    if not args.no_download:
        total_dl, total_skip, total_missing = 0, 0, 0
        for cls in args.classes:
            train_val = load_uuids(ann_dir, cls)
            exclude = set(load_uuids(exclude_dir, cls))
            picks = select_uuids(train_val, exclude, args.per_class, args.seed)
            dl, skip, missing = download_class(cls, picks, out_dir, args.workers)
            total_dl += dl
            total_skip += skip
            total_missing += missing
            print(f"  {cls}: {dl} downloaded, {skip} already present, "
                  f"{missing} missing-in-zip (target {len(picks)})\n")
        print(f"Done. {total_dl} downloaded, {total_skip} skipped, "
              f"{total_missing} missing-in-zip.")
        print(f"Images saved to: {out_dir.resolve()}")

    if not args.no_verify:
        verify_no_duplicates(
            out_dir, Path(args.test_dir), args.classes,
            remove_leaks=not args.keep_leaks,
        )


if __name__ == "__main__":
    main()
