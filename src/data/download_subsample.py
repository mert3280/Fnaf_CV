"""
Download subsample images from HaGRID sbercloud ZIPs using HTTP range requests.

Only fetches the ~100 images per class referenced in DATA/ann_subsample/*.json —
no need to download the full 25-63 GB per-class ZIP files.

Usage:
    python src/data/download_subsample.py
    python src/data/download_subsample.py --classes like fist palm
    python src/data/download_subsample.py --out DATA/images/subsample
"""

import argparse
import json
import os
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


def download_class(cls: str, uuids: list[str], out_dir: Path) -> tuple[int, int]:
    """Returns (downloaded, skipped) counts."""
    cls_dir = out_dir / cls
    cls_dir.mkdir(parents=True, exist_ok=True)

    needed = [u for u in uuids if not (cls_dir / f"{u}.jpg").exists()]
    skipped = len(uuids) - len(needed)

    if not needed:
        return 0, skipped

    url = f"{BASE_URL}/{cls}.zip"
    downloaded = 0

    with RemoteZip(url) as zf:
        for uuid in tqdm(needed, desc=cls, unit="img"):
            zip_path = f"{cls}/{uuid}.jpg"
            try:
                data = zf.read(zip_path)
                (cls_dir / f"{uuid}.jpg").write_bytes(data)
                downloaded += 1
            except KeyError:
                tqdm.write(f"  WARNING: {uuid}.jpg not found in ZIP")

    return downloaded, skipped


def main():
    parser = argparse.ArgumentParser(description="Download HaGRID subsample images")
    parser.add_argument(
        "--classes", nargs="+", default=WORKING_CLASSES,
        help="Gesture classes to download (default: all 8 working classes)"
    )
    parser.add_argument(
        "--ann", default="DATA/ann_subsample",
        help="Path to annotation JSON directory"
    )
    parser.add_argument(
        "--out", default="DATA/images/subsample",
        help="Output directory for images"
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann)
    out_dir = Path(args.out)

    print(f"Annotations: {ann_dir}")
    print(f"Output:      {out_dir}")
    print(f"Classes:     {args.classes}\n")

    total_dl, total_skip = 0, 0
    for cls in args.classes:
        uuids = load_uuids(ann_dir, cls)
        dl, skip = download_class(cls, uuids, out_dir)
        total_dl += dl
        total_skip += skip
        print(f"  {cls}: {dl} downloaded, {skip} already present\n")

    print(f"Done. {total_dl} downloaded, {total_skip} skipped.")
    print(f"Images saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
