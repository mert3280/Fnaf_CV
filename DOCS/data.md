# Data Reference

**Dataset:** HaGRID (HAnd Gesture Recognition Image Dataset) — WACV 2024  
**Source repo:** https://github.com/hukenovs/hagrid  
**Status:** Annotations in repo; images downloaded separately (git-ignored).

---

## What's in this repo (`DATA/`)

Annotation JSONs only — no pixel data. Three splits, one JSON per class:

```
DATA/
├── ann_subsample/    # small per-class JSON, matches the subsample image set
├── ann_train_val/    # full train + val annotations (~90 MB per class)
└── ann_test/         # test annotations (~8 MB per class)
```

Each JSON maps an image UUID to a record:

```jsonc
{
  "0035fcf9-...": {
    "bboxes":    [[x, y, w, h]],        // normalized, COCO-style (top-left + w/h), values ∈ [0,1]
    "labels":    ["fist"],
    "landmarks": [[[lx, ly], ...]]      // 21 normalized hand keypoints
    // full HaGRID also includes: leading_hand, user_id
  }
}
```

Image filename: `<uuid>.jpg`

---

## Working class subset (AD-03)

Only 8 of 18 classes are downloaded and trained on:

`like`, `dislike`, `fist`, `one`, `two_up`, `palm`, `ok`, `mute`

This avoids near-duplicate classes (`peace`/`peace_inverted`, etc.) that degrade live reliability. See [architecture-and-decisions.md](architecture-and-decisions.md) AD-03.

---

## Image variants & download plan (AD-02)

| Variant | Resolution | Use | Status |
|---|---|---|---|
| **subsample** | ~1920px (full HaGRID) | Phase 1–2 fast iteration | downloading via `src/data/download_subsample.py` |
| **512px** | 512px | Phase 3 final training | planned (Phase 3) |
| full | 1920px | — | **never download** (716 GB) |

### How the subsample download works

The official HaGRID per-class ZIPs are 25–63 GB each. Rather than downloading everything, `src/data/download_subsample.py` uses **HTTP range requests** (`remotezip`) to fetch only the 100 images per class referenced in `ann_subsample/*.json` — no intermediate ZIP stored locally.

```
python src/data/download_subsample.py                        # all 8 working classes
python src/data/download_subsample.py --classes like fist    # specific classes only
```

Source URL pattern: `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/{class}.zip`

---

## Local image paths

| Variant | Local path | Approx. size | Downloaded |
|---|---|---|---|
| subsample | `DATA/images/subsample/{class}/{uuid}.jpg` | ~160–400 MB (781 images) | [x] 2026-07-01 |
| 512px | `DATA/images/512px/{class}/{uuid}.jpg` | ~12 GB | [ ] Phase 3 |

Images are **git-ignored**. To re-download the subsample from scratch, run the script above.

### Actual per-class counts (781/800, 97.6%)

| Class | Downloaded | Missing |
|---|---|---|
| fist | 100 | 0 |
| ok | 100 | 0 |
| like | 98 | 2 |
| one | 98 | 2 |
| palm | 98 | 2 |
| mute | 98 | 2 |
| dislike | 96 | 4 |
| two_up | 93 | 7 |

19 UUIDs from `ann_subsample/*.json` returned `KeyError` (not present in the sbercloud per-class ZIP) — likely a version drift between the `ann_subsample` annotation snapshot and the current dataset release. Not a bug in the download script; re-running it will not recover these. 781 images is sufficient for Phase 1 dataloader/EDA validation.

---

## UUID ↔ filename mapping

The image file for annotation record `"0035fcf9-..."` is `0035fcf9-....jpg` in the corresponding class subfolder. Spot-check by overlaying the `bboxes` value (COCO normalized → pixel: `x_px = x * W`, `w_px = w * W`) on a loaded image.

---

## Augmentation note — horizontal flips (AD-09)

Horizontal flip is **not label-safe for all 8 classes.** `like` (thumbs-up) and `dislike` are hand-symmetric; `one`, `two_up`, `palm`, `fist`, `ok`, `mute` are too. None of the working-subset classes have a mirrored partner in the subset, so **flip can be applied freely** within this 8-class set. Revisit if the class list changes.

---

## Change log

| Date | Change |
|---|---|
| 2026-07-01 | File created; subsample download in progress. |
