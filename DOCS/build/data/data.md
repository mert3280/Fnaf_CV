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
| **subsample** | ~1920px (full HaGRID) | **held-out test set** (~100/class) | downloaded via `src/data/download_subsample.py` |
| **train_val** | ~1920px (full HaGRID) | **train + val pool** (1000/class) | downloaded via `src/data/download_train_val.py` |
| **512px** | 512px | Phase 3 final training (optional) | planned (Phase 3) |
| full | 1920px | — | **never download** (716 GB) |

### How the downloads work

The official HaGRID per-class ZIPs are 25–63 GB each. Rather than downloading everything, both scripts use **HTTP range requests** (`remotezip`) to fetch only the specific `{uuid}.jpg` entries they need out of `{class}.zip` — no intermediate ZIP stored locally.

```
# Test set — ~100 images/class referenced in ann_subsample
python src/data/download_subsample.py

# Train/val pool — 1000 images/class from ann_train_val, excluding the test UUIDs
python src/data/download_train_val.py
python src/data/download_train_val.py --classes like fist --per-class 50
python src/data/download_train_val.py --no-download        # re-run dedup verification only
```

`download_train_val.py` parallelizes across `--workers` (default 8) RemoteZip connections per class, since range requests are latency-bound.

Source URL pattern: `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/{class}.zip`

---

## Train / val / test split (AD-16)

The model split is a **single 70/15/15 split drawn at load time across ALL
downloaded images** (`train_val` + `subsample`, 8594 total), **grouped by
`user_id`** so no subject appears in two splits — an honest generalization test
(AD-10). Which folder an image lives in no longer implies its split. Realized
counts: **train 5919 / val 1273 / test 1402**. Full mechanics — index build,
grouped-split guarantees, crop/normalize/augment, config reference — live in
**[data-preparation.md](data-preparation.md)**.

> The two on-disk folders below are just **where the pixels were downloaded**
> (see history), not split roles. `download_*.py` still governs acquisition; the
> split above governs what the model trains/evaluates on.

### Download provenance — how the two folders stay byte-disjoint

The images were pulled in two batches that are guaranteed not to share content,
which is why they can be safely pooled and re-split:

1. **Within a folder:** UUIDs are unique JSON keys, so no image is fetched twice.
2. **Identity:** `download_train_val.py` **removes every `ann_subsample` UUID
   from the `train_val` candidate pool** before selecting — so the two folders
   share no UUID. (HaGRID's `ann_subsample` set is *not* disjoint from
   `train_val`: 92–96 of each class's 100 subsample UUIDs come from it.)
3. **Byte-level:** after downloading, the script MD5-hashes every `train_val`
   and `subsample` image and reports/removes any content-identical file (a
   `--no-download` run repeats just this check). Last run: **0 leaks, 0 dups.**

Both downloads are deterministic for a given `--seed` (default 42).

---

## Local image paths

| Variant | Local path | Role | Approx. size | Downloaded |
|---|---|---|---|---|
| subsample | `DATA/images/subsample/{class}/{uuid}.jpg` | split source (~781 images) | ~160–400 MB | [x] 2026-07-01 |
| train_val | `DATA/images/train_val/{class}/{uuid}.jpg` | split source (7813 images) | ~2–4 GB | [x] 2026-07-06 |
| 512px | `DATA/images/512px/{class}/{uuid}.jpg` | optional retrain | ~12 GB | [ ] Phase 3 |

Both image folders are pooled and re-split 70/15/15 by user at load time (see the split section above); the folder is no longer a split label.

Images are **git-ignored**. To re-create either set from scratch, run its script above.

### Test-set per-class counts (subsample: 781/800, 97.6%)

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

19 UUIDs from `ann_subsample/*.json` returned `KeyError` (not present in the sbercloud per-class ZIP) — likely version drift between the `ann_subsample` annotation snapshot and the current dataset release. Not a bug in the download script; re-running it will not recover these.

### Train/val per-class counts (7813/8000, 97.7%)

Target 1000/class. As with the subsample, some UUIDs `KeyError` out of the sbercloud ZIP (version drift between the annotation snapshot and the current release) — 187 total.

| Class | Downloaded | Missing-in-zip |
|---|---|---|
| fist | 997 | 3 |
| ok | 994 | 6 |
| mute | 990 | 10 |
| palm | 984 | 16 |
| like | 983 | 17 |
| dislike | 982 | 18 |
| one | 960 | 40 |
| two_up | 923 | 77 |

**Duplicate verification (MD5, 2026-07-06): 0 leaks, 0 intra-set duplicates across all 8 classes.** Re-run any time with `python src/data/download_train_val.py --no-download`.

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
| 2026-07-06 | Defined train/val/test split: subsample (781 imgs) → **test set**; new 1000/class pull from `ann_train_val` (excluding subsample UUIDs) → **train+val**, via `src/data/download_train_val.py`. Added identity- and byte-level dedup guarantees. |
| 2026-07-08 | Built the data-prep pipeline (`src/data/{config,hagrid_annotations,splits,transforms,dataset}.py` + `configs/data.yaml`): index, **70/15/15 split grouped by `user_id`** across all images (no subject leakage), crop/normalize/augment, `HagridDataset` + `build_dataloaders`. Retired the fixed subsample-as-test design; both folders now pooled and re-split. Documented in [data-preparation.md](data-preparation.md). |
