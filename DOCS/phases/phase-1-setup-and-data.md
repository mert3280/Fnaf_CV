# Phase 1 — Setup & Data Validation (Week 1)

**Phase goal:** Stand up a reproducible project and *prove the HaGRID data is real, understood, and loadable* into PyTorch. No modeling yet — this phase de-risks the dataset.

---

## 1. Context

The `DATA/` folder contains **HaGRID** annotations (not images) in three splits:

```
DATA/
├── ann_subsample/    # small per-class JSON (fast iteration; ~300–450 KB each)
├── ann_train_val/    # full train+val annotations (~90 MB per class)
└── ann_test/         # test annotations (~8 MB per class)
```

18 classes: `call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted`.

Each JSON maps an **image UUID** → record:

```jsonc
{
  "0035fcf9-...": {
    "bboxes":   [[x, y, w, h]],          // normalized, COCO-style (top-left x,y + w,h)
    "labels":   ["fist"],
    "landmarks":[[[lx, ly], ... 21 pts]] // normalized 21 hand keypoints
    // plus (in full HaGRID): leading_hand, user_id, etc.
  }
}
```

> **Key fact:** the images themselves are a **separate download**. The full set is 716 GB; we will use the **subsample** for iteration and the **512px** variant for final training.

## 2. Tasks

### 2.1 Repository & environment
- [ ] Create `src/`, `notebooks/`, `models/`, `configs/`, `DOCS/` structure.
- [ ] `requirements.txt` / `environment.yml`: `torch`, `torchvision`, `timm`, `opencv-python`, `mediapipe`, `numpy`, `pandas`, `pillow`, `matplotlib`, `tensorboard`, `pydirectinput`, `onnx`, `onnxruntime`.
- [ ] `.gitignore` for `DATA/` images, `models/*.pt`, `__pycache__`, large artifacts. Keep annotation JSONs out of Git too if large — document where they live.
- [ ] Verify CUDA availability on the **local RTX 4060** (`torch.cuda.is_available()` → `True`, correct device name); record GPU/driver/torch/CUDA versions in `DOCS/env.md`. All training is local — no cloud (AD-02b).

### 2.2 Annotation parsing
- [ ] `src/data/hagrid_annotations.py`: load a class JSON → list of records `{uuid, label, bbox, landmarks}`.
- [ ] Sanity checks: every record has ≥1 bbox; labels match the file's class; bbox values ∈ [0,1].
- [ ] Build a unified index (Pandas DataFrame) across all classes for a chosen subset.

### 2.3 Image acquisition
- [ ] Download the HaGRID **subsample images** matching `ann_subsample`. Document the source URL, total size, and local path in `DOCS/data.md`.
- [ ] Confirm UUID ↔ image filename mapping; spot-check that bboxes overlay correctly on real images.
- [ ] Plan (don't fully download yet) the **512px** set for Phase 3.

### 2.4 PyTorch dataset
- [ ] `src/data/dataset.py`: `HagridDataset(Dataset)` returning `(image_tensor, label_idx)`.
  - Option A (baseline): classify the **full frame** resized to model input size.
  - Option B (fallback): **crop to bbox** (optionally padded) before resize — more robust if the hand is small in frame.
- [ ] Transforms: resize, normalize (ImageNet stats for pretrained backbones), train-time augmentation hooks (added in Phase 3).
- [ ] `DataLoader` with sane `num_workers`/`pin_memory`.

### 2.5 EDA
- [ ] `notebooks/01_eda.ipynb`: class balance bar chart, sample grid with bbox overlays, image-size distribution, brightness/quality spot checks.
- [ ] Decide the **working class subset** for control (see §4).

## 3. Deliverables
- Reproducible environment + repo scaffold.
- Verified annotation parser + `HagridDataset` + `DataLoader`.
- EDA notebook with rendered, labeled batches.
- `DOCS/data.md` (sources, sizes, paths) and the finalized **gesture → action map** (§4).

## 4. Design artifact — Gesture → FNAF action map (draft)

FNAF is mouse-driven with two interaction contexts. A **mode model** keeps the per-state vocabulary small and reliable. Final mapping is tuned in Phase 4; this is the starting proposal, chosen for low inter-class confusion.

**Office mode (camera down):**
| Gesture | Action |
|---|---|
| `like` (thumbs up) | Toggle **left** door |
| `fist` | Toggle **right** door |
| `one` | Tap **left** light |
| `two_up` | Tap **right** light |
| `palm` | Raise camera monitor → *Camera mode* |
| `mute` | Idle / no-op (explicit rest state) |

**Camera mode (monitor up):**
| Gesture | Action |
|---|---|
| `like` | Previous camera |
| `dislike` | Next camera |
| `ok` | (reserved) select / confirm |
| `fist` | Lower monitor → *Office mode* |
| `mute` | Idle / no-op |

> Working subset (~7–8 classes): `like, dislike, fist, one, two_up, palm, ok, mute`. Fewer, well-separated classes → higher real-time reliability than all 18.

## 5. Definition of done
Running one notebook cell loads the chosen subset and renders an augmented, correctly-labeled batch with bbox overlays.

## 6. Behind-schedule signal
End of Week 1 and I cannot load/visualize a single `(image, label)` batch, or I have annotations with no matching images.

## 7. Exit criteria → Phase 2
Verified `DataLoader` over the working subset + finalized class list + confirmed input resolution and normalization constants.
