# Results

_Regenerated from `models/bbox_frozen_mnv3_large/best.pt` on the deterministic
by-user test split (seed 42). device=cpu, input=224px, crop_mode=bbox._

## Two-stage bbox-crop — mobilenetv3_large_100 (frozen backbone, head-only)

- **Backbone:** `mobilenetv3_large_100` (pretrained, frozen)
- **Crop mode:** `bbox` (+15% pad) — the two-stage detect-and-crop pipeline (AD-04)
- **Best val accuracy:** 0.9403
- **Held-out test accuracy:** 0.9165  (random = 0.1250)

### Per-class report (test)

```
              precision    recall  f1-score   support

        like     0.9344    0.9096    0.9218       188
     dislike     0.9824    0.9489    0.9653       176
        fist     0.9415    0.9471    0.9443       170
         one     0.8228    0.8075    0.8150       161
      two_up     0.8579    0.8920    0.8747       176
        palm     0.9179    0.9521    0.9347       188
          ok     0.9464    0.8883    0.9164       179
        mute     0.9253    0.9817    0.9527       164

    accuracy                         0.9165      1402
   macro avg     0.9161    0.9159    0.9156      1402
weighted avg     0.9171    0.9165    0.9165      1402
```

### Confusion matrix (test, rows = true)

| true\pred | like | dislike | fist | one | two_up | palm | ok | mute |
|---|---|---|---|---|---|---|---|---|
| **like** | 171 | 3 | 2 | 5 | 1 | 2 | 0 | 4 |
| **dislike** | 3 | 167 | 2 | 0 | 0 | 0 | 0 | 4 |
| **fist** | 0 | 0 | 161 | 8 | 1 | 0 | 0 | 0 |
| **one** | 5 | 0 | 3 | 130 | 17 | 2 | 2 | 2 |
| **two_up** | 0 | 0 | 2 | 14 | 157 | 0 | 2 | 1 |
| **palm** | 2 | 0 | 0 | 0 | 3 | 179 | 4 | 0 |
| **ok** | 1 | 0 | 0 | 1 | 4 | 12 | 159 | 2 |
| **mute** | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 161 |

> **Note.** These are **offline** numbers on HaGRID's *own annotated* bbox crops.
> The live pipeline crops with **MediaPipe** instead; on detected HaGRID stills
> that drops to ~0.71 (the train/serve gap — see
> [strategies/2-two-stage-mediapipe-crop](../../build/strategies/2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md)).
