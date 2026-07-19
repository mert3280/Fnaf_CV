# Results

_Frozen per-model copy of the auto-generated report (`src/train.py`, 2026-07-15;
device=cpu, input=224px, linear-probe on cached features)._

## palmfist_frozen_mnv3_large — mobilenetv3_large_100 (frozen backbone, head-only)

- **Backbone:** `mobilenetv3_large_100` (pretrained, frozen)
- **Best val accuracy:** 0.9639
- **Held-out test accuracy:** 0.9815  (random = 0.5000)

### Per-class report (test)

```
              precision    recall  f1-score   support

        palm     0.9874    0.9752    0.9812       161
        fist     0.9759    0.9878    0.9818       164

    accuracy                         0.9815       325
   macro avg     0.9817    0.9815    0.9815       325
weighted avg     0.9816    0.9815    0.9815       325

```

### Confusion matrix (test, rows = true)

| true\pred | palm | fist |
|---|---|---|
| **palm** | 157 | 4 |
| **fist** | 2 | 162 |
