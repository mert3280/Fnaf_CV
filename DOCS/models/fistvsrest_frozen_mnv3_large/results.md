# Results

_Frozen per-model copy of the auto-generated report (`src/train.py`, 2026-07-20;
device=cpu, input=224px, linear-probe on cached features). Strategy 3.2 / AD-21._

## fistvsrest_frozen_mnv3_large — mobilenetv3_large_100 (frozen backbone, head-only)

- **Backbone:** `mobilenetv3_large_100` (pretrained, frozen)
- **Best val accuracy:** 0.9221
- **Held-out test accuracy:** 0.9358  (random = 0.5000)

### Per-class report (test)

```
              precision    recall  f1-score   support

    not_fist     0.9156    0.9463    0.9307       149
        fist     0.9538    0.9270    0.9402       178

    accuracy                         0.9358       327
   macro avg     0.9347    0.9366    0.9354       327
weighted avg     0.9364    0.9358    0.9359       327

```

### Confusion matrix (test, rows = true)

| true\pred | not_fist | fist |
|---|---|---|
| **not_fist** | 141 | 8 |
| **fist** | 13 | 165 |
