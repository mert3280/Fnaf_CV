# fistvsrest_frozen_mnv3_large — the fist-vs-rest click classifier (Strategy 3.2)

The second click model of the cursor-control pivot ([AD-21](../../build/architecture-and-decisions.md#ad-21--reframe-the-click-classifier-as-fist-vs-not_fist--accepted-2026-07-20-strategy-32)).
Same two-stage pipeline and same head size as [`palmfist_frozen`](../palmfist_frozen_mnv3_large/README.md),
but the negative class is redefined: **`not_fist`** (idx 0) = no click, trained on
**six diverse non-fist gestures**; **`fist`** (idx 1) = click. The point is that an
*unrecognised* hand shape now resolves to no-click by construction instead of
being forced onto palm/fist and risking a false click.
Full design: [Strategy 3.2](../../build/strategies/3-cursor-and-click/03.2-fist-vs-rest.md).

## Headline numbers (2026-07-20)

| Metric | Value |
|---|---|
| Best val accuracy | 0.9221 |
| Held-out test accuracy | 0.9358 (random = 0.50) |
| not_fist F1 / fist F1 (test) | 0.9307 / 0.9402 |
| Test errors | 21 of 327 (8 not_fist→fist, 13 fist→not_fist) |

Raw test accuracy is **not** the headline — it's lower than palm/fist's 0.9815
*by design* (`not_fist` is a much harder negative: it spans six gestures, some
partly closed like `mute`). The metric that matters is below.

## The number that matters — open-set false-click rate

`src/eval_openset.py` runs both models over whole gesture folders and reports how
often each fires `fist` (a click). For a **non-fist** gesture, lower is better.
`ok` is held out of *both* models' training — the fair unseen-pose A/B. 250-image
seeded sample per gesture, conf-gate 0.70 (the FSM's click threshold, AD-20):

| gesture | palm/fist — click% | **fist-vs-rest — click%** | seen by fist-vs-rest? |
|---|---|---|---|
| **fist** *(want HIGH)* | 99.2 | **96.0** | yes (positive) |
| palm | 0.4 | **0.8** | yes |
| one *(pointing!)* | 75.6 | **9.2** | yes |
| two_up | 81.2 | **2.4** | yes |
| like | 41.2 | **2.8** | yes |
| dislike | 74.4 | **3.2** | yes |
| mute | 60.4 | **0.0** | yes |
| **ok** *(unseen by BOTH)* | 6.8 | **0.8** | **no — open-set** |

Reproduce:

```bash
python -m src.eval_openset --limit 250 \
    --checkpoints models/palmfist_frozen_mnv3_large/best.pt \
                  models/fistvsrest_frozen_mnv3_large/best.pt
```

### What this says

- **The palm/fist model false-clicks constantly on ordinary gestures** it never
  trained on — a pointing hand (`one`) fires a click 76% of the time, `two_up`
  81%, `mute` 60%. In live cursor use those are exactly the incidental poses a
  hand drifts through. This is the accidental-click problem 3.2 set out to fix.
- **Fist-vs-rest collapses those to single digits (0–9%)** because its negative
  class learned them.
- **On `ok`, unseen by both, fist-vs-rest still wins 0.8% vs 6.8%** — the diverse
  negative generalises "unknown → no-click" even to a pose it never saw. That's
  the open-set evidence the reframe works, not just memorisation.
- **The honest cost:** `fist` click-rate drops 99.2% → 96.0% (~3% of real fists
  read as not_fist — the 13/327 fist→not_fist test errors). Fewer false clicks
  bought with slightly less eager true clicks; the FSM's K-frame confirm means a
  briefly-missed fist just needs the squeeze held one more frame.

## Recipe — deliberately identical to `palmfist_frozen`

Same backbone, seed, epochs, freeze policy, `crop_mode: bbox`; the only changes
are the class definition and `balance` (see AD-21). Keeps the two directly
comparable.

- **Config:** [`configs/fistvsrest_frozen.yaml`](../../../configs/fistvsrest_frozen.yaml) → [`configs/data_fistvsrest.yaml`](../../../configs/data_fistvsrest.yaml) (snapshot saved next to the checkpoint).
- **Data:** 2,194 images balanced to not_fist 1,097 / fist 1,097 (not_fist drawn
  ~183 evenly across palm/one/two_up/like/dislike/mute), split 1,559 / 308 / 327
  (70/15/15 grouped by `user_id`, seed 42 — AD-16; leak assertions pass). `ok`
  excluded from training on purpose (open-set hold-out).
- **Training:** Stage A (AD-08) — frozen backbone, head-only linear-probe on
  cached features; 40 epochs, AdamW, lr 1e-3, no augmentation. Val still climbing
  at ep40 (0.8799 @ ep20 → 0.9221 @ ep40) with train 0.973 — no overfitting;
  levers (more epochs, `unfreeze_blocks`, augmentation) remain Ted's call.
- `label_idx`: `not_fist=0, fist=1`.

## Honest caveats

- All numbers are on HaGRID's **annotated** crops. Live, the crop comes from
  MediaPipe at arm's-length distances — the Strategy-2 domain gap applies here
  too, and is **unmeasured until the live test**.
- The ~3% fist-recall cost is real; if live clicking feels unreliable, raise
  `not_fist` diversity/balance or revisit unfreezing before blaming the FSM.
- `mute` reads as 0% fist here but is a *closed-ish* hand — the one negative most
  visually like a fist; watch it live as the likeliest confuser.

## How to run

```bash
python -m src.rt.webcam_demo --checkpoint models/fistvsrest_frozen_mnv3_large/best.pt
python -m src.control.play    --checkpoint models/fistvsrest_frozen_mnv3_large/best.pt --dry-run
python -m src.control.play    --checkpoint models/fistvsrest_frozen_mnv3_large/best.pt
```
