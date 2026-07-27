# week4_tuned_fistvsrest — the tuned, registered click classifier (Week 4)

The Week-4 tuned successor to [`fistvsrest_frozen`](../fistvsrest_frozen_mnv3_large/README.md)
(AD-21) and the **first model in this project to be registered in the MLflow Model
Registry** — `fnaf-click-classifier` **version 1**, alias `champion`
([AD-24](../../build/architecture-and-decisions.md#ad-24--mlflow-model-registry--a-pyfunc-artifact-as-the-deployment-contract)).
Same task and label space as its predecessor: **`not_fist` (0) = no click,
`fist` (1) = click**. Full write-up:
[Week-4 report](../../class-related/week4/tuning-orchestration-report.md).

## Headline numbers (2026-07-26)

| Metric | Value | Predecessor |
|---|---|---|
| Best val accuracy | **1.0000** (308/308) | 0.9221 |
| Held-out test accuracy | **1.0000** (327/327, random 0.50) | 0.9358 |
| not_fist F1 / fist F1 (test) | 1.0000 / 1.0000 | 0.9307 / 0.9402 |
| Test errors | **0 of 327** | 21 of 327 |
| Clicks retained @ gate 0.70 | 99.4% | 91.0% |
| False clicks @ gate 0.70 (held out) | 0.0% | 3.4% |
| False clicks on **unseen** `ok` | **0.8%** | 0.8% |

**Read the perfect score correctly.** It is real — it survived an independent
evaluation path and a leakage audit (0 duplicate files across splits; test→train
nearest-neighbour cosine max 0.845 against a train→train control of 0.891, i.e. test
images are no closer to train than train is to itself; see
[`split-leakage-audit.json`](../../class-related/week4/split-leakage-audit.json)).
What it means is that **the offline benchmark is saturated**: 327/327 cannot
distinguish this model from the two runs behind it, and the one metric with headroom
left — false clicks on a gesture in **no** split (`ok`) — **did not improve at all**
(0.8% before and after). Fine-tuning sharpened the boundary around poses the model was
shown; it did not measurably improve generalization to poses it wasn't.

## Where the improvement actually came from

| step | val | Δ |
|---|---|---|
| deployed baseline (`fistvsrest_frozen`) | 0.9221 | — |
| **head init fixed only** (AD-22) | 0.9740 | **+5.19** |
| + 60-trial TPE search (3-seed mean) | 0.9827 | +0.87 |
| + unfreeze 2 backbone blocks (AD-08 Stage B) | 0.9968 | +1.41 |
| + augmentation (within noise) | 1.0000 | +0.32 |

The dominant term is **not** hyper-parameter search. timm's EfficientNet-family head
init uses `r = 1/sqrt(out_features)`, so a **2-class** head starts at ±0.707 weights →
±27 logits → CE **2.70** instead of ln(2)=0.69; the predecessor's 40 epochs were
largely spent undoing that, which is exactly why its record read *"val still climbing
at ep40."* See [AD-22](../../build/architecture-and-decisions.md#ad-22--head-initialization-for-a-2-class-head--proposed-2026-07-26--teds-call)
— **whether `pytorch_linear` becomes the project default is Ted's call**; the knob
ships with `timm_default` preserved so every older run stays reproducible.

## Recipe

Selected **by validation only** across both tuning arms; the test split was read
**once**, after selection (AD-10).

- **Config:** [`configs/tune_fistvsrest.yaml`](../../../configs/tune_fistvsrest.yaml)
  → data [`configs/data_fistvsrest.yaml`](../../../configs/data_fistvsrest.yaml)
  (snapshot saved next to the checkpoint as `config.snapshot.json`).
- **Winning cell:** Arm B `unfreeze2_aug` — backbone **top 2 blocks unfrozen**
  (AD-08 Stage B) at `lr × 0.1`, AD-09 augmentation **on**, head init
  `pytorch_linear`.
- **Optimizer (from Arm A's search):** `adam`, lr **4.98e-3**, weight decay
  **9.45e-3**, batch 64, label smoothing **0.062**, 15 epochs (best at 12), seed 42.
- **Data:** identical to the predecessor — 2,194 images balanced to
  not_fist 1,097 / fist 1,097, split **1,559 / 308 / 327** (70/15/15 grouped by
  `user_id`, seed 42), `ok` held out of training entirely. Data version
  **`f69624bde9e7`** (split-manifest hash).
- **Trainable params:** ~1.5 M (top blocks + head) vs 2,562 for the frozen probe.
- **Cost:** 23.7 min on CPU (the whole 5-cell Arm-B grid was 1 h 56 m).

## Honest caveats

- **`unfreeze2_aug` is not demonstrably the best of the five structural runs.** It
  beats `unfreeze2` by **one validation image** (308/308 vs 307/308) and *ties*
  `unfreeze2_auglive` exactly. Arm A's winner was chosen by 3-seed means because
  differences this size are noise; Arm B was not seed-repeated (~2 h/seed). The
  defensible claim is "unfreezing 2 blocks ≈ +1.4 points; augmentation on top is
  within noise" — the registered choice among the top three is arbitrary but recorded.
- **Every val number in the sweep is a best-of-N-epochs maximum**, hence optimistically
  biased (the 0.9221 baseline included, so comparisons stay fair).
- **Ground-truth boxes, not MediaPipe boxes.** All numbers use HaGRID's *annotated*
  bbox; live, MediaPipe crops. The Strategy-2 measurement put that gap at ~0.92 → 0.71
  for the 8-class analogue. **Unmeasured here and untouched by tuning.**
- **Nothing at arm's length.** The cursor-driving regime is absent from HaGRID.
- **Two cells were still improving at the epoch cap** (best epoch 14/15), so 15 epochs
  under-states them.
- **`mute`** (a closed-ish hand) stays the likeliest live confuser despite reading 0.0%
  offline.

## How to run

```bash
# live demo / play, straight from the checkpoint
python -m src.rt.webcam_demo --checkpoint models/week4_tuned_fistvsrest/best.pt
python -m src.control.play    --checkpoint models/week4_tuned_fistvsrest/best.pt --dry-run

# or serve the registered version over HTTP (champion alias)
python -m src.serve.app
curl -s http://127.0.0.1:8765/health

# reproduce the model itself (~2.5 h CPU)
python -m src.pipeline.run --dag training
```

Artifacts (git-ignored, under `models/week4_tuned_fistvsrest/`): `best.pt`,
`config.snapshot.json`, `model.onnx` (17.1 MB, torch parity 7.15e-07),
`tuning/{arm_a,arm_b,final}.json`, and the `struct_*/best.pt` checkpoint per Arm-B cell.
