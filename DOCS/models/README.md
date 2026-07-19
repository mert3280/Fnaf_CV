# Trained models

Per-model documentation. Each subfolder captures what a checkpoint *is*, how it
was trained, its honest test numbers, and how to run it — self-contained so the
record survives even though the weights themselves (`models/**/best.pt`) are
git-ignored.

| Model | Classes | Crop mode | Test acc | Live crop source | Doc |
|---|---|---|---|---|---|
| `baseline_mnv3_large` | 8 | `full_frame` | **0.530** | raw frame (`--no-detector`) | [baseline_mnv3_large/](baseline_mnv3_large/README.md) |
| `bbox_frozen_mnv3_large` | 8 | `bbox` | **0.917** | MediaPipe cropper (default) | [bbox_frozen_mnv3_large/](bbox_frozen_mnv3_large/README.md) |
| `palmfist_frozen_mnv3_large` | **2** (AD-18) | `bbox` | **0.9815** | MediaPipe cropper (default) | [palmfist_frozen_mnv3_large/](palmfist_frozen_mnv3_large/README.md) |

Both are `mobilenetv3_large_100`, frozen backbone, head-only, no augmentation,
same seed/epochs/split — so the only difference is the input crop. That
controlled A/B is the evidence for the two-stage pipeline
([AD-04](../build/architecture-and-decisions.md#ad-04--two-stage-pipeline-detect-and-crop-the-hand-then-classify)).

> **Note:** the two 8-class models are **pre-pivot** era (kept as the AD-04 A/B
> record; still runnable from their saved configs). `palmfist_frozen_mnv3_large`
> (2026-07-15) is the **current** model — the binary click classifier of the
> cursor-control pivot ([AD-17/AD-18](../build/architecture-and-decisions.md#scope-pivot--cursor-control-2026-07-14),
> [Strategy 3](../build/strategies/3-cursor-and-click/03-cursor-and-click.md)),
> same recipe as `bbox_frozen` with the class list trimmed to palm/fist.

> `src/train.py` writes its report to `DOCS/results.md` (overwritten every run),
> so that file always reflects the *most recent* training. Each subfolder here
> keeps a frozen per-model copy — the bbox report was regenerated from its
> checkpoint after the DOCS reorg removed the transient root copy.
