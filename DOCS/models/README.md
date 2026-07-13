# Trained models

Per-model documentation. Each subfolder captures what a checkpoint *is*, how it
was trained, its honest test numbers, and how to run it — self-contained so the
record survives even though the weights themselves (`models/**/best.pt`) are
git-ignored.

| Model | Crop mode | Test acc | Live crop source | Doc |
|---|---|---|---|---|
| `baseline_mnv3_large` | `full_frame` | **0.530** | raw frame (`--no-detector`) | [baseline_mnv3_large/](baseline_mnv3_large/README.md) |
| `bbox_frozen_mnv3_large` | `bbox` | **0.917** | MediaPipe cropper (default) | [../results.md](../results.md) *(auto-gen report)* |

Both are `mobilenetv3_large_100`, frozen backbone, head-only, no augmentation,
same seed/epochs/split — so the only difference is the input crop. That
controlled A/B is the evidence for the two-stage pipeline
([AD-04](../architecture-and-decisions.md#ad-04--two-stage-pipeline-detect-and-crop-the-hand-then-classify)).

> `DOCS/results.md` is regenerated (and overwritten) by `src/train.py` on every
> run, so it always reflects the *most recent* training. Frozen per-model copies
> live in each subfolder here.
