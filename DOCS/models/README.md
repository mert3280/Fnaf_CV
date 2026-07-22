# Trained models

Per-model documentation. Each subfolder captures what a checkpoint *is*, how it
was trained, its honest test numbers, and how to run it — self-contained so the
record survives even though the weights themselves (`models/**/best.pt`) are
git-ignored.

| Model | Classes | Crop mode | Test acc | Live crop source | Doc |
|---|---|---|---|---|---|
| `baseline_mnv3_large` | 8 | `full_frame` | **0.530** | raw frame (`--no-detector`) | [baseline_mnv3_large/](baseline_mnv3_large/README.md) |
| `bbox_frozen_mnv3_large` | 8 | `bbox` | **0.917** | MediaPipe cropper (default) | [bbox_frozen_mnv3_large/](bbox_frozen_mnv3_large/README.md) |
| `palmfist_frozen_mnv3_large` | **2** (AD-18) palm/fist | `bbox` | **0.9815** | MediaPipe cropper (default) | [palmfist_frozen_mnv3_large/](palmfist_frozen_mnv3_large/README.md) |
| `fistvsrest_frozen_mnv3_large` | **2** (AD-21) not_fist/fist | `bbox` | 0.9358 ¹ | MediaPipe cropper (default) | [fistvsrest_frozen_mnv3_large/](fistvsrest_frozen_mnv3_large/README.md) |

Both are `mobilenetv3_large_100`, frozen backbone, head-only, no augmentation,
same seed/epochs/split — so the only difference is the input crop. That
controlled A/B is the evidence for the two-stage pipeline
([AD-04](../build/architecture-and-decisions.md#ad-04--two-stage-pipeline-detect-and-crop-the-hand-then-classify)).

¹ For `fistvsrest`, raw test accuracy is the wrong headline (harder negative,
lower by design). The real metric is the **open-set false-click rate** — on a
*pointing* hand the palm/fist model fires 76% of the time vs fist-vs-rest's 9%,
and on the unseen `ok` gesture 6.8% vs 0.8%. See its README's A/B table.

> **Note:** the two 8-class models are **pre-pivot** era (kept as the AD-04 A/B
> record; still runnable from their saved configs). `palmfist_frozen_mnv3_large`
> (2026-07-15, AD-18) trims the classes to palm/fist;
> **`fistvsrest_frozen_mnv3_large`** (2026-07-20, AD-21) is the **current** model
> — same recipe, but the negative class becomes `not_fist` (six diverse
> gestures) so unknown poses resolve to no-click and accidental clicks drop
> ([Strategy 3.2](../build/strategies/3-cursor-and-click/03.2-fist-vs-rest.md)).
> The palm/fist model stays as the A/B baseline.

> `src/train.py` writes its report to `DOCS/results.md` (overwritten every run),
> so that file always reflects the *most recent* training. Each subfolder here
> keeps a frozen per-model copy — the bbox report was regenerated from its
> checkpoint after the DOCS reorg removed the transient root copy.
