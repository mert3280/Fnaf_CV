# Legacy — the gesture-vocabulary plan (superseded 2026-07-14)

Everything in this folder describes the project's **original control design**:
train an ~8-class gesture classifier (`like, dislike, fist, one, two_up, palm,
ok, mute`) and map each gesture to a discrete FNAF action through a mode-aware
(Office/Camera) state machine.

That design was **superseded on 2026-07-14** by the **cursor-control pivot**
([AD-17](../build/architecture-and-decisions.md#ad-17--pivot-to-cursor-control-motion-tracked-cursor--binary-click-classifier)):

> MediaPipe hand tracking (pipeline stage 1) moves the mouse cursor directly;
> the transfer-learned classifier (stage 2) shrinks to a **binary palm/fist**
> decision — palm = no click, fist = click. The player points at what they want
> and "squeezes" to click, instead of memorizing a gesture vocabulary.

## Why the pivot

- **One universal interaction instead of a vocabulary.** FNAF is entirely
  mouse-driven (hover to pan the camera, click doors/lights/monitor). A cursor
  + click covers *all* of it — no gesture→action map, no Office/Camera mode
  state machine, nothing for the player to memorize.
- **Reliability concentrates where the model is strongest.** The live tests
  (see [Strategy 2](../build/strategies/2-two-stage-mediapipe-crop/02-two-stage-mediapipe-crop.md))
  showed the 8-class classifier reads gestures well but degrades with distance
  and inter-class confusion. `palm` vs `fist` are the two most visually distinct
  poses in the set; a binary task removes almost every confusion pair.
- **The learning objective is untouched.** Stage 2 is still the same
  transfer-learned MobileNetV3 (freeze → progressively unfreeze) — only its
  class count changes. Stage 1 (MediaPipe) was already in the pipeline as the
  cropper; it now additionally emits the hand position that drives the cursor.

## What lives here

| Doc | What it was |
|---|---|
| [proposal.md](proposal.md) | Original project proposal (2026-06-30): 8-gesture vocabulary, gesture→action mapping, 5-week schedule. Still accurate about dataset, stack, and learning goals. |
| [phases/](phases/) | The five phase plans + schedule + implementation plan built around the old control design. Phases 1–2 were **completed as written** (their outputs — data pipeline, splits, baseline, bbox model — all carry forward). Phases 3–5 are replaced by the new plan. |

## Where the current plan lives

- **New plan / roadmap:** [DOCS/build/plan.md](../build/plan.md)
- **New strategy (cursor + click):** [DOCS/build/strategies/3-cursor-and-click/](../build/strategies/3-cursor-and-click/03-cursor-and-click.md)
- **Decision records for the pivot:** AD-17…AD-20 in
  [architecture-and-decisions.md](../build/architecture-and-decisions.md)

Nothing in this folder is maintained; it is kept as the honest record of what
was planned, what was built under that plan, and why the direction changed.
