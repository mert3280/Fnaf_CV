# Claude / AI Usage Plan

> This file states my **intent** for using Claude and other AI tools on this project. It is a living document and will be updated each week. The actual week-by-week record of what I used AI for lives in [AI-usage.md](AI-usage.md). The agent-context file that tells Claude its role and guardrails on this repo is at the repo root: [`CLAUDE.md`](../CLAUDE.md).

## Guiding principle

AI is used to **accelerate scaffolding and debugging, not to replace the learning objectives.** The whole point of this project (see the original [proposal.md](../legacy/proposal.md) → "My Takeaway") is for *me* to internalize transfer learning and real-time model deployment. So I draw a hard line:

- **AI may drive:** boilerplate, glue code, documentation, error-message triage, API lookups.
- **I must drive:** the freezing/fine-tuning strategy, interpretation of training curves, the control design (cursor mapping and click semantics — decided by me 2026-07-14, AD-17…AD-20), and the real-time control tuning. AI is a *sounding board* for these, never the decision-maker.

## Where I plan to use AI (AI-assisted)

| Task | How AI helps | My oversight |
|---|---|---|
| Project/code scaffolding | Generate `Dataset`/`DataLoader`, transforms, argparse CLIs, plotting helpers | I review every line; I rename/restructure to match my mental model. |
| HaGRID annotation parsing | Draft the JSON-schema parser from a sample record | I verify against the real files and check landmark/bbox normalization. |
| Debugging | Interpret stack traces, CUDA/dtype/shape mismatches, dependency conflicts | I confirm the *root cause*, not just the patch. |
| `timm`/HF API usage | Look up how to swap heads, freeze params, set discriminative LRs | I decide *which* layers and *why*. |
| Input-simulation quirks | Explain why DirectX games ignore some synthetic clicks; suggest `pydirectinput` | I test registration in-game myself. |
| Documentation | Draft docstrings, READMEs, this phase plan, the demo script | I fact-check claims and keep numbers honest. |
| Code review | Spot bugs/simplifications in my diffs | I accept/reject each finding deliberately. |

## Where I will NOT lean on AI (handled manually)

- Choosing the **freezing schedule** and reading loss/accuracy curves to decide when to unfreeze.
- Diagnosing **under- vs. over-fitting** and choosing the response (augmentation, regularization, LR).
- Designing and tuning the **cursor mapping** (control box, smoothing) and the **click-FSM debounce/cooldown** logic — this is judgment built from playtesting, not from a prompt.
- Final **evaluation honesty**: I report real test-set numbers, including failures.

## Tools

- **Claude Code** (primary) — in-repo agent for scaffolding, debugging, docs, and code review, governed by [`CLAUDE.md`](../CLAUDE.md).
- **Claude (chat)** — design discussion and concept explanation (transfer learning, ONNX export).
- **MLflow** (added Week 3) — experiment store for the training runs. Claude wrote the logging/export glue (`scripts/mlflow_*.py`); **I own which metrics matter and the honesty of every number logged.**
- Possible: Copilot-style inline completion for small boilerplate.

## Honesty & attribution policy

- Every non-trivial AI contribution is logged weekly in [AI-usage.md](AI-usage.md): the task, the prompt/context that worked, and any output I had to correct.
- AI-generated code that I do not understand does not get merged. If I cannot explain it, I rewrite it until I can.
- Commit messages note where AI materially shaped an implementation.

## Weekly update cadence

At the end of each week I will: (1) append that week's entry to `AI-usage.md`, and (2) revise this file if my intended usage changed (e.g. I leaned on AI more/less than planned, or a new tool entered the workflow).

**Week-2 note (2026-07-12):** usage matched the plan — AI drove the EDA notebook, the overfit-single-batch validation, and the data-understanding report scaffolding; I retained the modeling calls those docs report (crop adoption, by-user split, ≥90% target). The living plan now lives in [implementation-plan.md](../legacy/phases/implementation-plan.md) *(moved to legacy with the 2026-07-14 pivot — current plan: [plan.md](../build/plan.md))*, and the Week-2 deliverables in [week2/](../class-related/week2/).

**Week-3 note (2026-07-19):** one **new tool** entered the workflow — **MLflow** (logged above). Usage still matched the plan: AI wrote the MLflow logging/export glue and drafted the [ML experimentation report](../class-related/week3/ml-experimentation-report.md), but every logged number is a real recorded result (transcribed from `DOCS/models/`), and the **model-selection call** (`palmfist_frozen` → Week-4 tuning) and the **train/serve honesty caveat** are mine. Full detail in [AI-usage.md → Week 3](AI-usage.md).
