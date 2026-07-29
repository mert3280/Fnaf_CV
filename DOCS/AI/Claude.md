# Claude / AI Usage Plan

> **Status: final (2026-07-29).** This file began as a statement of **intent** and was revised weekly; it now records how AI was **actually** used across the five weeks. Where the plan and reality diverged, [§How it actually went](#how-it-actually-went-final-2026-07-29) says so rather than editing the original intent to match. The week-by-week record and the closing AI retrospective live in [AI-usage.md](AI-usage.md). The agent-context file that tells Claude its role and guardrails on this repo is at the repo root: [`CLAUDE.md`](../../CLAUDE.md).

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

- **Claude Code** (primary) — in-repo agent for scaffolding, debugging, docs, and code review, governed by [`CLAUDE.md`](../../CLAUDE.md).
- **Claude (chat)** — design discussion and concept explanation (transfer learning, ONNX export).
- **MLflow** (added Week 3; **Model Registry** added Week 4) — experiment store for the training runs and the version/alias record for the deployed model. Claude wrote the logging/export/registration glue (`scripts/mlflow_*.py`, `src/serve/pyfunc_model.py`); **I own which metrics matter and the honesty of every number logged.**
- **Optuna** (added Week 4) — the TPE sampler behind the hyper-parameter search in `src/tune.py`. Claude wrote the search harness; **I own the search space, the budget, and the reading of the results.**
- **Flask + ONNX Runtime** (added Week 4) — the inference endpoint and the parity-checked export (AD-11/AD-24). Serving plumbing is squarely in the "AI may drive" column.
- **Playwright** (added Week 4–5) — headless-browser verification of the eval dashboard, and then `scripts/capture_ui_screenshots.js`, which regenerates the Week-5 UI screenshots by driving the real app. This one changed how I think about AI verification: it means Claude can *check the UI it wrote actually works* instead of reporting that it should.
- Copilot-style inline completion for small boilerplate — **never adopted**; Claude Code in the repo covered it.

## Honesty & attribution policy

- Every non-trivial AI contribution is logged weekly in [AI-usage.md](AI-usage.md): the task, the prompt/context that worked, and any output I had to correct.
- AI-generated code that I do not understand does not get merged. If I cannot explain it, I rewrite it until I can.
- Commit messages note where AI materially shaped an implementation.

## How it actually went (final, 2026-07-29)

The plan above survived five weeks essentially intact, which surprised me. What changed is not *where* I used AI but *what I used it for* — and one line of the plan turned out to be worth more than all the rest.

**The line that earned its keep.** "I must drive: interpretation of training curves… AI is a *sounding board*, never the decision-maker." In practice that stopped being a matter of principle and became a working method: **make the new code reproduce the old number before believing a new one.** It caught two conclusions that would otherwise have gone into reports as findings — the ~5-point "tuning win" that was really timm's 2-class head init (AD-22), and my own confident "the model is overfit" diagnosis that was really a train/serve crop-geometry mismatch (AD-25). Both times the correction came from a control run I had asked for. Neither would have surfaced from a prompt that asked for a fix.

**The pattern that emerged and became standard.** Every time Claude was in a position to change a project default, it shipped the change as an **opt-in flag with the old default preserved** and recorded the adoption question as a **Proposed** decision record pending my call — AD-22 (head init) and AD-25 (crop pad) both went through that gate, and AD-25 I later accepted explicitly. This wasn't in the original plan. It's the mechanism that actually enforced it.

**Where my usage exceeded the plan.** The plan's "Documentation" row anticipated docstrings and READMEs. What actually happened is that AI wrote most of the project's *evidence-generation* layer: the tuning harness, the DAG engine, the serving artifact, the measurement scripts (`eval_crop_geometry.py`, `bench_pipeline.py`, `audit_split_leakage.py`), the diagram rendered from the DAG definitions, and the usability dashboard. That is far more than "boilerplate and glue," and I'm recording it as an honest expansion rather than pretending it fit the Week-0 table. What kept it inside the spirit of the plan is that none of it makes a modeling decision — it produces numbers, and reading them stayed mine.

**Where the plan was too optimistic.** "I confirm the *root cause*, not just the patch" held for ML bugs but not for infrastructure ones: several root causes (thread-pool contention starving MediaPipe, 67 ms of camera backlog, a 0 ms click pulse a DirectX game can't see, a missing `sim.tick()` leaving the mouse button stuck down, `[hidden]` losing to an author `display` rule) were found by Claude measuring, and I confirmed them by reading the evidence rather than by independently deriving them. That's a weaker form of ownership than the plan implies, and it's the honest version.

**What I would put in this file if I were starting over.** One extra row: *"Before you believe a number from a new harness, make the harness reproduce the old number."* It is the only rule here that changed an outcome twice.

## Weekly update cadence

At the end of each week I will: (1) append that week's entry to `AI-usage.md`, and (2) revise this file if my intended usage changed (e.g. I leaned on AI more/less than planned, or a new tool entered the workflow).

**Week-5 note (2026-07-29) — final.** One **new tool** entered the workflow: **Playwright** (logged above), used first to verify the dashboard overhaul against a real running server and then to make the Week-5 UI screenshots a regenerable artifact rather than a manual chore. Usage matched the plan, and the boundary held on the two calls that came up: the crop-tightness default (AD-25) moved **only** at my explicit direction, and the MVP verdict — that Night 1 completed hands-free but is the *least-measured* result in the project, on a checkpoint that is not the registered champion — is my reading, not a drafted one. The one thing AI did this week that I hadn't asked for and kept: when told to "make sure all the deliverables are met," it stopped and asked where Step 5 actually landed instead of inferring it from a repo that still said "built, not run against the game." Closing AI retrospective: [AI-usage.md → Final retrospective](AI-usage.md#final-retrospective--ai-usage-across-the-project).

**Week-2 note (2026-07-12):** usage matched the plan — AI drove the EDA notebook, the overfit-single-batch validation, and the data-understanding report scaffolding; I retained the modeling calls those docs report (crop adoption, by-user split, ≥90% target). The living plan now lives in [implementation-plan.md](../legacy/phases/implementation-plan.md) *(moved to legacy with the 2026-07-14 pivot — current plan: [plan.md](../build/plan.md))*, and the Week-2 deliverables in [week2/](../class-related/week2/).

**Week-4 note (2026-07-26):** three **new tools** entered the workflow — **Optuna**, the **MLflow Model Registry**, and **Flask/ONNX Runtime** for serving (all logged above). Usage matched the plan, and the "I must drive" line proved its worth: the tuner's first results beat the deployed baseline by ~5 points, and because I insisted on a **control run of the old recipe inside the new harness**, that gain turned out to be timm's 2-class **head initialization**, not hyper-parameter search. I own that reading — the report presents it as a three-rung ladder (baseline → init fixed → search), and whether the fixed init becomes the project default is recorded as **[AD-22 Proposed](../build/architecture-and-decisions.md#ad-22--head-initialization-for-a-2-class-head--proposed-2026-07-26--teds-call), pending my decision**, not silently flipped. Standing rule going forward: **any new harness must reproduce the previous number before I believe a new one.** Full detail in [AI-usage.md → Week 4](AI-usage.md).

**Week-3 note (2026-07-19):** one **new tool** entered the workflow — **MLflow** (logged above). Usage still matched the plan: AI wrote the MLflow logging/export glue and drafted the [ML experimentation report](../class-related/week3/ml-experimentation-report.md), but every logged number is a real recorded result (transcribed from `DOCS/models/`), and the **model-selection call** (`palmfist_frozen` → Week-4 tuning) and the **train/serve honesty caveat** are mine. Full detail in [AI-usage.md → Week 3](AI-usage.md).
