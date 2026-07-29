# Week 5 — UI, documentation & delivery

Deliverables for [M5A1](M5A1.md). Everything below is committed on the `week5`
branch and merged to `main`.

## In this folder

| M5A1 section | File | What it covers |
|---|---|---|
| §1 Business-facing UI | [ui-screenshots/](ui-screenshots/) | 12 screenshots of the live front-end, regenerable via `scripts/capture_ui_screenshots.js` |
| §1 Business-facing UI | [ui-walkthrough.md](ui-walkthrough.md) | Screen-by-screen walkthrough: inputs/outputs/controls, how the UI reaches the model, how freshness is surfaced, design decisions for non-technical users |
| §2 How to use it | [how-to-use.md](how-to-use.md) | Non-technical user guide — what it is, step-by-step use, how to read the outputs, known limitations |
| §5 Retrospective | [retrospective.md](retrospective.md) | Proudest technical work, biggest challenge, what five more weeks would buy, and the connection back to the Week-1 takeaway |
| — | [audience-notes-week5.md](audience-notes-week5.md) | Peer-presentation notes + self-reflection |

## §3 / §4 — files finalized in place

The assignment asks for four root-docs files by name. Three of them predate the
assignment in this repo under different names and are cross-linked from dozens of
places, so they were **finalized in place** rather than renamed:

| Assignment name | This repo | Week-5 change |
|---|---|---|
| `README.md` | [/README.md](../../../README.md) | **Rewritten.** Was still at "planning / proposal stage" with dead doc links. Now: problem statement, mermaid architecture diagram, component table, versioned tech stack, developer setup, run commands for pipeline / model / endpoint / UI, and an explicit "things worth knowing before you trust a number here" section |
| `implementation-plan.md` | [DOCS/build/plan.md](../../build/plan.md) | **Week-5 final status note added** — MVP verdict against the three proposal criteria, the measured-vs-played gap, and seven descoped items with reasons |
| `claude.md` | [DOCS/AI/Claude.md](../../AI/Claude.md) | **Finalized** — reflects how AI was actually used across five weeks, not the Week-0 plan |
| `ai-usage-log.md` | [DOCS/AI/AI-usage.md](../../AI/AI-usage.md) | **Week-5 entry completed + final retrospective section added** |

> `CLAUDE.md` at the repo root is a *different* file — the agent-context file that
> governs how Claude Code behaves in this repo. On a case-insensitive Windows
> filesystem it cannot share a name with the `Claude.md` deliverable, which is why
> the two live in different folders.

## Headline result

**Night 1 of FNAF was completed hands-free** on
`fistvsrest_frozen_mnv3_large` — the project's headline success criterion. It was
not a clean run and was not instrumented or recorded, so it is reported as a
qualitative result throughout. Notably, the checkpoint that beat the game is
**not** the registered `champion` that scores 1.0000 on the offline test set, and
that model has still never been validated live. Full accounting:
[plan.md → Week-5 final status](../../build/plan.md#week-5-final-status-2026-07-29--mvp-met-honestly-scoped).
