# AI Usage Log (Weekly)

> A living, weekly record of how AI tools were actually used on this project. Intent and policy live in [Claude.md](Claude.md). For each week I record: **what tasks used AI**, **prompts/context that worked well**, and **cases where AI output needed correction**.

Template for each entry:

```
## Week N — <dates>
### Tasks AI assisted with
- ...
### Prompts / context that worked well
- ...
### AI output that needed correction
- ...
### Net assessment
- Did AI usage match the plan in Claude.md? More / less / as planned, and why.
```

---

## Week 0 — Proposal & Planning (2026-06-30)

### Tasks AI assisted with
- **Research:** Used Claude Code (web search) to confirm FNAF's control scheme (mouse-driven: doors, lights, camera panel) and to compare gesture-recognition architectures (custom image classifier vs. MediaPipe landmarks vs. hybrid).
- **Dataset identification:** Claude inspected the local `DATA/` folder and correctly identified it as the **HaGRID** dataset (18 static classes; JSON annotations with bboxes + 21 landmarks; `ann_subsample`/`ann_train_val`/`ann_test` splits) and flagged that the large JSONs are **annotations only** — images are a separate download.
- **Planning docs:** Claude drafted this documentation set — `proposal.md`, `schedule.md`, `Claude.md`, this file, the five phase files, and the root `CLAUDE.md` agent file — from my answers to four scoping questions (solo; transfer-learning with custom head + freezing; drive the real game via simulated input; static poses only).

### Prompts / context that worked well
- Giving Claude the **assignment rubric verbatim** plus the timm/HF tutorial link produced docs structured to the exact required sections.
- Asking Claude to **ask clarifying questions before planning** surfaced the four decisions that actually shaped the plan (team size, ML approach, control target, gesture type) instead of guessing.
- Letting Claude **inspect the `DATA/` folder directly** rather than describing it — it identified HaGRID and the annotations-vs-images gap on its own.

### AI output that needed correction
- Claude initially had to be steered on the **`CLAUDE.md` vs `Claude.md` Windows filename collision** (case-insensitive filesystem) — resolved by placing the deliverable `Claude.md` in `DOCS/` and the agent file `CLAUDE.md` at the repo root.
- *(To be extended as I review the generated docs and adjust scope/wording to my own voice.)*

### Net assessment
- As planned for a planning week: AI drove research synthesis and document scaffolding; I made every scoping decision. No core ML learning objective was delegated.

---

<!-- Append Week 1..5 entries below as the project progresses. -->
