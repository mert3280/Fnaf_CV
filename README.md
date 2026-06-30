# Fnaf_CV — Gesture-Controlled Five Nights at Freddy's

Play the original **Five Nights at Freddy's** hands-free. A webcam watches your hand, a
transfer-learned image classifier recognizes **static hand gestures**, and the system
translates them into the mouse inputs FNAF expects — closing doors, flicking lights, and
raising the camera monitor.

- **ML approach:** transfer learning with a pretrained `timm`/Hugging Face backbone + a custom
  classification head, trained on the **HaGRID** gesture dataset (frozen → progressively unfrozen).
- **Control:** gestures → simulated mouse input driving the real Steam game.
- **Scope:** solo · static poses only · FNAF 1 only.

## Status
📋 **Planning / proposal stage.** Implementation begins in Phase 1.

## Documentation
| Doc | Purpose |
|---|---|
| [DOCS/proposal.md](DOCS/proposal.md) | Project proposal (7 required sections) |
| [DOCS/architecture-and-decisions.md](DOCS/architecture-and-decisions.md) | System architecture + decision records (ADRs) |
| [DOCS/schedule.md](DOCS/schedule.md) | 5-week plan + risk register |
| [DOCS/Claude.md](DOCS/Claude.md) | AI-usage plan (intent) |
| [DOCS/AI-usage.md](DOCS/AI-usage.md) | Weekly AI-usage log |
| [DOCS/phases/](DOCS/phases/) | Per-phase implementation plans (1–5) |
| [CLAUDE.md](CLAUDE.md) | Agent-context file for Claude Code |

## Phases at a glance
1. **Setup & data validation** — env, HaGRID parsing, `DataLoader`, EDA, gesture→action map.
2. **Baseline model** — pretrained backbone + custom head, frozen, first metrics.
3. **Training & fine-tuning** — progressive unfreezing, augmentation, test-set eval, export.
4. **Real-time & control** — webcam loop, debouncing, simulated input into FNAF.
5. **Integration & demo** — robustness, full hands-free Night 1, docs, demo video.

## Dataset
[HaGRID](https://github.com/hukenovs/hagrid) — 18 static hand-gesture classes. `DATA/` contains
**annotations only**; images are downloaded separately (see
[DOCS/phases/phase-1-setup-and-data.md](DOCS/phases/phase-1-setup-and-data.md)).
