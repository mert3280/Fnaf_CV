# Phase 5 — Integration, Robustness & Demo (Week 5)

**Phase goal:** Make the system **survivable, robust, and presentable** — complete a full hands-free Night 1 of FNAF, harden against real-world conditions, and ship documentation + a demo.

---

## 1. Calibration & session setup
- [ ] A short **calibration routine** at startup: confirm webcam, lighting, region-of-interest, and per-gesture confidence thresholds for the current environment.
- [ ] Persist calibration to `configs/session.yaml` so a known-good setup is repeatable.
- [ ] Confirm FNAF window position matches `fnaf_layout.yaml` (or re-detect).

## 2. Robustness passes
The model was trained on HaGRID; the living room is not HaGRID. Stress-test and fix:
- [ ] **Lighting:** bright, dim, side-lit. If recognition degrades, lean on the Phase 3 color/brightness augmentation; consider a small **fine-tune on self-captured frames** of my own hands in my real environment (closes the train/serve gap).
- [ ] **Background clutter / other people** in frame.
- [ ] **Idle handling:** resting hands / out-of-frame should map to no action (`mute` / low-confidence → idle).
- [ ] **Latency under load:** ensure the game running alongside inference still hits the FPS budget; downscale input or backbone if needed.
- [ ] Tune debounce/cooldown so doors don't flap and lights respond crisply.

## 3. Full playthrough (the real evaluation)
- [ ] Complete **Night 1 of FNAF entirely hands-free.** This is the project's headline success metric.
- [ ] Log failure incidents (missed gestures, false toggles) and iterate on thresholds/mapping.
- [ ] Stretch: attempt later nights / a quick second player to test generalization.

## 4. Documentation & demo
- [ ] `README.md`: what it is, setup, how to run, the gesture map, known limitations.
- [ ] **Architecture diagram** (data → model → real-time loop → input → game).
- [ ] `DOCS/results.md` finalized: model metrics + real-world reliability notes (gesture-level hit rate during play).
- [ ] **Known limitations & future work** (dynamic gestures, more games, edge deployment).
- [ ] Record a **demo video** of a hands-free night.
- [ ] Final `AI-usage.md` weekly entry; update `Claude.md` if usage drifted from plan.
- [ ] Branch → PR → merge to `main` with complete `DOCS/`.

## 5. Deliverables
- Hands-free Night 1 completion (recorded).
- Calibration + robustness improvements.
- Complete documentation set + demo video + final merge.

## 6. Definition of done
A full Night 1 completed using only hand gestures, captured on video; documentation complete and merged to `main`.

## 7. Behind-schedule signal
Cannot complete a full hands-free night, or recurring misclassifications make play infeasible and remain unresolved at week's end.

## 8. Project retrospective (to write here at the end)
- What worked, what I'd do differently, which learning goals (transfer learning; offline→live deployment) I actually hit, and the honest reliability numbers from real play.
