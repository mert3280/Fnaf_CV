"""Named click-FSM tuning presets for the Strategy-3 control layer.

A "strategy variant" here is **only** the click-FSM tuning laid on top of a
binary click checkpoint -- the model, cursor mapping, crop contract, grace
window, and cooldown are all unchanged. 3.2 and 3.2.1 run the **same**
fist-vs-rest checkpoint (AD-21); they differ solely in how eagerly the FSM
commits a click:

    3.2    conf >= 0.70, K = 3   -- the AD-20 starting points (steady)
    3.2.1  conf >= 0.80, K = 2   -- stricter per-frame gate, faster confirm (snappy)

3.2.1 tightens the per-frame confidence bar (0.70 -> 0.80) while loosening the
K-frame confirm (3 -> 2): fewer marginal poses count, but a confirmed pose fires
one frame sooner. The two changes partly offset on false clicks -- see
DOCS/build/strategies/3-cursor-and-click/03.2.1-snappier-fsm.md and AD-20.

This module is the single source of truth so `src.control.play` and the eval
dashboard resolve the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

# The AD-20 starting points -- what the FSM falls back to with no preset chosen
# and no explicit override. Kept here so there's exactly one definition.
BASE_FSM_CONF = 0.70
BASE_FSM_K = 3


@dataclass(frozen=True)
class FSMPreset:
    """One named (conf, K) tuning of the click FSM. Only the two knobs that
    *define* the variant live here; grace and cooldown stay at their AD-20
    defaults for every preset."""

    id: str
    label: str      # shown in the dashboard Overview + strategy picker
    blurb: str      # one-line description for the UI / --help
    fsm_conf: float
    fsm_k: int


FSM_PRESETS: dict[str, FSMPreset] = {
    "3.2": FSMPreset(
        id="3.2",
        label="3.2 · fist-vs-rest",
        blurb="Steady: confidence ≥ 0.70, 3-frame confirm (AD-20 starting points).",
        fsm_conf=BASE_FSM_CONF,
        fsm_k=BASE_FSM_K,
    ),
    "3.2.1": FSMPreset(
        id="3.2.1",
        label="3.2.1 · fist-vs-rest (snappy)",
        blurb="Snappy: confidence ≥ 0.80, 2-frame confirm — faster clicks, stricter gate.",
        fsm_conf=0.80,
        fsm_k=2,
    ),
}

DEFAULT_STRATEGY = "3.2"


def get_preset(strategy_id: str) -> FSMPreset:
    try:
        return FSM_PRESETS[strategy_id]
    except KeyError:
        valid = ", ".join(FSM_PRESETS)
        raise KeyError(f"unknown strategy {strategy_id!r}; choose one of: {valid}") from None


def resolve_fsm(strategy_id: str | None,
                conf: float | None = None,
                k: int | None = None) -> tuple[float, int]:
    """Resolve the FSM (conf, K) for a run.

    Precedence: an explicit `conf`/`k` wins over the preset, and the preset wins
    over the base AD-20 defaults. So `--strategy 3.2.1` sets 0.80/2, but a
    `--fsm-k 4` alongside it still forces K=4.
    """
    preset = get_preset(strategy_id) if strategy_id else None
    out_conf = conf if conf is not None else (preset.fsm_conf if preset else BASE_FSM_CONF)
    out_k = k if k is not None else (preset.fsm_k if preset else BASE_FSM_K)
    return out_conf, out_k
