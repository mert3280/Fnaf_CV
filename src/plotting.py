"""Shared figure style for the committed report figures.

One place for the palette + rcParams so every PNG in DOCS/ reads as one system
(and so a color choice is never re-invented per script).

Palette roles (validated categorical slots, light surface #fcfcfb):
  series 1 blue #2a78d6, series 2 orange #eb6834 -- the pair used for the
  two-series charts; all-pairs CVD dE 24.7 / normal-vision 33.6, both well clear
  of the 8 / 15 floors, so the two lines stay distinguishable in grayscale print
  and for colorblind readers. Sequential magnitude (confusion matrices) uses the
  single blue ramp light->dark; reference lines are chrome (muted ink, dashed),
  never a series color.

Figures are rendered on an explicit light surface rather than transparent, so
they stay legible in GitHub's dark mode.
"""

from __future__ import annotations

from pathlib import Path

PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    # blue ramp, light -> dark (sequential magnitude)
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "good": "#0ca30c",
    "critical": "#d03b3b",
}


def apply_style() -> None:
    """Set matplotlib rcParams: recessive chrome, thin marks, system sans."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink_secondary"],
        "axes.labelsize": 9,
        "axes.edgecolor": PALETTE["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["ink_muted"],
        "ytick.color": PALETTE["ink_muted"],
        "xtick.labelcolor": PALETTE["ink_secondary"],
        "ytick.labelcolor": PALETTE["ink_secondary"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })


def save(fig, path: str | Path) -> Path:
    """Save with a tight box and report where it went."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    print(f"  wrote {path}")
    return path


def seq_color(value: float, vmax: float) -> str:
    """Pick a step off the blue ramp for `value` in [0, vmax] (magnitude)."""
    if vmax <= 0:
        return PALETTE["seq"][0]
    steps = PALETTE["seq"]
    i = min(len(steps) - 1, int(round((value / vmax) * (len(steps) - 1))))
    return steps[i]


def text_on(color: str) -> str:
    """Readable ink for a label drawn on top of `color` (a seq-ramp step)."""
    return "#ffffff" if color in PALETTE["seq"][4:] else PALETTE["ink"]
