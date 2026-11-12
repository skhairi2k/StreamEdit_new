"""Main-paper trade-off panels: preservation (y) vs. alignment (x) for one editor, uniform-control sweeps vs. SNaC.

One small PDF per editor (sized for half a CVPR column), read from the R57 suite's per-clip tables through
``paper_tables.load`` (plain per-clip means over the 415 FiVE-Bench-Correct pairs, cross-checked against _overall.csv):

  * equal-step uniform sweep (solid, the fair reference: same number of solver steps as SNaC, draft included);
  * native uniform sweep (dashed grey, the editor's own step count);
  * the editor's official setting (open ring on the native sweep);
  * SNaC (orange star, the colour of the qualitative figures' "ours" row).

Axes: x = FiVE-Acc (x100), y = background SSIM (x100); both higher-is-better, so up-right is better.

usage: python -I evaluation/paper_fig_tradeoff.py --editor se_ll --out paper/figures/tradeoff_se_ll.pdf
       python -I evaluation/paper_fig_tradeoff.py --editor we    --out paper/figures/tradeoff_we.pdf   (needs R71)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_tables as pt  # noqa: E402

BLUE = "#2a78d6"        # equal-step sweep (reference palette, slot 1)
GREY = "#8a8985"        # native sweep (secondary, dashed)
ORANGE = "#e67814"      # SNaC = \definecolor{qours}{RGB}{230,120,20} in sec/4_experiments.tex
INK, MUTED = "#0b0b0b", "#52514e"

X_KEY, X_SCALE, X_LABEL = "five_acc", 1e2, r"FiVE-Acc $\uparrow$"
Y_KEY, Y_SCALE, Y_LABEL = "ssim_unedit_part", 1e2, r"Background SSIM ($\times 10^2$) $\uparrow$"


@dataclass(frozen=True)
class Editor:
    title: str
    knob: str                              # LaTeX name of the global control
    equal: Tuple[Tuple[str, str], ...]     # (knob value, stem), knob order
    equal_label: str
    native: Tuple[Tuple[str, str], ...]
    native_label: str
    official: str                          # stem of the official setting (a native point)
    ours: str
    ours_label: str
    annotate: Dict[str, Tuple[float, float, str]]   # knob value -> label offset (points) and alignment, equal-step sweep


RHOS = ("2", "3", "4", "6", "8", "10", "20", "50")
EDITORS: Dict[str, Editor] = {
    "se_ll": Editor(
        "StreamEdit (LongLive)", r"\rho",
        tuple((r, f"r61_r61_ll_rho{r}_n20") for r in RHOS), "Uniform, 20 steps",
        tuple((r, f"r61_r61_ll_rho{r}_n15") for r in RHOS), "Uniform, 15 steps",
        "r61_r61_ll_rho2_n15", "r61_r61_max_ll", "SE + SNaC",
        {"2": (4, 5, "left"), "6": (-4, -4, "right"), "50": (-5, -1, "right")}),
    "we": Editor(
        "Wan-Edit", "s",
        tuple((s, f"r71_wan_matched63_s{s.replace('.', 'p')}_correct")
              for s in ("0.996", "0.978", "0.95", "0.92", "0.85", "0.5")), "Uniform, 63 steps",
        tuple((s, f"r58_wan_native_s{s.replace('.', 'p')}_correct")
              for s in ("0.978", "0.95", "0.92", "0.85", "0.75", "0.6", "0.4")), "Uniform, 50 steps",
        "r58_wan_native_s0p92_correct", "r71_wan_max_s1_correct", "WE + SNaC",
        {"0.996": (4, 5, "left"), "0.92": (-4, -4, "right"), "0.5": (-4, -1, "right")}),
}


def xy(stem: str) -> Tuple[float, float]:
    d = pt.load(stem)
    if d is None or d.get(X_KEY) is None or d.get(Y_KEY) is None:
        raise SystemExit(f"[paper_fig_tradeoff] missing table or value for {stem} (suite_{stem}.csv)")
    return d[X_KEY] * X_SCALE, d[Y_KEY] * Y_SCALE


def curve(points: Tuple[Tuple[str, str], ...]) -> List[Tuple[str, float, float]]:
    return [(k, *xy(s)) for k, s in points]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 5.8, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "axes.edgecolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.labelcolor": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def draw(ed: Editor, out: Path, width: float, height: float) -> None:
    eq, nat = curve(ed.equal), curve(ed.native)
    ox, oy = xy(ed.official)
    ux, uy = xy(ed.ours)
    style()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, lw=0.3, color="#e4e3df", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([p[1] for p in nat], [p[2] for p in nat], ls=(0, (3, 1.5)), lw=1.0, color=GREY, marker="o", ms=2.2,
            mfc=GREY, mec="white", mew=0.3, label=ed.native_label, zorder=2)
    ax.plot([p[1] for p in eq], [p[2] for p in eq], lw=1.2, color=BLUE, marker="o", ms=2.8, mfc=BLUE, mec="white",
            mew=0.4, label=ed.equal_label, zorder=3)
    ax.plot([ox], [oy], ls="none", marker="o", ms=5.5, mfc="none", mec=INK, mew=0.7, label="Official",
            zorder=4)
    ax.plot([ux], [uy], ls="none", marker="*", ms=8.5, color=ORANGE, mec="white", mew=0.5, label=ed.ours_label,
            zorder=5)
    for k, x, y in eq:   # selective direct labels on the reference sweep
        if k in ed.annotate:
            dx, dy, ha = ed.annotate[k]
            ax.annotate(f"${ed.knob}{{=}}{k}$", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.5,
                        color=MUTED, ha=ha, va="center")
    ax.set_xlabel(X_LABEL, labelpad=1.5)
    ax.set_ylabel(Y_LABEL, labelpad=1.5)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
    ax.margins(x=0.08, y=0.08)
    fig.tight_layout(pad=0.15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[paper_fig_tradeoff] wrote {out}: official ({ox:.2f}, {oy:.2f}), ours ({ux:.2f}, {uy:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--editor", choices=sorted(EDITORS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=float, default=1.62, help="inches (half a CVPR column = 1.62)")
    ap.add_argument("--height", type=float, default=1.55, help="inches")
    a = ap.parse_args()
    draw(EDITORS[a.editor], a.out, a.width, a.height)


if __name__ == "__main__":
    main()
