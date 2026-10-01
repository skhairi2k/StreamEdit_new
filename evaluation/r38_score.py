"""R38 score -- three Pareto figures: does a 5-step stage-1 draft route as well as R35's
15-step one?

WHAT IS PLOTTED
---------------
Preservation on y (``lpips_unedit_part``, lower = better) against one editability measure
on x:

    r38_clip_vs_lpips.pdf     x = clip_similarity_target_image  (CLIP-T)
    r38_clipd_vs_lpips.pdf    x = clip_d_prompt                 (CLIP-D, R33's axis of record)
    r38_fiveacc_vs_lpips.pdf  x = five_acc_yes_no               (FiVE-Acc, yes/no)

* R36 -- StreamEdit + VP at rho in {2,3,4,6,8,10,20,50}: the gray reference CURVE, as in R35.
* For each (divergence metric, stage-1 regime), a thin line from its N=15 point to its N=5
  point, each point labelled with its N:
    - N=15 = R35's own arms (r35_* CSVs) -- R35's ``first2`` IS R38's ``tgt0.9`` at 15 steps
      (smoke gate G1, job 1016420), so it is the N=15 end of the ``anchored`` line;
    - N=5  = R38's arms (r38_* CSVs).
  Hue = divergence metric (DINO blue, LPIPS orange), marker = regime (unblended filled
  circle, anchored hollow triangle) -- the same encoding as R35's figures.

AGGREGATION -- plain mean over the 419 clips on every side, from the per-clip CSVs, NEVER
``{stem}_avg.csv`` (a mean of six per-type means). R36's sweep CSV averages the same way.
The loaders are R35's own (imported from ``r35_score``, not copied), so both sides are read
by literally the same code; their error messages therefore carry an ``[r35_score]`` prefix.

DROPPED COMBOS. If gate-lowN dropped R38 combos, pass them with ``--dropped``: their N=5
point is omitted (the N=15 point stays, unlinked). Any OTHER missing input hard-fails.

Output: the three PDFs in --fig_dir and ``r38_arms.csv`` (one row per plotted point).

Example
-------
    python evaluation/r38_score.py --all
    python evaluation/r38_score.py --all --dropped dino_n5_tgt09 lpips_n5_tgt09
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r35_score import (  # noqa: E402
    ARM_COLOR, GRID, INK, MUTED, N_CLIPS, REF_COLOR, RHOS, SURFACE, Y_METRIC,
    clip_index, load_clipd, load_harness, r36_points,
)

# (source, stem combo, N, metric, regime). Order = R38's SLURM order within each N.
POINTS: Tuple[Tuple[str, str, int, str, str], ...] = (
    ("R35", "dino_unblended",     15, "dino",  "unblended"),
    ("R35", "dino_first2",        15, "dino",  "anchored"),
    ("R35", "lpips_unblended",    15, "lpips", "unblended"),
    ("R35", "lpips_first2",       15, "lpips", "anchored"),
    ("R38", "dino_n5_unblended",   5, "dino",  "unblended"),
    ("R38", "dino_n5_tgt09",       5, "dino",  "anchored"),
    ("R38", "lpips_n5_unblended",  5, "lpips", "unblended"),
    ("R38", "lpips_n5_tgt09",      5, "lpips", "anchored"),
)
R38_COMBOS = tuple(p[1] for p in POINTS if p[0] == "R38")

X_PLOTTED: Tuple[str, ...] = ("clip_similarity_target_image", "clip_d_prompt", "five_acc_yes_no")
FIGURES: Dict[str, Tuple[str, str, str]] = {
    "clip":    ("clip_similarity_target_image", "CLIP-T (target image)  → more edit",
                "r38_clip_vs_lpips.pdf"),
    "clipd":   ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit",
                "r38_clipd_vs_lpips.pdf"),
    "fiveacc": ("five_acc_yes_no", "FiVE-Acc (yes/no)  → more edit",
                "r38_fiveacc_vs_lpips.pdf"),
}
LINE_LABEL = {("dino", "unblended"): "DINO · unblended",
              ("dino", "anchored"): "DINO · anchored (first2@15 = tgt0.9@5)",
              ("lpips", "unblended"): "LPIPS · unblended",
              ("lpips", "anchored"): "LPIPS · anchored (first2@15 = tgt0.9@5)"}


def arm_points(csv_dir: Path, clips: Dict[Tuple[int, str], str],
               dropped: Sequence[str]) -> List[Dict[str, object]]:
    """One point per (non-dropped) combo: plain mean over 419 clips of each needed metric."""
    pts = []
    for source, combo, n, metric, regime in POINTS:
        if combo in dropped:
            print(f"[r38_score] skip {combo} (dropped at gate-lowN)")
            continue
        pre = source.lower()                       # r35 / r38: stem + clip-D file prefix
        per = {
            Y_METRIC: load_harness(csv_dir, f"{pre}_{combo}", Y_METRIC, clips),
            "clip_similarity_target_image":
                load_harness(csv_dir, f"{pre}_{combo}", "clip_similarity_target_image", clips),
            "five_acc_yes_no": load_harness(csv_dir, f"{pre}_fiveacc_{combo}", "five_acc_yes_no", clips),
            "clip_d_prompt": load_clipd(csv_dir / f"{pre}_clip_directional.csv", f"{pre}_{combo}", clips),
        }
        pts.append({"source": source, "combo": combo, "N": n, "metric": metric, "regime": regime,
                    "n_clips": N_CLIPS,
                    **{m: float(np.mean(list(v.values()))) for m, v in per.items()}})
    return pts


def plot(key: str, r36: List[Dict[str, object]], arms: List[Dict[str, object]],
         fig_dir: Path) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # reference curve -- identical to R35's figures
    ax.plot([p[x_metric] for p in r36], [p[Y_METRIC] for p in r36], color=REF_COLOR, lw=2,
            marker="o", ms=5, zorder=2, label="R36 · VP, ρ sweep")
    for p, rho in zip(r36, RHOS):
        ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                    textcoords="offset points", fontsize=7, color=MUTED, ha="right", va="bottom")

    # one arrow per (metric, regime), N=15 -> N=5. N is encoded by OPACITY, not per-point
    # text: R35's four N=15 points sit within ~0.004 of each other, so "N=15" labels
    # collide (seen on the fixture render). Faded = N=15 (R35), solid = N=5 (R38).
    for (metric, regime), label in LINE_LABEL.items():
        by_n = {p["N"]: p for p in arms if p["metric"] == metric and p["regime"] == regime}
        if not by_n:
            continue
        col = ARM_COLOR[metric]
        filled = regime == "unblended"
        if 15 in by_n and 5 in by_n:
            a, b = by_n[15], by_n[5]
            ax.annotate("", xy=(b[x_metric], b[Y_METRIC]), xytext=(a[x_metric], a[Y_METRIC]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1, alpha=0.7,
                                        shrinkA=6, shrinkB=7), zorder=2.5)
        for n, alpha in ((15, 0.4), (5, 1.0)):
            if n not in by_n:
                continue
            p = by_n[n]
            ax.scatter([p[x_metric]], [p[Y_METRIC]], s=80, zorder=3, alpha=alpha,
                       marker="o" if filled else "^",
                       facecolors=col if filled else SURFACE, edgecolors=SURFACE if filled else col,
                       linewidths=2, label=label if n == 5 or 5 not in by_n else None)
    # legend-only entry explaining the opacity / arrow encoding
    ax.plot([], [], color=MUTED, lw=1, marker=">", ms=4,
            label="faded N=15 (R35) → solid N=5 (R38)")

    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    ax.set_title(f"R38 draft steps N=15 → 5 vs R36 VP ρ sweep — full FiVE-Bench "
                 f"({N_CLIPS} pairs, per-clip mean)", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="best")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / fname
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def write_table(path: Path, r36: List[Dict[str, object]], arms: List[Dict[str, object]]) -> None:
    cols = ["source", "combo", "N", "regime", "metric", "n_clips", *X_PLOTTED, Y_METRIC]
    rows = [{"source": "R36", "combo": p["label"], "N": 15, "regime": "", "metric": "",
             "n_clips": p["n_clips"], **{m: p[m] for m in (*X_PLOTTED, Y_METRIC)}} for p in r36]
    rows += [{k: p[k] for k in cols} for p in arms]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Build all three figures.")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+", help="Build only these figures.")
    ap.add_argument("--dropped", nargs="*", default=[], choices=R38_COMBOS,
                    help="R38 combos dropped at gate-lowN; their N=5 point is omitted.")
    ap.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    ap.add_argument("--r36_sweep", type=Path, default=Path("evaluation/csv/r36_rho_sweep.csv"))
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    ap.add_argument("--out_csv", type=Path, default=Path("evaluation/csv/r38_arms.csv"))
    args = ap.parse_args(argv)
    if not args.all and not args.x:
        ap.error(f"pass --all or --x {{{','.join(FIGURES)}}}")
    keys = list(FIGURES) if args.all else args.x

    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r38_score] benchmark has {len(clips)} clips, expected {N_CLIPS}")

    r36 = r36_points(args.r36_sweep)                  # fail fast on the cross-task input
    arms = arm_points(args.csv_dir, clips, args.dropped)

    write_table(args.out_csv, r36, arms)
    print(f"[r38_score] {len(r36)} R36 + {len(arms)} arm points -> {args.out_csv}")
    for p in arms:
        print(f"  {p['source']} N={p['N']:<2} {p['combo']:<20} lpips={p[Y_METRIC]:.4f} "
              f"clipT={p['clip_similarity_target_image']:.3f} "
              f"clipD={p['clip_d_prompt']:+.4f} fa_yn={p['five_acc_yes_no']:.3f}")
    for k in keys:
        print(f"[r38_score] wrote {plot(k, r36, arms, args.fig_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
