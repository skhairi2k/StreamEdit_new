"""R40 score -- Pareto figures: R36's VP rho curve vs R35's 4 draft arms vs R40's 2
draft-free arms.

WHAT IS PLOTTED
---------------
The same five figures as ``r35_score.py --all`` (y = ``lpips_unedit_part``, lower =
better preservation; x = one editability measure), written as ``r40_*_vs_lpips.pdf``:

    r40_clip_vs_lpips.pdf          x = clip_similarity_target_image  (CLIP-T)
    r40_clipd_vs_lpips.pdf         x = clip_d_prompt                 (CLIP-D)
    r40_fiveacc_vs_lpips.pdf       x = five_acc_yes_no
    r40_fiveacc_mc_vs_lpips.pdf    x = five_acc_multi_choice
    r40_fiveacc_full_vs_lpips.pdf  x = five_acc                      (FiVE-Acc proper)

* R36 -- VP at rho in {2,3,4,6,8,10,20,50}: the reference curve, neutral gray.
* R35 -- the 4 draft-based arms: hue = divergence metric (DINO blue, LPIPS orange),
  marker = stage-1 regime (unblended filled circle, first2 hollow triangle), exactly as
  in R35's own figures.
* R40 -- the 2 draft-free arms (anchor map + flow warp): same hue as their metric,
  filled SQUARE with an ink ring, so the R35-vs-R40 pair for one metric shares a colour
  and differs by shape. Identified through the legend only (colour + shape): direct
  labels collided whenever the two R40 squares landed close together (test render).

The story these figures serve: the draft is necessary, because the draft-free
alternative is worse -- i.e. the R40 squares should sit up-left of their R35 points
(less edit at the same or worse background LPIPS).

AGGREGATION -- plain per-clip mean over the 419 clips, on all three sources
-----------------------------------------------------------------------------
R35 and R40 points are computed here from the per-clip CSVs with ``r35_score``'s own
loaders (``load_harness`` / ``load_clipd``, which hard-fail on any missing CSV, wrong row
count or clip-set mismatch); R36 points come from ``r36_rho_sweep.csv`` via
``r35_score.r36_points`` (full-bench rows only). Never ``{stem}_avg.csv``.

``r35_score.py`` is imported, not modified: R35 is analyzed and its figures must not
change under it. Its ``plot`` hard-codes R35's four combos, so this module draws its own.

Output: the five PDFs in --fig_dir and ``evaluation/csv/r40_arms.csv`` (one row per
plotted point: source R36|R35|R40, label, n_clips, the five x metrics, lpips_unedit_part).

Example
-------
    python evaluation/r40_score.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r35_score as R35  # noqa: E402  (also sets the Agg backend)
from r35_score import (  # noqa: E402
    ARM_COLOR, ARM_LABEL, FIVEACC_X, GRID, INK, MUTED, N_CLIPS, REF_COLOR, RHOS,
    SURFACE, Y_METRIC, clip_index, load_clipd, load_harness, r36_points, write_table,
)

import matplotlib.pyplot as plt  # noqa: E402

# Same order as r40_warp.sh / r40_stage3.sh / r40_eval.sh -- do not reorder one alone.
R40_COMBOS: Tuple[str, ...] = ("dino_warp", "lpips_warp")
R40_LABEL: Dict[str, str] = {"dino_warp": "DINO · anchor+warp",
                             "lpips_warp": "LPIPS · anchor+warp"}

# R35's figure set, re-pointed at r40_ file names.
FIGURES: Dict[str, Tuple[str, str, str]] = {
    k: (x, lab, fname.replace("r35_", "r40_", 1)) for k, (x, lab, fname) in R35.FIGURES.items()
}


def arm_points(csv_dir: Path, clips: Dict[Tuple[int, str], str], prefix: str,
               combos: Sequence[str], source: str) -> List[Dict[str, object]]:
    """One point per combo: plain mean over the 419 clips of each metric.

    Same per-metric sources as ``r35_score.r35_points``, parameterised by the stem prefix
    (``r35`` / ``r40``): ``{prefix}_{combo}`` (harness), ``{prefix}_fiveacc_{combo}``
    (FiVE-Acc), ``{prefix}_clip_directional.csv`` (CLIP-D).
    """
    pts = []
    for c in combos:
        stem = f"{prefix}_{c}"
        per = {
            Y_METRIC: load_harness(csv_dir, stem, Y_METRIC, clips),
            "clip_similarity_target_image":
                load_harness(csv_dir, stem, "clip_similarity_target_image", clips),
            **{m: load_harness(csv_dir, f"{prefix}_fiveacc_{c}", m, clips) for m in FIVEACC_X},
            "clip_d_prompt": load_clipd(csv_dir / f"{prefix}_clip_directional.csv", stem, clips),
        }
        pts.append({"source": source, "label": c, "n_clips": N_CLIPS,
                    **{m: float(np.mean(list(v.values()))) for m, v in per.items()}})
    return pts


def plot(key: str, r36: List[Dict[str, object]], r35: List[Dict[str, object]],
         r40: List[Dict[str, object]], fig_dir: Path) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # reference curve, as in r35_score.plot
    ax.plot([p[x_metric] for p in r36], [p[Y_METRIC] for p in r36], color=REF_COLOR,
            lw=2, marker="o", ms=5, zorder=2, label="R36 · VP, ρ sweep")
    for p, rho in zip(r36, RHOS):
        ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                    textcoords="offset points", fontsize=7, color=MUTED,
                    ha="right", va="bottom")

    # R35 draft arms: hue = metric, marker = stage-1 regime (R35's own encoding)
    for p in r35:
        arm, regime = p["label"].split("_", 1)
        col = ARM_COLOR[arm]
        filled = regime == "unblended"
        ax.scatter([p[x_metric]], [p[Y_METRIC]], s=90, zorder=3,
                   marker="o" if filled else "^",
                   facecolors=col if filled else SURFACE,
                   edgecolors=col if not filled else SURFACE,
                   linewidths=2, label=f"R35 · {ARM_LABEL[p['label']]} (draft)")

    # R40 draft-free arms: same hue as the metric, square with an ink ring
    for p in r40:
        col = ARM_COLOR[p["label"].split("_", 1)[0]]
        ax.scatter([p[x_metric]], [p[Y_METRIC]], s=100, zorder=4, marker="s",
                   facecolors=col, edgecolors=INK, linewidths=1.5,
                   label=f"R40 · {R40_LABEL[p['label']]}")

    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    ax.set_title(f"Draft-free (R40) vs draft (R35) divergence routing vs VP ρ sweep — "
                 f"full FiVE-Bench ({N_CLIPS} pairs, per-clip mean)", fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=7, frameon=False, loc="best")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / fname
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Build all five figures.")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+",
                    help="Build only these figures (default with --all: all five).")
    ap.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    ap.add_argument("--r36_sweep", type=Path, default=Path("evaluation/csv/r36_rho_sweep.csv"))
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    ap.add_argument("--out_csv", type=Path, default=Path("evaluation/csv/r40_arms.csv"))
    args = ap.parse_args(argv)
    if not args.all and not args.x:
        ap.error(f"pass --all or --x {{{','.join(FIGURES)}}}")
    keys = list(FIGURES) if args.all else args.x

    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r40_score] benchmark has {len(clips)} clips, expected {N_CLIPS}")

    r36 = r36_points(args.r36_sweep)
    r35 = arm_points(args.csv_dir, clips, "r35", R35.COMBOS, "R35")
    r40 = arm_points(args.csv_dir, clips, "r40", R40_COMBOS, "R40")

    write_table(args.out_csv, r36 + r35 + r40)
    print(f"[r40_score] {len(r36)} R36 + {len(r35)} R35 + {len(r40)} R40 points "
          f"-> {args.out_csv}")
    for p in r36 + r35 + r40:
        print(f"  {p['source']} {p['label']:<16} lpips={p[Y_METRIC]:.4f} "
              f"clipT={p['clip_similarity_target_image']:.3f} "
              f"clipD={p['clip_d_prompt']:+.4f} fa_yn={p['five_acc_yes_no']:.3f} "
              f"fa_mc={p['five_acc_multi_choice']:.3f} fa_full={p['five_acc']:.3f}")
    for k in keys:
        print(f"[r40_score] wrote {plot(k, r36, r35, r40, args.fig_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
