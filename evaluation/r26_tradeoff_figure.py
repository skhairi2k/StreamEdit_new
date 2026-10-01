#!/usr/bin/env python
"""R26 CLIP vs background-preservation trade-off across the full (tau_bg, tau_fg) grid.

Up to FOUR figures (`--which`), each with the SAME three series:

  BLUE curve   -- the degenerate diagonal tau_bg == tau_fg, i.e. the plain scalar Eq. 4
                  baseline swept over blend_power in {2,3,4,6,8,10,20,50}. No spatial
                  signal; the reference a spatial tau has to beat.
  RED cloud    -- every (tau_bg, tau_fg) with tau_bg < tau_fg, one point per cell, each
                  the DATASET-WIDE mean over all 22 clips. A single global choice of the
                  two-level field, fixed for every video.
  GREEN oracle -- for alpha swept over [0, 1], the PER-VIDEO argmax cell (tau_bg < tau_fg
                  only) under alpha*CLIP + (1-alpha)*Y (direction-corrected, see below),
                  averaged back across clips. The ceiling an all-knowing per-clip oracle
                  could reach with this field; it costs both a mask (R26's oracle-test
                  premise) AND per-clip supervision neither blue nor red have, so it is a
                  ceiling, not a candidate setting.

x = clip_similarity_target_image (higher is better) throughout. y is one of, selected via
`--which` (`_Y_CONFIGS`):

  part               ssim_unedit_part        (higher better) FiVE-Bench background, no R28
  union_fixed        ssim_unedit_union_fixed (higher better) R28 region-controlled family
  lpips_part         lpips_unedit_part        (LOWER better) same background, LPIPS instead
  lpips_union_fixed  lpips_unedit_union_fixed (LOWER better) same region, LPIPS instead

`both` (default) = part+union_fixed, i.e. the original two SSIM figures; `all` = all four.

Normalization -- used ONLY for the oracle's selection criterion, never for display
-------------------------------------------------------------------------------------
DISPLAY axes plot TRUE, UN-NORMALIZED values throughout -- raw CLIP on x, raw y on y, for
all three series (blue, red, green alike) -- so a LOWER-is-better y metric (LPIPS) simply
reads low-is-good on the axis exactly as its raw numbers do everywhere else in this
codebase; the plot does not flip orientation.

Min-max normalization is used ONLY internally, to pick alpha's per-video winner: each clip
independently argmaxes `alpha*CLIP_norm + (1-alpha)*Y_norm` over its OWN tau_bg<tau_fg
cells, PER-VIDEO min-max (computed independently within each clip's own off-diagonal
cells, NOT a global scale) -- reusing a global scale here would let one clip's naturally
wider raw metric range dominate the argmax for every clip and every alpha identically.
This is the same convention r26_grid_figure.py's `pick_best_tau_per_video` already uses,
for the same reason. CLIP and SSIM are both higher-is-better, but LPIPS is not, and
`alpha*CLIP_norm + (1-alpha)*Y_norm` implicitly assumes "higher normalized = more
preferred" for BOTH terms -- summing a maximize-me term with a minimize-me term without
correcting for that would push alpha's low end toward WORSE LPIPS, not better. The fix
(`oracle_frontier`'s `higher_is_better` flag): a lower-is-better y is normalized as
`1 - minmax(raw)` instead of `minmax(raw)`, so "higher normalized" means "more preferred"
uniformly regardless of the underlying metric's own direction, and the rest of the
criterion (the weighted sum, the argmax, the tie-break) is identical either way. The RAW
y value of the winning cell is recorded and averaged across clips to produce the frontier
point actually drawn for that alpha -- so the oracle curve, like blue and red, is always
true units on both axes, in the metric's own native direction.

fig2/fig4's (union_fixed variants) dataset-wide CLIP is read from r26_spatial_tau.csv (the
R26 summary), NOT from r28_union_vs_part.csv, because R28's summary carries no CLIP column
at all -- its --metrics list is the four background bases (psnr/lpips/mse/ssim) x three
families only. Their ORACLE side, by contrast, reads CLIP and the R28 y-metric TOGETHER
from the same R28 per-video CSV row (`load_per_video_metrics`), avoiding any cross-run
join for the value that actually decides the argmax.

None of the four figures is a like-for-like Pareto claim by itself -- see R26's verdict
and R28's `*_unedit_union_fixed` caveats. These figures visualize the trade-off surface a
spatial tau traces out, not a substitute for that verdict.

Usage
-----
    python evaluation/r26_tradeoff_figure.py \\
        --summary evaluation/csv/r26_spatial_tau.csv \\
        --r28_summary evaluation/csv/r28_union_vs_part.csv \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --cases evaluation/cases.json \\
        --which all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r26_grid_figure import (  # noqa: E402
    load_summary, load_r28_union_metric, load_per_video_metrics,
    _minmax, _PER_VIDEO_R26, _PER_VIDEO_R28,
)

CLIP_METRIC = "clip_similarity_target_image"
Cell = Tuple[int, int]


def oracle_frontier(per_video: Dict[Tuple[int, str], Dict[Cell, Dict[str, float]]],
                    off_diag: Sequence[Cell],
                    n_alpha: int, higher_is_better: bool = True
                    ) -> List[Tuple[float, float, float]]:
    """``[(alpha, mean_raw_clip, mean_raw_y), ...]`` -- the per-video oracle frontier.

    For each alpha, each clip INDEPENDENTLY argmaxes ``alpha*CLIP_norm + (1-alpha)*Y_norm``
    over its OWN ``tau_bg<tau_fg`` cells (per-video min-max -- see the module docstring),
    and the WINNING CELL's raw (CLIP, Y) is recorded. The frontier point for that alpha is
    the mean of those raw values across every clip with at least one off-diagonal cell
    scored. Ties break on higher CLIP then higher/lower Y per ``higher_is_better``.

    ``higher_is_better`` matters ONLY for the argmax, never for what gets returned/plotted:
    SSIM wants MAX, LPIPS wants MIN. Both metrics must be pushed onto the same
    "higher-normalized-value = more-preferred" scale before they can be summed with CLIP,
    so a lower-is-better metric is normalized as ``1 - minmax(raw)`` instead of
    ``minmax(raw)`` -- everything downstream (the weighted sum, the argmax) is then
    identical for both directions. The RAW y value recorded for the winning cell is
    untouched either way, so the returned frontier is always true units.
    """
    off_diag_set = set(off_diag)
    out: List[Tuple[float, float, float]] = []
    for alpha in np.linspace(0.0, 1.0, n_alpha):
        raw_clips, raw_ys = [], []
        for by_cell in per_video.values():
            cells = sorted(c for c in by_cell if c in off_diag_set)
            if not cells:
                continue
            clip_by_cell = {c: by_cell[c]["clip"] for c in cells}
            # `load_per_video_metrics` always keys its second value "ssim" regardless of
            # which metric was actually requested (see r26_grid_figure.py) -- it may hold
            # LPIPS here, the name is just a fixed dict key, not a claim about the metric.
            y_by_cell = {c: by_cell[c]["ssim"] for c in cells}
            cn = _minmax(clip_by_cell)
            yn = _minmax(y_by_cell)
            if not higher_is_better:
                yn = {c: 1.0 - v for c, v in yn.items()}
            score = {c: alpha * cn[c] + (1 - alpha) * yn[c] for c in cells}
            tie_y = (lambda v: v) if higher_is_better else (lambda v: -v)
            best = max(cells, key=lambda c: (score[c], clip_by_cell[c], tie_y(y_by_cell[c])))
            raw_clips.append(clip_by_cell[best])
            raw_ys.append(y_by_cell[best])
        if raw_clips:
            out.append((float(alpha), float(np.mean(raw_clips)), float(np.mean(raw_ys))))
    return out


def draw_tradeoff(ax, cells_clip: Dict[Cell, float], cells_y: Dict[Cell, float],
                  frontier: List[Tuple[float, float, float]],
                  ylabel: str, title: str, y_short: str) -> None:
    """Plots TRUE, un-normalized values throughout -- see the module docstring."""
    diag = sorted(c for c in cells_clip if c[0] == c[1])
    off_diag = sorted(c for c in cells_clip if c[0] < c[1])

    bx = [cells_clip[c] for c in diag]
    by = [cells_y[c] for c in diag]
    ax.plot(bx, by, "-o", color="tab:blue", label="baseline (tau_bg=tau_fg)", zorder=2)
    for c, x, y in zip(diag, bx, by):
        ax.annotate(str(c[0]), (x, y), fontsize=6, color="tab:blue",
                    textcoords="offset points", xytext=(3, 3))

    rx = [cells_clip[c] for c in off_diag]
    ry = [cells_y[c] for c in off_diag]
    ax.scatter(rx, ry, color="tab:red", s=14, alpha=0.7,
              label="spatial tau (tau_bg<tau_fg), single global cell", zorder=1)

    if frontier:
        gx = [c for _, c, _ in frontier]
        gy = [s for _, _, s in frontier]
        ax.plot(gx, gy, "-", color="tab:green", linewidth=2,
               label="per-video oracle (tau_bg<tau_fg), alpha sweep", zorder=3)
        ax.annotate(f"alpha=0\n({y_short})", (gx[0], gy[0]), fontsize=6, color="tab:green")
        ax.annotate("alpha=1\n(CLIP)", (gx[-1], gy[-1]), fontsize=6, color="tab:green")

    ax.set_xlabel(CLIP_METRIC)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.2)


def _one_figure(cells_clip: Dict[Cell, float], cells_y: Dict[Cell, float],
                off_diag_cells: Sequence[Cell], data_root: str, cases,
                per_video_tmpl: str, y_metric: str, higher_is_better: bool,
                ylabel: str, title: str, y_short: str,
                n_alpha: int, out_path: str) -> int:
    common = sorted(set(cells_clip) & set(cells_y))
    if not common:
        print(f"[r26_tradeoff] no cells with both CLIP and {y_metric}", file=sys.stderr)
        return 1
    off_diag_common = [c for c in off_diag_cells if c in common]
    pv = load_per_video_metrics(cases, off_diag_common, data_root, per_video_tmpl,
                                clip_metric=CLIP_METRIC, ssim_metric=y_metric)
    frontier = oracle_frontier(pv, off_diag_common, n_alpha, higher_is_better=higher_is_better)

    fig, ax = plt.subplots(figsize=(6, 5))
    draw_tradeoff(ax, {c: cells_clip[c] for c in common}, {c: cells_y[c] for c in common},
                 frontier, ylabel=ylabel, title=title, y_short=y_short)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[r26_tradeoff] wrote {out_path} ({len(common)} cells, "
          f"{len(off_diag_common)} off-diagonal, {len(frontier)} alpha points)")
    return 0


# One entry per plottable y-metric. `source` selects where the DATASET-WIDE mean comes
# from ('r26' -> --summary, 'r28' -> --r28_summary via load_r28_union_metric); the
# per-video CSV template always matches (R28 metrics need R28's per-video rows even
# though R26's own CSV also carries lpips_unedit_part -- keeping source consistent
# between the two reads is what let fig2 avoid a cross-run join for CLIP earlier).
_Y_CONFIGS: Dict[str, Dict] = {
    "part": dict(source="r26", metric="ssim_unedit_part", higher_is_better=True,
                y_short="SSIM", fig="fig1", desc="FiVE-Bench background",
                default_out="evaluation/figures/r26_tradeoff_part.pdf"),
    "union_fixed": dict(source="r28", metric="ssim_unedit_union_fixed", higher_is_better=True,
                        y_short="SSIM", fig="fig2", desc="R28, region-controlled",
                        default_out="evaluation/figures/r26_tradeoff_union_fixed.pdf"),
    "lpips_part": dict(source="r26", metric="lpips_unedit_part", higher_is_better=False,
                       y_short="LPIPS", fig="fig3", desc="FiVE-Bench background",
                       default_out="evaluation/figures/r26_tradeoff_lpips_part.pdf"),
    "lpips_union_fixed": dict(source="r28", metric="lpips_unedit_union_fixed",
                              higher_is_better=False, y_short="LPIPS", fig="fig4",
                              desc="R28, region-controlled",
                              default_out="evaluation/figures/r26_tradeoff_lpips_union_fixed.pdf"),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="r26_summarize.py's CSV")
    ap.add_argument("--r28_summary", default=None,
                    help="r28_summarize.py's CSV; required for any *_union_fixed variant")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--n_alpha", type=int, default=21, help="alpha grid points in [0, 1]")
    ap.add_argument("--which", nargs="+",
                    choices=list(_Y_CONFIGS) + ["both", "all"], default=["both"],
                    help="'both' = part+union_fixed (SSIM, original two figures); "
                         "'all' = all four; or name any subset explicitly")
    for key, cfg in _Y_CONFIGS.items():
        ap.add_argument(f"--out_{key}", default=cfg["default_out"])
    args = ap.parse_args(argv)

    which = set(args.which)
    if "both" in which:
        which |= {"part", "union_fixed"}
        which.discard("both")
    if "all" in which:
        which = set(_Y_CONFIGS)

    data_root = os.path.expanduser(args.data_root)
    with open(os.path.expanduser(args.cases), encoding="utf-8") as fh:
        cases = json.load(fh)

    summary = load_summary(args.summary)
    if args.subset not in summary:
        print(f"[r26_tradeoff] subset {args.subset!r} not in {args.summary!r}; "
              f"available: {sorted(summary)}", file=sys.stderr)
        return 1
    by_cell = summary[args.subset]
    all_cells = sorted(by_cell)
    off_diag_cells = [c for c in all_cells if c[0] < c[1]]
    diag_cells = [c for c in all_cells if c[0] == c[1]]
    print(f"[r26_tradeoff] {len(all_cells)} cells total, {len(off_diag_cells)} "
          f"off-diagonal (tau_bg<tau_fg), {len(diag_cells)} diagonal (baseline)")
    if len(all_cells) < 52:
        print(f"[r26_tradeoff] NOTE: fewer than 52 cells in {args.summary!r} -- this is a "
              "partial grid (the 40-arm R26 extension may still be running); the figure(s) "
              "below are computed over whatever is present.", file=sys.stderr)

    cells_clip = {c: v[CLIP_METRIC] for c, v in by_cell.items() if CLIP_METRIC in v}
    rc = 0

    for key in sorted(which):
        cfg = _Y_CONFIGS[key]
        if cfg["source"] == "r26":
            cells_y = {c: v[cfg["metric"]] for c, v in by_cell.items() if cfg["metric"] in v}
            pv_tmpl = _PER_VIDEO_R26
        else:
            if args.r28_summary is None:
                print(f"[r26_tradeoff] --which={key} requires --r28_summary", file=sys.stderr)
                return 1
            cells_y = load_r28_union_metric(args.r28_summary, metric=cfg["metric"],
                                            subset=args.subset)
            pv_tmpl = _PER_VIDEO_R28
        rc |= _one_figure(cells_clip, cells_y, off_diag_cells, data_root, cases,
                          pv_tmpl, cfg["metric"], cfg["higher_is_better"],
                          ylabel=cfg["metric"],
                          title=f"{cfg['fig']}: CLIP vs {cfg['metric']} ({cfg['desc']})",
                          y_short=cfg["y_short"], n_alpha=args.n_alpha,
                          out_path=getattr(args, f"out_{key}"))

    return rc


if __name__ == "__main__":
    sys.exit(main())
