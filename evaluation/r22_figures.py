#!/usr/bin/env python3
"""R22 convergence figures: where in (prefix length, denoising step) quality arrives.

Consumes the long matrix written by ``evaluation/r22_build_matrix.py``.

Two families
------------
1. ``r22_{metric}_{arm}.pdf`` -- a 22-panel grid, one panel per clip, at that
   clip's native row count (21-93 frames here). x = denoising step ``j``,
   y = prefix length ``i``.

   **Row axis follows MATRIX convention, not plot convention**: ``i`` increases
   DOWNWARD, so ``i=1`` is the top row and the panel's bottom-right cell is
   ``M(nb_frames, 14)`` -- exactly the cell the plan, the L3 identity check and
   the reports all call "the bottom-right entry". An earlier version drew
   ``i`` increasing upward, which put ``M(1,14)`` bottom-right and read as the
   opposite claim about how quality moves through a clip. Do not flip it back
   without also rewording every "bottom-right" reference.
2. ``r22_mean_surfaces.pdf`` -- one page, 8 metrics x 3 arms, each the
   clip-averaged surface on NORMALISED prefix fraction ``i/nb_frames``
   resampled to 50 bins. Clips run 21-93 frames, so a raw row-wise mean across
   clips is not defined; the normalised fraction is what makes them stackable.

What is plotted: deviation from the final column
------------------------------------------------
Every panel shows ``D(i,j) = M(i,j) - M(i,14)`` on a diverging map centred at 0,
so the last column is white **by construction** and the panel literally reads as
convergence-to-final -- the plan's stated intent.

This is the one point where the plan's wording ("diverging colormap centred on
the column-14 value") admits two readings: centre on the scalar
``M(nb_frames,14)``, or on the column ``M(i,14)``. The column is used, for two
reasons: (a) the scalar leaves the last column non-white wherever ``M(i,14)``
varies with ``i``, which it does everywhere, so the figure would NOT read as
convergence-to-final; (b) the 22 clips have very different absolute levels
(per-clip PSNR spans ~18-32 dB), so absolute values force either a useless
shared colourbar or 22 incomparable per-panel ones, whereas deviations share one
scale and make the grid directly comparable. Absolute levels are not lost --
they live in ``r22_matrix.csv`` and in the printed summary.

Note: ``clip_similarity_source_image`` is computed on the SOURCE video
(evaluate.py:107), so it is constant along ``j`` and its panels are uniformly
white. That is correct, and it doubles as a visual null control.

``--absolute``: the actual-value view
-------------------------------------
The deviation view deliberately discards level, so ``--absolute`` plots
``M(i,j)`` itself for the CLIP and LPIPS metrics (``--abs_metrics``) on a
SEQUENTIAL colormap, writing ``r22_abs_{metric}_{arm}.pdf`` and
``r22_abs_mean_surfaces.pdf``. Its colour scale is shared across the 22 panels
of a figure AND across the three arms of a metric, since cross-clip and
cross-arm comparability is the only reason to look at absolute numbers; the
cost is that within-clip structure is compressed, which is what the deviation
family is for. The two modes are complementary -- read them together.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_BINS = 50          # normalised-fraction resolution for the mean surface
CMAP = "RdBu_r"          # deviation mode: diverging, centred at 0
ABS_CMAP = "viridis"     # absolute mode: sequential, no meaningful centre
PAIR_KEY = ["editing_type_id", "file_id", "video_name"]

# --absolute defaults to the CLIP and LPIPS families. clip_similarity_source_image
# is excluded on purpose: it is computed on the SOURCE video, so it is constant
# along j and an absolute panel of it carries nothing.
ABS_METRICS = ["clip_similarity_target_image",
               "clip_similarity_target_image_edit_part",
               "lpips_unedit_part"]


def clip_surface(g: pd.DataFrame) -> tuple[np.ndarray, int]:
    """(i x j) array of M for one (arm, pair, metric) cell, rows i=1..nb_frames."""
    piv = g.pivot_table(index="i", columns="j", values="value")
    piv = piv.sort_index().sort_index(axis=1)
    return piv.to_numpy(float), int(piv.shape[0])


def deviation(S: np.ndarray) -> np.ndarray:
    """D(i,j) = M(i,j) - M(i,j_last): convergence-to-final, last column exactly 0."""
    return S - S[:, [-1]]


def robust_vmax(arrs: list[np.ndarray], pct: float = 98.0) -> float:
    """Symmetric colour limit from a high percentile of |D|, so outliers do not flatten."""
    allv = np.concatenate([np.abs(a).ravel() for a in arrs])
    allv = allv[np.isfinite(allv)]
    v = float(np.percentile(allv, pct)) if allv.size else 0.0
    return v if v > 0 else 1e-12


def abs_vlim(M: pd.DataFrame, metric: str, pct: float = 1.0) -> tuple[float, float]:
    """Absolute colour limits for one metric, SHARED across arms and clips.

    Cross-clip and cross-arm comparability is the only reason to plot absolute
    values, so the limits are computed once per metric over every arm. Robust
    percentiles (1/99) keep one extreme clip from flattening the rest.
    """
    v = M.loc[M["metric"] == metric, "value"].to_numpy(float)
    v = v[np.isfinite(v)]
    lo, hi = float(np.percentile(v, pct)), float(np.percentile(v, 100 - pct))
    return (lo, hi) if hi > lo else (lo, lo + 1e-12)


def per_clip_grid(M: pd.DataFrame, arm: str, metric: str, out: Path,
                  absolute: bool = False,
                  vlim: tuple[float, float] | None = None) -> Path:
    """22-panel grid at native row count.

    ``absolute=False`` (default, unchanged): per-clip D(i,j) on a diverging map,
    symmetric scale shared across the 22 panels of this figure.
    ``absolute=True``: M(i,j) itself on a sequential map, with ``vlim`` shared
    across arms so the three arms of a metric are directly comparable.
    """
    sub = M[(M["arm"] == arm) & (M["metric"] == metric)]
    cells = list(sub.groupby(PAIR_KEY, sort=True))
    surfs = [clip_surface(g)[0] for _, g in cells]
    if absolute:
        vmin, vmax = vlim if vlim is not None else abs_vlim(M, metric)
        cmap = ABS_CMAP
    else:
        surfs = [deviation(S) for S in surfs]
        vmax = robust_vmax(surfs)
        vmin, cmap = -vmax, CMAP

    ncol = 6
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.15 * nrow))
    axes = np.atleast_1d(axes).ravel()

    im = None
    for ax, (key, g), D in zip(axes, cells, surfs):
        et, _fid, name = key
        nb = D.shape[0]
        # MATRIX convention: i=1 at the TOP, i=nb at the BOTTOM, so the panel's
        # bottom-right cell is M(nb_frames,14) -- the same cell the plan, the
        # L3 check and the reports all call "the bottom-right entry".
        im = ax.imshow(D, aspect="auto", origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       extent=(-0.5, D.shape[1] - 0.5, nb + 0.5, 0.5))
        ax.set_title(f"{name}  (e{et}, {nb}f)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xticks([0, 7, 14])
    for ax in axes[len(cells):]:
        ax.axis("off")

    for k, ax in enumerate(axes[:len(cells)]):
        if k % ncol == 0:
            ax.set_ylabel("prefix $i$", fontsize=7)
        if k >= len(cells) - ncol:
            ax.set_xlabel("step $j$", fontsize=7)

    title = (f"{metric} — {arm} — actual values  $M(i,j)$  "
             f"(scale shared across arms)" if absolute else
             f"{metric} — {arm} — deviation from final column  $M(i,j)-M(i,14)$")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))
    cax = fig.add_axes((0.945, 0.12, 0.012, 0.76))
    if im is not None:
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)
    stem = "r22_abs" if absolute else "r22"
    p = out / f"{stem}_{metric}_{arm}.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def resample_fraction(D: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Resample rows onto normalised prefix fraction i/nb_frames on a fixed grid."""
    nb, nj = D.shape
    f_src = np.arange(1, nb + 1) / nb
    f_dst = np.arange(1, n_bins + 1) / n_bins
    return np.stack([np.interp(f_dst, f_src, D[:, j]) for j in range(nj)], axis=1)


def mean_surfaces(M: pd.DataFrame, out: Path, absolute: bool = False,
                  metrics: list[str] | None = None) -> Path:
    """One page: clip-averaged surface on normalised prefix fraction, metrics x arms.

    ``absolute=False`` (default, unchanged) averages the deviation D; ``True``
    averages M itself and shares one sequential scale per metric row.
    """
    metrics = metrics if metrics is not None else sorted(M["metric"].unique())
    arms = sorted(M["arm"].unique())
    fig, axes = plt.subplots(len(metrics), len(arms),
                             figsize=(3.15 * len(arms), 2.5 * len(metrics)),
                             squeeze=False, layout="constrained")

    for r, metric in enumerate(metrics):
        surfaces = {}
        for arm in arms:
            sub = M[(M["arm"] == arm) & (M["metric"] == metric)]
            stack = [resample_fraction(clip_surface(g)[0] if absolute
                                       else deviation(clip_surface(g)[0]))
                     for _, g in sub.groupby(PAIR_KEY, sort=True)]
            surfaces[arm] = np.mean(np.stack(stack, axis=0), axis=0)
        if absolute:
            allv = np.concatenate([s.ravel() for s in surfaces.values()])
            vmin, vmax, cmap = float(allv.min()), float(allv.max()), ABS_CMAP
        else:
            vmax = robust_vmax(list(surfaces.values()), pct=100.0)  # already averaged
            vmin, cmap = -vmax, CMAP

        for c, arm in enumerate(arms):
            ax = axes[r][c]
            S = surfaces[arm]
            # same matrix convention as per_clip_grid: prefix fraction 0 at the
            # top, the full clip (i/n = 1) at the bottom
            im = ax.imshow(S, aspect="auto", origin="upper", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=(-0.5, S.shape[1] - 0.5, 1.0, 0.0))
            ax.set_xticks([0, 7, 14])
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(arm, fontsize=9)
            if c == 0:
                ax.set_ylabel(metric.replace("_", "\n"), fontsize=6.5)
            if r == len(metrics) - 1:
                ax.set_xlabel("step $j$", fontsize=8)
        fig.colorbar(im, ax=axes[r].tolist(), fraction=0.02, pad=0.01
                     ).ax.tick_params(labelsize=6)

    fig.suptitle(
        ("R22 clip-averaged surfaces — actual values $M(i,j)$, "
         "y = prefix fraction $i/n$ (50 bins)") if absolute else
        ("R22 clip-averaged convergence surfaces — deviation from final "
         "column, y = prefix fraction $i/n$ (50 bins)"), fontsize=11)
    p = out / ("r22_abs_mean_surfaces.pdf" if absolute else "r22_mean_surfaces.pdf")
    fig.savefig(p)
    plt.close(fig)
    return p


def print_summary(M: pd.DataFrame, tol_frac: float = 0.01) -> None:
    """Earliest step at which the FULL-clip prefix is within tol of its final value."""
    full = M[M["i"] == M["nb_frames"]]
    print(f"\n=== earliest step j with |M(n,j) - M(n,14)| <= "
          f"{tol_frac:.0%} of |M(n,14)| (median over 22 clips) ===")
    for metric in sorted(full["metric"].unique()):
        row = []
        for arm in sorted(full["arm"].unique()):
            sub = full[(full["metric"] == metric) & (full["arm"] == arm)]
            js = []
            for _, g in sub.groupby(PAIR_KEY, sort=False):
                g = g.sort_values("j")
                v = g["value"].to_numpy(float)
                ok = np.abs(v - v[-1]) <= tol_frac * abs(v[-1])
                # first j from which it stays inside the band
                stay = np.where(~ok)[0]
                js.append(int(stay[-1] + 1) if stay.size else 0)
            row.append(f"{arm}={np.median(js):.1f}")
        print(f"  {metric:<42} " + "  ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=Path("evaluation/csv/r22_matrix.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("evaluation/figures"))
    ap.add_argument("--only_metric", nargs="*", default=None,
                    help="restrict the per-clip grids to these metrics")
    ap.add_argument("--absolute", action="store_true",
                    help="plot actual values M(i,j) for --abs_metrics instead of "
                         "the deviation from the final column")
    ap.add_argument("--abs_metrics", nargs="*", default=ABS_METRICS,
                    help=f"metrics for --absolute (default: {' '.join(ABS_METRICS)})")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    M = pd.read_csv(args.matrix)
    print(f"[r22_figures] read {args.matrix}: {len(M)} rows, "
          f"arms={sorted(M['arm'].unique())}, metrics={M['metric'].nunique()}")

    if args.absolute:
        known = set(M["metric"].unique())
        metrics = [m for m in args.abs_metrics if m in known]
        missing = [m for m in args.abs_metrics if m not in known]
        if missing:
            print(f"[r22_figures] WARNING: not in the matrix, skipped: {missing}")
        if not metrics:
            raise SystemExit("[r22_figures] no --abs_metrics present in the matrix")
        # limits computed once per metric so all arms of a row share one scale
        vlims = {m: abs_vlim(M, m) for m in metrics}
        n = 0
        for arm in sorted(M["arm"].unique()):
            for metric in metrics:
                per_clip_grid(M, arm, metric, args.outdir,
                              absolute=True, vlim=vlims[metric])
                n += 1
        print(f"[r22_figures] wrote {n} absolute per-clip grids -> "
              f"{args.outdir}/r22_abs_<metric>_<arm>.pdf")
        for m, (lo, hi) in vlims.items():
            print(f"[r22_figures]   shared scale {m:<42} [{lo:.4g}, {hi:.4g}]")
        p = mean_surfaces(M, args.outdir, absolute=True, metrics=metrics)
        print(f"[r22_figures] wrote {p}")
        return

    metrics = sorted(M["metric"].unique())
    if args.only_metric:
        metrics = [m for m in metrics if m in set(args.only_metric)]

    n = 0
    for arm in sorted(M["arm"].unique()):
        for metric in metrics:
            per_clip_grid(M, arm, metric, args.outdir)
            n += 1
    print(f"[r22_figures] wrote {n} per-clip grids -> {args.outdir}/r22_<metric>_<arm>.pdf")

    p = mean_surfaces(M, args.outdir)
    print(f"[r22_figures] wrote {p}")

    print_summary(M)


if __name__ == "__main__":
    main()
