"""R39 pareto -- compute-matched uniform baselines vs the divergence-routing arms.

WHAT IS PLOTTED
---------------
Preservation on y (``lpips_unedit_part``, lower = better) against one editability measure
on x, one PDF per measure:

    r39_clip_vs_lpips.pdf          x = clip_similarity_target_image  (CLIP-T)
    r39_clipd_vs_lpips.pdf         x = clip_d_prompt                 (CLIP-D, R33's axis of record)
    r39_fiveacc_full_vs_lpips.pdf  x = five_acc                      (FiVE-Acc proper: per-clip
                                                                      mean of yes/no, MC, union, inter)
    r39_fiveacc_vs_lpips.pdf       x = five_acc_yes_no               (FiVE-Acc, yes/no only)

Three uniform VP rho curves (rho in {2,3,4,6,8,10,20,50}, joined in rho order):
* R36, N=15  -- 15 NFE, neutral gray reference, every point rho-labelled (as in R35/R38);
* R39, N=20  -- 20 NFE, blue  (= R38's 15+5 budget: 5-step draft + 15-step render);
* R39, N=30  -- 30 NFE, orange (= R35's 15+15 budget).
The routing arms are points coloured like their NFE-MATCHED curve: R38 (15+5) blue,
R35 (15+15) orange. Marker = divergence metric (DINO circle, LPIPS square); fill = stage-1
regime (unblended filled, anchored -- first2 / tgt0.9 -- hollow). Rho is NOT comparable
across N (blend_power acts per step), so only the R36 curve carries rho labels; the R39
curves are labelled at their ends.

MATCHED DELTAS (--deltas_csv, r39_matched_deltas.csv)
-----------------------------------------------------
One row per (routing arm, x metric): the arm's value minus each curve linearly interpolated
at the arm's LPIPS -- against R36 (N=15) and against the NFE-matched curve (R38 -> N=20,
R35 -> N=30). Positive = the arm edits more than the curve at equal preservation.
``flag_*``: ``ok`` | ``outside_range`` (arm LPIPS beyond the curve's span -> NaN, never
extrapolated) | ``nonmonotone`` (curve LPIPS not strictly increasing in rho; the
interpolation is still taken on the LPIPS-sorted points, but read it with care) |
``no_curve`` (that curve's sweep CSV is absent). No significance test (out of scope).

AGGREGATION. Every point is a PLAIN MEAN OVER THE 419 CLIPS, as everywhere since R35.
* Curves: the ``n_pairs == 419`` rows of r36_rho_sweep.csv / r39_rho_sweep_n{20,30}.csv,
  written by r36_summarize.py (plain per-clip mean), loaded with r35_score.r36_points.
* Arms: computed HERE from the per-clip CSVs with r35_score's loaders -- not from
  r35_arms.csv / r38_arms.csv: r38_arms.csv does not carry five_acc (full), and one code
  path for both sources keeps them comparable. Same computation as r35_score, so the R35
  values equal r35_arms.csv's. Roster = r38_score.POINTS.

MISSING INPUTS. R36's sweep and R35's per-clip CSVs exist and are HARD requirements. An
R39 sweep CSV, or an R38 combo's per-clip CSVs (not yet scored, or dropped at R38's
gate-lowN), is skipped with a printed NOTICE and shows up as ``no_curve`` / absent rows in
the deltas table -- never silently.

Example
-------
    python evaluation/r39_pareto.py --outdir evaluation/figures
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r35_score import (  # noqa: E402
    N_CLIPS, RHOS, Y_METRIC, clip_index, load_clipd, load_harness, r36_points,
    INK, MUTED, GRID, SURFACE, REF_COLOR,
)
from r38_score import POINTS  # noqa: E402  (source, combo, draft N, metric, regime)

# figure key -> (x metric, x-axis label, file name)
FIGURES: Dict[str, Tuple[str, str, str]] = {
    "clip":         ("clip_similarity_target_image", "CLIP-T (target image)  → more edit",
                     "r39_clip_vs_lpips.pdf"),
    "clipd":        ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit",
                     "r39_clipd_vs_lpips.pdf"),
    "fiveacc_full": ("five_acc", "FiVE-Acc (mean of yes/no, MC, union, inter)  → more edit",
                     "r39_fiveacc_full_vs_lpips.pdf"),
    "fiveacc":      ("five_acc_yes_no", "FiVE-Acc (yes/no only)  → more edit",
                     "r39_fiveacc_vs_lpips.pdf"),
}
X_METRICS: Tuple[str, ...] = tuple(v[0] for v in FIGURES.values())
FIVEACC_X: Tuple[str, ...] = ("five_acc", "five_acc_yes_no")

# NFE budget -> colour. Slots 1-2 of the reference categorical palette (blue, orange), the
# pair R35/R38 already use; the gray reference is R36 at 15 NFE.
NFE_COLOR = {20: "#2a78d6", 30: "#eb6834"}
SOURCE_NFE = {"R38": 20, "R35": 30}                    # routing arm -> matched single-pass N
SOURCE_LABEL = {"R38": "15+5", "R35": "15+15"}
METRIC_MARKER = {"dino": "o", "lpips": "s"}


# ----------------------------------------------------------------------- inputs -------
def load_curves(r36_sweep: Path, r39_fmt: str) -> Dict[int, List[Dict[str, object]]]:
    """{N: 8 rho points}. R36 (N=15) is required; a missing R39 sweep is a NOTICE."""
    curves = {15: r36_points(r36_sweep)}
    for n in (20, 30):
        path = Path(r39_fmt.format(n=n))
        if not path.is_file():
            print(f"[r39_pareto] NOTICE: {path} missing -- N={n} curve skipped "
                  f"(R39 summarize not run for it yet)")
            continue
        curves[n] = r36_points(path)
    return curves


def arm_points(csv_dir: Path, clips: Dict[Tuple[int, str], str]) -> List[Dict[str, object]]:
    """One point per routing combo with all four x metrics; R35 required, R38 optional."""
    pts = []
    for source, combo, n_draft, metric, regime in POINTS:
        pre = source.lower()
        try:
            per = {
                Y_METRIC: load_harness(csv_dir, f"{pre}_{combo}", Y_METRIC, clips),
                "clip_similarity_target_image":
                    load_harness(csv_dir, f"{pre}_{combo}", "clip_similarity_target_image", clips),
                **{m: load_harness(csv_dir, f"{pre}_fiveacc_{combo}", m, clips) for m in FIVEACC_X},
                "clip_d_prompt": load_clipd(csv_dir / f"{pre}_clip_directional.csv",
                                            f"{pre}_{combo}", clips),
            }
        except SystemExit as e:
            if source == "R35":
                raise
            print(f"[r39_pareto] NOTICE: {source} {combo} skipped -- {e}")
            continue
        pts.append({"source": source, "combo": combo, "N_draft": n_draft, "metric": metric,
                    "regime": regime, "n_clips": N_CLIPS,
                    **{m: float(np.mean(list(v.values()))) for m, v in per.items()}})
    return pts


# ----------------------------------------------------------------------- deltas -------
def interp_at(curve: List[Dict[str, object]], lpips: float, x: str) -> Tuple[float, str]:
    """Metric x along a rho curve, linearly interpolated at a given LPIPS; never extrapolated."""
    ls_rho = [float(p[Y_METRIC]) for p in curve]                   # rho order
    flag = "ok" if all(a < b for a, b in zip(ls_rho, ls_rho[1:])) else "nonmonotone"
    pts = sorted((float(p[Y_METRIC]), float(p[x])) for p in curve)
    ls, xs = zip(*pts)
    if not ls[0] <= lpips <= ls[-1]:
        return math.nan, "outside_range"
    return float(np.interp(lpips, ls, xs)), flag


def matched_deltas(arms: List[Dict[str, object]],
                   curves: Dict[int, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    rows = []
    for a in arms:
        n_match = SOURCE_NFE[a["source"]]
        for x in X_METRICS:
            ref15, f15 = interp_at(curves[15], a[Y_METRIC], x)
            if n_match in curves:
                refm, fm = interp_at(curves[n_match], a[Y_METRIC], x)
            else:
                refm, fm = math.nan, "no_curve"
            rows.append({
                "source": a["source"], "combo": a["combo"], "nfe": SOURCE_LABEL[a["source"]],
                "matched_curve": f"n{n_match}", "x_metric": x,
                "lpips": a[Y_METRIC], "value": a[x],
                "ref_r36_n15": ref15, "delta_vs_r36_n15": a[x] - ref15, "flag_r36_n15": f15,
                "ref_matched": refm, "delta_vs_matched": a[x] - refm, "flag_matched": fm,
            })
    return rows


def write_deltas(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------- figures ------
def plot(key: str, curves: Dict[int, List[Dict[str, object]]], arms: List[Dict[str, object]],
         fig_dir: Path) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # uniform curves: 8 rho points joined in rho order
    for n, curve in sorted(curves.items()):
        col = REF_COLOR if n == 15 else NFE_COLOR[n]
        name = "R36" if n == 15 else "R39"
        xs = [p[x_metric] for p in curve]
        ys = [p[Y_METRIC] for p in curve]
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4, zorder=2,
                label=f"{name} · uniform VP, N={n} ({n} NFE)")
        # R36: every rho (the reference, as in R35/R38); R39: ends only -- rho is not
        # comparable across N. Above-left, away from the arms that beat a curve.
        idx = range(len(RHOS)) if n == 15 else (0, len(RHOS) - 1)
        for i in idx:
            ax.annotate(f"ρ={RHOS[i]}", (xs[i], ys[i]), xytext=(-5, 5),
                        textcoords="offset points", fontsize=7, color=MUTED,
                        ha="right", va="bottom")

    # routing arms: colour = NFE-matched curve, marker = divergence metric, fill = regime
    for a in arms:
        col = NFE_COLOR[SOURCE_NFE[a["source"]]]
        filled = a["regime"] == "unblended"
        ax.scatter([a[x_metric]], [a[Y_METRIC]], s=70, marker=METRIC_MARKER[a["metric"]],
                   facecolors=col if filled else SURFACE, edgecolors=col, linewidths=2,
                   zorder=4)

    handles, _ = ax.get_legend_handles_labels()
    for src in ("R38", "R35"):
        if any(a["source"] == src for a in arms):
            n = SOURCE_NFE[src]
            handles.append(Line2D([], [], ls="", marker="o", ms=7, color=NFE_COLOR[n],
                                  label=f"{src} · routing, {SOURCE_LABEL[src]} ({n} NFE)"))
    handles += [
        Line2D([], [], ls="", marker="o", ms=7, mfc=MUTED, mec=MUTED, label="DINO divergence"),
        Line2D([], [], ls="", marker="s", ms=7, mfc=MUTED, mec=MUTED, label="LPIPS divergence"),
        Line2D([], [], ls="", marker="o", ms=7, mfc=MUTED, mec=MUTED, label="unblended draft (filled)"),
        Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=MUTED, mew=2,
               label="anchored draft (hollow)"),
    ]

    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    ax.set_title(f"R39 compute-matched baselines vs divergence routing — full FiVE-Bench "
                 f"({N_CLIPS} pairs, per-clip mean)", fontsize=9, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / fname
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------- main ---------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+", default=list(FIGURES),
                    help="Figures to build (default: all four).")
    ap.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    ap.add_argument("--r36_sweep", type=Path, default=Path("evaluation/csv/r36_rho_sweep.csv"))
    ap.add_argument("--r39_sweep_fmt", default="evaluation/csv/r39_rho_sweep_n{n}.csv",
                    help="R39 sweep CSV per N; {n} is replaced by 20 / 30.")
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--outdir", type=Path, default=Path("evaluation/figures"))
    ap.add_argument("--deltas_csv", type=Path, default=Path("evaluation/csv/r39_matched_deltas.csv"))
    args = ap.parse_args(argv)

    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r39_pareto] benchmark has {len(clips)} clips, expected {N_CLIPS}")

    curves = load_curves(args.r36_sweep, args.r39_sweep_fmt)
    arms = arm_points(args.csv_dir, clips)
    print(f"[r39_pareto] curves N={sorted(curves)}; arms: "
          f"{sum(a['source'] == 'R35' for a in arms)} R35 + {sum(a['source'] == 'R38' for a in arms)} R38")

    rows = matched_deltas(arms, curves)
    write_deltas(args.deltas_csv, rows)
    print(f"[r39_pareto] {len(rows)} matched-delta rows -> {args.deltas_csv}")
    for r in rows:
        print(f"  {r['source']} {r['combo']:<20} {r['x_metric']:<29} lpips={r['lpips']:.4f} "
              f"Δ vs R36 N=15 {r['delta_vs_r36_n15']:+.4f} ({r['flag_r36_n15']}) · "
              f"Δ vs {r['matched_curve']} {r['delta_vs_matched']:+.4f} ({r['flag_matched']})")
    for k in args.x:
        print(f"[r39_pareto] wrote {plot(k, curves, arms, args.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
