"""R45 score -- does the divergence routing generalise beyond Self-Forcing?

For each SELECTED routing combo (``evaluation/r45_method.json``) and each backbone, the
routed arm is read against that backbone's OWN uniform VP rho sweep, at matched background
LPIPS:

    SF : R36 curve (N=15) + R39 curve at the matched N (15 + draft N);
         arm = the R35 (draft N=15) / R38 (draft N=5) counterpart of the combo.
    LL : R45 sweep at N=15 + R45 sweep at the matched N; arm = the R45 combo.

Only within-backbone gains are compared across backbones -- the backbones run their native
fg_boost / VP settings, which shift absolute scores (plan Decisions).

OUTPUTS
    evaluation/figures/r45_{clip,clipd,fiveacc}_vs_lpips.pdf   two panels (SF | LL) each:
        x = CLIP-T / CLIP-D / FiVE-Acc yes-no (alignment), y = lpips_unedit_part (preservation);
        gray curves (solid N=15, dashed matched N, rho-labelled at the ends),
        arms: hue = divergence metric, marker = stage-1 regime.
    evaluation/csv/r45_arms.csv   one row per (backbone, combo, x metric): the arm's value and
        LPIPS, each curve interpolated at that LPIPS, ``gain_vs_n15`` / ``gain_vs_matched``
        (arm minus curve; > 0 = more edit at equal preservation) and their flags
        (``ok`` | ``outside_range`` -> NaN, never extrapolated | ``nonmonotone``).
    evaluation/csv/r45_rho_sweep_n{15,M}.csv   the LL curves, built here with
        r36_summarize.py from the R45 per-clip CSVs (skip with --no_summarize).

AGGREGATION: every point is a PLAIN MEAN OVER THE 419 CLIPS from the per-clip CSVs
(r35_score.py's loaders, which hard-fail on any missing clip) -- never {stem}_avg.csv.
Every input is required; a missing CSV is a hard failure naming the step that writes it.

    python evaluation/r45_score.py --all
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
    ARM_COLOR, GRID, INK, MUTED, N_CLIPS, REF_COLOR, RHOS, SURFACE, Y_METRIC,
    clip_index, load_clipd, load_harness, r36_points,
)
from r39_pareto import interp_at  # noqa: E402
import r36_summarize  # noqa: E402
import r45_select as sel  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CSV_DIR = REPO / "evaluation" / "csv"

# figure key -> (x metric, x-axis label, file name)
FIGURES: Dict[str, Tuple[str, str, str]] = {
    "clip": ("clip_similarity_target_image", "CLIP-T (target image)  → more edit",
             "r45_clip_vs_lpips.pdf"),
    "clipd": ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit",
              "r45_clipd_vs_lpips.pdf"),
    "fiveacc": ("five_acc_yes_no", "FiVE-Acc (yes/no)  → more edit",
                "r45_fiveacc_vs_lpips.pdf"),
}
X_METRICS: Tuple[str, ...] = tuple(v[0] for v in FIGURES.values())
REGIME_MARKER = {"unblended": "o", "tgt09": "^"}          # filled circle / hollow triangle
REGIME_LABEL = {"unblended": "unblended draft", "tgt09": "τ = 0.9 draft"}
BACKBONE_TITLE = {"sf": "Self-Forcing", "ll": "LongLive"}


def parse_combo(combo: str) -> Tuple[str, int, str]:
    """'ll_dino_n5_tgt09' -> ('dino', 5, 'tgt09')."""
    _, metric, ntag, regime = combo.split("_", 3)
    return metric, int(ntag[1:]), regime


def sf_stems(combo: str) -> Tuple[str, str, Path, str]:
    """(harness stem, fiveacc stem, clipd csv, clipd method) of the SF counterpart."""
    metric, n, regime = parse_combo(combo)
    if n == 15:
        c = f"{metric}_{'first2' if regime == 'tgt09' else 'unblended'}"   # tau=0.9 == first2
        return f"r35_{c}", f"r35_fiveacc_{c}", CSV_DIR / "r35_clip_directional.csv", f"r35_{c}"
    c = f"{metric}_n5_{regime}"
    return f"r38_{c}", f"r38_fiveacc_{c}", CSV_DIR / "r38_clip_directional.csv", f"r38_{c}"


def ll_stems(combo: str) -> Tuple[str, str, Path, str]:
    """Stems written by r45_eval.sh / r45_fiveacc.sh / r45_clipd.sh for a routed combo."""
    return (f"r45_{combo}", f"r45_fiveacc_{combo}", CSV_DIR / "r45_clip_directional.csv",
            f"r45_{combo}")


def arm_point(stems: Tuple[str, str, Path, str], clips) -> Dict[str, float]:
    harness, fiveacc, clipd_csv, clipd_method = stems
    per = {
        Y_METRIC: load_harness(CSV_DIR, harness, Y_METRIC, clips),
        "clip_similarity_target_image":
            load_harness(CSV_DIR, harness, "clip_similarity_target_image", clips),
        "five_acc_yes_no": load_harness(CSV_DIR, fiveacc, "five_acc_yes_no", clips),
        "clip_d_prompt": load_clipd(clipd_csv, clipd_method, clips),
    }
    return {m: float(np.mean(list(v.values()))) for m, v in per.items()}


def summarize_ll(n: int) -> Path:
    """Build the LL rho curve at N with r36_summarize.py (plain per-clip mean)."""
    out = CSV_DIR / f"r45_rho_sweep_n{n}.csv"
    r36_summarize.main([
        "--harness_fmt", f"r45_ll_rho{{rho}}_n{n}_vp",
        "--fiveacc_fmt", f"r45_fiveacc_ll_rho{{rho}}_n{n}_vp",
        "--clipd_csv", str(CSV_DIR / "r45_clip_directional.csv"),
        "--clipd_method_fmt", f"r45_ll_rho{{rho}}_n{n}_vp",
        "--out", str(out)])
    return out


def gains(arm: Dict[str, float], curves: Dict[str, List[Dict[str, object]]], x: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, curve in curves.items():                       # key: "n15" | "matched"
        ref, flag = interp_at(curve, arm[Y_METRIC], x)
        out[f"ref_{key}"] = ref
        out[f"gain_vs_{key}"] = arm[x] - ref if not math.isnan(ref) else math.nan
        out[f"flag_{key}"] = flag
    return out


def plot(key: str, panels: Dict[str, Dict[str, object]], matched_n: int, fig_dir: Path) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, bb in zip(axes, ("sf", "ll")):
        ax.set_facecolor(SURFACE)
        for ckey, ls in (("n15", "-"), ("matched", "--")):
            curve = panels[bb]["curves"][ckey]
            xs = [p[x_metric] for p in curve]
            ys = [p[Y_METRIC] for p in curve]
            n = 15 if ckey == "n15" else matched_n
            ax.plot(xs, ys, color=REF_COLOR, lw=2, ls=ls, marker="o", ms=3.5, zorder=2,
                    label=f"uniform VP, N={n}")
            for i in (0, len(RHOS) - 1):
                ax.annotate(f"ρ={RHOS[i]}", (xs[i], ys[i]), xytext=(-5, 5),
                            textcoords="offset points", fontsize=7, color=MUTED,
                            ha="right", va="bottom")
        for combo, a in panels[bb]["arms"].items():
            metric, _, regime = parse_combo(combo)
            col = ARM_COLOR[metric]
            filled = regime == "unblended"
            ax.scatter([a[x_metric]], [a[Y_METRIC]], s=75, marker=REGIME_MARKER[regime],
                       facecolors=col if filled else SURFACE, edgecolors=col, linewidths=2,
                       zorder=4)
        ax.set_title(BACKBONE_TITLE[bb], fontsize=10, color=INK)
        ax.set_xlabel(x_label, color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    handles, _ = axes[1].get_legend_handles_labels()
    metrics = sorted({parse_combo(c)[0] for c in panels["ll"]["arms"]})
    regimes = sorted({parse_combo(c)[2] for c in panels["ll"]["arms"]})
    handles += [Line2D([], [], ls="", marker="s", ms=7, color=ARM_COLOR[m],
                       label=f"{m.upper()} divergence routing") for m in metrics]
    handles += [Line2D([], [], ls="", marker=REGIME_MARKER[r], ms=7, mec=MUTED, mew=2,
                       mfc=MUTED if r == "unblended" else SURFACE, label=REGIME_LABEL[r])
                for r in regimes]
    axes[1].legend(handles=handles, fontsize=7, frameon=False, loc="upper left",
                   bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(f"R45 — routing vs same-backbone uniform VP sweeps, full FiVE-Bench "
                 f"({N_CLIPS} pairs, per-clip mean)", fontsize=10, color=INK)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / fname
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="build all three figures (default)")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+", default=list(FIGURES))
    ap.add_argument("--no_summarize", action="store_true",
                    help="read existing r45_rho_sweep_n*.csv instead of rebuilding them")
    ap.add_argument("--sf_n15_sweep", type=Path, default=CSV_DIR / "r36_rho_sweep.csv")
    ap.add_argument("--sf_matched_fmt", default=str(CSV_DIR / "r39_rho_sweep_n{n}.csv"))
    ap.add_argument("--ll_sweep_fmt", default=str(CSV_DIR / "r45_rho_sweep_n{n}.csv"))
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--outdir", type=Path, default=REPO / "evaluation" / "figures")
    ap.add_argument("--arms_csv", type=Path, default=CSV_DIR / "r45_arms.csv")
    args = ap.parse_args(argv)

    m = sel.load()
    mn = m["matched_n"]
    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r45_score] benchmark has {len(clips)} clips, expected {N_CLIPS}")

    if not args.no_summarize:
        for n in (15, mn):
            summarize_ll(n)
    panels = {
        "sf": {"curves": {"n15": r36_points(args.sf_n15_sweep),
                          "matched": r36_points(Path(args.sf_matched_fmt.format(n=mn)))}},
        "ll": {"curves": {"n15": r36_points(Path(args.ll_sweep_fmt.format(n=15))),
                          "matched": r36_points(Path(args.ll_sweep_fmt.format(n=mn)))}},
    }
    for bb, stems_of in (("sf", sf_stems), ("ll", ll_stems)):
        panels[bb]["arms"] = {c: arm_point(stems_of(c), clips) for c in m["combos"]}

    rows: List[Dict[str, object]] = []
    for combo in m["combos"]:
        metric, n, regime = parse_combo(combo)
        for bb in ("sf", "ll"):
            arm = panels[bb]["arms"][combo]
            stem = (sf_stems if bb == "sf" else ll_stems)(combo)[0]
            for x in X_METRICS:
                g = gains(arm, panels[bb]["curves"], x)
                rows.append({"backbone": bb, "combo": combo, "arm_stem": stem, "draft_n": n,
                             "matched_n": mn, "metric": metric, "regime": regime,
                             "n_clips": N_CLIPS, "x_metric": x, "value": arm[x],
                             Y_METRIC: arm[Y_METRIC], **g})
                print(f"[r45_score] {bb} {combo:<24} {x:<29} lpips={arm[Y_METRIC]:.4f} "
                      f"gain vs N=15 {g['gain_vs_n15']:+.4f} ({g['flag_n15']}) · "
                      f"vs N={mn} {g['gain_vs_matched']:+.4f} ({g['flag_matched']})", flush=True)

    args.arms_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.arms_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[r45_score] {len(rows)} rows -> {args.arms_csv}")
    for k in args.x:
        print(f"[r45_score] wrote {plot(k, panels, mn, args.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
