"""R44 score -- is the divergence-routing gain tied to visual prompting?

For every selected combo (evaluation/r44_selected_combos.txt) the routing gain is measured
TWICE, each time against a uniform-rho curve of the SAME modality:

    text-only : R44 arm r44_{combo}        vs  R44 text-only uniform curves
    VP        : its VP counterpart         vs  R36 (N=15) / R39 (N=20, 30) VP uniform curves
                n15_tgt09     -> R35 {metric}_first2    (tau = 0.9 IS first2 at N=15)
                n15_unblended -> R35 {metric}_unblended
                n5_{regime}   -> R38 {combo}

EQUAL NFE IS MANDATORY. A routed arm costs draft + render = 15 + N_d model steps, so the
HEADLINE gain is against the uniform curve at N = 15 + N_d (``--matched_n``, 20 or 30,
chosen at plan step choose-matched); the N=15 gain is reported as secondary. The final
run hard-fails without both equal-NFE curves, or if any selected combo has 15 + N_d !=
--matched_n. Only ``--interim`` (N=15 curves only) may omit them, and it writes
``*_interim`` outputs with "INTERIM" in every title -- never the final file names.

GAIN. arm's x minus the curve linearly interpolated at the arm's bg LPIPS
(``r39_pareto.interp_at``: never extrapolated, ``outside_range`` -> NaN; ``nonmonotone``
flags a curve whose LPIPS is not increasing in rho). Positive = the arm edits more than the
uniform sweep at equal preservation. ``diff_*`` = gain_text - gain_vp: ~0 means routing
helps text-only as much as VP (not tied to VP); clearly negative means the gain needs VP.
No significance test (out of scope, as in R39).

AGGREGATION. Plain mean over the 419 clips everywhere, from the per-clip CSVs, NEVER
``{stem}_avg.csv``. Arms use r35_score's loaders; curves are r36_summarize.py outputs read
with ``r35_score.r36_points`` (its messages therefore carry an ``[r35_score]`` prefix).

OUTPUTS (``_interim`` suffix under --interim)
    evaluation/figures/r44_{clip,clipd,fiveacc}_vs_lpips.pdf   two panels: VP | text-only
    evaluation/csv/r44_arms.csv          one row per plotted point (curves + arms)
    evaluation/csv/r44_routing_gain.csv  one row per (combo, x metric)

Example
-------
    python evaluation/r44_score.py --all --interim           # early look, N=15 only
    python evaluation/r44_score.py --all --matched_n 30      # the result
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r35_score import (  # noqa: E402
    ARM_COLOR, GRID, INK, MUTED, N_CLIPS, REF_COLOR, RHOS, SURFACE, Y_METRIC,
    clip_index, load_clipd, load_harness, r36_points,
)
from r39_pareto import interp_at  # noqa: E402
from r44_div_stats import parse_combo, read_selection  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

X_METRICS: Tuple[str, ...] = ("clip_similarity_target_image", "clip_d_prompt", "five_acc_yes_no")
FIGURES: Dict[str, Tuple[str, str, str]] = {
    "clip":    ("clip_similarity_target_image", "CLIP-T (target image)  → more edit", "r44_clip_vs_lpips"),
    "clipd":   ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit", "r44_clipd_vs_lpips"),
    "fiveacc": ("five_acc_yes_no", "FiVE-Acc (yes/no)  → more edit", "r44_fiveacc_vs_lpips"),
}
# Equal-NFE curve colour: categorical slot 3 (aqua) of the reference palette, after the
# blue / orange slots 1-2 that ARM_COLOR already gives DINO / LPIPS. Curves are also told
# apart by marker + legend, arms by marker shape + fill, so no identity is colour-only.
EQ_COLOR = "#1baf7a"
REGIME_MARKER = {"unblended": "o", "tgt09": "^"}
REGIME_LABEL = {"unblended": "unblended", "tgt09": "τ=0.9"}


# ----------------------------------------------------------------------- inputs -------
def load_curve(path: Path, what: str,
               expected_rhos: Sequence[int] = RHOS) -> List[Dict[str, object]]:
    """Rho points (full-bench rows) of an r36_summarize.py output; hard-fail if absent.

    ``expected_rhos`` defaults to the standard 8-point grid. Pass a subset to validate a
    reduced sweep instead (e.g. R44's own equal-NFE text curve, run at fewer rhos to save
    compute when the VP-side comparison curve already exists at full resolution)."""
    if not path.is_file():
        raise SystemExit(f"[r44_score] missing {path} ({what}) -- run its summarize step first")
    return r36_points(path, expected_rhos=expected_rhos)


def arm_point(csv_dir: Path, clips: Dict[Tuple[int, str], str], stem: str, fiveacc_stem: str,
              clipd_csv: str, clipd_method: str) -> Dict[str, float]:
    """Plain mean over the 419 clips of y + the three x metrics; hard-fail on any gap."""
    per = {
        Y_METRIC: load_harness(csv_dir, stem, Y_METRIC, clips),
        "clip_similarity_target_image": load_harness(csv_dir, stem, "clip_similarity_target_image", clips),
        "five_acc_yes_no": load_harness(csv_dir, fiveacc_stem, "five_acc_yes_no", clips),
        "clip_d_prompt": load_clipd(csv_dir / clipd_csv, clipd_method, clips),
    }
    return {m: float(np.mean(list(v.values()))) for m, v in per.items()}


def vp_counterpart(combo: str) -> Tuple[str, str]:
    """(source, stem combo) of the VP arm a text-only combo is compared with."""
    metric, n, regime = parse_combo(combo)
    if n == 15:
        return "R35", f"{metric}_{'first2' if regime == 'tgt09' else 'unblended'}"
    return "R38", combo


def gain(pt: Dict[str, float], curve: Optional[List[Dict[str, object]]], x: str) -> Tuple[float, str]:
    ref, flag = interp_at(curve, pt[Y_METRIC], x)
    return pt[x] - ref, flag


# ----------------------------------------------------------------------- figures ------
def plot(key: str, curves: Dict[str, Dict[int, List[Dict[str, object]]]],
         arms: Dict[str, Dict[str, Dict[str, float]]], combos: Sequence[str],
         matched_n: Optional[int], fig_dir: Path, suffix: str) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, modality, title in zip(axes, ("vp", "text"), ("VP (anchor frame + text)", "Text only (no VP)")):
        ax.set_facecolor(SURFACE)
        c15 = curves[modality][15]
        ax.plot([p[x_metric] for p in c15], [p[Y_METRIC] for p in c15], color=REF_COLOR, lw=2,
                marker="o", ms=5, zorder=2, label="uniform ρ, N=15")
        for p, rho in zip(c15, RHOS):
            ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                        textcoords="offset points", fontsize=7, color=MUTED, ha="right", va="bottom")
        if matched_n is not None:
            ce = curves[modality][matched_n]
            ax.plot([p[x_metric] for p in ce], [p[Y_METRIC] for p in ce], color=EQ_COLOR, lw=2,
                    marker="s", ms=5, zorder=2.2, label=f"uniform ρ, N={matched_n} (equal NFE)")
            # rho is not comparable across N (blend_power acts per step): label the ends only
            # Read each end point's own rho from its label, not RHOS[0]/[-1]: the
            # equal-NFE curve may be a reduced subset (e.g. R44's text side at 4 rhos).
            for p in (ce[0], ce[-1]):
                rho = int(p["label"].split("=")[1])
                ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(5, -9),
                            textcoords="offset points", fontsize=7, color=MUTED, ha="left")
        for c in combos:
            metric, n, regime = parse_combo(c)
            p = arms[modality][c]
            col = ARM_COLOR[metric]
            filled = n == 15
            ax.scatter([p[x_metric]], [p[Y_METRIC]], s=90, zorder=3, marker=REGIME_MARKER[regime],
                       facecolors=col if filled else SURFACE, edgecolors=SURFACE if filled else col,
                       linewidths=2, label=f"{metric.upper()} · {REGIME_LABEL[regime]} · draft N={n}")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel(x_label, color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    axes[0].set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    head = ("INTERIM -- no equal-NFE reference (N=15 curves only)" if matched_n is None
            else f"headline reference: equal NFE, N={matched_n}")
    fig.suptitle(f"R44 routing vs uniform ρ, VP | text only — full FiVE-Bench ({N_CLIPS} pairs, "
                 f"per-clip mean) — {head}", fontsize=10, color=INK)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{fname}{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------- tables -------
def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Build all three figures.")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+", help="Build only these figures.")
    ap.add_argument("--matched_n", type=int, choices=(20, 30),
                    help="Equal-NFE N (= 15 + N_d). REQUIRED unless --interim.")
    ap.add_argument("--interim", action="store_true",
                    help="N=15 curves only; writes *_interim outputs, never the final names.")
    ap.add_argument("--text_matched_rhos", type=int, nargs="+", default=None,
                    help="Rhos actually present in R44's own equal-NFE text-only sweep, "
                         "if it was run at fewer than the standard 8 (e.g. to save compute "
                         "once higher NFE was judged unlikely to help). Validates/loads "
                         "r44_rho_sweep_n{N}.csv against this set instead of the full 8. "
                         "Does NOT affect the VP side (R39's curve, always the full 8). "
                         "Keep the grid's endpoints (2 and 50) when subsetting -- interp_at() "
                         "never extrapolates, so dropping an endpoint risks outside_range "
                         "for any point beyond the remaining span.")
    ap.add_argument("--selection", type=Path, default=REPO / "evaluation" / "r44_selected_combos.txt")
    ap.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark").expanduser())
    ap.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    args = ap.parse_args(argv)
    if not (args.all or args.x):
        ap.error("pass --all or --x")
    if args.interim == (args.matched_n is not None):
        ap.error("equal NFE is mandatory: pass --matched_n {20,30} for the final run, "
                 "or --interim (and no --matched_n) for the N=15-only early look")
    keys = list(FIGURES) if args.all else args.x
    suffix = "_interim" if args.interim else ""
    nn = args.matched_n

    combos = read_selection(args.selection)
    if nn is not None:
        bad = [c for c in combos if 15 + parse_combo(c)[1] != nn]
        if bad:
            raise SystemExit(f"[r44_score] --matched_n {nn} is not 15 + N_d for {bad}: equal NFE "
                             f"needs N = 15 + N_d for every selected combo (n5 -> 20, n15 -> 30)")

    clips = clip_index(args.data_root)
    d = args.csv_dir
    curves: Dict[str, Dict[int, List[Dict[str, object]]]] = {
        "vp": {15: load_curve(d / "r36_rho_sweep.csv", "R36 VP curve, N=15")},
        "text": {15: load_curve(d / "r44_rho_sweep_n15.csv", "R44 text-only curve, N=15")},
    }
    if nn is not None:
        curves["vp"][nn] = load_curve(d / f"r39_rho_sweep_n{nn}.csv", f"R39 VP curve, N={nn}")
        text_rhos = args.text_matched_rhos if args.text_matched_rhos is not None else RHOS
        curves["text"][nn] = load_curve(d / f"r44_rho_sweep_n{nn}.csv", f"R44 text-only curve, N={nn}",
                                        expected_rhos=text_rhos)

    arms: Dict[str, Dict[str, Dict[str, float]]] = {"vp": {}, "text": {}}
    vp_src: Dict[str, Tuple[str, str]] = {}
    for c in combos:
        arms["text"][c] = arm_point(d, clips, f"r44_{c}", f"r44_fiveacc_{c}",
                                    "r44_clip_directional_arms.csv", f"r44_{c}")
        src, stem = vp_src[c] = vp_counterpart(c)
        pre = src.lower()
        arms["vp"][c] = arm_point(d, clips, f"{pre}_{stem}", f"{pre}_fiveacc_{stem}",
                                  f"{pre}_clip_directional.csv", f"{pre}_{stem}")

    # ---- r44_arms: every plotted point
    point_rows: List[Dict[str, object]] = []
    for modality, src in (("vp", "R36/R39"), ("text", "R44")):
        for n, curve in sorted(curves[modality].items()):
            # Each point's own rho, from its label -- not zip(curve, RHOS): a reduced
            # curve (fewer than 8 points) would silently pair with the wrong rhos.
            for p in curve:
                rho = p["label"].split("=")[1]
                point_rows.append({"modality": modality, "kind": "uniform", "source": src,
                                   "label": f"N={n} rho={rho}", "N": n, "n_clips": p["n_clips"],
                                   **{m: p[m] for m in (*X_METRICS, Y_METRIC)}})
    for c in combos:
        n_d = parse_combo(c)[1]
        for modality in ("vp", "text"):
            src = f"{vp_src[c][0]} {vp_src[c][1]}" if modality == "vp" else f"R44 r44_{c}"
            point_rows.append({"modality": modality, "kind": "routed", "source": src, "label": c,
                               "N": f"15+{n_d}", "n_clips": N_CLIPS,
                               **{m: arms[modality][c][m] for m in (*X_METRICS, Y_METRIC)}})
    write_csv(args.csv_dir / f"r44_arms{suffix}.csv", point_rows)

    # ---- r44_routing_gain: one row per (combo, x metric); equal NFE first when present
    gain_rows: List[Dict[str, object]] = []
    for c in combos:
        n_d = parse_combo(c)[1]
        for x in X_METRICS:
            row: Dict[str, object] = {"combo": c, "N_d": n_d, "vp_source": "_".join(vp_src[c]),
                                      "x_metric": x,
                                      "text_lpips": arms["text"][c][Y_METRIC], "text_x": arms["text"][c][x],
                                      "vp_lpips": arms["vp"][c][Y_METRIC], "vp_x": arms["vp"][c][x]}
            for tag, n in ((("eqnfe", nn),) if nn is not None else ()) + (("n15", 15),):
                gt, ft = gain(arms["text"][c], curves["text"][n], x)
                gv, fv = gain(arms["vp"][c], curves["vp"][n], x)
                row.update({f"curve_N_{tag}": n,
                            f"gain_text_{tag}": gt, f"flag_text_{tag}": ft,
                            f"gain_vp_{tag}": gv, f"flag_vp_{tag}": fv,
                            f"diff_{tag}": gt - gv})
            gain_rows.append(row)
    write_csv(args.csv_dir / f"r44_routing_gain{suffix}.csv", gain_rows)

    head = "eqnfe" if nn is not None else "n15"
    for r in gain_rows:
        def f(v: object) -> str:
            return "nan" if isinstance(v, float) and math.isnan(v) else f"{v:+.4f}"
        print(f"[r44_score] {r['combo']:<20} {r['x_metric']:<29} "
              f"gain_text_{head} {f(r[f'gain_text_{head}'])} ({r[f'flag_text_{head}']})  "
              f"gain_vp_{head} {f(r[f'gain_vp_{head}'])} ({r[f'flag_vp_{head}']})  "
              f"diff {f(r[f'diff_{head}'])}")

    for key in keys:
        out = plot(key, curves, arms, combos, nn, args.fig_dir, suffix)
        print(f"[r44_score] figure -> {out}")
    print(f"[r44_score] -> {args.csv_dir / f'r44_arms{suffix}.csv'}, "
          f"{args.csv_dir / f'r44_routing_gain{suffix}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
