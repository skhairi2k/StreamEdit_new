#!/usr/bin/env python
"""R31 stage 3 -- CLIP-target vs. preservation (LPIPS, SSIM), R31's 4 continuous
per-token divergence arms overlaid on R26's constant-b SPATIAL curve, UNIFORM baseline,
and per-clip oracle frontier -- the SAME reference machinery r30_score.py draws, reused
verbatim (``CONST_BS``, ``build_join_tables``, ``load_r26_metric``, ``oracle_frontier``
imported directly from that module), so R31's point is read against the identical axes
R30's arms are. Local, no GPU.

WHY A SEPARATE SCRIPT, NOT A FLAG ON r30_score.py
--------------------------------------------------
R30's 8 arms are each swept over the SAME constant-b grid as R26's SPATIAL curve, so they
plot as a curve. R31's 4 arms are each ONE continuous per-token routing setting (no b to
sweep) -- one point per arm, not a curve -- so they are drawn as single STAR markers
(``marker="*", s=260``) to be visually distinct from R30's square-marker scalar arms in
any side-by-side, while sharing the exact same R26 reference curves/frontier underneath.

INPUT: R31's per-clip, per-edit-type CSVs -- NOT the top-level ``r31_{arm}_avg.csv``
-------------------------------------------------------------------------------------
``evaluate.py`` writes two things per arm: the real per-clip data, one file per edit type
(``edit{T}_FiVE_r31_{arm}_frame_stride8.csv``, same column layout/scale as R26's own
per-edit-type files), AND a top-level ``r31_{arm}_avg.csv`` that is supposed to be the
mean over all 6 edit types' clips but is NOT -- its numbers are off by roughly two orders
of magnitude on some columns (e.g. ``lpips_unedit_part`` reading ~196 instead of ~0.2),
a pre-existing bug in the harness's own "final averaging" step, unrelated to anything in
this pipeline and not something this script can fix from the outside. So, exactly like
``r30_score.py``'s ``load_r26_metric``, this script reads the per-edit-type per-clip
files and joins/means them itself through ``cases.json`` (``build_join_tables``), and
never touches the top-level avg CSV.

``--suffix`` lets the SAME script score either the calibrated (``budget_linear``) run's
CSVs (default, no suffix: ``edit{T}_FiVE_r31_{arm}_frame_stride8.csv``) or the
exploratory ``linear_threshold`` run's (``--suffix _lin04`` ->
``edit{T}_FiVE_r31_{arm}_lin04_frame_stride8.csv``). ``--tag`` independently controls the
output filenames so the two runs never collide (default tag = suffix, so the exploratory
run naturally writes ``r31_clip_vs_lpips_lin04.pdf`` etc. without extra flags).

Usage
-----
    # calibrated (budget_linear) run -- r31_clip_vs_lpips.pdf / r31_clip_vs_ssim.pdf
    python evaluation/r31_score.py

    # exploratory (linear_threshold) run -- r31_clip_vs_lpips_lin04.pdf / _ssim_lin04.pdf
    python evaluation/r31_score.py --suffix _lin04

RECREATED 2026-09-15 after this file was found deleted from disk by an external process
(root cause unknown, no destructive command in any tracked shell/terminal history) --
recreated from conversation context. The calibrated run's already-built
``evaluation/csv/r31_arms.csv`` / ``r31_clip_vs_lpips.pdf`` / ``r31_clip_vs_ssim.pdf``
were untouched by the deletion and are unaffected by this recreation (this script
overwrites them again with --suffix "" -- values should reproduce byte-for-byte since
the underlying avg CSVs were never touched).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import (  # noqa: E402
    CONST_BS,
    build_join_tables,
    load_r26_metric,
    load_stem_metric,
    oracle_frontier,
)

# The four R31 divergence arms (depth dropped 2026-09-11). Kept in sync with
# r31_divergence.py:ARMS / r31_stage2.sh / r31_stage3.sh / r31_eval.sh.
ARMS: Sequence[str] = ("lpips", "dino_patch", "normals", "latent")

#✨ 2026-09-22: the achievement axis is now selectable. `clip_similarity_target_image`
# (the default) preserves every previous output byte-for-byte. The FiVE-Acc axes are READ
# FROM R33, not recomputed: R31's own 9-metric eval never ran five_acc, and R33's fiveacc
# step already scored both R26's reference families and R31's own 4 arms at the same
# frame_stride 8 as every stored LPIPS/SSIM column.
#
# ⚠ COLUMN NAMES. R33's FiVE-Acc CSVs are written by evaluate.py and carry ITS raw column
# names, matched suffix-wise as `step14|five_acc_yes_no`. XM_FIVEACC maps the short axis
# label used on the CLI and in the output filenames to that raw column.
XM_FIVEACC = {"yn_acc": "five_acc_yes_no", "mc_acc": "five_acc_multi_choice"}

# ⚠ STEM. R33 stored these under `r33_fiveacc_r31_{arm}`, NOT R31's own `r31_{arm}`, so
# load_r31_percli_metric cannot serve the FiVE-Acc axis -- it globs the wrong stem. The
# promoted load_stem_metric takes a fully-resolved stem for exactly this reason.
FIVEACC_R31_STEM = "r33_fiveacc_r31_{arm}"
FIVEACC_STRIDE = 8

XSHORT = {"clip_similarity_target_image": "CLIP-target",
          "yn_acc": "FiVE-Acc(yes/no)", "mc_acc": "FiVE-Acc(multi-choice)"}
# CSV column name. The default keeps the historical short alias `clip_target` so
# re-running with no flags reproduces r31_arms.csv byte-for-byte (r30_arms.csv uses
# the same alias). The FiVE-Acc axes name the column after the metric itself.
XCOL = {"clip_similarity_target_image": "clip_target",
        "yn_acc": "yn_acc", "mc_acc": "mc_acc"}
XLABEL = {
    "clip_similarity_target_image":
        "mean clip_similarity_target_image  (higher = more achieved)",
    "yn_acc": "mean yn_acc  (higher = more achieved; 22 binary clips, steps of 1/22)",
    "mc_acc": "mean mc_acc  (higher = more achieved; 22 binary clips, steps of 1/22)",
}


def load_r31_percli_metric(csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                           name2case: Dict[Tuple[int, str], str],
                           arm: str, suffix: str, metric: str) -> Dict[str, float]:
    """{case_id: value} for one R31 arm's metric -- joined exactly like
    ``r30_score.load_r26_metric``, just against R31's own per-edit-type filename
    pattern (``edit{T}_FiVE_r31_{arm}{suffix}_frame_stride8.csv``) instead of R26's.
    See the module docstring for why this reads these files and not the top-level
    ``r31_{arm}{suffix}_avg.csv``.
    """
    pattern = str(csv_dir / f"edit*_FiVE_r31_{arm}{suffix}_frame_stride8.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[r31_score] no per-clip CSV matched {pattern} -- has "
                         f"stage3-eval (suffix={suffix!r}) been run for arm={arm!r}?")
    out: Dict[str, float] = {}
    for f in files:
        m = re.search(r"edit(\d+)_", os.path.basename(f))
        t = int(m.group(1))
        with open(f) as fh:
            for r in csv.DictReader(fh):
                col = next((k for k in r if k.endswith(f"|{metric}")), None)
                if col is None:
                    raise SystemExit(f"[r31_score] {f} has no *|{metric} column.")
                vid = idx2vid.get((t, r["file_id"]))
                if vid is None:
                    continue                      # scored row outside the 22 cases
                case_id = name2case.get((t, vid))
                if case_id is None:
                    continue
                out[case_id] = float(r[col])
    return out


def _x_r26(x_metric: str, r26_csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
           name2case: Dict[Tuple[int, str], str],
           method_fmt: str) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): x} for one R26 family, from whichever source holds `x_metric`."""
    if x_metric in XM_FIVEACC:
        # Imported lazily: r33_axis_compare pulls in scipy at module level, and the
        # default CLIP-target path has never needed it. Keeps `python r31_score.py`
        # unchanged in dependencies as well as in output.
        from r33_axis_compare import load_fiveacc_metric
        # Bare method_fmt -- load_fiveacc_metric builds the r33_fiveacc_{method} stem
        # itself, so no "r26_" prefix here.
        return load_fiveacc_metric(r26_csv_dir, idx2vid, name2case, CONST_BS,
                                   XM_FIVEACC[x_metric], method_fmt=method_fmt)
    return load_r26_metric(r26_csv_dir, idx2vid, name2case, CONST_BS, x_metric,
                           method_fmt=method_fmt)


def load_r31_arm_x(csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                   name2case: Dict[Tuple[int, str], str],
                   arm: str, suffix: str, x_metric: str) -> Dict[str, float]:
    """{case_id: x} for one R31 arm, from whichever source holds `x_metric`.

    The FiVE-Acc branch ignores `suffix`: R33 only ever scored the calibrated
    (budget_linear) run, so there is no `_lin04` FiVE-Acc to read.
    """
    if x_metric in XM_FIVEACC:
        return load_stem_metric(csv_dir, idx2vid, name2case,
                                FIVEACC_R31_STEM.format(arm=arm), FIVEACC_STRIDE,
                                XM_FIVEACC[x_metric])
    return load_r31_percli_metric(csv_dir, idx2vid, name2case, arm, suffix, x_metric)


def load_r31_metric(csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                    name2case: Dict[Tuple[int, str], str], clips: Sequence[str],
                    arms: Sequence[str], suffix: str,
                    x_metric: str = "clip_similarity_target_image"
                    ) -> Dict[str, Dict[str, float]]:
    """{arm: {<x_metric>, lpips, ssim, n_clips}}, each metric meant over `clips`.

    The achievement value keeps the generic dict key "x" regardless of which metric
    filled it; only the CSV column is named after the metric, matching r33_score.py.
    """
    out: Dict[str, Dict[str, float]] = {}
    for arm in arms:
        try:
            x = load_r31_arm_x(csv_dir, idx2vid, name2case, arm, suffix, x_metric)
            lpips = load_r31_percli_metric(csv_dir, idx2vid, name2case, arm, suffix,
                                           "lpips_unedit_part")
            ssim = load_r31_percli_metric(csv_dir, idx2vid, name2case, arm, suffix,
                                          "ssim_unedit_part")
        except SystemExit as ex:
            print(f"[r31_score] SKIP {arm}: {ex}")
            continue
        arm_clips = sorted(set(x) & set(lpips) & set(ssim))
        if len(arm_clips) != len(clips):
            print(f"[r31_score] WARNING {arm}: {len(arm_clips)}/{len(clips)} clips "
                  f"present -- scoring on the subset actually rendered.")
        if not arm_clips:
            continue
        out[arm] = {
            "x": sum(x[c] for c in arm_clips) / len(arm_clips),
            "lpips": sum(lpips[c] for c in arm_clips) / len(arm_clips),
            "ssim": sum(ssim[c] for c in arm_clips) / len(arm_clips),
            "n_clips": len(arm_clips),
        }
    return out


# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})

    # ---- R26 reference curves (identical to r30_score.py's run(), see that module for
    # the full rationale on method_fmt / the whole-frame CLIP convention / etc.)
    lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                            "lpips_unedit_part")
    clip_target = _x_r26(args.x_metric, args.r26_csv_dir, idx2vid, name2case,
                         "taubg0_taufg{b}_vp")
    ssim = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                           "ssim_unedit_part")
    uniform_lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                    "lpips_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")
    uniform_clip_target = _x_r26(args.x_metric, args.r26_csv_dir, idx2vid, name2case,
                                 "taubg{b}_taufg{b}_vp")
    uniform_ssim = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                   "ssim_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")

    clip_curve = [(sum(clip_target[(c, b)] for c in clips) / len(clips),
                  sum(lpips[(c, b)] for c in clips) / len(clips), b) for b in CONST_BS]
    uniform_curve = [(sum(uniform_clip_target[(c, b)] for c in clips) / len(clips),
                     sum(uniform_lpips[(c, b)] for c in clips) / len(clips), b)
                    for b in CONST_BS]
    ssim_curve = [(sum(clip_target[(c, b)] for c in clips) / len(clips),
                  sum(ssim[(c, b)] for c in clips) / len(clips), b) for b in CONST_BS]
    uniform_ssim_curve = [(sum(uniform_clip_target[(c, b)] for c in clips) / len(clips),
                          sum(uniform_ssim[(c, b)] for c in clips) / len(clips), b)
                         for b in CONST_BS]

    lpips_frontier = oracle_frontier(clips, clip_target, lpips, CONST_BS,
                                     n_alpha=args.n_alpha, higher_is_better=False)
    ssim_frontier = oracle_frontier(clips, clip_target, ssim, CONST_BS,
                                    n_alpha=args.n_alpha, higher_is_better=True)

    # ---- R31's own 4 arms, one point each.
    r31 = load_r31_metric(args.csv_dir, idx2vid, name2case, clips, ARMS, args.suffix,
                          args.x_metric)
    if not r31:
        raise SystemExit(f"[r31_score] no R31 arm had data under {args.csv_dir} "
                         f"(suffix={args.suffix!r}). Has stage3-eval been run?")

    out_rows: List[Dict[str, object]] = []
    for arm in ARMS:
        if arm not in r31:
            continue
        row = {"arm": arm, **r31[arm]}
        out_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    x_col = XCOL[args.x_metric]
    fieldnames = ["arm", x_col, "lpips", "ssim", "n_clips"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        # The CSV column is named via XCOL; internally the value keeps the generic
        # key "x" (r33_score.py's convention).
        w.writerows([{"arm": r["arm"], x_col: r["x"], "lpips": r["lpips"],
                      "ssim": r["ssim"], "n_clips": r["n_clips"]} for r in out_rows])
    print(f"[r31_score] wrote {args.out} ({len(out_rows)} arm(s), suffix={args.suffix!r})")

    print(f"\n{'arm':<12} {x_col:>12} {'lpips':>9} {'ssim':>9}")
    for row in out_rows:
        print(f"{row['arm']:<12} {row['x']:>12.4f} {row['lpips']:>9.5f} "
              f"{row['ssim']:>9.5f}")

    if args.fig_dir is not None:
        make_figures(args.fig_dir, out_rows, clip_curve, uniform_curve, ssim_curve,
                    uniform_ssim_curve, lpips_frontier, ssim_frontier, args.tag,
                    args.x_metric, args.fig_stem)
    return 0


# --------------------------------------------------------------------------------------
# figures: r31_clip_vs_lpips{tag}.pdf, r31_clip_vs_ssim{tag}.pdf
# --------------------------------------------------------------------------------------
def make_figures(fig_dir: Path, out_rows, clip_curve, uniform_curve, ssim_curve,
                 uniform_ssim_curve, lpips_frontier, ssim_frontier, tag: str,
                 x_metric: str = "clip_similarity_target_image",
                 fig_stem: str = "r31_clip") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    # Axis convention matches r26_tradeoff_figure.py / r30_score.py exactly: CLIP
    # (clip_similarity_target_image, whole-frame) always on x, preservation always on y.
    def draw_curve_scatter(filename, title, x_label, y_label, spatial_curve,
                           uniform_curve, y_key, frontier, y_short):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        xs = [p for p, _, _ in spatial_curve]
        ys = [c for _, c, _ in spatial_curve]
        ax.plot(xs, ys, "-o", color="black",
                label="constant-$b$ SPATIAL curve (R26, $\\tau_{bg}$=0)", zorder=3)
        for p, c, b in spatial_curve:
            ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(4, 4),
                        fontsize=8)
        uxs = [p for p, _, _ in uniform_curve]
        uys = [c for _, c, _ in uniform_curve]
        ax.plot(uxs, uys, marker="D", color="dimgray", linestyle="--",
                label="uniform baseline ($b_{bg}=b_{fg}$, Eq. 4, R26)", zorder=2)
        for p, c, b in uniform_curve:
            ax.annotate(f"$b$={b}", (p, c), textcoords="offset points",
                        xytext=(4, -10), fontsize=8, color="dimgray")

        # R31's four continuous per-token arms -- one point each, star marker to set
        # them visually apart from R30's square-marker scalar arms in any side-by-side.
        for i, row in enumerate(out_rows):
            if y_key not in row:
                continue
            ax.scatter([row["x"]], [row[y_key]], marker="*", s=260,
                      color=cmap(i % 10), edgecolor="black", linewidth=0.8, zorder=5,
                      label=f"r31_{row['arm']} ({y_key}={row[y_key]:.3f})")

        fx = [c for _, c, _ in frontier]
        fy = [y for _, _, y in frontier]
        ax.plot(fx, fy, "-", color="tab:green", linewidth=2, zorder=1.5,
                label="per-clip oracle ($\\alpha$ sweep, 8-$b$ grid)")
        ax.annotate(f"$\\alpha$=0\n({y_short})", (fx[0], fy[0]), fontsize=7,
                    color="tab:green")
        ax.annotate(f"$\\alpha$=1\n({XSHORT[x_metric]})", (fx[-1], fy[-1]), fontsize=7,
                    color="tab:green")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_path = fig_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    p1 = draw_curve_scatter(
        f"{fig_stem}_vs_lpips{tag}.pdf",
        f"R31: {XSHORT[x_metric]} vs. LPIPS -- continuous per-token divergence "
        "routing vs. R26's constant-$b$ spatial gate",
        XLABEL[x_metric],
        "mean lpips_unedit_part  (lower = better preserved)",
        clip_curve, uniform_curve, "lpips", lpips_frontier, "LPIPS")

    p2 = draw_curve_scatter(
        f"{fig_stem}_vs_ssim{tag}.pdf",
        f"R31: {XSHORT[x_metric]} vs. SSIM -- continuous per-token divergence "
        "routing vs. R26's constant-$b$ spatial gate",
        XLABEL[x_metric],
        "mean ssim_unedit_part  (higher = better preserved)",
        ssim_curve, uniform_ssim_curve, "ssim", ssim_frontier, "SSIM")

    print(f"[r31_score] wrote {p1}, {p2}")


# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"),
                   help="Where R31's r31_{arm}{suffix}_avg.csv live.")
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"),
                   help="Where R26's per-edit-type per-clip CSVs live (reference curves).")
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--suffix", type=str, default="",
                   help="Appended to 'r31_{arm}' to select which run's CSVs to read, "
                        "e.g. '_lin04' for the linear_threshold exploratory run. "
                        "Default '' reads the calibrated budget_linear run.")
    p.add_argument("--tag", type=str, default=None,
                   help="Appended to output filenames ({fig_stem}_vs_lpips{tag}.pdf). "
                        "Defaults to --suffix, so the two runs never collide.")
    p.add_argument("--x_metric", type=str, default="clip_similarity_target_image",
                   choices=("clip_similarity_target_image", "yn_acc", "mc_acc"),
                   help="Achievement axis. Default reproduces every previous output "
                        "byte-for-byte. yn_acc / mc_acc are READ from R33's FiVE-Acc "
                        "CSVs (not recomputed) -- pass a distinct --fig_stem and -o "
                        "for those, or the default r31_arms.csv / r31_clip_vs_*.pdf "
                        "are overwritten.")
    p.add_argument("--fig_stem", type=str, default="r31_clip",
                   help="Figures are {fig_stem}_vs_lpips{tag}.pdf / _vs_ssim{tag}.pdf. "
                        "Default keeps the existing r31_clip_vs_*.pdf filenames.")
    p.add_argument("--n_alpha", type=int, default=21)
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Defaults to evaluation/csv/r31_arms{suffix}.csv")
    p.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    args = p.parse_args(argv)
    if args.tag is None:
        args.tag = args.suffix
    if args.out is None:
        args.out = Path(f"evaluation/csv/r31_arms{args.suffix}.csv")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
