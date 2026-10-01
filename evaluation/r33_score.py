#!/usr/bin/env python
"""R33 -- aggregate (mean-over-22-clip) CLIP-D-vs-LPIPS and CLIP-D-vs-SSIM trade-off
figures for ALL 28 arms R33's CLIP-D pass scored (R26's 16 constant-b + R30's 8 routed +
R31's 4 divergence arms), superseding r30_clip_vs_lpips.pdf / r31_clip_vs_lpips.pdf on
the corrected achievement axis.

WHY THIS EXISTS. r30_score.py and r31_score.py each already draw R26's constant-b
SPATIAL curve + UNIFORM baseline + per-clip oracle frontier, with their own task's arms
overlaid (R30's 8 as SQUARE markers, R31's 4 as STAR markers) -- but both read the
x-axis from clip_similarity_target_image, which R33 measured does not rank renders by
edit strength on this case set (Spearman(b, clip_target) = +0.098 mean, 11/22 positive).
R33's own axis-compare step chose clip_d_prompt as the replacement (+0.408 mean, 17/22
positive). This script is the SAME two figures, x swapped to clip_d_prompt, with BOTH
tasks' arms drawn together on one shared, corrected axis rather than two separate
figures each still using the broken one.

y-AXES (lpips_unedit_part / ssim_unedit_part) ARE UNCHANGED -- R33 only disputes the
achievement axis, not the preservation one. R26's curves and R31's arms are read exactly
as r30_score.py / r31_score.py already do (same loaders, unmodified); R30's arms are
read from evaluation/csv/r30_arms.csv, which already has per-arm mean lpips/ssim over
the 22 cases -- only its own clip_target column is swapped out, for clip_d_prompt from
r33_clip_directional.csv.

Usage
-----
    python evaluation/r33_score.py \\
        -o evaluation/csv/r33_arms.csv \\
        --fig_dir evaluation/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import (  # noqa: E402
    ARMS as R30_ARMS,
    CONST_BS,
    build_join_tables,
    load_r26_metric,
    load_stem_metric,
    oracle_frontier,
)
from r31_score import ARMS as R31_ARMS, load_r31_percli_metric  # noqa: E402
from r33_axis_compare import load_clipd_metric, load_fiveacc_metric  # noqa: E402


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals)


#✨ R33-subset work 2026-09-22: the achievement axis is now selectable. `clip_d_prompt`
# (the default) preserves every previous output byte-for-byte; `clip_similarity_target_image`
# draws the same 28 arms on the INCUMBENT axis, so the two figures differ only in x and
# the comparison between them is like-for-like. Internally the x value keeps the dict key
# "x" regardless of which metric filled it; only the CSV column is named after the metric.
XM_CLIPD = ("clip_d_prompt", "clip_d_word")

#✨ 2026-09-22: the FiVE-Acc axes. ⚠ THE COLUMN NAME DIFFERS BY SOURCE. R26's and R31's
# FiVE-Acc CSVs are written by evaluate.py and carry ITS raw column names
# (five_acc_yes_no / five_acc_multi_choice / five_acc, matched suffix-wise as
# `step14|five_acc_yes_no`), while r30_fiveacc_arms.csv -- written by R30's own stage-2
# aggregation -- carries the short labels verbatim (yn_acc/mc_acc/union/inter) and has NO
# precomputed five_acc column at all (see load_r30_arms's five_acc branch, which computes
# it there instead). So XM_FIVEACC maps label -> raw column for the R26 and R31 loaders
# ONLY; load_r30_arms must keep using its own per-axis logic, and applying this map there
# raises "no *|five_acc_yes_no column" on r30_fiveacc_arms.csv.
XM_FIVEACC = {"yn_acc": "five_acc_yes_no", "mc_acc": "five_acc_multi_choice",
              "five_acc": "five_acc"}

# five_acc (added 2026-09-22) is evaluate.py's own combined verdict, verified against
# its source (evaluation/fivebench/evaluate.py, ~line 519-527) rather than assumed:
# union = int(yn_acc or mc_acc), inter = int(yn_acc and mc_acc),
# five_acc = mean([yn_acc, mc_acc, union, inter]) = (yn+mc+union+inter)/4 exactly.
# BEFORE THIS AXIS EXISTED, --fig_stem r33_fiveacc (the bare, unqualified name) was used
# for the yn_acc run -- misleading, since "FiVE-Acc" is the whole 4-column family, not
# just its yes/no column. That run is now --fig_stem r33_fiveacc_yn; the bare
# r33_fiveacc_vs_*.pdf / r33_arms_fiveacc.csv names are freed up for what they should
# have meant all along: this combined axis.
XSHORT = {"clip_d_prompt": "CLIP-D", "clip_d_word": "CLIP-D(word)",
          "clip_similarity_target_image": "CLIP-target",
          "yn_acc": "FiVE-Acc(yes/no)", "mc_acc": "FiVE-Acc(multi-choice)",
          "five_acc": "FiVE-Acc(combined)"}
XNOTE = {"clip_d_prompt": "the corrected achievement axis",
         "clip_d_word": "the corrected achievement axis (word pair)",
         "clip_similarity_target_image":
             "the INCUMBENT axis -- R33 measured it does NOT rank b (Spearman +0.098)",
         "yn_acc":
             "FiVE-Bench's own directional axis -- monotone in b, but quantised to 1/22",
         "mc_acc":
             "FiVE-Bench's multi-choice axis -- SATURATES from b=4, non-monotone on "
             "UNIFORM; read as a null result, not a curve",
         "five_acc":
             "FiVE-Bench's combined axis, (yn+mc+union+inter)/4 -- at least as coarse as "
             "yn_acc alone (only 3 achievable per-clip values: 0, 0.5, 1.0)"}
XLABEL = {
    "clip_d_prompt": "mean clip_d_prompt  (higher = more achieved, directional CLIP)",
    "clip_d_word": "mean clip_d_word  (higher = more achieved, directional CLIP)",
    "clip_similarity_target_image":
        "mean clip_similarity_target_image  (higher = more achieved; INCUMBENT axis)",
    "yn_acc": "mean yn_acc  (higher = more achieved; 22 binary clips, steps of 1/22)",
    "mc_acc": "mean mc_acc  (higher = more achieved; 22 binary clips, steps of 1/22)",
    "five_acc": "mean five_acc  (higher = more achieved; (yn+mc+union+inter)/4)",
}


def _x_r26(x_metric: str, r26_csv_dir: Path, clipd_csv: Path, idx2vid, name2case,
           method_fmt: str) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): x} for one R26 family, from whichever source holds `x_metric`."""
    if x_metric in XM_CLIPD:
        return load_clipd_metric(clipd_csv, x_metric, method_fmt="r26_" + method_fmt)
    if x_metric in XM_FIVEACC:
        # Bare method_fmt, NOT the "r26_" prefix the CLIP-D branch prepends:
        # load_fiveacc_metric builds the r33_fiveacc_{method} stem itself.
        return load_fiveacc_metric(r26_csv_dir, idx2vid, name2case, CONST_BS,
                                   XM_FIVEACC[x_metric], method_fmt=method_fmt)
    return load_r26_metric(r26_csv_dir, idx2vid, name2case, CONST_BS, x_metric,
                           method_fmt=method_fmt)


def load_r30_arms(r30_arms_csv: Path, clipd_csv: Path, clips: Sequence[str],
                  r30_perclip_csv: Path,
                  x_metric: str = "clip_d_prompt") -> List[Dict[str, object]]:
    """{arm, source='r30', clip_d_prompt, lpips, ssim, n_clips} for R30's 8 arms.

    ⚠️ FIXED 2026-09-22 -- this function used to IGNORE `clips`. lpips/ssim were read
    straight out of r30_arms.csv, which stores means already aggregated over all 22
    cases, and the clip_d_prompt loop appended every `r30_*` row without checking
    case_id. On the full 22-clip cases.json that is a no-op, so R33's published numbers
    are unaffected -- but under ANY subset the eight R30 arms silently stayed at their
    22-clip values while every other family moved, producing a figure that mixed two
    different clip sets on one pair of axes. Now every metric is averaged per-clip over
    exactly `clips`: lpips/ssim from r30_fiveacc_arms.csv (which carries the per-clip
    rows, 8 arms x 22 clips), clip_d_prompt from the CLIP-D table, both filtered by
    case_id. `r30_arms_csv` is still read, but only to decide which arms exist.
    """
    with open(r30_arms_csv) as fh:
        r30_rows = {r["arm"]: r for r in csv.DictReader(fh)}

    keep = set(clips)
    clipd_by_arm: Dict[str, List[float]] = {}
    if x_metric in XM_CLIPD:
        with open(clipd_csv) as fh:
            for r in csv.DictReader(fh):
                if r["method"].startswith("r30_") and r["case_id"] in keep:
                    clipd_by_arm.setdefault(r["method"][len("r30_"):], []).append(
                        float(r[x_metric]))
    elif x_metric == "five_acc":
        # r30_fiveacc_arms.csv has no precomputed five_acc column (unlike R26/R31's own
        # CSVs, where evaluate.py writes one directly) -- it has yn_acc/mc_acc/union/inter,
        # the same four inputs evaluate.py itself averages, so compute it here instead of
        # requiring a rescore. Same formula verified against evaluate.py's own source (see
        # the XM_FIVEACC comment above): (yn_acc + mc_acc + union + inter) / 4.
        with open(r30_perclip_csv) as fh:
            for r in csv.DictReader(fh):
                if r["case_id"] in keep:
                    five = (float(r["yn_acc"]) + float(r["mc_acc"])
                           + float(r["union"]) + float(r["inter"])) / 4
                    clipd_by_arm.setdefault(r["arm"], []).append(five)
    else:
        # The incumbent CLIP column lives per-clip in r30_fiveacc_arms.csv, same file
        # the lpips/ssim fix below reads -- so both axes are filtered identically.
        with open(r30_perclip_csv) as fh:
            for r in csv.DictReader(fh):
                if r["case_id"] in keep:
                    clipd_by_arm.setdefault(r["arm"], []).append(float(r[x_metric]))

    # Per-clip lpips/ssim, so a subset of `clips` actually changes these numbers.
    perclip: Dict[str, Dict[str, List[float]]] = {}
    with open(r30_perclip_csv) as fh:
        for r in csv.DictReader(fh):
            if r["case_id"] not in keep:
                continue
            d = perclip.setdefault(r["arm"], {"lpips": [], "ssim": []})
            d["lpips"].append(float(r["lpips_unedit_part"]))
            d["ssim"].append(float(r["ssim_unedit_part"]))

    out: List[Dict[str, object]] = []
    for arm in R30_ARMS:
        if arm not in r30_rows:
            print(f"[r33_score] SKIP r30_{arm} -- not in {r30_arms_csv}")
            continue
        if arm not in clipd_by_arm:
            print(f"[r33_score] SKIP r30_{arm} -- no {x_metric} rows for it")
            continue
        if arm not in perclip:
            print(f"[r33_score] SKIP r30_{arm} -- no per-clip rows in {r30_perclip_csv}")
            continue
        out.append({
            "arm": f"r30_{arm}", "source": "r30",
            "x": round(_mean(clipd_by_arm[arm]), 6),
            "lpips": round(_mean(perclip[arm]["lpips"]), 6),
            "ssim": round(_mean(perclip[arm]["ssim"]), 6),
            "n_clips": len(clipd_by_arm[arm]),
        })
    return out


def load_r31_arms(csv_dir: Path, clipd_csv: Path,
                  idx2vid: Dict[Tuple[int, str], str],
                  name2case: Dict[Tuple[int, str], str], clips: Sequence[str],
                  x_metric: str = "clip_d_prompt") -> List[Dict[str, object]]:
    """{arm, source='r31', clip_d_prompt, lpips, ssim, n_clips} for R31's 4 arms.

    lpips/ssim via r31_score.load_r31_percli_metric, unmodified (R31's own per-clip
    per-edit-type CSVs); clip_d_prompt from r33_clip_directional.csv's `r31_{arm}` rows.
    """
    clipd_by_arm: Dict[str, Dict[str, float]] = {}
    if x_metric in XM_CLIPD:
        with open(clipd_csv) as fh:
            for r in csv.DictReader(fh):
                if r["method"].startswith("r31_"):
                    arm = r["method"][len("r31_"):]
                    clipd_by_arm.setdefault(arm, {})[r["case_id"]] = float(r[x_metric])
    elif x_metric in XM_FIVEACC:
        # load_r31_percli_metric cannot serve this: it globs the r31_{arm} stem, and
        # R33's FiVE-Acc for these same arms lives under r33_fiveacc_r31_{arm}.
        for arm in R31_ARMS:
            try:
                clipd_by_arm[arm] = load_stem_metric(csv_dir, idx2vid, name2case,
                                                     f"r33_fiveacc_r31_{arm}", 8,
                                                     XM_FIVEACC[x_metric])
            except SystemExit as ex:
                print(f"[r33_score] SKIP r31_{arm}: {ex}")
    else:
        for arm in R31_ARMS:
            try:
                clipd_by_arm[arm] = load_r31_percli_metric(csv_dir, idx2vid, name2case,
                                                           arm, "", x_metric)
            except SystemExit as ex:
                print(f"[r33_score] SKIP r31_{arm}: {ex}")

    out: List[Dict[str, object]] = []
    for arm in R31_ARMS:
        if arm not in clipd_by_arm:
            print(f"[r33_score] SKIP r31_{arm} -- no {x_metric} rows for it")
            continue
        try:
            lpips = load_r31_percli_metric(csv_dir, idx2vid, name2case, arm, "",
                                           "lpips_unedit_part")
            ssim = load_r31_percli_metric(csv_dir, idx2vid, name2case, arm, "",
                                          "ssim_unedit_part")
        except SystemExit as ex:
            print(f"[r33_score] SKIP r31_{arm}: {ex}")
            continue
        common = sorted(set(clipd_by_arm[arm]) & set(lpips) & set(ssim)
                        & set(clips))
        if not common:
            continue
        out.append({
            "arm": f"r31_{arm}", "source": "r31",
            "x": round(_mean([clipd_by_arm[arm][c] for c in common]), 6),
            "lpips": round(_mean([lpips[c] for c in common]), 6),
            "ssim": round(_mean([ssim[c] for c in common]), 6),
            "n_clips": len(common),
        })
    return out


def load_r26_arms(r26_csv_dir: Path, clipd_csv: Path,
                  idx2vid: Dict[Tuple[int, str], str],
                  name2case: Dict[Tuple[int, str], str], clips: Sequence[str],
                  x_metric: str = "clip_d_prompt") -> List[Dict[str, object]]:
    """{arm, source='r26', x, lpips, ssim, n_clips} for R26's 16
    constant-b arms (8 SPATIAL taubg0_taufg{b}_vp + 8 UNIFORM taubg{b}_taufg{b}_vp),
    one row per (family, b) -- these are also just "arms" on R33's shared axis, not
    only the reference curve.
    """
    out: List[Dict[str, object]] = []
    for family, method_fmt in (("spatial", "taubg0_taufg{b}_vp"),
                               ("uniform", "taubg{b}_taufg{b}_vp")):
        lpips = load_r26_metric(r26_csv_dir, idx2vid, name2case, CONST_BS,
                                "lpips_unedit_part", method_fmt=method_fmt)
        ssim = load_r26_metric(r26_csv_dir, idx2vid, name2case, CONST_BS,
                               "ssim_unedit_part", method_fmt=method_fmt)
        cd = _x_r26(x_metric, r26_csv_dir, clipd_csv, idx2vid, name2case, method_fmt)
        for b in CONST_BS:
            common = sorted(c for c in clips
                            if (c, b) in lpips and (c, b) in ssim and (c, b) in cd)
            if not common:
                continue
            method = method_fmt.format(b=b)
            out.append({
                "arm": f"r26_{method}", "source": f"r26_{family}",
                "x": round(_mean([cd[(c, b)] for c in common]), 6),
                "lpips": round(_mean([lpips[(c, b)] for c in common]), 6),
                "ssim": round(_mean([ssim[(c, b)] for c in common]), 6),
                "n_clips": len(common),
            })
    return out


# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})

    # ---- R26 reference curves, x swapped clip_target -> clip_d_prompt.
    lpips_sp = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                               "lpips_unedit_part")
    lpips_un = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                               "lpips_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")
    ssim_sp = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                              "ssim_unedit_part")
    ssim_un = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                              "ssim_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")
    cd_sp = _x_r26(args.x_metric, args.r26_csv_dir, args.clipd_csv, idx2vid, name2case,
                   "taubg0_taufg{b}_vp")
    cd_un = _x_r26(args.x_metric, args.r26_csv_dir, args.clipd_csv, idx2vid, name2case,
                   "taubg{b}_taufg{b}_vp")

    clip_curve = [(_mean([cd_sp[(c, b)] for c in clips]),
                  _mean([lpips_sp[(c, b)] for c in clips]), b) for b in CONST_BS]
    uniform_curve = [(_mean([cd_un[(c, b)] for c in clips]),
                     _mean([lpips_un[(c, b)] for c in clips]), b) for b in CONST_BS]
    ssim_curve = [(_mean([cd_sp[(c, b)] for c in clips]),
                  _mean([ssim_sp[(c, b)] for c in clips]), b) for b in CONST_BS]
    uniform_ssim_curve = [(_mean([cd_un[(c, b)] for c in clips]),
                          _mean([ssim_un[(c, b)] for c in clips]), b) for b in CONST_BS]

    lpips_frontier = oracle_frontier(clips, cd_sp, lpips_sp, CONST_BS,
                                     n_alpha=args.n_alpha, higher_is_better=False)
    ssim_frontier = oracle_frontier(clips, cd_sp, ssim_sp, CONST_BS,
                                    n_alpha=args.n_alpha, higher_is_better=True)

    # ---- All 28 arms as rows (R26's 16 + R30's 8 + R31's 4).
    out_rows = (load_r26_arms(args.r26_csv_dir, args.clipd_csv, idx2vid, name2case, clips,
                              args.x_metric)
               + load_r30_arms(args.r30_arms_csv, args.clipd_csv, clips,
                               args.r30_perclip_csv, args.x_metric)
               + load_r31_arms(args.csv_dir, args.clipd_csv, idx2vid, name2case, clips,
                               args.x_metric))
    if not out_rows:
        raise SystemExit("[r33_score] no arm had data. Nothing to score.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The CSV column is named after the metric that filled it, so a clip_target run and
    # a clip_d run are never mistaken for each other downstream.
    for r in out_rows:
        r[args.x_metric] = r.pop("x")
    fieldnames = ["arm", "source", args.x_metric, "lpips", "ssim", "n_clips"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[r33_score] wrote {args.out} ({len(out_rows)} arm(s))")

    r30_rows = [r for r in out_rows if r["source"] == "r30"]
    r31_rows = [r for r in out_rows if r["source"] == "r31"]
    if args.fig_dir is not None:
        make_figures(args.fig_dir, r30_rows, r31_rows, clip_curve, uniform_curve,
                    ssim_curve, uniform_ssim_curve, lpips_frontier, ssim_frontier,
                    args.x_metric, args.fig_stem)
    return 0


# --------------------------------------------------------------------------------------
# figures: {fig_stem}_vs_lpips.pdf, {fig_stem}_vs_ssim.pdf
# --------------------------------------------------------------------------------------
def make_figures(fig_dir: Path, r30_rows, r31_rows, clip_curve, uniform_curve,
                 ssim_curve, uniform_ssim_curve, lpips_frontier, ssim_frontier,
                 x_metric: str = "clip_d_prompt",
                 fig_stem: str = "r33_clip") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    # x-axis convention: clip_d_prompt (R33's chosen achievement axis) always on x,
    # preservation always on y -- same convention r26_tradeoff_figure.py / r30_score.py
    # / r31_score.py already use, just with the corrected x metric.
    def draw_curve_scatter(filename, title, x_label, y_label, spatial_curve,
                           uniform_curve, y_key, frontier, y_short):
        fig, ax = plt.subplots(figsize=(7.5, 6))
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

        # R30's 8 static-divergence-routed arms -- SQUARE markers, r30_score.py's
        # convention, to stay visually distinct from R31's STAR markers below.
        for i, row in enumerate(r30_rows):
            if y_key not in row:
                continue
            ax.scatter([row[x_metric]], [row[y_key]], marker="s", s=70,
                      color=cmap(i % 10), edgecolor="black", zorder=4,
                      label=f"{row['arm']} ({y_key}={row[y_key]:.3f})")

        # R31's 4 continuous per-token arms -- STAR markers, r31_score.py's convention.
        # Colour index offset from R30's so the same tab10 colour is not reused between
        # a square and a star that are unrelated arms.
        for i, row in enumerate(r31_rows):
            if y_key not in row:
                continue
            ax.scatter([row[x_metric]], [row[y_key]], marker="*", s=260,
                      color=cmap((i + 4) % 10), edgecolor="black", linewidth=0.8,
                      zorder=5, label=f"{row['arm']} ({y_key}={row[y_key]:.3f})")

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
        # Wrapped, not left as one line: savefig(bbox_inches="tight") expands the
        # CANVAS to fit whatever the title needs, so an axis with a long XNOTE (e.g.
        # five_acc's coarseness caveat) silently produced a much WIDER PDF page than
        # CLIP-D's short one -- same plot height, ~2x the width, which then read as a
        # visibly SHORTER figure once displayed at matching width in the HTML report.
        # Wrapping keeps every axis's output page at a consistent, comparable aspect
        # ratio; width=72 keeps CLIP-D's own (already-short) title on one line.
        ax.set_title(textwrap.fill(title, width=72), fontsize=10)
        ax.legend(fontsize=6.5, loc="best", ncol=1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out_path = fig_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    p1 = draw_curve_scatter(
        f"{fig_stem}_vs_lpips.pdf",
        f"R33: {XSHORT[x_metric]} vs. LPIPS -- all 28 arms on {XNOTE[x_metric]}",
        XLABEL[x_metric],
        "mean lpips_unedit_part  (lower = better preserved)",
        clip_curve, uniform_curve, "lpips", lpips_frontier, "LPIPS")

    p2 = draw_curve_scatter(
        f"{fig_stem}_vs_ssim.pdf",
        f"R33: {XSHORT[x_metric]} vs. SSIM -- all 28 arms on {XNOTE[x_metric]}",
        XLABEL[x_metric],
        "mean ssim_unedit_part  (higher = better preserved)",
        ssim_curve, uniform_ssim_curve, "ssim", ssim_frontier, "SSIM")

    print(f"[r33_score] wrote {p1}, {p2}")


# --------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--r30_arms_csv", type=Path,
                   default=Path("evaluation/csv/r30_arms.csv"))
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"),
                   help="Where R31's per-clip per-edit-type CSVs live.")
    p.add_argument("--clipd_csv", type=Path,
                   default=Path("evaluation/csv/r33_clip_directional.csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--x_metric", type=str, default="clip_d_prompt",
                   choices=("clip_d_prompt", "clip_d_word",
                            "clip_similarity_target_image", "yn_acc", "mc_acc",
                            "five_acc"),
                   help="Achievement axis. Default reproduces every previous output. "
                        "yn_acc / mc_acc / five_acc draw the same 28 arms on "
                        "FiVE-Bench's own accuracy axes (five_acc is the combined "
                        "(yn+mc+union+inter)/4 verdict) -- pass a distinct --fig_stem "
                        "and -o for those, or the default r33_arms.csv / "
                        "r33_clip_vs_*.pdf are overwritten.")
    p.add_argument("--fig_stem", type=str, default="r33_clip",
                   help="Figures are {fig_stem}_vs_lpips.pdf / _vs_ssim.pdf. The default "
                        "keeps the existing r33_clip_vs_*.pdf filenames.")
    p.add_argument("--r30_perclip_csv", type=Path,
                   default=Path("evaluation/csv/r30_fiveacc_arms.csv"),
                   help="Per-clip lpips/ssim for R30's arms, so --cases subsets "
                        "actually filter them (see load_r30_arms).")
    p.add_argument("--n_alpha", type=int, default=21)
    p.add_argument("-o", "--out", type=Path, default=Path("evaluation/csv/r33_arms.csv"))
    p.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
