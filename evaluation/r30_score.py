#!/usr/bin/env python
"""R30 stage 2 -- CLIP-target vs. preservation (LPIPS, SSIM) for the constant-b SPATIAL
curve, the UNIFORM baseline, the eight rendered divergence arms, and a per-clip oracle
frontier. Local, no GPU. Writes evaluation/csv/r30_arms.csv, r30_clip_vs_lpips.pdf and
r30_clip_vs_ssim.pdf.

⚠ 2026-09-11: this script previously also built a discrete FiVE-Acc (yn_acc + mc_acc)
"operating curve" against a single-point per-clip oracle (smallest grid b reaching each
clip's own accuracy max), rated every arm against it by Pareto dominance / vertical gap /
Spearman(d, b*) / McNemar's exact test, and wrote r30_curve.pdf + r30_b_scatter.pdf. That
whole analysis is REMOVED (dropped as unsatisfying relative to the simpler continuous
frontier below -- N/44's integer counts also gave McNemar b+c=0 for every arm, i.e. no
significance test was ever computable on this data). The two files it wrote are deleted;
nothing in this script produces them any more. See the plan doc's own note for the
historical record of what that analysis found before removal.

ORACLE FRONTIER (replaces the old single-point oracle). Following r26_tradeoff_figure.py's
`oracle_frontier` convention exactly, just adapted from R26's 2-D (tau_bg, tau_fg) grid to
R30's 1-D grid of 8 constant b values: for alpha swept over [0, 1], each clip
INDEPENDENTLY argmaxes `alpha*CLIP_norm + (1-alpha)*Y_norm` over its own 8 b's (CLIP and Y
each min-max normalized PER CLIP across those 8 values -- never a global scale, so one
clip's wider raw range can't dominate every clip's argmax), and the winning b's RAW (CLIP,
Y) is recorded; the frontier point for that alpha is the mean of those raw values across
all 22 clips. Y is direction-corrected (`higher_is_better`): LPIPS normalizes as
`1 - minmax(raw)` so "higher normalized" means "more preferred" for both terms before they
are summed, exactly like R26. This is the ceiling an all-knowing per-clip oracle could
reach on the 8-point grid -- ties break on higher CLIP then better Y, matching R26 bit for
bit. No non-monotone exclusion is needed here (unlike the removed discrete oracle): this
frontier never touches yn_acc/mc_acc, only continuous per-clip metrics, so there is no VLM
yes/no sequence to be non-monotone in.

JOIN. R26's per-clip metric CSVs (edit{T}_FiVE_r26_taubg0_taufg{B}_vp_frame_stride8.csv) do
NOT carry case_id, so this script joins them itself: file_id (position in the FULL
edit{T}_FiVE.json, not the 22-case subset) -> video_name -> case_id, via evaluation/
cases.json -- a video can appear under two edit types with different case_ids
(0011_lucia in edit2 AND edit5), so the join must go through the annotation file's own
order, not video_name alone.

r30_fiveacc_arms.csv (from stage2-eval) supplies each rendered arm's own per-clip
clip_similarity_target_image / lpips_unedit_part / ssim_unedit_part -- read via
`load_fiveacc_arms`, which returns every stored column per row (including yn_acc, mc_acc,
and the routed b) even though this script itself now only consumes the three continuous
metrics; the accuracy/routing columns remain in the CSV for other uses (e.g. building the
per-arm reference video grids) even though no analysis here depends on them any more.

Usage
-----
    python evaluation/r30_score.py \\
        --fiveacc_arms evaluation/csv/r30_fiveacc_arms.csv \\
        --r26_csv_dir evaluation/csv \\
        -o evaluation/csv/r30_arms.csv \\
        --fig_dir evaluation/figures
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

# The eight constant-b arms R26 stored. "Constant" means constant ACROSS CLIPS, not
# b_bg == b_fg -- R26's taubg2_taufg2 (is_control) is the uniform Eq.4 case and is
# deliberately not in this set. Order matches r30_stage1.sh / r30_stage2.sh's ARMS.
CONST_BS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 20, 50)
ARMS: Tuple[str, ...] = ("lpips", "dino_cls", "dino_patch", "clip_image",
                         "clip_prompt", "depth", "normals", "selfsim")


# --------------------------------------------------------------------------------------
# joins
# --------------------------------------------------------------------------------------
def build_join_tables(data_root: Path, cases: List[Dict[str, object]]
                      ) -> Tuple[Dict[Tuple[int, str], str], Dict[Tuple[int, str], str]]:
    """(edit_type, file_id_str) -> video_name, and (edit_type, video_name) -> case_id.

    file_id indexes the FULL edit{T}_FiVE.json (100/100/100/100/9/10 entries), not the
    22-case subset, and a video can appear under two edit types with different case_ids
    (0011_lucia in edit2 AND edit5) -- so the join goes through the annotation file's own
    order, exactly as r30_stage2.sh's grid/arms join does.
    """
    idx2vid: Dict[Tuple[int, str], str] = {}
    for t in range(1, 7):
        items = json.loads((data_root / "edit_prompt" / f"edit{t}_FiVE.json").read_text())
        items = items if isinstance(items, list) else list(items.values())
        for i, e in enumerate(items):
            idx2vid[(t, str(i))] = e["video_name"]
    name2case = {(int(c["edit_type"]), c["video_name"]): c["case_id"] for c in cases}
    return idx2vid, name2case


def load_r26_metric(r26_csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                    name2case: Dict[Tuple[int, str], str],
                    bs: Sequence[int],
                    metric: str = "lpips_unedit_part",
                    method_fmt: str = "taubg0_taufg{b}_vp",
                    stem_fmt: str = "r26_{method}",
                    stride: int = 8) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): <metric>} from R26's stored per-edit-type CSVs.

    `metric` is any column R26's 9-metric harness wrote (e.g. lpips_unedit_part or
    ssim_unedit_part -- the preservation axis -- or clip_similarity_target_image -- the
    WHOLE-FRAME achievement axis, matching r26_tradeoff_figure.py's own "x =
    clip_similarity_target_image throughout" convention; NOT the _edit_part variant,
    which is masked to the edit region and not comparable across arms with different
    masks); all live in the same per-edit-type files, just different columns.

    `method_fmt` selects WHICH of R26's stored arms to read, `{b}` filled in per value in
    `bs`. Default is R30's own reference family -- tau_bg FIXED at 0, tau_fg = b, the
    SPATIAL arm the whole task is built around. Pass "taubg{b}_taufg{b}_vp" for the
    UNIFORM baseline instead: tau_bg == tau_fg == b, no fg/bg split at all -- the plain
    scalar Eq. 4 rho swept across the same 8 values, R26's degenerate/is_control family
    (52-arm sweep, tasks where bg==fg skip the mask/tau-field path entirely and run the
    scalar --blend_power path -- see r26_infer.sh). Both families cover the same 22 cases
    at the same 8 b values, so they're directly comparable with no extra rendering.
    """
    out: Dict[Tuple[str, int], float] = {}
    for b in bs:
        method = method_fmt.format(b=b)
        #✨ R34: `stem_fmt`/`stride` default to R26's stored whole-video convention, so
        # every pre-R34 call site is unchanged. R34 passes stem_fmt="r34_{method}_chunk1"
        # and stride=1 to read its own first-chunk CSVs through this same loader/join.
        stem = stem_fmt.format(method=method)
        pattern = str(r26_csv_dir / f"edit*_FiVE_{stem}_frame_stride{stride}.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"[r30_score] no R26 per-clip CSV matched {pattern}. "
                             f"Wrong --r26_csv_dir, or b={b} was never rendered.")
        for f in files:
            m = re.search(r"edit(\d+)_", os.path.basename(f))
            t = int(m.group(1))
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    col = next((k for k in r if k.endswith(f"|{metric}")), None)
                    if col is None:
                        raise SystemExit(f"[r30_score] {f} has no *|{metric} column.")
                    vid = idx2vid.get((t, r["file_id"]))
                    if vid is None:
                        continue                  # scored row outside the 22 cases
                    case_id = name2case.get((t, vid))
                    if case_id is None:
                        continue
                    out[(case_id, b)] = float(r[col])
    return out


#✨ Moved here from r34_score.py 2026-09-22, unchanged. It lives in the shared base so
# r31_score.py and r33_score.py can read single-arm CSVs without importing r34_score.py;
# r34_score.py re-exports it, so R34's own call sites are untouched.
def load_stem_metric(csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                     name2case: Dict[Tuple[int, str], str],
                     stem: str, stride: int, metric: str) -> Dict[str, float]:
    """{case_id: value} for ONE fully-resolved stem.

    Same per-edit-type join as load_r26_metric, but for a single arm with no `b` to
    sweep -- R31's arms. The stem is passed in already resolved precisely so the
    r31_/r34_r31_ prefix collision cannot happen inside a formatter here.

    Reads the per-edit-type per-clip files and never the top-level {stem}_avg.csv: R31
    established that evaluate.py's own "final averaging" step is wrong on some columns
    (lpips_unedit_part reading ~196 instead of ~0.2), a pre-existing harness bug this
    script cannot fix from the outside.
    """
    pattern = str(csv_dir / f"edit*_FiVE_{stem}_frame_stride{stride}.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[r30_score] no per-clip CSV matched {pattern}")
    out: Dict[str, float] = {}
    for f in files:
        t = int(re.search(r"edit(\d+)_", os.path.basename(f)).group(1))
        with open(f) as fh:
            for r in csv.DictReader(fh):
                col = next((k for k in r if k.endswith(f"|{metric}")), None)
                if col is None:
                    raise SystemExit(f"[r30_score] {f} has no *|{metric} column.")
                vid = idx2vid.get((t, r["file_id"]))
                if vid is None:
                    continue                      # scored row outside the 22 cases
                case_id = name2case.get((t, vid))
                if case_id is None:
                    continue
                out[case_id] = float(r[col])
    return out


def load_r26_lpips(r26_csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                   name2case: Dict[Tuple[int, str], str],
                   bs: Sequence[int]) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): lpips_unedit_part} -- thin wrapper, kept for the existing call sites."""
    return load_r26_metric(r26_csv_dir, idx2vid, name2case, bs, "lpips_unedit_part")


def load_fiveacc_arms(path: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    """{arm: {case_id: row_dict}}. Only arms actually present are keys.

    Every stored column comes back per row (yn_acc, mc_acc, routed b, the 9 metrics, ...)
    even though this script only reads clip_similarity_target_image / lpips_unedit_part /
    ssim_unedit_part below -- the rest stays available to other consumers of this same CSV.
    """
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["arm"], {})[r["case_id"]] = r
    return out


# --------------------------------------------------------------------------------------
# per-clip oracle frontier (alpha*CLIP + (1-alpha)*Y, swept), following
# r26_tradeoff_figure.py's oracle_frontier exactly -- see the module docstring.
# --------------------------------------------------------------------------------------
def _minmax(d: Dict[int, float]) -> Dict[int, float]:
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def oracle_frontier(clips: Sequence[str], clip_target: Dict[Tuple[str, int], float],
                    y_metric: Dict[Tuple[str, int], float], bs: Sequence[int],
                    n_alpha: int = 21, higher_is_better: bool = True
                    ) -> List[Tuple[float, float, float]]:
    """[(alpha, mean_raw_clip, mean_raw_y), ...] -- the per-clip oracle frontier over `bs`.

    For each alpha, each clip independently argmaxes `alpha*CLIP_norm + (1-alpha)*Y_norm`
    over its own b in `bs` (per-clip min-max, not a global scale -- see the module
    docstring), and the winning b's RAW (clip_target, y) is recorded. The frontier point
    for that alpha is the mean of those raw values across every clip. `higher_is_better`
    direction-corrects Y before the sum (LPIPS: `1 - minmax(raw)`); it never affects what's
    returned, only which b wins the argmax. Ties break on higher CLIP then better Y.
    """
    out: List[Tuple[float, float, float]] = []
    for i in range(n_alpha):
        alpha = i / (n_alpha - 1) if n_alpha > 1 else 0.0
        raw_clips, raw_ys = [], []
        for c in clips:
            clip_by_b = {b: clip_target[(c, b)] for b in bs}
            y_by_b = {b: y_metric[(c, b)] for b in bs}
            cn = _minmax(clip_by_b)
            yn = _minmax(y_by_b)
            if not higher_is_better:
                yn = {b: 1.0 - v for b, v in yn.items()}
            score = {b: alpha * cn[b] + (1 - alpha) * yn[b] for b in bs}
            tie_y = (lambda v: v) if higher_is_better else (lambda v: -v)
            best = max(bs, key=lambda b: (score[b], clip_by_b[b], tie_y(y_by_b[b])))
            raw_clips.append(clip_by_b[best])
            raw_ys.append(y_by_b[best])
        out.append((alpha, sum(raw_clips) / len(raw_clips), sum(raw_ys) / len(raw_ys)))
    return out


# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    if not args.fiveacc_arms.exists():
        raise SystemExit(f"[r30_score] {args.fiveacc_arms} does not exist -- has "
                         f"stage2-eval been run? (R30_PHASE=eval sbatch "
                         f"slurm_scripts/five_bench/r30_stage2.sh)")

    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})
    lpips = load_r26_lpips(args.r26_csv_dir, idx2vid, name2case, CONST_BS)
    # WHOLE-IMAGE CLIP-to-target, not the edit-region variant. r26_tradeoff_figure.py
    # fixes "x = clip_similarity_target_image (higher is better) throughout" as the one
    # CLIP definition for every trade-off figure in this research program, precisely so
    # the achievement axis means the same thing everywhere it's used; never touches the
    # union mask, so it is also directly comparable across arms with different masks.
    clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                  "clip_similarity_target_image")
    # Tuple order is (x, y, b) = (CLIP, preservation, b) throughout, matching
    # r26_tradeoff_figure.py's axis convention exactly: CLIP always on x, preservation
    # always on y.
    clip_curve = [(sum(clip_target[(c, b)] for c in clips) / len(clips),
                  sum(lpips[(c, b)] for c in clips) / len(clips), b)
                 for b in CONST_BS]

    # UNIFORM BASELINE (b_bg == b_fg, no fg/bg split at all): the plain scalar Eq. 4 rho
    # swept across the same 8 values, from R26's degenerate/is_control family -- the same
    # series r26_tradeoff_figure.py draws in BLUE, "the reference a spatial tau has to
    # beat". Free -- already stored, same 22 cases, no new rendering.
    uniform_lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                    "lpips_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")
    uniform_clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                          "clip_similarity_target_image",
                                          method_fmt="taubg{b}_taufg{b}_vp")
    uniform_curve = [(sum(uniform_clip_target[(c, b)] for c in clips) / len(clips),
                     sum(uniform_lpips[(c, b)] for c in clips) / len(clips), b)
                    for b in CONST_BS]

    # SSIM-vs-CLIP: same idea, SSIM instead of LPIPS on the preservation (y) axis (HIGHER
    # is better here, unlike LPIPS). Same two families, same free reuse of stored metrics.
    ssim = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                           "ssim_unedit_part")
    uniform_ssim = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                   "ssim_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")
    ssim_curve = [(sum(clip_target[(c, b)] for c in clips) / len(clips),
                  sum(ssim[(c, b)] for c in clips) / len(clips), b)
                 for b in CONST_BS]
    uniform_ssim_curve = [(sum(uniform_clip_target[(c, b)] for c in clips) / len(clips),
                          sum(uniform_ssim[(c, b)] for c in clips) / len(clips), b)
                         for b in CONST_BS]

    # Per-clip oracle frontier over the SPATIAL family's own 8-b grid -- the ceiling an
    # all-knowing per-clip policy could reach, not a candidate setting (costs per-clip
    # supervision neither the constant curve nor the arms have).
    lpips_frontier = oracle_frontier(clips, clip_target, lpips, CONST_BS,
                                     n_alpha=args.n_alpha, higher_is_better=False)
    ssim_frontier = oracle_frontier(clips, clip_target, ssim, CONST_BS,
                                    n_alpha=args.n_alpha, higher_is_better=True)

    arms_data = load_fiveacc_arms(args.fiveacc_arms)
    if not arms_data:
        raise SystemExit(f"[r30_score] {args.fiveacc_arms} has no rows.")

    out_rows: List[Dict[str, object]] = []
    for arm in ARMS:
        if arm not in arms_data:
            print(f"[r30_score] SKIP {arm} -- not present in {args.fiveacc_arms} "
                  f"(never rendered, or disqualified at stage 1).")
            continue
        rows = arms_data[arm]
        arm_clips = sorted(rows)
        if len(arm_clips) != len(clips):
            print(f"[r30_score] WARNING {arm}: {len(arm_clips)}/{len(clips)} clips "
                  f"present -- scoring on the subset actually rendered.")

        p_a = sum(float(rows[c]["lpips_unedit_part"]) for c in arm_clips) / len(arm_clips)
        clip_a = sum(float(rows[c]["clip_similarity_target_image"])
                    for c in arm_clips) / len(arm_clips)
        ssim_a = sum(float(rows[c]["ssim_unedit_part"])
                    for c in arm_clips) / len(arm_clips)

        out_rows.append({
            "arm": arm, "n_clips": len(arm_clips),
            "clip_target": round(clip_a, 6), "lpips": round(p_a, 6),
            "ssim": round(ssim_a, 6),
        })

    if not out_rows:
        raise SystemExit("[r30_score] no arm had data in --fiveacc_arms. Nothing to score.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[r30_score] wrote {args.out} ({len(out_rows)} arm(s))")

    print(f"\n{'arm':<12} {'clip_target':>12} {'lpips':>9} {'ssim':>9}")
    for row in out_rows:
        print(f"{row['arm']:<12} {row['clip_target']:>12.4f} {row['lpips']:>9.5f} "
              f"{row['ssim']:>9.5f}")

    if args.fig_dir is not None:
        make_figures(args.fig_dir, out_rows, clip_curve, uniform_curve, ssim_curve,
                    uniform_ssim_curve, lpips_frontier, ssim_frontier)
    return 0


# --------------------------------------------------------------------------------------
# figures: r30_clip_vs_lpips.pdf, r30_clip_vs_ssim.pdf
# --------------------------------------------------------------------------------------
def make_figures(fig_dir: Path, out_rows, clip_curve, uniform_curve, ssim_curve,
                 uniform_ssim_curve, lpips_frontier, ssim_frontier) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    # Axis convention matches r26_tradeoff_figure.py exactly: CLIP (clip_similarity_
    # target_image, whole-frame) always on x, preservation always on y.
    def draw_curve_scatter(filename, title, x_label, y_label, spatial_curve,
                           uniform_curve, x_key, y_key, frontier, y_short):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        xs = [p for p, _, _ in spatial_curve]
        ys = [c for _, c, _ in spatial_curve]
        ax.plot(xs, ys, "-o", color="black",
                label="constant-$b$ SPATIAL curve (R26, $\\tau_{bg}$=0)", zorder=3)
        for p, c, b in spatial_curve:
            ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(4, 4),
                        fontsize=8)
        # UNIFORM baseline: b_bg == b_fg, the plain scalar Eq. 4 rho, no fg/bg split.
        # Free -- already stored, same 22 cases, needs no new rendering.
        uxs = [p for p, _, _ in uniform_curve]
        uys = [c for _, c, _ in uniform_curve]
        ax.plot(uxs, uys, marker="D", color="dimgray", linestyle="--",
                label="uniform baseline ($b_{bg}=b_{fg}$, Eq. 4, R26)", zorder=2)
        for p, c, b in uniform_curve:
            ax.annotate(f"$b$={b}", (p, c), textcoords="offset points",
                        xytext=(4, -10), fontsize=8, color="dimgray")
        for i, row in enumerate(out_rows):
            if x_key not in row or y_key not in row:
                continue
            ax.scatter([row[x_key]], [row[y_key]], marker="s", s=70,
                      color=cmap(i % 10), edgecolor="black", zorder=4,
                      label=f"{row['arm']} ({y_key}={row[y_key]:.3f})")
        # Per-clip oracle frontier (alpha*CLIP + (1-alpha)*Y sweep, R26 convention).
        fx = [c for _, c, _ in frontier]
        fy = [y for _, _, y in frontier]
        ax.plot(fx, fy, "-", color="tab:green", linewidth=2, zorder=5,
                label="per-clip oracle ($\\alpha$ sweep, 8-$b$ grid)")
        ax.annotate(f"$\\alpha$=0\n({y_short})", (fx[0], fy[0]), fontsize=7,
                    color="tab:green")
        ax.annotate("$\\alpha$=1\n(CLIP)", (fx[-1], fy[-1]), fontsize=7, color="tab:green")
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
        "r30_clip_vs_lpips.pdf",
        "R30: CLIP-target vs. LPIPS, with per-clip oracle frontier",
        "mean clip_similarity_target_image  (higher = more achieved)",
        "mean lpips_unedit_part  (lower = better preserved)",
        clip_curve, uniform_curve, "clip_target", "lpips", lpips_frontier, "LPIPS")

    p2 = draw_curve_scatter(
        "r30_clip_vs_ssim.pdf",
        "R30: CLIP-target vs. SSIM, with per-clip oracle frontier",
        "mean clip_similarity_target_image  (higher = more achieved)",
        "mean ssim_unedit_part  (higher = better preserved)",
        ssim_curve, uniform_ssim_curve, "clip_target", "ssim", ssim_frontier, "SSIM")

    print(f"[r30_score] wrote {p1}, {p2}")


# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fiveacc_arms", type=Path, required=True,
                   help="Each rendered arm's FiVE-Acc + 9-metric results, from "
                        "stage2-eval.")
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"),
                   help="Where R26's per-edit-type per-clip CSVs live (for "
                        "lpips_unedit_part / ssim_unedit_part / clip_similarity_"
                        "target_image, the two panels' axes).")
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"),
                   help="FiVE-Bench root, for the file_id -> video_name join.")
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--n_alpha", type=int, default=21,
                   help="alpha grid points in [0, 1] for the oracle frontier sweep, "
                        "matching r26_tradeoff_figure.py's default.")
    p.add_argument("-o", "--out", type=Path, default=Path("evaluation/csv/r30_arms.csv"))
    p.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"),
                   help="Where r30_clip_vs_lpips.pdf / r30_clip_vs_ssim.pdf are written.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
