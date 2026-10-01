#!/usr/bin/env python
"""R33 report: R31's per-video qualitative grid + per-video Pareto panel, regenerated
with the achievement (x) axis swapped from the incumbent whole-frame CLIP to whichever
axis R33's axis-compare step found actually ranks edit strength (default: clip_d_prompt).

WHY THIS EXISTS. evaluation/r31_html_report.py already builds this exact report --
source/baseline/lpips/dino/normals row strips stacked above each video's own
clip_similarity_target_image-vs-LPIPS Pareto point -- but that x-axis is the incumbent
CLIP-to-target metric R33 measured does not rank R26's own constant-b sweep by edit
strength (Spearman(b, clip_target) = +0.098 mean, 11/22 positive). This script is the
SAME report, generalised to take any of R33's candidate axes as --x_metric, so the
figures the paper actually uses are read off an axis that means what the caption claims.

ROW STRIPS. Unchanged from R31's report and reused, not reimplemented: the visual crops
(source / baseline / lpips-arm / dino-arm / normals-arm) are pixel crops of R31's
existing evaluation/figures/r31_arm_grids/*.png and evaluation/figures/r26_grids/*.pdf --
which imagery to show never depended on which achievement axis is used to score it, so
crop_arm_rows / extract_baseline_row / label_strip (and the build_strips orchestration
that calls them) are imported straight from r31_html_report.py.

PARETO PANEL. Same axes/curves/frontier structure as r31_html_report.build_video_pareto
(constant-b SPATIAL curve, uniform baseline, per-clip oracle frontier, R31's 4 arms as
star markers), but the x-VALUE for every point is read via --x_metric instead of being
hardcoded to clip_similarity_target_image:
  - clip_similarity_target_image / _edit_part: r30_score.load_r26_metric, unchanged
    (already stored in R26/R31's own per-edit-type CSVs).
  - clip_d_prompt / clip_d_word: r33_axis_compare.load_clipd_metric against
    evaluation/csv/r33_clip_directional.csv (method_fmt carries the r26_/r31_ prefix
    that CSV's method column uses -- see R33's clip-d-script todo).
  - yn_acc / mc_acc / five_acc: r33_axis_compare.load_fiveacc_metric against R33's own
    edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv files (evaluate.py's raw
    column names, five_acc_yes_no / five_acc_multi_choice / five_acc -- see
    r33_axis_compare.py's AXES table for why yn_acc/mc_acc's labels differ from their
    column names; five_acc's label IS its column name, no renaming needed).
    five_acc (added 2026-09-22) is evaluate.py's own combined verdict, computed at job
    time and already sitting in every FiVE-Acc CSV on disk -- no new scoring required.
    Confirmed in evaluate.py (~line 519-527): given yn_acc/mc_acc in {0,1},
    union = int(yn_acc or mc_acc), inter = int(yn_acc and mc_acc), and
    five_acc = mean([yn_acc, mc_acc, union, inter]) -- i.e. (yn+mc+union+inter)/4,
    exactly. R26/R31's per-clip CSVs already carry all five raw columns
    (five_acc_yes_no/_multi_choice/_union/_inter/five_acc); R30's r30_fiveacc_arms.csv
    carries yn_acc/mc_acc/union/inter but not a precomputed five_acc column (out of
    scope here -- this script never reads R30's per-video data, only R26/R31's).
The y-axis (lpips_unedit_part, preservation) is unchanged throughout -- R33 only
disputes the achievement axis, not the preservation one.

OUTPUTS are r33_-prefixed throughout (evaluation/figures/r33_html_strips/,
r33_video_pareto/, r33_report.html) so R31's existing report is never overwritten.

MULTI-AXIS MODE (--x_metric_more / --pareto_out_more, added 2026-09-22, superseding the
same day's earlier --x_metric2/--pareto_out2 single-pair flags). r33_report.html
originally showed one achievement axis at a time -- a separate --x_metric run wrote its
own standalone HTML (e.g. r33_report_fiveacc.html for yn_acc). At the user's request the
canonical r33_report.html shows every requested axis's per-video panel side by side in
the same section, so a reader compares them on one clip without switching files. The
first axis is still --x_metric/--pareto_out (required, as always); each ADDITIONAL axis
is one more --x_metric_more paired POSITIONALLY with one more --pareto_out_more (repeat
both flags once per extra axis, same count of each -- a mismatched count is a hard
SystemExit, not a silent truncation). Two flags rather than one packed string
("metric:path") to match this script's existing plain-flag style. All panels render
through write_html_multi (defined in this file, not r31_html_report.py -- write_html
there only accepts one pareto image per section, and patching N images in through the
existing sentinel-string-replace trick main() uses for the single-axis title/caption
would be more fragile than a small dedicated writer for arbitrarily many panels).
Omitting --x_metric_more entirely keeps the original single-axis behaviour unchanged,
which is what the standalone --pareto_out clipd/ and fiveacc/ split-folder runs still use.

Usage
-----
    python evaluation/r33_report_figures.py --x_metric clip_d_prompt
    python evaluation/r33_report_figures.py --x_metric clip_d_prompt \\
        --pareto_out evaluation/figures/r33_video_pareto/clipd \\
        --x_metric_more yn_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_yn \\
        --x_metric_more mc_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_mc \\
        --x_metric_more five_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_combined
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import CONST_BS, build_join_tables, load_r26_metric, oracle_frontier  # noqa: E402
from r31_score import ARMS, load_r31_percli_metric  # noqa: E402
from r31_html_report import (  # noqa: E402
    ROW_ORDER,
    build_strips,
    write_html as _r31_write_html,
)
from r33_axis_compare import load_clipd_metric, load_fiveacc_metric  # noqa: E402

# label -> (evaluate.py / r33_clip_directional.py column name, axis is stored per-frame
# in R26's own CSVs vs. read from r33's own CLIP-D / FiVE-Acc CSVs).
AXIS_COLUMN: Dict[str, str] = {
    "clip_similarity_target_image": "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part": "clip_similarity_target_image_edit_part",
    "clip_d_prompt": "clip_d_prompt",
    "clip_d_word": "clip_d_word",
    "yn_acc": "five_acc_yes_no",
    "mc_acc": "five_acc_multi_choice",
    "five_acc": "five_acc",
}
AXIS_LABEL: Dict[str, str] = {
    "clip_similarity_target_image": "clip_similarity_target_image (achieved edit)",
    "clip_similarity_target_image_edit_part": "clip_similarity_target_image_edit_part (achieved edit)",
    "clip_d_prompt": "CLIP-D, prompt pair (achieved edit)",
    "clip_d_word": "CLIP-D, word pair (achieved edit)",
    "yn_acc": "FiVE-Acc yes/no accuracy (achieved edit)",
    "mc_acc": "FiVE-Acc multi-choice accuracy (achieved edit)",
    "five_acc": "FiVE-Acc combined, (yn+mc+union+inter)/4 (achieved edit)",
}
# Short, filesystem-safe stem per axis, for the per-video PNG filenames -- mirrors
# r31_html_report.py's precedent (that script only ever plotted whole-frame CLIP, so it
# could hardcode "clip"; this one is parametrized over --x_metric, so the stem must be
# too, or two runs with different --x_metric values silently overwrite each other's PNGs
# under the same generic name).
AXIS_STEM: Dict[str, str] = {
    "clip_similarity_target_image": "clip",
    "clip_similarity_target_image_edit_part": "clip_edit",
    "clip_d_prompt": "clipd",
    "clip_d_word": "clipd_word",
    "yn_acc": "fiveacc_yn",
    "mc_acc": "fiveacc_mc",
    "five_acc": "fiveacc_combined",
}


def load_axis_curve(x_metric: str, method_fmt: str, r26_csv_dir: Path, clipd_csv: Path,
                    idx2vid: Dict[Tuple[int, str], str],
                    name2case: Dict[Tuple[int, str], str]
                    ) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): x_value} for one constant-b family (SPATIAL or UNIFORM).

    `method_fmt` is R26's own naming ("taubg0_taufg{b}_vp" or "taubg{b}_taufg{b}_vp");
    the clip-d branch prefixes it with "r26_" to match r33_clip_directional.csv's
    method column (see the clip-d-script todo's note on the r26_/r30_/r31_ prefix
    every CLIP-D method label carries).
    """
    col = AXIS_COLUMN[x_metric]
    if x_metric in ("clip_similarity_target_image",
                    "clip_similarity_target_image_edit_part"):
        return load_r26_metric(r26_csv_dir, idx2vid, name2case, CONST_BS, col,
                               method_fmt=method_fmt)
    if x_metric in ("clip_d_prompt", "clip_d_word"):
        return load_clipd_metric(clipd_csv, col, method_fmt="r26_" + method_fmt)
    if x_metric in ("yn_acc", "mc_acc", "five_acc"):
        return load_fiveacc_metric(r26_csv_dir, idx2vid, name2case, CONST_BS, col,
                                   method_fmt=method_fmt)
    raise SystemExit(f"[r33_report_figures] unknown --x_metric {x_metric!r}")


def load_axis_r31_point(x_metric: str, arm: str, r26_csv_dir: Path, clipd_csv: Path,
                        idx2vid: Dict[Tuple[int, str], str],
                        name2case: Dict[Tuple[int, str], str]) -> Dict[str, float]:
    """{case_id: x_value} for one R31 arm -- a single point per clip, no b to sweep."""
    col = AXIS_COLUMN[x_metric]
    if x_metric in ("clip_similarity_target_image",
                    "clip_similarity_target_image_edit_part"):
        return load_r31_percli_metric(r26_csv_dir, idx2vid, name2case, arm, "", col)
    if x_metric in ("clip_d_prompt", "clip_d_word"):
        out: Dict[str, float] = {}
        import csv
        with open(clipd_csv) as fh:
            for r in csv.DictReader(fh):
                if r["method"] == f"r31_{arm}":
                    out[r["case_id"]] = float(r[col])
        return out
    if x_metric in ("yn_acc", "mc_acc", "five_acc"):
        import csv
        import glob
        import os
        import re
        pattern = str(r26_csv_dir / f"edit*_FiVE_r33_fiveacc_r31_{arm}_frame_stride8.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"[r33_report_figures] no FiVE-Acc per-clip CSV matched "
                             f"{pattern} for r31_{arm}.")
        out = {}
        for f in files:
            m = re.search(r"edit(\d+)_", os.path.basename(f))
            t = int(m.group(1))
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    c = next((k for k in r if k.endswith(f"|{col}")), None)
                    vid = idx2vid.get((t, r["file_id"]))
                    if vid is None:
                        continue
                    case_id = name2case.get((t, vid))
                    if case_id is None:
                        continue
                    out[case_id] = float(r[c])
        return out
    raise SystemExit(f"[r33_report_figures] unknown --x_metric {x_metric!r}")


# ----------------------------------------------------------------------- pareto point --

def build_video_pareto(case_id: str, x_metric: str,
                       x_spatial: Dict[Tuple[str, int], float],
                       lpips: Dict[Tuple[str, int], float],
                       x_uniform: Dict[Tuple[str, int], float],
                       uniform_lpips: Dict[Tuple[str, int], float],
                       r31_x: Dict[str, Dict[str, float]],
                       r31_lpips: Dict[str, Dict[str, float]],
                       out_dir: Path, n_alpha: int, video_name: str, edit_type: int
                       ) -> Path:
    """This video's own x_metric-vs-LPIPS panel, restricted to `case_id` alone --
    same structure as r31_html_report.build_video_pareto, x-axis parametrized.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spatial_curve = [(x_spatial[(case_id, b)], lpips[(case_id, b)], b) for b in CONST_BS]
    uniform_curve = [(x_uniform[(case_id, b)], uniform_lpips[(case_id, b)], b)
                     for b in CONST_BS]
    frontier = oracle_frontier([case_id], x_spatial, lpips, CONST_BS,
                               n_alpha=n_alpha, higher_is_better=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [p for p, _, _ in spatial_curve]
    ys = [c for _, c, _ in spatial_curve]
    ax.plot(xs, ys, "-o", color="black", label="constant-$b$ SPATIAL ($\\tau_{bg}$=0)",
           zorder=3)
    for p, c, b in spatial_curve:
        ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(3, 3),
                    fontsize=6.5)
    uxs = [p for p, _, _ in uniform_curve]
    uys = [c for _, c, _ in uniform_curve]
    ax.plot(uxs, uys, marker="D", color="dimgray", linestyle="--", markersize=4,
           label="uniform baseline ($b_{bg}=b_{fg}$)", zorder=2)

    cmap = plt.get_cmap("tab10")
    for i, arm in enumerate(ARMS):
        if case_id not in r31_x.get(arm, {}) or case_id not in r31_lpips.get(arm, {}):
            continue
        ax.scatter([r31_x[arm][case_id]], [r31_lpips[arm][case_id]], marker="*", s=220,
                  color=cmap(i % 10), edgecolor="black", linewidth=0.7, zorder=5,
                  label=f"r31_{arm} ({r31_lpips[arm][case_id]:.3f})")

    fx = [c for _, c, _ in frontier]
    fy = [y for _, _, y in frontier]
    ax.plot(fx, fy, "-", color="tab:green", linewidth=1.6, zorder=1.5,
           label="per-clip oracle ($\\alpha$ sweep)")

    ax.set_xlabel(AXIS_LABEL[x_metric], fontsize=8)
    ax.set_ylabel("lpips_unedit_part (lower = better preserved)", fontsize=8)
    ax.set_title(f"edit{edit_type} {video_name} -- {x_metric} vs. LPIPS (this clip only)",
               fontsize=8.5)
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = out_dir / f"{case_id}_{AXIS_STEM[x_metric]}_vs_lpips.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- multi html -----

def write_html_multi(cases: list, strips_by_case: Dict[str, Dict[str, Path]],
                     panels: Sequence[Tuple[str, Dict[str, Path]]],
                     out_path: Path, root: Path) -> None:
    """Same section layout as r31_html_report.write_html (source/baseline/arm strips,
    then a Pareto panel below), but with N Pareto panels side by side per clip instead of
    one -- one per achievement axis. `panels` is [(label, pareto_by_case), ...] in the
    order they render, left to right; each label is an AXIS_LABEL string shown as that
    panel's caption, so a reader never has to infer which image is which axis. Replaces
    the earlier write_html_dual (fixed at exactly two panels) now that a third axis
    (five_acc) is wanted alongside CLIP-D and yn_acc -- N=2 still works the same way,
    just as a 1-element `panels` list.
    """
    def rel(p: Path) -> str:
        return str(p.relative_to(root))

    sections = []
    for c in sorted(cases, key=lambda c: (int(c["edit_type"]), c["video_name"])):
        cid, t, name = c["case_id"], int(c["edit_type"]), c["video_name"]
        strips = strips_by_case.get(cid, {})
        rows_html = "\n".join(
            f'<img class="strip" src="{rel(strips[row])}" alt="{row}">'
            for row in ROW_ORDER if row in strips)

        def panel(pareto_by_case: Dict[str, Path], label: str) -> str:
            p = pareto_by_case.get(cid)
            if p is None:
                return "<figure><p><em>no data</em></p></figure>"
            return (f'<figure><img class="pareto" src="{rel(p)}" alt="{label} pareto">'
                    f'<figcaption>{label}</figcaption></figure>')

        panels_html = "\n    ".join(panel(pareto_by_case, label)
                                    for label, pareto_by_case in panels)
        prompt = f'{c.get("src_word", "")} → {c.get("trg_word", "")}'
        sections.append(f"""
<section class="video">
  <h2>edit{t} &mdash; {name}</h2>
  <p class="prompt">{prompt}</p>
  <div class="grid">{rows_html}</div>
  <div class="pareto-wrap dual">
    {panels_html}
  </div>
</section>""")

    axes_note = ", ".join(label for label, _ in panels)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>R33 qualitative report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #f7f7f8; color: #1a1a1a; }}
  header {{ padding: 20px 24px; background: #1a1a1a; color: #fff; }}
  header p {{ margin: 4px 0 0; color: #c9c9c9; font-size: 13px; }}
  .video {{ background: #fff; margin: 20px auto; max-width: 1100px; padding: 18px 22px;
           border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  .video h2 {{ margin: 0 0 2px; font-size: 17px; }}
  .prompt {{ margin: 0 0 12px; color: #555; font-size: 13px; font-style: italic; }}
  .grid img.strip {{ display: block; width: 100%; margin-bottom: 2px; border: 1px solid #eee; }}
  .pareto-wrap {{ margin-top: 14px; text-align: center; }}
  .pareto-wrap.dual {{ display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; }}
  .pareto-wrap.dual figure {{ margin: 0; flex: 1 1 420px; max-width: 480px; }}
  .pareto-wrap img.pareto {{ max-width: 100%; width: 100%; border: 1px solid #eee; }}
  .pareto-wrap figcaption {{ font-size: 12px; color: #666; margin-top: 4px; }}
  nav {{ position: sticky; top: 0; background: #fff; padding: 8px 24px; border-bottom: 1px
        solid #ddd; font-size: 12px; z-index: 10; }}
  nav a {{ margin-right: 10px; color: #444; text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1 style="margin:0;font-size:19px;">R33 &mdash; per-video qualitative grid + per-video Pareto, {len(panels)} axes side by side</h1>
  <p>Each section: source / streamedit baseline (uniform b=2) / lpips / dino / normals renders,
     with THAT video's own achievement-vs-LPIPS point (not averaged over the 22-clip set) below it,
     on {len(panels)} axes at once, left to right: {axes_note}.</p>
</header>
{"".join(sections)}
</body>
</html>"""
    out_path.write_text(html)


# --------------------------------------------------------------------------- main ------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x_metric", type=str, default="clip_d_prompt",
                   choices=sorted(AXIS_COLUMN))
    p.add_argument("--x_metric_more", type=str, action="append", default=None,
                   choices=sorted(AXIS_COLUMN),
                   help="One additional achievement axis, beyond --x_metric. Repeat "
                        "this flag once per extra axis; each occurrence pairs "
                        "POSITIONALLY with one --pareto_out_more occurrence (same "
                        "count of each, or a hard SystemExit). When given, every "
                        "axis's per-video panel renders side by side in one "
                        "r33_report.html section instead of writing separate "
                        "single-axis reports.")
    p.add_argument("--pareto_out_more", type=Path, action="append", default=None,
                   help="Where each --x_metric_more axis's per-video PNGs are "
                        "written, one occurrence per --x_metric_more, same order.")
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--clipd_csv", type=Path,
                   default=Path("evaluation/csv/r33_clip_directional.csv"))
    p.add_argument("--arm_grids_dir", type=Path,
                   default=Path("evaluation/figures/r31_arm_grids"))
    p.add_argument("--r26_grids_dir", type=Path, default=Path("evaluation/figures/r26_grids"))
    p.add_argument("--strips_out", type=Path,
                   default=Path("evaluation/figures/r33_html_strips"))
    p.add_argument("--pareto_out", type=Path,
                   default=Path("evaluation/figures/r33_video_pareto"))
    p.add_argument("--out", type=Path, default=Path("evaluation/figures/r33_report.html"))
    p.add_argument("--n_alpha", type=int, default=21)
    args = p.parse_args(argv)

    extra_metrics = args.x_metric_more or []
    extra_outs = args.pareto_out_more or []
    if len(extra_metrics) != len(extra_outs):
        raise SystemExit(f"[r33_report_figures] {len(extra_metrics)} --x_metric_more "
                         f"but {len(extra_outs)} --pareto_out_more -- these pair "
                         f"positionally and must come in equal counts.")

    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)

    lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                            "lpips_unedit_part")
    uniform_lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                    "lpips_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")

    # LPIPS for R31's 4 arms never depends on the achievement axis -- computed once and
    # reused by both build_video_pareto calls below (primary axis and, if given, the
    # second one), instead of being re-loaded per axis for no reason.
    r31_lpips: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        try:
            r31_lpips[arm] = load_r31_percli_metric(args.csv_dir, idx2vid, name2case, arm,
                                                     "", "lpips_unedit_part")
        except SystemExit as ex:
            print(f"[r33_report_figures] SKIP r31_{arm} lpips: {ex}")

    def load_axis(x_metric: str) -> Tuple[Dict, Dict, Dict[str, Dict[str, float]]]:
        x_spatial = load_axis_curve(x_metric, "taubg0_taufg{b}_vp",
                                    args.r26_csv_dir, args.clipd_csv, idx2vid, name2case)
        x_uniform = load_axis_curve(x_metric, "taubg{b}_taufg{b}_vp",
                                    args.r26_csv_dir, args.clipd_csv, idx2vid, name2case)
        r31_x: Dict[str, Dict[str, float]] = {}
        for arm in ARMS:
            try:
                r31_x[arm] = load_axis_r31_point(x_metric, arm, args.r26_csv_dir,
                                                 args.clipd_csv, idx2vid, name2case)
            except SystemExit as ex:
                print(f"[r33_report_figures] SKIP r31_{arm} on {x_metric}: {ex}")
        return x_spatial, x_uniform, r31_x

    # One (x_spatial, x_uniform, r31_x, pareto_out) tuple per axis, primary first --
    # loaded once up front so the per-case loop below just builds panels, not axes.
    axis_data = [(args.x_metric, args.pareto_out, load_axis(args.x_metric))]
    for m, out in zip(extra_metrics, extra_outs):
        axis_data.append((m, out, load_axis(m)))

    strips_by_case: Dict[str, Dict[str, Path]] = {}
    # pareto_by_axis[x_metric] = {case_id: png_path}, one dict per requested axis.
    pareto_by_axis: Dict[str, Dict[str, Path]] = {m: {} for m, _, _ in axis_data}
    for c in cases:
        cid = c["case_id"]
        try:
            strips_by_case[cid] = build_strips(c, args.arm_grids_dir, args.r26_grids_dir,
                                               args.strips_out)
        except Exception as ex:
            print(f"[r33_report_figures] STRIPS FAILED {cid}: {type(ex).__name__}: {ex}")
        for x_metric, pareto_out, (x_spatial, x_uniform, r31_x) in axis_data:
            try:
                pareto_by_axis[x_metric][cid] = build_video_pareto(
                    cid, x_metric, x_spatial, lpips, x_uniform, uniform_lpips,
                    r31_x, r31_lpips, pareto_out, args.n_alpha,
                    c["video_name"], int(c["edit_type"]))
            except Exception as ex:
                print(f"[r33_report_figures] PARETO[{x_metric}] FAILED {cid}: "
                     f"{type(ex).__name__}: {ex}")

    if extra_metrics:
        panels = [(AXIS_LABEL[m], pareto_by_axis[m]) for m, _, _ in axis_data]
        write_html_multi(cases, strips_by_case, panels, args.out, root=args.out.parent)
        counts = "+".join(str(len(pareto_by_axis[m])) for m, _, _ in axis_data)
        print(f"[r33_report_figures] axes={[m for m, _, _ in axis_data]} "
             f"wrote {args.out} ({len(strips_by_case)} grids, {counts} pareto points)")
        return 0

    pareto_by_case = pareto_by_axis[args.x_metric]
    _r31_write_html(cases, strips_by_case, pareto_by_case, args.out, root=args.out.parent)
    # r31_html_report.write_html hardcodes an "R31" title/caption; patch it to say R33
    # and name the achievement axis actually used, rather than duplicating the whole
    # function for a two-line difference.
    html = args.out.read_text()
    html = html.replace(
        "R31 &mdash; per-video qualitative grid + per-video CLIP-vs-LPIPS Pareto",
        f"R33 &mdash; per-video qualitative grid + per-video {args.x_metric}-vs-LPIPS Pareto")
    html = html.replace(
        "Each section: source / streamedit baseline (uniform b=2) / lpips / dino / normals renders,\n"
        "     with THAT video's own CLIP-vs-LPIPS point (not averaged over the 22-clip set) below it.",
        f"Each section: source / streamedit baseline (uniform b=2) / lpips / dino / normals renders,\n"
        f"     with THAT video's own {args.x_metric}-vs-LPIPS point (not averaged over the 22-clip "
        f"set) below it. Achievement axis: {AXIS_LABEL[args.x_metric]}.")
    html = html.replace("<title>R31 qualitative report</title>",
                        "<title>R33 qualitative report</title>")
    args.out.write_text(html)

    print(f"[r33_report_figures] x_metric={args.x_metric} wrote {args.out} "
         f"({len(strips_by_case)} grids, {len(pareto_by_case)} pareto points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
