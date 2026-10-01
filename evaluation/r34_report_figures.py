#!/usr/bin/env python
"""R34 per-clip report: chunk-1 Pareto panel (CLIP-D x, LPIPS y) for EACH clip
individually, paired with that clip's own qualitative grid (evaluation/r34_chunk1_grids.py),
plus evaluation/csv/r34_perclip_gap.csv -- the paired per-clip gap of each R31 arm
against that clip's OWN 8-b spatial Pareto envelope.

WHY THIS EXISTS
---------------
r34_score.py's arm-level means carry no dispersion: whether R31's `lpips`/`dino_patch`
"sitting above the spatial envelope" on the chunk-1 window is a real per-clip effect or
an artefact of the halved achievement spread being converted into vertical distance by a
steeper curve cannot be told from a single mean. This script computes the same
displacement PER CLIP, against that clip's OWN spatial envelope (not the aggregate
curve), and reports it with dispersion (SE, both n's) rather than folding it into one
number.

STRUCTURE. This is the per-clip counterpart of r34_score.py: the same loaders
(r30_score.build_join_tables / oracle_frontier, r34_score's load_r26_metric wrappers /
load_clip_d / load_stem_metric / pareto_envelope) are reused unmodified, just indexed
down to one clip at a time -- mirrors r33_report_figures.py's relationship to
r33_score.py, and r31_html_report.py's relationship to r31_score.py/r30_score.py.

ROW STRIPS come from evaluation/r34_chunk1_grids.py's OWN per-clip PNGs
(``edit{T}_{video_name}_chunk1.png``, already 6 rows x 9 cols: source / uniform
baseline / 4 R31 arms, at the chunk-1 window) -- embedded whole, not re-cropped. That
script already draws exactly the imagery this report wants; r31_html_report.py had to
crop row bands out of whole-video grids built for a different purpose, but r34_chunk1_
grids.py was purpose-built for this report, so there is nothing left to crop.

PARETO PANEL, PER CLIP: chunk-1 CLIP-D (x) vs. chunk-1 LPIPS (y) -- the achievement axis
of record (see r34_score.py's docstring) -- this clip's own 8-point SPATIAL curve, its
own 8-point UNIFORM curve, R31's 4 arms as stars, and its own oracle frontier (n_alpha
sweep, computed with `clips=[case_id]` alone). The clip's own SPATIAL Pareto envelope's
x-span is shaded; any arm whose CLIP-D falls OUTSIDE that span is drawn hollow rather
than filled, since the paired gap below is undefined for it -- this is the annotation
the plan asks for, made visually inline rather than as a separate caveat.

PAIRED GAP (evaluation/csv/r34_perclip_gap.csv). Per arm, per clip:

    gap = arm_lpips - envelope_lpips_at(arm_clip_d)

linearly interpolated along that clip's own SPATIAL Pareto envelope (same interpolation
r34_score.oracle_gap uses against the frontier, applied here to a single arm point --
see that function's docstring for why the ENVELOPE, not raw sorted points, is what gets
interpolated against on a curve that is not monotone in achievement). Positive = the arm
is WORSE preserved than the fixed-b envelope reaches at the same achievement, i.e.
displaced ABOVE it. Clips whose arm CLIP-D falls outside the envelope's x-span are
EXCLUDED from the mean/SE and counted in `n_out_of_range`, not silently dropped -- the
aggregate analysis (r34_score.py) had to drop 6-12/22 clips for exactly this reason, and
this file is what makes that count visible PER ARM instead of buried in a caveat.

⚠️ `n_in_range` differs per arm (each arm's CLIP-D lands inside a different subset of
clips' own envelope spans) and the SE is uncorrected for testing 4 arms at once on small
n, so these are not a clean set of paired samples across a fixed clip set -- report both
n's next to every mean in the HTML, per the plan's own instruction, rather than the mean
alone.

Usage
-----
    python evaluation/r34_report_figures.py \\
        --chunk1_csv_dir evaluation/csv \\
        --clipd_csv evaluation/csv/r34_clip_directional_chunk1.csv \\
        --grids_dir evaluation/figures/r34_chunk1_grids \\
        --pareto_out evaluation/figures/r34_video_pareto \\
        --gap_csv evaluation/csv/r34_perclip_gap.csv \\
        --out evaluation/figures/r34_report.html
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import CONST_BS, build_join_tables, load_r26_metric, oracle_frontier  # noqa: E402
from r34_score import (  # noqa: E402
    CHUNK1_STEM,
    CHUNK1_STRIDE,
    R31_ARMS,
    SPATIAL_FMT,
    UNIFORM_FMT,
    load_clip_d,
    load_stem_metric,
    pareto_envelope,
)

Curve = List[Tuple[float, float, int]]              # (x, y, b)
ArmPoint = Tuple[float, float, bool, Optional[float]]  # (clip_d, lpips, in_range, gap)


# --------------------------------------------------------------------------- envelope --
def interp_envelope(env: Sequence[Tuple[float, float]], x: float) -> Optional[float]:
    """Linear interpolation of the (x, y) envelope at `x`; None if `x` is outside its span.

    Same interpolation r34_score.oracle_gap performs against the frontier -- see that
    function's docstring for why the ENVELOPE (not raw sorted fixed-b points) is the
    thing interpolated against on a curve that is not monotone in achievement.
    """
    if len(env) < 2 or x < env[0][0] or x > env[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(env, env[1:]):
        if x0 <= x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


# ----------------------------------------------------------------------- pareto panel --
def build_video_pareto(case_id: str, video_name: str, edit_type: int,
                       spatial_curve: Curve, uniform_curve: Curve,
                       frontier: Sequence[Tuple[float, float, float]],
                       env: Sequence[Tuple[float, float]],
                       arm_points: Dict[str, ArmPoint], out_dir: Path,
                       clip_d_column: str) -> Path:
    """This clip's own chunk-1 CLIP-D-vs-LPIPS panel: 8-b SPATIAL curve, 8-b UNIFORM
    curve, this clip's own oracle frontier, R31's 4 arms as stars (hollow = outside
    this clip's own envelope span, shaded green), matching r31_html_report.build_video_
    pareto's structure with the x-metric fixed to chunk-1 CLIP-D and the envelope shading
    added.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))

    xs = [p for p, _, _ in spatial_curve]
    ys = [c for _, c, _ in spatial_curve]
    ax.plot(xs, ys, "-o", color="black", label="constant-$b$ SPATIAL ($\\tau_{bg}$=0)",
           zorder=3)
    for p, c, b in spatial_curve:
        ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(3, 3),
                    fontsize=6.5)

    if uniform_curve:
        uxs = [p for p, _, _ in uniform_curve]
        uys = [c for _, c, _ in uniform_curve]
        ax.plot(uxs, uys, marker="D", color="dimgray", linestyle="--", markersize=4,
               label="uniform baseline ($b_{bg}=b_{fg}$)", zorder=2)

    # This clip's own SPATIAL Pareto envelope span -- the step's main job: shows WHERE
    # a paired gap is even defined, rather than leaving that as an unstated caveat.
    if len(env) >= 2:
        ax.axvspan(env[0][0], env[-1][0], color="tab:green", alpha=0.08, zorder=0,
                  label="this clip's own envelope span")

    cmap = plt.get_cmap("tab10")
    for i, arm in enumerate(R31_ARMS):
        if arm not in arm_points:
            continue
        ax_val, ay_val, in_range, gap = arm_points[arm]
        lab = f"r31_{arm} (lpips={ay_val:.3f}"
        lab += f", gap {gap:+.4f})" if gap is not None else ", OUT OF RANGE)"
        if in_range:
            ax.scatter([ax_val], [ay_val], marker="*", s=220, color=cmap(i % 10),
                      edgecolor="black", linewidth=0.7, zorder=5, label=lab)
        else:
            ax.scatter([ax_val], [ay_val], marker="*", s=220, facecolors="none",
                      edgecolors=cmap(i % 10), linewidth=1.6, zorder=5, label=lab)

    if len(frontier) >= 2:
        fx = [c for _, c, _ in frontier]
        fy = [y for _, _, y in frontier]
        ax.plot(fx, fy, "-", color="tab:green", linewidth=1.6, zorder=1.5,
               label="per-clip oracle ($\\alpha$ sweep)")

    ax.set_xlabel(f"{clip_d_column}, chunk 1 (achieved edit)", fontsize=8)
    ax.set_ylabel("lpips_unedit_part, chunk 1 (lower = better preserved)", fontsize=8)
    ax.set_title(f"edit{edit_type} {video_name} -- chunk-1 CLIP-D vs. LPIPS "
                f"(this clip only)", fontsize=8.5)
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = out_dir / f"{case_id}_clipd_vs_lpips.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- html ------
def write_html(cases: List[dict], grids_by_case: Dict[str, Path],
              pareto_by_case: Dict[str, Path], gap_rows: List[Dict[str, object]],
              out_path: Path, root: Path, clip_d_column: str) -> None:
    def rel(p: Path) -> str:
        return str(p.relative_to(root))

    gap_table_rows = "\n".join(
        f"<tr><td>{g['arm']}</td>"
        f"<td>{g['mean_gap']}</td><td>&plusmn;{g['se_gap']}</td>"
        f"<td>{g['n_in_range']}</td><td>{g['n_out_of_range']}</td>"
        f"<td>{g['n_worse']}</td></tr>"
        for g in gap_rows)

    sections = []
    for c in sorted(cases, key=lambda c: (int(c["edit_type"]), c["video_name"])):
        cid, t, name = c["case_id"], int(c["edit_type"]), c["video_name"]
        grid = grids_by_case.get(cid)
        pareto = pareto_by_case.get(cid)
        grid_html = (f'<img class="grid-img" src="{rel(grid)}" alt="chunk-1 grid">'
                    if grid else "<p><em>no chunk-1 grid (see r34_chunk1_grids.py "
                                 "output for this clip)</em></p>")
        pareto_html = (f'<img class="pareto" src="{rel(pareto)}" alt="pareto">'
                      if pareto else "<p><em>no per-clip pareto (incomplete "
                                     "8-b spatial curve for this clip)</em></p>")
        prompt = f'{c.get("src_word", "")} \u2192 {c.get("trg_word", "")}'
        sections.append(f"""
<section class="video">
  <h2>edit{t} &mdash; {name} <span class="case-id">({cid})</span></h2>
  <p class="prompt">{prompt}</p>
  <div class="grid-wrap">{grid_html}</div>
  <div class="pareto-wrap">{pareto_html}</div>
</section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>R34 per-clip chunk-1 report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #f7f7f8; color: #1a1a1a; }}
  header {{ padding: 20px 24px; background: #1a1a1a; color: #fff; }}
  header p {{ margin: 4px 0 0; color: #c9c9c9; font-size: 13px; }}
  .gap-summary {{ background: #fff; margin: 20px auto; max-width: 1100px;
                 padding: 16px 22px; border-radius: 8px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  table.gap-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.gap-table th, table.gap-table td {{ border: 1px solid #ddd; padding: 5px 10px;
                                            text-align: right; }}
  table.gap-table th:first-child, table.gap-table td:first-child {{ text-align: left; }}
  .caveat {{ font-size: 12px; color: #a33; margin-top: 8px; }}
  .video {{ background: #fff; margin: 20px auto; max-width: 1100px; padding: 18px 22px;
           border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  .video h2 {{ margin: 0 0 2px; font-size: 17px; }}
  .video h2 .case-id {{ color: #888; font-size: 13px; font-weight: normal; }}
  .prompt {{ margin: 0 0 12px; color: #555; font-size: 13px; font-style: italic; }}
  .grid-wrap img.grid-img {{ display: block; width: 100%; border: 1px solid #eee; }}
  .pareto-wrap {{ margin-top: 14px; text-align: center; }}
  .pareto-wrap img.pareto {{ max-width: 560px; width: 100%; border: 1px solid #eee; }}
</style>
</head>
<body>
<header>
  <h1 style="margin:0;font-size:19px;">R34 &mdash; per-clip chunk-1 (pixel frames 0-8)
  qualitative grid + paired Pareto-envelope gap</h1>
  <p>Each section: that clip's own 6-row/9-column chunk-1 grid (source / uniform b=2
     baseline / R31's 4 arms) above its own chunk-1 {clip_d_column}-vs-LPIPS panel --
     8-b SPATIAL curve, 8-b UNIFORM curve, per-clip oracle frontier, and R31's 4 arms
     (hollow star = this clip's arm falls outside its OWN spatial envelope's x-span,
     so the paired gap is undefined for it here).</p>
</header>
<div class="gap-summary">
  <h2 style="margin-top:0;font-size:15px;">Paired per-clip gap vs. each clip's own
  spatial Pareto envelope</h2>
  <p style="font-size:12px;color:#555;margin-top:-6px;">gap = arm_lpips &minus;
  envelope_lpips(arm_clip_d), interpolated along that clip's own envelope. Positive =
  arm sits ABOVE the envelope (worse preserved at matched achievement). Mean/SE over
  only the clips where the arm's CLIP-D falls INSIDE that clip's own envelope span.</p>
  <table class="gap-table">
    <thead><tr><th>arm</th><th>mean gap</th><th>SE</th><th>n in range</th>
    <th>n out of range</th><th>n worse (gap&gt;0)</th></tr></thead>
    <tbody>{gap_table_rows}</tbody>
  </table>
  <p class="caveat">&#9888; n differs per arm and per window; the SE above is
  uncorrected for testing 4 arms at once on small n. Enough to say whether a
  displacement survives per-clip pairing, not yet a number for a paper.</p>
</div>
{"".join(sections)}
</body>
</html>"""
    out_path.write_text(html)


# --------------------------------------------------------------------------- main ------
def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})
    cdir = args.chunk1_csv_dir

    sp_lpips = load_r26_metric(cdir, idx2vid, name2case, CONST_BS, "lpips_unedit_part",
                              method_fmt=SPATIAL_FMT, stem_fmt=CHUNK1_STEM,
                              stride=CHUNK1_STRIDE)
    un_lpips = load_r26_metric(cdir, idx2vid, name2case, CONST_BS, "lpips_unedit_part",
                              method_fmt=UNIFORM_FMT, stem_fmt=CHUNK1_STEM,
                              stride=CHUNK1_STRIDE)

    cd_c1 = load_clip_d(args.clipd_csv, args.clip_d_column)

    def cd_by_b(fmt: str) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        for b in CONST_BS:
            m = f"r26_{fmt.format(b=b)}"
            for c, v in cd_c1.get(m, {}).items():
                out[(c, b)] = v
        return out

    sp_clipd = cd_by_b(SPATIAL_FMT)
    un_clipd = cd_by_b(UNIFORM_FMT)

    r31_lpips: Dict[str, Dict[str, float]] = {}
    r31_clipd: Dict[str, Dict[str, float]] = {}
    for arm in R31_ARMS:
        r31_lpips[arm] = load_stem_metric(cdir, idx2vid, name2case,
                                         f"r34_r31_{arm}_chunk1", CHUNK1_STRIDE,
                                         "lpips_unedit_part")
        r31_clipd[arm] = cd_c1.get(f"r31_{arm}", {})

    # ---- per-clip panels + paired gap collection -------------------------------------
    gap_records: Dict[str, List[Tuple[Optional[float], bool]]] = {a: [] for a in R31_ARMS}
    pareto_by_case: Dict[str, Path] = {}
    grids_by_case: Dict[str, Path] = {}

    for c in cases:
        cid, t, name = c["case_id"], int(c["edit_type"]), c["video_name"]

        grid = args.grids_dir / f"edit{t}_{name}_chunk1.png"
        if grid.exists():
            grids_by_case[cid] = grid
        else:
            print(f"[r34_report_figures] no chunk-1 grid for {cid} at {grid}")

        curve = [(sp_clipd[(cid, b)], sp_lpips[(cid, b)], b) for b in CONST_BS
                if (cid, b) in sp_clipd and (cid, b) in sp_lpips]
        if len(curve) < len(CONST_BS):
            print(f"[r34_report_figures] SKIP pareto {cid}: incomplete spatial curve "
                 f"({len(curve)}/{len(CONST_BS)} b's)")
            continue
        uniform_curve = [(un_clipd[(cid, b)], un_lpips[(cid, b)], b) for b in CONST_BS
                         if (cid, b) in un_clipd and (cid, b) in un_lpips]
        env = pareto_envelope(curve)
        frontier = oracle_frontier([cid], sp_clipd, sp_lpips, CONST_BS,
                                  n_alpha=args.n_alpha, higher_is_better=False)

        arm_points: Dict[str, ArmPoint] = {}
        for arm in R31_ARMS:
            if cid not in r31_clipd.get(arm, {}) or cid not in r31_lpips.get(arm, {}):
                continue
            ax_val = r31_clipd[arm][cid]
            ay_val = r31_lpips[arm][cid]
            env_y = interp_envelope(env, ax_val)
            in_range = env_y is not None
            gap = (ay_val - env_y) if in_range else None
            gap_records[arm].append((gap, in_range))
            arm_points[arm] = (ax_val, ay_val, in_range, gap)

        try:
            pareto_by_case[cid] = build_video_pareto(
                cid, name, t, curve, uniform_curve, frontier, env, arm_points,
                args.pareto_out, args.clip_d_column)
        except Exception as ex:
            print(f"[r34_report_figures] PARETO FAILED {cid}: {type(ex).__name__}: {ex}")

    # ---- paired-gap CSV ---------------------------------------------------------------
    gap_rows: List[Dict[str, object]] = []
    for arm in R31_ARMS:
        recs = gap_records[arm]
        in_range_gaps = [g for g, ir in recs if ir]
        n_in, n_out = len(in_range_gaps), len(recs) - len(in_range_gaps)
        n_worse = sum(1 for g in in_range_gaps if g > 0)
        mean = st.mean(in_range_gaps) if in_range_gaps else None
        se = (st.stdev(in_range_gaps) / (n_in ** 0.5)) if n_in > 1 else None
        gap_rows.append({
            "arm": f"r31_{arm}",
            "n_clips": len(recs),
            "n_in_range": n_in,
            "n_out_of_range": n_out,
            "n_worse": n_worse,
            "mean_gap": round(mean, 6) if mean is not None else "",
            "se_gap": round(se, 6) if se is not None else "",
        })

    args.gap_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.gap_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gap_rows[0].keys()))
        w.writeheader()
        w.writerows(gap_rows)
    print(f"[r34_report_figures] wrote {args.gap_csv} ({len(gap_rows)} arms)")

    print(f"\n{'arm':<16} {'mean_gap':>10} {'se':>9} {'n_in':>5} {'n_out':>6} "
         f"{'n_worse':>8}")
    for r in gap_rows:
        print(f"{r['arm']:<16} {r['mean_gap']:>10} {r['se_gap']:>9} "
             f"{r['n_in_range']:>5} {r['n_out_of_range']:>6} {r['n_worse']:>8}")

    # ---- html ---------------------------------------------------------------------
    write_html(cases, grids_by_case, pareto_by_case, gap_rows, args.out,
              root=args.out.parent, clip_d_column=args.clip_d_column)
    print(f"\n[r34_report_figures] wrote {args.out} ({len(grids_by_case)} grids, "
         f"{len(pareto_by_case)} pareto panels, {len(clips)} clips total)")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                  default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--chunk1_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--clipd_csv", type=Path,
                  default=Path("evaluation/csv/r34_clip_directional_chunk1.csv"))
    p.add_argument("--clip_d_column", type=str, default="clip_d_prompt",
                  choices=("clip_d_prompt", "clip_d_word"))
    p.add_argument("--grids_dir", type=Path,
                  default=Path("evaluation/figures/r34_chunk1_grids"))
    p.add_argument("--pareto_out", type=Path,
                  default=Path("evaluation/figures/r34_video_pareto"))
    p.add_argument("--gap_csv", type=Path,
                  default=Path("evaluation/csv/r34_perclip_gap.csv"))
    p.add_argument("--out", type=Path, default=Path("evaluation/figures/r34_report.html"))
    p.add_argument("--n_alpha", type=int, default=21)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
