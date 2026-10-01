#!/usr/bin/env python
"""R34 -- the first-chunk (pixel frames 0-8) trade-off picture for R26's constant-b
SPATIAL curve, R26's UNIFORM baseline, R31's 4 divergence arms and a per-clip oracle,
on three axes: LPIPS, CLIP-target and CLIP-D. Local, no GPU.

Structure mirrors r31_score.py, and the reference machinery (``CONST_BS``,
``build_join_tables``, ``load_r26_metric``, ``oracle_frontier``) is imported from
r30_score.py rather than reimplemented, so R34's points are read against the identical
axes and joins R30/R31 used.

WHAT IS DIFFERENT FROM R30/R31, AND WHY IT MATTERS
---------------------------------------------------
The WINDOW. R26/R30/R31 scored whole-video at --frame_stride 8. R34 re-scores the FIRST
ROLLOUT CHUNK only: num_frame_per_block=3 latent frames = pixel frames 0-8, at stride 1.
That grid is not a subset of the stride-8 grid (which inside this window is just {0, 8}),
so chunk-1 and whole-video numbers are NOT two views of the same frames and neither is a
subset of the other. This script therefore keeps them in SEPARATE columns
(chunk1_* / whole_*) and reports the delta and the rank correlation between the two
windows, rather than presenting either as a refinement of the other.

MEASURED ON THIS DATA (2026-09-22, the run this script consumes): the two windows do not
behave alike. Preservation keeps its spread -- LPIPS across the 20 arms spans 0.0618 on
chunk 1 vs 0.0673 whole-video (0.92x) -- while BOTH achievement axes roughly halve:
CLIP-target 0.5929 vs 1.3899 (0.43x), CLIP-D 0.0359 vs 0.0746 (0.48x). So the arms
separate on preservation about as well as before but on achievement half as well. The
practical consequence, and the reason `oracle_gap` is computed and printed as a headline
column here: at half the achievement spread, the ORDERING of arms is the fragile readout,
while the per-clip oracle GAP (how much an all-knowing per-clip choice of b buys over the
best fixed b) is a distance and stays readable.

TWO NAMING CONVENTIONS COLLIDE ON R31'S ARMS -- the one real trap in this join. R31's
stored whole-video CSVs use the BARE arm name (edit{T}_FiVE_r31_lpips_frame_stride8.csv),
R34's chunk-1 CSVs use it nested (edit{T}_FiVE_r34_r31_lpips_chunk1_frame_stride1.csv),
and BOTH CLIP-D tables label that same arm "r31_lpips". Feeding the wrong one to a shared
formatter yields "r31_r31_lpips", which matches no file -- and because a missing arm
loads as an empty dict rather than an error, it would silently produce nan for all four
R31 rows and a table that still looks complete. Hence `load_stem_metric` takes a fully
resolved stem and every stem is spelled out at its call site.

AXIS OF RECORD. R33 measured clip_similarity_target_image not to rank these renders by
edit strength (mean Spearman(b, clip_target) = +0.098, positive on 11/22 clips) and chose
clip_d_prompt instead (+0.4080, 17/22, 0 flat clips). Both are computed here; CLIP-D is
the axis the figures lead with. On the chunk-1 window the case is if anything stronger:
CLIP-D stays monotone in b across the 8 spatial arms while chunk-1 CLIP-target does not.

Usage
-----
    python evaluation/r34_score.py \\
        --chunk1_csv_dir evaluation/csv \\
        --clip_d_chunk1 evaluation/csv/r34_clip_directional_chunk1.csv \\
        --clip_d_whole evaluation/csv/r33_clip_directional.csv \\
        -o evaluation/csv/r34_arms.csv --fig_dir evaluation/figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import (  # noqa: E402
    CONST_BS,
    build_join_tables,
    load_r26_metric,
    #✨ 2026-09-22: load_stem_metric MOVED to r30_score.py (the shared base) so
    # r31_score.py / r33_score.py can reach it without importing this module. Re-exported
    # here unchanged -- this module's own call sites and r34_report_figures.py's
    # `from r34_score import load_stem_metric` both keep working.
    load_stem_metric,
    oracle_frontier,
)

R31_ARMS: Sequence[str] = ("lpips", "dino_patch", "normals", "latent")

# Stem conventions, spelled out once. See the docstring's naming-collision note.
CHUNK1_STEM = "r34_{method}_chunk1"      # + _frame_stride1
CHUNK1_STRIDE = 1
WHOLE_R26_STEM = "r26_{method}"          # + _frame_stride8
WHOLE_STRIDE = 8

SPATIAL_FMT = "taubg0_taufg{b}_vp"
UNIFORM_FMT = "taubg{b}_taufg{b}_vp"


# --------------------------------------------------------------------------- loaders ---
def load_clip_d(path: Path, column: str = "clip_d_prompt") -> Dict[str, Dict[str, float]]:
    """{method: {case_id: value}} -- CLIP-D tables already carry case_id, so no join."""
    out: Dict[str, Dict[str, float]] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["method"], {})[r["case_id"]] = float(r[column])
    return out


def mean_over(d: Dict[str, float], clips: Sequence[str]) -> Optional[float]:
    vals = [d[c] for c in clips if c in d]
    return st.mean(vals) if vals else None


# ------------------------------------------------------------------------ statistics ---
def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rho with average ranks for ties; None if either side is flat.

    Local rather than scipy: r33_axis_compare.py hit exactly this and found scipy returns
    nan on a flat input, which then silently propagates through a mean. Returning None
    forces the caller to decide.
    """
    n = len(xs)
    if n < 2:
        return None

    def rank(v: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    if len(set(rx)) == 1 or len(set(ry)) == 1:
        return None
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def pareto_envelope(curve: Sequence[Tuple[float, float, object]]
                    ) -> List[Tuple[float, float]]:
    """Non-dominated (max achievement, min preservation-cost) points, sorted by x.

    A fixed-b point is dominated when some other b reaches at least as much achievement
    at no greater LPIPS. Dominated b's are exactly the settings nobody would choose, so
    they must not define the reference the oracle is measured against.
    """
    pts = sorted(((x, y) for x, y, _ in curve), key=lambda p: (p[0], p[1]))
    keep: List[Tuple[float, float]] = []
    best_y = float("inf")
    for x, y in reversed(pts):          # right to left: keep a point only if it is
        if y < best_y:                  # cheaper than everything achieving more
            keep.append((x, y))
            best_y = y
    return sorted(keep)


def oracle_gap(frontier: Sequence[Tuple[float, float, float]],
               curve: Sequence[Tuple[float, float, object]]
               ) -> Tuple[Optional[float], int, int]:
    """(mean vertical gap, n_points_used, n_points_skipped) vs the curve's Pareto envelope.

    The headline number when the achievement axis is compressed: a DISTANCE on the
    preservation axis at matched achievement, so it survives a squeezed x-range that makes
    the ORDER of arms unreliable. Positive = the oracle preserves better at the same
    achievement, i.e. what a per-clip choice of b buys over the best fixed b.

    ⚠️ MEASURED ON THIS DATA, AND THE REASON THIS IS NOT A PLAIN INTERPOLATION: the
    fixed-b curve is NOT monotone in achievement. Chunk-1 CLIP-target rises to 28.342 at
    b=8 then falls back to 28.046 at b=50; chunk-1 CLIP-D has two smaller inversions. An
    earlier version of this function sorted the raw points by x and interpolated between
    consecutive sorted pairs, which on a non-monotone curve interpolates between points
    that are not adjacent along it. With several near-equal x values the pairing was
    decided by the last bits of statistics.mean -- which changed between Python 3.11 and
    3.12 -- and the CLIP-target gap read 0.0303 under one interpreter and 0.0253 under
    another from byte-identical inputs. Reducing to the Pareto envelope first removes the
    ambiguity: the envelope is monotone by construction, so the interpolation is
    well-defined and the result is interpreter-independent.

    Frontier points beyond the envelope's x-range are skipped, not extrapolated, and the
    count is returned so a gap averaged over few points cannot be mistaken for one
    averaged over many.

    ⚠️ AND ON THIS DATA THAT CAVEAT BITES, which is why n_points/n_skipped are written to
    the CSV rather than kept internal. The per-clip oracle reaches achievement levels no
    fixed b attains -- on chunk-1 CLIP-target the oracle runs to 29.27 while the best
    fixed b reaches 28.34 -- so 15 of 21 frontier points fall outside the envelope
    entirely and the "gap" is averaged over the 6 that remain (CLIP-D: 10 of 21 usable).
    A residual cross-interpreter wobble of ~2e-4 survives on the 6-point CLIP-target
    average, from last-bit statistics.mean differences between Python 3.11 and 3.12 that
    a 6-point mean cannot absorb; CLIP-D's 10-point average is stable to all printed
    digits. Read the CLIP-target gap as "small and weakly determined", not as a measured
    quantity, and prefer the CLIP-D column.
    """
    env = pareto_envelope(curve)
    if len(env) < 2:
        return None, 0, len(frontier)
    gaps = []
    skipped = 0
    for _, fx, fy in frontier:
        if fx < env[0][0] or fx > env[-1][0]:
            skipped += 1
            continue
        for (x0, y0), (x1, y1) in zip(env, env[1:]):
            if x0 <= fx <= x1:
                t = 0.0 if x1 == x0 else (fx - x0) / (x1 - x0)
                gaps.append((y0 + t * (y1 - y0)) - fy)
                break
    return (st.mean(gaps) if gaps else None), len(gaps), skipped


# ------------------------------------------------------------------------------ run ---
def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})
    cdir = args.chunk1_csv_dir

    print(f"[r34_score] {len(clips)} clips; chunk-1 window = pixel frames 0-8 at stride 1")

    # ---- per-(clip, b) tables, both windows, for the two b-swept families ------------
    # CLIP-D is keyed by method NAME, not (method_fmt, b), so it is reshaped to the same
    # (clip, b) keying here to go through oracle_frontier unchanged.
    cd_c1 = load_clip_d(args.clip_d_chunk1, args.clip_d_column)
    cd_wv = load_clip_d(args.clip_d_whole, args.clip_d_column)

    def cd_by_b(table: Dict[str, Dict[str, float]], fmt: str) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        for b in CONST_BS:
            m = f"r26_{fmt.format(b=b)}"
            for c, v in table.get(m, {}).items():
                out[(c, b)] = v
        return out

    fam: Dict[str, Dict[str, object]] = {}
    for label, fmt in (("spatial", SPATIAL_FMT), ("uniform", UNIFORM_FMT)):
        fam[label] = {
            "c1_lpips": load_r26_metric(cdir, idx2vid, name2case, CONST_BS,
                                        "lpips_unedit_part", method_fmt=fmt,
                                        stem_fmt=CHUNK1_STEM, stride=CHUNK1_STRIDE),
            "c1_clip_target": load_r26_metric(cdir, idx2vid, name2case, CONST_BS,
                                              "clip_similarity_target_image",
                                              method_fmt=fmt, stem_fmt=CHUNK1_STEM,
                                              stride=CHUNK1_STRIDE),
            "c1_clip_d": cd_by_b(cd_c1, fmt),
            "wv_lpips": load_r26_metric(args.whole_csv_dir, idx2vid, name2case, CONST_BS,
                                        "lpips_unedit_part", method_fmt=fmt,
                                        stem_fmt=WHOLE_R26_STEM, stride=WHOLE_STRIDE),
            "wv_clip_target": load_r26_metric(args.whole_csv_dir, idx2vid, name2case,
                                              CONST_BS, "clip_similarity_target_image",
                                              method_fmt=fmt, stem_fmt=WHOLE_R26_STEM,
                                              stride=WHOLE_STRIDE),
            "wv_clip_d": cd_by_b(cd_wv, fmt),
        }

    # ---- rows: 8 spatial + 8 uniform + 4 R31 ----------------------------------------
    rows: List[Dict[str, object]] = []
    for label, fmt in (("spatial", SPATIAL_FMT), ("uniform", UNIFORM_FMT)):
        f = fam[label]
        for b in CONST_BS:
            row: Dict[str, object] = {"method": f"r26_{fmt.format(b=b)}",
                                      "family": label, "b": b, "n_clips": len(clips)}
            for key in ("lpips", "clip_target", "clip_d"):
                c1 = st.mean([f[f"c1_{key}"][(c, b)] for c in clips
                              if (c, b) in f[f"c1_{key}"]])
                wvv = [f[f"wv_{key}"][(c, b)] for c in clips if (c, b) in f[f"wv_{key}"]]
                wv = st.mean(wvv) if wvv else None
                row[f"chunk1_{key}"] = round(c1, 6)
                row[f"whole_{key}"] = round(wv, 6) if wv is not None else ""
                row[f"delta_{key}"] = round(c1 - wv, 6) if wv is not None else ""
            rows.append(row)

    for arm in R31_ARMS:
        row = {"method": f"r31_{arm}", "family": "r31", "b": "", "n_clips": len(clips)}
        pairs = {
            "lpips": ("lpips_unedit_part", f"r34_r31_{arm}_chunk1", f"r31_{arm}"),
            "clip_target": ("clip_similarity_target_image", f"r34_r31_{arm}_chunk1",
                            f"r31_{arm}"),
        }
        for key, (metric, c1_stem, wv_stem) in pairs.items():
            c1 = mean_over(load_stem_metric(cdir, idx2vid, name2case, c1_stem,
                                            CHUNK1_STRIDE, metric), clips)
            wv = mean_over(load_stem_metric(args.whole_csv_dir, idx2vid, name2case,
                                            wv_stem, WHOLE_STRIDE, metric), clips)
            row[f"chunk1_{key}"] = round(c1, 6) if c1 is not None else ""
            row[f"whole_{key}"] = round(wv, 6) if wv is not None else ""
            row[f"delta_{key}"] = (round(c1 - wv, 6)
                                   if c1 is not None and wv is not None else "")
        c1d = mean_over(cd_c1.get(f"r31_{arm}", {}), clips)
        wvd = mean_over(cd_wv.get(f"r31_{arm}", {}), clips)
        row["chunk1_clip_d"] = round(c1d, 6) if c1d is not None else ""
        row["whole_clip_d"] = round(wvd, 6) if wvd is not None else ""
        row["delta_clip_d"] = (round(c1d - wvd, 6)
                               if c1d is not None and wvd is not None else "")
        rows.append(row)

    # ---- curves + oracle frontiers (spatial family, per the plan's Decisions) --------
    sp = fam["spatial"]
    un = fam["uniform"]

    def curve(t: Dict[Tuple[str, int], float], y: Dict[Tuple[str, int], float]):
        return [(st.mean([t[(c, b)] for c in clips if (c, b) in t]),
                 st.mean([y[(c, b)] for c in clips if (c, b) in y]), b)
                for b in CONST_BS
                if any((c, b) in t for c in clips) and any((c, b) in y for c in clips)]

    curves = {
        "clip_d": (curve(sp["c1_clip_d"], sp["c1_lpips"]),
                   curve(un["c1_clip_d"], un["c1_lpips"]),
                   oracle_frontier(clips, sp["c1_clip_d"], sp["c1_lpips"], CONST_BS,
                                   n_alpha=args.n_alpha, higher_is_better=False)),
        "clip_target": (curve(sp["c1_clip_target"], sp["c1_lpips"]),
                        curve(un["c1_clip_target"], un["c1_lpips"]),
                        oracle_frontier(clips, sp["c1_clip_target"], sp["c1_lpips"],
                                        CONST_BS, n_alpha=args.n_alpha,
                                        higher_is_better=False)),
    }

    # ---- window comparison -----------------------------------------------------------
    delta_rows: List[Dict[str, object]] = []
    for key, lab in (("lpips", "lpips_unedit_part"),
                     ("clip_target", "clip_similarity_target_image"),
                     ("clip_d", args.clip_d_column)):
        pairs = [(r[f"chunk1_{key}"], r[f"whole_{key}"]) for r in rows
                 if r[f"chunk1_{key}"] != "" and r[f"whole_{key}"] != ""]
        c1v = [float(a) for a, _ in pairs]
        wvv = [float(b) for _, b in pairs]
        rho = spearman(c1v, wvv) if pairs else None
        c1_spread = (max(c1v) - min(c1v)) if c1v else None
        wv_spread = (max(wvv) - min(wvv)) if wvv else None
        d = {
            "metric": lab,
            "n_arms_paired": len(pairs),
            "chunk1_min": round(min(c1v), 6) if c1v else "",
            "chunk1_max": round(max(c1v), 6) if c1v else "",
            "chunk1_spread": round(c1_spread, 6) if c1_spread is not None else "",
            "whole_min": round(min(wvv), 6) if wvv else "",
            "whole_max": round(max(wvv), 6) if wvv else "",
            "whole_spread": round(wv_spread, 6) if wv_spread is not None else "",
            "spread_ratio": (round(c1_spread / wv_spread, 4)
                             if c1_spread is not None and wv_spread else ""),
            "spearman_chunk1_vs_whole": round(rho, 4) if rho is not None else "",
        }
        if key in curves:
            sc, uc, fr = curves[key]
            g_s, n_s, skip_s = oracle_gap(fr, sc)
            g_u, _, _ = oracle_gap(fr, uc)
            d["oracle_gap_vs_spatial_lpips"] = round(g_s, 6) if g_s is not None else ""
            d["oracle_gap_vs_uniform_lpips"] = round(g_u, 6) if g_u is not None else ""
            d["oracle_gap_n_points"] = n_s
            d["oracle_gap_n_skipped"] = skip_s
            d["spatial_pareto_b_kept"] = "|".join(
                str(b) for x, _, b in sorted(sc, key=lambda p: p[0])
                if (x, ) in {(px, ) for px, _ in pareto_envelope(sc)})
        delta_rows.append(d)

    # ---- write ------------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[r34_score] wrote {args.out} ({len(rows)} arms)")

    delta_path = args.out.parent / args.delta_name
    fields = sorted({k for d in delta_rows for k in d},
                    key=lambda k: (k != "metric", k))
    with delta_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(delta_rows)
    print(f"[r34_score] wrote {delta_path} ({len(delta_rows)} metrics)")

    print(f"\n{'method':<26} {'fam':<8} {'c1_lpips':>9} {'c1_clipT':>9} {'c1_clipD':>9} "
          f"{'d_clipD':>9}")
    for r in rows:
        print(f"{r['method']:<26} {r['family']:<8} {r['chunk1_lpips']:>9} "
              f"{r['chunk1_clip_target']:>9} {r['chunk1_clip_d']:>9} "
              f"{r['delta_clip_d']:>9}")

    print(f"\n{'metric':<32} {'c1_spread':>10} {'wv_spread':>10} {'ratio':>7} "
          f"{'rho(c1,wv)':>11} {'oracle_gap':>11}")
    for d in delta_rows:
        print(f"{d['metric']:<32} {d['chunk1_spread']:>10} {d['whole_spread']:>10} "
              f"{d['spread_ratio']:>7} {d['spearman_chunk1_vs_whole']:>11} "
              f"{d.get('oracle_gap_vs_spatial_lpips', ''):>11}")

    if args.fig_dir is not None:
        make_figures(args.fig_dir, rows, curves, delta_rows, args.clip_d_column)
    return 0


# -------------------------------------------------------------------------- figures ---
def make_figures(fig_dir: Path, rows, curves, delta_rows, clip_d_column: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    r31_rows = [r for r in rows if r["family"] == "r31"]

    # Axis convention inherited from r26_tradeoff_figure.py / r30_score.py / r31_score.py:
    # achievement on x, preservation on y. Only the achievement DEFINITION changes here.
    def panel(filename, title, x_label, x_key):
        spatial_curve, uniform_curve, frontier = curves[x_key]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot([p for p, _, _ in spatial_curve], [c for _, c, _ in spatial_curve],
                "-o", color="black",
                label="constant-$b$ SPATIAL curve (R26, $\\tau_{bg}$=0)", zorder=3)
        for p, c, b in spatial_curve:
            ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(4, 4),
                        fontsize=8)
        ax.plot([p for p, _, _ in uniform_curve], [c for _, c, _ in uniform_curve],
                marker="D", color="dimgray", linestyle="--",
                label="uniform baseline ($b_{bg}=b_{fg}$, Eq. 4, R26)", zorder=2)
        for p, c, b in uniform_curve:
            ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(4, -10),
                        fontsize=8, color="dimgray")
        for i, r in enumerate(r31_rows):
            if r[f"chunk1_{x_key}"] == "" or r["chunk1_lpips"] == "":
                continue
            ax.scatter([float(r[f"chunk1_{x_key}"])], [float(r["chunk1_lpips"])],
                       marker="*", s=260, color=cmap(i % 10), edgecolor="black",
                       linewidth=0.8, zorder=5, label=f"{r['method']}")
        fx = [c for _, c, _ in frontier]
        fy = [y for _, _, y in frontier]
        ax.plot(fx, fy, "-", color="tab:green", linewidth=2, zorder=1.5,
                label="per-clip oracle ($\\alpha$ sweep, 8-$b$ grid)")
        ax.annotate("$\\alpha$=0\n(LPIPS)", (fx[0], fy[0]), fontsize=7, color="tab:green")
        ax.annotate("$\\alpha$=1\n(achv.)", (fx[-1], fy[-1]), fontsize=7,
                    color="tab:green")
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean lpips_unedit_part  (lower = better preserved)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = fig_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    p1 = panel("r34_clipd_vs_lpips.pdf",
               "R34 first chunk (pixel frames 0-8): CLIP-D vs. LPIPS",
               f"mean {clip_d_column}  (higher = more achieved)", "clip_d")
    p2 = panel("r34_cliptgt_vs_lpips.pdf",
               "R34 first chunk (pixel frames 0-8): CLIP-target vs. LPIPS\n"
               "(axis R33 retired -- shown for continuity with R26/R30/R31)",
               "mean clip_similarity_target_image  (higher = more achieved)",
               "clip_target")

    # Window panel: chunk-1 vs whole-video per arm per metric, with the identity line.
    # An arm off the line is one whose first chunk does not represent its clip. Each
    # metric is min-max normalised over its own 20 arms, since LPIPS (~0.1) and
    # CLIP-target (~28) cannot share a raw axis.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (key, lab) in zip(axes, (("lpips", "lpips_unedit_part"),
                                     ("clip_target", "clip_similarity_target_image"),
                                     ("clip_d", clip_d_column))):
        pts = [(float(r[f"whole_{key}"]), float(r[f"chunk1_{key}"]), r["family"])
               for r in rows if r[f"whole_{key}"] != "" and r[f"chunk1_{key}"] != ""]
        if not pts:
            ax.set_visible(False)
            continue
        allv = [v for p in pts for v in p[:2]]
        lo, hi = min(allv), max(allv)
        pad = (hi - lo) * 0.08 or 1e-6
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="gray",
                linewidth=1, label="identity (window makes no difference)")
        for fam_lab, colour, marker in (("spatial", "black", "o"),
                                        ("uniform", "dimgray", "D"),
                                        ("r31", "tab:orange", "*")):
            sel = [(x, y) for x, y, f in pts if f == fam_lab]
            if sel:
                ax.scatter([x for x, _ in sel], [y for _, y in sel], c=colour,
                           marker=marker, s=90 if marker == "*" else 34,
                           edgecolor="black" if marker == "*" else "none",
                           linewidth=0.6, label=fam_lab, zorder=3)
        d = next((x for x in delta_rows if x["metric"] == lab), {})
        ax.set_title(f"{lab}\nspread ratio {d.get('spread_ratio', 'n/a')}, "
                     f"rho {d.get('spearman_chunk1_vs_whole', 'n/a')}", fontsize=9)
        ax.set_xlabel("whole video (stride 8)")
        ax.set_ylabel("first chunk (frames 0-8)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    p3 = fig_dir / "r34_window_delta.pdf"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    print(f"[r34_score] wrote {p1}, {p2}, {p3}")


# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--chunk1_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--whole_csv_dir", type=Path, default=None,
                   help="Defaults to --chunk1_csv_dir; the stored stride-8 CSVs live "
                        "alongside R34's.")
    p.add_argument("--clip_d_chunk1", type=Path,
                   default=Path("evaluation/csv/r34_clip_directional_chunk1.csv"))
    p.add_argument("--clip_d_whole", type=Path,
                   default=Path("evaluation/csv/r33_clip_directional.csv"))
    p.add_argument("--clip_d_column", type=str, default="clip_d_prompt",
                   choices=("clip_d_prompt", "clip_d_word"))
    p.add_argument("--n_alpha", type=int, default=21)
    p.add_argument("--delta_name", type=str, default="r34_window_delta.csv")
    p.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r34_arms.csv"))
    a = p.parse_args(argv)
    if a.whole_csv_dir is None:
        a.whole_csv_dir = a.chunk1_csv_dir
    return a


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
