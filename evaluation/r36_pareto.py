#!/usr/bin/env python
"""R36 figures -- the VP rho sweep as a trade-off curve, three figures.

x = semantic alignment (one per figure), y = preservation (lpips_unedit_part, lower = better):
    r36_cliptgt_vs_lpips.pdf   x = clip_similarity_target_image  (CLIP-T)
    r36_clipd_vs_lpips.pdf     x = clip_d_prompt                 (CLIP-D, R33's axis of record)
    r36_fiveacc_vs_lpips.pdf   x = five_acc_yes_no               (FiVE-Acc, yes/no)

Series:
* R36 full bench  -- r36_rho_sweep.csv subset=full (419 pairs), 8 rho joined in rho
  order, each point labelled with its rho.
* R36 22 clips    -- r36_rho_sweep.csv subset=cases22, same rho, dashed.
* R26 22 clips    -- R26's uniform diagonal taubg{R}_taufg{R}_vp on the same 22 clips,
  markers only. R26 and R36 run the same scalar blend path, per-pair seed and sampler,
  so these must sit on the dashed curve; the gap is printed per rho as a check.

Every point is a plain per-clip mean (each clip weight 1). R26's values are computed here
from its per-clip CSVs:
    edit{T}_FiVE_r26_taubg{R}_taufg{R}_vp_frame_stride8.csv          (lpips, CLIP-T)
    edit{T}_FiVE_r33_fiveacc_taubg{R}_taufg{R}_vp_frame_stride8.csv  (FiVE-Acc)
    r33_clip_directional.csv, method == r26_taubg{R}_taufg{R}_vp     (CLIP-D)

Also writes r36_pareto_points.csv: every plotted point (the table view of the figures).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r35_score import clip_index  # noqa: E402

RHOS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 20, 50)
Y_METRIC = "lpips_unedit_part"
FIGURES: Dict[str, Tuple[str, str, str]] = {
    "cliptgt": ("clip_similarity_target_image", "CLIP-T (target image)  → more edit",
                "r36_cliptgt_vs_lpips.pdf"),
    "clipd":   ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit",
                "r36_clipd_vs_lpips.pdf"),
    "fiveacc": ("five_acc_yes_no", "FiVE-Acc (yes/no)  → more edit",
                "r36_fiveacc_vs_lpips.pdf"),
}
METRICS: Tuple[str, ...] = (Y_METRIC, *(v[0] for v in FIGURES.values()))

# Categorical slots 1-2 of the reference palette (validated light mode: CVD ΔE 24.7,
# normal ΔE 33.6, contrast >= 3:1 on SURFACE); R26 is a reference series -> neutral gray.
FULL_COLOR, SUB_COLOR, REF_COLOR = "#2a78d6", "#eb6834", "#8c8c87"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b67", "#e6e6e3", "#fcfcfb"

Key = Tuple[int, str]


# ------------------------------------------------------------------ inputs -----------

def load_sweep(path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    """{subset: {rho: {metric: value}}} from r36_rho_sweep.csv, one row per (subset, rho)."""
    if not path.is_file():
        raise SystemExit(f"[r36_pareto] missing {path} -- run the summarize step first")
    out: Dict[str, Dict[int, Dict[str, float]]] = {"full": {}, "cases22": {}}
    for r in csv.DictReader(path.open()):
        sub, rho = r["subset"], int(r["rho"])
        if sub not in out:
            continue
        if rho in out[sub]:
            raise SystemExit(f"[r36_pareto] {path.name}: duplicate row subset={sub} rho={rho}")
        out[sub][rho] = {m: float(r[m]) for m in METRICS}
    for sub, n_exp in (("full", 419), ("cases22", 22)):
        if sorted(out[sub]) != sorted(RHOS):
            raise SystemExit(f"[r36_pareto] {path.name}: subset={sub} has rho "
                             f"{sorted(out[sub])}, expected {list(RHOS)}")
    return out


def _per_clip_csv(csv_dir: Path, stem: str, metric: str,
                  clips: Dict[Tuple[int, str], str], keys: List[Key]) -> Dict[Key, float]:
    out: Dict[Key, float] = {}
    for t in sorted({k[0] for k in keys}):
        path = csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv"
        if not path.is_file():
            raise SystemExit(f"[r36_pareto] missing {path}")
        for r in csv.DictReader(path.open()):
            col = next((k for k in r if k == metric or k.endswith(f"|{metric}")), None)
            if col is None:
                raise SystemExit(f"[r36_pareto] {path.name}: no '{metric}' column")
            out[(t, clips[(t, r["file_id"])])] = float(r[col])
    missing = set(keys) - set(out)
    if missing:
        raise SystemExit(f"[r36_pareto] {stem}:{metric} lacks {len(missing)} of the "
                         f"{len(keys)} cases.json clips: {sorted(missing)[:3]}...")
    return {k: out[k] for k in keys}


def _clipd_r33(path: Path, method: str, keys: List[Key]) -> Dict[Key, float]:
    out = {(int(r["edit_type"]), r["video_name"]): float(r["clip_d_prompt"])
           for r in csv.DictReader(path.open()) if r["method"] == method}
    missing = set(keys) - set(out)
    if missing:
        raise SystemExit(f"[r36_pareto] {path.name}:{method} lacks {len(missing)} clips")
    return {k: out[k] for k in keys}


def r26_points(args: argparse.Namespace, clips: Dict[Tuple[int, str], str],
               keys: List[Key]) -> Dict[int, Dict[str, float]]:
    """{rho: {metric: plain mean over the 22 clips}} for R26's uniform diagonal."""
    pts: Dict[int, Dict[str, float]] = {}
    for rho in RHOS:
        arm = f"taubg{rho}_taufg{rho}_vp"
        per = {
            Y_METRIC: _per_clip_csv(args.csv_dir, f"r26_{arm}", Y_METRIC, clips, keys),
            "clip_similarity_target_image":
                _per_clip_csv(args.csv_dir, f"r26_{arm}", "clip_similarity_target_image",
                              clips, keys),
            "five_acc_yes_no":
                _per_clip_csv(args.csv_dir, f"r33_fiveacc_{arm}", "five_acc_yes_no",
                              clips, keys),
            "clip_d_prompt": _clipd_r33(args.r33_clipd, f"r26_{arm}", keys),
        }
        pts[rho] = {m: float(np.mean(list(v.values()))) for m, v in per.items()}
    return pts


# ------------------------------------------------------------------ figure -----------

def plot(key: str, sweep: Dict[str, Dict[int, Dict[str, float]]],
         r26: Dict[int, Dict[str, float]], outdir: Path) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    full = [sweep["full"][r] for r in RHOS]
    sub = [sweep["cases22"][r] for r in RHOS]
    ax.plot([p[x_metric] for p in full], [p[Y_METRIC] for p in full], color=FULL_COLOR,
            lw=2, marker="o", ms=6, zorder=3, label="R36 · full bench (419 pairs)")
    for p, rho in zip(full, RHOS):
        ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                    textcoords="offset points", fontsize=7, color=MUTED,
                    ha="right", va="bottom")
    ax.plot([p[x_metric] for p in sub], [p[Y_METRIC] for p in sub], color=SUB_COLOR,
            lw=2, ls="--", marker="s", ms=7, mfc=SURFACE, mec=SUB_COLOR, mew=2, zorder=2,
            label="R36 · 22 cases.json clips")
    ax.scatter([r26[r][x_metric] for r in RHOS], [r26[r][Y_METRIC] for r in RHOS],
               marker="x", s=40, color=REF_COLOR, linewidths=2, zorder=4,
               label="R26 uniform diagonal · same 22 clips")

    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    ax.set_title("R36 — StreamEdit + VP, ρ sweep (per-clip mean)", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="best")

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / fname
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def write_points(path: Path, sweep: Dict[str, Dict[int, Dict[str, float]]],
                 r26: Dict[int, Dict[str, float]]) -> None:
    rows = [{"series": s, "rho": rho, **vals[rho]}
            for s, vals in (("r36_full", sweep["full"]), ("r36_cases22", sweep["cases22"]),
                            ("r26_cases22", r26))
            for rho in RHOS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "rho", *METRICS])
        w.writeheader()
        w.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=Path, default=Path("evaluation/csv/r36_rho_sweep.csv"))
    p.add_argument("--outdir", type=Path, default=Path("evaluation/figures"))
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--r33_clipd", type=Path,
                   default=Path("evaluation/csv/r33_clip_directional.csv"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--points_csv", type=Path,
                   default=Path("evaluation/csv/r36_pareto_points.csv"))
    args = p.parse_args(argv)
    args.data_root = args.data_root.expanduser()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    clips = clip_index(args.data_root)
    keys = sorted({(int(c["edit_type"]), c["video_name"])
                   for c in json.loads(args.cases.read_text())})
    sweep = load_sweep(args.sweep)
    r26 = r26_points(args, clips, keys)

    # R36 and R26 share renders on these 22 clips (check-parity), so this should be ~0.
    print("[r36_pareto] |R36 cases22 - R26 diagonal| per rho:")
    for rho in RHOS:
        d = {m: abs(sweep["cases22"][rho][m] - r26[rho][m]) for m in METRICS}
        print(f"  rho={rho:<3} " + "  ".join(f"{m}={v:.2e}" for m, v in d.items()))

    write_points(args.points_csv, sweep, r26)
    print(f"[r36_pareto] wrote {args.points_csv}")
    for k in FIGURES:
        print(f"[r36_pareto] wrote {plot(k, sweep, r26, args.outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
