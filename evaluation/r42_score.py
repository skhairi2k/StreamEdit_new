#!/usr/bin/env python
"""R42 rescore -- R35's four arms vs R36's VP rho curve on FiVE-Bench-Shape and on the rest.

No new renders: every number comes from per-clip CSVs that R35 and R36 already wrote for
all 419 pairs. ``evaluation/five_bench_shape.json`` (frozen by ``r42_inspect.py --freeze``)
splits them into ``shape`` (100 pairs) and ``rest`` (319 pairs).

PER-CLIP SOURCES (never ``_avg.csv``; never ``r36_rho_sweep.csv``, a full-bench aggregate)
  R35 arm c:  edit{T}_FiVE_r35_{c}_frame_stride8.csv          lpips_unedit_part, CLIP-T
              edit{T}_FiVE_r35_fiveacc_{c}_frame_stride8.csv  FiVE-Acc (3 variants)
              r35_clip_directional.csv  method r35_{c}         CLIP-D
  R36 rho p:  same stems with r36_rho{p}_vp / r36_fiveacc_rho{p}_vp, r36_clip_directional.csv
Loaders are imported from r35_score.py unchanged (it is analyzed; its figures must not move).
CHECK before splitting: the full-bench R36 means rebuilt here must match R36's own
r36_rho_sweep.csv (max |diff| printed, hard fail above 1e-4) -- a wrong join would pass
every later step silently.

AGGREGATION: plain per-clip mean within a split (each clip weight 1).

GAIN of an R35 arm on metric x = x_arm - x_R36(lpips_arm), where x_R36 is the R36 curve
(8 rho points, sorted by bg LPIPS) linearly interpolated at the arm's bg LPIPS. Outside the
curve's LPIPS range the gain is NaN -- no extrapolation.

BOOTSTRAP: B draws (default 10,000, seed 0) of the split's clips with replacement; the SAME
draw is used for R35 and R36 (paired), and both the arm point and the whole R36 curve are
recomputed on it. 95% percentile CI per (arm, metric, split); for ``diff = shape - rest``
the two splits are resampled independently (disjoint clips). ``frac_valid`` is the share
of draws whose gain was defined. 20 (arm, metric) intervals, uncorrected: descriptive.

Outputs
  evaluation/figures/r42_{key}_vs_lpips_{shape,rest}.pdf   (5 keys x 2 splits)
  evaluation/csv/r42_arms.csv        split, source, label, n_clips, 5 x metrics, bg LPIPS
  evaluation/csv/r42_bootstrap.csv   arm, metric, split, gain, ci_lo, ci_hi, frac_valid

Usage
-----
    python evaluation/r42_score.py --subset evaluation/five_bench_shape.json --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))

from r35_score import (  # noqa: E402
    ARM_COLOR, ARM_LABEL, COMBOS, FIGURES, FIVEACC_X, GRID, INK, MUTED, N_CLIPS, REF_COLOR,
    RHOS, SURFACE, X_METRICS, Y_METRIC, clip_index, load_clipd, load_harness, r36_points,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Clip = Tuple[int, str]
PerClip = Dict[str, Dict[Clip, float]]          # metric -> {(edit_type, video): value}
SPLITS: Tuple[str, ...] = ("shape", "rest")
SWEEP_TOL: float = 1e-4


# --------------------------------------------------------------------------------------
# per-clip tables
# --------------------------------------------------------------------------------------
def method_table(csv_dir: Path, harness_stem: str, fiveacc_stem: str, clipd_csv: Path,
                 clipd_method: str, clips: Dict[Tuple[int, str], str]) -> PerClip:
    return {
        Y_METRIC: load_harness(csv_dir, harness_stem, Y_METRIC, clips),
        "clip_similarity_target_image":
            load_harness(csv_dir, harness_stem, "clip_similarity_target_image", clips),
        **{m: load_harness(csv_dir, fiveacc_stem, m, clips) for m in FIVEACC_X},
        "clip_d_prompt": load_clipd(clipd_csv, clipd_method, clips),
    }


def load_tables(csv_dir: Path, clips) -> Tuple[Dict[str, PerClip], Dict[int, PerClip]]:
    r35 = {c: method_table(csv_dir, f"r35_{c}", f"r35_fiveacc_{c}",
                           csv_dir / "r35_clip_directional.csv", f"r35_{c}", clips)
           for c in COMBOS}
    r36 = {p: method_table(csv_dir, f"r36_rho{p}_vp", f"r36_fiveacc_rho{p}_vp",
                           csv_dir / "r36_clip_directional.csv", f"r36_rho{p}_vp", clips)
           for p in RHOS}
    return r35, r36


def check_against_sweep(r36: Dict[int, PerClip], sweep: Path) -> None:
    """Full-bench means rebuilt from per-clip CSVs must reproduce R36's own summary."""
    ref = {int(p["label"].split("=")[1]): p for p in r36_points(sweep)}
    worst = 0.0
    for p in RHOS:
        for m in (Y_METRIC, *X_METRICS):
            d = abs(float(np.mean(list(r36[p][m].values()))) - ref[p][m])
            worst = max(worst, d)
    print(f"[r42_score] R36 per-clip rebuild vs {sweep.name}: max |diff| = {worst:.2e}")
    if worst > SWEEP_TOL:
        raise SystemExit(f"[r42_score] R36 rebuild disagrees with {sweep.name} by {worst:.3g} "
                         f"> {SWEEP_TOL}: the per-clip join is wrong, refusing to continue")


def load_subset(path: Path, clips) -> List[Clip]:
    d = json.loads(path.read_text())
    sub = [(int(c["edit_type"]), c["video_name"]) for c in d["cases"]]
    universe = {(t, v) for (t, _), v in clips.items()}
    missing = [c for c in sub if c not in universe]
    if missing or len(set(sub)) != len(sub) or len(sub) != d["k"]:
        raise SystemExit(f"[r42_score] {path}: {len(sub)} cases (k={d['k']}), "
                         f"{len(set(sub))} unique, {len(missing)} not in the benchmark")
    print(f"[r42_score] subset {d['name']}: {len(sub)} pairs, chosen score '{d['chosen_score']}', "
          f"frozen {d['frozen_on']}")
    return sub


def as_array(tab: PerClip, order: Sequence[Clip]) -> Dict[str, np.ndarray]:
    return {m: np.array([tab[m][c] for c in order]) for m in (Y_METRIC, *X_METRICS)}


# --------------------------------------------------------------------------------------
# gain + bootstrap
# --------------------------------------------------------------------------------------
def interp_curve(lp_curve: np.ndarray, x_curve: np.ndarray, lp: np.ndarray) -> np.ndarray:
    """x of the R36 curve at bg LPIPS `lp`, curve sorted by LPIPS; NaN outside its range.

    Shapes: lp_curve/x_curve [..., R], lp [...]. Loops over the leading axis (B draws).
    """
    lp_curve, x_curve, lp = np.atleast_2d(lp_curve), np.atleast_2d(x_curve), np.atleast_1d(lp)
    out = np.full(lp.shape, np.nan)
    for b in range(lp.shape[0]):
        o = np.argsort(lp_curve[b])
        xs, ys = lp_curve[b][o], x_curve[b][o]
        if xs[0] <= lp[b] <= xs[-1]:
            out[b] = np.interp(lp[b], xs, ys)
    return out


def gains(arm: Dict[str, np.ndarray], curve: Dict[int, Dict[str, np.ndarray]], metric: str,
          idx: Optional[np.ndarray]) -> np.ndarray:
    """Gain on the full split (idx None -> shape [1]) or per bootstrap draw (idx [B, n])."""
    take = (lambda v: v.mean(keepdims=True)) if idx is None else (lambda v: v[idx].mean(axis=1))
    lp_arm, x_arm = take(arm[Y_METRIC]), take(arm[metric])
    lp_c = np.stack([take(curve[p][Y_METRIC]) for p in RHOS], axis=-1)     # [B, R]
    x_c = np.stack([take(curve[p][metric]) for p in RHOS], axis=-1)
    return x_arm - interp_curve(lp_c, x_c, lp_arm)


def ci(v: np.ndarray) -> Tuple[float, float, float]:
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan"), float("nan"), 0.0
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), float(ok.mean())


def bootstrap(r35: Dict[str, PerClip], r36: Dict[int, PerClip], split_clips: Dict[str, List[Clip]],
              n_boot: int, seed: int) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    idx = {s: rng.integers(0, len(split_clips[s]), size=(n_boot, len(split_clips[s])))
           for s in SPLITS}
    arr35 = {s: {c: as_array(r35[c], split_clips[s]) for c in COMBOS} for s in SPLITS}
    arr36 = {s: {p: as_array(r36[p], split_clips[s]) for p in RHOS} for s in SPLITS}
    rows = []
    for c in COMBOS:
        for m in X_METRICS:
            point, draws = {}, {}
            for s in SPLITS:
                point[s] = float(gains(arr35[s][c], arr36[s], m, None)[0])
                draws[s] = gains(arr35[s][c], arr36[s], m, idx[s])
            point["diff"] = point["shape"] - point["rest"]
            draws["diff"] = draws["shape"] - draws["rest"]
            for s in (*SPLITS, "diff"):
                lo, hi, fv = ci(draws[s])
                rows.append({"arm": c, "metric": m, "split": s, "gain": point[s],
                             "ci_lo": lo, "ci_hi": hi, "frac_valid": fv})
    return rows


# --------------------------------------------------------------------------------------
# points + figures
# --------------------------------------------------------------------------------------
def split_points(r35, r36, clips_s: List[Clip], split: str) -> List[Dict[str, object]]:
    pts = []
    for p in RHOS:
        a = as_array(r36[p], clips_s)
        pts.append({"split": split, "source": "R36", "label": f"rho={p}", "n_clips": len(clips_s),
                    **{m: float(v.mean()) for m, v in a.items()}})
    for c in COMBOS:
        a = as_array(r35[c], clips_s)
        pts.append({"split": split, "source": "R35", "label": c, "n_clips": len(clips_s),
                    **{m: float(v.mean()) for m, v in a.items()}})
    return pts


def plot(key: str, pts: List[Dict[str, object]], split: str, n: int, fig_dir: Path) -> Path:
    """r35_score.plot's encoding (gray R36 curve, hue = metric, marker = regime), split title."""
    x_metric, x_label, _ = FIGURES[key]
    r36 = [p for p in pts if p["source"] == "R36"]
    r35 = [p for p in pts if p["source"] == "R35"]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([p[x_metric] for p in r36], [p[Y_METRIC] for p in r36], color=REF_COLOR, lw=2,
            marker="o", ms=5, zorder=2, label="R36 · VP, ρ sweep")
    for p, rho in zip(r36, RHOS):
        ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                    textcoords="offset points", fontsize=7, color=MUTED, ha="right", va="bottom")
    for p in r35:
        arm, regime = p["label"].split("_", 1)
        col = ARM_COLOR[arm]
        filled = regime == "unblended"
        ax.scatter([p[x_metric]], [p[Y_METRIC]], s=90, zorder=3, marker="o" if filled else "^",
                   facecolors=col if filled else SURFACE, edgecolors=col if not filled else SURFACE,
                   linewidths=2, label=f"R35 · {ARM_LABEL[p['label']]}")
        ax.annotate(ARM_LABEL[p["label"]], (p[x_metric], p[Y_METRIC]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8, color=INK)
    name = "FiVE-Bench-Shape" if split == "shape" else "FiVE-Bench minus Shape"
    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    ax.set_title(f"R35 divergence routing vs R36 VP ρ sweep — {name} ({n} pairs, per-clip mean)",
                 fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"r42_{key}_vs_lpips_{split}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in cols})


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=Path, default=_REPO_ROOT / "evaluation/five_bench_shape.json")
    ap.add_argument("--all", action="store_true", help="all five figure keys")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+")
    ap.add_argument("--csv_dir", type=Path, default=_REPO_ROOT / "evaluation/csv")
    ap.add_argument("--r36_sweep", type=Path, default=_REPO_ROOT / "evaluation/csv/r36_rho_sweep.csv")
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--fig_dir", type=Path, default=_REPO_ROOT / "evaluation/figures")
    ap.add_argument("--out_csv", type=Path, default=_REPO_ROOT / "evaluation/csv/r42_arms.csv")
    ap.add_argument("--boot_csv", type=Path, default=_REPO_ROOT / "evaluation/csv/r42_bootstrap.csv")
    ap.add_argument("-B", "--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.all and not args.x:
        ap.error(f"pass --all or --x {{{','.join(FIGURES)}}}")
    keys = list(FIGURES) if args.all else args.x
    print(f"[r42_score] args={vars(args)}")

    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r42_score] benchmark has {len(clips)} clips, expected {N_CLIPS}")
    r35, r36 = load_tables(args.csv_dir, clips)
    check_against_sweep(r36, args.r36_sweep)

    shape = load_subset(args.subset, clips)
    shape_set = set(shape)
    rest = sorted(c for c in {(t, v) for (t, _), v in clips.items()} if c not in shape_set)
    split_clips = {"shape": shape, "rest": rest}
    print(f"[r42_score] splits: shape={len(shape)} rest={len(rest)}")

    pts = []
    for s in SPLITS:
        sp = split_points(r35, r36, split_clips[s], s)
        pts += sp
        for k in keys:
            print(f"[r42_score] wrote {plot(k, sp, s, len(split_clips[s]), args.fig_dir)}")
    write_csv(args.out_csv, pts, ["split", "source", "label", "n_clips", *X_METRICS, Y_METRIC])
    print(f"[r42_score] {len(pts)} points -> {args.out_csv}")

    rows = bootstrap(r35, r36, split_clips, args.n_boot, args.seed)
    write_csv(args.boot_csv, rows, ["arm", "metric", "split", "gain", "ci_lo", "ci_hi", "frac_valid"])
    print(f"[r42_score] {len(rows)} bootstrap rows (B={args.n_boot}, seed={args.seed}) -> {args.boot_csv}")
    for r in rows:
        if r["split"] == "diff":
            flag = "" if not (r["ci_lo"] <= 0 <= r["ci_hi"]) else "  (CI spans 0)"
            print(f"  {r['arm']:<16} {r['metric']:<30} shape-rest gain {r['gain']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] valid={r['frac_valid']:.2f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
