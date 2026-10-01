"""R35 score -- five Pareto figures: R36's VP rho-sweep curve vs R35's four arms.

WHAT IS PLOTTED
---------------
Each figure puts preservation on y (``lpips_unedit_part``, lower = better) against one
editability measure on x:

    r35_clip_vs_lpips.pdf     x = clip_similarity_target_image  (CLIP-T)
    r35_clipd_vs_lpips.pdf    x = clip_d_prompt                 (CLIP-D, R33's axis of record)
    r35_fiveacc_vs_lpips.pdf       x = five_acc_yes_no          (FiVE-Acc, yes/no only)
    r35_fiveacc_mc_vs_lpips.pdf    x = five_acc_multi_choice    (FiVE-Acc, multi-choice only)
    r35_fiveacc_full_vs_lpips.pdf  x = five_acc                 (FiVE-Acc proper: per-clip mean
                                                                 of yes/no, MC, union, inter)

* R36 -- StreamEdit + visual prompting at rho in {2,3,4,6,8,10,20,50}: a CURVE, the 8
  points joined in rho order, each labelled with its rho. Neutral gray: it is the
  reference the arms are read against.
* R35 -- the 4 (arm, regime) combos as points. Divergence metric by hue (DINO blue,
  LPIPS orange), stage-1 regime by marker (unblended = filled circle, first2 = hollow
  triangle), so identity is never colour alone. Each point is also direct-labelled.

R7 is NOT evaluated: R36 re-renders and scores rho=2 itself.

AGGREGATION -- the one thing that must match on both sides
----------------------------------------------------------
Every point is a PLAIN MEAN OVER THE 419 CLIPS (each clip weight 1).

* R35 side: computed here from the per-clip CSVs --
      edit{T}_FiVE_r35_{combo}_frame_stride8.csv          (lpips_unedit_part, CLIP-T)
      edit{T}_FiVE_r35_fiveacc_{combo}_frame_stride8.csv  (five_acc_yes_no, _multi_choice, five_acc)
      r35_clip_directional.csv  (method == r35_{combo})   (clip_d_prompt)
  NEVER from ``{stem}_avg.csv``: evaluate.py's top-level average is a mean of the six
  per-edit-type means, so edit5's 9 clips would weigh as much as edit1's 100.
* R36 side: read from R36's own ``r36_rho_sweep.csv``, full-bench rows only
  (``n_pairs == 419``, exactly one per rho). R36's summarize step averages the same way
  (plain per-clip mean over 419 -- R36 plan, Decisions: Averaging), so the two sides are
  comparable. Its 22-clip subset rows are ignored.

HARD FAILURES (a figure that silently drops data looks exactly like a correct one)
---------------------------------------------------------------------------------
* any R35 per-clip CSV missing, or any stem not exactly 100/100/100/100/9/10 clips, or a
  metric's clips not matching the benchmark's clip list;
* r36_rho_sweep.csv missing, or not exactly one full-bench row per rho;
* a needed column absent from r36_rho_sweep.csv. R36's plan does not pin the exact
  FiVE-Acc or rho column names, so they are matched against the header, and on failure
  the available columns are printed rather than guessed at.

CLIP IDENTITY. evaluate.py writes ``file_id`` = the row index into edit{T}_FiVE.json, and
prefixes every metric column with the scored folder's name (``step14|lpips_unedit_part``),
so clips are joined through the annotation file's own order and columns are matched on the
``|metric`` suffix -- the same convention as r30_score.build_join_tables.

Output: the five PDFs in --fig_dir, and ``r35_arms.csv`` (one row per plotted point:
source, label, n_clips, the five x metrics and lpips_unedit_part) -- the table view of the
figures.

Example
-------
    python evaluation/r35_score.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Same order as every other R35 script -- do not reorder one without the others.
COMBOS: Tuple[str, ...] = ("dino_unblended", "dino_first2", "lpips_unblended", "lpips_first2")
RHOS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 20, 50)
EXPECT: Dict[int, int] = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
N_CLIPS: int = sum(EXPECT.values())            # 419
Y_METRIC: str = "lpips_unedit_part"
X_METRICS: Tuple[str, ...] = ("clip_similarity_target_image", "clip_d_prompt", "five_acc_yes_no",
                               "five_acc_multi_choice", "five_acc")
FIVEACC_X: Tuple[str, ...] = ("five_acc_yes_no", "five_acc_multi_choice", "five_acc")

# figure key -> (x metric, x-axis label, file name)
FIGURES: Dict[str, Tuple[str, str, str]] = {
    "clip":    ("clip_similarity_target_image", "CLIP-T (target image)  → more edit",
                "r35_clip_vs_lpips.pdf"),
    "clipd":   ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit",
                "r35_clipd_vs_lpips.pdf"),
    "fiveacc": ("five_acc_yes_no", "FiVE-Acc (yes/no only)  → more edit",
                "r35_fiveacc_vs_lpips.pdf"),
    "fiveacc_mc": ("five_acc_multi_choice", "FiVE-Acc (multi-choice only)  → more edit",
                   "r35_fiveacc_mc_vs_lpips.pdf"),
    "fiveacc_full": ("five_acc", "FiVE-Acc (mean of yes/no, MC, union, inter)  → more edit",
                     "r35_fiveacc_full_vs_lpips.pdf"),
}

# Colour by divergence metric, marker by stage-1 regime. Blue/orange are slots 1-2 of the
# reference categorical palette, documented to validate all-pairs (scatter) in both modes.
ARM_COLOR = {"dino": "#2a78d6", "lpips": "#eb6834"}
REF_COLOR = "#8c8c87"                      # neutral gray: the reference curve
INK, MUTED, GRID = "#1f1f1e", "#6b6b67", "#e6e6e3"
SURFACE = "#fcfcfb"
ARM_LABEL = {"dino_unblended": "DINO · unblended", "dino_first2": "DINO · first2",
             "lpips_unblended": "LPIPS · unblended", "lpips_first2": "LPIPS · first2"}


# ----------------------------------------------------------------------- R35 side -----

def clip_index(data_root: Path) -> Dict[Tuple[int, str], str]:
    """(edit_type, file_id) -> video_name, through each annotation file's own order."""
    out: Dict[Tuple[int, str], str] = {}
    for t in EXPECT:
        items = json.loads((data_root / "edit_prompt" / f"edit{t}_FiVE.json").read_text())
        for i, e in enumerate(items):
            out[(t, str(i))] = e["video_name"]
    return out


def _check_coverage(vals: Dict[Tuple[int, str], float], what: str,
                    clips: Dict[Tuple[int, str], str]) -> None:
    want = {(t, v) for (t, _), v in clips.items()}
    got = set(vals)
    if got != want:
        raise SystemExit(f"[r35_score] {what}: covers {len(got)} clips, expected the "
                         f"{len(want)} benchmark clips (missing {len(want - got)}, "
                         f"unexpected {len(got - want)}).")


def load_harness(csv_dir: Path, stem: str, metric: str,
                 clips: Dict[Tuple[int, str], str]) -> Dict[Tuple[int, str], float]:
    """{(edit_type, video_name): value} for one metric of one evaluate.py stem."""
    out: Dict[Tuple[int, str], float] = {}
    for t, n_exp in EXPECT.items():
        path = csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv"
        if not path.is_file():
            raise SystemExit(f"[r35_score] missing {path} -- has its scoring step run?")
        rows = list(csv.DictReader(path.open()))
        if len(rows) != n_exp:
            raise SystemExit(f"[r35_score] {path.name}: {len(rows)} rows, expected {n_exp}")
        col = next((k for k in rows[0] if k == metric or k.endswith(f"|{metric}")), None)
        if col is None:
            raise SystemExit(f"[r35_score] {path.name}: no '{metric}' column; "
                             f"have {list(rows[0])}")
        for r in rows:
            out[(t, clips[(t, r["file_id"])])] = float(r[col])
    _check_coverage(out, f"{stem}:{metric}", clips)
    return out


def load_clipd(path: Path, method: str,
               clips: Dict[Tuple[int, str], str]) -> Dict[Tuple[int, str], float]:
    if not path.is_file():
        raise SystemExit(f"[r35_score] missing {path} -- has the clipd step run?")
    out = {(int(r["edit_type"]), r["video_name"]): float(r["clip_d_prompt"])
           for r in csv.DictReader(path.open()) if r["method"] == method}
    _check_coverage(out, f"{path.name}:{method}", clips)
    return out


def r35_points(csv_dir: Path, clips: Dict[Tuple[int, str], str]) -> List[Dict[str, object]]:
    pts = []
    for c in COMBOS:
        per = {
            Y_METRIC: load_harness(csv_dir, f"r35_{c}", Y_METRIC, clips),
            "clip_similarity_target_image":
                load_harness(csv_dir, f"r35_{c}", "clip_similarity_target_image", clips),
            **{m: load_harness(csv_dir, f"r35_fiveacc_{c}", m, clips) for m in FIVEACC_X},
            "clip_d_prompt": load_clipd(csv_dir / "r35_clip_directional.csv", f"r35_{c}", clips),
        }
        # plain mean over the 419 clips, each clip weight 1
        pts.append({"source": "R35", "label": c, "n_clips": N_CLIPS,
                    **{m: float(np.mean(list(v.values()))) for m, v in per.items()}})
    return pts


# ----------------------------------------------------------------------- R36 side -----

def _resolve(cols: Sequence[str], name: str, pattern: Optional[str] = None) -> str:
    """Exact name, then `|name` suffix, then an optional regex -- exactly one hit or fail."""
    for cands in ([k for k in cols if k == name],
                  [k for k in cols if k.endswith(f"|{name}")],
                  [k for k in cols if pattern and re.search(pattern, k)]):
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            raise SystemExit(f"[r35_score] r36_rho_sweep.csv: '{name}' is ambiguous "
                             f"({cands}); available columns: {list(cols)}")
    raise SystemExit(f"[r35_score] r36_rho_sweep.csv: no column for '{name}'; "
                     f"available columns: {list(cols)}")


def r36_points(sweep: Path) -> List[Dict[str, object]]:
    if not sweep.is_file():
        raise SystemExit(f"[r35_score] missing {sweep} -- R36's summarize step (a separate "
                         f"task) must have run first.")
    rows = list(csv.DictReader(sweep.open()))
    if not rows:
        raise SystemExit(f"[r35_score] {sweep} is empty")
    cols = list(rows[0])
    c_rho = _resolve(cols, "rho", r"(^|_)rho($|_)")
    c_n = _resolve(cols, "n_pairs")
    c_met = {m: _resolve(cols, m, r"five_acc.*(yes|yn)|(yes_no|yn).*acc"
                         if m == "five_acc_yes_no" else None)
             for m in (Y_METRIC, *X_METRICS)}

    full = [r for r in rows if int(float(r[c_n])) == N_CLIPS]
    by_rho: Dict[int, Dict[str, str]] = {}
    for r in full:
        rho = int(round(float(r[c_rho])))
        if rho in by_rho:
            raise SystemExit(f"[r35_score] {sweep.name}: two full-bench rows for rho={rho}")
        by_rho[rho] = r
    if sorted(by_rho) != sorted(RHOS):
        raise SystemExit(f"[r35_score] {sweep.name}: full-bench (n_pairs=={N_CLIPS}) rows "
                         f"for rho {sorted(by_rho)}, expected {list(RHOS)}")
    return [{"source": "R36", "label": f"rho={rho}", "n_clips": N_CLIPS,
             **{m: float(by_rho[rho][c]) for m, c in c_met.items()}} for rho in RHOS]


# ----------------------------------------------------------------------- figures ------

def plot(key: str, r36: List[Dict[str, object]], r35: List[Dict[str, object]],
         fig_dir: Path) -> Path:
    x_metric, x_label, fname = FIGURES[key]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # reference curve: 8 rho points joined in rho order, each labelled with its rho
    xs = [p[x_metric] for p in r36]
    ys = [p[Y_METRIC] for p in r36]
    ax.plot(xs, ys, color=REF_COLOR, lw=2, marker="o", ms=5, zorder=2,
            label="R36 · VP, ρ sweep")
    for p, rho in zip(r36, RHOS):
        # above-left of the curve: arms that beat it land below-right (more edit, less
        # LPIPS), so labels there would collide with exactly the points of interest
        ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                    textcoords="offset points", fontsize=7, color=MUTED,
                    ha="right", va="bottom")

    # the four arms: hue = divergence metric, marker = stage-1 regime
    for p in r35:
        arm, regime = p["label"].split("_", 1)
        col = ARM_COLOR[arm]
        filled = regime == "unblended"
        ax.scatter([p[x_metric]], [p[Y_METRIC]], s=90, zorder=3,
                   marker="o" if filled else "^",
                   facecolors=col if filled else SURFACE, edgecolors=col if not filled else SURFACE,
                   linewidths=2, label=f"R35 · {ARM_LABEL[p['label']]}")
        ax.annotate(ARM_LABEL[p["label"]], (p[x_metric], p[Y_METRIC]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8, color=INK)

    ax.set_xlabel(x_label, color=INK)
    ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation", color=INK)
    ax.set_title(f"R35 divergence routing vs R36 VP ρ sweep — full FiVE-Bench "
                 f"({N_CLIPS} pairs, per-clip mean)", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="best")

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / fname
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def write_table(path: Path, pts: List[Dict[str, object]]) -> None:
    cols = ["source", "label", "n_clips", *X_METRICS, Y_METRIC]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in pts:
            w.writerow({k: p[k] for k in cols})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Build all five figures.")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+",
                    help="Build only these figures (default with --all: all five).")
    ap.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    ap.add_argument("--r36_sweep", type=Path, default=Path("evaluation/csv/r36_rho_sweep.csv"))
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--fig_dir", type=Path, default=Path("evaluation/figures"))
    ap.add_argument("--out_csv", type=Path, default=Path("evaluation/csv/r35_arms.csv"))
    args = ap.parse_args(argv)
    if not args.all and not args.x:
        ap.error(f"pass --all or --x {{{','.join(FIGURES)}}}")
    keys = list(FIGURES) if args.all else args.x

    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r35_score] benchmark has {len(clips)} clips, expected {N_CLIPS}")

    r36 = r36_points(args.r36_sweep)                  # fail fast on the cross-task input
    r35 = r35_points(args.csv_dir, clips)

    write_table(args.out_csv, r36 + r35)
    print(f"[r35_score] {len(r36)} R36 + {len(r35)} R35 points -> {args.out_csv}")
    for p in r36 + r35:
        print(f"  {p['source']} {p['label']:<16} lpips={p[Y_METRIC]:.4f} "
              f"clipT={p['clip_similarity_target_image']:.3f} "
              f"clipD={p['clip_d_prompt']:+.4f} fa_yn={p['five_acc_yes_no']:.3f} "
              f"fa_mc={p['five_acc_multi_choice']:.3f} fa_full={p['five_acc']:.3f}")
    for k in keys:
        print(f"[r35_score] wrote {plot(k, r36, r35, args.fig_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
