"""R41 score -- binary vs continuous rho mapping, at the draft length N set in r41_config.sh.

WHAT IS PLOTTED
---------------
R35's five planes: preservation on y (``lpips_unedit_part``, lower = better) against one
editability measure on x, one PDF each:

    r41_n{N}_clip_vs_lpips.pdf          x = clip_similarity_target_image  (CLIP-T)
    r41_n{N}_clipd_vs_lpips.pdf         x = clip_d_prompt                 (CLIP-D)
    r41_n{N}_fiveacc_vs_lpips.pdf       x = five_acc_yes_no
    r41_n{N}_fiveacc_mc_vs_lpips.pdf    x = five_acc_multi_choice
    r41_n{N}_fiveacc_full_vs_lpips.pdf  x = five_acc (mean of yes/no, MC, union, inter)

One panel per combo ({DINO, LPIPS} x {unblended, tau = 0.9}), shared axes:
* R36 -- StreamEdit + VP at rho in {2,...,50}: the gray reference curve, as in R35/R38.
* continuous -- the stored budget-linear arm at the SAME N (R35's at N=15, R38's at N=5).
* binary -- one point per cut theta, joined in theta order: the binary family as a curve,
  so no single cut carries the comparison.

AGGREGATION -- plain mean over the 419 clips from the per-clip CSVs, never ``{stem}_avg.csv``
(a mean of six per-type means). The loaders are R35's own (imported from ``r35_score``), so
every side is read by the same code; their errors therefore carry an ``[r35_score]`` prefix.
Any missing input -- continuous or binary -- is a hard failure.

Output: the five PDFs in --fig_dir and ``r41_n{N}_arms.csv`` (one row per plotted point).

Example
-------
    python evaluation/r41_score.py --all --draft_n 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r41_config  # noqa: E402
from r35_score import (  # noqa: E402
    ARM_COLOR, FIVEACC_X, GRID, INK, MUTED, N_CLIPS, REF_COLOR, RHOS, SURFACE, X_METRICS,
    Y_METRIC, clip_index, load_clipd, load_harness, r36_points,
)

FIGURES: Dict[str, Tuple[str, str, str]] = {
    "clip":         ("clip_similarity_target_image", "CLIP-T (target image)  → more edit", "clip"),
    "clipd":        ("clip_d_prompt", "CLIP-D (directional, prompt)  → more edit", "clipd"),
    "fiveacc":      ("five_acc_yes_no", "FiVE-Acc (yes/no only)  → more edit", "fiveacc"),
    "fiveacc_mc":   ("five_acc_multi_choice", "FiVE-Acc (multi-choice only)  → more edit",
                     "fiveacc_mc"),
    "fiveacc_full": ("five_acc", "FiVE-Acc (mean of yes/no, MC, union, inter)  → more edit",
                     "fiveacc_full"),
}
COMBO_LABEL = {"dino_unblended": "DINO · unblended", "dino_tgt09": "DINO · τ = 0.9",
               "lpips_unblended": "LPIPS · unblended", "lpips_tgt09": "LPIPS · τ = 0.9"}


def point(csv_dir: Path, clips: Dict[Tuple[int, str], str], stem: str, fiveacc_stem: str,
          clipd_csv: Path, clipd_method: str) -> Dict[str, float]:
    """Plain mean over the 419 clips of the y metric and the five x metrics."""
    per = {
        Y_METRIC: load_harness(csv_dir, stem, Y_METRIC, clips),
        "clip_similarity_target_image":
            load_harness(csv_dir, stem, "clip_similarity_target_image", clips),
        **{m: load_harness(csv_dir, fiveacc_stem, m, clips) for m in FIVEACC_X},
        "clip_d_prompt": load_clipd(clipd_csv, clipd_method, clips),
    }
    return {m: float(np.mean(list(v.values()))) for m, v in per.items()}


def arm_points(cfg: dict, csv_dir: Path,
               clips: Dict[Tuple[int, str], str]) -> List[Dict[str, object]]:
    n, tag, pre = cfg["draft_n"], cfg["tag"], cfg["cont_prefix"]
    pts: List[Dict[str, object]] = []
    for combo in cfg["combos"]:
        src = cfg["src_name"][combo]
        cont = f"{pre}_{src}"
        pts.append({"source": "continuous", "combo": combo, "theta": "", "N": n,
                    "n_clips": N_CLIPS,
                    **point(csv_dir, clips, cont, f"{pre}_fiveacc_{src}",
                            cfg["cont_clipd_csv"], cont)})
    for a in cfg["arms"]:
        stem = f"r41_{tag}_{a['arm']}"
        th_tag = str(a["theta"]).replace(".", "")
        pts.append({"source": "binary", "combo": a["combo"], "theta": a["theta"], "N": n,
                    "n_clips": N_CLIPS,
                    **point(csv_dir, clips, stem, f"r41_fiveacc_{tag}_{a['arm']}",
                            csv_dir / f"r41_{tag}_clip_directional_t{th_tag}.csv", stem)})
    return pts


def plot(key: str, cfg: dict, r36: List[Dict[str, object]], pts: List[Dict[str, object]],
         fig_dir: Path) -> Path:
    x_metric, x_label, fkey = FIGURES[key]
    combos = cfg["combos"]
    n_cols = 2 if len(combos) > 1 else 1
    n_rows = int(np.ceil(len(combos) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel()[len(combos):]:
        ax.set_visible(False)

    for ax, combo in zip(axes.ravel(), combos):
        ax.set_facecolor(SURFACE)
        ax.plot([p[x_metric] for p in r36], [p[Y_METRIC] for p in r36], color=REF_COLOR,
                lw=2, marker="o", ms=4, zorder=2, label="R36 · VP, ρ sweep")
        for p, rho in zip(r36, RHOS):
            ax.annotate(f"ρ={rho}", (p[x_metric], p[Y_METRIC]), xytext=(-5, 5),
                        textcoords="offset points", fontsize=6.5, color=MUTED,
                        ha="right", va="bottom")

        col = ARM_COLOR[combo.split("_")[0]]
        cont = next(p for p in pts if p["source"] == "continuous" and p["combo"] == combo)
        unblended = combo.endswith("unblended")
        ax.scatter([cont[x_metric]], [cont[Y_METRIC]], s=95, zorder=4,
                   marker="o" if unblended else "^",
                   facecolors=col if unblended else SURFACE,
                   edgecolors=SURFACE if unblended else col, linewidths=2,
                   label=f"continuous (budget-linear), N={cfg['draft_n']}")

        binary = sorted((p for p in pts if p["source"] == "binary" and p["combo"] == combo),
                        key=lambda p: p["theta"])
        ax.plot([p[x_metric] for p in binary], [p[Y_METRIC] for p in binary], color=col,
                lw=1.2, ls="--", marker="s", ms=5, mfc=SURFACE, mec=col, mew=1.5, zorder=3,
                label="binary, θ curve (ρ=2 if m<θ else 50)")
        for p in binary:
            ax.annotate(f"θ={p['theta']:g}", (p[x_metric], p[Y_METRIC]), xytext=(6, -3),
                        textcoords="offset points", fontsize=7, color=INK, va="top")

        ax.set_title(COMBO_LABEL[combo], fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        ax.legend(fontsize=6.5, frameon=False, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel(x_label, color=INK, fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("background LPIPS (unedited region)  ↓ better preservation",
                      color=INK, fontsize=8.5)
    fig.suptitle(f"R41 binary vs continuous ρ mapping — draft N={cfg['draft_n']} — full "
                 f"FiVE-Bench ({N_CLIPS} pairs, per-clip mean)", fontsize=10, color=INK)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"r41_{cfg['tag']}_{fkey}_vs_lpips.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def write_table(path: Path, r36: List[Dict[str, object]], pts: List[Dict[str, object]],
                n: int) -> None:
    cols = ["source", "combo", "theta", "N", "n_clips", *X_METRICS, Y_METRIC]
    rows = [{"source": "R36", "combo": p["label"], "theta": "", "N": 15,
             "n_clips": p["n_clips"], **{m: p[m] for m in (*X_METRICS, Y_METRIC)}}
            for p in r36]
    rows += [{k: p[k] for k in cols} for p in pts]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft_n", type=int, required=True, choices=(5, 15))
    ap.add_argument("--all", action="store_true", help="Build all five figures.")
    ap.add_argument("--x", choices=tuple(FIGURES), nargs="+", help="Build only these figures.")
    ap.add_argument("--csv_dir", type=Path, default=r41_config.REPO / "evaluation" / "csv")
    ap.add_argument("--r36_sweep", type=Path,
                    default=r41_config.REPO / "evaluation" / "csv" / "r36_rho_sweep.csv")
    ap.add_argument("--data_root", type=Path,
                    default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--fig_dir", type=Path, default=r41_config.REPO / "evaluation" / "figures")
    ap.add_argument("--out_csv", type=Path, help="default: <csv_dir>/r41_n{N}_arms.csv")
    args = ap.parse_args(argv)
    if not args.all and not args.x:
        ap.error(f"pass --all or --x {{{','.join(FIGURES)}}}")
    keys = list(FIGURES) if args.all else args.x

    cfg = r41_config.load(args.draft_n)
    # continuous clip-D lives with the other per-clip CSVs, i.e. under --csv_dir too
    cfg["cont_clipd_csv"] = args.csv_dir / Path(cfg["cont_clipd_csv"]).name
    clips = clip_index(args.data_root.expanduser())
    if len(clips) != N_CLIPS:
        raise SystemExit(f"[r41_score] benchmark has {len(clips)} clips, expected {N_CLIPS}")

    r36 = r36_points(args.r36_sweep)                  # fail fast on the cross-task input
    pts = arm_points(cfg, args.csv_dir, clips)

    out_csv = args.out_csv or args.csv_dir / f"r41_{cfg['tag']}_arms.csv"
    write_table(out_csv, r36, pts, cfg["draft_n"])
    print(f"[r41_score] {len(r36)} R36 + {len(pts)} R41 points -> {out_csv}")
    for p in pts:
        th = f"θ={p['theta']:g}" if p["theta"] != "" else "cont."
        print(f"  {p['source']:<10} {p['combo']:<16} {th:<6} lpips={p[Y_METRIC]:.4f} "
              f"clipT={p['clip_similarity_target_image']:.3f} "
              f"clipD={p['clip_d_prompt']:+.4f} fa_yn={p['five_acc_yes_no']:.3f} "
              f"fa_full={p['five_acc']:.3f}")
    for k in keys:
        print(f"[r41_score] wrote {plot(k, cfg, r36, pts, args.fig_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
