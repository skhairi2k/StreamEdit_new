#!/usr/bin/env python
"""R42 inspect -- the material for choosing between the two FiVE-Bench-Shape candidates.

Reads ``evaluation/csv/r42_shape_scores.csv`` (+ the per-pair npz from r42_shape_scores.py)
and writes, WITHOUT touching any R35/R36 result:

  evaluation/csv/r42_subset_composition.csv
      one row per group -- top-100 of `patch`, top-100 of `selfsim`, their intersection,
      and the two "only" sets: size, count per edit type, and the case_ids.
  evaluation/csv/r42_type_prior.csv
      per score: median by edit type, and the PAIRED check on the 100 videos present in
      both type 1 and type 2 -- fraction with score(type 2) > score(type 1).
  evaluation/figures/r42_rank_patch.pdf, r42_rank_selfsim.pdf
      all 419 pairs in descending score, 8 rows per page, a red rule after rank K.
      Columns: source f0 + M_src (red) | anchor + M_trg (cyan) | score map over the anchor
      with the contour of M.
  evaluation/figures/r42_disagree.pdf
      pairs in exactly one top-K (only-patch first, then only-selfsim), with BOTH maps.

``--freeze {patch,selfsim}`` writes ``evaluation/five_bench_shape.json`` from that score's
top-K. It refuses to overwrite an existing file without ``--force``: the subset is meant to
be fixed once, before any R35/R36 number on it is looked at.

PDF pages are rendered by matplotlib to RGB and assembled by PIL as JPEG pages; embedding
~1300 full images as vector-PDF rasters would make each file > 100 MB.

Usage
-----
    python evaluation/r42_inspect.py
    python evaluation/r42_inspect.py --freeze selfsim
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]

SCORES = ("patch", "selfsim")
EDIT_TYPES = (1, 2, 3, 4, 5, 6)
HW = (480, 832)
DISP_HW = (240, 416)            # displayed resolution (half), keeps pages light
ROWS_PER_PAGE = 8
SRC_RGB = np.array([1.0, 0.2, 0.2])
TRG_RGB = np.array([0.1, 0.9, 1.0])


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_rows(path: Path) -> List[Dict]:
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["edit_type"] = int(r["edit_type"])
        for s in SCORES:
            r[s] = float(r[s])
    if len(rows) != 419:
        raise SystemExit(f"[r42_inspect] {path}: {len(rows)} rows, expected 419")
    return rows


def ranked(rows: List[Dict], score: str) -> List[Dict]:
    """Descending; NaN (empty M) sorts last and is reported."""
    nan = [r["case_id"] for r in rows if math.isnan(r[score])]
    if nan:
        print(f"[r42_inspect] WARNING {score}: {len(nan)} NaN scores ranked last: {nan}")
    return sorted(rows, key=lambda r: (math.isnan(r[score]), -r[score] if not math.isnan(r[score]) else 0))


def per_type(cases: Sequence[Dict]) -> Dict[int, int]:
    return {t: sum(1 for r in cases if r["edit_type"] == t) for t in EDIT_TYPES}


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def write_composition(path: Path, top: Dict[str, List[Dict]], by_id: Dict[str, Dict]) -> None:
    a = {r["case_id"] for r in top["patch"]}
    b = {r["case_id"] for r in top["selfsim"]}
    groups = {
        "top_patch": [r["case_id"] for r in top["patch"]],
        "top_selfsim": [r["case_id"] for r in top["selfsim"]],
        "both": sorted(a & b),
        "only_patch": [c for c in groups_order(top["patch"]) if c not in b],
        "only_selfsim": [c for c in groups_order(top["selfsim"]) if c not in a],
    }
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n"] + [f"n_edit{t}" for t in EDIT_TYPES] + ["case_ids"])
        for g, ids in groups.items():
            counts = per_type([by_id[c] for c in ids])
            w.writerow([g, len(ids)] + [counts[t] for t in EDIT_TYPES] + [";".join(ids)])
            print(f"  {g:<13} n={len(ids):<3} " + " ".join(f"e{t}={counts[t]}" for t in EDIT_TYPES))


def groups_order(cases: Sequence[Dict]) -> List[str]:
    return [r["case_id"] for r in cases]


def write_type_prior(path: Path, rows: List[Dict]) -> None:
    by_tv = {(r["edit_type"], r["video_name"]): r for r in rows}
    shared = sorted({v for (t, v) in by_tv if t == 1} & {v for (t, v) in by_tv if t == 2})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score"] + [f"median_edit{t}" for t in EDIT_TYPES]
                   + ["n_paired_videos", "frac_type2_gt_type1"])
        for s in SCORES:
            meds = [median(r[s] for r in rows if r["edit_type"] == t) for t in EDIT_TYPES]
            frac = float(np.mean([by_tv[(2, v)][s] > by_tv[(1, v)][s] for v in shared]))
            w.writerow([s] + [f"{m:.6g}" for m in meds] + [len(shared), f"{frac:.3f}"])
            print(f"  {s:<8} medians " + " ".join(f"e{t}={m:.4g}" for t, m in zip(EDIT_TYPES, meds))
                  + f" | type2>type1 on {frac:.0%} of {len(shared)} videos")


# --------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------
class Pages:
    """Loads images / npz lazily for one pair at a time, at display resolution."""

    def __init__(self, data_root: Path, anchor_root: Path, npz_root: Path, ss_vmax: float):
        self.data_root, self.anchor_root, self.npz_root = data_root, anchor_root, npz_root
        self.ss_vmax = ss_vmax

    def load(self, r: Dict):
        T, v = r["edit_type"], r["video_name"]
        size = (DISP_HW[1], DISP_HW[0])
        src = Image.open(self.data_root / "images" / v / "00001.jpg").convert("RGB").resize(size, Image.BILINEAR)
        anc = Image.open(self.anchor_root / f"edit{T}" / f"{v}.png").convert("RGB").resize(size, Image.BILINEAR)
        z = np.load(self.npz_root / f"edit{T}" / f"{v}.npz")
        ds = lambda m: np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
        return (np.asarray(src) / 255.0, np.asarray(anc) / 255.0, ds(z["M_src"]), ds(z["M_trg"]),
                z["d_patch"], z["selfsim_row_err"])


def tint(img: np.ndarray, mask: np.ndarray, rgb: np.ndarray, a: float = 0.45) -> np.ndarray:
    out = img.copy()
    out[mask] = (1 - a) * out[mask] + a * rgb
    return out


def map_panel(ax, anc: np.ndarray, m: np.ndarray, M: np.ndarray, vmax: float, label: str) -> None:
    gray = anc.mean(axis=2)
    h, w = gray.shape
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, extent=(0, w, h, 0))
    ax.imshow(m, cmap="magma", vmin=0, vmax=vmax, alpha=0.6, extent=(0, w, h, 0),
              interpolation="nearest")
    if M.any():
        ax.contour(M.astype(float), levels=[0.5], colors="white", linewidths=0.8)
    ax.text(4, 14, label, color="white", fontsize=7, bbox=dict(fc="black", alpha=0.5, pad=1, lw=0))


def render_pdf(path: Path, order: List[Dict], pages: Pages, ranks: Dict[str, Dict[str, int]],
               show: Sequence[str], rule_at: Optional[int], rule_label: str, header: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ncol = 2 + len(show)
    imgs: List[Image.Image] = []
    for p0 in range(0, len(order), ROWS_PER_PAGE):
        chunk = order[p0:p0 + ROWS_PER_PAGE]
        fig, axes = plt.subplots(ROWS_PER_PAGE, ncol, figsize=(3.6 * ncol, 2.35 * ROWS_PER_PAGE),
                                 squeeze=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.32, wspace=0.03)
        fig.suptitle(f"{header}  —  page {p0 // ROWS_PER_PAGE + 1}", fontsize=10, x=0.01, y=0.995, ha="left")
        for i in range(ROWS_PER_PAGE):
            for ax in axes[i]:
                ax.axis("off")
            if i >= len(chunk):
                continue
            r = chunk[i]
            src, anc, ms, mt, dp, ss = pages.load(r)
            M = ms | mt
            axes[i, 0].imshow(tint(src, ms, SRC_RGB))
            axes[i, 1].imshow(tint(anc, mt, TRG_RGB))
            maps = [m for m in (("patch", dp, 1.0), ("selfsim", ss, pages.ss_vmax)) if m[0] in show]
            for j, (name, m, vmax) in enumerate(maps):
                map_panel(axes[i, 2 + j], anc, m, M, vmax, f"{name} map")
            title = (f"#{p0 + i + 1}  {r['case_id']}  (type {r['edit_type']})   "
                     f"patch={r['patch']:.3f} [rank {ranks['patch'][r['case_id']]}]   "
                     f"selfsim={r['selfsim']:.4f} [rank {ranks['selfsim'][r['case_id']]}]   "
                     f"grounded '{r['phrase']}' on {r['ground_image']}  area_M={float(r['area_M']):.3f}"
                     + (f"   AUDIT {r['audit']}" if r["audit"] else ""))
            axes[i, 0].set_title(title, loc="left", fontsize=7.5,
                                 color="darkred" if r["audit"] else "black")
        if rule_at is not None and p0 < rule_at < p0 + len(chunk):
            k = rule_at - p0                       # first row below the cut
            y = 0.5 * (axes[k - 1, 0].get_position().y0 + axes[k, 0].get_position().y1)
            fig.add_artist(Line2D([0, 1], [y, y], transform=fig.transFigure, color="red", lw=3))
            fig.text(0.99, y + 0.003, rule_label, color="red", fontsize=9, ha="right")
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], resolution=100.0, quality=85)
    print(f"[r42_inspect] {path}  ({len(imgs)} pages)")


# --------------------------------------------------------------------------------------
# freeze
# --------------------------------------------------------------------------------------
def freeze(path: Path, score: str, top: List[Dict], k: int, src_csv: Path, force: bool) -> int:
    if path.exists() and not force:
        print(f"[r42_inspect] {path} exists; refusing to overwrite (pass --force to replace "
              "the frozen subset deliberately)")
        return 1
    out = {
        "name": "FiVE-Bench-Shape",
        "task": "R42",
        "chosen_score": score,
        "k": k,
        "source_csv": str(src_csv.relative_to(_REPO_ROOT) if src_csv.is_absolute() else src_csv),
        "frozen_on": dt.date.today().isoformat(),
        "cases": [{"case_id": r["case_id"], "edit_type": r["edit_type"], "video_name": r["video_name"],
                   "rank": i + 1, "score": round(r[score], 6)} for i, r in enumerate(top)],
    }
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[r42_inspect] froze {len(top)} cases by '{score}' -> {path}  "
          + " ".join(f"e{t}={c}" for t, c in per_type(top).items()))
    return 0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores", type=Path, default=_REPO_ROOT / "evaluation/csv/r42_shape_scores.csv")
    p.add_argument("--npz_root", type=Path, default=Path("~/Data/dataggen/outputs/five_bench/r42_scores"))
    p.add_argument("--data_root", type=Path, default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--anchor_root", type=Path, default=Path("~/Data/dataggen/outputs/five_bench/anchors"))
    p.add_argument("--csv_dir", type=Path, default=_REPO_ROOT / "evaluation/csv")
    p.add_argument("--fig_dir", type=Path, default=_REPO_ROOT / "evaluation/figures")
    p.add_argument("-k", type=int, default=100, help="subset size (fixed by the plan)")
    p.add_argument("--freeze", choices=SCORES, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--subset_out", type=Path, default=_REPO_ROOT / "evaluation/five_bench_shape.json")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.scores)
    by_id = {r["case_id"]: r for r in rows}
    order = {s: ranked(rows, s) for s in SCORES}
    top = {s: order[s][: args.k] for s in SCORES}

    if args.freeze:
        return freeze(args.subset_out, args.freeze, top[args.freeze], args.k, args.scores, args.force)

    ranks = {s: {r["case_id"]: i + 1 for i, r in enumerate(order[s])} for s in SCORES}
    print("[r42_inspect] composition")
    write_composition(args.csv_dir / "r42_subset_composition.csv", top, by_id)
    print("[r42_inspect] type prior")
    write_type_prior(args.csv_dir / "r42_type_prior.csv", rows)

    # one fixed colour scale for every selfsim map: the 99th percentile of all row errors
    npz_root = args.npz_root.expanduser()
    allv = np.concatenate([np.load(npz_root / f"edit{r['edit_type']}" / f"{r['video_name']}.npz")
                           ["selfsim_row_err"].ravel() for r in rows])
    pages = Pages(args.data_root.expanduser(), args.anchor_root.expanduser(), npz_root,
                  ss_vmax=float(np.percentile(allv, 99)))

    for s in SCORES:
        render_pdf(args.fig_dir / f"r42_rank_{s}.pdf", order[s], pages, ranks, show=(s,),
                   rule_at=args.k, rule_label=f"cutoff K={args.k}", header=f"{s} ranking (all 419, descending)")
    a = {r["case_id"] for r in top["patch"]}
    b = {r["case_id"] for r in top["selfsim"]}
    disagree = [r for r in top["patch"] if r["case_id"] not in b] + \
               [r for r in top["selfsim"] if r["case_id"] not in a]
    n_only_patch = len(a - b)
    render_pdf(args.fig_dir / "r42_disagree.pdf", disagree, pages, ranks, show=SCORES,
               rule_at=n_only_patch if 0 < n_only_patch < len(disagree) else None,
               rule_label="↑ only patch   ·   only selfsim ↓",
               header=f"disagreement — first {n_only_patch} only in patch top-{args.k}, "
                      f"then {len(b - a)} only in selfsim top-{args.k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
