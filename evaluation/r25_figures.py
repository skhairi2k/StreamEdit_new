#!/usr/bin/env python
"""R25 step 11 -- the two diagnostic pages for the adaptive-tau arm.

(a) r25_tau_iou.pdf -- the REALIZED mapping: tau against measured IoU, one marker per
    edit type, with the Eq. 4 rho = 2 baseline drawn across. Answers whether colour and
    material edits actually landed at the tau_min end and object swaps at tau_max, or
    whether the edit types interleave.

(b) r25_delta.pdf -- per-case delta (adaptive minus Eq. 4) for the two verdict metrics,
    plotted against |tau - 2|. THIS IS THE PAGE THAT DECIDES THE TASK. With Eq. 4 as the
    only comparator, "adaptive routing helps" and "some constant tau != 2 helps" predict
    the same overall means; they differ only in WHERE the gain sits. If the IoU signal is
    doing real work, the gain must concentrate on the cases whose tau moved furthest from
    the baseline exponent. A flat cloud means any win is a constant-tau effect or
    sampling noise, whichever the mean happens to show.

    Because "flat" is a judgement, each panel is annotated with Pearson r and its
    two-sided p-value over the per-case points. That is a small addition beyond the plan's
    wording, made because the page's stated purpose is to distinguish concentration from
    flatness, and eyeballing 22 points is exactly how that call goes wrong. n = 22 is
    small: read r as a descriptive summary of the cloud, not as an inference.

Reads evaluation/csv/r25_adaptive.csv (from r25_summarize.py) and ignores its trailing
MEAN_* aggregate rows.

Usage
-----
    python evaluation/r25_figures.py \\
        --summary_csv evaluation/csv/r25_adaptive.csv \\
        --clip_col clip_similarity_target_image \\
        --lpips_col lpips_unedit_part \\
        --tau_baseline 2.0 \\
        --out_dir evaluation/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One marker + label per FiVE edit type, so the mapping page reads without a legend key.
TYPE_STYLE: Dict[int, tuple] = {
    1: ("o", "#d62728", "type 1 · swap"),
    2: ("o", "#1f77b4", "type 2 · swap"),
    3: ("s", "#2ca02c", "type 3 · colour"),
    4: ("D", "#9467bd", "type 4 · material"),
    5: ("^", "#ff7f0e", "type 5 · addition"),
    6: ("v", "#8c564b", "type 6 · removal"),
}


# Mask tint colours, matching r25_iou.py's audit panels so the two artefacts read
# the same way: source-side object red, target-side object blue.
COLOR_SRC = (1.0, 0.25, 0.25)
COLOR_TRG = (0.25, 0.55, 1.0)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> tuple:
    """Pearson r and a two-sided p-value, without pulling in scipy."""
    pts = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    n = len(pts)
    if n < 3:
        return float("nan"), float("nan"), n
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan"), n
    r = sxy / math.sqrt(sxx * syy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0, n
    # t = r*sqrt(n-2)/sqrt(1-r^2); two-sided p via the incomplete beta / Student t CDF.
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    x = df / (df + t * t)
    p = _betainc_half(df / 2.0, 0.5, x)
    return r, min(1.0, max(0.0, p)), n


def _betainc_half(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def read_cases(path: Path) -> List[Dict[str, object]]:
    """Per-case rows only; the MEAN_* aggregates r25_summarize appends are dropped."""
    rows = [r for r in csv.DictReader(path.open())
            if not str(r["video_name"]).startswith("MEAN_")]
    if not rows:
        raise SystemExit(f"{path} holds no per-case rows")
    return rows


def fnum(r: Dict[str, object], key: str) -> float:
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def page_tau_iou(rows, tau_baseline: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for t, (marker, colour, label) in TYPE_STYLE.items():
        sub = [r for r in rows if int(r["edit_type"]) == t]
        if not sub:
            continue
        ax.scatter([fnum(r, "iou") for r in sub], [fnum(r, "tau") for r in sub],
                   marker=marker, c=colour, s=54, edgecolors="black", linewidths=0.5,
                   label=label, zorder=3)
    ax.axhline(tau_baseline, color="black", lw=1.2, ls="--", zorder=2,
               label=f"Eq. 4 baseline  $\\rho$ = {tau_baseline:g}")
    ax.set_xlabel("measured IoU  ($M_{src}$ vs $M_{edit}$)")
    ax.set_ylabel(r"assigned release exponent  $\tau$")
    ax.set_title("R25 realized IoU $\\rightarrow$ $\\tau$ mapping")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def page_delta(rows, clip_col: str, lpips_col: str, out: Path) -> None:
    panels = ((clip_col, "higher is better"), (lpips_col, "LOWER is better"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, (col, sense) in zip(axes, panels):
        key = f"delta|{col}"
        if key not in rows[0]:
            raise SystemExit(f"{key} not in the summary CSV -- was {col} scored? "
                             f"(the arm ran a reduced --metrics list)")
        xs = [fnum(r, "abs_tau_minus_2") for r in rows]
        ys = [fnum(r, key) for r in rows]
        for t, (marker, colour, _) in TYPE_STYLE.items():
            idx = [i for i, r in enumerate(rows) if int(r["edit_type"]) == t]
            if idx:
                ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], marker=marker,
                           c=colour, s=52, edgecolors="black", linewidths=0.5, zorder=3)
        ax.axhline(0.0, color="black", lw=1.1, zorder=2)
        r, p, n = pearson(xs, ys)
        ax.set_title(f"$\\Delta$ {col}\n({sense})", fontsize=10)
        ax.set_xlabel(r"$|\tau - 2|$   (distance from the Eq. 4 exponent)")
        ax.set_ylabel("adaptive $-$ Eq. 4")
        ax.grid(alpha=0.25, zorder=0)
        ax.annotate(f"Pearson r = {r:+.3f}\np = {p:.3f}   n = {n}",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=9,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))
    fig.suptitle("R25 per-case gain vs how far $\\tau$ moved from the baseline "
                 "— a flat cloud means the IoU routing contributed nothing",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)



# --------------------------------------------------------------------------------------
# (c) per-video qualitative grids
# --------------------------------------------------------------------------------------
def _load_masks(mask_dir: Path, case_id: str):
    """Unpack the bit-packed M_src / M_edit that r25_iou.py --dump_masks wrote."""
    import numpy as np
    f = mask_dir / f"{case_id}.npz"
    if not f.exists():
        return None, None
    z = np.load(f, allow_pickle=True)
    h, w = (int(x) for x in z["shape"])
    n = h * w
    return (np.unpackbits(z["m_src"])[:n].reshape(h, w).astype(bool),
            np.unpackbits(z["m_edit"])[:n].reshape(h, w).astype(bool))


def _mask_overlay(img, m_src, m_edit):
    """M_src red, M_edit blue, their INTERSECTION -- what the IoU numerator is -- white.

    Drawn on the source first frame. Both masks are first-frame quantities, so this panel
    is the same in every row; it is the reference the IoU number was computed from, not a
    per-frame segmentation.
    """
    import numpy as np
    base = np.asarray(img, dtype=np.float32) / 255.0
    if m_src is None or m_edit is None:
        return np.clip(base, 0, 1)
    inter, a = 0.55, 0.45
    only_s, only_e, both = m_src & ~m_edit, m_edit & ~m_src, m_src & m_edit
    for m, c in ((only_s, COLOR_SRC), (only_e, COLOR_TRG), (both, (1.0, 1.0, 1.0))):
        if m.any():
            base[m] = (1 - a) * base[m] + a * np.array(c, dtype=np.float32)
    return np.clip(base, 0, 1)


def page_grids(rows, mask_dir: Path, src_root: Path, base_root: Path, arm_root: Path,
               out_dir: Path, stride: int, cases_json=None) -> int:
    """One F x 4 page per video: source | Eq.4 baseline | IoU masks | adaptive.

    Rows are every `stride`-th RENDERED frame. The source column is aligned by NORMALIZED
    TIME, not by index: the VAE changes the frame count (80 source frames -> 69 rendered
    on 0001_bus), so indexing both by the same integer would drift the source ahead of
    the renders and make the comparison look worse than it is.
    """
    import numpy as np
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    # The summary CSV keys on (video_name, edit_type); the mask npz files and the output
    # names key on case_id. Those differ for the two duplicated videos -- 0011_lucia is
    # both edit2 and edit5, 0028_kite-walk both edit2 and edit4 -- so resolving through
    # cases.json is what stops a page loading the other case's mask and overwriting it.
    case_id = {}
    if cases_json is not None and Path(cases_json).exists():
        for c in json.loads(Path(cases_json).read_text()):
            case_id[(c["video_name"], int(c["edit_type"]))] = c["case_id"]
    written = 0
    for r in rows:
        v, t = r["video_name"], int(r["edit_type"])
        cid = case_id.get((v, t), v)
        arm_dir = arm_root / f"edit{t}" / v
        base_dir = base_root / f"edit{t}" / v
        src_dir = src_root / "images" / v
        arm_f = sorted(arm_dir.glob("*.png"))
        base_f = sorted(base_dir.glob("*.png"))
        src_f = sorted(src_dir.glob("*.jpg"))
        if not arm_f or not base_f or not src_f:
            print(f"  [skip] {cid}: missing frames "
                  f"(src {len(src_f)} base {len(base_f)} arm {len(arm_f)})", flush=True)
            continue

        idx = list(range(0, len(arm_f), stride))
        m_src, m_edit = _load_masks(mask_dir, str(cid))
        iou_v, tau_v = fnum(r, "iou"), fnum(r, "tau")

        fig, axes = plt.subplots(len(idx), 4,
                                 figsize=(4 * 2.6, len(idx) * 1.75), squeeze=False)
        titles = ("source $I_0$",
                  "baseline  Eq. 4  $\\rho$=2",
                  "IoU masks (frame 0)",
                  f"adaptive  $\\tau$={tau_v:.2f}")
        for row_i, k in enumerate(idx):
            # normalized-time alignment for the two galleries that differ in length
            frac = k / max(len(arm_f) - 1, 1)
            s_k = min(int(round(frac * (len(src_f) - 1))), len(src_f) - 1)
            b_k = min(int(round(frac * (len(base_f) - 1))), len(base_f) - 1)
            src_img = Image.open(src_f[s_k]).convert("RGB")
            hw = (src_img.size[1], src_img.size[0])
            mask_panel = None
            if row_i == 0:
                f0 = Image.open(src_f[0]).convert("RGB")
                mask_panel = _mask_overlay(f0, m_src, m_edit)
            panels = [
                np.asarray(src_img),
                np.asarray(Image.open(base_f[b_k]).convert("RGB").resize(
                    (hw[1], hw[0]), Image.BILINEAR)),
                mask_panel,
                np.asarray(Image.open(arm_f[k]).convert("RGB").resize(
                    (hw[1], hw[0]), Image.BILINEAR)),
            ]
            for col in range(4):
                ax = axes[row_i][col]
                if panels[col] is None:
                    ax.set_facecolor("#f2f2f2")
                else:
                    ax.imshow(panels[col])
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if row_i == 0:
                    ax.set_title(titles[col], fontsize=9)
            axes[row_i][0].set_ylabel(f"f{k}", fontsize=8, rotation=0,
                                      labelpad=14, va="center")
        fig.suptitle(f"{cid}   ·   type {t}   ·   IoU = {iou_v:.4f}   ·   "
                     f"$\\tau$ = {tau_v:.2f}   (Eq. 4 baseline $\\rho$ = 2)",
                     fontsize=11)
        fig.text(0.5, 0.005,
                 "masks: $M_{src}$ red · $M_{edit}$ blue · intersection white — "
                 "frame-0 quantities, shown once; source aligned by normalized time",
                 ha="center", fontsize=7.5, color="0.35")
        fig.tight_layout(rect=(0, 0.02, 1, 0.965))
        out = out_dir / f"{cid}.pdf"
        fig.savefig(out)
        plt.close(fig)
        written += 1
    print(f"wrote {written} per-video grids to {out_dir}", flush=True)
    return written

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_csv", type=Path,
                   default=Path("evaluation/csv/r25_adaptive.csv"))
    p.add_argument("--clip_col", type=str, default="clip_similarity_target_image")
    p.add_argument("--lpips_col", type=str, default="lpips_unedit_part")
    p.add_argument("--tau_baseline", type=float, default=2.0)
    p.add_argument("--out_dir", type=Path, default=Path("evaluation/figures"))
    p.add_argument("--grid_dir", type=Path,
                   default=Path("evaluation/figures/r25_grids"),
                   help="Per-video F x 4 qualitative grids go here.")
    p.add_argument("--mask_dir", type=Path,
                   default=Path("evaluation/figures/r25_iou_masks"),
                   help="npz masks from r25_iou.py --dump_masks.")
    p.add_argument("--src_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--baseline_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r7_visual_prompting"),
                   help="The Eq. 4 vp reference render (R7 = r21_ref_vp).")
    p.add_argument("--arm_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r25_adaptive_tau/r25_adaptive_vp"))
    p.add_argument("--frame_stride", type=int, default=10,
                   help="Take every Nth RENDERED frame as a grid row.")
    p.add_argument("--cases_json", type=Path,
                   default=Path("evaluation/cases.json"),
                   help="Maps (video_name, edit_type) -> case_id.")
    p.add_argument("--no_grids", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = read_cases(args.summary_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== R25 figures: {len(rows)} cases from {args.summary_csv} ===", flush=True)
    page_tau_iou(rows, args.tau_baseline, args.out_dir / "r25_tau_iou.pdf")
    page_delta(rows, args.clip_col, args.lpips_col, args.out_dir / "r25_delta.pdf")
    if not args.no_grids:
        page_grids(rows, args.mask_dir.expanduser(), args.src_root.expanduser(),
                   args.baseline_root.expanduser(), args.arm_root.expanduser(),
                   args.grid_dir.expanduser(), args.frame_stride, args.cases_json)

    # Print the same correlations to stdout so `verdict` can be recorded without
    # opening the PDFs.
    for col in (args.clip_col, args.lpips_col):
        r, p, n = pearson([fnum(x, "abs_tau_minus_2") for x in rows],
                          [fnum(x, f"delta|{col}") for x in rows])
        print(f"  delta|{col:38s} vs |tau-2|:  r={r:+.3f}  p={p:.3f}  n={n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
