#!/usr/bin/env python
"""R28: join the arm CSVs and compare the three background-preservation families.

    *_unedit_part         1 - M_src                     today's metric
    *_unedit_union        1 - (M_src | M_tgt_arm)       per-arm, faithful per method
    *_unedit_union_fixed  1 - (M_src | U_arms M_tgt)    one region for every arm

The headline question is whether scoring on a union changes the ORDERING of the arms. A
Spearman near 1.0 would mean the fix is a refinement with no scientific consequence --
itself a publishable negative. A low correlation means R26's conclusions, which rest on
`lpips_unedit_part`, have to be re-read.

MICRO, NOT MACRO. Values are recomputed from the per-video CSVs rather than read from
`{stem}_avg.csv`, because evaluate.py's `_avg` is a MACRO-average over the six edit types
WITH per-metric scaling (lpips x1000, ssim x100, ...). R28's case set is unbalanced --
edit2 holds 13 of the 22 pairs while edit1 and edit6 hold one each -- so macro weighting
would give each singleton clip 1/6 of the table instead of 1/22. The scaled macro value is
still emitted alongside, for continuity with numbers already read off those files.

BACKGROUND AREA IS REPORTED PER ARM. Under the per-arm union each arm is scored over a
DIFFERENT region: an arm whose grounding under-fired gets a smaller M_tgt, hence a larger
background and an easier score. Without the area column that failure is indistinguishable
from a genuine preservation win -- which is precisely the shape of result R28 is looking
for, so the control has to travel with it.

TWO COMPARATORS, TWO QUESTIONS. `r7_visual_prompting` and the R26 `(2,2)` cell are Eq.4
under vp (in fact bit-identical renders), so either isolates the tau field against the
vp-anchored R26 arms. `baseline` is novp, so an arm-vs-baseline gap moves the tau field
AND visual prompting together; it answers "what does the full pipeline buy over the paper
method", not "does spatial tau help". They are reported in separate blocks.

Usage
-----
    python evaluation/r28_summarize.py \\
        --arm_glob 'evaluation/csv/r28_*_avg.csv' \\
        --control taubg2_taufg2_vp \\
        --cases evaluation/cases.json \\
        -o evaluation/csv/r28_union_vs_part.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r28_target_masks import list_images, load_src_masks, load_npz  # noqa: E402
from r28_fixed_union import load_fixed_union  # noqa: E402

FAMILIES = ("part", "union", "union_fixed")
BASES = ("psnr", "lpips", "mse", "ssim")
HIGHER_IS_BETTER = {"psnr", "ssim"}

# Non-arm CSVs that share the evaluation/csv/ directory. Swept in by a loose glob they
# would appear as phantom arms -- the R21 bug.
#
# Deliberately does NOT exclude "baseline": in R26 that name was a stored REFERENCE, but
# in R28 `baseline` is a scored arm (the novp Eq.4 comparator), and inheriting R26's
# pattern rejected r28_baseline_avg.csv outright. Every `r28_*_avg.csv` is written by
# r28_eval.sh and is by construction an arm; the r20-r27 references this guards against do
# not carry an `r28_` prefix. The per-edit-type intermediates are named
# `edit{T}_FiVE_r28_...` and so never match the arm glob either.
_NOT_AN_ARM = re.compile(r"(_r2[0-7]_|_per_frame|final_averaged)")
_ARM_RE = re.compile(r"r28_(.+)_avg\.csv$")
_PER_VIDEO = "evaluation/csv/edit{T}_FiVE_r28_{arm}_frame_stride8.csv"
EDIT_TYPES = (1, 2, 3, 4, 5, 6)


def read_per_video(arm: str) -> Tuple[List[str], Dict[Tuple[int, str], Dict[str, float]]]:
    """Per-video rows for one arm across all six edit types, keyed (edit_type, file_id)."""
    metrics: Optional[List[str]] = None
    out: Dict[Tuple[int, str], Dict[str, float]] = {}
    for t in EDIT_TYPES:
        p = _PER_VIDEO.format(T=t, arm=arm)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} is missing. Every arm must be scored on all six edit types; a "
                "partial arm would still produce a full-looking row."
            )
        r = list(csv.reader(open(p, newline="", encoding="utf-8")))
        m = [c.split("|")[-1] for c in r[0][1:]]
        if metrics is None:
            metrics = m
        elif m != metrics:
            raise ValueError(f"{p} has metric set {m}, earlier types had {metrics}")
        for row in r[1:]:
            out[(t, row[0])] = {k: float(v) for k, v in zip(m, row[1:])}
    assert metrics is not None
    return metrics, out


def read_macro(path: str) -> Dict[str, float]:
    """The scaled macro row evaluate.py writes; carried for continuity only."""
    r = list(csv.reader(open(path, newline="", encoding="utf-8")))
    names = [c.split("|")[-1] for c in r[0]]
    return {n: float(v) for n, v in zip(names, r[1]) if n != "file_id"}


def background_areas(arm: str, cases, mask_root: str, data_root: str,
                     stride: int) -> Tuple[float, float]:
    """(per-arm, fixed) mean background fraction for this arm: the region each family scores.

    An arm whose grounding under-fired has a LARGER per-arm background and an easier score;
    the fixed value is identical for every arm by construction and is returned as a check.
    """
    per, fix = [], []
    for c in cases:
        t, v = int(c["edit_type"]), c["video_name"]
        mp = os.path.join(mask_root, arm, f"edit{t}", f"{v}.npz")
        fp = os.path.join(mask_root, "_fixed_union", f"edit{t}", f"{v}.npz")
        if not (os.path.exists(mp) and os.path.exists(fp)):
            continue
        M = load_npz(mp, expect_stride=stride)
        U = load_fixed_union(fp, expect_stride=stride)
        sp = list_images(os.path.join(data_root, "images", v))[::stride][: M.shape[0]]
        if not sp:
            continue
        from PIL import Image
        w, h = Image.open(sp[0]).size
        S = load_src_masks(os.path.join(data_root, "bmasks", v), sp, (h, w))
        if S is None:
            continue
        n = min(M.shape[0], U.shape[0], S.shape[0])
        per.append(float((~(S[:n] | M[:n])).mean()))
        fix.append(float((~U[:n]).mean()))
    return (mean(per) if per else float("nan"),
            mean(fix) if fix else float("nan"))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm_glob", default="evaluation/csv/r28_*_avg.csv")
    ap.add_argument("--control", default="taubg2_taufg2_vp",
                    help="vp Eq.4 reference the tau deltas are taken against")
    ap.add_argument("--novp_baseline", default="baseline",
                    help="novp arm, reported separately (moves vp AND tau)")
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--mask_root", default="/projects/dataggen/outputs/five_bench/r28_tgt_masks")
    ap.add_argument("--data_root", default=os.path.expanduser(
        "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--frame_stride", type=int, default=8)
    ap.add_argument("--metric", default="lpips",
                    help="base metric the ordering/Spearman is computed on")
    ap.add_argument("--skip_areas", action="store_true",
                    help="skip the background-area columns (they reread every mask)")
    ap.add_argument("-o", "--out", default="evaluation/csv/r28_union_vs_part.csv")
    args = ap.parse_args(argv)

    matches = sorted(glob.glob(args.arm_glob))
    if not matches:
        print(f"[r28_sum] no files matched {args.arm_glob!r}", file=sys.stderr)
        return 1
    bad = [p for p in matches if _NOT_AN_ARM.search(os.path.basename(p))]
    if bad:
        print("[r28_sum] --arm_glob matched non-arm CSVs; refusing:", file=sys.stderr)
        for p in bad:
            print(f"    {p}", file=sys.stderr)
        return 1
    arms = []
    for p in matches:
        m = _ARM_RE.search(os.path.basename(p))
        if not m:
            print(f"[r28_sum] cannot parse an arm from {p!r}", file=sys.stderr)
            return 1
        arms.append((m.group(1), p))
    print(f"[r28_sum] {len(arms)} arms: {', '.join(a for a, _ in arms)}")

    cases = json.load(open(os.path.expanduser(args.cases), encoding="utf-8"))
    data = {}
    macro = {}
    metrics = None
    for arm, p in arms:
        m, rows = read_per_video(arm)
        if metrics is None:
            metrics = m
        elif m != metrics:
            print(f"[r28_sum] arm {arm} has a different metric set", file=sys.stderr)
            return 1
        data[arm] = rows
        macro[arm] = read_macro(p)
    counts = {a: len(d) for a, d in data.items()}
    if len(set(counts.values())) != 1:
        print(f"[r28_sum] arms disagree on clip count: {counts}", file=sys.stderr)
        return 1
    n_clips = next(iter(counts.values()))
    print(f"[r28_sum] {n_clips} clips per arm, {len(metrics)} metrics")

    names = [a for a, _ in arms]
    if args.control not in names:
        print(f"[r28_sum] control {args.control!r} not among the arms", file=sys.stderr)
        return 1
    keys = sorted(data[args.control])
    # The degenerate R26 clip carries no spatial signal; report both subsets.
    degen = {(6, k[1]) for k in keys if k[0] == 6}
    subsets = [("all", keys), ("excl_edit6", [k for k in keys if k[0] != 6])]

    areas = {}
    if not args.skip_areas:
        print("[r28_sum] computing per-arm background areas ...")
        for a in names:
            areas[a] = background_areas(a, cases, args.mask_root, args.data_root,
                                        args.frame_stride)

    header = ["subset", "arm", "n_clips", "is_control", "is_novp",
              "bg_area_per_arm", "bg_area_fixed"]
    for b in BASES:
        for fam in FAMILIES:
            header += [f"{b}_unedit_{fam}", f"{b}_unedit_{fam}_delta",
                       f"{b}_unedit_{fam}_macro"]
    out_rows = []
    for sub, ks in subsets:
        ctl = {f"{b}_unedit_{f}": mean(data[args.control][k][f"{b}_unedit_{f}"] for k in ks)
               for b in BASES for f in FAMILIES}
        for a in names:
            ar = areas.get(a, (float("nan"), float("nan")))
            row = [sub, a, str(len(ks)), "1" if a == args.control else "0",
                   "1" if a == args.novp_baseline else "0",
                   f"{ar[0]:.6f}", f"{ar[1]:.6f}"]
            for b in BASES:
                for f in FAMILIES:
                    col = f"{b}_unedit_{f}"
                    v = mean(data[a][k][col] for k in ks)
                    row += [f"{v:.6f}", f"{v - ctl[col]:+.6f}",
                            f"{macro[a].get(col, float('nan')):.6f}"]
            out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out_rows)
    print(f"[r28_sum] wrote {args.out} ({len(out_rows)} rows)")

    # ---- the headline: does the union change the ORDERING? ------------------------
    base = args.metric
    higher = base in HIGHER_IS_BETTER
    tau_arms = [a for a in names if a.startswith("taubg")]
    for sub, ks in subsets:
        vals = {f: {a: mean(data[a][k][f"{base}_unedit_{f}"] for k in ks) for a in tau_arms}
                for f in FAMILIES}
        print(f"\n=== {base} across the {len(tau_arms)} tau arms — subset={sub} "
              f"({'higher' if higher else 'lower'} is better) ===")
        print(f"  {'arm':24s}" + "".join(f"{f:>14s}" for f in FAMILIES))
        for a in tau_arms:
            print(f"  {a:24s}" + "".join(f"{vals[f][a]:14.4f}" for f in FAMILIES))
        print(f"  {'SPREAD (max-min)':24s}" +
              "".join(f"{max(vals[f].values()) - min(vals[f].values()):14.4f}" for f in FAMILIES))
        order = {f: sorted(tau_arms, key=lambda a: vals[f][a], reverse=higher) for f in FAMILIES}
        print(f"  best by part : {order['part'][0]}")
        print(f"  best by union: {order['union'][0]}")
        print(f"  best by fixed: {order['union_fixed'][0]}")
        for f in ("union", "union_fixed"):
            rho = spearman([vals["part"][a] for a in tau_arms], [vals[f][a] for a in tau_arms])
            print(f"  Spearman(part, {f}) = {rho:+.4f}"
                  + ("   <- ordering essentially unchanged" if rho > 0.9 else
                     "   <- ORDERING CHANGES: R26's part-based conclusions must be re-read"))

    # ---- the two comparators, kept apart ------------------------------------------
    print("\n=== comparators (vp-matched vs novp) ===")
    for a in names:
        if not a.startswith("taubg"):
            tag = "novp — moves vp AND tau, NOT a tau comparison" if a == args.novp_baseline \
                  else "vp Eq.4 — isolates tau"
            v = {f: mean(data[a][k][f"{base}_unedit_{f}"] for k in keys) for f in FAMILIES}
            print(f"  {a:24s} part={v['part']:9.4f} union={v['union']:9.4f} "
                  f"fixed={v['union_fixed']:9.4f}   [{tag}]")

    if areas:
        print("\n=== per-arm background area (the region each family scores) ===")
        print("  a LARGER per-arm background = weaker grounding = easier score; the fixed")
        print("  column must be identical for every arm by construction")
        for a in names:
            print(f"  {a:24s} per_arm={areas[a][0]:.4f}  fixed={areas[a][1]:.4f}")
        fx = [areas[a][1] for a in names]
        if max(fx) - min(fx) > 1e-9:
            print(f"  ⚠ fixed background varies by {max(fx) - min(fx):.2e} across arms; "
                  "it must not — the arms were scored on different fixed regions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
