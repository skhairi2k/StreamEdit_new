#!/usr/bin/env python3
"""R20 blend-schedule comparison table.

Joins the per-arm ``r20_*_avg.csv`` metric files produced by
``evaluation/fivebench/evaluate.py`` into one wide comparison CSV, adding, for
every arm, the per-metric delta against the paper (Eq.4) reference of the SAME
VP mode:

    novp arms  ->  --ref_novp   (five_bench/baseline,            R1)
    vp   arms  ->  --ref_vp      (five_bench/r7_visual_prompting, R7)
    pvp  arms  ->  --ref_pvp     (r20_blend_sched/paper_pvp)

Only same-VP-mode deltas are meaningful: the blend schedule and the anchor
mechanism move the same quantity, so a schedule is judged against its own
regime's Eq.4 baseline, never across regimes.

Schema of the input _avg.csv (written by evaluate.py): a header row then ONE
mean row. Metric columns are the ``group|metric`` names (e.g.
``all|structure_distance``); ``file_id`` and any other bare columns are skipped.
Scaling (x1000 struct/lpips, x10000 mse, x100 ssim/motion) is already applied by
evaluate.py -- we do not rescale here.

Metric direction (for the printed verdict and the ``better?`` flag):
    lower is better : structure_distance, lpips_*, mse_*
    higher is better: psnr_*, ssim_*, clip_similarity_*, motion_fidelity_score
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Optional

VP_MODES = ("novp", "vp", "pvp")
# canonical row order within each VP block (missing arms are simply skipped)
SCHED_ORDER = ["paper", "cos_full", "cos_half", "cos_third", "const", "zero"]
SKIP_COLS = {"file_id", "video_name", "editing_type_id", "method",
             "frame_idx", "metric", "value"}


def is_lower_better(metric: str) -> bool:
    m = metric.lower()
    return any(tok in m for tok in ("structure_distance", "lpips", "mse", "niqe"))


def read_avg_csv(path: str) -> dict[str, float]:
    """Return {bare_metric_name: value} for the single mean row; skip non-metrics.

    Metric columns are named ``<method>|<metric>`` where <method> is the file's
    own arm/reference directory name (e.g. ``cos_full_novp|structure_distance``,
    ``baseline|structure_distance``). We key by the bare metric so arm and
    reference columns align across files.
    """
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"{path}: expected header + 1 mean row, got {len(rows)}")
    header, values = rows[0], rows[1]
    out: dict[str, float] = {}
    for name, val in zip(header, values):
        if name in SKIP_COLS or "|" not in name:
            continue                 # file_id and any bare column
        bare = name.split("|", 1)[1]
        try:
            out[bare] = float(val)
        except (ValueError, TypeError):
            pass                     # "N/A" etc.
    return out


def arm_from_filename(path: str) -> tuple[str, str, str]:
    """r20_cos_full_novp_avg.csv -> ('cos_full_novp', 'cos_full', 'novp')."""
    stem = os.path.basename(path)
    if stem.startswith("r20_"):
        stem = stem[len("r20_"):]
    if stem.endswith("_avg.csv"):
        stem = stem[:-len("_avg.csv")]
    for vp in VP_MODES:
        if stem.endswith("_" + vp):
            return stem, stem[: -(len(vp) + 1)], vp
    raise ValueError(f"cannot parse vp_mode from arm file: {path}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm_csv_glob",
                    default=str(repo / "evaluation/csv/r20_*_avg.csv"))
    ap.add_argument("--ref_novp",
                    default=str(repo / "evaluation/csv/r20_ref_novp_avg.csv"))
    ap.add_argument("--ref_vp",
                    default=str(repo / "evaluation/csv/r20_ref_vp_avg.csv"))
    ap.add_argument("--ref_pvp",
                    default=str(repo / "evaluation/csv/r20_paper_pvp_avg.csv"))
    ap.add_argument("-o", "--out",
                    default=str(repo / "evaluation/csv/r20_blend_sched.csv"))
    args = ap.parse_args()

    refs = {"novp": args.ref_novp, "vp": args.ref_vp, "pvp": args.ref_pvp}
    missing = [p for p in refs.values() if not os.path.isfile(p)]
    if missing:
        raise SystemExit("[r20_summarize] reference CSV(s) not found -- has "
                         "r20_eval.sh finished?\n  " + "\n  ".join(missing))
    ref_metrics = {mode: read_avg_csv(p) for mode, p in refs.items()}
    ref_real = {os.path.realpath(p) for p in refs.values()}

    # collect arm CSVs, excluding the reference files themselves
    arm_files = [p for p in sorted(glob.glob(args.arm_csv_glob))
                 if os.path.realpath(p) not in ref_real]
    if not arm_files:
        raise SystemExit(f"[r20_summarize] no arm CSVs match {args.arm_csv_glob} "
                         "-- has r20_eval.sh finished?")

    arms: dict[str, dict] = {}          # arm_name -> {sched, vp, metrics}
    metric_order: list[str] = []
    for p in arm_files:
        try:
            arm, sched, vp = arm_from_filename(p)
        except ValueError as e:
            print(f"[warn] skip {p}: {e}")
            continue
        m = read_avg_csv(p)
        arms[arm] = {"sched": sched, "vp": vp, "metrics": m}
        for k in m:
            if k not in metric_order:
                metric_order.append(k)

    # also expose the reference rows in the table (delta = 0 by definition)
    for mode in VP_MODES:
        arms[f"paper_{mode}"] = {"sched": "paper", "vp": mode,
                                 "metrics": ref_metrics[mode], "is_ref": True}
        for k in ref_metrics[mode]:
            if k not in metric_order:
                metric_order.append(k)

    # ---- write wide CSV: one row per arm, value + delta per metric ----
    header = ["arm", "sched", "vp_mode", "ref"]
    for m in metric_order:
        header += [m, m + "|Δref"]

    def sort_key(name: str):
        a = arms[name]
        s = a["sched"]
        return (VP_MODES.index(a["vp"]),
                SCHED_ORDER.index(s) if s in SCHED_ORDER else len(SCHED_ORDER))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for name in sorted(arms, key=sort_key):
            a = arms[name]
            ref = ref_metrics[a["vp"]]
            row = [name, a["sched"], a["vp"],
                   os.path.basename(refs[a["vp"]])]
            for m in metric_order:
                v = a["metrics"].get(m)
                if v is None:
                    row += ["", ""]
                    continue
                if a.get("is_ref"):
                    row += [f"{v:.4f}", "0.0000"]
                else:
                    rv = ref.get(m)
                    d = "" if rv is None else f"{v - rv:+.4f}"
                    row += [f"{v:.4f}", d]
            w.writerow(row)
    print(f"[r20_summarize] wrote {args.out}  "
          f"({len(arms)} arms x {len(metric_order)} metrics)")

    # ---- stdout verdict: per VP mode, which cosine schedule beats Eq.4 ----
    print("\n=== does any cosine schedule beat Eq.4 (per VP mode)? ===")
    for mode in VP_MODES:
        ref = ref_metrics[mode]
        cos = [n for n, a in arms.items()
               if a["vp"] == mode and a["sched"].startswith("cos_")]
        if not cos:
            continue
        print(f"\n[{mode}]  (+ = arm better than Eq.4)")
        for m in metric_order:
            rv = ref.get(m)
            if rv is None:
                continue
            lo = is_lower_better(m)
            parts = []
            for n in sorted(cos, key=lambda x: SCHED_ORDER.index(arms[x]["sched"])):
                v = arms[n]["metrics"].get(m)
                if v is None:
                    continue
                better = (v < rv) if lo else (v > rv)
                parts.append(f"{arms[n]['sched']}={v:.2f}{'+' if better else '-'}")
            arrow = "↓" if lo else "↑"
            print(f"  {m:<44} {arrow} Eq.4={rv:.2f} | " + "  ".join(parts))


if __name__ == "__main__":
    main()
