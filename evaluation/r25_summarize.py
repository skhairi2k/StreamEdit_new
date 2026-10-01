#!/usr/bin/env python
"""R25 step 10 -- join the adaptive-tau arm against the stored Eq. 4 vp reference.

For each edit type, the arm's per-case rows are joined to the stored full-bench
reference on `file_id`. That key is sound because evaluate.py assigns file_id from
`enumerate` over the FULL annotation file and `--cases_json` only `continue`s past
unlisted pairs (evaluate.py:229 / :268-272) -- ids do NOT renumber under a subset, so a
22-case run and a 419-case run agree on them. Verified on edit2: the arm's 13 ids all
appear in the reference's 0..99.

THE REFERENCE IS RE-AVERAGED OVER ONLY THE JOINED CASES, never against its 419-pair
mean. Comparing a 22-case arm to a 419-case reference mean would confound the arm's
effect with the difference between the two case sets, which is large here: R25's set is
deliberately 13/22 object swaps.

METRIC SET: the arm was scored with a reduced --metrics list (9 metrics) because the
H100 partition was unavailable and five_acc (Qwen2.5-VL) + motion_fidelity (CoTracker)
do not fit alongside each other on a 40GB card. The stored reference has all 16. Deltas
are therefore computed over the INTERSECTION, and the reference's extra columns are
carried through unpaired so the table still records what the baseline achieved on them.
`motion_fidelity` is the only temporal-consistency metric, so no delta in this table
speaks to temporal quality -- see the plan's decision row before drawing conclusions.

Delta convention: `delta = arm - ref`, i.e. positive means the adaptive arm scored
HIGHER. Whether higher is better depends on the metric (psnr/ssim/clip up is good;
lpips/mse/structure_distance/niqe down is good), so the sign is not a verdict on its own.

Usage
-----
    python evaluation/r25_summarize.py \\
        --arm_csv_glob 'evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8.csv' \\
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \\
        --tau_map evaluation/csv/r25_tau_map.csv \\
        --iou_csv evaluation/csv/r25_iou.csv \\
        -o evaluation/csv/r25_adaptive.csv
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

# Metrics where a LOWER value is better; used only to annotate the printed summary.
LOWER_IS_BETTER = {
    "structure_distance", "lpips_unedit_part", "mse_unedit_part", "niqe_target_image",
    "niqe_source_image",
}


def edit_type_of(path: str) -> int:
    """Pull the edit type out of an `edit{T}_FiVE_...csv` filename."""
    m = re.search(r"edit(\d)_FiVE", os.path.basename(path))
    if not m:
        raise SystemExit(f"cannot read an edit type from {path}")
    return int(m.group(1))


def read_metric_csv(path: str) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
    """Read one evaluate.py per-case CSV -> {file_id: {metric: value}}, metric order.

    Column names are `{method}|{metric}`; the method prefix is stripped so the arm and
    the reference become directly comparable despite being produced by different runs.
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    metrics = [c.split("|", 1)[1] for c in rows[0] if c != "file_id"]
    out: Dict[int, Dict[str, float]] = {}
    for r in rows:
        vals = {}
        for c, v in r.items():
            if c == "file_id":
                continue
            try:
                vals[c.split("|", 1)[1]] = float(v)
            except (TypeError, ValueError):
                vals[c.split("|", 1)[1]] = float("nan")
        out[int(r["file_id"])] = vals
    return out, metrics


def load_video_names(data_root: Path, edit_type: int) -> List[str]:
    """file_id -> video_name, by position in the FULL annotation file."""
    p = data_root / "edit_prompt" / f"edit{edit_type}_FiVE.json"
    if not p.exists():
        raise SystemExit(f"missing annotation file {p}")
    return [e["video_name"] for e in json.loads(p.read_text())]


def mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]          # drop NaN
    return st.mean(xs) if xs else float("nan")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm_csv_glob", type=str,
                   default="evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8.csv")
    p.add_argument("--ref_csv_glob", type=str,
                   default="evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv")
    p.add_argument("--tau_map", type=Path,
                   default=Path("evaluation/csv/r25_tau_map.csv"))
    p.add_argument("--iou_csv", type=Path, default=Path("evaluation/csv/r25_iou.csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"),
                   help="For the file_id -> video_name mapping.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r25_adaptive.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()

    tau = {(r["video_name"], int(r["edit_type"])): float(r["tau"])
           for r in csv.DictReader(args.tau_map.open())}
    iou = {(r["video_name"], int(r["edit_type"])): float(r["iou"])
           for r in csv.DictReader(args.iou_csv.open())
           if r["iou"] not in ("", "nan")}

    arm_files = sorted(glob.glob(args.arm_csv_glob))
    ref_files = {edit_type_of(f): f for f in sorted(glob.glob(args.ref_csv_glob))}
    if not arm_files:
        raise SystemExit(f"no arm CSVs matched {args.arm_csv_glob}")

    cases: List[Dict[str, object]] = []
    shared: Optional[List[str]] = None
    ref_only: List[str] = []

    for af in arm_files:
        t = edit_type_of(af)
        if t not in ref_files:
            raise SystemExit(f"no reference CSV for edit{t} (glob {args.ref_csv_glob})")
        arm, arm_metrics = read_metric_csv(af)
        ref, ref_metrics = read_metric_csv(ref_files[t])
        names = load_video_names(data_root, t)

        # Deltas only where both arms measured the same thing. The arm ran a reduced
        # --metrics list, so this intersection is 9 wide, not 16.
        s = [m for m in arm_metrics if m in ref_metrics]
        if shared is None:
            shared, ref_only = s, [m for m in ref_metrics if m not in arm_metrics]
        elif s != shared:
            raise SystemExit(f"edit{t} metric set {s} differs from {shared}; refusing to "
                             f"average columns that are not the same quantity")

        for fid, avals in sorted(arm.items()):
            if fid not in ref:
                raise SystemExit(f"edit{t}: file_id {fid} is in the arm but not the "
                                 f"reference -- the join key is not stable, stop")
            if fid >= len(names):
                raise SystemExit(f"edit{t}: file_id {fid} beyond the annotation file "
                                 f"({len(names)} entries)")
            v = names[fid]
            row: Dict[str, object] = {
                "video_name": v, "edit_type": t, "file_id": fid,
                "iou": iou.get((v, t), float("nan")),
                "tau": tau.get((v, t), float("nan")),
            }
            row["abs_tau_minus_2"] = abs(row["tau"] - 2.0) if row["tau"] == row["tau"] \
                else float("nan")
            for m in shared:
                row[f"arm|{m}"] = avals[m]
                row[f"ref|{m}"] = ref[fid][m]
                row[f"delta|{m}"] = avals[m] - ref[fid][m]
            for m in ref_only:
                row[f"ref_only|{m}"] = ref[fid][m]
            cases.append(row)

    assert shared is not None
    fields = (["video_name", "edit_type", "file_id", "iou", "tau", "abs_tau_minus_2"]
              + [f"{p}|{m}" for m in shared for p in ("arm", "ref", "delta")]
              + [f"ref_only|{m}" for m in ref_only])

    def block(rows: List[Dict[str, object]], label: str) -> Dict[str, object]:
        agg: Dict[str, object] = {"video_name": label, "edit_type": "",
                                  "file_id": "", "iou": round(mean([r["iou"] for r in rows]), 6),
                                  "tau": round(mean([r["tau"] for r in rows]), 4),
                                  "abs_tau_minus_2": round(mean([r["abs_tau_minus_2"] for r in rows]), 4)}
        for f in fields[6:]:
            agg[f] = mean([r[f] for r in rows])
        return agg

    out_rows = list(cases)
    for t in sorted({r["edit_type"] for r in cases}):
        sub = [r for r in cases if r["edit_type"] == t]
        out_rows.append(block(sub, f"MEAN_edit{t}_n{len(sub)}"))
    out_rows.append(block(cases, f"MEAN_ALL_n{len(cases)}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # ---- console summary -------------------------------------------------------
    overall = out_rows[-1]
    print("=== R25 summarize: adaptive tau vs Eq. 4 vp reference ===", flush=True)
    print(f"cases joined      {len(cases)}", flush=True)
    print(f"metrics compared  {len(shared)}  {shared}", flush=True)
    if ref_only:
        print(f"reference-only    {len(ref_only)}  {ref_only}", flush=True)
        print("                  (arm ran a reduced --metrics list; no delta possible, "
              "and NO temporal metric is compared)", flush=True)
    print(f"mean tau          {overall['tau']:.3f}   mean |tau-2| "
          f"{overall['abs_tau_minus_2']:.3f}", flush=True)
    print()
    print(f"{'metric':42s} {'arm':>11s} {'ref':>11s} {'delta':>11s}  better?", flush=True)
    for m in shared:
        a, r_, d = overall[f"arm|{m}"], overall[f"ref|{m}"], overall[f"delta|{m}"]
        good = (d < 0) if m in LOWER_IS_BETTER else (d > 0)
        arrow = "ARM" if good else "ref"
        print(f"{m:42s} {a:11.5f} {r_:11.5f} {d:+11.5f}  {arrow}", flush=True)
    print(f"\nwrote {args.out}  ({len(out_rows)} rows = {len(cases)} cases + "
          f"{len(out_rows)-len(cases)} mean rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
