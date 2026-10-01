#!/usr/bin/env python
"""R25 phase 2 -- three-way attribution table: adaptive vs reversed vs constant.

Phase 1 showed the adaptive arm beats Eq. 4, but changed TWO things relative to the
baseline at once -- the SET of tau values and their ASSIGNMENT to videos -- so a win
does not say which one did the work. Two more arms, built by r25_tau_map.py --assign,
isolate each:

    reversed   SAME tau multiset as adaptive, pairing inverted by IoU rank
               -> does the PAIRING carry signal?
    constant   every pair gets the realized mean tau
               -> does VARYING tau help at all?

Two contrasts decide it:

    adaptive - reversed   adaptive ~= reversed  =>  the IoU routing contributes nothing
    adaptive - constant   adaptive ~= constant   =>  the IoU machinery is unnecessary

Both are computed as PAIRED per-case deltas (same 22 cases, same file_id join every
arm shares with the stored Eq. 4 reference), because a mean alone hides a lopsided
split -- phase 1's own +1.064 mean CLIP gain over Eq. 4 was a 15-positive/7-negative
split, sign test p = 0.134, not the clean win the mean suggested. A two-sided exact
binomial sign test on each contrast x metric is reported for exactly that reason: it is
distribution-free and does not assume the per-case deltas are normally distributed,
unlike a paired t-test.

The Eq. 4 reference is carried through for context only (same re-averaging-over-the-
joined-cases convention as r25_summarize.py) -- it is not part of either contrast, since
both contrasts are arm-vs-arm and the reference cancels out of them entirely.

Column/join conventions duplicated from r25_summarize.py rather than imported, matching
this repo's self-contained-script convention (see r25_tau_map.py's t_next_schedule for
the same choice): edit_type_of, read_metric_csv, load_video_names.

Usage
-----
    python evaluation/r25_controls.py \\
        --arms adaptive reversed constant \\
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \\
        --iou_csv evaluation/csv/r25_iou.csv \\
        -o evaluation/csv/r25_controls.csv
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

from scipy.stats import binomtest

# Metrics where a LOWER value is better; used only to annotate the printed summary.
LOWER_IS_BETTER = {
    "structure_distance", "lpips_unedit_part", "mse_unedit_part", "niqe_target_image",
    "niqe_source_image",
}

# The two metrics r25_figures.py's delta page and every prior verdict have used as the
# primary read (edit alignment, background preservation). Printed first and always,
# even though every shared metric gets a full row.
HEADLINE_METRICS = ["clip_similarity_target_image", "lpips_unedit_part"]


def edit_type_of(path: str) -> int:
    """Pull the edit type out of an `edit{T}_FiVE_...csv` filename."""
    m = re.search(r"edit(\d)_FiVE", os.path.basename(path))
    if not m:
        raise SystemExit(f"cannot read an edit type from {path}")
    return int(m.group(1))


def read_metric_csv(path: str) -> Tuple[Dict[int, Dict[str, float]], List[str]]:
    """Read one evaluate.py per-case CSV -> {file_id: {metric: value}}, metric order."""
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
    xs = [x for x in xs if x == x]           # drop NaN
    return st.mean(xs) if xs else float("nan")


def load_arm(csv_glob: str, data_root: Path,
             ) -> Tuple[Dict[Tuple[str, int], Dict[str, float]], List[str]]:
    """One arm's per-edit-type CSVs -> {(video_name, edit_type): {metric: value}}."""
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise SystemExit(f"no CSVs matched {csv_glob}")
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    metrics: Optional[List[str]] = None
    for f in files:
        t = edit_type_of(f)
        vals, m = read_metric_csv(f)
        if metrics is None:
            metrics = m
        elif m != metrics:
            raise SystemExit(f"{f}: metric set {m} differs from {metrics} -- arms must "
                             f"be scored with the identical --metrics list")
        names = load_video_names(data_root, t)
        for fid, row in vals.items():
            if fid >= len(names):
                raise SystemExit(f"{f}: file_id {fid} beyond the annotation file "
                                 f"({len(names)} entries)")
            out[(names[fid], t)] = row
    assert metrics is not None
    return out, metrics


def sign_test(deltas: Sequence[float]) -> Tuple[int, int, int, float]:
    """n_pos, n_neg, n_ties, two-sided exact binomial p-value on pos vs neg (ties excluded).

    Distribution-free by design -- phase 1's mean CLIP gain (+1.064) hid a 15/7 split
    that a paired t-test's normality assumption would have papered over.
    """
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = sum(1 for d in deltas if d == 0)
    p = binomtest(pos, pos + neg, 0.5).pvalue if (pos + neg) > 0 else float("nan")
    return pos, neg, ties, p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=str, nargs="+", default=["adaptive", "reversed", "constant"],
                   help="Arm names; each maps to method dir r25_{arm}_vp and CSV glob "
                        "evaluation/csv/edit?_FiVE_r25_{arm}_vp_frame_stride8.csv. "
                        "The two contrasts are always arms[0]-arms[1] and arms[0]-arms[2], "
                        "i.e. adaptive is always the reference arm the others are read "
                        "against, so the DEFAULT ORDER is load-bearing.")
    p.add_argument("--csv_glob_template", type=str,
                   default="evaluation/csv/edit?_FiVE_r25_{arm}_vp_frame_stride8.csv")
    p.add_argument("--ref_csv_glob", type=str,
                   default="evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv")
    p.add_argument("--iou_csv", type=Path, default=Path("evaluation/csv/r25_iou.csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("-o", "--out", type=Path, default=Path("evaluation/csv/r25_controls.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if len(args.arms) != 3:
        raise SystemExit(f"--arms needs exactly 3 names (got {args.arms}); the two "
                         f"contrasts are arms[0]-arms[1] and arms[0]-arms[2]")
    a0, a1, a2 = args.arms

    iou = {(r["video_name"], int(r["edit_type"])): float(r["iou"])
           for r in csv.DictReader(args.iou_csv.open())
           if r["iou"] not in ("", "nan")}

    arm_data: Dict[str, Dict[Tuple[str, int], Dict[str, float]]] = {}
    arm_metrics: Dict[str, List[str]] = {}
    for arm in args.arms:
        d, m = load_arm(args.csv_glob_template.format(arm=arm), data_root)
        arm_data[arm], arm_metrics[arm] = d, m

    # COVERAGE: every arm must score the exact same case set, or the join below would
    # silently compare apples to oranges on the missing keys.
    keysets = {arm: set(d.keys()) for arm, d in arm_data.items()}
    ref_arm = args.arms[0]
    for arm in args.arms[1:]:
        if keysets[arm] != keysets[ref_arm]:
            only_a = keysets[arm] - keysets[ref_arm]
            only_r = keysets[ref_arm] - keysets[arm]
            raise SystemExit(f"coverage mismatch: {arm} vs {ref_arm} -- "
                             f"only in {arm}: {sorted(only_a)}; "
                             f"only in {ref_arm}: {sorted(only_r)}")
    cases_keys = sorted(keysets[ref_arm])

    if any(arm_metrics[arm] != arm_metrics[ref_arm] for arm in args.arms[1:]):
        raise SystemExit(f"metric sets differ across arms: {arm_metrics}")
    arm_shared = arm_metrics[ref_arm]

    # ref, joined the same way r25_summarize.py does (file_id -> video_name), but keyed
    # by (video_name, edit_type) here so it lines up with the three arms directly.
    ref_files = {edit_type_of(f): f for f in sorted(glob.glob(args.ref_csv_glob))}
    ref_data: Dict[Tuple[str, int], Dict[str, float]] = {}
    ref_metrics: Optional[List[str]] = None
    for t in sorted({t for _, t in cases_keys}):
        if t not in ref_files:
            raise SystemExit(f"no reference CSV for edit{t} (glob {args.ref_csv_glob})")
        ref_vals, r_m = read_metric_csv(ref_files[t])
        ref_metrics = r_m
        names = load_video_names(data_root, t)
        for fid, row in ref_vals.items():
            if fid < len(names):
                ref_data[(names[fid], t)] = row
    assert ref_metrics is not None
    shared = [m for m in arm_shared if m in ref_metrics]   # arm-vs-ref intersection (context only)

    contrasts = [(a0, a1, "adaptive_minus_reversed"), (a0, a2, "adaptive_minus_constant")]

    # ---- per-case rows -------------------------------------------------------------
    rows: List[Dict[str, object]] = []
    for key in cases_keys:
        v, t = key
        row: Dict[str, object] = {"video_name": v, "edit_type": t,
                                  "iou": iou.get(key, float("nan"))}
        for arm in args.arms:
            for m in arm_shared:
                row[f"{arm}|{m}"] = arm_data[arm][key][m]
        if key in ref_data:
            for m in shared:
                row[f"ref|{m}"] = ref_data[key][m]
        for lo, hi, label in contrasts:
            for m in arm_shared:
                row[f"{label}|{m}"] = arm_data[lo][key][m] - arm_data[hi][key][m]
        rows.append(row)

    fields = (["video_name", "edit_type", "iou"]
              + [f"{arm}|{m}" for arm in args.arms for m in arm_shared]
              + [f"ref|{m}" for m in shared]
              + [f"{label}|{m}" for _, _, label in contrasts for m in arm_shared])

    def agg_row(label: str, sub: List[Dict[str, object]]) -> Dict[str, object]:
        out: Dict[str, object] = {"video_name": label, "edit_type": "",
                                  "iou": round(mean([r["iou"] for r in sub]), 6)}
        for f in fields[3:]:
            out[f] = mean([r[f] for r in sub])
        return out

    out_rows = list(rows)
    for t in sorted({t for _, t in cases_keys}):
        sub = [r for r in rows if r["edit_type"] == t]
        out_rows.append(agg_row(f"MEAN_edit{t}_n{len(sub)}", sub))
    out_rows.append(agg_row(f"MEAN_ALL_n{len(rows)}", rows))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # ---- sign tests, one per contrast x metric, sidecar JSON since a scalar p-value
    # does not fit the per-case row shape --------------------------------------------
    sign_results: Dict[str, Dict[str, object]] = {}
    for lo, hi, label in contrasts:
        sign_results[label] = {}
        for m in arm_shared:
            deltas = [r[f"{label}|{m}"] for r in rows]
            pos, neg, ties, p = sign_test(deltas)
            sign_results[label][m] = {
                "mean_delta": round(mean(deltas), 6), "n_pos": pos, "n_neg": neg,
                "n_ties": ties, "sign_test_p": round(p, 6) if p == p else None,
            }
    meta = {
        "arms": args.arms, "n_cases": len(rows),
        "contrasts": {label: f"{lo} - {hi}" for lo, hi, label in contrasts},
        "sign_tests": sign_results,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # ---- console summary ------------------------------------------------------------
    overall = out_rows[-1]
    print("=== R25 controls: adaptive vs reversed vs constant ===", flush=True)
    print(f"cases              {len(rows)}", flush=True)
    print(f"metrics compared   {len(arm_shared)}  {arm_shared}", flush=True)
    print(f"arms               {args.arms}  (contrasts: {a0}-{a1}, {a0}-{a2})", flush=True)
    print()
    print(f"{'metric':42s} {a0:>10s} {a1:>10s} {a2:>10s} {'ref':>10s}", flush=True)
    for m in arm_shared:
        vals = [overall.get(f"{arm}|{m}", float('nan')) for arm in args.arms]
        r_ = overall.get(f"ref|{m}", float("nan"))
        print(f"{m:42s} " + " ".join(f"{v:10.5f}" for v in vals) + f" {r_:10.5f}",
              flush=True)
    print()
    for lo, hi, label in contrasts:
        print(f"--- {lo} - {hi} ({label}) ---", flush=True)
        for m in HEADLINE_METRICS + [x for x in arm_shared if x not in HEADLINE_METRICS]:
            sr = sign_results[label][m]
            good = (sr["mean_delta"] < 0) if m in LOWER_IS_BETTER else (sr["mean_delta"] > 0)
            arrow = lo if good else hi
            star = " *" if m in HEADLINE_METRICS else ""
            print(f"  {m:40s} mean_delta {sr['mean_delta']:+9.5f}  "
                  f"{sr['n_pos']:2d}pos/{sr['n_neg']:2d}neg/{sr['n_ties']:d}tie  "
                  f"sign_p={sr['sign_test_p']}  {arrow}{star}", flush=True)
    print("\n(* headline metric)", flush=True)
    print(f"\nwrote {args.out}  ({len(out_rows)} rows = {len(rows)} cases + "
          f"{len(out_rows)-len(rows)} mean rows)", flush=True)
    print(f"wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
