#!/usr/bin/env python
"""R26: join the 9 spatial-tau arm CSVs into one table with deltas vs the (2,2) control.

Pass 2 of R26 sweeps ``tau_bg in {0,1,2} x tau_fg in {2,6,10}``; the ``(2,2)`` cell is a
degenerate control that must reproduce StreamGVE Eq. 4. This script joins the arms, emits
per-metric deltas against that control, and cross-checks the control against the stored
R21 visual-prompting reference restricted to the same pairs.

Two weightings are emitted for every metric, on purpose:

``micro``
    The plain mean over the clips actually scored (22 for R26). This is the primary
    reading and the one the deltas are computed from.

``macro``
    What ``evaluate.py`` writes into ``{stem}_avg.csv``: the mean over the SIX edit-type
    means, with per-metric scaling (structure_distance x1000, lpips x1000, mse x10000,
    ssim x100). Because R26's case set is unbalanced -- edit2 holds 13 of the 22 pairs
    while edit1 and edit6 hold one each -- macro weighting gives each singleton clip
    1/6 = 16.7% of the table instead of 1/22 = 4.5%. It is carried here for continuity
    with numbers already read off the ``_avg.csv`` files, NOT as the primary reading.

That distinction matters for R26 specifically: ``0042_gym-ball`` is the sole edit6 pair
and its grounding mask came out all-ones (see the ``wait-dump`` run-log entry), so it
carries no spatial signal at all. Under macro weighting that one degenerate clip drives a
sixth of every number. Rows are therefore emitted twice, once over all clips and once
with the degenerate clips dropped, so the verdict can be read against both.

Usage
-----
    python evaluation/r26_summarize.py \\
        --arm_glob 'evaluation/csv/r26_taubg*_taufg*_vp_avg.csv' \\
        --control taubg2_taufg2_vp \\
        --cases evaluation/cases.json \\
        -o evaluation/csv/r26_spatial_tau.csv
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

# ---------------------------------------------------------------------------------
# The arm glob is deliberately narrow. `evaluation/csv/` also holds the surviving R21
# reference tables (`*_r21_ref_*`, `*_r21_*`), and in R21 a loose glob swept those in as
# if they were experiment arms, silently adding phantom rows to the summary. Anything
# matching these patterns is refused rather than skipped, so a mistyped glob fails loudly.
# ---------------------------------------------------------------------------------
_NOT_AN_ARM = re.compile(r"(_r21_|_ref_|_r2[0-5]_|_baseline_)")

_ARM_RE = re.compile(r"r26_(taubg(\d+)_taufg(\d+)_vp)_avg\.csv$")

# Per-video CSVs written by evaluate.py, one per edit type.
_PER_VIDEO_TMPL = "evaluation/csv/edit{T}_FiVE_r26_{arm}_frame_stride8.csv"
_R21_REF_TMPL = "evaluation/csv/edit{T}_FiVE_r21_ref_vp_frame_stride8.csv"

# How evaluate.py scales each metric before writing {stem}_avg.csv. Anything absent is
# written unscaled. Reproduced here only so the macro columns can be reported in the same
# units the _avg.csv files use.
_MACRO_SCALE = {
    "structure_distance": 1000.0,
    "lpips_unedit_part": 1000.0,
    "mse_unedit_part": 10000.0,
    "ssim_unedit_part": 100.0,
}

# Higher-is-better metrics; everything else reads lower-is-better. Used only for the
# console summary, never for the CSV.
_HIGHER_IS_BETTER = {
    "psnr_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
}

EDIT_TYPES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------
def _read_per_video(path: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Read one ``edit{T}_..._frame_stride8.csv`` into ``{file_id: {metric: value}}``."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"{path}: no data rows")
    # Header cells are '<method>|<metric>'; the first is the bare 'file_id'.
    metrics = [c.split("|")[-1] for c in rows[0][1:]]
    out: Dict[str, Dict[str, float]] = {}
    for row in rows[1:]:
        if len(row) != len(metrics) + 1:
            raise ValueError(
                f"{path}: row for file_id={row[0]!r} has {len(row)} fields, "
                f"expected {len(metrics) + 1}"
            )
        out[row[0]] = {m: float(v) for m, v in zip(metrics, row[1:])}
    return metrics, out


def load_arm(arm: str) -> Tuple[List[str], Dict[Tuple[int, str], Dict[str, float]]]:
    """Load an arm's per-video rows across all six edit types, keyed ``(edit_type, file_id)``."""
    metrics: Optional[List[str]] = None
    data: Dict[Tuple[int, str], Dict[str, float]] = {}
    for t in EDIT_TYPES:
        path = _PER_VIDEO_TMPL.format(T=t, arm=arm)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"R26: {path} is missing. Every arm must be scored on all six edit types; "
                "a partial arm would still produce a full-looking row."
            )
        m, rows = _read_per_video(path)
        if metrics is None:
            metrics = m
        elif m != metrics:
            raise ValueError(
                f"R26: {path} has metric columns {m} but earlier edit types had {metrics}. "
                "Arms scored with different metric sets are not comparable."
            )
        for fid, vals in rows.items():
            data[(t, fid)] = vals
    assert metrics is not None
    return metrics, data


def load_macro(avg_path: str) -> Dict[str, float]:
    """Read a ``{stem}_avg.csv`` row.

    evaluate.py writes one header row and one value row of equal width, where the leading
    header cell is the literal 'file_id' and its value slot is not a metric. The remaining
    cells align one-to-one with their headers.
    """
    with open(avg_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) != 2:
        raise ValueError(f"{avg_path}: expected exactly 2 rows, found {len(rows)}")
    names = [c.split("|")[-1] for c in rows[0]]
    if len(names) != len(rows[1]):
        raise ValueError(f"{avg_path}: header/value width mismatch")
    return {n: float(v) for n, v in zip(names, rows[1]) if n != "file_id"}


# ---------------------------------------------------------------------------------
# degenerate-clip detection
# ---------------------------------------------------------------------------------
def find_degenerate(mask_dir: str, cases: Sequence[dict]) -> Dict[Tuple[int, str], str]:
    """Return ``{(edit_type, video_name): reason}`` for clips whose union mask is degenerate.

    A mask that is all-ones (or all-zeros) on every latent frame carries no spatial signal:
    ``tau(f, p)`` collapses to a single global exponent, so the clip cannot speak to whether
    a SPATIAL tau helps. Detected rather than hardcoded, so a re-dump that fixes a clip
    silently stops excluding it.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is present in every env used here
        print("[r26_summarize] numpy unavailable; skipping degenerate-clip detection")
        return {}

    flagged: Dict[Tuple[int, str], str] = {}
    for case in cases:
        t, name = int(case["edit_type"]), case["video_name"]
        path = os.path.join(mask_dir, f"edit{t}", f"{name}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as d:
            shape = d["shape"]
            m = np.unpackbits(d["M"], axis=-1)[:, : int(shape[1])].astype(bool)
        frac = m.mean(axis=1)
        if float(frac.min()) == 1.0:
            flagged[(t, name)] = "all-ones"
        elif float(frac.max()) == 0.0:
            flagged[(t, name)] = "all-zeros"
    return flagged


# ---------------------------------------------------------------------------------
# control cross-check
# ---------------------------------------------------------------------------------
def crosscheck_control(
    control_data: Dict[Tuple[int, str], Dict[str, float]],
    metrics: Sequence[str],
    tol: float,
) -> Tuple[bool, List[str]]:
    """Compare the control's per-video rows against the stored R21 vp reference.

    The control is Eq. 4 under vp anchoring, which is what the R21 reference recorded on
    the full bench. Restricting the reference to the pairs R26 actually rendered, the two
    should agree; a mismatch means this 22-pair subset is NOT reproducing the full run and
    every delta in the table is suspect.
    """
    notes: List[str] = []
    worst: List[Tuple[float, str]] = []
    compared = 0
    missing_ref = 0

    for t in EDIT_TYPES:
        path = _R21_REF_TMPL.format(T=t)
        if not os.path.exists(path):
            notes.append(f"reference missing for edit{t}: {path}")
            continue
        _, ref_rows = _read_per_video(path)
        for (et, fid), vals in control_data.items():
            if et != t:
                continue
            ref = ref_rows.get(fid)
            if ref is None:
                missing_ref += 1
                continue
            for m in metrics:
                if m not in ref:
                    continue
                a, b = vals[m], ref[m]
                denom = max(abs(a), abs(b), 1e-12)
                rel = abs(a - b) / denom
                compared += 1
                worst.append((rel, f"edit{t} file_id={fid} {m}: control={a:.6g} ref={b:.6g} rel={rel:.3g}"))

    if not compared:
        return False, ["no overlapping (edit_type, file_id) rows between control and R21 reference"]

    worst.sort(reverse=True)
    max_rel = worst[0][0]
    ok = max_rel <= tol
    notes.append(f"compared {compared} (pair, metric) cells; max relative difference {max_rel:.3g} (tol {tol:g})")
    if missing_ref:
        notes.append(f"{missing_ref} control pairs had no matching row in the R21 reference")
    if not ok:
        notes.append("worst offenders:")
        notes.extend("    " + w[1] for w in worst[:8])
    return ok, notes


# ---------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm_glob", default="evaluation/csv/r26_taubg*_taufg*_vp_avg.csv",
                    help="glob identifying the arms (matched on the _avg.csv files)")
    ap.add_argument("--control", default="taubg2_taufg2_vp", help="arm name of the degenerate control")
    ap.add_argument("--cases", default="evaluation/cases.json", help="case manifest rendered by pass 2")
    ap.add_argument("--masks", default="/projects/dataggen/outputs/five_bench/r26_masks",
                    help="pass-1 union-mask root, used to detect degenerate (all-ones/all-zeros) clips")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="relative tolerance for the control vs R21-reference cross-check")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if the control cross-check fails")
    ap.add_argument("-o", "--out", default="evaluation/csv/r26_spatial_tau.csv")
    args = ap.parse_args(argv)

    # ---- resolve arms -------------------------------------------------------------
    matches = sorted(glob.glob(args.arm_glob))
    if not matches:
        print(f"[r26_summarize] no files matched --arm_glob {args.arm_glob!r}", file=sys.stderr)
        return 1
    bad = [p for p in matches if _NOT_AN_ARM.search(os.path.basename(p))]
    if bad:
        print("[r26_summarize] --arm_glob matched non-arm reference CSVs; refusing to continue:",
              file=sys.stderr)
        for p in bad:
            print(f"    {p}", file=sys.stderr)
        print("  Narrow the glob (the R21 bug: reference tables swept in as phantom arms).",
              file=sys.stderr)
        return 1

    arms: List[Tuple[int, int, str, str]] = []
    for p in matches:
        m = _ARM_RE.search(os.path.basename(p))
        if not m:
            print(f"[r26_summarize] cannot parse an arm name from {p!r}", file=sys.stderr)
            return 1
        arms.append((int(m.group(2)), int(m.group(3)), m.group(1), p))
    arms.sort()
    print(f"[r26_summarize] {len(arms)} arms: {', '.join(a[2] for a in arms)}")

    if args.control not in {a[2] for a in arms}:
        print(f"[r26_summarize] control {args.control!r} is not among the matched arms", file=sys.stderr)
        return 1

    # ---- load ----------------------------------------------------------------------
    with open(args.cases, encoding="utf-8") as fh:
        cases = json.load(fh)
    case_keys = {(int(c["edit_type"]), c["video_name"]) for c in cases}
    print(f"[r26_summarize] {len(cases)} cases in {args.cases}")

    metrics: Optional[List[str]] = None
    per_arm: Dict[str, Dict[Tuple[int, str], Dict[str, float]]] = {}
    macro: Dict[str, Dict[str, float]] = {}
    for _bg, _fg, arm, avg_path in arms:
        m, data = load_arm(arm)
        if metrics is None:
            metrics = m
        elif m != metrics:
            print(f"[r26_summarize] arm {arm} has a different metric set; not comparable", file=sys.stderr)
            return 1
        per_arm[arm] = data
        macro[arm] = load_macro(avg_path)
    assert metrics is not None

    counts = {arm: len(d) for arm, d in per_arm.items()}
    if len(set(counts.values())) != 1:
        print(f"[r26_summarize] arms disagree on clip count: {counts}", file=sys.stderr)
        print("  Every delta compares different clip sets; refusing to continue.", file=sys.stderr)
        return 1
    n_clips = next(iter(counts.values()))
    print(f"[r26_summarize] {n_clips} clips per arm, {len(metrics)} metrics")
    if n_clips != len(cases):
        print(f"[r26_summarize] WARNING: {n_clips} scored clips vs {len(cases)} cases in the manifest")

    keys = sorted(per_arm[args.control])

    # ---- degenerate clips ----------------------------------------------------------
    degenerate = find_degenerate(args.masks, cases) if os.path.isdir(args.masks) else {}
    if degenerate:
        for (t, name), why in sorted(degenerate.items()):
            print(f"[r26_summarize] DEGENERATE edit{t} {name}: union mask is {why}; "
                  "tau is globally uniform there, so it carries no spatial signal")
    elif not os.path.isdir(args.masks):
        print(f"[r26_summarize] mask dir {args.masks} not found; skipping degenerate-clip detection")

    # Map the degenerate (edit_type, video_name) onto the (edit_type, file_id) keys the
    # CSVs use, via position within each edit type's case list.
    degenerate_keys = set()
    if degenerate:
        by_type: Dict[int, List[str]] = {}
        for c in cases:
            by_type.setdefault(int(c["edit_type"]), []).append(c["video_name"])
        for (t, name) in degenerate:
            fids = sorted([k[1] for k in keys if k[0] == t], key=lambda s: int(s))
            names = by_type.get(t, [])
            if len(fids) == len(names) and name in names:
                degenerate_keys.add((t, fids[names.index(name)]))
    subsets: List[Tuple[str, List[Tuple[int, str]]]] = [("all", keys)]
    if degenerate_keys:
        subsets.append(("excl_degenerate", [k for k in keys if k not in degenerate_keys]))

    # ---- build the table -----------------------------------------------------------
    header = ["subset", "tau_bg", "tau_fg", "arm", "n_clips", "is_control"]
    for m in metrics:
        header += [m, f"{m}_delta", f"{m}_macro"]

    out_rows: List[List[str]] = []
    for subset_name, subset_keys in subsets:
        ctl = {m: mean(per_arm[args.control][k][m] for k in subset_keys) for m in metrics}
        for bg, fg, arm, _ in arms:
            vals = {m: mean(per_arm[arm][k][m] for k in subset_keys) for m in metrics}
            row = [subset_name, str(bg), str(fg), arm, str(len(subset_keys)),
                   "1" if arm == args.control else "0"]
            for m in metrics:
                row.append(f"{vals[m]:.6f}")
                row.append(f"{vals[m] - ctl[m]:+.6f}")
                # macro comes straight from _avg.csv and is always all-clip derived.
                row.append(f"{macro[arm][m]:.6f}" if subset_name == "all" and m in macro[arm] else "")
            out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(out_rows)
    print(f"[r26_summarize] wrote {args.out} ({len(out_rows)} rows)")

    # ---- control cross-check -------------------------------------------------------
    print(f"\n[r26_summarize] cross-checking control {args.control} against the R21 vp reference")
    ok, notes = crosscheck_control(per_arm[args.control], metrics, args.tol)
    for n in notes:
        print(f"    {n}")
    print(f"[r26_summarize] CONTROL-CROSSCHECK {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("    A mismatch means this 22-pair subset is not reproducing the stored full run,")
        print("    so every delta above is suspect. Investigate before reading the grid.")

    # ---- console summary -----------------------------------------------------------
    for m in ("clip_similarity_target_image", "lpips_unedit_part"):
        if m not in metrics:
            continue
        arrow = "higher is better" if m in _HIGHER_IS_BETTER else "lower is better"
        for subset_name, subset_keys in subsets:
            print(f"\n=== {m} ({arrow}) — subset={subset_name}, n={len(subset_keys)} ===")
            fgs = sorted({fg for _, fg, _, _ in arms})
            bgs = sorted({bg for bg, _, _, _ in arms})
            print("          " + "".join(f"tau_fg={f:<9}" for f in fgs))
            for bg in bgs:
                # Grid may be TRIANGULAR (tau_bg <= tau_fg only, since 2026-09-01's
                # extension), so not every (bg, fg) pair is a scored arm -- print a
                # blank rather than crashing on the missing cross-product cell.
                cells = []
                for fg in fgs:
                    arm = next((a for b, f, a, _ in arms if b == bg and f == fg), None)
                    cells.append(mean(per_arm[arm][k][m] for k in subset_keys) if arm else None)
                row = "".join(f"{c:<14.4f}" if c is not None else f"{'--':<14}" for c in cells)
                print(f"tau_bg={bg}  " + row)

    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
