"""Join the per-metric CSVs of one method into the unified suite tables -- R57 unified metric suite (stdlib only).

Inputs (all in --csv_dir, produced by slurm_scripts/five_bench/eval_suite_{metrics,cf}.sh with this --tag/--method):
  edit{T}_FiVE_suite_{tag}_{method}_frame_stride8.csv            FiVE harness (+ motion fidelity), one file per edit type
  edit{T}_FiVE_suite_fiveacc_{tag}_{method}_frame_stride8.csv    FiVE-Acc
  suite_clipd_{tag}_{method}.csv                                 CLIP-D (r33_clip_directional.py)
  suite_editreward_{tag}_{method}.csv                            editreward_score.py
  suite_psc_{tag}_{method}.csv                                   psc_score.py
  suite_cf_{tag}_{method}.csv                                    cf_score.py
The harness CSVs carry no video name, only ``file_id`` = the index inside edit{T}_FiVE.json of --edit_json_dir; that file
maps it back.

Outputs
  suite_{tag}_{method}.csv            one row per pair (edit_type, video_name, every metric)
  suite_{tag}_{method}_per_type.csv   plain per-clip means per edit type and overall (never the harness's _avg.csv, whose
                                      columns are rescaled). PSC: ``psc`` is the mean over edit types 1-5 only; edit type
                                      6 (removal) is reported as ``psc_removal`` and never averaged with them. CF and the
                                      rest are averaged over all pairs.

Exit 1 (before writing) on a missing file / metric column, a pair set that differs from the expected one, or duplicated
rows; exit 1 (after writing) when any metric cell is NaN, unless its column is listed in --allow_nan (the harness's
motion_fidelity_score_edit_part is NaN on clips with an empty edit mask, ~1 in 50: allow it for full-bench runs, never for
the smoke).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite_common as sc  # noqa: E402

HARNESS = ("structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part",
           "clip_similarity_source_image", "clip_similarity_target_image", "clip_similarity_target_image_edit_part",
           "niqe_target_image", "motion_fidelity_score", "motion_fidelity_score_edit_part")
FIVEACC = ("five_acc_yes_no", "five_acc_multi_choice")
CLIPD = ("clip_d_prompt", "clip_d_word", "delta_img_norm")
EDITREWARD = ("editreward", "editreward_if", "editreward_vq")
PSC = ("psc", "s_edit", "s_src")
CF = ("cf",)
NUMERIC = HARNESS + FIVEACC + CLIPD + EDITREWARD + PSC + CF
MEAN_COLS = HARNESS + FIVEACC + CLIPD + EDITREWARD + ("psc",) + CF      # per-type table (s_edit / s_src: per-pair only)
Key = Tuple[int, str]


class MergeError(Exception):
    pass


def num(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def read_harness(csv_dir: Path, stem: str, method: str, wanted: Sequence[str],
                 pairs: Sequence[Dict[str, object]]) -> Dict[Key, Dict[str, float]]:
    """Per-type harness CSVs -> {(edit_type, video_name): {metric: value}}."""
    by_type: Dict[int, Dict[int, str]] = {}
    for q in pairs:
        by_type.setdefault(int(q["edit_type"]), {})[int(q["file_id"])] = str(q["video_name"])
    out: Dict[Key, Dict[str, float]] = {}
    for t, names in sorted(by_type.items()):
        path = csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv"
        if not path.exists():
            raise MergeError(f"missing {path}")
        rows = list(csv.DictReader(path.open()))
        if not rows:
            raise MergeError(f"{path} has no rows")
        cols: Dict[str, str] = {}
        for h in rows[0].keys():
            if "|" in h:
                key, metric = h.split("|", 1)
                if key != method:
                    raise MergeError(f"{path}: column '{h}' is not of method '{method}'")
                cols[metric] = h
        missing = [m for m in wanted if m not in cols]
        if missing:
            raise MergeError(f"{path}: missing metric column(s) {missing}")
        for r in rows:
            fid = int(r["file_id"])
            if fid not in names:
                raise MergeError(f"{path}: file_id {fid} is not one of the expected pairs")
            k = (t, names[fid])
            if k in out:
                raise MergeError(f"{path}: duplicated row for {k}")
            out[k] = {m: num(r[cols[m]]) for m in wanted}
    return out


def read_keyed(path: Path, wanted: Sequence[str], extra: Sequence[str] = (), method: str = "") -> Dict[Key, Dict[str, object]]:
    """A suite CSV with edit_type + video_name (+ method column for CLIP-D) -> {(edit_type, video_name): row}."""
    if not path.exists():
        raise MergeError(f"missing {path}")
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise MergeError(f"{path} has no rows")
    missing = [c for c in list(wanted) + list(extra) + ["edit_type", "video_name"] if c not in rows[0]]
    if missing:
        raise MergeError(f"{path}: missing column(s) {missing}")
    out: Dict[Key, Dict[str, object]] = {}
    for r in rows:
        if method and r.get("method") != method:
            continue
        k = (int(r["edit_type"]), r["video_name"])
        if k in out:
            raise MergeError(f"{path}: duplicated row for {k}")
        out[k] = {c: (num(r[c]) if c in wanted else r[c]) for c in list(wanted) + list(extra)}
    return out


def mean(x: Sequence[float]) -> float:
    x = [v for v in x if math.isfinite(v)]
    return sum(x) / len(x) if x else float("nan")


def fmt(v: object) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--edit_json_dir", default="evaluation/five_bench_correct")
    p.add_argument("--cases", default=None)
    p.add_argument("--csv_dir", default="evaluation/csv")
    p.add_argument("--allow_nan", nargs="*", default=[], metavar="COL")
    a = p.parse_args()

    csv_dir = Path(a.csv_dir)
    pairs = sc.load_pairs(a.edit_json_dir, a.cases)
    expected = {(int(q["edit_type"]), str(q["video_name"])) for q in pairs}
    unknown = [c for c in a.allow_nan if c not in NUMERIC]
    if unknown:
        print(f"[merge_suite] FAILED: --allow_nan names unknown column(s) {unknown}")
        return 1
    try:
        parts: List[Dict[Key, Dict[str, object]]] = [
            read_harness(csv_dir, f"suite_{a.tag}_{a.method}", a.method, HARNESS, pairs),
            read_harness(csv_dir, f"suite_fiveacc_{a.tag}_{a.method}", a.method, FIVEACC, pairs),
            read_keyed(csv_dir / f"suite_clipd_{a.tag}_{a.method}.csv", CLIPD, method=a.method),
            read_keyed(csv_dir / f"suite_editreward_{a.tag}_{a.method}.csv", EDITREWARD),
            read_keyed(csv_dir / f"suite_psc_{a.tag}_{a.method}.csv", PSC, extra=("psc_rule",)),
            read_keyed(csv_dir / f"suite_cf_{a.tag}_{a.method}.csv", CF, extra=("cf_raw",)),
        ]
        names = ("harness", "five_acc", "clip_d", "editreward", "psc", "cf")
        for n, part in zip(names, parts):
            if set(part) != expected:
                raise MergeError(f"{n}: {len(part)} rows, expected {len(expected)}; "
                                 f"missing {sorted(expected - set(part))[:5]}, extra {sorted(set(part) - expected)[:5]}")
    except MergeError as e:
        print(f"[merge_suite] FAILED: {e}")
        return 1

    fields = ["edit_type", "video_name"] + list(HARNESS + FIVEACC + CLIPD + EDITREWARD + PSC) + ["psc_rule"] + list(CF) + ["cf_raw"]
    table: List[Dict[str, object]] = []
    for q in pairs:
        k = (int(q["edit_type"]), str(q["video_name"]))
        row: Dict[str, object] = {"edit_type": k[0], "video_name": k[1]}
        for part in parts:
            row.update(part[k])
        table.append(row)

    nan_cells = [(r["edit_type"], r["video_name"], c) for r in table for c in NUMERIC
                 if c not in a.allow_nan and not math.isfinite(float(r[c]))]  # type: ignore[arg-type]
    allowed_nan = {c: sum(1 for r in table if not math.isfinite(float(r[c]))) for c in a.allow_nan}  # type: ignore[arg-type]

    out_pair = csv_dir / f"suite_{a.tag}_{a.method}.csv"
    with out_pair.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: fmt(v) for k, v in r.items()} for r in table])

    out_type = csv_dir / f"suite_{a.tag}_{a.method}_per_type.csv"
    type_fields = ["scope", "n"] + list(MEAN_COLS) + ["psc_removal"]
    lines: List[Dict[str, object]] = []
    for scope, sel in [(f"edit{t}", [r for r in table if r["edit_type"] == t]) for t in sc.EDIT_TYPES] + [("overall", table)]:
        if not sel:
            continue
        line: Dict[str, object] = {"scope": scope, "n": len(sel)}
        for c in MEAN_COLS:
            vals = [float(r[c]) for r in sel if not (c == "psc" and r["edit_type"] == 6)]  # type: ignore[arg-type]
            line[c] = mean(vals) if vals else ""
        rem = [float(r["psc"]) for r in sel if r["edit_type"] == 6]  # type: ignore[arg-type]
        line["psc_removal"] = mean(rem) if rem else ""
        lines.append(line)
    with out_type.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=type_fields)
        w.writeheader()
        w.writerows([{k: fmt(v) for k, v in ln.items()} for ln in lines])

    print(f"[merge_suite] {len(table)} pairs -> {out_pair}")
    print(f"[merge_suite] per-type means -> {out_type}")
    for ln in lines:
        if ln["scope"] == "overall":
            print("[merge_suite] overall: " + "  ".join(f"{c}={fmt(ln[c])}" for c in ("editreward", "psc", "psc_removal", "cf",
                  "clip_d_prompt", "five_acc_yes_no", "motion_fidelity_score") if ln[c] != ""))
    if any(allowed_nan.values()):
        print(f"[merge_suite] NaN cells in allowed columns (left out of the means): "
              f"{ {c: n for c, n in allowed_nan.items() if n} }")
    if nan_cells:
        print(f"[merge_suite] FAILED: {len(nan_cells)} NaN cell(s), e.g. {nan_cells[:5]}")
        return 1
    print("[merge_suite] merged: every metric finite" + (f" (except --allow_nan: {', '.join(a.allow_nan)})" if a.allow_nan else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
