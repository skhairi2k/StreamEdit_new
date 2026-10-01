#!/usr/bin/env python
"""R33 -- pass/fail test for every candidate edit-alignment axis, on the same footing.

WHY THIS EXISTS. FiVE-Bench's clip_similarity_target_image does not rank R26's own
constant-b SPATIAL sweep by edit strength: Spearman(b, clip_target) = +0.098 mean,
positive on only 11/22 clips -- a coin flip. This script measures the same statistic
for every candidate replacement axis, over the identical 8-b SPATIAL sweep
(taubg0_taufg{b}_vp, b in CONST_BS), so the incumbent and its replacements are read off
the same footing rather than compared across different case sets or b-grids.

AXES. Two already stored in R26's own per-edit-type CSVs (the incumbent
clip_similarity_target_image, and its edit-region variant); two from R33's CLIP-D pass
(clip_d_prompt, clip_d_word, evaluation/csv/r33_clip_directional.csv); two from R33's
FiVE-Acc pass (yn_acc, mc_acc -- evaluate.py's own column names are
five_acc_yes_no / five_acc_multi_choice, renamed here to match the plan's axis labels).

0040_TENNIS. Its clip_similarity_* columns are corrupted by a torchmetrics 1.9.0 bug
(clip_score.py:145-146 raw-slices input_ids[:77] on an 85/84-token prompt pair, dropping
the EOT token CLIP pools its text embedding at -- src_prompt and trg_prompt score a
bit-identical 5.528 there). Excluded from the two clip_similarity_* axes only. CLIP-D
tokenises with truncation=True and FiVE-Acc is a VLM asking short questions -- both
immune -- so 0040_tennis is KEPT for clip_d_prompt, clip_d_word, yn_acc, mc_acc.

JOIN. Reuses r30_score.build_join_tables for the (edit_type, file_id) -> video_name ->
case_id join R26's per-edit-type CSVs need (they carry no case_id of their own), and
r30_score.load_r26_metric for the two CLIP columns already in those files. FiVE-Acc's
per-edit-type CSVs (edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv) are read with a
small local analogue of load_r26_metric, since their filename prefix (r33_fiveacc_) and
metric column names (five_acc_yes_no / five_acc_multi_choice) differ from R26's own
files. CLIP-D's case_id is already a column in r33_clip_directional.csv -- no join
needed there.

Usage
-----
    python evaluation/r33_axis_compare.py \\
        --r26_csv_dir evaluation/csv \\
        --clipd_csv evaluation/csv/r33_clip_directional.csv \\
        -o evaluation/csv/r33_axis_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scipy.stats import spearmanr

from r30_score import CONST_BS, build_join_tables, load_r26_metric

TENNIS_CASE_ID = "0040_tennis"

# (axis label, column-loading strategy). "r26" axes read straight out of R26's own
# per-edit-type CSVs; "clipd" axes read out of r33_clip_directional.csv; "fiveacc" axes
# read out of R33's own per-edit-type CSVs. tennis_immune marks whether 0040_tennis's
# known EOT-truncation corruption applies to this axis.
AXES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("clip_similarity_target_image",            "r26",     "clip_similarity_target_image",            False),
    ("clip_similarity_target_image_edit_part",  "r26",     "clip_similarity_target_image_edit_part",  False),
    ("clip_d_prompt",                            "clipd",   "clip_d_prompt",                            True),
    ("clip_d_word",                              "clipd",   "clip_d_word",                              True),
    ("yn_acc",                                   "fiveacc", "five_acc_yes_no",                          True),
    ("mc_acc",                                   "fiveacc", "five_acc_multi_choice",                    True),
    # five_acc added 2026-09-22: evaluate.py's own combined verdict,
    # (yn_acc+mc_acc+union+inter)/4 -- already a raw column in every FiVE-Acc per-clip
    # CSV (evaluate.py writes it directly), read here the same generic suffix-matched
    # way as yn_acc/mc_acc, no new loader needed.
    ("five_acc",                                 "fiveacc", "five_acc",                                 True),
)


def load_fiveacc_metric(csv_dir: Path, idx2vid: Dict[Tuple[int, str], str],
                        name2case: Dict[Tuple[int, str], str],
                        bs: Sequence[int], metric: str,
                        method_fmt: str = "taubg0_taufg{b}_vp"
                        ) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): <metric>} from R33's own FiVE-Acc per-edit-type CSVs.

    Mirrors r30_score.load_r26_metric's join logic exactly, but against
    edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv (R33's own stem prefix, not
    R26's) and evaluate.py's raw five_acc column names.
    """
    out: Dict[Tuple[str, int], float] = {}
    for b in bs:
        method = method_fmt.format(b=b)
        pattern = str(csv_dir / f"edit*_FiVE_r33_fiveacc_{method}_frame_stride8.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"[r33_axis_compare] no FiVE-Acc per-clip CSV matched "
                             f"{pattern}. Has r33_fiveacc.sh finished for b={b}?")
        for f in files:
            m = re.search(r"edit(\d+)_", os.path.basename(f))
            t = int(m.group(1))
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    col = next((k for k in r if k.endswith(f"|{metric}")), None)
                    if col is None:
                        raise SystemExit(f"[r33_axis_compare] {f} has no *|{metric} column.")
                    vid = idx2vid.get((t, r["file_id"]))
                    if vid is None:
                        continue                  # scored row outside the 22 cases
                    case_id = name2case.get((t, vid))
                    if case_id is None:
                        continue
                    out[(case_id, b)] = float(r[col])
    return out


def load_clipd_metric(clipd_csv: Path, metric: str,
                      method_fmt: str = "r26_taubg0_taufg{b}_vp"
                      ) -> Dict[Tuple[str, int], float]:
    """{(case_id, b): <metric>} from r33_clip_directional.csv.

    case_id is already a column here (no per-edit-type join needed) -- CLIP-D was
    computed directly against the same 22-case registry the other axes join through.
    """
    wanted = {method_fmt.format(b=b): b for b in CONST_BS}
    out: Dict[Tuple[str, int], float] = {}
    with open(clipd_csv) as fh:
        for r in csv.DictReader(fh):
            b = wanted.get(r["method"])
            if b is None:
                continue
            out[(r["case_id"], b)] = float(r[metric])
    return out


def per_clip_spearman(clips: Sequence[str], values: Dict[Tuple[str, int], float],
                      bs: Sequence[int]) -> Tuple[Dict[str, float], List[str]]:
    """({case_id: rho}, [case_ids with a FLAT axis over bs]) for Spearman(b, axis).

    A clip missing any of the bs (should not happen once all arms are scored, but
    checked rather than assumed) is skipped with a loud warning instead of silently
    dropped -- a silent drop could quietly shrink the denominator without anyone
    noticing the axis is short a clip relative to the others.

    A clip whose axis value is IDENTICAL across every b (expected for the coarse
    accuracy axes -- yn_acc/mc_acc are 0/1 per clip, and a clip that answers the same
    way regardless of edit strength is not a measurement error) has an undefined
    Spearman rho (scipy returns nan with a ConstantInputWarning). Returned separately
    rather than folded into `out` as nan, so one flat clip cannot silently nan out the
    whole axis's mean.
    """
    out: Dict[str, float] = {}
    flat: List[str] = []
    for c in clips:
        row = [values.get((c, b)) for b in bs]
        if any(v is None for v in row):
            print(f"[r33_axis_compare] WARNING: {c} missing {sum(v is None for v in row)}/"
                  f"{len(bs)} b-values -- excluded from this axis's mean.")
            continue
        if len(set(row)) == 1:
            flat.append(c)
            continue
        rho, _ = spearmanr(bs, row)
        out[c] = rho
    return out, flat


def run(args: argparse.Namespace) -> int:
    cases = __import__("json").loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})
    if len(clips) != 22:
        raise SystemExit(f"[r33_axis_compare] expected 22 cases, got {len(clips)}.")

    r26_dir = args.r26_csv_dir
    rows: List[Dict[str, object]] = []
    for label, kind, col, tennis_immune in AXES:
        if kind == "r26":
            values = load_r26_metric(r26_dir, idx2vid, name2case, CONST_BS, col)
        elif kind == "clipd":
            values = load_clipd_metric(args.clipd_csv, col)
        elif kind == "fiveacc":
            values = load_fiveacc_metric(r26_dir, idx2vid, name2case, CONST_BS, col)
        else:
            raise AssertionError(kind)

        per_clip, flat = per_clip_spearman(clips, values, CONST_BS)
        tennis_seen = TENNIS_CASE_ID in per_clip or TENNIS_CASE_ID in flat
        eval_clips = list(per_clip) if tennis_immune else \
            [c for c in per_clip if c != TENNIS_CASE_ID]
        rhos = [per_clip[c] for c in eval_clips]
        mean_rho = sum(rhos) / len(rhos) if rhos else float("nan")
        n_pos = sum(1 for r in rhos if r > 0)
        n_flat = len([c for c in flat if tennis_immune or c != TENNIS_CASE_ID])

        rows.append({
            "axis": label,
            "n_clips": len(eval_clips),
            "mean_spearman": round(mean_rho, 4),
            "n_positive": n_pos,
            "n_flat": n_flat,
            "tennis_excluded": (not tennis_immune) and tennis_seen,
        })
        print(f"[r33_axis_compare] {label:<42} mean_rho={mean_rho:+.4f}  "
              f"n_pos={n_pos}/{len(eval_clips)}  n_flat={n_flat}  "
              f"tennis_excluded={(not tennis_immune) and tennis_seen}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["axis", "n_clips", "mean_spearman",
                                           "n_positive", "n_flat", "tennis_excluded"])
        w.writeheader()
        w.writerows(rows)
    print(f"[r33_axis_compare] wrote {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--clipd_csv", type=Path,
                   default=Path("evaluation/csv/r33_clip_directional.csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("evaluation/csv/r33_axis_comparison.csv"))
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
