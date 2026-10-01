#!/usr/bin/env python3
"""R22 step/frame metric-convergence matrix.

Folds the 45 per-frame CSVs written by ``evaluation/fivebench/evaluate.py``
(3 arms x 15 denoising steps, ``--per_frame --frame_stride 1``) into the long
matrix

    M(i, j) = metric evaluated on the first ``i`` pixel frames of the video
              as it stood at denoising step ``j``

for every (arm, pair, metric).

Why a cumulative mean is exact, not an approximation
----------------------------------------------------
``evaluate.py`` reduces each of the 8 metrics carried here with
``calculate_mean`` over its per-frame list (evaluate.py:484), so the value it
would report for the truncated video ``V_j[:i]`` is precisely the mean of the
first ``i`` per-frame values it already wrote. That is what turns an
``nb_frames x 15`` cost into a ``15`` cost. It holds ONLY for per-frame-mean
metrics -- the 8 in the R22 set. Do not extend the metric set without
re-deriving it (``motion_fidelity_*`` and ``five_acc`` are whole-clip and are
emitted by ``--per_frame`` with ``frame_idx=-1``).

Keying
------
Pairs are keyed by ``(arm, editing_type_id, file_id, video_name)``, NOT by
``video_name`` alone: ``0011_lucia`` appears under two edit types in
``cases.json``, so a name-only key would silently merge two different pairs
into one interleaved prefix sequence.

Note on ``clip_similarity_source_image``: it is computed on the SOURCE video
(evaluate.py:107), so it is constant along ``j`` by construction -- flat rows
in the output are correct, not a defect. It still varies along ``i``.

Outputs
-------
``--out``       long CSV: one row per (arm, pair, metric, j, i) with M(i,j).
``--check_out`` the L3 identity check: M(nb_frames, 14) against the plain mean
                of the step-14 per-frame column, per (arm, pair, metric).
                Exits non-zero if any row exceeds ``--tol``.

The L3 check is an internal regression assert. It must NOT be pointed at
``r21_*_avg.csv`` / ``r20_*_avg.csv``: those ran at ``--frame_stride 8`` and
their overall row is a mean of per-edit-type means, so they legitimately
disagree. The cross-arm check lives at L2 (bit-identical step-14 PNGs).
"""
from __future__ import annotations

import argparse
import glob
import re
import sys

import numpy as np
import pandas as pd

# method column is written by the eval script as ``r22_{arm}_s{jj}``
METHOD_RE = re.compile(r"^r22_(?P<arm>.+)_s(?P<j>\d{2})$")

PAIR_KEY = ["arm", "editing_type_id", "file_id", "video_name"]
CELL_KEY = PAIR_KEY + ["metric"]

N_STEPS = 15


def load_per_frame(pattern: str) -> pd.DataFrame:
    """Read every per-frame CSV matching ``pattern`` into one long frame."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"[r22_build_matrix] no files match {pattern!r}")

    frames = []
    for p in paths:
        d = pd.read_csv(p)
        missing = {"file_id", "video_name", "editing_type_id", "method",
                   "frame_idx", "metric", "value"} - set(d.columns)
        if missing:
            raise SystemExit(f"[r22_build_matrix] {p}: missing columns {sorted(missing)}")
        # one file == one (arm, step); take the arm/j from the method column
        methods = d["method"].unique()
        if len(methods) != 1:
            raise SystemExit(f"[r22_build_matrix] {p}: expected 1 method, got {list(methods)}")
        m = METHOD_RE.match(str(methods[0]))
        if m is None:
            raise SystemExit(
                f"[r22_build_matrix] {p}: method {methods[0]!r} does not match r22_<arm>_s<jj>")
        d["arm"] = m.group("arm")
        d["j"] = int(m.group("j"))
        frames.append(d)

    A = pd.concat(frames, ignore_index=True)

    # whole-clip metrics are emitted with frame_idx=-1; none are requested for
    # R22, but drop them defensively so they cannot enter a prefix mean.
    n_whole = int((A["frame_idx"] < 0).sum())
    if n_whole:
        print(f"[r22_build_matrix] dropping {n_whole} whole-clip rows (frame_idx<0)")
        A = A[A["frame_idx"] >= 0]

    A["value"] = pd.to_numeric(A["value"], errors="coerce")
    n_nan = int(A["value"].isna().sum())
    if n_nan:
        print(f"[r22_build_matrix] WARNING: {n_nan} non-numeric/nan values "
              f"-- they are skipped by the prefix mean, so affected cells "
              f"average over fewer frames than their prefix length")

    print(f"[r22_build_matrix] read {len(paths)} files, {len(A)} rows, "
          f"arms={sorted(A['arm'].unique())}, steps={A['j'].nunique()}")
    return A


def validate(A: pd.DataFrame) -> None:
    """Fail loudly on holes -- a missing row silently shifts every later prefix mean."""
    problems: list[str] = []

    # every cell must carry all 15 steps
    steps = A.groupby(CELL_KEY, sort=False)["j"].nunique()
    bad = steps[steps != N_STEPS]
    if len(bad):
        problems.append(f"{len(bad)} (arm,pair,metric) cells do not have {N_STEPS} steps "
                        f"(e.g. {bad.index[0]} -> {bad.iloc[0]})")

    # frame_idx must be contiguous 0..n-1 within each (cell, step): that is what
    # --frame_stride 1 buys, and it is what makes i = frame_idx + 1 valid
    g = A.groupby(CELL_KEY + ["j"], sort=False)["frame_idx"]
    stats = g.agg(["min", "max", "nunique", "size"])
    ok = (stats["min"] == 0) & (stats["max"] == stats["nunique"] - 1) & \
         (stats["nunique"] == stats["size"])
    if not ok.all():
        problems.append(f"{int((~ok).sum())} (cell,step) groups have non-contiguous "
                        f"frame_idx (e.g. {stats[~ok].index[0]})")

    # a pair's frame count must not depend on the step
    per_step = A.groupby(CELL_KEY + ["j"], sort=False).size().rename("n")
    spread = per_step.groupby(level=list(range(len(CELL_KEY)))).nunique()
    if (spread != 1).any():
        problems.append(f"{int((spread != 1).sum())} cells change frame count across steps")

    if problems:
        for p in problems:
            print(f"[r22_build_matrix] ERROR: {p}", file=sys.stderr)
        raise SystemExit(1)

    n_pairs = A.groupby(PAIR_KEY, sort=False).ngroups
    print(f"[r22_build_matrix] validated: {n_pairs} (arm,pair) groups x "
          f"{A['metric'].nunique()} metrics x {N_STEPS} steps, frame_idx contiguous")


def build_matrix(A: pd.DataFrame) -> pd.DataFrame:
    """Cumulative prefix mean over frame_idx within each (arm, pair, metric, step)."""
    A = A.sort_values(CELL_KEY + ["j", "frame_idx"], kind="mergesort").reset_index(drop=True)

    grp = A.groupby(CELL_KEY + ["j"], sort=False)
    # nan-skipping cumulative mean: running sum of present values over their count
    csum = grp["value"].cumsum()                      # pandas cumsum skips NaN
    cnt = grp["value"].transform(lambda s: s.notna().cumsum()) \
        if A["value"].isna().any() else pd.Series(grp.cumcount() + 1, index=A.index)

    out = pd.DataFrame({
        "arm": A["arm"],
        "editing_type_id": A["editing_type_id"],
        "file_id": A["file_id"],
        "video_name": A["video_name"],
        "metric": A["metric"],
        "j": A["j"],
        "i": A["frame_idx"] + 1,                      # prefix length, 1..nb_frames
        "nb_frames": grp["value"].transform("size"),
        "value": csum / cnt.replace(0, np.nan),
    })
    print(f"[r22_build_matrix] built M(i,j): {len(out)} rows")
    return out


def bottom_right_check(M: pd.DataFrame, A: pd.DataFrame, tol: float) -> pd.DataFrame:
    """L3: M(nb_frames, 14) must equal the plain mean of the step-14 per-frame column."""
    last = M["j"].max()
    br = M[(M["j"] == last) & (M["i"] == M["nb_frames"])].set_index(CELL_KEY)["value"]
    plain = A[A["j"] == last].groupby(CELL_KEY, sort=False)["value"].mean()

    chk = pd.DataFrame({"m_bottom_right": br}).join(
        plain.rename("plain_mean_step14"), how="outer").reset_index()
    chk["abs_diff"] = (chk["m_bottom_right"] - chk["plain_mean_step14"]).abs()
    chk["ok"] = chk["abs_diff"] <= tol
    return chk.sort_values("abs_diff", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per_frame_glob",
                    default="evaluation/csv/r22_raw/*_frame_stride1_per_frame.csv",
                    help="glob for the per-frame CSVs written by evaluate.py")
    ap.add_argument("-o", "--out", default="evaluation/csv/r22_matrix.csv",
                    help="long-format M(i,j) output")
    ap.add_argument("--check_out", default="evaluation/csv/r22_bottom_right_check.csv",
                    help="L3 bottom-right identity check output")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="absolute tolerance for the L3 identity (default 1e-9)")
    args = ap.parse_args()

    A = load_per_frame(args.per_frame_glob)
    validate(A)
    M = build_matrix(A)

    M.to_csv(args.out, index=False)
    print(f"[r22_build_matrix] wrote {args.out}")

    chk = bottom_right_check(M, A, args.tol)
    chk.to_csv(args.check_out, index=False)
    n_bad = int((~chk["ok"]).sum())
    print(f"[r22_build_matrix] wrote {args.check_out} "
          f"({len(chk)} cells, max |diff| = {chk['abs_diff'].max():.3g})")

    if n_bad:
        print(f"[r22_build_matrix] L3 IDENTITY FAIL: {n_bad}/{len(chk)} cells "
              f"exceed tol={args.tol:g}", file=sys.stderr)
        print(chk.head(10).to_string(index=False), file=sys.stderr)
        raise SystemExit(1)
    print(f"[r22_build_matrix] L3 identity PASS on all {len(chk)} cells "
          f"(tol={args.tol:g})")


if __name__ == "__main__":
    main()
