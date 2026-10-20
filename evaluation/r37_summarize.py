"""R37 summary: dataset tables, per-category rankings + flips, and official-column parity.

Inputs: the per-sample long CSVs written by `bg_metrics.py` (one per arm; rows
`method, sample_id, file_id, video_name, category, alpha, psnr_w, ssim_w, lpips_w`,
alpha in {0.0..1.0, 'avg', 'official'}).

Outputs:
  --out_summary  method x category (all, 1..6) x column {official, 0.0, 0.1, 0.25, 0.5, avg}
                 x metric {psnr, ssim, lpips}: plain per-clip mean + n (never `_avg.csv`
                 means, which weigh edit5's 9 clips like edit1's 100).
  --out_rank     per matched-compute group (GROUPS), for each metric x category x column
                 {0.0, 0.1, 0.25, 0.5, avg, official}:
                 the arm ranking (best first), Kendall tau and every swapped pair vs
                 (a) alpha = 0 and (b) the official column.
  --out_parity   per arm: max |official here - official in the R35/R36 per-clip CSVs|,
                 matched on (edit type, file_id); pass iff <= 1e-4.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

METRICS = ["psnr", "ssim", "lpips"]
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False}
OFFICIAL_COL = {"psnr": "psnr_unedit_part", "ssim": "ssim_unedit_part", "lpips": "lpips_unedit_part"}
SUMMARY_COLUMNS = ["official", "0.0", "0.1", "0.25", "0.5", "avg"]
RANK_COLUMNS = ["0.0", "0.1", "0.25", "0.5", "avg", "official"]
REFERENCES = ["0.0", "official"]
# Rankings are only compared within matched-compute groups (decided 2026-10-07):
#   pass1: the original 12 arms (R35 routing N=15+15 next to the R36 N=15 rho curve)
#   n30:   R35 routing (15-step draft + 15-step render) vs the R39 N=30 rho curve
#   n20:   R38 routing (5-step draft + 15-step render) vs the R39 N=20 rho curve
GROUPS = {
    "pass1": ("r35_", "r36_"),
    "n30": ("r35_", "r39_n30_"),
    "n20": ("r38_", "r39_n20_"),
}
NON_SAMPLE = ("edgecases", "summary", "rankings", "parity")
PARITY_TOL = 1e-4


def load_per_sample(pattern: str) -> pd.DataFrame:
    # The glob r37_bg_*.csv also matches the edge-case logs and this script's outputs.
    paths = sorted(p for p in glob.glob(pattern)
                   if not any(tag in os.path.basename(p) for tag in NON_SAMPLE))
    if not paths:
        raise FileNotFoundError(f"no per-sample CSVs match {pattern}")
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"alpha": str})
        missing = {"method", "file_id", "category", "alpha", "psnr_w"} - set(df.columns)
        if missing:
            raise ValueError(f"{p}: missing columns {missing}")
        frames.append(df)
        print(f"[r37_sum] {os.path.basename(p)}: {len(df)} rows, "
              f"{df['sample_id'].nunique()} clips")
    df = pd.concat(frames, ignore_index=True)
    # Normalise alpha labels: '0.0', '0.05', ..., 'avg', 'official'.
    def norm(a: str) -> str:
        try:
            return repr(round(float(a), 10))
        except ValueError:
            return a
    df["alpha"] = df["alpha"].map(norm)
    return df.rename(columns={"psnr_w": "psnr", "ssim_w": "ssim", "lpips_w": "lpips"})


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["alpha"].isin(SUMMARY_COLUMNS)]
    rows = []
    for (method, col), g in sub.groupby(["method", "alpha"]):
        for cat, gc in [("all", g)] + [(str(c), x) for c, x in g.groupby("category")]:
            for m in METRICS:
                v = gc[m].to_numpy(dtype=float)
                finite = np.isfinite(v)
                rows.append({"method": method, "category": cat, "column": col, "metric": m,
                             "mean": float(v[finite].mean()) if finite.any() else np.nan,
                             "n": int(finite.sum()), "n_nonfinite": int((~finite).sum())})
    out = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(SUMMARY_COLUMNS)}
    cats = {c: i for i, c in enumerate(["all", "1", "2", "3", "4", "5", "6"])}
    return out.sort_values(["metric", "category", "column", "method"],
                           key=lambda s: s.map(order) if s.name == "column"
                           else s.map(cats) if s.name == "category" else s).reset_index(drop=True)


def _order(scores: pd.Series, higher: bool) -> list[str]:
    return list(scores.sort_values(ascending=not higher).index)


def rankings(summary: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, prefixes in GROUPS.items():
        sub = summary[summary["method"].str.startswith(prefixes)]
        if sub["method"].nunique() < 2:
            print(f"[r37_sum] group {name}: fewer than 2 arms present -- skipped")
            continue
        r = _rankings_one(sub)
        r.insert(0, "group", name)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def _rankings_one(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, cat), g in summary.groupby(["metric", "category"]):
        piv = g.pivot(index="method", columns="column", values="mean")
        higher = HIGHER_IS_BETTER[m]
        for col in RANK_COLUMNS:
            if col not in piv:
                continue
            order = _order(piv[col], higher)
            for ref in REFERENCES:
                if ref == col or ref not in piv:
                    continue
                tau = kendalltau(piv[col], piv[ref]).statistic
                flips = []
                for a, b in itertools.combinations(piv.index, 2):
                    d_col = piv.at[a, col] - piv.at[b, col]
                    d_ref = piv.at[a, ref] - piv.at[b, ref]
                    if d_col * d_ref < 0:  # strict reversal of the pair's order
                        better_ref = a if (d_ref > 0) == higher else b
                        worse_ref = b if better_ref == a else a
                        flips.append(f"{better_ref}>{worse_ref}→{worse_ref}>{better_ref}")
                rows.append({"metric": m, "category": cat, "column": col, "reference": ref,
                             "kendall_tau": tau, "n_flips": len(flips),
                             "ranking_best_first": ";".join(order),
                             "flips": "; ".join(flips)})
    return pd.DataFrame(rows)


def parity(df: pd.DataFrame, official_glob: str) -> pd.DataFrame:
    off = df[df["alpha"] == "official"]
    rows = []
    for method, g in off.groupby("method"):
        deltas = {m: [] for m in METRICS}
        n_match = n_missing = n_nan_mismatch = 0
        for T, gt in g.groupby("category"):
            path = official_glob.format(T=T, stem=method)
            if not os.path.exists(path):
                n_missing += len(gt)
                continue
            ref = pd.read_csv(path)
            cols = {m: next(c for c in ref.columns if c.endswith("|" + OFFICIAL_COL[m]))
                    for m in METRICS}
            ref = ref.set_index("file_id")
            for _, r in gt.iterrows():
                if r["file_id"] not in ref.index:
                    n_missing += 1
                    continue
                n_match += 1
                for m in METRICS:
                    a, b = float(r[m]), float(ref.at[r["file_id"], cols[m]])
                    if np.isnan(a) and np.isnan(b):
                        deltas[m].append(0.0)
                    elif np.isnan(a) or np.isnan(b):
                        n_nan_mismatch += 1
                    else:
                        deltas[m].append(abs(a - b))
        row = {"method": method, "n_matched": n_match, "n_missing": n_missing,
               "n_nan_mismatch": n_nan_mismatch}
        for m in METRICS:
            row[f"max_abs_delta_{m}"] = max(deltas[m]) if deltas[m] else np.nan
        row["pass"] = bool(n_missing == 0 and n_nan_mismatch == 0 and
                           all(row[f"max_abs_delta_{m}"] <= PARITY_TOL for m in METRICS))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R37 summary / rankings / parity.")
    p.add_argument("--per_sample_glob", required=True)
    p.add_argument("--official_glob", required=True,
                   help="path template with {T} and {stem}, e.g. "
                        "evaluation/csv/edit{T}_FiVE_{stem}_frame_stride8.csv")
    p.add_argument("--out_summary", required=True)
    p.add_argument("--out_rank", required=True)
    p.add_argument("--out_parity", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    df = load_per_sample(args.per_sample_glob)
    n_arms, n_clips = df["method"].nunique(), df.groupby("method")["sample_id"].nunique()
    print(f"[r37_sum] {n_arms} arms; clips per arm: {sorted(set(n_clips))}")
    short = n_clips[n_clips < n_clips.max()]
    if len(short):  # e.g. an arm whose CSV is still being written by the launcher
        raise ValueError(f"incomplete arms (clips < {n_clips.max()}): {short.to_dict()}")

    summary = summarize(df)
    summary.to_csv(args.out_summary, index=False)
    rank = rankings(summary)
    rank.to_csv(args.out_rank, index=False)
    par = parity(df, args.official_glob)
    par.to_csv(args.out_parity, index=False)

    print(f"[r37_sum] summary -> {args.out_summary} ({len(summary)} rows)")
    print(f"[r37_sum] rankings -> {args.out_rank} ({len(rank)} rows, "
          f"{int((rank['n_flips'] > 0).sum())} with >=1 flip)")
    print(f"[r37_sum] parity -> {args.out_parity}: "
          f"{int(par['pass'].sum())}/{len(par)} arms pass (tol {PARITY_TOL})")
    if not par["pass"].all():
        print(par[~par["pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
