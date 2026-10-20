"""R45 divergence stats: does the LongLive draft shift the raw divergence that the fixed
``div_max`` (0.65 DINO / 1.0 LPIPS, set on Self-Forcing drafts) was calibrated on?

R45 keeps SF's absolute normalisation on both backbones (same d -> m function), so a LongLive
draft whose raw ``d`` sits higher saturates more tokens at m = 1, and one that sits lower
compresses m everywhere -- either changes the routing for a reason unrelated to WHERE the
map points. This table puts every SELECTED LL combo (``evaluation/r45_method.json``) next to
its SF counterpart at the same draft N, metric and regime:

    N=15: r35_div/{dino,lpips}_{unblended,first2}      (R35's first2 == tau = 0.9 at N=15)
    N=5 : r38_div/{dino,lpips}_n5_{unblended,tgt09}

Per combo, pooled over every token of every clip (as r38_div_stats.py): mean, p50, p90, p99 of
``d`` and the fraction of tokens with ``d >= div_max``. Prints LL/SF ratios and FLAGs any LL
combo whose saturated fraction is more than 2x its SF counterpart's (plan Decisions: raise it
at ``divstats`` before stage 3). Gates nothing by itself.

Writes ``evaluation/csv/r45_div_stats.csv``. Numpy only; the per-clip loader and the exact
100/100/100/100/9/10 check are r38_div_stats.py's.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r38_div_stats import DIV_MAX, FIVE, pooled_d   # noqa: E402
import r45_select as sel                             # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SAT_FLAG_RATIO = 2.0


def sf_dir(metric: str, regime: str, n: int) -> Path:
    """The SF combo directory matching an LL combo's (metric, regime, draft N)."""
    if n == 15:
        return FIVE / "r35_div" / f"{metric}_{'first2' if regime == 'tgt09' else 'unblended'}"
    return FIVE / "r38_div" / f"{metric}_n5_{regime}"


def stats(d: np.ndarray, dmax: float) -> Dict[str, object]:
    p50, p90, p99 = np.percentile(d, [50, 90, 99])
    return dict(n_tokens=int(d.size), d_mean=round(float(d.mean()), 5),
                d_p50=round(float(p50), 5), d_p90=round(float(p90), 5),
                d_p99=round(float(p99), 5), sat_frac=round(float(np.mean(d >= dmax)), 5))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("-o", "--out", type=Path, default=REPO / "evaluation" / "csv" / "r45_div_stats.csv")
    args = p.parse_args(argv)

    m = sel.load()
    rows: List[Dict[str, object]] = []
    flags = 0
    for combo in m["combos"]:                        # e.g. ll_dino_n5_tgt09
        _, metric, ntag, regime = combo.split("_", 3)
        n = int(ntag[1:])
        dmax = DIV_MAX[metric]
        pair = {}
        for backbone, cdir in (("sf", sf_dir(metric, regime, n)),
                               ("ll", FIVE / "r45_div" / combo)):
            row = dict(backbone=backbone, combo=cdir.name, N=n, metric=metric, regime=regime,
                       div_max=dmax, **stats(pooled_d(cdir), dmax))
            rows.append(row)
            pair[backbone] = row
            print(f"[r45_div_stats] {row}", flush=True)
        sf, ll = pair["sf"], pair["ll"]
        ratio = ll["sat_frac"] / sf["sat_frac"] if sf["sat_frac"] > 0 else float("inf")
        flag = ratio > SAT_FLAG_RATIO
        flags += flag
        print(f"[r45_div_stats] {combo}: d_mean LL/SF = {ll['d_mean'] / sf['d_mean']:.3f}, "
              f"sat_frac SF {sf['sat_frac']:.4f} -> LL {ll['sat_frac']:.4f} (x{ratio:.2f})"
              f"{'   FLAG: > 2x SF saturation -- raise before stage 3' if flag else ''}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[r45_div_stats] {flags} saturation flag(s) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
