"""R38 divergence stats: does a 5-step draft shift the raw divergence that R35's fixed
``div_max`` was calibrated on?

R38 keeps R35's ABSOLUTE normalisation for every N -- ``m = clip(d, 0, div_max) / div_max``
with ``div_max`` 0.65 (DINO) / 1.0 (LPIPS), set on 15-step drafts. A 5-step draft's final
x0 is predicted from t=0.2 (vs 0.067), so its raw ``d`` may move: up => more tokens
saturate at m=1 (max rho in stage 3), down => m shrinks everywhere. Either changes the
routing for a reason unrelated to WHERE the map points. This table makes the shift visible
next to the N=15 counterpart of each combo; it gates nothing.

Per combo (R38's N=5 four + R35's N=15 four), pooled over every token of every clip
(``d`` is [n_latent_frames, 1560] per clip, so longer clips weigh more -- the same weighting
the stage-3 render sees): mean, p50, p90, p99 of ``d``, and the fraction of tokens with
``d >= div_max``. ``metric`` + ``regime`` pair each N=5 row with its N=15 row (R35's
``first2`` is R38's ``tgt0.9`` at 15 steps -- smoke gate G1).

Writes ``evaluation/csv/r38_div_stats.csv``. Numpy only; hard-fails on any combo that is
not exactly 100/100/100/100/9/10 rather than summarising fewer clips. Combos absent on disk
can be skipped explicitly with ``--dropped`` (the gate-lowN subset).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIVE = Path("~/Data/dataggen/outputs/five_bench").expanduser()
EXPECT = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
DIV_MAX = {"dino": 0.65, "lpips": 1.0}   # R35 Decisions, kept for every N (R38 Decisions)

# combo dir, N, metric, regime ("anchored" = R35 first2 at 15 / R38 tgt0.9 at 5)
COMBOS = [
    (FIVE / "r35_div" / "dino_unblended",      15, "dino",  "unblended"),
    (FIVE / "r35_div" / "dino_first2",         15, "dino",  "anchored"),
    (FIVE / "r35_div" / "lpips_unblended",     15, "lpips", "unblended"),
    (FIVE / "r35_div" / "lpips_first2",        15, "lpips", "anchored"),
    (FIVE / "r38_div" / "dino_n5_unblended",    5, "dino",  "unblended"),
    (FIVE / "r38_div" / "dino_n5_tgt09",        5, "dino",  "anchored"),
    (FIVE / "r38_div" / "lpips_n5_unblended",   5, "lpips", "unblended"),
    (FIVE / "r38_div" / "lpips_n5_tgt09",       5, "lpips", "anchored"),
]


def pooled_d(combo_dir: Path) -> np.ndarray:
    """All tokens' raw d for one combo, after asserting the exact per-type breakdown."""
    chunks: List[np.ndarray] = []
    for T, n_exp in EXPECT.items():
        files = sorted((combo_dir / f"edit{T}").glob("*.npz"))
        if len(files) != n_exp:
            raise SystemExit(f"[r38_div_stats] {combo_dir}/edit{T}: {len(files)} npz, expect {n_exp}")
        for f in files:
            chunks.append(np.load(f)["d"].astype(np.float64).ravel())
    return np.concatenate(chunks)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dropped", nargs="*", default=[],
                   help="R38 combo names dropped at gate-lowN (e.g. dino_n5_tgt09); skipped")
    p.add_argument("-o", "--out", type=Path, default=REPO / "evaluation" / "csv" / "r38_div_stats.csv")
    args = p.parse_args(argv)

    unknown = set(args.dropped) - {c[0].name for c in COMBOS if c[1] == 5}
    if unknown:
        raise SystemExit(f"[r38_div_stats] --dropped names not R38 combos: {sorted(unknown)}")

    rows: List[Dict[str, object]] = []
    for combo_dir, n, metric, regime in COMBOS:
        if combo_dir.name in args.dropped:
            print(f"[r38_div_stats] skip {combo_dir.name} (dropped at gate-lowN)", flush=True)
            continue
        d = pooled_d(combo_dir)
        dmax = DIV_MAX[metric]
        p50, p90, p99 = np.percentile(d, [50, 90, 99])
        row = dict(combo=combo_dir.name, N=n, metric=metric, regime=regime,
                   div_max=dmax, n_tokens=int(d.size),
                   d_mean=round(float(d.mean()), 5), d_p50=round(float(p50), 5),
                   d_p90=round(float(p90), 5), d_p99=round(float(p99), 5),
                   sat_frac=round(float(np.mean(d >= dmax)), 5))
        rows.append(row)
        print(f"[r38_div_stats] {row}", flush=True)

    # N=5 vs N=15, per (metric, regime): the shift this table exists to show.
    by = {(r["metric"], r["regime"], r["N"]): r for r in rows}
    for metric in DIV_MAX:
        for regime in ("unblended", "anchored"):
            a, b = by.get((metric, regime, 5)), by.get((metric, regime, 15))
            if a and b:
                print(f"[r38_div_stats] {metric}/{regime}: d_mean N5/N15 = "
                      f"{a['d_mean'] / b['d_mean']:.3f}, sat_frac {b['sat_frac']:.4f} -> {a['sat_frac']:.4f}",
                      flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[r38_div_stats] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
