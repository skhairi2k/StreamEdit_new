#!/usr/bin/env python
"""R30 metric redundancy: are the 8 divergences measuring the same thing?

The free, no-fitting precondition for asking whether a COMBINATION of divergences could
beat the best single one. A monotone remap of one metric can never change how it ranks
clips, so no d->b regression can lift a single arm's rank correlation with the human
ratings (confirmed empirically in r30_retest.py: d and routed b give identical rho). A
function of TWO metrics is not bounded that way -- it can reorder clips -- but only if
the two carry INDEPENDENT information. If every arm ranks clips nearly identically, a
combination has nothing extra to draw on and the whole regression exercise is moot.

This script answers that with no model, no fitting, and no overfitting risk: the pairwise
Spearman matrix among the arms themselves, plus each arm's own agreement with the human
rating, plus every pair ranked by how COMPLEMENTARY it looks (low mutual correlation,
both members individually informative). A pair that is individually strong and mutually
weak is the only kind worth spending a regression on -- and at n=21 it should be
pre-specified on those grounds rather than searched over, since 28 pairs against 21
points will manufacture a winner by chance.

Human agreement is taken against the MEAN of the two rating passes (r30_retest.py),
which is the more stable target: the rater's rank agreement with themselves is 0.942 but
their absolute picks drifted upward slightly on pass 2.

Usage
-----
    python evaluation/r30_metric_redundancy.py \\
        --pass1 evaluation/csv/r30_human_b_template.csv \\
        --pass2 evaluation/csv/r30_human_b_pass2.csv \\
        --key <scratchpad>/r30_blind_key.csv \\
        -o evaluation/csv/r30_metric_redundancy.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import ARMS  # noqa: E402
from r30_rank_check import load_divergence_by_arm, spearman  # noqa: E402
from r30_retest import load_key, load_pass1, load_pass2  # noqa: E402


def run(args: argparse.Namespace) -> int:
    div = load_divergence_by_arm(args.divergence_csv, ARMS)

    # Metric-vs-metric redundancy is a property of the metrics alone, so it uses every
    # clip they were measured on -- the human ratings are not involved in it at all.
    all_clips = sorted(set.intersection(*(set(div[a]) for a in ARMS)))
    print(f"=== Pairwise Spearman AMONG the {len(ARMS)} divergences "
          f"(n={len(all_clips)} clips) ===")
    header = " " * 13 + "".join(f"{a[:10]:>11}" for a in ARMS)
    print(header)
    matrix: Dict[str, Dict[str, float]] = {}
    for a in ARMS:
        matrix[a] = {}
        line = f"{a:<13}"
        for b in ARMS:
            r = 1.0 if a == b else spearman([div[a][c] for c in all_clips],
                                            [div[b][c] for c in all_clips])
            matrix[a][b] = r
            line += f"{r:>11.3f}"
        print(line)

    # Human agreement per arm, on the clips the rater actually scored.
    human: Dict[str, float] = {}
    if args.key is not None and args.pass2 is not None:
        key = load_key(args.key)
        p1 = load_pass1(args.pass1)
        p2 = load_pass2(args.pass2, key)
        rated = sorted(set(p1) & set(p2))
        mean_rating = {c: (p1[c] + p2[c]) / 2.0 for c in rated}
        for a in ARMS:
            cs = [c for c in rated if c in div[a]]
            human[a] = spearman([mean_rating[c] for c in cs], [div[a][c] for c in cs])
        print(f"\n=== Each arm vs the human mean rating (n={len(rated)}), "
              f"ceiling 0.942 ===")
        for a in sorted(ARMS, key=lambda k: -human[k]):
            print(f"  {a:<13}{human[a]:>7.3f}")

    # The only pairs worth a regression: individually informative, mutually weak.
    print(f"\n=== Pairs ranked by COMPLEMENTARITY (low mutual rho, both informative) ===")
    print(f"{'pair':<26}{'mutual':>9}{'human A':>9}{'human B':>9}{'min human':>11}")
    rows: List[Dict[str, object]] = []
    for a, b in combinations(ARMS, 2):
        mutual = matrix[a][b]
        ha, hb = human.get(a, float("nan")), human.get(b, float("nan"))
        rows.append({"arm_a": a, "arm_b": b, "mutual_rho": round(mutual, 4),
                     "human_rho_a": round(ha, 4), "human_rho_b": round(hb, 4),
                     "min_human_rho": round(min(ha, hb), 4)})
    # Sort so the most promising (weak mutual, strong weakest-member) float to the top.
    rows.sort(key=lambda r: (r["mutual_rho"] - r["min_human_rho"]))
    for r in rows[:12]:
        print(f"{r['arm_a'] + ' + ' + r['arm_b']:<26}{r['mutual_rho']:>9.3f}"
              f"{r['human_rho_a']:>9.3f}{r['human_rho_b']:>9.3f}"
              f"{r['min_human_rho']:>11.3f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[r30_metric_redundancy] wrote {args.out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--divergence_csv", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"))
    p.add_argument("--pass1", type=Path,
                   default=Path("evaluation/csv/r30_human_b_template.csv"))
    p.add_argument("--pass2", type=Path,
                   default=Path("evaluation/csv/r30_human_b_pass2.csv"))
    p.add_argument("--key", type=Path, default=None,
                   help="Scramble key for pass 2 (kept outside the repo).")
    p.add_argument("-o", "--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
