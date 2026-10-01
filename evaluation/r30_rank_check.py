#!/usr/bin/env python
"""R30 rank check: does each arm's raw divergence `d` rank clips the same way the
per-clip oracle would route them?

For a fixed alpha, the oracle picks each clip's own best b in CONST_BS by argmaxing
`alpha*CLIP_norm + (1-alpha)*LPIPS_norm` (per-clip min-max, LPIPS direction-corrected --
see r30_score.py's `oracle_frontier`, which this reuses the same per-clip argmax logic
from, just returning the WINNING b per clip instead of the alpha-aggregate mean). That
gives one b* per clip, per alpha. This script then asks: across the 22 clips, does an
arm's raw d rank clips the same way b* does? Spearman(d, b*) answers exactly this --
it is invariant to any monotonic rescaling of d (so normalizing d first, as
r30_video_grids.py's `m` column does, would not change these numbers at all), which is
what makes it a check on the divergence MEASURE's ranking, independent of however it
later gets mapped to b.

alpha=0 is pure LPIPS-preservation-driven routing (b* is whichever b preserves best,
almost always b_min); alpha=1 is pure CLIP-achievement-driven routing (b* is whichever b
achieves most, almost always b_max) -- both ends are close to constant across clips by
construction and TIES DOMINATE the ranking there (see the printed tie-count), which
Spearman's average-rank tie-breaking handles but which also means a high or low rho at
the extremes carries much less information than one at alpha=0.5. The interesting
alphas are the interior ones, where b* actually varies clip to clip.

Usage
-----
    python evaluation/r30_rank_check.py \\
        --divergence_csv evaluation/csv/r30_divergence.csv \\
        --r26_csv_dir evaluation/csv \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --cases evaluation/cases.json \\
        --alphas 0,0.25,0.5,0.75,1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import (  # noqa: E402
    ARMS, CONST_BS, build_join_tables, load_r26_metric, _minmax,
)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, ties broken by average rank. No scipy dependency,
    matching this codebase's existing convention (see git history of r30_score.py)."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def oracle_b_star(clips: Sequence[str], clip_target: Dict[Tuple[str, int], float],
                  lpips: Dict[Tuple[str, int], float], bs: Sequence[int],
                  alpha: float) -> Dict[str, int]:
    """Per-clip winning b at this alpha -- alpha*CLIP_norm + (1-alpha)*(1-LPIPS_norm),
    per-clip min-max, LPIPS direction-corrected (lower is better). Ties break on higher
    CLIP then lower LPIPS, matching r30_score.py's oracle_frontier exactly."""
    out: Dict[str, int] = {}
    for c in clips:
        clip_by_b = {b: clip_target[(c, b)] for b in bs}
        y_by_b = {b: lpips[(c, b)] for b in bs}
        cn = _minmax(clip_by_b)
        yn = {b: 1.0 - v for b, v in _minmax(y_by_b).items()}
        score = {b: alpha * cn[b] + (1 - alpha) * yn[b] for b in bs}
        best = max(bs, key=lambda b: (score[b], clip_by_b[b], -y_by_b[b]))
        out[c] = best
    return out


def load_divergence_by_arm(path: Path, arms: Sequence[str]
                           ) -> Dict[str, Dict[str, float]]:
    """{arm: {case_id: d_masked}}."""
    out: Dict[str, Dict[str, float]] = {a: {} for a in arms}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for arm in arms:
                v = r.get(f"d_{arm}_masked", "")
                if v not in ("", None):
                    out[arm][r["case_id"]] = float(v)
    return out


def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})

    clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                  "clip_similarity_target_image")
    lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                            "lpips_unedit_part")
    div_by_arm = load_divergence_by_arm(args.divergence_csv, ARMS)

    alphas = [float(a) for a in args.alphas.split(",")]
    b_star_by_alpha: Dict[float, Dict[str, int]] = {
        a: oracle_b_star(clips, clip_target, lpips, CONST_BS, a) for a in alphas
    }

    print(f"=== R30 rank check: Spearman(d_arm, b*(alpha)) over {len(clips)} clips ===")
    header = "arm".ljust(12) + "".join(f"alpha={a:<9.2g}" for a in alphas)
    print(header)
    rows_out = []
    for arm in ARMS:
        d_by_case = div_by_arm.get(arm, {})
        common = [c for c in clips if c in d_by_case]
        missing = len(clips) - len(common)
        if missing:
            print(f"[r30_rank_check] WARNING {arm}: {missing}/{len(clips)} clips have "
                 f"no d_{arm}_masked, correlating on the remaining {len(common)}")
        d_vals = [d_by_case[c] for c in common]
        line = arm.ljust(12)
        row = {"arm": arm, "n_clips": len(common)}
        for a in alphas:
            b_vals = [b_star_by_alpha[a][c] for c in common]
            rho = spearman(d_vals, b_vals)
            n_distinct_b = len(set(b_vals))
            line += f"{rho:>9.3f} " if rho == rho else f"{'n/a':>9} "
            row[f"rho_alpha_{a:g}"] = round(rho, 4) if rho == rho else ""
            row[f"n_distinct_b_alpha_{a:g}"] = n_distinct_b
        print(line)
        rows_out.append(row)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows_out[0].keys())
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n[r30_rank_check] wrote {args.out}")

    print("\nNote: alpha=0/1 push b* toward a near-constant value across clips (see "
         "n_distinct_b in the CSV) -- ties dominate there, so rho is most informative "
         "at the interior alphas (0.25-0.75), where b* actually varies clip to clip.")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--divergence_csv", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"))
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Optional CSV to write (arm x alpha rho table + n_distinct_b).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
