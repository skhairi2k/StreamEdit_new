#!/usr/bin/env python
"""R30 explicit per-clip table: the oracle's b*(alpha) next to each arm's own routed b.

The companion listing to r30_rank_check.py's aggregate Spearman table -- this is the
raw per-clip data that table is computed FROM, at one fixed alpha, so a reader can see
exactly which clips the oracle and an arm agree or disagree on rather than only the
correlation summary.

Usage
-----
    python evaluation/r30_b_star_table.py --alpha 0.5 \\
        -o evaluation/csv/r30_b_star_table_alpha0.5.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import ARMS, CONST_BS, build_join_tables, load_r26_metric  # noqa: E402
from r30_rank_check import oracle_b_star  # noqa: E402


def load_routed_b(path: Path) -> dict:
    """{(case_id, arm): b}."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("b") not in ("", None):
                out[(r["case_id"], r["arm"])] = float(r["b"])
    return out


def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    clips = sorted({c["case_id"] for c in cases})

    clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                  "clip_similarity_target_image")
    lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                            "lpips_unedit_part")
    b_star = oracle_b_star(clips, clip_target, lpips, CONST_BS, args.alpha)
    routed = load_routed_b(args.fiveacc_arms_csv)

    rows = []
    for c in clips:
        row = {"case_id": c, "oracle_b_star": b_star[c]}
        for arm in ARMS:
            row[f"{arm}_b"] = routed.get((c, arm), "")
        rows.append(row)
    rows.sort(key=lambda r: r["oracle_b_star"])

    header = f"{'case_id':<24}{'oracle*':>9}" + "".join(f"{a:>13}" for a in ARMS)
    print(f"=== R30 per-clip b*(alpha={args.alpha:g}) vs. each arm's routed b ===")
    print(header)
    for row in rows:
        line = f"{row['case_id']:<24}{row['oracle_b_star']:>9}"
        for arm in ARMS:
            v = row[f"{arm}_b"]
            line += f"{v:>13.2f}" if v != "" else f"{'n/a':>13}"
        print(line)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[r30_b_star_table] wrote {args.out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--fiveacc_arms_csv", type=Path,
                   default=Path("evaluation/csv/r30_fiveacc_arms.csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("-o", "--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
