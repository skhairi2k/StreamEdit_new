#!/usr/bin/env python
"""R30 human-rating rank check: how well does a human's own visual pick of the best
constant b (from the r30_video_grids pages) rank clips against (a) the alpha-oracle's
b*(alpha), (b) each arm's raw divergence d, and (c) each arm's own routed b?

Reads evaluation/csv/r30_human_b_template.csv (case_id, human_b, grid_path). Rows whose
human_b is blank or "None" are DROPPED, not treated as a value -- this is how
0042_gym-ball (the one removal-type clip, confirmed elsewhere in this run to be
degenerate: its trg_word is a negation with no visual referent, so it never grounds
correctly regardless of b) opts itself out, without a separate exclusion list to keep in
sync by hand. All comparisons below run on whatever clips are left.

Three correlations per arm, all against the SAME human_b list, so they are directly
comparable:
  - Spearman(human_b, oracle_b*(alpha))   -- does the human agree with the oracle's own
    definition of "best", across the alpha sweep? (arm-independent)
  - Spearman(human_b, d_arm)              -- does the arm's raw divergence rank clips the
    way the human would, mapping-invariant (same spirit as r30_rank_check.py, human
    ground truth instead of the algorithmic oracle)
  - Spearman(human_b, arm_routed_b)       -- does the arm's ACTUAL output (metric + its
    current d->b mapping) land where the human would put it

Usage
-----
    python evaluation/r30_human_rank_check.py \\
        --human_csv evaluation/csv/r30_human_b_template.csv \\
        --alphas 0,0.25,0.5,0.75,1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import ARMS, CONST_BS, build_join_tables, load_r26_metric  # noqa: E402
from r30_rank_check import oracle_b_star, spearman, load_divergence_by_arm  # noqa: E402
from r30_b_star_table import load_routed_b  # noqa: E402


def load_human_b(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    dropped = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            v = (r.get("human_b") or "").strip()
            if v == "" or v.lower() == "none":
                dropped.append(r["case_id"])
                continue
            out[r["case_id"]] = float(v)
    if dropped:
        print(f"[r30_human_rank_check] excluded (no rating): {dropped}")
    return out


def run(args: argparse.Namespace) -> int:
    human_b = load_human_b(args.human_csv)
    clips = sorted(human_b)
    print(f"[r30_human_rank_check] {len(clips)} rated clips: {clips}\n")

    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)
    all_clips = sorted({c["case_id"] for c in cases})

    clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                  "clip_similarity_target_image")
    lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                            "lpips_unedit_part")
    alphas = [float(a) for a in args.alphas.split(",")]

    print("=== Spearman(human_b, oracle_b*(alpha)) ===")
    for a in alphas:
        b_star = oracle_b_star(all_clips, clip_target, lpips, CONST_BS, a)
        rho = spearman([human_b[c] for c in clips], [b_star[c] for c in clips])
        print(f"  alpha={a:<5g} rho={rho:>7.3f}   n={len(clips)}")

    div_by_arm = load_divergence_by_arm(args.divergence_csv, ARMS)
    routed = load_routed_b(args.fiveacc_arms_csv)

    print("\n=== Spearman(human_b, d_arm)  [metric only, mapping-invariant] ===")
    print(f"{'arm':<12}{'rho':>8}{'n':>5}")
    for arm in ARMS:
        d_by_case = div_by_arm.get(arm, {})
        cs = [c for c in clips if c in d_by_case]
        rho = spearman([human_b[c] for c in cs], [d_by_case[c] for c in cs])
        print(f"{arm:<12}{rho:>8.3f}{len(cs):>5}")

    print("\n=== Spearman(human_b, arm_routed_b)  [metric + its own mapping] ===")
    print(f"{'arm':<12}{'rho':>8}{'n':>5}")
    for arm in ARMS:
        cs = [c for c in clips if (c, arm) in routed]
        rho = spearman([human_b[c] for c in cs], [routed[(c, arm)] for c in cs])
        print(f"{arm:<12}{rho:>8.3f}{len(cs):>5}")

    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human_csv", type=Path,
                   default=Path("evaluation/csv/r30_human_b_template.csv"))
    p.add_argument("--divergence_csv", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"))
    p.add_argument("--fiveacc_arms_csv", type=Path,
                   default=Path("evaluation/csv/r30_fiveacc_arms.csv"))
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
