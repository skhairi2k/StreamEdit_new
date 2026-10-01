#!/usr/bin/env python
"""R30 test-retest: the rater's own self-consistency, i.e. the NOISE CEILING for every
human-vs-metric correlation in r30_human_rank_check.py.

Pass 1 (evaluation/csv/r30_human_b_template.csv) was rated off the NAMED per-clip pages
in evaluation/figures/r30_video_grids/. Pass 2 (evaluation/csv/r30_human_b_pass2.csv) was
rated BLIND: the same 21 clips re-rendered by r30_video_grids.py --label_map --anonymize
into evaluation/figures/r30_blind_rating/ as trial_NN.png, in shuffled order, with the
case_id stripped from the title AND the per-arm m/b numbers stripped from the row labels
(both are per-clip unique, so either alone would identify a clip the rater had already
scored). The scramble key lives OUTSIDE the repo, in the session scratchpad, so it cannot
be stumbled on while rating.

Why this number matters more than any single rho in r30_human_rank_check.py: a metric
cannot agree with a rater better than the rater agrees with themselves. Spearman(pass 1,
pass 2) is therefore the ceiling every arm's human-agreement rho should be read against
-- selfsim/clip_prompt at 0.570 means something very different if the ceiling is 0.60
(essentially AT ceiling, nothing left to explain) than if it is 0.95 (a real, large gap).

Reports Spearman plus two readings the correlation alone hides: exact-agreement rate
(identical grid pick) and agreement within one grid step, since b lives on an 8-point
ordinal grid (2,3,4,6,8,10,20,50) where "off by one notch" is a very different error from
"off by four".

Usage
-----
    python evaluation/r30_retest.py \\
        --pass1 evaluation/csv/r30_human_b_template.csv \\
        --pass2 evaluation/csv/r30_human_b_pass2.csv \\
        --key <scratchpad>/r30_blind_key.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import ARMS  # noqa: E402
from r30_rank_check import load_divergence_by_arm, spearman  # noqa: E402

GRID: Sequence[int] = (2, 3, 4, 6, 8, 10, 20, 50)


def load_pass1(path: Path) -> Dict[str, float]:
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            v = (r.get("human_b") or "").strip()
            if v and v.lower() != "none":
                out[r["case_id"]] = float(v)
    return out


def load_pass2(path: Path, key: Dict[str, str]) -> Dict[str, float]:
    """key maps case_id -> label; invert it to turn trial_NN back into a case_id."""
    label2case = {lab: case for case, lab in key.items()}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            v = (r.get("human_b_pass2") or "").strip()
            if not v or v.lower() == "none":
                continue
            case = label2case.get(r["trial_id"])
            if case is None:
                raise SystemExit(f"[r30_retest] {r['trial_id']} is not in the key")
            out[case] = float(v)
    return out


def load_key(path: Path) -> Dict[str, str]:
    with open(path) as fh:
        return {r["case_id"]: r["label"] for r in csv.DictReader(fh)}


def grid_index(b: float) -> int:
    """Position on the 8-point b grid, for 'within one notch' agreement."""
    return min(range(len(GRID)), key=lambda i: abs(GRID[i] - b))


def run(args: argparse.Namespace) -> int:
    key = load_key(args.key)
    p1 = load_pass1(args.pass1)
    p2 = load_pass2(args.pass2, key)
    clips = sorted(set(p1) & set(p2))
    if not clips:
        raise SystemExit("[r30_retest] no clips rated in both passes")

    x = [p1[c] for c in clips]
    y = [p2[c] for c in clips]
    rho = spearman(x, y)

    exact = sum(1 for c in clips if p1[c] == p2[c])
    within1 = sum(1 for c in clips
                  if abs(grid_index(p1[c]) - grid_index(p2[c])) <= 1)

    print(f"=== R30 test-retest (n={len(clips)}) ===")
    print(f"{'case_id':<24}{'pass1':>7}{'pass2':>7}{'notch delta':>13}")
    for c in clips:
        d = grid_index(p2[c]) - grid_index(p1[c])
        print(f"{c:<24}{p1[c]:>7.0f}{p2[c]:>7.0f}{d:>+13d}")

    print(f"\nSpearman(pass1, pass2)   rho = {rho:.3f}   <-- the noise ceiling")
    print(f"exact agreement          {exact}/{len(clips)} ({100*exact/len(clips):.0f}%)")
    print(f"within one grid notch    {within1}/{len(clips)} ({100*within1/len(clips):.0f}%)")

    # Every arm's human-agreement rho, recomputed against BOTH passes and against their
    # per-clip mean, read against the ceiling above. A metric at or above the ceiling is
    # not "better than the human" -- it is at the limit of what this rating can resolve.
    div = load_divergence_by_arm(args.divergence_csv, ARMS)
    mean_rating = {c: (p1[c] + p2[c]) / 2.0 for c in clips}
    print(f"\n{'arm':<12}{'vs pass1':>10}{'vs pass2':>10}{'vs mean':>10}"
          f"{'% of ceiling':>14}")
    for arm in ARMS:
        d = div.get(arm, {})
        cs = [c for c in clips if c in d]
        r1 = spearman([p1[c] for c in cs], [d[c] for c in cs])
        r2 = spearman([p2[c] for c in cs], [d[c] for c in cs])
        rm = spearman([mean_rating[c] for c in cs], [d[c] for c in cs])
        pct = 100 * rm / rho if rho and rho == rho and rho != 0 else float("nan")
        print(f"{arm:<12}{r1:>10.3f}{r2:>10.3f}{rm:>10.3f}{pct:>13.0f}%")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "human_b_pass1", "human_b_pass2", "notch_delta"])
            for c in clips:
                w.writerow([c, p1[c], p2[c], grid_index(p2[c]) - grid_index(p1[c])])
        print(f"\n[r30_retest] wrote {args.out}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pass1", type=Path,
                   default=Path("evaluation/csv/r30_human_b_template.csv"))
    p.add_argument("--pass2", type=Path,
                   default=Path("evaluation/csv/r30_human_b_pass2.csv"))
    p.add_argument("--key", type=Path, required=True,
                   help="case_id,label scramble key (kept outside the repo).")
    p.add_argument("--divergence_csv", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"))
    p.add_argument("-o", "--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
