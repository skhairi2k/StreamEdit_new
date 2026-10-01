"""Join R10's two metric sources into one final table.

R10's five metrics come from two places and cannot come from one: three are
benchmark metrics computed by the unmodified fivebench evaluator (via
r10_metrics.py), and two are anchor-referenced and have no benchmark equivalent
(r10_anchor_sim.py -- calculate_clip_similarity is image-to-TEXT only). This
script joins them on (arm, video_name, editing_type_id) and keeps ONLY the five,
so the final CSV cannot be misread by picking up a superseded column.

WHY THE KEY IS A TRIPLE
-----------------------
0011_lucia is scored twice, under edit types 2 and 5 -- same source video, two
target prompts, therefore two clips. Keying on video_name alone would silently
collapse them into one row.

MISSING ROWS ARE REPORTED, NOT DROPPED
--------------------------------------
A row present in one source and absent from the other means an arm failed to
render or an anchor was missing. That is a result about the run, so it is printed
and written with blank cells rather than quietly excluded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# The five metrics, and nothing else. Anything not named here is dropped, which
# is the point: the superseded run wrote whole-frame columns into a file of the
# same name, and those must not survive into the table people read.
ANCHOR_COLS = ["anchor_sim_f0", "anchor_sim_mean", "anchor_sim_last",
               "anchor_sim_retention", "anchor_sim_slope_per100f"]
BENCH_COLS = ["motion_fidelity_score_edit_part",
              "clip_similarity_target_image_edit_part_mean",
              "clip_similarity_target_image_edit_part_slope_per100f",
              "lpips_unedit_part_mean"]
KEY = ("arm", "video_name", "editing_type_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench_csv", type=Path, required=True,
                   help="r10_arms.csv written by r10_metrics.py")
    p.add_argument("--anchor_csv", type=Path, required=True,
                   help="r10_anchor_sim.csv written by r10_anchor_sim.py")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load(path: Path, cols: list[str]) -> tuple[dict, list[str]]:
    if not path.exists():
        raise SystemExit(f"[join] missing {path}")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"[join] {path} is empty")
    have = [c for c in cols if c in rows[0]]
    if len(have) != len(cols):
        print(f"[join] WARNING {path.name} lacks: {sorted(set(cols) - set(have))}")
    out = {}
    for r in rows:
        out[tuple(str(r[k]) for k in KEY)] = {c: r.get(c, "") for c in have}
    return out, have


def main() -> None:
    args = parse_args()
    bench, bench_cols = load(args.bench_csv, BENCH_COLS)
    anchor, anchor_cols = load(args.anchor_csv, ANCHOR_COLS)

    keys = sorted(set(bench) | set(anchor))
    only_bench = sorted(set(bench) - set(anchor))
    only_anchor = sorted(set(anchor) - set(bench))
    for k in only_bench:
        print(f"[join] no anchor row for {k} -- anchor columns blank")
    for k in only_anchor:
        print(f"[join] no benchmark row for {k} -- benchmark columns blank")

    fields = list(KEY) + anchor_cols + bench_cols
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in keys:
            row = dict(zip(KEY, k))
            row.update({c: "" for c in anchor_cols + bench_cols})
            row.update(anchor.get(k, {}))
            row.update(bench.get(k, {}))
            w.writerow(row)

    arms = sorted({k[0] for k in keys})
    clips = {(k[1], k[2]) for k in keys}
    print(f"[join] {len(keys)} rows -> {args.out}")
    print(f"[join] {len(arms)} arms x {len(clips)} clips: {', '.join(arms)}")
    if only_bench or only_anchor:
        print(f"[join] INCOMPLETE: {len(only_bench)} missing anchor, "
              f"{len(only_anchor)} missing benchmark")


if __name__ == "__main__":
    main()
