#!/usr/bin/env python
"""R36 summarize -- one row per rho: plain per-clip means over the 419 FiVE-Bench pairs.

Inputs (per rho R):
    edit{T}_FiVE_r36_rho{R}_vp_frame_stride8.csv          9 harness metrics  (r36_eval.sh)
    edit{T}_FiVE_r36_fiveacc_rho{R}_vp_frame_stride8.csv  FiVE-Acc           (r36_fiveacc.sh)
    r36_clip_directional.csv, method == r36_rho{R}_vp     CLIP-D             (r36_clipd.sh)

AGGREGATION: a PLAIN MEAN OVER CLIPS, each clip weight 1, computed from the per-clip CSVs.
Never evaluate.py's {stem}_avg.csv: that is a mean of the six per-edit-type means (edit5's
9 clips weigh as much as edit1's 100), and R31 found its final averaging wrong on some
columns (lpips_unedit_part ~196 instead of ~0.2).

Output rows (evaluation/csv/r36_rho_sweep.csv):
    subset=full     n_pairs=419  -- one per rho; the rows r35_score.py plots
    subset=cases22  n_pairs=22   -- same mean restricted to evaluation/cases.json, for the
                                    overlay against R26's 22-clip uniform diagonal
Metric columns carry their bare names (lpips_unedit_part, clip_d_prompt, five_acc_yes_no,
...), which r35_score.py resolves by exact match.

Hard failures: any per-clip CSV missing, any stem not exactly 100/100/100/100/9/10 rows, or
any metric not covering exactly the 419 benchmark clips (checks reused from r35_score.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r35_score import _check_coverage, clip_index, load_harness  # noqa: E402

RHOS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 20, 50)
HARNESS_METRICS: Tuple[str, ...] = (
    "structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
    "ssim_unedit_part", "clip_similarity_source_image", "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part", "niqe_target_image",
)
# five_acc = FiVE-Acc proper: evaluate.py's per-clip mean of yes/no, multi-choice, union, inter
FIVEACC_METRICS: Tuple[str, ...] = ("five_acc_yes_no", "five_acc_multi_choice", "five_acc")
CLIPD_METRICS: Tuple[str, ...] = ("clip_d_prompt", "clip_d_word")

Key = Tuple[int, str]  # (edit_type, video_name)


def load_clipd(path: Path, method: str, metric: str,
               clips: Dict[Tuple[int, str], str]) -> Dict[Key, float]:
    """{(edit_type, video_name): metric} for one method of r36_clip_directional.csv."""
    if not path.is_file():
        raise SystemExit(f"[r36_summarize] missing {path} -- has the clip-d step run?")
    out: Dict[Key, float] = {}
    for r in csv.DictReader(path.open()):
        if r["method"] != method:
            continue
        key = (int(r["edit_type"]), r["video_name"])
        if key in out:
            raise SystemExit(f"[r36_summarize] {path.name}: duplicate row {method} {key}")
        out[key] = float(r[metric])
    _check_coverage(out, f"{path.name}:{method}:{metric}", clips)
    return out


def per_clip_values(args: argparse.Namespace, rho: int,
                    clips: Dict[Tuple[int, str], str]) -> Dict[str, Dict[Key, float]]:
    harness = args.harness_fmt.format(rho=rho)
    fiveacc = args.fiveacc_fmt.format(rho=rho)
    method = args.clipd_method_fmt.format(rho=rho)
    vals: Dict[str, Dict[Key, float]] = {}
    for m in HARNESS_METRICS:
        vals[m] = load_harness(args.csv_dir, harness, m, clips)
    for m in FIVEACC_METRICS:
        vals[m] = load_harness(args.csv_dir, fiveacc, m, clips)
    for m in CLIPD_METRICS:
        vals[m] = load_clipd(args.clipd_csv, method, m, clips)
    return vals


def summarize(vals: Dict[str, Dict[Key, float]], keys: List[Key]) -> Dict[str, float]:
    return {m: float(np.mean([v[k] for k in keys])) for m, v in vals.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rhos", type=int, nargs="+", default=list(RHOS))
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--harness_fmt", default="r36_rho{rho}_vp",
                   help="evaluate.py stem of the 9-metric pass, formatted with rho")
    p.add_argument("--fiveacc_fmt", default="r36_fiveacc_rho{rho}_vp",
                   help="evaluate.py stem of the FiVE-Acc pass, formatted with rho")
    p.add_argument("--clipd_csv", type=Path,
                   default=Path("evaluation/csv/r36_clip_directional.csv"))
    p.add_argument("--clipd_method_fmt", default="r36_rho{rho}_vp",
                   help="`method` value in the CLIP-D CSV, formatted with rho")
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--out", type=Path, default=Path("evaluation/csv/r36_rho_sweep.csv"))
    args = p.parse_args(argv)
    args.data_root = args.data_root.expanduser()
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    clips = clip_index(args.data_root)
    full_keys = sorted({(t, v) for (t, _), v in clips.items()})
    cases_keys = sorted({(int(c["edit_type"]), c["video_name"])
                         for c in json.loads(args.cases.read_text())})
    missing = set(cases_keys) - set(full_keys)
    if missing:
        raise SystemExit(f"[r36_summarize] cases.json clips not in the benchmark: {sorted(missing)}")

    metrics = (*HARNESS_METRICS, *FIVEACC_METRICS, *CLIPD_METRICS)
    rows = []
    for rho in args.rhos:
        vals = per_clip_values(args, rho, clips)
        for subset, keys in (("full", full_keys), ("cases22", cases_keys)):
            rows.append({"rho": rho, "subset": subset, "n_pairs": len(keys),
                         **summarize(vals, keys)})
        full = rows[-2]
        print(f"[r36_summarize] rho={rho:<3} n={full['n_pairs']} "
              f"lpips={full['lpips_unedit_part']:.4f} "
              f"clipT={full['clip_similarity_target_image']:.3f} "
              f"clipD={full['clip_d_prompt']:+.4f} "
              f"fiveacc_yn={full['five_acc_yes_no']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rho", "subset", "n_pairs", *metrics])
        w.writeheader()
        w.writerows(rows)
    print(f"[r36_summarize] wrote {args.out} ({len(rows)} rows: "
          f"{len(args.rhos)} rho x full/cases22)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
