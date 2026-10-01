#!/usr/bin/env python
"""R27 -- tabulate the 0011_lucia_e5 tau sweep: what each tau bought and what it cost.

Reads the per-tau edit5 CSVs written by r27_tau_sweep_eval.sh and prints one row per
tau, ordered numerically. Two columns carry the decision:

    clip_similarity_target_image_edit_part   did the dog actually get added
    lpips_unedit_part / psnr_unedit_part     did the woman and background survive

Their behaviour across tau IS the calibration curve. If edit-part CLIP rises while the
unedited-part metrics stay flat, higher tau is free and the ceiling should be high; if
the unedited part degrades as CLIP rises, there is a trade-off to place and the ceiling
is wherever the curves cross.

⚠️ n = 1. One clip, one edit, no error bar. These numbers locate a knee; they do not
support a claim. Read them WITH evaluation/figures/r27_tau_sweep_lucia_e5.png -- the
grid is the primary evidence here and the metrics are corroboration.

⚠️ The *_unedit_part metrics are computed against a mask, and for an ADDITION the added
object lands OUTSIDE the source's edited region. So a large, correct dog can read as
"unedited part got worse" purely because it occupies pixels the mask calls background.
Do not read a falling psnr_unedit_part as damage without checking the grid.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

METRICS = [
    ("clip_similarity_target_image_edit_part", "CLIP-edit", "+"),
    ("clip_similarity_target_image", "CLIP-tgt", "+"),
    ("clip_similarity_source_image", "CLIP-src", "."),
    ("lpips_unedit_part", "LPIPS-un", "-"),
    ("psnr_unedit_part", "PSNR-un", "+"),
    ("ssim_unedit_part", "SSIM-un", "+"),
    ("structure_distance", "struct-d", "-"),
    ("niqe_target_image", "NIQE", "-"),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--edit_type", type=int, default=5)
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r27_tau_sweep_summary.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pat = re.compile(rf"edit{args.edit_type}_FiVE_r27_sweep_tau([0-9.]+)_frame_stride8\.csv$")

    rows: List[Dict[str, object]] = []
    for f in sorted(args.csv_dir.glob(
            f"edit{args.edit_type}_FiVE_r27_sweep_tau*_frame_stride8.csv")):
        m = pat.search(f.name)
        if not m:
            continue
        tau = float(m.group(1))
        recs = list(csv.DictReader(f.open()))
        if not recs:
            print(f"[warn] {f.name} has no rows -- skipped", flush=True)
            continue
        if len(recs) != 1:
            print(f"[warn] {f.name} has {len(recs)} rows, expected 1 (single clip)",
                  flush=True)
        r = recs[0]
        row: Dict[str, object] = {"tau": tau}
        for key, _, _ in METRICS:
            col = next((c for c in r if c.endswith(f"|{key}")), None)
            row[key] = float(r[col]) if col and r[col] not in ("", "nan") else float("nan")
        rows.append(row)

    if not rows:
        raise SystemExit(f"no sweep CSVs found under {args.csv_dir}")
    rows.sort(key=lambda x: x["tau"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tau"] + [k for k, _, _ in METRICS])
        w.writeheader(); w.writerows(rows)

    hdr = f"{'tau':>6} " + " ".join(f"{lab:>9s}" for _, lab, _ in METRICS)
    print("=== R27 tau sweep on 0011_lucia_e5 (add a dog) -- n = 1 ===")
    print("    arrows: + higher is better, - lower is better, . neither (source fidelity)")
    print()
    print(hdr)
    print(f"{'':>6} " + " ".join(f"{d:>9s}" for _, _, d in METRICS))
    print("-" * len(hdr))
    for r in rows:
        tag = "  <- Eq. 4" if abs(float(r["tau"]) - 2.0) < 1e-9 else ""
        print(f"{r['tau']:>6.0f} " +
              " ".join(f"{r[k]:>9.4f}" for k, _, _ in METRICS) + tag)

    # Deltas against Eq. 4, which is the contrast that matters: the arm is only worth
    # anything if it beats the baseline the paper already ships.
    base = rows[0]
    print(f"\n--- delta vs tau = {base['tau']:.0f} (Eq. 4) ---")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[1:]:
        print(f"{r['tau']:>6.0f} " +
              " ".join(f"{r[k] - base[k]:>+9.4f}" for k, _, _ in METRICS))
    print(f"\nwrote {args.out}  ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
