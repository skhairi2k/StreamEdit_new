"""R58 -- reproducibility gate: may the new renders be mixed with the R49 renders of the unchanged pairs?

The gate pairs (evaluation/r58_cases_repro.json: Wan-Edit inputs unchanged by Correct) were re-rendered by
r58_wan_render.sh repro into $D/r58_repro/{config}_full/. Each is compared frame by frame with the R49 render of the
same config, read only (R49 folders are never written):

  native_s0.92      $D/r48_wan_arms/native_s0.92_full      (r48_wan_render.py, the official Wan-Edit setting)
  matched59_s0.92   $D/r49_wan_arms/matched59_s0.92_full   (r49_wan_render.py)

PASS iff every pair of both configs has the same frame names and max |diff| <= --tol (of 255) on every frame.
Writes evaluation/csv/r58_repro_check.csv (config, case_id, n_frames, max_abs_diff, worst_frame, min_psnr, pass); exit 1 on FAIL.
usage: python evaluation/r58_repro_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
D = Path.home() / "Data/dataggen/outputs/five_bench"
CONFIGS = {"native_s0.92": "r48_wan_arms", "matched59_s0.92": "r49_wan_arms"}   # config -> R49 render folder


def compare_dirs(new_dir: Path, ref_dir: Path) -> Tuple[int, int, str, float]:
    """-> (n_frames, max |diff| over all frames, worst frame name, min PSNR); raises on missing or differing frame names."""
    new = sorted(p.name for p in new_dir.glob("*.png"))
    ref = sorted(p.name for p in ref_dir.glob("*.png"))
    if not new or new != ref:
        raise ValueError(f"frame names differ: {len(new)} new vs {len(ref)} R49 ({new_dir} vs {ref_dir})")
    worst, worst_name, min_psnr = 0, new[0], float("inf")
    for name in new:
        a = np.asarray(Image.open(new_dir / name).convert("RGB"), dtype=np.int16)
        b = np.asarray(Image.open(ref_dir / name).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape {a.shape} vs {b.shape}")
        diff = np.abs(a - b)
        m = int(diff.max())
        mse = float((diff.astype(np.float64) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        min_psnr = min(min_psnr, psnr)
        if m > worst:
            worst, worst_name = m, name
    return len(new), worst, worst_name, min_psnr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=REPO / "evaluation/r58_cases_repro.json")
    ap.add_argument("--new_root", type=Path, default=D / "r58_repro")
    ap.add_argument("--tol", type=int, default=2, help="max |diff| (of 255) allowed on any frame")
    ap.add_argument("--out_csv", type=Path, default=REPO / "evaluation/csv/r58_repro_check.csv")
    a = ap.parse_args()

    cases: List[dict] = json.loads(a.cases.read_text())
    rows, n_fail = [], 0
    for config, ref_folder in CONFIGS.items():
        for c in cases:
            t, vid = int(c["edit_type"]), c["video_name"]
            new_dir = a.new_root / f"{config}_full" / f"edit{t}" / vid
            ref_dir = D / ref_folder / f"{config}_full" / f"edit{t}" / vid
            try:
                n, worst, worst_name, min_psnr = compare_dirs(new_dir, ref_dir)
                ok = worst <= a.tol
            except (ValueError, OSError) as e:
                print(f"[r58_repro] ERROR {config} {c['case_id']}: {e}")
                n, worst, worst_name, min_psnr, ok = 0, -1, "", float("nan"), False
            n_fail += not ok
            rows.append([config, c["case_id"], n, worst, worst_name, f"{min_psnr:.2f}", int(ok)])
            print(f"[r58_repro] {config:16s} {c['case_id']:24s} frames={n:3d} max|diff|={worst:3d} (frame {worst_name}) "
                  f"min PSNR={min_psnr:.2f} dB  {'ok' if ok else 'FAIL'}")
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "case_id", "n_frames", "max_abs_diff", "worst_frame", "min_psnr", "pass"])
        w.writerows(rows)
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"[r58_repro] {verdict}: {len(rows) - n_fail}/{len(rows)} (config, pair) within {a.tol}/255 -> {a.out_csv}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
