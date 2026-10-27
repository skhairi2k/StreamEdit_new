"""R60 -- reproducibility gate for reusing R44's text-only renders on FiVE-Bench-Correct.

The 2 pairs of ``evaluation/r60_cases_repro.json`` (Correct left their prompts and trigger words unchanged) were rendered
again by ``r60_render.sh repro`` (``n15_rho2`` and ``n20_rho2``, Correct annotations). This compares those frames with R44's
renders of the same pairs. PASS iff every frame has max |diff| <= ``--tol`` (of 255); only then may R44's renders of the
225 unchanged pairs be mixed with the new renders of the 190 changed ones.

R44's n20 renders live on /store: run this on a node that mounts it (nodecpu12, node39, gpu-gw).
Writes ``evaluation/csv/r60_repro_check.csv`` (config, case_id, frame, max_abs_diff, psnr). Exit 1 on FAIL.
usage: python evaluation/r60_repro_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
D = Path.home() / "Data/dataggen/outputs/five_bench"
CONFIGS = ("n15_rho2", "n20_rho2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=REPO / "evaluation/r60_cases_repro.json")
    ap.add_argument("--new_root", type=Path, default=D / "r60_repro")
    ap.add_argument("--ref_root", type=Path, default=D / "r44_rho_sweep")
    ap.add_argument("--csv", type=Path, default=REPO / "evaluation/csv/r60_repro_check.csv")
    ap.add_argument("--tol", type=int, default=2, help="max |diff| allowed per frame, of 255")
    a = ap.parse_args()

    cases = json.loads(a.cases.read_text())
    rows, errors = [], []
    for cfg in CONFIGS:
        for c in cases:
            t, vid = int(c["edit_type"]), c["video_name"]
            new_dir = a.new_root / f"r60_{cfg}_novp" / f"edit{t}" / vid
            ref_dir = a.ref_root / f"r44_{cfg}_novp" / f"edit{t}" / vid
            new = sorted(new_dir.glob("*.png"))
            ref = sorted(ref_dir.glob("*.png"))
            if not new or not ref:
                errors.append(f"{cfg} {c['case_id']}: {len(new)} new frames in {new_dir}, {len(ref)} reference frames in {ref_dir}")
                continue
            if [p.name for p in new] != [p.name for p in ref]:
                errors.append(f"{cfg} {c['case_id']}: frame sets differ ({len(new)} new vs {len(ref)} reference)")
                continue
            worst = 0
            for pn, pr in zip(new, ref):
                x = np.asarray(Image.open(pn).convert("RGB"), dtype=np.int16)
                y = np.asarray(Image.open(pr).convert("RGB"), dtype=np.int16)
                if x.shape != y.shape:
                    errors.append(f"{cfg} {c['case_id']} {pn.name}: shape {x.shape} vs {y.shape}")
                    continue
                diff = np.abs(x - y)
                mx = int(diff.max())
                mse = float((diff.astype(np.float64) ** 2).mean())
                psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
                rows.append((cfg, c["case_id"], pn.name, mx, f"{psnr:.2f}"))
                worst = max(worst, mx)
            ok = worst <= a.tol
            print(f"[r60_repro_check] {cfg} {c['case_id']}: {len(new)} frames, max |diff| {worst} -> {'ok' if ok else 'FAIL'}")
            if not ok:
                errors.append(f"{cfg} {c['case_id']}: max |diff| {worst} > {a.tol}")
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "case_id", "frame", "max_abs_diff", "psnr"])
        w.writerows(rows)
    print(f"[r60_repro_check] {a.csv}: {len(rows)} frame rows")
    if errors:
        for e in errors:
            print(f"[r60_repro_check] ERROR: {e}")
        print("[r60_repro_check] FAIL")
        return 1
    print(f"[r60_repro_check] PASS (every frame max |diff| <= {a.tol})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
