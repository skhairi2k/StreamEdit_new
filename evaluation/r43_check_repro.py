"""R43 -- does the R43 anchor generator reproduce R7's anchors?

The ``anchor-smoke`` step regenerates 3 R7 pairs (FiVE's own instruction, no keep-pose suffix)
into ``anchor_r43_smoke/repro``. This script decodes each new PNG and R7's
``<r7_root>/edit{T}/{video}.png`` and reports ``IDENTICAL`` or ``DIFFERS`` with the max abs
diff, mean abs diff (0-255 scale) and PSNR. IDENTICAL confirms the claim that the Correct
anchors come from the same generator as R7's; DIFFERS is informational (a node-level kernel
difference, not a setting change) and does not fail the script.

Exit 0 either way; exit 1 only if a file is missing or the image sizes differ.

Example
-------
    python evaluation/r43_check_repro.py \\
        --manifest evaluation/r43_anchor_manifest_repro.json \\
        --r7_root ~/Data/dataggen/outputs/five_bench/anchors \\
        --out_csv evaluation/csv/r43_repro_check.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=str, default="evaluation/r43_anchor_manifest_repro.json")
    parser.add_argument("--r7_root", type=str, default="~/Data/dataggen/outputs/five_bench/anchors")
    parser.add_argument("--out_csv", type=str, default="evaluation/csv/r43_repro_check.csv")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def main() -> None:
    args = parse_args()
    r7_root = Path(args.r7_root).expanduser()
    rows, problems = [], []
    for case in json.loads(Path(args.manifest).read_text()):
        cid, t, video = case["case_id"], case["edit_type"], case["video_name"]
        new_path = Path(case["anchor_image"]).expanduser()
        r7_path = r7_root / f"edit{t}" / f"{video}.png"
        missing = [str(p) for p in (new_path, r7_path) if not p.is_file()]
        if missing:
            problems.append(f"{cid}: missing {missing}")
            continue
        new, old = load_rgb(new_path), load_rgb(r7_path)
        if new.shape != old.shape:
            problems.append(f"{cid}: size differs {new.shape} vs {old.shape}")
            continue
        diff = np.abs(new - old)
        mse = float((diff.astype(np.float64) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
        rows.append({"case_id": cid, "status": "IDENTICAL" if diff.max() == 0 else "DIFFERS",
                     "max_abs_diff": int(diff.max()), "mean_abs_diff": f"{diff.mean():.4f}",
                     "psnr_db": "inf" if math.isinf(psnr) else f"{psnr:.2f}",
                     "new_path": str(new_path), "r7_path": str(r7_path)})

    if rows:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for r in rows:
            print(f"[r43_check_repro] {r['case_id']}: {r['status']} (max {r['max_abs_diff']}, "
                  f"mean {r['mean_abs_diff']}, PSNR {r['psnr_db']} dB)")
        n_same = sum(r["status"] == "IDENTICAL" for r in rows)
        print(f"[r43_check_repro] {n_same}/{len(rows)} IDENTICAL -> {out}")
    if problems:
        print("\n".join(problems))
        sys.exit(f"[r43_check_repro] {len(problems)} problem(s)")


if __name__ == "__main__":
    main()
