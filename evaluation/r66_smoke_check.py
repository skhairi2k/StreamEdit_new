"""R66 smoke check: contact sheet, timing table and the projected Jean Zay H100 cost of the 41-frame render.

Reads --smoke_root (default ~/Data/dataggen/outputs/five_bench/r66_contextflow/smoke, written by r66_smoke_jz.sh):
manifest_n{41,81}_shard0.csv and the renders n{41,81}/edit{T}/{video}/*.png. The cases' anchor_image / src_first_frame are
relative to --data_dir (default ~/Data; on Jean Zay ~/Data -> $WORK/Data).

Writes
  evaluation/figures/r66_smoke_sheet.jpg   per smoke pair, rows = frame 0 / middle / last of each clip, columns =
                                           source (at the n41 indices) | anchor | ContextFlow n81 | ContextFlow n41;
                                           every tile is labelled with its frame index, a missing render (e.g. OOM) is a grey tile
  evaluation/csv/r66_smoke_timing.csv      per pair and frame count: mode, frames, sec_inversion, sec_edit, peak_mem_gb, status,
                                           then one 'mean n{F}' row per frame count; the n41 row carries the projection
Projection (41 frames only; 81 is a diagnostic), every job holding 2 H100 (the authors' setup):
  job-hours = (n_videos x mean inversion + n_pairs x mean edit) x 1.15 for loading / overhead; H100-hours = job-hours x 2;
  K = the smallest number of video shards with ceil(n_videos / K) x (mean inversion + pairs per video x mean edit) x 1.3 < 18 h
  (the gate's rule, under the 20 h qos limit); with the K jobs running at once the render takes about one shard's time.
n_videos and n_pairs from evaluation/r66_cases_correct.json, means over the ok n41 pairs (inversion over the pairs that ran one).
Peak memory (max over the 2 GPUs) is flagged against 76 GB (H100 80 GB).

Usage (Jean Zay frontal, in the clone): $WORK/envs/contextflow/bin/python -I evaluation/r66_smoke_check.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
CASES_SMOKE = EVAL_DIR / "r66_cases_smoke.json"
CASES_ALL = EVAL_DIR / "r66_cases_correct.json"
SHEET = EVAL_DIR / "figures" / "r66_smoke_sheet.jpg"
TIMING = EVAL_DIR / "csv" / "r66_smoke_timing.csv"

FRAME_COUNTS = (41, 81)
OVERHEAD = 1.15
GPUS_PER_JOB = 2
SHARD_SAFETY = 1.3
SHARD_LIMIT_H = 18.0
PEAK_LIMIT_GB = 76.0
TILE_W, TILE_H = 312, 180
GREY = 96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke_root", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/r66_contextflow/smoke").expanduser())
    p.add_argument("--data_dir", type=Path, default=Path("~/Data").expanduser())
    return p.parse_args()


def read_manifest(smoke: Path, frames: int) -> dict[str, dict]:
    """Last row per case_id (a resubmit appends)."""
    path = smoke / f"manifest_n{frames}_shard0.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["case_id"]: row for row in csv.DictReader(f)}


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (TILE_W, 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def tile(path: Path | None, text: str) -> np.ndarray:
    if path is None or not path.is_file():
        return label(np.full((TILE_H, TILE_W, 3), GREY, np.uint8), f"{text}: missing")
    return label(cv2.resize(cv2.imread(str(path)), (TILE_W, TILE_H), interpolation=cv2.INTER_AREA), text)


def three(n: int) -> list[int]:
    return [0, (n - 1) // 2, n - 1]


def shards_needed(n_videos: int, sec_per_video: float) -> tuple[int, float]:
    """Smallest K whose largest shard stays under SHARD_LIMIT_H with the safety factor, and that shard's hours."""
    for k in range(1, n_videos + 1):
        hours = math.ceil(n_videos / k) * sec_per_video * SHARD_SAFETY / 3600
        if hours < SHARD_LIMIT_H:
            return k, hours
    return n_videos, sec_per_video * SHARD_SAFETY / 3600


def main() -> int:
    args = parse_args()
    smoke = args.smoke_root
    manifests = {f: read_manifest(smoke, f) for f in FRAME_COUNTS}
    if not manifests[41]:
        print(f"ERROR: no {smoke}/manifest_n41_shard0.csv", file=sys.stderr)
        return 1
    smoke_cases = json.loads(CASES_SMOKE.read_text())
    all_cases = json.loads(CASES_ALL.read_text())
    n_videos, n_pairs = len({c["video_name"] for c in all_cases}), len(all_cases)

    # ---- contact sheet ------------------------------------------------------------------------------------
    blocks = []
    for c in smoke_cases:
        src_files = sorted((args.data_dir / c["src_first_frame"]).parent.glob("*.jpg"))
        renders = {f: sorted((smoke / f"n{f}" / f"edit{c['edit_type']}" / c["video_name"]).glob("*.png")) for f in FRAME_COUNTS}
        title = np.zeros((22, 4 * TILE_W, 3), np.uint8)
        cv2.putText(title, f"{c['case_id']}  [{c['mode']}]  {c['target_prompt'][:110]}", (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows = [title]
        for r, i41 in enumerate(three(c["n41"])):
            src = tile(src_files[i41], f"source f{i41}")
            anchor = tile(args.data_dir / c["anchor_image"], "anchor") if r == 0 else np.full((TILE_H, TILE_W, 3), 0, np.uint8)
            cols = [src, anchor]
            for f in (81, 41):
                fr = renders[f]
                if len(fr) == c[f"n{f}"]:
                    i = three(len(fr))[r]
                    cols.append(tile(fr[i], f"ContextFlow n{f} f{i}"))
                else:
                    status = manifests[f].get(c["case_id"], {}).get("status", "not run")
                    cols.append(tile(None, f"ContextFlow n{f} ({status})"))
            rows.append(np.concatenate(cols, axis=1))
        blocks.append(np.concatenate(rows, axis=0))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEET), np.concatenate(blocks, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[r66_smoke_check] contact sheet -> {SHEET}")

    # ---- timing + projection ------------------------------------------------------------------------------
    fields = ["pair", "mode", "frames", "sec_inversion", "sec_edit", "peak_mem_gb", "status"]
    out_rows, verdict = [], 0
    for f in FRAME_COUNTS:
        rows = list(manifests[f].values())
        for row in rows:
            out_rows.append({"pair": row["case_id"], "mode": row["mode"], "frames": row["n_frames"],
                             "sec_inversion": row["sec_inversion"], "sec_edit": row["sec_edit"],
                             "peak_mem_gb": row["peak_mem_gb"], "status": row["status"]})
        ok = [r for r in rows if r["status"] == "ok"]
        inv = [float(r["sec_inversion"]) for r in ok if float(r["sec_inversion"]) > 0]
        edit = [float(r["sec_edit"]) for r in ok]
        peak = max((float(r["peak_mem_gb"]) for r in rows), default=float("nan"))
        mean_inv = float(np.mean(inv)) if inv else float("nan")
        mean_edit = float(np.mean(edit)) if edit else float("nan")
        note = f"{len(ok)}/{len(rows)} ok; max peak {peak:.1f} GB ({'OK' if peak < PEAK_LIMIT_GB else 'OVER'} vs {PEAK_LIMIT_GB:.0f} GB)"
        if f == 41:
            job_hours = (n_videos * mean_inv + n_pairs * mean_edit) * OVERHEAD / 3600
            k, shard_hours = shards_needed(n_videos, mean_inv + n_pairs / n_videos * mean_edit)
            note += (f"; projected {n_videos} inversions + {n_pairs} edits x {OVERHEAD}: {job_hours:.1f} h of {GPUS_PER_JOB}-GPU jobs = "
                     f"{job_hours * GPUS_PER_JOB:.1f} H100-h; K = {k} shards ({shard_hours:.1f} h each with x{SHARD_SAFETY}, "
                     f"< {SHARD_LIMIT_H:.0f} h), about {shard_hours:.1f} h wall-clock if the {k} jobs run at once")
            verdict = int(len(ok) < len(rows) or not peak < PEAK_LIMIT_GB)
        else:
            note += " (diagnostic only)"
        out_rows.append({"pair": f"mean n{f}", "mode": "", "frames": f, "sec_inversion": round(mean_inv, 1),
                         "sec_edit": round(mean_edit, 1), "peak_mem_gb": round(peak, 2), "status": note})
        print(f"[r66_smoke_check] n{f}: mean inversion {mean_inv:.0f}s, mean edit {mean_edit:.0f}s; {note}")
    TIMING.parent.mkdir(parents=True, exist_ok=True)
    with TIMING.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[r66_smoke_check] timing -> {TIMING}")
    if verdict:
        print("[r66_smoke_check] WARNING: n41 has non-ok pairs or peak memory over the gate limit; see the table", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
