#!/usr/bin/env python
"""R36 render-parity gates: frame-by-frame sha256 of R36 renders against stored references.

Two references are exact by construction, so any mismatch is a regression, not noise:

* R26 diagonal (``r26_spatial_tau/taubg{R}_taufg{R}_vp``): same scalar ``--blend_power``
  path, same per-pair seed, same sampler, same anchors -> every ``cases.json`` clip must
  be byte-identical at every rho.
* R7 (``r7_visual_prompting``, rho=2): rendered BEFORE per-pair reseeding (2026-07-22),
  so only the index-0 pair of each ``edit{T}_FiVE.json`` drew the same noise as today.
  Only those pairs are compared.

Modes:
  --smoke  3 gates on the r36_smoke.sh renders (0001_bus edit1, 0007_guitar-violin edit5):
           GATE1 rho=3 == R26 (3,3); GATE2 rho=2 == R7; GATE3 rho=3 != rho=2.
  default  full-bench checks over the 8 R36 arms: 419 real dirs + all-ok manifests per
           arm, 22 cases.json clips x 8 rho vs R26, index-0 pairs at rho=2 vs R7.
           Writes one row per comparison to --out.

Exits non-zero if any gate/check fails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RHOS = (2, 3, 4, 6, 8, 10, 20, 50)
EXPECTED_PER_TYPE = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
FIVE_ROOT = Path("~/Data/dataggen/outputs/five_bench").expanduser()
# R26/R7 reference renders were moved off the home quota on 2026-09-25.
ARCHIVE_ROOT = Path("/work/PERSO/skhairi/outputs/five_bench")


def frame_hashes(clip_dir: Path) -> list[str]:
    """sha256 of every frame PNG in sorted order; empty list if the dir is missing."""
    if not clip_dir.is_dir():
        return []
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(clip_dir.glob("*.png"))]


def compare(a: Path, b: Path) -> dict:
    ha, hb = frame_hashes(a), frame_hashes(b)
    n_diff = sum(x != y for x, y in zip(ha, hb)) + abs(len(ha) - len(hb))
    return {
        "n_frames": len(ha),
        "n_frames_ref": len(hb),
        "n_diff_frames": n_diff,
        "missing": int(not ha or not hb),
        "identical": int(bool(ha) and ha == hb),
    }


def index0_videos(data_root: Path) -> dict[int, str]:
    """video_name of the first pair in each edit{T}_FiVE.json."""
    out = {}
    for t in EXPECTED_PER_TYPE:
        entries = json.loads((data_root / "edit_prompt" / f"edit{t}_FiVE.json").read_text())
        out[t] = entries[0]["video_name"]
    return out


def arm_name(rho: int) -> str:
    return f"r36_rho{rho}_vp"


def run_smoke(args: argparse.Namespace) -> int:
    cases = json.loads(args.smoke_cases.read_text())
    idx0 = index0_videos(args.data_root)
    failures = 0

    for c in cases:
        t, vid = int(c["edit_type"]), c["video_name"]
        if idx0[t] != vid:
            print(f"GATE2-FAIL setup: {vid} is not index 0 of edit{t}_FiVE.json "
                  f"(index 0 is {idx0[t]}); R7 parity is not expected to hold for it")
            failures += 1

    def clip(root: Path, arm: str, c: dict) -> Path:
        return root / arm / f"edit{int(c['edit_type'])}" / c["video_name"]

    print("=== GATE1: rho=3 == R26 taubg3_taufg3_vp ===")
    for c in cases:
        r = compare(clip(args.smoke_root, arm_name(3), c),
                    clip(args.r26_root, "taubg3_taufg3_vp", c))
        ok = r["identical"]
        failures += not ok
        print(f"{'IDENTICAL' if ok else 'GATE1-FAIL'} gate1 {c['video_name']}: {r}")

    print("=== GATE2: rho=2 == R7 r7_visual_prompting (index-0 pairs) ===")
    for c in cases:
        r = compare(clip(args.smoke_root, arm_name(2), c),
                    args.r7_root / f"edit{int(c['edit_type'])}" / c["video_name"])
        ok = r["identical"]
        failures += not ok
        print(f"{'IDENTICAL' if ok else 'GATE2-FAIL'} gate2 {c['video_name']}: {r}")

    print("=== GATE3: rho=3 != rho=2 (flag reaches the bridge) ===")
    for c in cases:
        r = compare(clip(args.smoke_root, arm_name(3), c),
                    clip(args.smoke_root, arm_name(2), c))
        ok = not r["missing"] and not r["identical"]
        failures += not ok
        print(f"{'DIFFERS' if ok else 'GATE3-FAIL'} gate3 {c['video_name']}: {r}")

    print(f"[r36_check_parity] smoke failures: {failures}")
    return int(failures > 0)


def count_arm(arm_dir: Path) -> tuple[dict[int, int], int]:
    """Real frame dirs per edit type (excluding *_resize) and total non-ok manifest rows."""
    counts, n_bad = {}, 0
    for t in EXPECTED_PER_TYPE:
        d = arm_dir / f"edit{t}"
        counts[t] = sum(1 for p in d.glob("*/") if p.is_dir() and not p.name.endswith("_resize")) \
            if d.is_dir() else 0
        man = d / "_manifest.csv"
        if man.is_file():
            with man.open() as f:
                n_bad += sum(1 for row in csv.DictReader(f) if row["status"] != "ok")
        else:
            n_bad += 1
    return counts, n_bad


def run_full(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.read_text())
    idx0 = index0_videos(args.data_root)
    rows, failures = [], 0

    for rho in RHOS:
        arm_dir = args.r36_root / arm_name(rho)
        counts, n_bad = count_arm(arm_dir)
        ok = counts == EXPECTED_PER_TYPE and n_bad == 0
        failures += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {arm_name(rho)} dirs={counts} "
              f"total={sum(counts.values())}/419 non_ok_manifest={n_bad}")

        for c in cases:
            t, vid = int(c["edit_type"]), c["video_name"]
            r = compare(arm_dir / f"edit{t}" / vid,
                        args.r26_root / f"taubg{rho}_taufg{rho}_vp" / f"edit{t}" / vid)
            failures += not r["identical"]
            rows.append({"check": "r26_diagonal", "arm": arm_name(rho), "edit_type": t,
                         "clip": vid, "ref": f"taubg{rho}_taufg{rho}_vp", **r})

    for t, vid in idx0.items():
        r = compare(args.r36_root / arm_name(2) / f"edit{t}" / vid,
                    args.r7_root / f"edit{t}" / vid)
        failures += not r["identical"]
        rows.append({"check": "r7_index0", "arm": arm_name(2), "edit_type": t,
                     "clip": vid, "ref": "r7_visual_prompting", **r})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for check in ("r26_diagonal", "r7_index0"):
        sub = [r for r in rows if r["check"] == check]
        n_id = sum(r["identical"] for r in sub)
        print(f"[r36_check_parity] {check}: {n_id}/{len(sub)} identical")
    print(f"[r36_check_parity] wrote {args.out}; failures: {failures}")
    return int(failures > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="run the 3 smoke gates instead of the full check")
    p.add_argument("--smoke_root", type=Path, default=FIVE_ROOT / "r36_smoke")
    p.add_argument("--smoke_cases", type=Path, default=Path("evaluation/r36_smoke_cases.json"))
    p.add_argument("--r36_root", type=Path, default=FIVE_ROOT / "r36_rho_sweep")
    p.add_argument("--r26_root", type=Path, default=ARCHIVE_ROOT / "r26_spatial_tau")
    p.add_argument("--r7_root", type=Path, default=ARCHIVE_ROOT / "r7_visual_prompting")
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark").expanduser())
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--out", type=Path, default=Path("evaluation/csv/r36_parity.csv"))
    args = p.parse_args(argv)
    for k in ("smoke_root", "r36_root", "r26_root", "r7_root", "data_root"):
        setattr(args, k, getattr(args, k).expanduser())
    return args


if __name__ == "__main__":
    a = parse_args()
    sys.exit(run_smoke(a) if a.smoke else run_full(a))
