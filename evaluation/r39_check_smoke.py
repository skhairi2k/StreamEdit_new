#!/usr/bin/env python
"""R39 smoke gates: frame-by-frame sha256 of the r39_smoke.sh renders.

Gates (clips from --smoke_cases: 0001_bus edit1, 0007_guitar-violin edit5):
  GATE1 (IDENTICAL) N=15 rho=2 == R36 ``r36_rho_sweep/r36_rho2_vp``. Same flags as
        r36_infer.sh, so any mismatch means the render path moved since R36.
  GATE2 (DIFFERS)   N=20 rho=2 and N=30 rho=2, each vs N=15 rho=2: same frame count, EVERY
        frame different (--step reaches the sampler; R36's rho gate already differed on
        every frame); plus N=30 rho=2 != N=20 rho=2 (the two budgets are distinct runs).
  GATE3 (DIFFERS)   N=20 rho=3 != N=20 rho=2 (--blend_power reaches the bridge at N>15).

Exits non-zero if any gate fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r36_check_parity import FIVE_ROOT, compare  # noqa: E402


def arm_name(steps: int, rho: int) -> str:
    return f"r39_n{steps}_rho{rho}_vp"


def clip(root: Path, arm: str, c: dict) -> Path:
    return root / arm / f"edit{int(c['edit_type'])}" / c["video_name"]


def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.smoke_cases.read_text())
    failures = 0

    print("=== GATE1: N=15 rho=2 == R36 r36_rho2_vp ===")
    for c in cases:
        r = compare(clip(args.smoke_root, arm_name(15, 2), c),
                    clip(args.r36_root, "r36_rho2_vp", c))
        ok = r["identical"]
        failures += not ok
        print(f"{'IDENTICAL' if ok else 'GATE1-FAIL'} gate1 {c['video_name']}: {r}")

    print("=== GATE2: N=20 / N=30 rho=2 differ from N=15 rho=2 on every frame; N=30 != N=20 ===")
    for steps in (20, 30):
        for c in cases:
            r = compare(clip(args.smoke_root, arm_name(steps, 2), c),
                        clip(args.smoke_root, arm_name(15, 2), c))
            ok = (not r["missing"] and r["n_frames"] == r["n_frames_ref"]
                  and r["n_diff_frames"] == r["n_frames"])
            failures += not ok
            print(f"{'DIFFERS' if ok else 'GATE2-FAIL'} gate2 n{steps}-vs-n15 {c['video_name']}: {r}")
    for c in cases:
        r = compare(clip(args.smoke_root, arm_name(30, 2), c),
                    clip(args.smoke_root, arm_name(20, 2), c))
        ok = not r["missing"] and not r["identical"]
        failures += not ok
        print(f"{'DIFFERS' if ok else 'GATE2-FAIL'} gate2 n30-vs-n20 {c['video_name']}: {r}")

    print("=== GATE3: N=20 rho=3 != N=20 rho=2 ===")
    for c in cases:
        r = compare(clip(args.smoke_root, arm_name(20, 3), c),
                    clip(args.smoke_root, arm_name(20, 2), c))
        ok = not r["missing"] and not r["identical"]
        failures += not ok
        print(f"{'DIFFERS' if ok else 'GATE3-FAIL'} gate3 {c['video_name']}: {r}")

    print(f"[r39_check_smoke] failures: {failures}")
    return int(failures > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke_root", type=Path, default=FIVE_ROOT / "r39_smoke")
    p.add_argument("--smoke_cases", type=Path, default=Path("evaluation/r36_smoke_cases.json"))
    p.add_argument("--r36_root", type=Path, default=FIVE_ROOT / "r36_rho_sweep")
    args = p.parse_args(argv)
    for k in ("smoke_root", "r36_root"):
        setattr(args, k, getattr(args, k).expanduser())
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
