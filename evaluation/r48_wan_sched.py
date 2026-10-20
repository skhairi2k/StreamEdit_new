"""R48 Wan-Edit sweep configuration: which (sample_steps, skip_timesteps) each baseline point uses.

Wan-Edit's sigma at step i of an N-step schedule (shift 5) is  sigma = shift*t / (1 + (shift-1)*t),  t = (N-i)/N
(``FlowUniPCMultistepScheduler.set_timesteps``: linspace(1, 0, N+1)[:-1] pushed through the shift). Inverting it,
a target start sigma sigma_g sits at  t = sigma_g / (shift - (shift-1)*sigma_g).

NATIVE curve  (N = 50 fixed, NFE varies per point): skip = round(N * (1 - t)).
MATCHED curve (same number A of ACTIVE steps for every point): active steps = N - skip, so t = A/N exactly, hence
               N = round(A / t) and skip = N - A. A = 55 is the adaptive run's worst case (14 draft + 41 final).
               N blows up at low sigma_g (A/N must shrink), so the matched curve is restricted to {0.95, 0.92, 0.85}
               (N = 69 / 79 / 104). 0.75 / 0.6 / 0.4 would need N = 147 / 234 / 459.

Config ids (what r48_wan_render.py takes as --config):  native_s{sigma_g}, matched_s{sigma_g}, adapt_{arm}.
Every table value is checked against the REAL scheduler class, not just the formula (``--selftest``, also run on import
by r48_wan_render.py).
usage: python evaluation/r48_wan_sched.py            # print the tables, assert them
       python evaluation/r48_wan_sched.py --list     # baseline config ids, one per line (for the sbatch script)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SHIFT = 5.0
SIGMA_GRID = (0.95, 0.92, 0.85, 0.75, 0.6, 0.4)          # native curve
MATCHED_GRID = (0.95, 0.92, 0.85)                        # matched curve (subset of the above)
N_NATIVE = 50
A_ACTIVE = 55                                            # adaptive worst case: 14 draft + 41 final
ADAPT_SKIP = 9                                           # E = 1 lands on step 9 of the 50-step grid
ARMS = ("dino_SD", "lpips_SD", "dino_D", "S")
# R48 v2a: two global starts EARLIER than the draft's step 9, on the native 50-step grid (does freeing more help?).
# Kept out of configs() on purpose: configs() is the fixed 9-unit list r48_units.py and the full-run arrays index into.
EXPLORE = {"native_s0.978": (N_NATIVE, 5), "native_s1": (N_NATIVE, 0)}


def sigma_at(i: int, n: int, shift: float = SHIFT) -> float:
    t = (n - i) / n
    return shift * t / (1 + (shift - 1) * t)


def t_of(sigma_g: float, shift: float = SHIFT) -> float:
    return sigma_g / (shift - (shift - 1) * sigma_g)


def native_skip(sigma_g: float, n: int = N_NATIVE) -> int:
    return int(round(n * (1 - t_of(sigma_g))))


def matched_n(sigma_g: float, active: int = A_ACTIVE) -> Tuple[int, int]:
    """(sample_steps N, skip) such that exactly ``active`` steps run and the start sits at sigma_g."""
    n = int(round(active / t_of(sigma_g)))
    return n, n - active


def configs() -> Dict[str, Tuple[int, int]]:
    """config id -> (sample_steps, skip_timesteps) for the 6 native + 3 matched baseline points."""
    out: Dict[str, Tuple[int, int]] = {}
    for sg in SIGMA_GRID:
        out[f"native_s{sg:g}"] = (N_NATIVE, native_skip(sg))
    for sg in MATCHED_GRID:
        out[f"matched_s{sg:g}"] = matched_n(sg)
    return out


def explore_configs() -> Dict[str, Tuple[int, int]]:
    """config id -> (sample_steps, skip_timesteps) for the R48 v2a global starts above step 9 (not baseline units)."""
    return dict(EXPLORE)


def real_sigmas(n: int) -> np.ndarray:
    """The host scheduler's own sigma array for an n-step schedule."""
    here = Path(__file__).resolve().parent.parent / "wan-edit"
    sys.path.insert(0, str(here))
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    s = FlowUniPCMultistepScheduler(shift=1, use_dynamic_shifting=False)
    s.set_timesteps(n, shift=SHIFT)
    return s.sigmas.numpy()[:n]


def selftest(verbose: bool = True) -> None:
    cfg = configs()
    real50 = real_sigmas(N_NATIVE)
    assert abs(real50[9] - 0.958) < 1e-3 and abs(real50[49] - 0.093) < 1e-3, (real50[9], real50[49])
    expect_native = {0.95: 10, 0.92: 15, 0.85: 23, 0.75: 31, 0.6: 38, 0.4: 44}
    for sg, k in expect_native.items():
        assert native_skip(sg) == k, (sg, native_skip(sg), k)
    expect_matched = {0.95: 69, 0.92: 79, 0.85: 104}
    if verbose:
        print(f"{'config':16s} {'sample_steps':>12s} {'skip':>5s} {'active':>6s} {'sigma_g':>8s} {'real sigma(skip)':>17s}")
    for cid, (n, skip) in cfg.items():
        sg = float(cid.split("_s")[1])
        real = real_sigmas(n)
        assert abs(real[skip] - sigma_at(skip, n)) < 1e-3, (cid, real[skip], sigma_at(skip, n))
        assert abs(real[skip] - sg) < 0.015, f"{cid}: starts at {real[skip]:.3f}, wanted {sg}"
        if cid.startswith("matched"):
            assert n - skip == A_ACTIVE and n == expect_matched[sg], (cid, n, skip)
        if verbose:
            print(f"{cid:16s} {n:12d} {skip:5d} {n - skip:6d} {sg:8.2f} {real[skip]:17.4f}")
    # E = 1 of the adaptive mapping (sigma 0.96) must land on the step used as ADAPT_SKIP
    assert int(np.abs(real50 - 0.96).argmin()) == ADAPT_SKIP
    # v2a explore starts: step 5 = sigma 0.978, step 0 = sigma 1.0 (pure noise) on the real 50-step scheduler
    for cid, (n, skip), want in (("native_s0.978", EXPLORE["native_s0.978"], 0.978), ("native_s1", EXPLORE["native_s1"], 1.0)):
        assert n == N_NATIVE and abs(real50[skip] - want) < 1e-3, (cid, skip, real50[skip])
        if verbose:
            print(f"{cid:16s} {n:12d} {skip:5d} {n - skip:6d} {want:8.3f} {real50[skip]:17.4f}  (v2a explore)")
    if verbose:
        print("[r48_wan_sched] selftest PASS (sigma(9)=0.958, sigma(49)=0.093, native skips 10/15/23/31/38/44, "
              "matched N 69/79/104 at 55 active steps, 0.96 -> step 9)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the baseline config ids, one per line")
    ap.add_argument("--list_explore", action="store_true", help="print the v2a explore config ids, one per line")
    a = ap.parse_args()
    if a.list:
        print("\n".join(configs()))
        return 0
    if a.list_explore:
        print("\n".join(explore_configs()))
        return 0
    selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
