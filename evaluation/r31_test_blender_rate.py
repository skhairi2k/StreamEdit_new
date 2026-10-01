"""Regression test for the VECTORIZED ``blender_rate_from_rho`` (R31 `vectorize-rho`).

Run directly::

    python evaluation/r31_test_blender_rate.py

WHY THIS EXISTS
---------------
``pipeline/utils.py:blender_rate_from_rho`` used to evaluate ``t ** rho`` once per
DISTINCT exponent with Python's ``**``, so R26's ``tau_bg == tau_fg`` control reproduced
the scalar Eq. 4 path bit-for-bit. R31's field is continuous (up to 21 * 1560 = 32,760
distinct exponents), which made that loop cost ~2.7 s per call, so it was replaced with a
single ``torch.pow`` in float64.

The replacement is only legitimate while it stays BIT-IDENTICAL to the Python-float
computation. That is not guaranteed by the language -- it is a property of the current
torch/libm on this architecture, and a future torch, a different GPU arch, or a fast-math
build could break it silently, changing every R26 arm's numbers with no error anywhere.
This test is the tripwire.

⚠️ IT CHECKS BOTH RETURNED TENSORS, AND THE COMPLEMENT IS THE SUBTLE ONE. The scalar
path builds ``blender_rate = 1 - t ** blend_power`` (causal_model.py:365) and then takes
``1 - blender_rate`` (causal_model.py:452), so the complement is ``1 - (1 - w_src)``,
NOT ``w_src``. Those diverge once ``w_src`` falls below ~4e-9 -- which ``t ** rho`` with
rho up to 50 reaches constantly. Measured at the time of the change: substituting
``w_src`` altered 33.1% of complement values. A "simplification" to ``w_src`` must fail
this test, so the test covers exponents large enough to reach that regime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluation"))
from r30_b_map import t_next_schedule  # noqa: E402

# Load pipeline/utils.py BY PATH: `import pipeline.utils` would run pipeline/__init__.py,
# which pulls in every inference pipeline (and their model deps) for one small function.
_UTILS_PATH = _REPO / "Self-Forcing_StreamEdit" / "pipeline" / "utils.py"
_spec = importlib.util.spec_from_file_location("_sf_pipeline_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
blender_rate_from_rho = _utils.blender_rate_from_rho

# R26's exact exponent grid -- the arms whose stored numbers must not move.
R26_EXPONENTS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0)
# R31's calibrated endpoints.
R31_TAU_MIN, R31_TAU_MAX = 2.0, 50.0
STEP, FLOW_SHIFT = 15, 1.0


def reference(rho: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ORIGINAL per-distinct-exponent Python-float implementation, verbatim.

    Deliberately slow and deliberately not refactored: it is the specification.
    """
    rate = torch.empty_like(rho, dtype=torch.float32)
    comp = torch.empty_like(rho, dtype=torch.float32)
    for r in torch.unique(rho).tolist():
        w_src = 1.0 if r == 0.0 else t ** r
        sel = (rho == r)
        rate[sel] = float(1.0 - w_src)
        comp[sel] = float(1.0 - (1.0 - w_src))
    return rate, comp


def build_exponents(device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.cat([
        torch.tensor(R26_EXPONENTS, dtype=torch.float64),
        torch.rand(5000, generator=g).double() * R31_TAU_MAX,       # continuous
        torch.linspace(R31_TAU_MIN, R31_TAU_MAX, 2000).double(),    # R31's real range
    ]).to(device)


def check_bit_identical(device: str) -> None:
    t_next = t_next_schedule(STEP, FLOW_SHIFT)
    rho = build_exponents(device)
    n = len(t_next) * rho.numel()
    bad_rate = bad_comp = 0
    for t in t_next:
        r_ref, c_ref = reference(rho, float(t))
        r_got, c_got = blender_rate_from_rho(rho, float(t))
        bad_rate += int((r_ref != r_got).sum())
        bad_comp += int((c_ref != c_got).sum())
    print(f"[{device}] {len(t_next)} t_next x {rho.numel()} exponents = {n:,} pairs: "
          f"rate {bad_rate} diffs, comp {bad_comp} diffs")
    if bad_rate or bad_comp:
        raise AssertionError(
            f"[{device}] blender_rate_from_rho is NO LONGER bit-identical to the "
            f"Python-float path ({bad_rate} rate, {bad_comp} comp differences out of "
            f"{n:,}). Every R26 arm's stored numbers depend on this. Do not ship."
        )


def check_complement_is_not_w_src(device: str) -> None:
    """Guard the specific 'simplification' that would pass a naive eyeball check."""
    t_next = t_next_schedule(STEP, FLOW_SHIFT)
    rho = torch.tensor([20.0, 50.0], dtype=torch.float64, device=device)
    found = False
    for t in t_next:
        if float(t) <= 0.0:
            continue
        _, comp = blender_rate_from_rho(rho, float(t))
        w_src = torch.pow(torch.as_tensor(float(t), dtype=torch.float64,
                                          device=device), rho).float()
        if bool((comp != w_src).any()):
            found = True
            break
    if not found:
        raise AssertionError(
            f"[{device}] complement appears to equal w_src everywhere. Either the "
            "exponent range no longer reaches w_src < ~4e-9, or the implementation was "
            "'simplified' to return w_src -- which breaks the tau_bg == tau_fg control."
        )
    print(f"[{device}] complement correctly differs from w_src in the small-w_src regime")


def check_guards(device: str) -> None:
    rho = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
    try:
        blender_rate_from_rho(rho, 0.5)
    except ValueError:
        print(f"[{device}] negative exponent rejected")
    else:
        raise AssertionError("a negative exponent must raise: W_src = t**rho diverges")
    try:
        blender_rate_from_rho(torch.ones(4, dtype=torch.float64, device=device), -0.1)
    except ValueError:
        print(f"[{device}] negative timestep_next rejected")
    else:
        raise AssertionError("a negative timestep_next must raise")

    # tau == 0 must pin the source at EVERY step, the final t == 0 included.
    z = torch.zeros(4, dtype=torch.float64, device=device)
    for t in (0.0, 0.5, 1.0):
        rate, comp = blender_rate_from_rho(z, t)
        if not (torch.all(rate == 0.0) and torch.all(comp == 1.0)):
            raise AssertionError(f"tau=0 at t={t} must give rate 0 / comp 1 "
                                 f"(source pinned), got {rate[0]} / {comp[0]}")
    print(f"[{device}] tau == 0 pins the source at t = 0, 0.5 and 1.0")


def main() -> int:
    print(f"testing {_UTILS_PATH}")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        check_bit_identical(dev)
        check_complement_is_not_w_src(dev)
        check_guards(dev)
    if "cuda" not in devices:
        print("[note] no CUDA visible here; CPU only. Re-run on a GPU node before "
              "trusting a torch upgrade.")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
