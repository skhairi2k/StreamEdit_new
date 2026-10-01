"""R31 stage 3, prep: turn each clip's normalised divergence field ``m`` into the
per-token exponent field ``rho`` that the StreamGVE Q/K blend consumes.

WHAT THIS COMPUTES
------------------
StreamGVE Eq. 4 releases the source with ``W_src(t_i) = t_i ** rho``. R26 gave every
token one of two exponents (background / foreground, from the grounding-mask union);
R31 gives every token its OWN exponent, driven by how much that token diverged.

  SEMANTICS: high divergence => the token changed => release the source FAST => HIGH rho.
             m = 0 -> tau_min,  m = 1 -> tau_max.

The mapping is BUDGET-LINEAR, not linear in the exponent. Define the discrete source
budget actually spent over the real denoising grid:

    A_disc(b) = mean_i t_i ** b          (t_i = the pipeline's t_next values)

and interpolate the BUDGET linearly, then invert:

    A_m  = (1 - m) * A_disc(tau_min) + m * A_disc(tau_max)
    rho  = A_disc^-1(A_m)

Interpolating the exponent instead (``tau_min + (tau_max - tau_min) * m``, R26's formula
with a continuous m) makes the injected source wildly non-linear in m: R30 measured the
fictitious mass this hides as 34% at b=5 and 97.1% at b=50. Budget-linear is what makes
"m halfway" mean "half the source budget".

⚠️ ``--step`` HERE IS STAGE 3's STEP COUNT, NOT STAGE 1's
---------------------------------------------------------
Easy to get wrong since 2026-09-15, when stage 1 became a 7-step run. A_disc is defined
over the grid the BLEND runs on, which is stage 3's rollout: **15 steps**. Stage 1's 7
steps produced the divergence field and have nothing to do with this schedule. The npz
records ``step`` and ``flow_shift`` precisely because A_disc is schedule-dependent --
at --step 15, A_disc(2) = 0.300 and A_disc(50) = 0.0021, and those are the endpoints
R30 calibrated tau_min/tau_max against.

WHAT IS REUSED, AND THE ONE THING THAT IS NEW
---------------------------------------------
``t_next_schedule`` / ``a_disc`` / ``invert_a_disc`` are imported VERBATIM from
``r30_b_map.py`` -- they already reproduce the pipeline's t_next grid exactly (arange ->
flow-shift warp -> shift by one) and already invert A_disc by bisection. The only new
thing here is VECTORIZATION: R30 inverted one scalar per clip, R31 needs one exponent per
token, i.e. [F_lat, 1560] ~ 32k inversions. Bisecting the whole array at once is
milliseconds; looping ``budget_linear_b`` would be ~32k separate 200-iteration bisections
per clip. ``--self_test`` asserts the vectorized result matches the scalar r30 function.

OUTPUTS
-------
  {out}/edit{T}/{video_name}.npz
      rho   float32 [F_lat, 1560]   the exponent field, ready for `rho_frames`
      plus step / flow_shift / tau_min / tau_max / grid shape / provenance.

``--mapping linear_threshold`` (added 2026-09-15, EXPLORATORY -- not R30-calibrated)
-------------------------------------------------------------------------------------
User-requested alternative to budget-linear, for comparison only: ``tau = 0`` for
``m < thresh`` (default 0.4, i.e. the background is FULLY pinned to source below the
threshold -- no source release at all, not just a small one), then LINEAR IN THE
EXPONENT from 0 to ``tau_max`` as ``m`` goes ``thresh -> 1``:

    tau(m) = 0                                    if m <  thresh
    tau(m) = tau_max * (m - thresh) / (1 - thresh) if m >= thresh

This is deliberately NOT budget-linear -- the whole point of the comparison is to see
what the OLD R26-style "linear in the exponent" mapping (the one this module's docstring
above says hides 34-97% fictitious released mass) looks like next to the calibrated one.
``--tau_min`` is not used in this mode: the floor is 0 by construction, matching the
user's spec verbatim, not R30's calibrated ``tau_min=2``.

RECREATED 2026-09-15 after this file was found deleted from disk by an external process
(root cause unknown, no destructive command in any tracked shell history) -- recreated
verbatim from conversation context. Its already-run outputs (r31_rho/, r31_rho_lin_t0.4/)
were untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_b_map import (  # noqa: E402
    a_disc,
    budget_linear_b,
    invert_a_disc,
    t_next_schedule,
)

# The four R31 divergence arms (depth dropped 2026-09-11 -- it needs an affine alignment
# R31 has no mask to fit). Kept in sync with r31_divergence.py:ARMS.
ARMS: Tuple[str, ...] = ("lpips", "dino_patch", "normals", "latent")

GRID_H: int = 30
GRID_W: int = 52
FRAME_SEQ_LENGTH: int = GRID_H * GRID_W          # 1560 == pipeline's frame_seq_length

# R30's calibrated pair, valid because stage 3 stayed at 15 steps.
DEFAULT_TAU_MIN: float = 2.0
DEFAULT_TAU_MAX: float = 50.0

# Stage 3's rollout length -- NOT stage 1's. See the module docstring.
DEFAULT_STAGE3_STEP: int = 15
DEFAULT_FLOW_SHIFT: float = 1.0


def a_disc_vec(b: np.ndarray, t_next: np.ndarray) -> np.ndarray:
    """``A_disc(b) = mean_i t_i ** b``, evaluated elementwise over an array of exponents.

    The scalar twin is ``r30_b_map.a_disc``; this broadcasts ``t_next`` against an
    arbitrarily shaped ``b``. float64 throughout -- the tail of the grid is ~1e-15 and
    float32 would flatten it, which would make the bisection terminate on noise.
    """
    return np.mean(np.power(t_next.reshape(*([1] * b.ndim), -1),
                            b[..., None]), axis=-1)


def budget_linear_tau(m: np.ndarray, t_next: np.ndarray,
                      tau_min: float, tau_max: float,
                      max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Vectorized budget-linear ``m -> rho`` over a whole ``[F_lat, 1560]`` field.

    ``A_disc`` is strictly decreasing in the exponent, so for ``m`` in [0, 1] the target
    budget ``A_m`` lies in ``[A_disc(tau_max), A_disc(tau_min)]`` and the root is bracketed
    by ``[tau_min, tau_max]`` by construction -- no bracket search is needed, unlike
    ``invert_a_disc``'s generic ``B_BRACKET``.

    Args:
        m: normalised divergence in [0, 1], any shape.
        t_next: the denoising grid the BLEND runs on (stage 3's, not stage 1's).
        tau_min: exponent at m = 0 (least source release).
        tau_max: exponent at m = 1 (most source release).

    Returns:
        float64 array of exponents, same shape as ``m``, with the endpoints exact.
    """
    if tau_max <= tau_min:
        raise ValueError(f"tau_max must exceed tau_min, got {tau_min} / {tau_max}")
    m = np.asarray(m, dtype=np.float64)
    if np.isnan(m).any():
        raise ValueError("m contains NaN -- refusing to route a gate on it")
    # Clip rather than raise: r31_divergence writes m in [0,1] by construction, and a
    # float32->float64 round-trip can land a hair outside.
    m = np.clip(m, 0.0, 1.0)

    a_lo = a_disc(float(tau_min), t_next)        # budget at tau_min (the LARGER budget)
    a_hi = a_disc(float(tau_max), t_next)        # budget at tau_max (the smaller)
    a_m = (1.0 - m) * a_lo + m * a_hi

    lo = np.full(m.shape, float(tau_min), dtype=np.float64)
    hi = np.full(m.shape, float(tau_max), dtype=np.float64)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # A_disc decreasing: too much budget still spent => the exponent must go UP.
        too_much = a_disc_vec(mid, t_next) > a_m
        lo = np.where(too_much, mid, lo)
        hi = np.where(too_much, hi, mid)
        if np.all(hi - lo < tol * np.maximum(1.0, hi)):
            break
    rho = 0.5 * (lo + hi)

    # Pin the endpoints exactly -- bisection lands within tol, but m == 0 and m == 1 are
    # the two values a reader will check by hand against the calibration.
    rho[m <= 0.0] = float(tau_min)
    rho[m >= 1.0] = float(tau_max)
    return rho


def linear_threshold_tau(m: np.ndarray, thresh: float, tau_max: float) -> np.ndarray:
    """EXPLORATORY alternative mapping, user-requested 2026-09-15 -- see module docstring.

    ``tau = 0`` below ``thresh``, then LINEAR IN THE EXPONENT (not budget-linear) up to
    ``tau_max`` at ``m = 1``. No schedule dependence at all (no ``t_next``, no A_disc) --
    it is a pure piecewise-linear function of ``m``, which is exactly what makes it the
    thing to contrast against the calibrated budget-linear curve above.
    """
    if not (0.0 <= thresh < 1.0):
        raise ValueError(f"thresh must be in [0, 1), got {thresh}")
    m = np.clip(np.asarray(m, dtype=np.float64), 0.0, 1.0)
    return tau_max * np.clip((m - thresh) / (1.0 - thresh), 0.0, 1.0)


def self_test(step: int, flow_shift: float,
              tau_min: float, tau_max: float) -> None:
    """Assert the vectorized inversion matches r30's scalar one, and print the curve.

    Cheap, and the guard against the bisection's monotonicity direction being flipped --
    which would route HIGH tau at LOW divergence and silently inverse the whole gate.
    """
    t_next = t_next_schedule(step, flow_shift)
    probes = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    vec = budget_linear_tau(probes, t_next, tau_min, tau_max)
    ref = np.array([budget_linear_b(float(x), t_next, tau_min, tau_max) for x in probes])
    err = np.abs(vec - ref).max()
    print(f"[self_test] step={step} flow_shift={flow_shift} "
          f"tau=[{tau_min}, {tau_max}]")
    print(f"[self_test] A_disc({tau_min}) = {a_disc(tau_min, t_next):.6f}   "
          f"A_disc({tau_max}) = {a_disc(tau_max, t_next):.6f}")
    for x, v in zip(probes, vec):
        print(f"[self_test]   m={x:.2f} -> tau={v:8.4f}  "
              f"(normalised {(v - tau_min) / (tau_max - tau_min):.3f})")
    if err > 1e-6:
        raise AssertionError(f"vectorized != r30 scalar budget_linear_b, max |delta| "
                             f"= {err:.3g}")
    print(f"[self_test] PASS: matches r30_b_map.budget_linear_b to {err:.3g}")

    # Monotonicity is what makes the gate mean what the plan says it means.
    if not np.all(np.diff(vec) >= 0.0):
        raise AssertionError("tau is not monotone increasing in m -- the gate is inverted")
    print("[self_test] PASS: tau monotone increasing in m "
          "(high divergence -> high tau -> fast source release)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--div_root", type=Path,
                   help="One ARM's stage-2 output, i.e. .../r31_div/{arm}, containing "
                        "edit{T}/{video}.npz with the `m` field.")
    p.add_argument("-o", "--out", type=Path,
                   help="Destination root; writes {out}/edit{T}/{video}.npz")
    p.add_argument("--tau_min", type=float, default=DEFAULT_TAU_MIN,
                   help="Exponent at m = 0. Default 2 (R30's calibrated floor). "
                        "0 pins the background to the source for the whole rollout -- "
                        "R26's best-scoring AND most artefact-prone arm, the aggressive "
                        "alternative, not the default.")
    p.add_argument("--tau_max", type=float, default=DEFAULT_TAU_MAX,
                   help="Exponent at m = 1. Default 50 (R30's calibrated ceiling).")
    p.add_argument("--mapping", choices=("budget_linear", "linear_threshold"),
                   default="budget_linear",
                   help="Default: R30's calibrated budget-linear inversion (unchanged "
                        "behavior). 'linear_threshold' is a 2026-09-15 EXPLORATORY "
                        "alternative for comparison only -- see module docstring: "
                        "tau=0 below --thresh, then linear IN THE EXPONENT to tau_max.")
    p.add_argument("--thresh", type=float, default=0.4,
                   help="Only used by --mapping linear_threshold: m below this routes "
                        "tau=0 (source fully pinned). Default 0.4.")
    p.add_argument("--step", type=int, default=DEFAULT_STAGE3_STEP,
                   help="STAGE 3's denoising step count -- the grid the blend runs on. "
                        "NOT stage 1's 7. A_disc is schedule-dependent, so changing this "
                        "invalidates R30's tau endpoints.")
    p.add_argument("--flow_shift", type=float, default=DEFAULT_FLOW_SHIFT)
    p.add_argument("--self_test", action="store_true",
                   help="Check the vectorized inversion against r30_b_map's scalar one, "
                        "print the m -> tau curve, and exit.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        self_test(args.step, args.flow_shift, args.tau_min, args.tau_max)
        return 0
    if args.div_root is None or args.out is None:
        raise SystemExit("[r31_rho_map] --div_root and -o are required "
                         "(or pass --self_test)")

    div_root = args.div_root.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    if not div_root.is_dir():
        raise SystemExit(f"[r31_rho_map] no such --div_root: {div_root}")

    t_next = t_next_schedule(args.step, args.flow_shift)
    linear = args.mapping == "linear_threshold"
    if linear:
        # No A_disc/schedule dependence at all in this mode -- see module docstring.
        eff_tau_min = 0.0
        a_lo = a_hi = float("nan")
        tau_half = float(linear_threshold_tau(np.array([0.5]), args.thresh, args.tau_max)[0])
        print(f"[r31_rho_map] div_root={div_root}")
        print(f"[r31_rho_map] EXPLORATORY mapping=linear_threshold thresh={args.thresh} "
              f"tau_max={args.tau_max} (tau_min forced to 0, --tau_min ignored)")
        print(f"[r31_rho_map] m=0.5 -> tau={tau_half:.4f} "
              f"(sanity: m<thresh -> 0, m=1 -> {args.tau_max})", flush=True)
    else:
        eff_tau_min = args.tau_min
        a_lo, a_hi = a_disc(args.tau_min, t_next), a_disc(args.tau_max, t_next)
        tau_half = float(budget_linear_tau(np.array([0.5]), t_next,
                                           args.tau_min, args.tau_max)[0])
        print(f"[r31_rho_map] div_root={div_root}")
        print(f"[r31_rho_map] STAGE-3 schedule: step={args.step} flow_shift={args.flow_shift} "
              f"(stage 1's 7 steps are irrelevant here)")
        print(f"[r31_rho_map] tau=[{args.tau_min}, {args.tau_max}] => "
              f"A_disc=[{a_lo:.6f}, {a_hi:.6f}]", flush=True)

    files = sorted(div_root.glob("edit*/*.npz"))
    if not files:
        raise SystemExit(f"[r31_rho_map] no edit*/*.npz under {div_root}")

    n_ok = 0
    for f in files:
        edit_dir = f.parent.name
        try:
            z = np.load(f)
            m = z["m"]
            gh, gw = int(z["grid_h"]), int(z["grid_w"])
            if (gh, gw) != (GRID_H, GRID_W):
                raise ValueError(f"grid {gh}x{gw} != expected {GRID_H}x{GRID_W}")
            if m.ndim != 2 or m.shape[1] != FRAME_SEQ_LENGTH:
                raise ValueError(f"m has shape {m.shape}, expected "
                                 f"[F_lat, {FRAME_SEQ_LENGTH}]")

            d_min, d_max = float(z["d_min"]), float(z["d_max"])
            # Per-clip min/max normalisation's one sharp edge: a clip that never really
            # diverged still gets a full-range m, i.e. pure noise stretched over the whole
            # tau span. r31_divergence emits m == 0 when the span collapses; say so loudly
            # either way, because check-stage2 gates on exactly this.
            if d_max - d_min <= 1e-12:
                print(f"[WARN] {edit_dir}/{f.stem}: d_min == d_max == {d_min:.6g} -- no "
                      f"divergence to route; rho is uniformly {eff_tau_min:g}.", flush=True)

            if linear:
                rho = linear_threshold_tau(m, args.thresh, args.tau_max)
            else:
                rho = budget_linear_tau(m, t_next, args.tau_min, args.tau_max)

            out = out_root / edit_dir / f.name
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                rho=rho.astype(np.float32),
                # A_disc is schedule-dependent: a rho field is only meaningful against
                # the schedule it was inverted on, so both are recorded (NaN for
                # linear_threshold, which has no schedule dependence at all).
                mapping=np.str_(args.mapping),
                thresh=np.float32(args.thresh if linear else float("nan")),
                tau_at_m_half=np.float32(tau_half),
                step=np.int32(args.step),
                flow_shift=np.float32(args.flow_shift),
                tau_min=np.float32(eff_tau_min),
                tau_max=np.float32(args.tau_max),
                a_disc_tau_min=np.float64(a_lo),
                a_disc_tau_max=np.float64(a_hi),
                grid_h=np.int32(gh), grid_w=np.int32(gw),
                n_latent_frames=np.int32(m.shape[0]),
                frame_seq_length=np.int32(FRAME_SEQ_LENGTH),
                d_min=np.float64(d_min), d_max=np.float64(d_max),
                arm=z["arm"], video_name=z["video_name"],
                edit_type=z["edit_type"],
                # Stage 1's gating regime, carried the whole way through: at --step 7 the
                # divergence was measured POST-injection, so a localised field is partly
                # mask-derived. The verdict step must be able to see this.
                measure_index=z["measure_index"],
                measure_kind=(z["measure_kind"] if "measure_kind" in z.files
                              else np.str_("unknown__pre_2026_09_15")),
            )
            print(f"[ok] {edit_dir}/{f.stem}: rho {rho.shape} "
                  f"range [{rho.min():.3f}, {rho.max():.3f}] "
                  f"mean {rho.mean():.3f} -> {out}", flush=True)
            n_ok += 1
        except Exception as ex:          # record and continue -- never swallow silently
            print(f"[ERROR] {edit_dir}/{f.stem}: {type(ex).__name__}: {ex}", flush=True)

    print(f"[r31_rho_map] done: {n_ok}/{len(files)} ok -> {out_root}", flush=True)
    return 0 if n_ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
