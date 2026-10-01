#!/usr/bin/env python
"""R30 stage 1 -- map one arm's measured divergence `d` onto the blend rate `b`.

`b` IS `blend_power`, read as ``1 - t_next ** blend_power`` at causal_model.py:334, so
the SOURCE weight at a step is W_src(t) = t ** b. Because t in (0, 1), t ** b DECREASES
in b: a small b holds the source anchor, a large b releases it early.

    small divergence (the anchor barely moved) -> b_min: source held longest
    large divergence (the anchor moved a lot)  -> b_max: source released early

THE MAP IS LINEAR IN THE INJECTED SOURCE BUDGET, NOT IN b
----------------------------------------------------------
b is an exponent and its effect saturates, so interpolating b linearly would put a
mid-divergence clip at b = 26, which spends almost no source anchoring at all -- "mid" on
the b axis is already near-full detachment. Interpolate the BUDGET instead and invert.

    A_disc(b) = (1/n) * sum_i t_i ** b     over the ACTUAL t_next grid
    A_m       = (1 - m) * A_disc(b_min) + m * A_disc(b_max)
    b         = A_disc^{-1}(A_m)           by bisection (A_disc is strictly decreasing)

WHY THE DISCRETE SUM AND NOT THE CLOSED FORM 1/(1+b)
-----------------------------------------------------
The closed form A(b) = integral of t**b over [0, 1] = 1/(1+b) is the budget of a grid
spanning t in [0, 1]. The real grid is `timestep_next` = [0.933, 0.866, ..., 0.066, 0.0]
at --step 15 and TOPS OUT AT t = 0.933, never reaching 1. As b grows, t**b concentrates
near t = 1, so the integral collects most of its mass in a sliver of t the schedule never
visits. Fraction of 1/(1+b) lying in t > 0.933:

    b = 1  -> 13%      b = 5  -> 34%      b = 20 -> 77%      b = 50 -> 97.1%

At the top of the range 97% of the reported budget is fictitious -- ask for 2% of full
source, get 0.21%. The fatal part is not the absolute error but that IT GROWS WITH b, so
constructing b to make 1/(1+b) linear in d leaves the ACTUALLY INJECTED source non-linear
in d, which is the single property budget-linearity exists to provide. Symptom under the
closed form: delivered budget near-linear for d in [0.1, 0.9], then a 40x drop in the last
decile. A_disc on the real grid is linear in d by construction at any --step.

A_disc IS SCHEDULE-DEPENDENT, so `step` and `flow_shift` are written into every row and
into the sidecar .meta.json. A map calibrated at --step 15 silently changes meaning at any
other step count. RECOMPUTE THE ENDPOINTS PER STEP COUNT: at --step 15, A_max = A_disc(2)
= 0.300 and A_min = A_disc(50) = 0.0021; at --step 7 they are 0.265 and 1e-4.

WHY b_min = 2 AND NOT 0
------------------------
b_min = 0 is unsafe: t ** 0 = 1 at every step (numpy's 0 ** 0 = 1 included), so
blender_rate = 0 is PURE SOURCE (pipeline/utils.py:18) and the foreground receives no edit
at all. Combined with min/max normalisation pinning the argmin clip to exactly d = 0, the
least-divergent clip in the dataset would be guaranteed to produce zero edit. b_min = 2 at
--step 15 leaves W_src = [0.87, 0.75, 0.64, 0.54, 0.44, ...], crossing 0.5 near step 4 of
15 -- substantial but finite anchoring for the smallest edit.

CONSEQUENCE TO CHECK, and it is the reason check-div gate 5 exists. Raising the floor from
b = 0 moved A_max from 1.00 to 0.300, compressing the total budget range from ~500x to
~14x. Every clip now lands between MODERATE and LITTLE anchoring. If that band is too
narrow to produce visibly different outputs, no divergence measure can matter and all
eight arms converge for reasons unrelated to the measures.

NORMALISATION IS AGAINST THE DATASET MIN/MAX, with the argmin/argmax video recorded and
the raw un-normalised d stored per clip, so switching to percentile anchors later is a CSV
operation rather than a rerun. Known consequence, accepted at planning: the argmin clip is
pinned to b_min and the argmax to b_max by construction.

The emitted CSV keeps R25's (video_name, edit_type, iou, tau) columns FIRST and unchanged,
so run_fivebench.py --tau_map needs no new flag -- the `tau` column carries b and the
`iou` column carries the raw d. Extra diagnostic columns follow; csv.DictReader ignores
them. Provenance goes in those columns and in a sidecar JSON rather than a leading `#`
comment, because run_fivebench.py:256 reads the file with a bare csv.DictReader and would
take a comment line as the header row.

Usage
-----
    python evaluation/r30_b_map.py --arm lpips \\
        --divergence evaluation/csv/r30_divergence.csv \\
        --b_min 2 --b_max 50 --step 15 \\
        -o evaluation/csv/r30_b_map_lpips.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# The eight arms, matching r30_divergence.py.
ARMS: Tuple[str, ...] = ("lpips", "dino_cls", "dino_patch", "clip_image",
                         "clip_prompt", "depth", "normals", "selfsim")

# Bisection bracket. A_disc is strictly decreasing, and A_disc(500) at --step 15 is ~1e-15,
# far below any A_m reachable with b_max <= 100, so the root is always bracketed.
B_BRACKET: Tuple[float, float] = (1e-6, 500.0)

# The Eq. 4 default exponent, printed alongside each row so the table shows at a glance
# which direction each case was moved relative to the baseline arm.
EQ4_BLEND_POWER: float = 2.0


# --------------------------------------------------------------------------------------
# the denoising schedule -- reproduced EXACTLY as the pipeline builds it
# --------------------------------------------------------------------------------------
def t_next_schedule(step: int, flow_shift: float = 1.0) -> np.ndarray:
    """The t_next values the blend actually sees, one per denoising step.

    Mirrors three places in the pipeline and must stay in sync with all three:

      inference_edit_streamedit.py:68
          config['denoising_step_list'] = np.arange(1000, 0, -1000 / step).astype(int)
      edit_causal_inference.py:37-39   (warp_denoising_step: true in the dmd config)
          timesteps = cat(scheduler.timesteps, [0]); dsl = timesteps[1000 - dsl]
      edit_causal_inference.py:721
          timestep_next = dsl[index + 1] / 1000  if index < len-1  else 0

    The scheduler is FlowMatchScheduler(shift=flow_shift, sigma_min=0.0,
    extra_one_step=True) with set_timesteps(1000, ...) -- wan_wrapper.py:131-133 -- so
    sigmas = linspace(1, 0, 1001)[:-1] pushed through the flow shift.

    At flow_shift = 1.0 (the run_fivebench default, and what R21/R25/R26/R27 all use) the
    shift is the identity and the warp is a no-op, so t_next reduces to the plain
    arange / 1000. This function does NOT assume that -- it applies the shift -- so a map
    built at a different flow_shift is still correct rather than silently wrong.

    Kept byte-identical to r27_tau_map.py:t_next_schedule; the two must not drift.
    """
    if step < 2:
        raise ValueError(f"--step must be >= 2, got {step}")

    dsl = np.arange(1000, 0, -1000.0 / step).astype(int)

    sigmas = np.linspace(1.0, 0.0, 1001)[:-1]
    sigmas = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)
    timesteps = np.concatenate([sigmas * 1000.0, np.zeros(1)])
    warped = timesteps[1000 - dsl]

    t_next = np.empty(len(warped), dtype=np.float64)
    t_next[:-1] = warped[1:] / 1000.0
    t_next[-1] = 0.0
    return t_next


def a_disc(b: float, t_next: np.ndarray) -> float:
    """A_disc(b) = mean_i t_i ** b -- the source budget spent over the REAL grid.

    Strictly decreasing in b for t_i in [0, 1) with at least one t_i > 0, which is what
    makes it invertible. The final t_i is exactly 0, and 0 ** b is 0 for b > 0.
    """
    return float(np.mean(np.power(t_next, b)))


def invert_a_disc(target: float, t_next: np.ndarray,
                  bracket: Tuple[float, float] = B_BRACKET,
                  tol: float = 1e-12, max_iter: int = 200) -> float:
    """b such that A_disc(b) == target, by bisection on the decreasing A_disc.

    Bisection rather than Brent because monotonicity is guaranteed here and bisection
    cannot be thrown by a flat tail: A_disc is ~1e-15 over most of the upper bracket, and
    a secant step there would jump wildly.
    """
    lo, hi = bracket
    a_lo, a_hi = a_disc(lo, t_next), a_disc(hi, t_next)
    if not (a_hi <= target <= a_lo):
        raise ValueError(f"A_disc target {target:.6g} is outside the bracket "
                         f"[A({hi})={a_hi:.6g}, A({lo})={a_lo:.6g}]")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if a_disc(mid, t_next) > target:      # still too much budget -> raise b
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def budget_linear_b(m: float, t_next: np.ndarray,
                    b_min: float, b_max: float) -> float:
    """Interpolate the injected budget linearly in m, then invert for b."""
    if m <= 0.0:
        return float(b_min)
    if m >= 1.0:
        return float(b_max)
    a_m = (1.0 - m) * a_disc(b_min, t_next) + m * a_disc(b_max, t_next)
    return invert_a_disc(a_m, t_next)


# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", type=str, required=True, choices=list(ARMS),
                   help="Which divergence column to route on. Reads d_{arm}_{reduction} "
                        "from the divergence CSV.")
    p.add_argument("--divergence", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"),
                   help="Output of r30_divergence.py.")
    p.add_argument("--reduction", choices=("masked", "global"), default="masked",
                   help="Which reduction of the arm to route on. 'masked' (default) is "
                        "the one stage 2 renders: spatial selectivity is already carried "
                        "by the stage-2 mask, so the scalar b need only encode how much "
                        "per PIXEL, and loading area into b double-counts a factor the "
                        "mechanism already handles. 'global' exists for the diagnostic "
                        "comparison only and does not earn a rollout set.")
    p.add_argument("--b_min", type=float, default=2.0,
                   help="b at m = 0, for the LEAST divergent clip. NOT 0: t ** 0 = 1 at "
                        "every step, so blender_rate = 0 is pure source and the "
                        "foreground receives no edit at all -- and min/max normalisation "
                        "pins the argmin clip to exactly m = 0, so that clip would be "
                        "guaranteed a null edit. 2.0 leaves W_src crossing 0.5 near step "
                        "4 of 15: substantial but finite anchoring.")
    p.add_argument("--b_max", type=float, default=50.0,
                   help="b at m = 1, for the MOST divergent clip. 50 rather than lower "
                        "because 0011_lucia_e5 (add a dog) is flat to four decimals for "
                        "b <= 10 and only edits at the top of the range.")
    p.add_argument("--step", type=int, default=15,
                   help="Sampler steps the map is calibrated for. A_disc is a mean over "
                        "THIS grid, so the map is valid only for the step count it was "
                        "built with -- which is why it is recorded in every row. "
                        "RECOMPUTE THE ENDPOINT BUDGETS when this changes.")
    p.add_argument("--flow_shift", type=float, default=1.0,
                   help="Must match run_fivebench.py --flow_shift (default 1.0). Enters "
                        "through the denoising-step warp; at 1.0 the warp is identity.")
    p.add_argument("--min_distinct", type=int, default=3,
                   help="Refuse to write the map if fewer than this many distinct b "
                        "result: a degenerate map carries no routing signal and the arm "
                        "is not worth 22 rollouts.")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Default: evaluation/csv/r30_b_map_{arm}.csv")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = args.out or Path(f"evaluation/csv/r30_b_map_{args.arm}.csv")
    col = f"d_{args.arm}_{args.reduction}"

    rows = list(csv.DictReader(args.divergence.open()))
    if not rows:
        raise SystemExit(f"{args.divergence} is empty")
    if col not in rows[0]:
        raise SystemExit(f"{args.divergence} has no column {col!r}. "
                         f"Available: {sorted(k for k in rows[0] if k.startswith('d_'))}")

    # A row whose divergence never got measured cannot be mapped, and silently dropping it
    # would leave the arm without coverage for that pair -- which run_fivebench.py rejects
    # at startup (run_fivebench.py:260). Fail here, where the cause is visible.
    def bad(v: str) -> bool:
        if v in ("", None, "nan"):
            return True
        try:
            return math.isnan(float(v))
        except ValueError:
            return True

    unmeasured = [r["case_id"] for r in rows if bad(r.get(col, ""))]
    if unmeasured:
        raise SystemExit(
            f"{len(unmeasured)} case(s) have no measured {col}: {unmeasured}. "
            f"Re-run r30_divergence.py for arm {args.arm!r} and clear the hard failures "
            f"before mapping. Nothing written.")

    t_next = t_next_schedule(args.step, args.flow_shift)
    a_max, a_min = a_disc(args.b_min, t_next), a_disc(args.b_max, t_next)
    if a_max <= a_min:
        raise SystemExit(f"A_disc({args.b_min}) = {a_max:.6g} is not above "
                         f"A_disc({args.b_max}) = {a_min:.6g}; b_min must be BELOW b_max.")

    d = {r["case_id"]: float(r[col]) for r in rows}
    order = sorted(d.items(), key=lambda kv: kv[1])
    argmin_video, d_min = order[0]
    argmax_video, d_max = order[-1]
    span = d_max - d_min
    if span <= 0.0:
        raise SystemExit(
            f"every case reads the same {col} ({d_min:.6g}); arm {args.arm!r} carries no "
            f"per-video information at all. Nothing written.")

    out_rows: List[Dict[str, object]] = []
    for r in rows:
        dv = float(r[col])
        m = float(np.clip((dv - d_min) / span, 0.0, 1.0))
        b = budget_linear_b(m, t_next, args.b_min, args.b_max)
        out_rows.append({
            "video_name": r["video_name"],
            "edit_type": int(r["edit_type"]),
            # NOTE: the `iou` header is R25 plumbing that run_fivebench.py reads past;
            # the column carries this arm's RAW un-normalised divergence.
            "iou": round(dv, 6),
            "tau": round(b, 4),                 # `tau` carries b -- see the docstring
            "case_id": r["case_id"],
            "arm": args.arm,
            "reduction": args.reduction,
            "d_raw": round(dv, 6),
            "m": round(m, 6),
            "A_b": round(a_disc(b, t_next), 8),
            "d_min": round(d_min, 6),
            "argmin_video": argmin_video,
            "d_max": round(d_max, 6),
            "argmax_video": argmax_video,
            "b_min": args.b_min,
            "b_max": args.b_max,
            "A_max": round(a_max, 8),
            "A_min": round(a_min, 8),
            "step": args.step,
            "flow_shift": args.flow_shift,
        })

    bs = [float(r["tau"]) for r in out_rows]
    n_distinct = len(set(bs))
    if n_distinct < args.min_distinct:
        raise SystemExit(
            f"only {n_distinct} distinct b value(s) over {len(out_rows)} cases "
            f"(need >= {args.min_distinct}): arm {args.arm!r} carries no per-video "
            f"routing signal here and is not worth rendering. Nothing written.")

    fieldnames = ["video_name", "edit_type", "iou", "tau", "case_id", "arm", "reduction",
                  "d_raw", "m", "A_b", "d_min", "argmin_video", "d_max", "argmax_video",
                  "b_min", "b_max", "A_max", "A_min", "step", "flow_shift"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # The delivered budget must be linear in m -- that is the whole point of inverting
    # A_disc rather than using 1/(1+b). Measured here rather than asserted: the maximum
    # relative departure from the straight line between A_max and A_min.
    m_vals = np.array([float(r["m"]) for r in out_rows])
    a_vals = np.array([float(r["A_b"]) for r in out_rows])
    a_want = (1.0 - m_vals) * a_max + m_vals * a_min
    lin_err = float(np.max(np.abs(a_vals - a_want)) / max(a_max, 1e-12))

    meta = {
        "arm": args.arm,
        "reduction": args.reduction,
        "divergence_csv": str(args.divergence),
        "n_cases": len(out_rows),
        "d_min": d_min, "argmin_video": argmin_video,
        "d_max": d_max, "argmax_video": argmax_video,
        "b_min": args.b_min, "b_max": args.b_max,
        "A_max": a_max, "A_min": a_min,
        "budget_range_x": a_max / a_min if a_min > 0 else None,
        "step": args.step, "flow_shift": args.flow_shift,
        "t_next": [round(float(t), 6) for t in t_next],
        "n_distinct_b": n_distinct,
        "b_deciles": [round(budget_linear_b(k / 10.0, t_next, args.b_min, args.b_max), 4)
                      for k in range(11)],
        "budget_linearity_max_rel_err": lin_err,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # ---- report -----------------------------------------------------------------------
    print(f"=== R30 stage 1: {args.arm} ({args.reduction}) -> b, budget-linear ===",
          flush=True)
    print(f"divergence   {args.divergence} ({len(rows)} cases), column {col}", flush=True)
    print(f"schedule     --step {args.step}, --flow_shift {args.flow_shift}", flush=True)
    print(f"             t_next = [{', '.join(f'{t:.3f}' for t in t_next)}]", flush=True)
    print(f"budget       A_disc({args.b_min:g}) = {a_max:.4f}   "
          f"A_disc({args.b_max:g}) = {a_min:.6f}   "
          f"range {a_max / a_min:.1f}x", flush=True)
    print(f"             b(d) deciles = "
          f"[{', '.join(f'{v:g}' for v in meta['b_deciles'])}]", flush=True)
    print(f"             delivered budget max rel. departure from linear in m: "
          f"{lin_err:.2e}", flush=True)
    print(f"floor/ceil   d_min {d_min:.6f} ({argmin_video})   "
          f"d_max {d_max:.6f} ({argmax_video})", flush=True)
    print("             ** both are PINNED by construction: argmin -> b_min, "
          "argmax -> b_max. A metric failure at either extreme drags the whole map. **",
          flush=True)
    print()
    print(f"{'case_id':22s} {'type':>4s} {'d_raw':>10s} {'m':>7s} {'b':>8s} "
          f"{'A(b)':>9s}  vs Eq.4", flush=True)
    for r in sorted(out_rows, key=lambda x: x["m"]):
        print(f"{str(r['case_id']):22s} {r['edit_type']:>4d} {r['d_raw']:>10.4f} "
              f"{r['m']:>7.4f} {r['tau']:>8.4f} {r['A_b']:>9.5f}  "
              f"{float(r['tau']) - EQ4_BLEND_POWER:+.2f}", flush=True)
    print()
    print(f"realized b spread: [{min(bs):.4f}, {max(bs):.4f}]  "
          f"{n_distinct} distinct", flush=True)
    print(f"cases at b_min: {sum(1 for b in bs if abs(b - args.b_min) < 1e-6)}/{len(bs)}   "
          f"at b_max: {sum(1 for b in bs if abs(b - args.b_max) < 1e-6)}/{len(bs)}",
          flush=True)
    print(f"\nwrote {out}  ({len(out_rows)} rows)", flush=True)
    print(f"wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
