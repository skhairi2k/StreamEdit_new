#!/usr/bin/env python
"""R27 step 4 -- map the measured per-case depth-delta magnitude onto the Eq. 4
release exponent tau.

`tau` IS `blend_power`, read directly as ``1 - t_next ** blend_power`` at
causal_model.py:334, so the SOURCE weight at a step is W_src(t) = t ** tau. Because
t in (0, 1), t ** tau DECREASES in tau: a large tau releases the source anchor fast,
a small tau holds it. R27 routes the direction R25 could not:

    small depth change (colour / material edit) -> tau_min: source held longest
    large depth change (swap / removal)         -> tau_max: source released early

Two design choices distinguish this from r25_tau_map.py, and both are load-bearing.

1. THE FLOOR IS A BOUNDARY CONDITION, NOT A DATASET MINIMUM
   ---------------------------------------------------------
   R25 normalised by the observed min/max of its signal, which its own verdict
   flagged as a calibration defect: `f` was undefined for a single new video and
   the mapping moved whenever the case set moved. Here the low anchor is each
   case's OWN detection threshold, expressed in the same normalised units:

       D_lo(i) = thr(i) / iqr_src(i)

   `thr` is the value a residual pixel must exceed to enter L at all (r27_depth_delta.py:
   ``max(mad_k * sigma, min_effect * IQR(D_src))``). So D_lo is the largest D_norm an
   edit can produce while still being statistically indistinguishable from the depth
   estimator's own noise on that scene. m = 0 therefore means "no measurable geometry
   change", and tau_min = 2.0 makes the schedule reduce EXACTLY to StreamGVE Eq. 4
   there. tau_min is not a tuned floor; it is the value the boundary condition forces.

   Consequence, stated because it is a real distortion and not an artefact of writing:
   D_lo is per-case, so a scene whose background depth estimate is noisy (large sigma,
   hence `thr_by == "stat"`) gets a HIGHER floor and is pushed toward tau_min. That is
   defensible -- a noisier estimate demands more evidence before we act on it -- but it
   means two edits with identical D_norm can receive different tau. The `d_lo` column
   is emitted per row so this is auditable rather than buried.

2. THE MAP IS LINEAR IN THE ANCHORING BUDGET, NOT IN TAU
   -----------------------------------------------------
   tau is an EXPONENT and its effect saturates. Define the source-anchoring budget
   actually spent over the schedule:

       A(tau) = sum_i t_i ** tau        (t_i = the t_next values of the --step schedule)

   Interpolating tau linearly on [2, 50] puts a mid-magnitude edit at tau = 25.5,
   which spends A = 0.29 against Eq. 4's A = 4.51 -- about 1/16 of the source
   anchoring, i.e. "mid" on the tau axis is already almost full detachment. So we
   interpolate the BUDGET linearly and invert:

       A_m = (1 - m) * A(tau_min) + m * A(tau_max)
       tau = A^{-1}(A_m)                (Brent, on the strictly decreasing A)

   At --step 15 this gives A(2) = 4.5062, A(50) = 0.0320, and m = 0.6 -> tau = 5.53.
   Equivalently: 1 + tau is interpolated harmonically, since the continuous form of
   the budget is the integral of t**tau over [0, 1], which is 1 / (1 + tau).

   A IS SCHEDULE-DEPENDENT. A map calibrated at --step 15 silently changes meaning at
   any other step count, so `step` and `flow_shift` are written into every row and into
   the sidecar .meta.json.

The emitted CSV keeps R25's (video_name, edit_type, iou, tau) columns FIRST and
unchanged, so run_fivebench.py --tau_map needs no modification -- the `iou` column
carries D_norm here. Extra diagnostic columns follow; csv.DictReader ignores them.
Provenance goes in those columns and in a sidecar JSON rather than in a leading `#`
comment, because run_fivebench.py:256 reads the file with a bare csv.DictReader and
would take a comment line as the header row.

Usage
-----
    python evaluation/r27_tau_map.py --assign depth \\
        --depth_csv evaluation/csv/r27_depth.csv \\
        --d_hi_case auto --tau_min 2.0 --tau_max 50.0 --step 15 \\
        -o evaluation/csv/r27_tau_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

# The Eq. 4 default exponent, printed alongside each row so the table shows at a
# glance which direction each case was moved relative to the baseline arm.
EQ4_BLEND_POWER: float = 2.0

# Upper bracket for the Brent inversion. A(400) is ~1e-12 at --step 15, far below any
# A_m we can request with tau_max <= 100, so the root is always bracketed.
TAU_BRACKET_HI: float = 400.0


# --------------------------------------------------------------------------------
# The denoising schedule -- reproduced EXACTLY as the pipeline builds it
# --------------------------------------------------------------------------------
def t_next_schedule(step: int, flow_shift: float = 1.0) -> np.ndarray:
    """Return the t_next values the blend actually sees, one per denoising step.

    Mirrors three places in the pipeline and must stay in sync with all three:

      inference_edit_streamedit.py:68
          config['denoising_step_list'] = np.arange(1000, 0, -1000 / step).astype(int)
      edit_causal_inference.py:37-39   (warp_denoising_step: true in the dmd config)
          timesteps = cat(scheduler.timesteps, [0]); dsl = timesteps[1000 - dsl]
      edit_causal_inference.py:721
          timestep_next = dsl[index + 1] / 1000  if index < len-1  else 0

    The scheduler is FlowMatchScheduler(shift=flow_shift, sigma_min=0.0,
    extra_one_step=True) with set_timesteps(1000, ...) -- wan_wrapper.py:131-133 --
    so sigmas = linspace(1, 0, 1001)[:-1] pushed through the flow shift.

    At flow_shift = 1.0 (the run_fivebench default, and what R21/R25/R27 all use) the
    shift is the identity and the warp is a no-op, so t_next reduces to the plain
    arange / 1000. This function does NOT assume that -- it applies the shift -- so a
    map built at a different flow_shift is still correct rather than silently wrong.
    """
    if step < 2:
        raise ValueError(f"--step must be >= 2, got {step}")

    dsl = np.arange(1000, 0, -1000.0 / step).astype(int)          # e.g. 15 values

    # FlowMatchScheduler.set_timesteps(1000, ...) with extra_one_step=True, sigma_min=0
    sigmas = np.linspace(1.0, 0.0, 1001)[:-1]                      # 1000 values
    sigmas = flow_shift * sigmas / (1.0 + (flow_shift - 1.0) * sigmas)
    timesteps = np.concatenate([sigmas * 1000.0, np.zeros(1)])     # 1001, index 1000 -> 0
    warped = timesteps[1000 - dsl]

    t_next = np.empty(len(warped), dtype=np.float64)
    t_next[:-1] = warped[1:] / 1000.0
    t_next[-1] = 0.0                                               # the final step
    return t_next


def budget(tau: float, t_next: np.ndarray) -> float:
    """A(tau) = sum_i t_i ** tau -- the total source weight spent over the schedule.

    Strictly decreasing in tau for t_i in [0, 1), which is what makes it invertible.
    """
    return float(np.sum(np.power(t_next, tau)))


def budget_linear_tau(m: float, t_next: np.ndarray,
                      tau_min: float, tau_max: float) -> float:
    """Interpolate the anchoring budget linearly in m and invert for tau."""
    a_lo, a_hi = budget(tau_min, t_next), budget(tau_max, t_next)
    a_m = (1.0 - m) * a_lo + m * a_hi
    if m <= 0.0:
        return float(tau_min)
    if m >= 1.0:
        return float(tau_max)
    return float(brentq(lambda x: budget(x, t_next) - a_m,
                        0.0, TAU_BRACKET_HI, xtol=1e-10, rtol=1e-12))


# --------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--depth_csv", type=Path,
                   default=Path("evaluation/csv/r27_depth.csv"),
                   help="Output of r27_depth_delta.py; needs case_id, video_name, "
                        "edit_type, d_norm, thr, iqr_src.")
    p.add_argument("--d_hi", type=float, default=None,
                   help="Pin the ceiling to an ABSOLUTE D_norm value, overriding "
                        "--d_hi_case. This is the transferable form: the map then applies "
                        "unchanged to a single new video, because nothing about it depends "
                        "on which other clips are in the set. Cases above it clip to "
                        "tau_max, which is semantically fine -- clipping reads as 'at least "
                        "full detachment', and a ceiling nobody reaches is wasted range. "
                        "NOTE: this value is NOT derivable from the depth CSV. It answers "
                        "'how much depth movement means the source must be abandoned "
                        "entirely', which is a property of the RENDER, not of the depth "
                        "statistics. Calibrate it against a single-clip tau sweep and "
                        "record the sweep in the run log; do not pick it by eye.")
    p.add_argument("--d_hi_case", type=str, default="auto",
                   help="Case whose D_norm defines m = 1. 'auto' (default) uses the "
                        "ARGMAX over the case set, which clips nobody by construction. "
                        "A case_id pins the ceiling to that case instead -- anchoring on "
                        "0011_lucia_e5 was measured to clip 11 of 21 cases to tau_max. "
                        "WARNING: 'auto' makes the map DEPEND ON THE CASE SET, which is "
                        "R25's calibration defect. Measured on this set: dropping the "
                        "single clip 0042_gym-ball moves 0011_lucia_e2 from tau 5.78 to "
                        "tau 50.00 -- same video, same edit, same D_norm. Prefer --d_hi.")
    p.add_argument("--tau_min", type=float, default=2.0,
                   help="tau at m = 0. 2.0 is a BOUNDARY CONDITION, not a tuned floor: "
                        "m = 0 means the depth change is indistinguishable from the "
                        "estimator's noise, and the schedule must then be Eq. 4 exactly.")
    p.add_argument("--tau_max", type=float, default=50.0,
                   help="tau at m = 1. Step-count dependent: at --step 15, tau = 50 "
                        "still leaves W_src = 0.031 at the first blended step; at "
                        "--step 7 it leaves ~4e-4, i.e. degenerate == blend_sched zero.")
    p.add_argument("--step", type=int, default=15,
                   help="Sampler steps the map is calibrated for. A(tau) is a sum over "
                        "THIS schedule; the map is only valid for the step count it was "
                        "built with, which is why it is recorded in every row.")
    p.add_argument("--flow_shift", type=float, default=1.0,
                   help="Must match run_fivebench.py --flow_shift (default 1.0). Enters "
                        "through the denoising-step warp; at 1.0 the warp is identity.")
    p.add_argument("--assign", type=str, default="depth",
                   choices=["depth", "constant"],
                   help="'depth' (default) is the adaptive arm. 'constant' gives every "
                        "pair the realized mean tau -- it isolates whether VARYING tau "
                        "helps at all, which is the contrast R25 needed a phase-2 split "
                        "to get.")
    p.add_argument("--min_distinct", type=int, default=3,
                   help="Refuse to write the depth arm if fewer than this many distinct "
                        "tau values result: a degenerate map carries no routing signal "
                        "and the arm is not worth the GPU time.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r27_tau_map.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rows = list(csv.DictReader(args.depth_csv.open()))
    if not rows:
        raise SystemExit(f"{args.depth_csv} is empty")

    # A row whose magnitude never got measured cannot be mapped, and silently dropping
    # it would leave the arm without coverage for that pair -- which run_fivebench.py
    # rejects at startup (run_fivebench.py:260). Fail here, where the cause is visible.
    unmeasured = [r["case_id"] for r in rows
                  if r.get("d_norm", "") in ("", "nan") or r.get("iqr_src", "") in ("", "nan")]
    if unmeasured:
        raise SystemExit(
            f"{len(unmeasured)} case(s) have no measured depth magnitude: {unmeasured}. "
            f"Re-run the `depth` step and clear the hard failures before mapping.")

    t_next = t_next_schedule(args.step, args.flow_shift)
    a_lo, a_hi = budget(args.tau_min, t_next), budget(args.tau_max, t_next)

    d_norm = {r["case_id"]: float(r["d_norm"]) for r in rows}

    # ---- resolve the ceiling ------------------------------------------------------
    order = sorted(d_norm.items(), key=lambda kv: -kv[1])
    if args.d_hi is not None:
        # Absolute ceiling: the only form that transfers to a single new video.
        d_hi_case, d_hi = f"<absolute {args.d_hi:g}>", float(args.d_hi)
        if d_hi < order[0][1]:
            print(f"[r27_tau_map] {sum(1 for _, v in order if v >= d_hi)} case(s) at or "
                  f"above D_hi = {d_hi:g} will clip to tau_max, top being "
                  f"{order[0][0]} at {order[0][1]:.4f}", flush=True)
    elif args.d_hi_case == "auto":
        d_hi_case, d_hi = order[0]
        print("[r27_tau_map] WARNING: --d_hi_case auto ties the map to THIS case set. "
              "Dropping one clip can move a case from tau 5.78 to 50.00. Use --d_hi "
              "for a map that transfers to a single new video.", flush=True)
    else:
        if args.d_hi_case not in d_norm:
            raise SystemExit(f"--d_hi_case {args.d_hi_case!r} is not a case_id in "
                             f"{args.depth_csv}. Known: {sorted(d_norm)}")
        d_hi_case, d_hi = args.d_hi_case, d_norm[args.d_hi_case]
    runner_up_case, runner_up = order[1] if len(order) > 1 else (None, float("nan"))

    # ---- per-case floor, m, tau ---------------------------------------------------
    out_rows: List[Dict[str, object]] = []
    floor_exceeds: List[str] = []
    for r in rows:
        dn = float(r["d_norm"])
        iqr = float(r["iqr_src"])
        # thr is in raw depth units; D_norm is d_raw / iqr_src, so the floor has to be
        # divided by the same IQR to live on the same axis.
        d_lo = float(r["thr"]) / iqr if iqr > 0 else float("inf")

        span = d_hi - d_lo
        if span <= 0.0:
            # This case's own noise floor sits at or above the ceiling, so no edit on
            # this scene could be called measurable relative to D_hi. m = 0 is the only
            # defensible reading; record it loudly rather than dividing by <= 0.
            m = 0.0
            floor_exceeds.append(r["case_id"])
        else:
            m = float(np.clip((dn - d_lo) / span, 0.0, 1.0))

        out_rows.append({
            "video_name": r["video_name"],
            "edit_type": int(r["edit_type"]),
            "iou": round(dn, 6),          # NOTE: carries D_norm; header kept for R25 plumbing
            "tau": round(budget_linear_tau(m, t_next, args.tau_min, args.tau_max), 4),
            "case_id": r["case_id"],
            "d_norm": round(dn, 6),
            "d_lo": round(d_lo, 6),
            "d_hi": round(d_hi, 6),
            "m": round(m, 6),
            "thr_by": r.get("thr_by", ""),
            "d_hi_case": d_hi_case,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "step": args.step,
            "flow_shift": args.flow_shift,
            "assign": args.assign,
        })

    depth_taus = [float(r["tau"]) for r in out_rows]
    n_distinct = len(set(depth_taus))
    if n_distinct < args.min_distinct:
        raise SystemExit(
            f"only {n_distinct} distinct tau value(s) over {len(out_rows)} cases "
            f"(need >= {args.min_distinct}): the depth signal carries no per-video "
            f"information here and the arm is not worth rendering. Nothing written.")

    # ---- the control arm ----------------------------------------------------------
    # The realized mean of the DEPTH arm, so the two arms differ only in the assignment
    # and every other knob -- schedule, seed, anchors, metrics -- is byte-identical.
    flat = round(float(np.mean(depth_taus)), 4)
    if args.assign == "constant":
        for r in out_rows:
            r["tau"] = flat

    fieldnames = ["video_name", "edit_type", "iou", "tau", "case_id", "d_norm",
                  "d_lo", "d_hi", "m", "thr_by", "d_hi_case", "tau_min", "tau_max",
                  "step", "flow_shift", "assign"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    meta = {
        "depth_csv": str(args.depth_csv),
        "assign": args.assign,
        "n_cases": len(out_rows),
        "d_hi": d_hi,
        "d_hi_case": d_hi_case,
        "d_hi_mode": ("absolute" if args.d_hi is not None
                      else "argmax" if args.d_hi_case == "auto" else "pinned_case"),
        "d_hi_runner_up": runner_up,
        "d_hi_runner_up_case": runner_up_case,
        "d_hi_margin": (d_hi / runner_up)
                       if (runner_up_case is not None and runner_up > 0) else None,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "step": args.step,
        "flow_shift": args.flow_shift,
        "A_tau_min": a_lo,
        "A_tau_max": a_hi,
        "t_next": [round(float(t), 6) for t in t_next],
        "realized_mean_tau": flat,
        "n_distinct_tau": n_distinct,
        "floor_exceeds_ceiling": floor_exceeds,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    # ---- report -------------------------------------------------------------------
    taus = [float(r["tau"]) for r in out_rows]
    print("=== R27 step 4: depth-delta -> tau (budget-linear) ===", flush=True)
    print(f"depth_csv    {args.depth_csv} ({len(rows)} cases)", flush=True)
    print(f"assign       {args.assign}", flush=True)
    print(f"schedule     --step {args.step}, --flow_shift {args.flow_shift}", flush=True)
    print(f"             t_next = [{', '.join(f'{t:.3f}' for t in t_next)}]", flush=True)
    print(f"budget       A({args.tau_min:g}) = {a_lo:.4f}   "
          f"A({args.tau_max:g}) = {a_hi:.4f}", flush=True)
    how = ("absolute" if args.d_hi is not None
           else "argmax" if args.d_hi_case == "auto" else "pinned to case")
    print(f"ceiling      D_hi = {d_hi:.4f} from {d_hi_case} ({how})", flush=True)
    margin = (d_hi / runner_up) if (runner_up_case is not None and runner_up > 0) else None
    if margin is not None:
        print(f"             runner-up {runner_up_case} {runner_up:.4f}, "
              f"margin {margin:.2f}x", flush=True)
        if margin > 2.0:
            # A lone outlier at the top sets the ceiling for everyone, so it compresses
            # every other case toward tau_min. Worth eyeballing before rendering 22 clips.
            print("             ** the top case leads the second by more than 2x: read "
                  "its panel before trusting the ceiling **", flush=True)
    if floor_exceeds:
        print(f"             ** {len(floor_exceeds)} case(s) have D_lo >= D_hi, forced "
              f"to m = 0: {floor_exceeds} **", flush=True)
    print()
    print(f"{'case_id':22s} {'type':>4s} {'D_norm':>8s} {'D_lo':>7s} {'thr_by':>7s} "
          f"{'m':>7s} {'tau':>8s}  vs Eq.4", flush=True)
    for r in sorted(out_rows, key=lambda x: x["m"]):
        d = float(r["tau"]) - EQ4_BLEND_POWER
        print(f"{str(r['case_id']):22s} {r['edit_type']:>4d} {r['d_norm']:>8.4f} "
              f"{r['d_lo']:>7.4f} {str(r['thr_by']):>7s} {r['m']:>7.4f} "
              f"{r['tau']:>8.4f}  {d:+.2f}", flush=True)
    print()
    print(f"realized tau spread: [{min(taus):.4f}, {max(taus):.4f}]  "
          f"span {max(taus) - min(taus):.4f}, {len(set(taus))} distinct", flush=True)
    print(f"cases at tau_min (== Eq. 4 exactly): "
          f"{sum(1 for t in taus if abs(t - args.tau_min) < 1e-6)}/{len(taus)}", flush=True)
    print(f"cases at tau_max: "
          f"{sum(1 for t in taus if abs(t - args.tau_max) < 1e-6)}/{len(taus)}", flush=True)
    print(f"realized mean tau (used by --assign constant): {flat:.4f}", flush=True)
    print(f"\nwrote {args.out}  ({len(out_rows)} rows)", flush=True)
    print(f"wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
