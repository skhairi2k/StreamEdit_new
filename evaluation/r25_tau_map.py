#!/usr/bin/env python
"""R25 step 3 -- map the measured per-case IoU onto the Eq. 4 release exponent tau.

`tau` IS `blend_power`, read directly as ``1 - t_next ** blend_power`` at
causal_model.py:334, so the SOURCE weight at a step is W_src(t) = t ** tau. Because
t in (0, 1), t ** tau DECREASES in tau: a large tau releases the source anchor fast,
a small tau holds it. The mapping is deliberately descending --

    high IoU (small shape change, e.g. a colour edit) -> tau_min: source held longest
    low  IoU (large shape change, e.g. person -> lion) -> tau_max: source released early

ANCHORING ON THE MEASURED RANGE is the locked design choice, not a convenience. Object
swaps still occupy roughly the same image region, so their IoU never approaches 0 (the
observed floor is ~0.07 for swaps; type-5 additions and type-6 removal now sit lower
still, see below), while colour/material edits sit near 1.0. On a nominal [0, 1] domain
the realized tau spread would be a fraction of [tau_min, tau_max] and the arm would be
far less adaptive than intended. The cost, which is real: `f` is calibrated on THIS case
set and is not transferable as-is to a different one.

MAPPING FAMILY (revised 2026-09-01, `--mapping budget_linear` now the default): tau is an
EXPONENT and its effect saturates -- the physically meaningful quantity is the anchoring
budget actually spent over the schedule,

    A(tau) = sum_i t_i ** tau        (t_i = the t_next values of the --step schedule)

not tau itself. Interpolating tau LINEARLY in IoU (the original R25 design, kept as
`--mapping linear`) means equal IoU steps do NOT correspond to equal anchoring-budget
steps: at [tau_min, tau_max] = [2, 20] the midpoint-IoU case lands at tau = 11.00 under
linear-in-tau, spending A(11) far closer to full detachment than the arithmetic midpoint
suggests, while under budget-linear the same case lands at tau = 4.25. This is the exact
distortion R27 identified and fixed for its own (wider) [1, 50] range; R25 adopts the same
fix (`budget_linear_tau`, reused verbatim from r27_tau_map.py) now that the type-5 mask
fix already forces a full re-render, so there is no separate cost to also correcting this.
Measured on the post-type-5-fix 22-case IoU distribution at [2, 20]: mean tau drops from
13.39 (linear) to 7.73 (budget-linear) -- widening under linear-in-tau drags the ENTIRE
distribution, including the swap-heavy middle, toward the near-zero-budget regime that
R21's `zero` schedule showed causes a ~3x LPIPS preservation blowout (143.0 vs `ref_vp`'s
53.6); under budget-linear the same widening mostly extends room for the already-extreme
low-IoU tail (removal, the most disjoint additions) without dragging the middle along.

`tau_max = 20` (revised 2026-09-01 from 10; only meaningful together with the
budget-linear switch -- see the docstring block above for why 20 under the OLD
linear-in-tau mapping would have been reckless). `tau_min = 2` is unchanged: it still
pins the highest measured IoU exactly at the Eq. 4 default, so the mildest edits behave
like the paper baseline rather than holding the source longer than it.

Refuses to write a degenerate range (< --min_range): if every case maps to nearly the same
tau, the IoU signal carries no per-video information and rendering the arm would burn GPU
time on 22 near-identical clips. This check is on the IoU axis and is unaffected by which
mapping family converts it to tau.

Usage
-----
    python evaluation/r25_tau_map.py \\
        --iou_csv evaluation/csv/r25_iou.csv \\
        --tau_min 2.0 --tau_max 20.0 --mapping budget_linear \\
        --step 15 --flow_shift 1.0 \\
        -o evaluation/csv/r25_tau_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

# The Eq. 4 default exponent, printed alongside each row so the table shows at a glance
# which direction each case was moved relative to the baseline arm.
EQ4_BLEND_POWER: float = 2.0

# Upper bracket for the Brent inversion in budget-linear mode. A(400) is ~1e-12 at
# --step 15, far below any A_u we can request with tau_max <= 100, so the root is
# always bracketed. Mirrors r27_tau_map.py's TAU_BRACKET_HI.
TAU_BRACKET_HI: float = 400.0


# --------------------------------------------------------------------------------
# The denoising schedule -- reproduced EXACTLY as the pipeline builds it.
# Duplicated verbatim from r27_tau_map.py rather than imported, matching this repo's
# convention of self-contained per-task scripts; the two copies must stay in sync.
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


def budget_linear_tau(u: float, t_next: np.ndarray,
                       tau_min: float, tau_max: float) -> float:
    """Interpolate the anchoring budget linearly in u and invert for tau.

    u = 0 (lowest measured IoU, largest shape change) -> tau_max: fastest release,
        smallest budget spent.
    u = 1 (highest measured IoU, smallest shape change) -> tau_min: slowest release,
        largest budget spent.

    Direction is the mirror of r27_tau_map.py's `budget_linear_tau(m, ...)` (there
    m=0 -> tau_min, m=1 -> tau_max) because R25's driving signal is IoU (descending in
    shape change) while R27's is a depth-delta magnitude (ascending in shape change).
    """
    a_fast, a_slow = budget(tau_max, t_next), budget(tau_min, t_next)   # a_fast < a_slow
    if u <= 0.0:
        return float(tau_max)
    if u >= 1.0:
        return float(tau_min)
    a_u = a_fast + (a_slow - a_fast) * u
    return float(brentq(lambda x: budget(x, t_next) - a_u,
                        0.0, TAU_BRACKET_HI, xtol=1e-10, rtol=1e-12))


def iou_to_tau_linear(iou: float, iou_lo: float, iou_hi: float,
                       tau_min: float, tau_max: float) -> float:
    """The ORIGINAL R25 mapping: tau linear in IoU. Kept for `--mapping linear` only --
    known to distort the anchoring budget at wide tau ranges (see module docstring)."""
    u = (min(max(iou, iou_lo), iou_hi) - iou_lo) / (iou_hi - iou_lo)   # 0 at lo, 1 at hi
    return tau_max - (tau_max - tau_min) * u


# --------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iou_csv", type=Path, default=Path("evaluation/csv/r25_iou.csv"),
                   help="Output of r25_iou.py; must carry video_name, edit_type, iou.")
    p.add_argument("--tau_min", type=float, default=2.0,
                   help="tau at the HIGHEST measured IoU. 2.0 pins the mildest edits at the "
                        "Eq. 4 default, so the arm only ever releases at least as fast "
                        "as the paper baseline, never slower.")
    p.add_argument("--tau_max", type=float, default=20.0,
                   help="tau at the LOWEST measured IoU. Set by prior, not derived. "
                        "Revised 2026-09-01 from 10.0 -> 20.0, valid only together with "
                        "--mapping budget_linear (see module docstring): under the old "
                        "linear-in-tau mapping this range would drag the whole case set "
                        "toward the zero-schedule preservation-collapse regime.")
    p.add_argument("--mapping", type=str, default="budget_linear",
                   choices=["budget_linear", "linear"],
                   help="'budget_linear' (default, revised 2026-09-01) interpolates the "
                        "anchoring budget A(tau)=sum t_i**tau linearly in normalised IoU "
                        "and inverts -- corrects the tau-saturation distortion R27 already "
                        "fixed for its own signal. 'linear' is the original R25 mapping, "
                        "tau linear in IoU; kept only to reproduce the phase-1 arm.")
    p.add_argument("--step", type=int, default=15,
                   help="Sampler steps the budget-linear map is calibrated for (ignored "
                        "under --mapping linear). Must match run_fivebench.py --step.")
    p.add_argument("--flow_shift", type=float, default=1.0,
                   help="Must match run_fivebench.py --flow_shift (ignored under "
                        "--mapping linear). Enters through the denoising-step warp; at "
                        "1.0 the warp is identity.")
    p.add_argument("--assign", type=str, default="iou",
                   choices=["iou", "reversed", "constant"],
                   help="How tau is PAIRED with videos. 'iou' (default) is the R25 "
                        "adaptive arm. 'reversed' keeps the SAME 22 tau values and inverts "
                        "the pairing against the IoU ranking -- it isolates whether the "
                        "PAIRING carries signal, because an identical tau distribution "
                        "means the *_unedit_part metric's bias against shape change "
                        "applies equally to both arms and cancels in the contrast. "
                        "'constant' gives every pair the realized mean tau -- it isolates "
                        "whether VARYING tau helps at all.")
    p.add_argument("--min_range", type=float, default=0.05,
                   help="Refuse to write if IoU_hi - IoU_lo is below this.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r25_tau_map.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rows = list(csv.DictReader(args.iou_csv.open()))
    if not rows:
        raise SystemExit(f"{args.iou_csv} is empty")

    # A row whose IoU never got measured cannot be mapped, and silently dropping it would
    # leave the arm without coverage for that pair -- which run_fivebench.py would then
    # reject at startup. Fail here instead, where the cause is visible.
    unmeasured = [r["case_id"] for r in rows if r["iou"] in ("", "nan")]
    if unmeasured:
        raise SystemExit(
            f"{len(unmeasured)} case(s) have no measured IoU: {unmeasured}. "
            f"Re-run the `iou` step and clear the hard failures before mapping.")

    ious = [float(r["iou"]) for r in rows]
    iou_lo, iou_hi = min(ious), max(ious)

    if iou_hi - iou_lo < args.min_range:
        raise SystemExit(
            f"degenerate IoU range [{iou_lo:.3f}, {iou_hi:.3f}] "
            f"(span {iou_hi - iou_lo:.3f} < {args.min_range}): the signal carries no "
            f"per-video information and the arm is not worth rendering. Nothing written.")

    t_next = t_next_schedule(args.step, args.flow_shift) if args.mapping == "budget_linear" else None
    a_lo = budget(args.tau_min, t_next) if t_next is not None else None
    a_hi = budget(args.tau_max, t_next) if t_next is not None else None

    out_rows: List[Dict[str, object]] = []
    for r in rows:
        iou = float(r["iou"])
        u = (min(max(iou, iou_lo), iou_hi) - iou_lo) / (iou_hi - iou_lo)
        if args.mapping == "budget_linear":
            tau = budget_linear_tau(u, t_next, args.tau_min, args.tau_max)
        else:
            tau = iou_to_tau_linear(iou, iou_lo, iou_hi, args.tau_min, args.tau_max)
        out_rows.append({
            "video_name": r["video_name"],
            "edit_type": int(r["edit_type"]),
            "iou": round(iou, 6),
            "tau": round(tau, 4),
        })

    # ---- PHASE 2: re-pair the SAME tau values, or flatten them ---------------------
    # Both controls deliberately leave the tau MULTISET untouched ('reversed') or replace
    # it with its own mean ('constant'); nothing else about the arm changes, so the three
    # arms differ only in the assignment. Agnostic to which mapping family produced the
    # multiset.
    if args.assign == "reversed":
        # Rank reversal, not an IoU mirror: mirroring the IoU axis through a linear map
        # would yield a DIFFERENT multiset unless the IoUs happened to be symmetric, and
        # the identical-multiset property is the whole reason this control cancels the
        # metric bias. Tie-break on (video_name, edit_type) so the permutation is
        # deterministic when two cases share an IoU.
        order = sorted(range(len(out_rows)),
                       key=lambda i: (out_rows[i]["iou"], out_rows[i]["video_name"],
                                      out_rows[i]["edit_type"]))
        taus_by_rank = [out_rows[i]["tau"] for i in order]
        for rank, i in enumerate(order):
            out_rows[i]["tau"] = taus_by_rank[len(order) - 1 - rank]
    elif args.assign == "constant":
        flat = round(sum(r["tau"] for r in out_rows) / len(out_rows), 4)
        for r in out_rows:
            r["tau"] = flat

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video_name", "edit_type", "iou", "tau"])
        writer.writeheader()
        writer.writerows(out_rows)

    meta = {
        "iou_csv": str(args.iou_csv),
        "mapping": args.mapping,
        "assign": args.assign,
        "n_cases": len(out_rows),
        "iou_lo": iou_lo,
        "iou_hi": iou_hi,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "step": args.step if args.mapping == "budget_linear" else None,
        "flow_shift": args.flow_shift if args.mapping == "budget_linear" else None,
        "A_tau_min": a_lo,
        "A_tau_max": a_hi,
        "t_next": [round(float(t), 6) for t in t_next] if t_next is not None else None,
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    taus = [float(r["tau"]) for r in out_rows]
    print(f"=== R25 step 3: IoU -> tau ({args.mapping}) ===", flush=True)
    print(f"iou_csv     {args.iou_csv} ({len(rows)} cases)", flush=True)
    print(f"IoU range   [{iou_lo:.4f}, {iou_hi:.4f}]  span {iou_hi - iou_lo:.4f}",
          flush=True)
    print(f"assign      {args.assign}", flush=True)
    print(f"tau range   [{args.tau_min}, {args.tau_max}]  "
          f"(Eq.4 baseline blend_power = {EQ4_BLEND_POWER})", flush=True)
    if args.mapping == "budget_linear":
        print(f"schedule    --step {args.step}, --flow_shift {args.flow_shift}", flush=True)
        print(f"budget      A({args.tau_min:g}) = {a_lo:.4f}   "
              f"A({args.tau_max:g}) = {a_hi:.4f}", flush=True)
    print()
    print(f"{'video_name':24s} {'type':>4s} {'iou':>8s} {'tau':>8s}  vs Eq.4", flush=True)
    for r in sorted(out_rows, key=lambda x: x["iou"]):
        d = float(r["tau"]) - EQ4_BLEND_POWER
        print(f"{r['video_name']:24s} {r['edit_type']:>4d} {r['iou']:>8.4f} "
              f"{r['tau']:>8.4f}  {d:+.2f}", flush=True)
    print()
    print(f"realized tau spread: [{min(taus):.4f}, {max(taus):.4f}]  "
          f"span {max(taus) - min(taus):.4f}", flush=True)
    print(f"cases routed FASTER than Eq.4 (tau > {EQ4_BLEND_POWER}): "
          f"{sum(1 for t in taus if t > EQ4_BLEND_POWER)}/{len(taus)}", flush=True)
    print(f"\nwrote {args.out}  ({len(out_rows)} rows)", flush=True)
    print(f"wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
