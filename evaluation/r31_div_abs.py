"""R31 stage 2b: re-normalise the stage-2 divergence fields ABSOLUTELY, not per clip.

WHY THIS EXISTS
---------------
``r31_divergence.py`` normalises every arm per clip::

    m = (d - d_min_clip) / (d_max_clip - d_min_clip)

which guarantees each clip spans the full [0, 1] tau range regardless of how much it
actually diverged. That is fine for a 22-clip qualitative pass, but it makes ``m`` -- and
therefore ``tau`` -- NON-COMPARABLE ACROSS CLIPS: the same LPIPS divergence routes a
different exponent depending on which clip it sits in. The sharp edge is already recorded
in ``r31_divergence.py``'s NORMALISATION note: a clip with no real divergence still gets a
full-range ``m``, i.e. pure noise stretched over the whole tau span.

This script replaces that with a FIXED, arm-specific scale, settled with the user
2026-09-24 ahead of running R32 on the full FiVE-Bench::

    m = (clip(d, div_min, div_max) - div_min) / (div_max - div_min)
        with  lpips: [0, 1.0]     dino_patch: [0, 0.65]

Everything above ``div_max`` saturates to m = 1 (and so to ``tau_max``). Nothing else in
the pipeline changes: the npz keeps every field ``r31_rho_map.py`` reads, so the SAME
budget-linear mapping runs on top of it unmodified.

⚠️ THE CLIPPED FRACTION IS THE NUMBER TO WATCH, and it is written into every npz and
printed per clip. Saturated tokens are tokens whose within-foreground structure is
DISCARDED -- and that structure (``std_fg`` ~0.20-0.28 vs ``std_bg`` ~0.10-0.16) is the
one thing R31's check-stage2 identified as expressible by R31 and not by R26. A cap that
saturates a large share of the foreground on the high-edit clips gives some of that back.

The raw ``d`` and the clip's own ``d_min``/``d_max`` are carried through UNCHANGED, so the
per-clip normalisation remains recoverable from these files and the two regimes can be
diffed without re-reading the originals.

``--detrend P`` -- PER-FRAME OFFSET REMOVAL (added 2026-09-24)
--------------------------------------------------------------
Measured on all 22 clips against R26's union masks: the stage-1 rollout drifts, and the
BACKGROUND divergence grows +80% (lpips 0.140 -> 0.253) / +75% (dino 0.086 -> 0.151) from
the first third of a clip to the last, in 19/21 clips. But the FOREGROUND grows by nearly
the same amount, so the fg-bg gap is almost constant (0.364 -> 0.332) and per-frame AUC
barely moves (-0.02, degrading in only 12/21 clips, i.e. chance). The map still knows
WHERE the edit is in late frames; it gets the LEVEL wrong.

That matters specifically because this module normalises ABSOLUTELY. A fixed cap means a
late-frame background token at d = 0.25 maps to m = 0.25 while the identical background
early maps to m = 0.14 -- systematically more source release in late backgrounds, which
is the opposite of what a drifting rollout needs. Per-clip min/max partly washed this out;
a fixed scale does not.

``--detrend P`` subtracts each LATENT FRAME's own P-th percentile and clamps at zero::

    d' = max(d - percentile_P(d, axis=tokens), 0)

⚠️ The load-bearing property: subtracting a per-frame CONSTANT and clamping at 0 is
monotone WITHIN each frame, so per-frame AUC is INVARIANT BY CONSTRUCTION -- this cannot
change which tokens rank above which inside a frame, and therefore cannot degrade
localisation. It only removes the frame-level offset.

Measured drift remaining after the subtraction (mean over clips, first third -> last):
    p     lpips bg drift   dino bg drift   fg tokens kept
    none      +0.113          +0.065           100%
    p20       +0.052          +0.035            95%      <- recommended
    p30       +0.041          +0.031            93%
    p50       +0.019          +0.021            85%      <- eats real signal
P must stay BELOW the background's share of the frame or the percentile lands inside the
edit: measured union coverage is median 0.29, max 0.64 across these clips, so p20/p30 are
safe on every one and p50 is not. It is a MITIGATION, not a cure -- ~half the drift
survives at p20, because the background's spread grows too, not only its mean.

OUTPUT
------
  {out_root}/{arm}/edit{T}/{video_name}.npz
      every key of the input npz, with ``m`` REPLACED by the absolute normalisation, plus
      norm_kind="absolute", div_min, div_max, frac_clipped, m_pervideo_max_equiv, and
      (when --detrend is used) detrend_p + detrend_offset [F_lat, 1] so d' is recomputable.
      Raw ``d`` is written UNCHANGED either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

# Settled with the user 2026-09-24. These are DIVERGENCE-SCALE constants, not calibrated
# quantities: LPIPS(alex, spatial) sits on a ~[0, 1.5] range here and DINOv3 patch (1-cos)
# on ~[0, 0.8]; see the measured per-clip table in the daily log for the distributions
# these were chosen against.
DEFAULT_DIV_MAX: Dict[str, float] = {"lpips": 1.0, "dino_patch": 0.65}
DEFAULT_DIV_MIN: Dict[str, float] = {"lpips": 0.0, "dino_patch": 0.0}


def parse_overrides(pairs: Optional[Sequence[str]], base: Dict[str, float],
                    what: str) -> Dict[str, float]:
    """``["lpips=1.2"]`` -> a copy of ``base`` with that key replaced."""
    out = dict(base)
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"[r31_div_abs] --{what} wants arm=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = float(v)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--div_root", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/r31_div"),
                   help="Stage-2 output root containing {arm}/edit{T}/{video}.npz")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/r31_div_abs"))
    p.add_argument("--arms", nargs="+", default=["lpips", "dino_patch"],
                   help="Arms to re-normalise. Default: the two the user scoped for R32.")
    p.add_argument("--div_max", nargs="*", default=None, metavar="ARM=VAL",
                   help="Override the absolute ceiling, e.g. dino_patch=0.7")
    p.add_argument("--div_min", nargs="*", default=None, metavar="ARM=VAL")
    p.add_argument("--detrend", type=float, default=None, metavar="P",
                   help="Per-latent-frame offset removal before the absolute cap: "
                        "subtract each frame's P-th percentile of d and clamp at 0. "
                        "Counters stage-1 rollout drift, which inflates late frames' "
                        "divergence LEVEL without hurting their ranking. p20 recommended; "
                        "P must stay below the background's share of the frame (measured "
                        "union coverage: median 0.29, max 0.64). Off by default.")
    args = p.parse_args(argv)

    div_root = args.div_root.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    if args.detrend is not None and not (0.0 <= args.detrend < 100.0):
        raise SystemExit(f"[r31_div_abs] --detrend must be in [0, 100), got {args.detrend}")
    d_max_by_arm = parse_overrides(args.div_max, DEFAULT_DIV_MAX, "div_max")
    d_min_by_arm = parse_overrides(args.div_min, DEFAULT_DIV_MIN, "div_min")

    n_ok = n_tot = 0
    for arm in args.arms:
        if arm not in d_max_by_arm:
            raise SystemExit(f"[r31_div_abs] no div_max for arm {arm!r}; pass "
                             f"--div_max {arm}=<value>")
        lo, hi = d_min_by_arm.get(arm, 0.0), d_max_by_arm[arm]
        if hi <= lo:
            raise SystemExit(f"[r31_div_abs] {arm}: div_max {hi} must exceed div_min {lo}")
        files = sorted((div_root / arm).glob("edit*/*.npz"))
        if not files:
            raise SystemExit(f"[r31_div_abs] no edit*/*.npz under {div_root / arm}")
        dt_note = "" if args.detrend is None else f" detrend=p{args.detrend:g}"
        print(f"\n[r31_div_abs] arm={arm} absolute range [{lo:g}, {hi:g}]{dt_note} "
              f"({len(files)} clips) -> {out_root / arm}", flush=True)
        for f in files:
            n_tot += 1
            edit_dir = f.parent.name
            try:
                z = np.load(f)
                d_raw = z["d"].astype(np.float64)
                if args.detrend is None:
                    d, offset = d_raw, None
                else:
                    # Per LATENT FRAME (axis 1 is the 1560 tokens of that frame).
                    offset = np.percentile(d_raw, args.detrend, axis=1, keepdims=True)
                    d = np.maximum(d_raw - offset, 0.0)
                frac_clipped = float((d > hi).mean())
                frac_floored = float((d < lo).mean())
                m = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

                # Every original key survives; only `m` is redefined. Re-saving the whole
                # npz (rather than writing a thin sidecar) is what lets r31_rho_map.py and
                # r31_stage3_grids.py run on this tree completely unmodified.
                payload = {k: z[k] for k in z.files}     # raw `d` carried through unchanged
                payload["m"] = m
                if offset is not None:
                    payload["detrend_p"] = np.float64(args.detrend)
                    payload["detrend_offset"] = offset.astype(np.float32)
                payload.update(
                    norm_kind=np.str_("absolute"),
                    div_min=np.float64(lo),
                    div_max=np.float64(hi),
                    frac_clipped=np.float64(frac_clipped),
                    frac_floored=np.float64(frac_floored),
                    # What this clip's OLD per-video m would have been at this clip's own
                    # d_max, expressed on the new absolute scale -- i.e. the factor by
                    # which the per-video normalisation was inflating (>1) or deflating
                    # (<1) this clip relative to the fixed scale.
                    m_pervideo_max_equiv=np.float64((float(z["d_max"]) - lo) / (hi - lo)),
                )
                out = out_root / arm / edit_dir / f.name
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, **payload)
                dt = ("" if offset is None else
                      f" detrend_p{args.detrend:g}=[{offset.min():.4f},{offset.max():.4f}]")
                print(f"[ok] {arm}/{edit_dir}/{f.stem}: d_max_clip={float(z['d_max']):.4f} "
                      f"-> m_abs_max={m.max():.4f} mean={m.mean():.4f} "
                      f"clipped={frac_clipped:.4%} frac(m>0.5)={float((m > 0.5).mean()):.3f}"
                      f"{dt}", flush=True)
                n_ok += 1
            except Exception as ex:
                print(f"[ERROR] {arm}/{edit_dir}/{f.stem}: {type(ex).__name__}: {ex}",
                      flush=True)

    print(f"\n[r31_div_abs] done: {n_ok}/{n_tot} ok -> {out_root}", flush=True)
    return 0 if n_ok == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
