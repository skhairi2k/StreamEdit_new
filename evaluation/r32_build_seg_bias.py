"""Build R32's per-head segment-bias tables from the R19 labels.

WHAT THE BIAS DOES
------------------
The target branch attends to a concatenation of three key segments: the
target's previous frames, the source's current frame (background tokens, and
only over the second half of the denoising steps), and the target's current
frame. Nothing in StreamEdit gives a head any control over their relative
contribution -- every head sees the same cache composition regardless of
whether it anchors appearance or carries motion.

R32 adds a per-head additive bias to the softmax logits of the FIRST segment
(previous frames), leaving both current-frame segments at 0. One scalar per
head spans the entire past-vs-current axis, because softmax is shift-invariant
per query row: only differences between segments change the distribution. So a
[num_layers, num_heads] table is the complete parameterisation of this axis.

Sign, from the R19 convention (verified against the tau payload -- all 117
SPATIAL heads have margin in [-0.997, -0.159], all 49 TEMPORAL in
[+0.184, +1.000]):

    s < 0  =>  down-weight previous frames  =>  spatial head, look at NOW
    s > 0  =>  up-weight   previous frames  =>  temporal head, look BACK

TWO PARAMETERISATIONS, BOTH BUILT
---------------------------------
``label`` uses the four-way labels R19 actually validated (wrong-arm ~1% on
video/step/block, 11-20x over count-matched random controls) and respects the
abstain: MIXED and DENSE heads get 0 rather than being forced to a side.

``cont`` avoids the threshold entirely -- R19's margin histogram is not
bimodal (~20% of mass within |margin|<0.2), so any cut is uncomfortable. It
grades by margin and attenuates by reconstruction quality:

    s = tanh(margin / T) * clip(1 - best_rel/tau_dense, 0, 1)

The attenuation is what re-introduces the abstain without a label cut: a DENSE
head has best_rel > tau_dense by construction, so its confidence clips to 0.
Note R19's own caveat -- the margin is NOT calibrated (a head with 1-3% of its
mass on the temporal line can read strongly spatial, because restricted softmax
renormalises) -- so tanh is used to keep the map monotone but deliberately not
the identity. Treating the margin as a proportion would be exactly the error
R19 warned about.

NORMALISATION
-------------
Each table is scaled so max|s| == 1, which makes ``b_max`` at the driver a hard
ceiling on the bias any head receives, identically in both parameterisations.
The tables still differ in MEAN strength (``cont`` is deliberately softer, and
gives the 11 MIXED heads a small bias) -- that is a property of the
parameterisation under test, not an artifact to be normalised away.

CONTROLS
--------
``*_rev``   negated. If routed and reversed degrade SYMMETRICALLY the heads are
            already at their preferred allocation and the margin has no
            headroom -- saturation, which is not the same finding as "routing
            does not work". This is the cheapest way to tell those apart.
``*_unif``  every head gets mean(s) over all 360. Same global shift, zero
            per-head variation: kills "you just globally up-weighted the past".
``*_shuf``  the exact multiset of s permuted across heads at a fixed seed. Same
            distribution, wrong assignment: kills "any per-head variation would
            have done". Both controls are needed -- they rule out different
            things.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import torch

PARAMS = ("label", "cont")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tau", type=Path, default=Path("evaluation/r19_tau_flat.pt"),
                   help="written by r19_analyze.py with --tau_route/--tau_dense set")
    p.add_argument("--out", type=Path, default=Path("evaluation/r32_seg_bias.pt"))
    p.add_argument("--temperature", type=float, default=0.5,
                   help="T in tanh(margin/T) for the continuous parameterisation")
    p.add_argument("--seed", type=int, default=0,
                   help="for the shuffled controls")
    p.add_argument("--no_normalize", action="store_true",
                   help="skip the max|s|==1 rescale (b_max stops being a ceiling)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tau.exists():
        raise SystemExit(
            f"{args.tau} not found. It is written by r19_analyze.py once "
            f"--tau_route/--tau_dense are set (the R19 verdict step)."
        )

    tau = torch.load(args.tau, map_location="cpu", weights_only=False)
    margin = tau["margin_flat"].float()
    best_rel = tau["best_rel"].float()
    spatial, temporal = tau["spatial"], tau["temporal"]
    tau_dense = float(tau["thresholds"]["tau_dense"])
    n_layers, n_heads = margin.shape

    # The sign convention is load-bearing: the whole experiment inverts if it is
    # wrong, and it would still run and still produce plausible numbers. Assert
    # it against the labels rather than trusting the comment.
    if spatial.any() and margin[spatial].max() >= 0:
        raise SystemExit(
            f"sign check failed: a SPATIAL head has margin "
            f"{margin[spatial].max():+.3f} >= 0. R32 assumes negative=spatial."
        )
    if temporal.any() and margin[temporal].min() <= 0:
        raise SystemExit(
            f"sign check failed: a TEMPORAL head has margin "
            f"{margin[temporal].min():+.3f} <= 0. R32 assumes positive=temporal."
        )

    tables: dict[str, torch.Tensor] = {}

    # ---- label-based: the validated object, with the abstain respected -----
    label = torch.zeros_like(margin)
    label[spatial] = -1.0
    label[temporal] = +1.0
    tables["label"] = label

    # ---- continuous: graded by margin, attenuated by reconstruction --------
    confidence = (1.0 - best_rel / tau_dense).clamp(0.0, 1.0)
    tables["cont"] = torch.tanh(margin / args.temperature) * confidence

    # ---- normalise so b_max is a ceiling, identically for both -------------
    scales = {}
    for name in PARAMS:
        peak = float(tables[name].abs().max())
        scale = 1.0 if (args.no_normalize or peak == 0.0) else 1.0 / peak
        scales[name] = scale
        tables[name] = tables[name] * scale

    # ---- controls ----------------------------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    for name in PARAMS:
        s = tables[name]
        tables[f"{name}_rev"] = -s
        tables[f"{name}_unif"] = torch.full_like(s, float(s.mean()))
        perm = torch.randperm(s.numel(), generator=g)
        tables[f"{name}_shuf"] = s.reshape(-1)[perm].reshape(s.shape).clone()

    tables["zero"] = torch.zeros_like(margin)

    census = collections.Counter(x for row in tau["labels"] for x in row)
    payload = {
        **tables,
        "margin_flat": tau["margin_flat"],
        "best_rel": tau["best_rel"],
        "labels": tau["labels"],
        "thresholds": tau["thresholds"],
        "temperature": args.temperature,
        "seed": args.seed,
        "normalized": not args.no_normalize,
        "scales": scales,
        "census": dict(census),
        "source_tau": str(args.tau),
        "provenance": (
            "R32 per-head segment bias on the target branch's previous-frames "
            "key segment (Axis 1: past vs current). Derived from R19 "
            "equal-budget disjoint key sets. Sign: negative=spatial (look at "
            "now), positive=temporal (look back). Apply as b_prev = b_max * s, "
            "with both current-frame segments held at 0."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    total = n_layers * n_heads
    print(f"[r32] from {args.tau}  (shape={tau['thresholds']['shape']}, "
          f"tau_route={tau['thresholds']['tau_route']}, tau_dense={tau_dense}, "
          f"{len(tau['cases'])} videos, T={args.temperature})")
    print(f"[r32] R19 census: " + "  ".join(f"{k}={v}" for k, v in sorted(census.items())))
    print(f"[r32] normalize={not args.no_normalize}  scales=" +
          "  ".join(f"{k}={v:.3f}" for k, v in scales.items()))
    print(f"[r32] {'table':<12} {'nonzero':>11}  {'mean|s|':>8} {'max|s|':>7} {'mean(s)':>8}")
    for k in ("zero", "label", "cont", "label_rev", "cont_rev",
              "label_unif", "cont_unif", "label_shuf", "cont_shuf"):
        s = tables[k]
        nz = s.abs() > 1e-9
        mean_abs = float(s[nz].abs().mean()) if bool(nz.any()) else 0.0
        print(f"[r32] {k:<12} {int(nz.sum()):4d}/{total}  {mean_abs:8.3f} "
              f"{float(s.abs().max()):7.3f} {float(s.mean()):+8.4f}")
    print(f"[r32] wrote {args.out}")


if __name__ == "__main__":
    main()
