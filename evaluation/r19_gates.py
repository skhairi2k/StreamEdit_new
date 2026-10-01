"""R19 blocking gates: prove the key sets measure head type, not key count.

Two synthetic checks on random / hand-built q,k,v at the real shapes. Both must
pass before any number from the model is interpreted. Runs on CPU in seconds.

    GATE A -- NULL
        A head with NO specialisation (random q,k,v) must score margin ~0 under
        both band shapes. This is the check the previous key sets failed: with
        1560 spatial keys against 18 temporal ones, a purely random head scored
        -0.97, i.e. it would have been labelled "strongly spatial". Any residual
        here is a budget mismatch, so the gate also reports achieved key counts.

    GATE B -- N-STABILITY
        A FIXED synthetic temporal head (mass on the same position in nearby
        frames, unchanged as the cache grows) must keep a flat margin as N goes
        3 -> 21. Includes a POWER CONTROL: the same head run through the
        superseded unequal sets must visibly drift (measured 0.545, and it even
        flips sign at N=3), because a check that cannot fail proves nothing.
        A drifting margin under the equal sets would mean per-block comparisons
        were measuring the probe rather than the head.

Exits non-zero if either gate fails, so it can gate the sbatch step.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r9_head_profiler import FRAME_TOKENS, MAP_HW, N_SAMPLE_ROWS, build_key_sets  # noqa: E402

N_HEADS, HEAD_DIM = 12, 64
BLOCK_FRAMES = 3
# Tolerances. The null margin is the headline: |m| <= NULL_TOL means an
# unspecialised head is not pushed toward either label. Budget mismatch feeds
# directly into it, hence the separate, tighter budget check.
NULL_TOL = 0.10
BUDGET_TOL = 0.05
DRIFT_TOL = 0.15
# Logit bias on the synthetic temporal line. Tuned so the head holds ~50-60% of
# its mass there: strong enough to be unambiguously temporal, weak enough that
# the margin stays off the +1.0 ceiling where drift would be invisible.
TEMPORAL_AMP = 8.0
# The unequal-budget control must drift at least this much, else gate B has no
# power and passing it would mean nothing.
POWER_MIN = 0.25


def sampled_rows(s_q: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(s_q, size=min(N_SAMPLE_ROWS, s_q), replace=False))


def probe(q, k, v, spatial, temporal, bias=None, line=None):
    """SVG Algorithm 1 on one synthetic head-set: returns (mass_on_line, margin).

    Masking is applied with -inf BEFORE softmax, so the kept keys renormalise to
    sum to 1 -- never zero the weights afterwards, which would leave the output
    scaled by the retained mass and confound the comparison.
    """
    logits = torch.einsum("phd,khd->hpk", q, k) / math.sqrt(q.shape[-1])
    if bias is not None:
        logits = logits + bias[None]
    probs = logits.softmax(-1)
    o_full = torch.einsum("hpk,khd->hpd", probs, v)
    mass = (probs * line[None]).sum(-1).mean().item() if line is not None else float("nan")

    def restricted(mask):
        masked = logits.masked_fill(~mask[None], float("-inf"))
        return torch.einsum("hpk,khd->hpd", masked.softmax(-1), v)

    err_s = ((restricted(spatial) - o_full) ** 2).mean(dim=(1, 2))
    err_t = ((restricted(temporal) - o_full) ** 2).mean(dim=(1, 2))
    margin = ((err_s - err_t) / (err_s + err_t)).mean().item()
    return mass, margin


def make_geometry(frames_vis: int, seed: int = 0):
    """Sampled query rows in the current block + their (frame, position)."""
    s_q = BLOCK_FRAMES * FRAME_TOKENS
    rows = sampled_rows(s_q, seed)
    rows_t = torch.as_tensor(rows)
    row_frame = frames_vis - BLOCK_FRAMES + rows_t // FRAME_TOKENS
    row_pos = rows_t % FRAME_TOKENS
    return rows_t, row_frame, row_pos


def gate_a_null(frames_vis: int = 18, seed: int = 0) -> bool:
    """Random q,k,v -- no structure at all -- must not be pushed toward a label."""
    torch.manual_seed(seed)
    _, row_frame, row_pos = make_geometry(frames_vis, seed)
    s_k = frames_vis * FRAME_TOKENS
    P = row_frame.shape[0]

    q = torch.randn(P, N_HEADS, HEAD_DIM)
    k = torch.randn(s_k, N_HEADS, HEAD_DIM)
    v = torch.randn(s_k, N_HEADS, HEAD_DIM)

    print(f"\n=== GATE A -- null control (N={frames_vis}, {P} rows, random q/k/v) ===")
    print(f"  {'shape':>6} {'spatial keys':>13} {'temporal keys':>14} "
          f"{'budget gap':>11} {'margin':>9}  verdict")
    ok = True
    for shape in ("flat", "disk"):
        spatial, temporal = build_key_sets(row_frame, row_pos, frames_vis, shape)
        ns = spatial.sum(1).float().mean().item()
        nt = temporal.sum(1).float().mean().item()
        gap = abs(ns - nt) / ns
        _, margin = probe(q, k, v, spatial, temporal)
        good = abs(margin) <= NULL_TOL and gap <= BUDGET_TOL
        ok &= good
        print(f"  {shape:>6} {ns:13.0f} {nt:14.0f} {gap:10.2%} "
              f"{margin:+9.4f}  {'PASS' if good else 'FAIL'}")
    print(f"  tolerances: |margin| <= {NULL_TOL}, budget gap <= {BUDGET_TOL:.0%}")
    print(f"  (the superseded key sets -- 1560 spatial vs 18 temporal -- scored -0.97 here)")
    return ok


def temporal_head(row_frame, row_pos, frames_vis, amp=TEMPORAL_AMP, seed=0):
    """A synthetic head that reads the same position across every other frame.

    Implemented as a FLAT logit bias `amp` on the whole temporal line, on top of
    random q/k/v. Flatness is what makes the head genuinely fixed in N: the line
    carries N-1 boosted keys while the diffuse background carries N*L, so the
    two grow together and the head's mass fraction stays ~constant (~50-60% at
    the default amp). A decaying bias would saturate while the background kept
    growing, so the head would drift toward diffuse on its own and gate B would
    be measuring the head instead of the probe.

    Returns (q, k, v, bias, line) where `line` marks the boosted keys.
    """
    g = torch.Generator().manual_seed(seed)
    s_k = frames_vis * FRAME_TOKENS
    P = row_frame.shape[0]
    q = torch.randn(P, N_HEADS, HEAD_DIM, generator=g)
    k = torch.randn(s_k, N_HEADS, HEAD_DIM, generator=g)
    v = torch.randn(s_k, N_HEADS, HEAD_DIM, generator=g)
    bias = torch.zeros(P, s_k)
    line = torch.zeros(P, s_k, dtype=torch.bool)
    for p in range(P):
        f0 = int(row_frame[p])
        for f in range(frames_vis):
            if f == f0:
                continue
            j = f * FRAME_TOKENS + int(row_pos[p])
            bias[p, j] = amp
            line[p, j] = True
    return q, k, v, bias, line


def legacy_key_sets(row_frame, row_pos, frames_vis):
    """The superseded UNEQUAL sets: own frame (1560) vs one position (N).

    Used only as gate B's power control -- a check that never fails proves
    nothing, so the same head must visibly drift under these.
    """
    s_k = frames_vis * FRAME_TOKENS
    key_idx = torch.arange(s_k)
    spatial = (key_idx // FRAME_TOKENS)[None, :] == row_frame[:, None]
    temporal = (key_idx % FRAME_TOKENS)[None, :] == row_pos[:, None]
    return spatial, temporal


def gate_b_stability(seed: int = 0) -> bool:
    """A fixed temporal head must score the same margin at every cache depth."""
    print(f"\n=== GATE B -- N-stability (fixed synthetic TEMPORAL head, amp={TEMPORAL_AMP}) ===")
    print(f"  {'N':>3} {'span':>6} {'mass on line':>13} {'margin flat':>12} "
          f"{'margin disk':>12} {'UNEQUAL (control)':>18}")
    margins = {"flat": [], "disk": [], "legacy": []}
    for frames_vis in (3, 6, 9, 12, 15, 18, 21):
        _, row_frame, row_pos = make_geometry(frames_vis, seed)
        q, k, v, bias, line = temporal_head(row_frame, row_pos, frames_vis, seed=seed)
        row, mass = {}, 0.0
        for shape in ("flat", "disk"):
            spatial, temporal = build_key_sets(row_frame, row_pos, frames_vis, shape)
            mass, margin = probe(q, k, v, spatial, temporal, bias, line)
            margins[shape].append(margin)
            row[shape] = margin
        sp_l, tp_l = legacy_key_sets(row_frame, row_pos, frames_vis)
        _, m_legacy = probe(q, k, v, sp_l, tp_l, bias, line)
        margins["legacy"].append(m_legacy)
        span = int(round(FRAME_TOKENS / (frames_vis - 1)))
        print(f"  {frames_vis:3d} {span:6d} {mass:12.0%} {row['flat']:+12.4f} "
              f"{row['disk']:+12.4f} {m_legacy:+18.4f}")

    ok = True
    for shape in ("flat", "disk"):
        vals = margins[shape]
        drift = max(vals) - min(vals)
        sign_ok = all(m > 0 for m in vals)
        good = drift <= DRIFT_TOL and sign_ok
        ok &= good
        print(f"  {shape:>6}: drift {drift:.4f} (tol {DRIFT_TOL}), reads temporal at "
              f"every N: {sign_ok}  {'PASS' if good else 'FAIL'}")

    # Power control: the check must be capable of detecting drift when it exists.
    legacy_drift = max(margins["legacy"]) - min(margins["legacy"])
    powered = legacy_drift >= POWER_MIN
    ok &= powered
    print(f"  control: the SAME head through the superseded unequal sets drifts "
          f"{legacy_drift:.4f} (need >= {POWER_MIN})  {'PASS' if powered else 'FAIL'}")
    print("  -> a fixed temporal head must read temporal at every cache depth with a")
    print("     flat margin; the unequal control shows the check can detect drift.")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames_vis", type=int, default=18,
                   help="cache depth for the null gate (default: the classification point)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    a = gate_a_null(args.frames_vis, args.seed)
    b = gate_b_stability(args.seed)

    print("\n" + "=" * 62)
    print(f"  GATE A (null)        : {'PASS' if a else 'FAIL'}")
    print(f"  GATE B (N-stability) : {'PASS' if b else 'FAIL'}")
    if not (a and b):
        print("  -> BLOCKED: do not launch profiling or interpret any margin.")
        sys.exit(1)
    print("  -> both gates pass; the key sets measure head type, not key count.")


if __name__ == "__main__":
    main()
