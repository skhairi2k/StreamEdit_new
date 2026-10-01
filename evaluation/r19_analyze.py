"""R19 analysis: four-way head labels, margin distribution, stability.

Consumes the npz written by ``r9_head_profiler.py`` and derives everything --
nothing is read back that was not measured. Margin and best_rel are formed here
rather than stored, so the band shape and the thresholds can be revisited
without touching the GPU job.

    margin   = (err_spat - err_temp) / (err_spat + err_temp)     in [-1, 1]
    best_rel = sqrt(min(err_spat, err_temp) / mean(golden^2))

Four-way routing (svg_routing_strat.md §4). The two failure modes are opposite
and must not be merged:

    best_rel > tau_dense            -> DENSE     neither pattern fits; the head
                                                 is diffuse and abstains rather
                                                 than being handed to whichever
                                                 mask happens to fit less badly
    margin   >  tau_route           -> TEMPORAL
    margin   < -tau_route           -> SPATIAL
    otherwise                       -> MIXED     both patterns fit about equally

THRESHOLDS ARE NOT SET BY DEFAULT. Run with no ``--tau_*`` to get the margin
distribution and a threshold suggestion; labels are only produced once you pass
them explicitly. Fixing a cut before seeing the distribution is how the previous
taxonomy went unquestioned.

CAVEAT WHEN READING THE MARGIN
------------------------------
It is NOT monotone in temporal mass. Restricted softmax renormalises, so a head
with only a few percent of its mass on the temporal line has that mass
over-weighted by the temporal set and lands FURTHER from the dense output than
the spatial set does -- it reads strongly negative. Measured on synthetic heads:
2.6% mass -> -0.89, 17% -> -0.60, 77% -> +0.79. A strongly negative margin
therefore means "spatial OR weakly temporal"; best_rel is what separates them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

LABELS = ("SPATIAL", "TEMPORAL", "MIXED", "DENSE")
_COLOR = {"SPATIAL": "#4C78A8", "TEMPORAL": "#F58518", "MIXED": "#9C9C9C", "DENSE": "#E45756"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r19_head_profile"))
    p.add_argument("--shape", choices=["flat", "disk"], default="flat",
                   help="band shape used for the labels; the other is still reported")
    p.add_argument("--tau_route", type=float, default=None,
                   help="|margin| past this => SPATIAL/TEMPORAL. Unset => distribution only.")
    p.add_argument("--tau_dense", type=float, default=None,
                   help="best_rel above this => DENSE. Unset => distribution only.")
    p.add_argument("--out_csv", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--out_fig", type=Path, default=Path("evaluation/figures"))
    p.add_argument("--out_tau", type=Path, default=None,
                   help="default: evaluation/r19_tau_{shape}.pt")
    return p.parse_args()


def load_cases(root: Path) -> Dict[str, dict]:
    """Load each case's npz and locate its classification point.

    The classification point is the LAST block with finite data -- the deepest
    cache, where the temporal band is narrowest and the head has the most
    context to reveal itself. Videos differ in length (6 or 7 blocks), so it is
    found per case rather than assumed.
    """
    out = {}
    for path in sorted(root.glob("*/r9_scalars.npz")):
        z = np.load(path)
        meta = json.loads(z["meta"].tobytes().decode())
        finite = np.isfinite(z["err_spat"]).any(axis=(1, 2, 3))
        if not finite.any():
            print(f"[r19] WARNING: {path.parent.name} has no finite captures -- skipped")
            continue
        last_block = int(np.nonzero(finite)[0].max())
        step_finite = np.isfinite(z["err_spat"][last_block]).any(axis=(1, 2))
        out[meta["case_id"]] = {
            "z": z, "meta": meta,
            "last_block": last_block,
            "last_step": int(np.nonzero(step_finite)[0].max()),
            "n_blocks_valid": int(finite.sum()),
        }
    return out


def margin_of(z, shape: str) -> np.ndarray:
    """Signed margin over the whole [block, step, layer, head] grid."""
    es = z["err_spat"]
    et = z[f"err_temp_{shape}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (es - et) / (es + et)


def mass_margin_of(z) -> np.ndarray:
    """Budget-free alternative to the MSE margin: which set captured more MASS.

    (W_T - W_S) / (W_T + W_S). Needs no size correction at all -- the two masks
    hold the same number of tokens, so their captured masses are directly
    comparable. Where this disagrees with the MSE margin the head is diffuse:
    the MSE margin inverts for heads whose winning mask captures little mass,
    while mass does not. Recorded for the FLAT band only.
    """
    ws, wt = z["mass_spat"], z["mass_temp"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wt - ws) / (wt + ws)


def best_rel_of(z, shape: str) -> np.ndarray:
    es, et = z["err_spat"], z[f"err_temp_{shape}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.minimum(es, et) / z["golden_sq"])


def classify(margin: np.ndarray, best_rel: np.ndarray,
             tau_route: float, tau_dense: float) -> np.ndarray:
    """Four-way label array (object dtype) with the DENSE test applied first."""
    lab = np.full(margin.shape, "MIXED", dtype=object)
    lab[margin > tau_route] = "TEMPORAL"
    lab[margin < -tau_route] = "SPATIAL"
    lab[best_rel > tau_dense] = "DENSE"          # abstain wins over any margin
    lab[~np.isfinite(margin)] = "MIXED"
    return lab


def check_invariants(cases: Dict[str, dict]) -> bool:
    """The equal-budget claim must be verified from the run, not assumed."""
    print("\n=== recorded invariants (the equal-budget claim, per case) ===")
    print(f"  {'case':>22} {'blocks':>7} {'N':>4} {'span':>6} {'n_spat':>7} {'n_temp':>7} {'gap':>7}")
    ok = True
    for cid, c in sorted(cases.items()):
        z, b = c["z"], c["last_block"]
        ns, nt, sp = int(z["n_spat"][b]), int(z["n_temp"][b]), int(z["span"][b])
        n = int(z["svis"][b]) // int(z["frame_tokens"])
        gap = abs(ns - nt) / max(ns, 1)
        ok &= gap < 0.02
        print(f"  {cid:>22} {c['n_blocks_valid']:7d} {n:4d} {sp:6d} {ns:7d} {nt:7d} {gap:6.2%}"
              + ("" if gap < 0.02 else "   <-- VIOLATED"))
    print(f"  invariant: |n_spat - n_temp| / n_spat < 2%  =>  {'HELD' if ok else 'VIOLATED'}")
    return ok


def describe(margins: np.ndarray, best: np.ndarray, shape: str) -> None:
    """The distribution, printed before any threshold is chosen."""
    m = margins[np.isfinite(margins)]
    b = best[np.isfinite(best)]
    print(f"\n=== margin distribution at the classification point ({shape}, "
          f"{m.size} head-observations) ===")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("  percentile " + " ".join(f"{q:>7d}" for q in qs))
    print("  margin     " + " ".join(f"{np.percentile(m, q):+7.3f}" for q in qs))
    print("  best_rel   " + " ".join(f"{np.percentile(b, q):7.3f}" for q in qs))

    hist, edges = np.histogram(m, bins=20, range=(-1, 1))
    peak = hist.max()
    print("\n  margin histogram (20 bins over [-1, 1]):")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(40 * h / max(peak, 1))
        print(f"   [{lo:+.2f},{hi:+.2f})  {h:5d}  {bar}")

    # Bimodality is the readable signal: a dip around 0 separating two modes.
    centre = hist[8:12].sum()
    flanks = hist[:8].sum() + hist[12:].sum()
    print(f"\n  mass within |margin| < 0.2 : {centre / max(hist.sum(), 1):.1%}")
    print(f"  mass outside               : {flanks / max(hist.sum(), 1):.1%}")
    print("  bimodal (a gap at 0) supports a clean two-way taxonomy; a single")
    print("  mode piled at 0 means the probe does not separate heads here.")

    lo_i = int(np.argmin(hist[6:14])) + 6
    suggest = abs((edges[lo_i] + edges[lo_i + 1]) / 2)
    print(f"\n  suggested --tau_route ~ {max(suggest, 0.05):.2f} (thinnest bin near 0)")
    print(f"  suggested --tau_dense ~ {np.percentile(b, 75):.2f} (75th pct of best_rel)")


def flip_rates(cases: Dict[str, dict], shape: str,
               tau_route: float, tau_dense: float) -> Dict[str, np.ndarray]:
    """Label instability across videos, steps and blocks, per head.

    Reported as the fraction of PAIRS that disagree, so the number does not
    depend on how many videos/steps happen to be available.
    """
    cid = sorted(cases)
    z0 = cases[cid[0]]["z"]
    n_layers, n_heads = z0["err_spat"].shape[2], z0["err_spat"].shape[3]

    def pair_disagreement(stack: np.ndarray) -> np.ndarray:
        """stack [M, L, H] labels -> [L, H] fraction of disagreeing pairs."""
        m = stack.shape[0]
        if m < 2:
            return np.zeros((n_layers, n_heads))
        diff = 0
        for i in range(m):
            for j in range(i + 1, m):
                diff = diff + (stack[i] != stack[j])
        return diff / (m * (m - 1) / 2)

    # across videos: classification point of each case
    per_video = []
    for c in (cases[k] for k in cid):
        z, b, s = c["z"], c["last_block"], c["last_step"]
        per_video.append(classify(margin_of(z, shape)[b, s], best_rel_of(z, shape)[b, s],
                                  tau_route, tau_dense))
    video_flip = pair_disagreement(np.stack(per_video))

    # across steps and blocks: averaged over cases so one clip cannot dominate
    step_flip, block_flip = [], []
    for c in (cases[k] for k in cid):
        z, b, s = c["z"], c["last_block"], c["last_step"]
        mg, br = margin_of(z, shape), best_rel_of(z, shape)
        steps = [classify(mg[b, si], br[b, si], tau_route, tau_dense)
                 for si in range(mg.shape[1]) if np.isfinite(mg[b, si]).any()]
        blocks = [classify(mg[bi, s], br[bi, s], tau_route, tau_dense)
                  for bi in range(mg.shape[0]) if np.isfinite(mg[bi, s]).any()]
        step_flip.append(pair_disagreement(np.stack(steps)))
        block_flip.append(pair_disagreement(np.stack(blocks)))
    return {"video": video_flip,
            "step": np.mean(step_flip, axis=0),
            "block": np.mean(block_flip, axis=0)}


OPPOSITE = {"SPATIAL": "TEMPORAL", "TEMPORAL": "SPATIAL"}


def confusion_rates(cases: Dict[str, dict], shape: str, tau_route: float,
                    tau_dense: float, per_head: np.ndarray) -> Dict[str, np.ndarray]:
    """Wrong-arm rate per head: the share of observations landing in the
    OPPOSITE routing arm, on each of the three axes.

    This is the measure that governs the injection experiment, and it is not
    what ``flip_rates`` reports. ``flip_rates`` counts any label change, so a
    head moving to DENSE registers as a flip -- but DENSE is an *abstain*, not a
    contradiction: it never places the head in the arm meant for its opposite.
    Counting abstains inflates the instability of TEMPORAL heads roughly
    thirty-fold (42.8% vs 1.3% across videos) because they abstain far more
    often than SPATIAL ones.

    Use ``flip_rates`` to describe how reproducible a head's label is; use this
    to decide whether a head can be trusted in a gate. ``dense_*`` is returned
    alongside because abstain churn is a real caveat -- a frozen gate injects
    where per-video routing would have held back -- it is simply a different
    failure mode (over-application, not misdirection).

    Heads whose frozen label is MIXED or DENSE have no opposite arm and are NaN.
    """
    cid = sorted(cases)
    opp = np.full(per_head.shape, None, dtype=object)
    for i in range(per_head.shape[0]):
        for j in range(per_head.shape[1]):
            opp[i, j] = OPPOSITE.get(per_head[i, j])
    routed = np.array([[o is not None for o in row] for row in opp])

    def rate(stack: np.ndarray, target: np.ndarray) -> np.ndarray:
        """stack [M, L, H] labels -> [L, H] fraction equal to `target`."""
        hit = np.zeros(per_head.shape, dtype=float)
        for k in range(stack.shape[0]):
            hit += (stack[k] == target)
        return hit / stack.shape[0]

    dense = np.full(per_head.shape, "DENSE", dtype=object)
    out: Dict[str, np.ndarray] = {}

    per_video = []
    for c in (cases[k] for k in cid):
        z, b, s = c["z"], c["last_block"], c["last_step"]
        per_video.append(classify(margin_of(z, shape)[b, s], best_rel_of(z, shape)[b, s],
                                  tau_route, tau_dense))
    pv = np.stack(per_video)
    out["video"], out["dense_video"] = rate(pv, opp), rate(pv, dense)

    for axis in ("step", "block"):
        wrong, abst = [], []
        for c in (cases[k] for k in cid):
            z, b, s = c["z"], c["last_block"], c["last_step"]
            mg, br = margin_of(z, shape), best_rel_of(z, shape)
            if axis == "step":
                labs = [classify(mg[b, si], br[b, si], tau_route, tau_dense)
                        for si in range(mg.shape[1]) if np.isfinite(mg[b, si]).any()]
            else:
                labs = [classify(mg[bi, s], br[bi, s], tau_route, tau_dense)
                        for bi in range(mg.shape[0]) if np.isfinite(mg[bi, s]).any()]
            st = np.stack(labs)
            wrong.append(rate(st, opp))
            abst.append(rate(st, dense))
        out[axis] = np.mean(wrong, axis=0)
        out[f"dense_{axis}"] = np.mean(abst, axis=0)

    for k in list(out):
        out[k] = np.where(routed, out[k], np.nan)
    return out


def sign_flips(cases: Dict[str, dict], shape: str) -> Dict[str, np.ndarray]:
    """R9-style stability: how often the SIGN of the margin disagrees.

    Needs no thresholds, so it can be reported before any cut is chosen. This is
    the direct analogue of R9's video/step/block flip rates.
    """
    cid = sorted(cases)
    z0 = cases[cid[0]]["z"]
    nl, nh = z0["err_spat"].shape[2], z0["err_spat"].shape[3]

    def disagree(stack):
        m = stack.shape[0]
        if m < 2:
            return np.zeros((nl, nh))
        d = 0
        for i in range(m):
            for j in range(i + 1, m):
                d = d + (np.sign(stack[i]) != np.sign(stack[j]))
        return d / (m * (m - 1) / 2)

    per_video = []
    for c in (cases[k] for k in cid):
        z, b, s = c["z"], c["last_block"], c["last_step"]
        per_video.append(margin_of(z, shape)[b, s])
    video = disagree(np.stack(per_video))

    step, block = [], []
    for c in (cases[k] for k in cid):
        z, b, s = c["z"], c["last_block"], c["last_step"]
        mg = margin_of(z, shape)
        st = [mg[b, si] for si in range(mg.shape[1]) if np.isfinite(mg[b, si]).any()]
        bl = [mg[bi, s] for bi in range(mg.shape[0]) if np.isfinite(mg[bi, s]).any()]
        step.append(disagree(np.stack(st)))
        block.append(disagree(np.stack(bl)))
    return {"video": video, "step": np.mean(step, axis=0), "block": np.mean(block, axis=0)}


def plot_extra(margins, mass_m, best, cases, shape, out_fig: Path):
    """MSE-vs-mass agreement, mass distribution, and the per-block ramp."""
    out_fig.mkdir(parents=True, exist_ok=True)
    ok = np.isfinite(margins) & np.isfinite(mass_m)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax[0].scatter(margins[ok], mass_m[ok], s=5, alpha=0.35,
                       c=np.clip(best[ok], 0, 1.5), cmap="viridis")
    ax[0].axhline(0, color="k", lw=.7, ls=":"); ax[0].axvline(0, color="k", lw=.7, ls=":")
    ax[0].set_xlabel("MSE margin"); ax[0].set_ylabel("mass margin  (W_T-W_S)/(W_T+W_S)")
    ax[0].set_title("agreement (colour = best_rel)", fontsize=10)
    plt.colorbar(sc, ax=ax[0], label="best_rel")
    ax[1].hist(mass_m[ok], bins=60, range=(-1, 1), color="#F58518", alpha=.85)
    ax[1].axvline(0, color="k", lw=.7, ls=":")
    ax[1].set_xlabel("mass margin"); ax[1].set_ylabel("head-observations")
    ax[1].set_title("mass-margin distribution", fontsize=10)
    fig.tight_layout(); fig.savefig(out_fig / f"r19_mse_vs_mass_{shape}.pdf"); plt.close(fig)

    # per-block ramp: does the margin drift as the cache fills?
    fig, ax = plt.subplots(figsize=(6.5, 4))
    maxb = max(c["z"]["err_spat"].shape[0] for c in cases.values())
    for label, arrs in (("MSE margin", [margin_of(c["z"], shape) for c in cases.values()]),):
        means = []
        for b in range(maxb):
            vals = [a[b, c["last_step"]] for a, c in zip(arrs, cases.values())
                    if b < a.shape[0] and np.isfinite(a[b, c["last_step"]]).any()]
            means.append(np.nanmean(np.stack(vals)) if vals else np.nan)
        ax.plot(range(maxb), means, marker="o", label=label)
    ax.axhline(0, color="k", lw=.7, ls=":")
    ax.set_xlabel("block (cache depth grows ->)"); ax.set_ylabel("mean margin over all heads")
    ax.set_title("R19 per-block ramp -- flat = the probe is N-stable", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_fig / f"r19_ramp_{shape}.pdf"); plt.close(fig)
    print(f"[r19] extra figures -> {out_fig}/r19_mse_vs_mass.pdf, r19_ramp.pdf")


def plot_all(margins_flat, margins_disk, labels, flips, out_fig: Path, shape: str):
    out_fig.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margins_flat[np.isfinite(margins_flat)], bins=60, range=(-1, 1),
            color="#4C78A8", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("margin   (< 0 spatial   |   > 0 temporal)")
    ax.set_ylabel("head-observations")
    ax.set_title(f"R19 margin distribution at the classification point ({shape})", fontsize=10)
    fig.tight_layout(); fig.savefig(out_fig / f"r19_margin_hist_{shape}.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ok = np.isfinite(margins_flat) & np.isfinite(margins_disk)
    ax.scatter(margins_flat[ok], margins_disk[ok], s=6, alpha=0.4, color="#4C78A8")
    ax.plot([-1, 1], [-1, 1], color="k", lw=0.8, ls=":")
    r = np.corrcoef(margins_flat[ok], margins_disk[ok])[0, 1] if ok.sum() > 1 else float("nan")
    ax.set_xlabel("margin (flat band)"); ax.set_ylabel("margin (disk)")
    ax.set_title(f"band shape agreement, r = {r:.3f}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_fig / "r19_flat_vs_disk.pdf"); plt.close(fig)

    if labels is None:
        return
    n_layers, n_heads = labels.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(n_layers)
    for name in LABELS:
        counts = (labels == name).sum(axis=1)
        ax.bar(range(n_layers), counts, bottom=bottom, label=name, color=_COLOR[name])
        bottom += counts
    ax.set_xlabel("layer"); ax.set_ylabel("heads"); ax.legend(fontsize=8, ncol=4)
    ax.set_title("R19 head census per layer", fontsize=10)
    fig.tight_layout(); fig.savefig(out_fig / f"r19_layer_hist_{shape}.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in flips.items():
        ax.hist(v.ravel(), bins=20, range=(0, 1), histtype="step", lw=1.5, label=f"{k} flip")
    ax.set_xlabel("fraction of disagreeing pairs"); ax.set_ylabel("heads")
    ax.legend(fontsize=8); ax.set_title("R19 label stability", fontsize=10)
    fig.tight_layout(); fig.savefig(out_fig / f"r19_stability_{shape}.pdf"); plt.close(fig)
    print(f"[r19] figures -> {out_fig}")


def main() -> None:
    args = parse_args()
    if args.out_tau is None:
        args.out_tau = Path(f"evaluation/r19_tau_{args.shape}.pt")
    cases = load_cases(args.profile_root)
    if not cases:
        raise SystemExit(f"[r19] no r9_scalars.npz under {args.profile_root}")
    print(f"[r19] {len(cases)} cases: {', '.join(sorted(cases))}")
    if not check_invariants(cases):
        raise SystemExit("[r19] budget invariant violated -- the margins are not "
                         "comparable; do not interpret them.")

    # Pool the classification point of every case.
    mf, md, br = [], [], []
    for c in cases.values():
        z, b, s = c["z"], c["last_block"], c["last_step"]
        mf.append(margin_of(z, "flat")[b, s])
        md.append(margin_of(z, "disk")[b, s])
        br.append(best_rel_of(z, args.shape)[b, s])
    margins_flat, margins_disk, best = np.stack(mf), np.stack(md), np.stack(br)
    mass_m = np.stack([mass_margin_of(c["z"])[c["last_block"], c["last_step"]]
                       for c in cases.values()])

    margins = margins_flat if args.shape == "flat" else margins_disk
    describe(margins, best, args.shape)

    # --- MSE margin vs budget-free mass margin -------------------------------
    ok = np.isfinite(margins) & np.isfinite(mass_m)
    agree = (np.sign(margins[ok]) == np.sign(mass_m[ok])).mean()
    q = [1, 25, 50, 75, 99]
    print(f"\n=== mass margin (budget-free cross-check, flat band) ===")
    print("  percentile " + " ".join(f"{x:>7d}" for x in q))
    print("  mass marg  " + " ".join(f"{np.percentile(mass_m[ok], x):+7.3f}" for x in q))
    print(f"\n  sign agreement with the MSE margin: {agree:.1%}")
    lo = best[ok] < np.nanpercentile(best, 25)
    hi = best[ok] > np.nanpercentile(best, 75)
    print(f"    among the best-reconstructed quartile (best_rel low) : "
          f"{(np.sign(margins[ok][lo])==np.sign(mass_m[ok][lo])).mean():.1%}")
    print(f"    among the worst-reconstructed quartile (best_rel high): "
          f"{(np.sign(margins[ok][hi])==np.sign(mass_m[ok][hi])).mean():.1%}")
    print("  the two should agree where a mask genuinely reconstructs, and diverge")
    print("  where nothing does -- that divergence IS the diffuse population.")

    # --- R9-style stability: sign flips, no thresholds needed -----------------
    sf = sign_flips(cases, args.shape)
    print(f"\n=== stability of the margin SIGN (R9-style, threshold-free) ===")
    for k, v in sf.items():
        print(f"  {k:>6} flip: mean {v.mean():6.1%}   median {np.median(v):6.1%}   "
              f"heads with 0 flips: {(v == 0).sum():3d}/{v.size}")

    plot_extra(margins, mass_m, best, cases, args.shape, args.out_fig)

    if args.tau_route is None or args.tau_dense is None:
        plot_all(margins_flat, margins_disk, None, {}, args.out_fig, args.shape)
        print("\n[r19] thresholds not given -- distribution only, no labels written.")
        print("      Inspect figures/r19_margin_hist.pdf, then re-run with")
        print("      --tau_route <x> --tau_dense <y> to produce labels.")
        return

    margins = margins_flat if args.shape == "flat" else margins_disk
    # MEDIAN over videos, not mean: labels are ~88% consistent rather than
    # unanimous, so a couple of outlier clips should not drag a head's summary.
    per_head = classify(np.nanmedian(margins, axis=0), np.nanmedian(best, axis=0),
                        args.tau_route, args.tau_dense)
    flips = flip_rates(cases, args.shape, args.tau_route, args.tau_dense)
    conf = confusion_rates(cases, args.shape, args.tau_route, args.tau_dense, per_head)

    print(f"\n=== head census (tau_route={args.tau_route}, tau_dense={args.tau_dense}) ===")
    total = per_head.size
    for name in LABELS:
        n = int((per_head == name).sum())
        print(f"  {name:>9}: {n:4d}/{total}  ({n / total:5.1%})")
    print(f"\n  any-label flip  video {flips['video'].mean():.1%}  "
          f"step {flips['step'].mean():.1%}  block {flips['block'].mean():.1%}")

    # The number that governs the gates: how often a head lands in the arm meant
    # for its opposite. Counting abstains as flips (above) overstates this by an
    # order of magnitude on TEMPORAL heads, which abstain often but rarely invert.
    print(f"\n=== wrong-arm rate (share of observations in the OPPOSITE arm) ===")
    print(f"  {'label':>9} {'n':>4} {'video':>8} {'step':>8} {'block':>8}"
          f" {'clean':>7}  {'(abstain: video)':>18}")
    for name in ("SPATIAL", "TEMPORAL"):
        m = per_head == name
        if not m.any():
            continue
        clean = int((m & (conf["video"] == 0) & (conf["step"] == 0)
                     & (conf["block"] == 0)).sum())
        print(f"  {name:>9} {int(m.sum()):>4} {conf['video'][m].mean():>8.1%}"
              f" {conf['step'][m].mean():>8.1%} {conf['block'][m].mean():>8.1%}"
              f" {clean:>4}/{int(m.sum()):<3} {conf['dense_video'][m].mean():>17.1%}")
    print("  'clean' = heads with zero wrong-arm observations on all three axes.")

    args.out_csv.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_csv / f"r19_head_labels_{args.shape}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "head", "label", "margin_flat", "margin_disk", "best_rel",
                    "video_flip", "step_flip", "block_flip",
                    "wrong_arm_video", "wrong_arm_step", "wrong_arm_block",
                    "dense_video"])
        for i in range(per_head.shape[0]):
            for j in range(per_head.shape[1]):
                w.writerow([i, j, per_head[i, j],
                            f"{np.nanmedian(margins_flat, axis=0)[i, j]:.6f}",
                            f"{np.nanmedian(margins_disk, axis=0)[i, j]:.6f}",
                            f"{np.nanmedian(best, axis=0)[i, j]:.6f}",
                            f"{flips['video'][i, j]:.4f}",
                            f"{flips['step'][i, j]:.4f}",
                            f"{flips['block'][i, j]:.4f}",
                            f"{conf['video'][i, j]:.4f}",
                            f"{conf['step'][i, j]:.4f}",
                            f"{conf['block'][i, j]:.4f}",
                            f"{conf['dense_video'][i, j]:.4f}"])
    print(f"[r19] labels -> {csv_path}")

    torch.save({
        "labels": per_head.tolist(),
        "margin_flat": torch.as_tensor(np.nanmedian(margins_flat, axis=0)),
        "margin_disk": torch.as_tensor(np.nanmedian(margins_disk, axis=0)),
        "best_rel": torch.as_tensor(np.nanmedian(best, axis=0)),
        "spatial": torch.as_tensor(per_head == "SPATIAL"),
        "temporal": torch.as_tensor(per_head == "TEMPORAL"),
        "thresholds": {"tau_route": args.tau_route, "tau_dense": args.tau_dense,
                       "shape": args.shape},
        "cases": sorted(cases),
    }, args.out_tau)
    print(f"[r19] tau -> {args.out_tau}")

    plot_all(margins_flat, margins_disk, per_head, flips, args.out_fig, args.shape)


if __name__ == "__main__":
    main()
