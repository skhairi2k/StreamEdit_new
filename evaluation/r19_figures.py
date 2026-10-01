"""R19 diagnostic figures, kept separate from the labelling pipeline.

Four families, all derived from the npz written by r9_head_profiler.py -- no
GPU, no re-run. Every shape-specific file is suffixed with the band shape so
flat and disk can sit side by side.

  1. scatter_<shape>.pdf            mass margin vs MSE margin, one point per
                                    (video, layer, head) at the classification
                                    point, coloured by best_rel.
  2. hists_<shape>.pdf              the two margin distributions side by side.
  3. bestrel/<video>_<shape>.pdf    best_rel vs MSE margin, ONE PER VIDEO.
                                    One hue per layer, light->dark within the
                                    layer for head 0..11, so a whole layer moving
                                    together is visible as a colour cluster.
  4. ramp/<video>_<shape>.pdf       MSE margin vs block, ONE PER VIDEO, as a
                                    6x5 grid of the 30 layers; each panel draws
                                    all 12 heads. Flat lines = a head that keeps
                                    its character as the cache fills; crossings
                                    through zero = a head that re-tasks.

Why per-video rather than pooled: pooling hides whether a pattern is a property
of the model or of one clip. Anything that only shows up pooled is not evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r19_analyze import load_cases, margin_of, mass_margin_of, best_rel_of  # noqa: E402

# 30 distinguishable base hues, one per layer; heads get shades within the hue.
_BASE = plt.get_cmap("hsv")(np.linspace(0, 1, 31)[:30])


def head_shades(layer: int, n_heads: int) -> np.ndarray:
    """light -> dark ramp inside this layer's hue, for head 0..n_heads-1."""
    base = np.array(to_rgb(_BASE[layer % len(_BASE)]))
    f = np.linspace(0.35, 1.0, n_heads)[:, None]      # 0.35 = washed out, 1.0 = full
    return 1.0 - f * (1.0 - base)                     # blend toward white


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r19_head_profile"))
    p.add_argument("--shape", choices=["flat", "disk"], default="flat")
    p.add_argument("--out", type=Path, default=Path("evaluation/figures/r19"))
    p.add_argument("--only", nargs="*", default=None,
                   help="figure families: scatter hists bybest bestrel ramp")
    return p.parse_args()


def at_point(cases, fn):
    """Stack one [layers, heads] slice per video, taken at the classification point."""
    return np.stack([fn(c["z"])[c["last_block"], c["last_step"]] for c in cases.values()])


# --------------------------------------------------------------------- 1 & 2
def fig_scatter(cases, shape, out: Path):
    mse = at_point(cases, lambda z: margin_of(z, shape)).ravel()
    mass = at_point(cases, mass_margin_of).ravel()
    best = at_point(cases, lambda z: best_rel_of(z, shape)).ravel()
    ok = np.isfinite(mse) & np.isfinite(mass) & np.isfinite(best)
    mse, mass, best = mse[ok], mass[ok], best[ok]

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    sc = ax.scatter(mse, mass, s=5, alpha=0.35, c=np.clip(best, 0, 1.2), cmap="viridis")
    ax.plot([-1, 1], [-1, 1], color="k", lw=0.9, ls="--", label="y = x")
    a, b = np.polyfit(mse, mass, 1)
    xs = np.array([-1, 1])
    ax.plot(xs, a * xs + b, color="crimson", lw=1.2,
            label=f"fit: {a:.3f}x{b:+.3f}  (r={np.corrcoef(mse, mass)[0,1]:.3f})")
    ax.axhline(0, color="k", lw=.5, ls=":"); ax.axvline(0, color="k", lw=.5, ls=":")
    ax.set_xlabel("MSE margin"); ax.set_ylabel("mass margin  (W_T - W_S)/(W_T + W_S)")
    ax.set_title(f"mass vs MSE margin ({shape})\n{mse.size} points = "
                 f"{len(cases)} videos x 30 layers x 12 heads", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    plt.colorbar(sc, ax=ax, label="best_rel  (low = reconstructs)")
    fig.tight_layout(); fig.savefig(out / f"scatter_{shape}.pdf"); plt.close(fig)
    print(f"  scatter_{shape}.pdf")


def fig_hists(cases, shape, out: Path):
    mse = at_point(cases, lambda z: margin_of(z, shape)).ravel()
    mass = at_point(cases, mass_margin_of).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, v, name, col in ((axes[0], mse, "MSE margin", "#4C78A8"),
                             (axes[1], mass, "mass margin", "#F58518")):
        v = v[np.isfinite(v)]
        ax.hist(v, bins=60, range=(-1, 1), color=col, alpha=0.85)
        ax.axvline(0, color="k", lw=.8, ls=":")
        ax.set_xlabel(name)
        ax.set_title(f"{name}   median {np.median(v):+.3f}", fontsize=10)
    axes[0].set_ylabel("head-observations")
    fig.suptitle(f"margin distributions at the classification point ({shape})", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"hists_{shape}.pdf"); plt.close(fig)
    print(f"  hists_{shape}.pdf")


def fig_margin_by_bestrel(cases, shape, out: Path):
    """The margin distribution, resolved by how well the winning mask reconstructed.

    This is the figure that decides the thresholds. A margin histogram pooled
    over all heads mixes two populations: heads the probe measured, and heads it
    failed on. Splitting by best_rel separates them, and the bimodality that is
    absent in the pooled view appears cleanly in the measured half.
    """
    mg = at_point(cases, lambda z: margin_of(z, shape)).ravel()
    br = at_point(cases, lambda z: best_rel_of(z, shape)).ravel()
    ok = np.isfinite(mg) & np.isfinite(br)
    mg, br = mg[ok], br[ok]
    qs = np.percentile(br, [25, 50, 75])

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1])

    # A: stacked composition -- what each margin bin is made of
    ax = fig.add_subplot(gs[0])
    bands = [(0, qs[0], f"best_rel < {qs[0]:.2f}  (Q1, sharpest)", "#1a5c8a"),
             (qs[0], qs[1], f"{qs[0]:.2f} - {qs[1]:.2f}  (Q2)", "#4C78A8"),
             (qs[1], qs[2], f"{qs[1]:.2f} - {qs[2]:.2f}  (Q3)", "#9ecae1"),
             (qs[2], 1e9, f"> {qs[2]:.2f}  (Q4, diffuse)", "#E45756")]
    edges = np.linspace(-1, 1, 41)
    bottom = np.zeros(len(edges) - 1)
    for lo, hi, lab, col in bands:
        h, _ = np.histogram(mg[(br >= lo) & (br < hi)], bins=edges)
        ax.bar(edges[:-1], h, width=np.diff(edges), bottom=bottom, align="edge",
               color=col, label=lab, linewidth=0)
        bottom += h
    ax.axvline(0, color="k", lw=.7, ls=":")
    ax.set_xlabel("MSE margin"); ax.set_ylabel("head-observations")
    ax.set_title("A. what each margin bin is made of", fontsize=10)
    ax.legend(fontsize=7, loc="upper center")

    # B: the sharpening -- tighten the best_rel cut, watch the gap at 0 open
    ax = fig.add_subplot(gs[1])
    for cut, col, ls in ((1e9, "#bbbbbb", "-"), (qs[2], "#9ecae1", "-"),
                         (qs[1], "#4C78A8", "-"), (qs[0], "#1a5c8a", "-")):
        s = mg[br < cut]
        h, e = np.histogram(s, bins=edges, density=True)
        lab = "all heads" if cut > 1e8 else f"best_rel < {cut:.2f}  ({s.size/mg.size:.0%} kept)"
        ax.plot((e[:-1] + e[1:]) / 2, h, color=col, ls=ls, lw=1.6, label=lab)
        near = np.mean(np.abs(s) < 0.2)
        ax.annotate("", xy=(0, 0), xytext=(0, 0))
    ax.axvspan(-0.2, 0.2, color="k", alpha=0.05)
    ax.axvline(0, color="k", lw=.7, ls=":")
    ax.set_xlabel("MSE margin"); ax.set_ylabel("density")
    ax.set_title("B. drop the heads the probe could not measure,\nand the gap at 0 opens",
                 fontsize=10)
    ax.legend(fontsize=7)

    # C: the joint density
    ax = fig.add_subplot(gs[2])
    hb = ax.hexbin(mg, np.clip(br, 0, 1.2), gridsize=45, bins="log", cmap="magma_r")
    for q, lab in zip(qs, ("Q1|Q2", "median", "Q3|Q4")):
        ax.axhline(q, color="#1f77b4", lw=.8, ls="--")
        ax.text(0.99, q, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=6, color="#1f77b4")
    ax.axvline(0, color="k", lw=.7, ls=":")
    ax.set_xlabel("MSE margin"); ax.set_ylabel("best_rel")
    ax.set_title("C. joint density (log counts)", fontsize=10)
    plt.colorbar(hb, ax=ax, label="count")

    fig.suptitle(f"R19 -- margin resolved by reconstruction quality ({shape})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / f"margin_by_bestrel_{shape}.pdf"); plt.close(fig)
    print(f"  margin_by_bestrel_{shape}.pdf")


# ------------------------------------------------------------------------- 3
def fig_bestrel_per_video(cases, shape, out: Path):
    d = out / "bestrel"; d.mkdir(parents=True, exist_ok=True)
    for cid, c in sorted(cases.items()):
        z, b, s = c["z"], c["last_block"], c["last_step"]
        mg = margin_of(z, shape)[b, s]                       # [L, H]
        br = best_rel_of(z, shape)[b, s]
        nl, nh = mg.shape
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for layer in range(nl):
            cols = head_shades(layer, nh)
            ax.scatter(mg[layer], br[layer], s=26, color=cols,
                       edgecolors="none", alpha=0.9)
        ax.axvline(0, color="k", lw=.6, ls=":")
        ax.axhline(1.0, color="crimson", lw=.9, ls="--")
        ax.text(0.98, 1.02, "best_rel = 1: error equals the output (useless)",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7, color="crimson")
        ax.set_yscale("log")
        ax.set_xlabel("MSE margin   (< 0 spatial   |   > 0 temporal)")
        ax.set_ylabel("best_rel  (log scale, lower = better reconstruction)")
        ax.set_title(f"{cid}  --  best_rel vs MSE margin ({shape})\n"
                     f"one hue per layer, light->dark = head 0->{nh-1}", fontsize=10)
        fig.tight_layout(); fig.savefig(d / f"{cid}_{shape}.pdf"); plt.close(fig)
    print(f"  bestrel/*_{shape}.pdf   ({len(cases)} files)")


# ------------------------------------------------------------------------- 4
def fig_ramp_per_video(cases, shape, out: Path):
    d = out / "ramp"; d.mkdir(parents=True, exist_ok=True)
    for cid, c in sorted(cases.items()):
        z, s = c["z"], c["last_step"]
        mg = margin_of(z, shape)[:, s]                       # [blocks, L, H]
        valid = np.isfinite(mg).any(axis=(1, 2))
        blocks = np.nonzero(valid)[0]
        frames = (z["svis"][blocks] // int(z["frame_tokens"])).astype(int)
        nl, nh = mg.shape[1], mg.shape[2]

        fig, axes = plt.subplots(6, 5, figsize=(16, 15), sharex=True, sharey=True)
        for layer in range(nl):
            ax = axes[layer // 5][layer % 5]
            cols = head_shades(layer, nh)
            for h in range(nh):
                ax.plot(frames, mg[blocks, layer, h], color=cols[h],
                        lw=1.1, marker="o", markersize=2.5)
            ax.axhline(0, color="k", lw=.6, ls=":")
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(f"layer {layer}", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("frames visible", fontsize=8)
        for row in axes:
            row[0].set_ylabel("MSE margin", fontsize=8)
        fig.suptitle(f"{cid}  --  MSE margin as the cache fills ({shape})\n"
                     f"one panel per layer, {nh} heads each (light->dark). "
                     f"Flat = stable head; a crossing of 0 = the label flips.",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(d / f"{cid}_{shape}.pdf"); plt.close(fig)
    print(f"  ramp/*_{shape}.pdf      ({len(cases)} files)")


def main() -> None:
    args = parse_args()
    cases = load_cases(args.profile_root)
    if not cases:
        raise SystemExit(f"[r19] no r9_scalars.npz under {args.profile_root}")
    args.out.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {"scatter", "hists", "bybest", "bestrel", "ramp"}
    print(f"[r19] {len(cases)} videos, shape={args.shape} -> {args.out}")
    if "scatter" in want:
        fig_scatter(cases, args.shape, args.out)
    if "hists" in want:
        fig_hists(cases, args.shape, args.out)
    if "bybest" in want:
        fig_margin_by_bestrel(cases, args.shape, args.out)
    if "bestrel" in want:
        fig_bestrel_per_video(cases, args.shape, args.out)
    if "ramp" in want:
        fig_ramp_per_video(cases, args.shape, args.out)


if __name__ == "__main__":
    main()
