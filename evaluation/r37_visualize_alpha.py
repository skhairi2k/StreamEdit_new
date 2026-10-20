"""R37 alpha-study figures: what the distance-weighted bg metric sees, on 5 fixed examples.

Per example, four PNGs in evaluation/figures/r37_alpha_study/:
  {video}_e{T}_tolerance.png  Fig 1a  rows = strided frames {0,16,32,48,64};
                                      cols = source | S | w at alpha in {0.1,0.25,0.5,1.0}
  {video}_e{T}_errors.png     Fig 1b  frame 32; rows = 12 arms;
                                      cols = edit | LPIPS | w*LPIPS at the same 4 alphas
  {video}_e{T}_alpha.png      Fig 2   PSNR_w / SSIM_w / LPIPS_w vs alpha in [0,1], one line
                                      per arm; FiVE official drawn as a separate marker
  {video}_e{T}_rings.png      Fig 3   ring means of LPIPS and MSE vs distance r in [0, 2L]
                                      (bin 0.05 L), pooled over the strided frames
plus README.md (one line per figure).

Everything is recomputed here from the renders with bg_metrics (no dependence on the
per-sample CSVs), so this can run as soon as the unit tests pass.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evaluation", "fivebench"))
import bg_metrics as bm  # noqa: E402

# (video, edit type, short description) -- fixed in the R37 plan.
EXAMPLES = [
    ("0011_lucia", 2, "woman -> lion (non-rigid)"),
    ("0058_boat", 3, "white -> pink boat (colour)"),
    ("0011_lucia", 5, "add a dog"),
    ("0042_gym-ball", 6, "remove the gym ball"),
    ("0001_bus", 1, "bus -> jeep (rigid)"),
]
R35_ARMS = ["dino_unblended", "dino_first2", "lpips_unblended", "lpips_first2"]
R36_RHOS = [2, 3, 4, 6, 8, 10, 20, 50]
FIG_ALPHAS = [0.1, 0.25, 0.5, 1.0]

# ---- colours (validated with the dataviz palette validator, light surface #fcfcfb)
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# R36 rho sweep is ordinal -> one blue hue, light (rho=2) -> dark (rho=50); OKLCH L
# 0.75 -> 0.28, adjacent dL >= 0.06 (ordinal check PASS).
RHO_RAMP = ["#73b1ff", "#5f9be8", "#4b86d1", "#3672bb", "#225ea5", "#084a90", "#043971", "#00265d"]
# R35 arms are a 2x2: colour = divergence (categorical slots 2, 3), linestyle = regime.
DIV_COLOR = {"dino": "#eb6834", "lpips": "#1baf7a"}
REGIME_STYLE = {"unblended": (0, (5, 2)), "first2": (0, (1.5, 1.5))}
CMAP_W, CMAP_ERR = "Blues", "Oranges"


def arm_list(out_root: str) -> list[dict]:
    arms = []
    for a in R35_ARMS:
        div, regime = a.split("_")
        arms.append({"stem": f"r35_{a}", "label": f"R35 {a}",
                     "dir": os.path.join(out_root, "r35_arms", f"r35_{a}", "step14"),
                     "color": DIV_COLOR[div], "ls": REGIME_STYLE[regime]})
    for r, c in zip(R36_RHOS, RHO_RAMP):
        arms.append({"stem": f"r36_rho{r}_vp", "label": f"R36 ρ={r}",
                     "dir": os.path.join(out_root, "r36_rho_sweep", f"r36_rho{r}_vp"),
                     "color": c, "ls": "-"})
    return arms


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def img_off(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def ring_means(maps: dict, masks: np.ndarray, L: float, n_bins: int = 40) -> dict:
    """Pixel-pooled mean of the LPIPS and MSE maps per distance ring [k, k+1)*0.05L,
    k < n_bins (r in [0, 2L)), over the non-empty strided frames."""
    width = 0.05 * L
    sums = {k: np.zeros(n_bins) for k in ("lpips", "mse")}
    cnt = np.zeros(n_bins)
    for t in range(masks.shape[0]):
        S = masks[t]
        if not S.any():
            continue
        dist = distance_transform_edt(~S)
        b = np.floor(dist / width).astype(int)
        sel = (~S) & (b < n_bins)
        bb = b[sel]
        cnt += np.bincount(bb, minlength=n_bins)
        for k in sums:
            sums[k] += np.bincount(bb, weights=maps[k][t][sel], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {k: sums[k] / cnt for k in sums} | {"count": cnt}


def compute_example(video: str, T: int, arms: list[dict], args, lpips_model, mc) -> dict:
    alphas = bm.alpha_grid(0.05, 1.0)
    src = masks = None
    per_arm = {}
    for arm in arms:
        s, e, m, _ = bm.load_clip(args.data_root, arm["dir"], T, video, args.stride)
        if src is None:
            src, masks = s, m
        elif not (np.array_equal(s, src) and np.array_equal(m, masks)):
            raise RuntimeError(f"{video} e{T}: source/mask differ across arms ({arm['stem']})")
        res = bm.distance_weighted_bg_metrics(s, e, m, alphas, lpips_model, args.alpha_max)
        off = bm.official_fivebench(s, e, m, mc)
        maps = bm.error_maps(s.astype(np.float32) / 255, e.astype(np.float32) / 255, lpips_model)
        per_arm[arm["stem"]] = {
            "res": res, "off": off, "rings": ring_means(maps, m, res["L"]),
            "edit_err_frame": e[args.err_pos], "lpips_err_frame": maps["lpips"][args.err_pos],
        }
    return {"src": src, "masks": masks, "alphas": alphas, "per_arm": per_arm,
            "L": per_arm[arms[0]["stem"]]["res"]["L"]}


# ------------------------------------------------------------------------- figures


def fig_tolerance(ex: dict, video: str, T: int, desc: str, args, path: str) -> None:
    """Fig 1a: method-independent -- source, S and w(alpha) on the sampled frames."""
    src, masks, L = ex["src"], ex["masks"], ex["L"]
    rows = [p for p in args.tol_pos if p < src.shape[0]]
    ncol = 2 + len(FIG_ALPHAS)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.3 * ncol, 1.4 * len(rows) + 0.9),
                             squeeze=False, facecolor=SURFACE, layout="constrained")
    im_w = None
    for i, p in enumerate(rows):
        S = masks[p]
        dist = distance_transform_edt(~S)
        axes[i, 0].imshow(src[p])
        axes[i, 1].imshow(S, cmap="gray_r", vmin=0, vmax=1)
        for j, a in enumerate(FIG_ALPHAS):
            ax = axes[i, 2 + j]
            if S.any():
                im_w = ax.imshow(bm.weight_map(dist, S, a, L), cmap=CMAP_W, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "empty S", ha="center", va="center", color=INK_2,
                        transform=ax.transAxes)
        frame = p * args.stride
        mark = "  ◀ Fig 1b" if p == args.err_pos else ""
        axes[i, 0].set_ylabel(f"frame {frame}{mark}", color=INK, fontsize=9,
                              fontweight="bold" if mark else "normal")
        for ax in axes[i]:
            img_off(ax)
    for j, title in enumerate(["source", "S (hard mask)"] +
                              [f"w, α={a} (ρ={a * L:.0f}px)" for a in FIG_ALPHAS]):
        axes[0, j].set_title(title, color=INK, fontsize=9)
    fig.suptitle(f"{video} e{T} — {desc} · tolerance weights w = clip(dist / αL, 0, 1), "
                 f"L = {L:.0f}px", color=INK, fontsize=10)
    if im_w is not None:
        cb = fig.colorbar(im_w, ax=axes[:, 2:].ravel().tolist(), shrink=0.6)
        cb.set_label("w", color=INK_2)
    fig.savefig(path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_errors(ex: dict, arms: list[dict], video: str, T: int, desc: str, args,
               path: str) -> None:
    """Fig 1b: frame err_pos, one row per arm: edit | LPIPS | w*LPIPS at 4 alphas."""
    S, L = ex["masks"][args.err_pos], ex["L"]
    dist = distance_transform_edt(~S)
    ws = [bm.weight_map(dist, S, a, L) for a in FIG_ALPHAS]
    lp_all = np.stack([ex["per_arm"][a["stem"]]["lpips_err_frame"] for a in arms])
    vmax = float(np.percentile(lp_all, 99))  # one shared scale per example
    ncol = 2 + len(FIG_ALPHAS)
    fig, axes = plt.subplots(len(arms), ncol, figsize=(2.3 * ncol, 1.3 * len(arms) + 0.9),
                             squeeze=False, facecolor=SURFACE, layout="constrained")
    im = None
    for i, arm in enumerate(arms):
        d = ex["per_arm"][arm["stem"]]
        lp = d["lpips_err_frame"]
        axes[i, 0].imshow(d["edit_err_frame"])
        im = axes[i, 1].imshow(lp, cmap=CMAP_ERR, vmin=0, vmax=vmax)
        for j, w in enumerate(ws):
            axes[i, 2 + j].imshow(w * lp, cmap=CMAP_ERR, vmin=0, vmax=vmax)
        for ax in axes[i, 1:]:
            ax.contour(S, levels=[0.5], colors=INK_2, linewidths=0.5)
        axes[i, 0].set_ylabel(arm["label"], color=INK, fontsize=8)
        for ax in axes[i]:
            img_off(ax)
    for j, title in enumerate(["edit", "LPIPS"] + [f"w·LPIPS, α={a}" for a in FIG_ALPHAS]):
        axes[0, j].set_title(title, color=INK, fontsize=9)
    fig.suptitle(f"{video} e{T} — {desc} · frame {args.err_pos * args.stride} · "
                 f"grey line = ∂S · shared scale (99th pct of LPIPS)", color=INK, fontsize=10)
    cb = fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=0.4)
    cb.set_label("LPIPS (alex)", color=INK_2)
    fig.savefig(path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _legend(fig, arms: list[dict], extra: list | None = None) -> None:
    handles = [Line2D([], [], color=a["color"], linestyle=a["ls"], linewidth=2, label=a["label"])
               for a in arms] + (extra or [])
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
               fontsize=8, labelcolor=INK)


def fig_alpha(ex: dict, arms: list[dict], video: str, T: int, desc: str, args,
              path: str) -> None:
    """Fig 2: clip-level PSNR_w / SSIM_w / LPIPS_w vs alpha; official at its own x."""
    alphas = ex["alphas"]
    x_off = -0.08
    panels = [("psnr_w", "psnr", "PSNR_w (dB) ↑"), ("ssim_w", "ssim", "SSIM_w ↑"),
              ("lpips_w", "lpips", "LPIPS_w ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURFACE)
    for ax, (key, okey, ylabel) in zip(axes, panels):
        style_axes(ax)
        for arm in arms:
            d = ex["per_arm"][arm["stem"]]
            ys = [d["res"]["per_alpha"][a][key] for a in alphas]
            ax.plot(alphas, ys, color=arm["color"], linestyle=arm["ls"], linewidth=1.6)
            ax.plot([x_off], [d["off"][okey]], marker="D", markersize=6, color=arm["color"],
                    markeredgecolor=SURFACE, markeredgewidth=0.8, linestyle="none")
        ax.axvline(0, color=INK_2, linewidth=0.6)
        ax.set_xticks([x_off, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["FiVE\nofficial", "0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(x_off - 0.05, 1.02)
        ax.set_xlabel("α (tolerance band ρ = αL)", color=INK_2, fontsize=9)
        ax.set_title(ylabel, color=INK, fontsize=10)
    fig.suptitle(f"{video} e{T} — {desc} · clip-level score vs α (◆ = FiVE official, "
                 f"not equal to α=0)", color=INK, fontsize=10)
    _legend(fig, arms, [Line2D([], [], marker="D", color=INK_2, linestyle="none",
                               label="FiVE official")])
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_rings(ex: dict, arms: list[dict], video: str, T: int, desc: str, args,
              path: str) -> None:
    """Fig 3: ring-mean LPIPS and MSE vs r / L, r in [0, 2L), bin 0.05 L."""
    centers = (np.arange(40) + 0.5) * 0.05
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, (key, ylabel) in zip(axes, [("lpips", "ring-mean LPIPS"), ("mse", "ring-mean MSE")]):
        style_axes(ax)
        for a in FIG_ALPHAS:  # w reaches 1 at r = αL
            ax.axvline(a, color=GRID, linewidth=1.0, zorder=0)
            ax.text(a, 1.0, f"{a:g}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color=INK_2)
        for arm in arms:
            ax.plot(centers, ex["per_arm"][arm["stem"]]["rings"][key], color=arm["color"],
                    linestyle=arm["ls"], linewidth=1.6)
        ax.set_xlim(0, 2)
        ax.set_xlabel("distance to S, r / L   (grey lines: r = αL, where w reaches 1)",
                      color=INK_2, fontsize=9)
        ax.set_title(ylabel, color=INK, fontsize=10, pad=14)
    fig.suptitle(f"{video} e{T} — {desc} · error vs distance to the object "
                 f"(L = {ex['L']:.0f}px, pooled over strided frames)", color=INK, fontsize=10)
    _legend(fig, arms)
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R37 alpha-study figures (5 examples x 12 arms).")
    p.add_argument("--data_root", default=os.path.expanduser(
        "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--out_root", default=os.path.expanduser("~/Data/dataggen/outputs/five_bench"),
                   help="holds r35_arms/ and r36_rho_sweep/")
    p.add_argument("--out_dir", default=os.path.join(REPO, "evaluation", "figures",
                                                     "r37_alpha_study"))
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--tol_frames", type=int, nargs="+", default=[0, 16, 32, 48, 64],
                   help="Fig 1a rows, as clip frame indices (multiples of --stride)")
    p.add_argument("--err_frame", type=int, default=32, help="Fig 1b frame (clip index)")
    p.add_argument("--alpha_max", type=float, default=0.5)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    for f in args.tol_frames + [args.err_frame]:
        if f % args.stride:
            p.error(f"frame {f} is not a multiple of --stride {args.stride}")
    args.tol_pos = [f // args.stride for f in args.tol_frames]
    args.err_pos = args.err_frame // args.stride
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[r37_fig] args: {json.dumps({k: v for k, v in vars(args).items()})}")
    print(f"[r37_fig] torch {torch.__version__} matplotlib {matplotlib.__version__} "
          f"device {args.device}")
    os.makedirs(args.out_dir, exist_ok=True)

    arms = arm_list(args.out_root)
    lpips_model = bm.build_lpips(args.device)
    mc = bm.build_official_calculator(args.device)
    readme = ["# R37 alpha study", "",
              f"Generated by `evaluation/r37_visualize_alpha.py`; 12 arms "
              f"(4 R35 + 8 R36 ρ), stride {args.stride}.", ""]
    for video, T, desc in EXAMPLES:
        print(f"[r37_fig] {video} e{T}: scoring {len(arms)} arms", flush=True)
        ex = compute_example(video, T, arms, args, lpips_model, mc)
        stem = os.path.join(args.out_dir, f"{video}_e{T}")
        fig_tolerance(ex, video, T, desc, args, f"{stem}_tolerance.png")
        fig_errors(ex, arms, video, T, desc, args, f"{stem}_errors.png")
        fig_alpha(ex, arms, video, T, desc, args, f"{stem}_alpha.png")
        fig_rings(ex, arms, video, T, desc, args, f"{stem}_rings.png")
        base = f"{video}_e{T}"
        readme += [
            f"- `{base}_tolerance.png` — Fig 1a ({desc}): source, S and w at α ∈ "
            f"{{0.1, 0.25, 0.5, 1.0}} on frames {args.tol_frames}; L = {ex['L']:.0f}px.",
            f"- `{base}_errors.png` — Fig 1b: frame {args.err_frame}, one row per arm: edit, "
            f"LPIPS map, w·LPIPS at the 4 α; one colour scale (99th pct of LPIPS over arms).",
            f"- `{base}_alpha.png` — Fig 2: clip-level PSNR_w / SSIM_w / LPIPS_w vs α ∈ [0, 1]; "
            f"◆ = FiVE official (blacks out S, full frame), drawn apart from α = 0.",
            f"- `{base}_rings.png` — Fig 3: ring-mean LPIPS and MSE vs r/L ∈ [0, 2), "
            f"bin 0.05L, pooled over the strided frames.",
        ]
        print(f"[r37_fig] {video} e{T}: 4 figures written", flush=True)
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("\n".join(readme) + "\n")
    print(f"[r37_fig] done: {4 * len(EXAMPLES)} figures + README -> {args.out_dir}")


if __name__ == "__main__":
    main()
