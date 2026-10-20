"""R41 grids -- per clip, what each binary cut marks as foreground next to the continuous map.

One page per clip. Per combo, ``2 + len(thetas)`` rows over the SAME latent tokens:

    m                    the chosen draft's absolute-normalised divergence (viridis, [0, 1])
    continuous rho       the stored budget-linear field the binary arms are compared to
    binary rho @ theta   one row per cut, ascending theta, labelled with the clip's FG share

All rho rows share one inferno scale over [2, 50], so a binary row reads directly against
the continuous one: black = rho 2 (source kept), white = rho 50 (source released).

Inputs come from r41_config.sh for --draft_n (which must equal its DRAFT_N): the fields and
continuous rho of R35 (N=15) or R38 (N=5), and ``r41_rho/n{N}/{arm}``. Any missing npz is an
[ERROR] for that page and a non-zero exit -- a grid with a silently blank row looks like a
real one. Runs no model: local / CPU.

Example
-------
    python evaluation/r41_grids.py --draft_n 5 --cases evaluation/r7_anchor_manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r41_config  # noqa: E402
from r31_stage3_grids import evenly_spaced, load_cases  # noqa: E402

TAU_MIN, TAU_MAX = 2.0, 50.0
COMBO_LABEL = {"dino_unblended": "DINO · unblended", "dino_tgt09": "DINO · τ = 0.9",
               "lpips_unblended": "LPIPS · unblended", "lpips_tgt09": "LPIPS · τ = 0.9"}


def build_page(cfg: dict, name: str, T: int, thetas: Sequence[float], out: Path,
               n_frames: int, cmap_m: str, cmap_rho: str, dpi: int) -> None:
    rows: List[tuple] = []                     # (label, field [F, H, W], kind)
    for combo in cfg["combos"]:
        src = cfg["src_name"][combo]
        rel = Path(f"edit{T}") / f"{name}.npz"
        zm = np.load(cfg["div_abs_root"] / src / rel)
        gh, gw = int(zm["grid_h"]), int(zm["grid_w"])
        shape = (-1, gh, gw)
        rows.append((f"{COMBO_LABEL[combo]}\nm = d/{float(zm['div_max']):.3g}",
                     zm["m"].reshape(shape), "m"))
        rows.append((f"{COMBO_LABEL[combo]}\ncontinuous ρ",
                     np.load(cfg["cont_rho_root"] / src / rel)["rho"].reshape(shape), "rho"))
        for th in thetas:
            arm = next(a["arm"] for a in cfg["arms"]
                       if a["combo"] == combo and a["theta"] == th)
            rho = np.load(cfg["rho_root"] / arm / rel)["rho"]
            rows.append((f"{COMBO_LABEL[combo]}\nbinary θ={th:g} (FG {(rho == TAU_MAX).mean():.1%})",
                         rho.reshape(shape), "rho"))

    n_lat = rows[0][1].shape[0]
    lat_idx = evenly_spaced(n_lat, n_frames)
    n_rows, n_cols = len(rows), len(lat_idx)
    fig_h = 0.55 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(0.95 * n_cols + 1.6, fig_h),
                             squeeze=False)
    im = {"m": None, "rho": None}
    for r, (label, field, kind) in enumerate(rows):
        for j, f in enumerate(lat_idx):
            ax = axes[r][j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if kind == "m":
                im["m"] = ax.imshow(field[f], cmap=cmap_m, vmin=0.0, vmax=1.0,
                                    interpolation="nearest", aspect="auto")
            else:
                im["rho"] = ax.imshow(field[f], cmap=cmap_rho, vmin=TAU_MIN, vmax=TAU_MAX,
                                      interpolation="nearest", aspect="auto")
            if r == 0:
                ax.set_title(f"f{f}", fontsize=5, pad=1.5)
        axes[r][0].set_ylabel(label, fontsize=4.6, rotation=0, ha="right", va="center",
                              labelpad=30)

    top = 1.0 - 0.45 / fig_h
    fig.subplots_adjust(left=0.17, right=0.90, top=top, bottom=0.15 / fig_h,
                        wspace=0.03, hspace=0.10)
    fig.suptitle(f"R41 {name} (edit{T}) — draft N={cfg['draft_n']}: m, continuous ρ and "
                 f"binary ρ (ρ=2 if m<θ else 50) per combo\n{n_lat} latent frames, "
                 f"{n_cols} shown; all ρ rows on one [2, 50] scale (black = source kept)",
                 fontsize=6.5, y=1.0 - 0.06 / fig_h)
    cax_m = fig.add_axes([0.905, 0.55, 0.010, 0.30])
    cb_m = fig.colorbar(im["m"], cax=cax_m)
    cb_m.set_label("m [0, 1]", fontsize=4.6)
    cb_m.ax.tick_params(labelsize=4)
    cax_r = fig.add_axes([0.905, 0.15, 0.010, 0.30])
    cb_r = fig.colorbar(im["rho"], cax=cax_r)
    cb_r.set_label("ρ [2, 50]", fontsize=4.6)
    cb_r.ax.tick_params(labelsize=4)

    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft_n", type=int, required=True, choices=(5, 15))
    ap.add_argument("--cases", type=Path, required=True,
                    help="JSON list of {video_name, edit_type}; the 419-entry "
                         "evaluation/r7_anchor_manifest.json for the full bench")
    ap.add_argument("--out_dir", type=Path,
                    help="default: evaluation/figures/r41_n{N}_grids")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--cmap_m", default="viridis")
    ap.add_argument("--cmap_rho", default="inferno")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args(argv)

    cfg = r41_config.load(args.draft_n)
    thetas = sorted(cfg["thetas"])                 # ascending: FG shrinks down each block
    out_dir = args.out_dir or (r41_config.REPO / "evaluation" / "figures"
                               / f"r41_n{args.draft_n}_grids")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.cases)
    n_ok = 0
    for c in cases:
        name, T = c["video_name"], int(c["edit_type"])
        out = out_dir / f"edit{T}_{name}.png"
        try:
            build_page(cfg, name, T, thetas, out, args.n_frames, args.cmap_m,
                       args.cmap_rho, args.dpi)
            print(f"[ok] {out.name}", flush=True)
            n_ok += 1
        except Exception as ex:          # record and continue -- never swallow silently
            print(f"[ERROR] {name} edit{T}: {type(ex).__name__}: {ex}", flush=True)
    print(f"[r41_grids] {n_ok}/{len(cases)} pages -> {out_dir}")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
