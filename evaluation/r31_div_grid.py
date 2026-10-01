"""R31 stage 2: the per-clip diagnostic page, in BOTH image space and latent space.

One page per clip, 12 rows, split into two halves so the reduction is visible rather than
assumed:

  IMAGE SPACE (what was measured)
    source video                              480x832
    stage-1 unblended target (index 7)        480x832
    lpips        native map                   480x832
    dino_patch   native map                   30x52    <- natively the token grid already
    normals      native map                   480x832
    latent       native map                   60x104   <- natively the VAE latent already

  LATENT SPACE (what the blend actually gates)
    z_src        clean VAE encode             30x52 (PCA->RGB)
    z_trg        unblended target at index 7  30x52 (PCA->RGB)
    lpips        reduced field                30x52
    dino_patch   reduced field                30x52
    normals      reduced field                30x52
    latent       reduced field                30x52

⚠ ONLY TWO ARMS GAIN ANYTHING FROM THE IMAGE HALF. `lpips` and `normals` are genuinely
480x832 before the 16x16 area-mean, so their two rows differ and that difference IS what
the reduction discards. `dino_patch` is natively 30x52 (DINOv3 patch 16) and `latent`
natively 60x104, so for those the image-space row is the same field at (nearly) the same
resolution -- kept for uniformity, not because it adds signal.

⚠ `z_src` IS NOT STEP-DEPENDENT. The source branch re-noises the same clean encode at every
denoising step and never evolves an x0 of its own, so "the source latent at step 7" is just
the VAE encode. The row is labelled accordingly rather than implying a trajectory.

(A 5th arm, `depth`, was specced and dropped 2026-09-11 -- monocular relative depth needs
an affine alignment R31 has no mask to fit. See r31_divergence.py's docstring.)

RENDERING NOTES
---------------
* Divergence rows use the NORMALISED ``m`` (latent half) or a per-row min-max of the native
  map (image half), on one shared colormap with a colorbar.
* ``z_src``/``z_trg`` are 16-channel tensors, so they are projected to RGB by PCA over the
  channel axis, FITTED JOINTLY on both so the two rows share one basis and are directly
  comparable. A separate fit per row would recolour them independently.
* Upsampling of any 30x52 field is NEAREST, deliberately: each cell is exactly one token
  the blend gates. A smooth upsample would hide the granularity the mechanism has.
* Every divergence row label carries the arm's RAW ``d_min``/``d_max``. ``m`` alone cannot
  tell "a strong, well-localised edit" from "barely changed, and per-clip normalisation
  stretched the noise to full range" -- the raw span can.

COLUMNS default to ALL LATENT FRAMES, so page width VARIES BY CLIP. Measured over the 22
cases: latent frames run 9 / 12 / 15 / 18 / 21 / 24 with counts 1 / 2 / 2 / 10 / 6 / 1 --
the mode is 18 (a 69-pixel-frame clip), NOT 21. Deliberately left variable: this is a
per-clip diagnostic and dropping frames to equalise width would hide real content.
``--n_frames K`` evenly subsamples K if a fixed width is wanted. There is no all-pixel-frame mode: half the rows exist only per latent frame
and would have to be repeated 4x.

This script runs no model. Every value and frame already exists on disk, so it is
local/CPU and needs no GPU allocation.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_ROWS: Sequence[str] = ("lpips", "dino_patch", "normals", "latent")


def load_frames(path: Path) -> List[Image.Image]:
    if path.is_dir():
        files = sorted(path.glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no PNG frames in {path}")
        return [Image.open(f).convert("RGB") for f in files]
    from diffusers.utils import load_video
    return [f.convert("RGB") for f in load_video(str(path))]


def pixel_index_for_latent(f: int) -> int:
    """Representative pixel frame for latent frame ``f`` -- the LAST of its group.

    Latent frame 0 <-> pixel 0; latent f >= 1 <-> pixels [4f-3, 4f]. Matches
    ``r31_divergence.native_frame_indices``, so the persisted native maps line up with the
    image rows column for column.
    """
    return 0 if f == 0 else 4 * f


def latent_to_rgb(z_a: np.ndarray, z_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project two ``[F, C, h, w]`` latents to ``[F, h, w, 3]`` RGB by a JOINT channel PCA.

    Fitted on both stacks together so they share one basis: a per-stack fit would give the
    source and target unrelated colour conventions and make them impossible to compare by
    eye, which is the entire reason both are on the page.
    """
    fa = z_a.transpose(0, 2, 3, 1).reshape(-1, z_a.shape[1])
    fb = z_b.transpose(0, 2, 3, 1).reshape(-1, z_b.shape[1])
    both = np.concatenate([fa, fb], axis=0).astype(np.float64)
    mu = both.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(both - mu, full_matrices=False)
    proj = (both - mu) @ vt[:3].T
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    proj = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0.0, 1.0)
    n = fa.shape[0]
    out_a = proj[:n].reshape(z_a.shape[0], z_a.shape[2], z_a.shape[3], 3)
    out_b = proj[n:].reshape(z_b.shape[0], z_b.shape[2], z_b.shape[3], 3)
    return out_a, out_b


def pool2x2(z: np.ndarray) -> np.ndarray:
    """``[F, h, w, 3]`` 60x104 -> 30x52 by the DiT's own 2x2 patchify mean."""
    f, h, w, c = z.shape
    return z.reshape(f, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--div_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_div"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--target_root", type=Path, required=True,
                   help=".../r31_stage1/r31_unblended_s7/step06")
    p.add_argument("--latent_dir", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_latents"),
                   help="Stage-1 npz root, for the z_src / z_trg rows")
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("evaluation/figures/r31_div_grids"))
    p.add_argument("--n_frames", type=int, default=-1,
                   help="-1 = every latent frame (default); K > 0 = evenly subsample K")
    p.add_argument("--cmap", type=str, default="magma")
    p.add_argument("--dpi", type=int, default=110)
    args = p.parse_args(argv)

    div_root = args.div_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    latent_dir = args.latent_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(args.cases.expanduser().read_text())
    n_ok = 0
    for c in cases:
        name, T = c["video_name"], int(c["edit_type"])
        try:
            fields: Dict[str, np.ndarray] = {}
            natives: Dict[str, np.ndarray] = {}
            spans: Dict[str, Tuple[float, float]] = {}
            missing: List[str] = []
            for arm in ARM_ROWS:
                f = div_root / arm / f"edit{T}" / f"{name}.npz"
                if not f.exists():
                    missing.append(arm)
                    continue
                z = np.load(f)
                gh, gw = int(z["grid_h"]), int(z["grid_w"])
                fields[arm] = z["m"].reshape(-1, gh, gw)
                spans[arm] = (float(z["d_min"]), float(z["d_max"]))
                if "d_native" in z.files:
                    natives[arm] = z["d_native"].astype(np.float32)
            if not fields:
                raise FileNotFoundError(f"no divergence npz for {name} under {div_root}")

            n_lat = next(iter(fields.values())).shape[0]
            lat_idx = (list(range(n_lat)) if args.n_frames < 0
                       else sorted(set(np.linspace(0, n_lat - 1, args.n_frames)
                                       .round().astype(int).tolist())))

            src = load_frames(data_root / "videos" / f"{name}.mp4")
            trg = load_frames(target_root / f"edit{T}" / name)
            n_have = min(len(src), len(trg))
            lat_idx = [f for f in lat_idx if pixel_index_for_latent(f) < n_have]
            col_idx = [pixel_index_for_latent(f) for f in lat_idx]
            n_cols = len(col_idx)

            # z_src / z_trg -> RGB on the token grid, joint PCA basis.
            zp = latent_dir / f"edit{T}" / f"{name}.npz"
            zrgb_src = zrgb_trg = None
            if zp.exists():
                zz = np.load(zp)
                a, b = latent_to_rgb(zz["z_src"], zz["z_trg"])
                zrgb_src, zrgb_trg = pool2x2(a), pool2x2(b)

            n_rows = 12
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(max(1.0, 1.05 * n_cols), 0.70 * n_rows),
                                     squeeze=False)
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

            def label(r: int, text: str) -> None:
                axes[r][0].set_ylabel(text, fontsize=4.6, rotation=0, ha="right",
                                      va="center", labelpad=30)

            # ---------------- IMAGE SPACE ----------------
            for j, pi in enumerate(col_idx):
                axes[0][j].imshow(src[pi])
                axes[1][j].imshow(trg[pi])
                axes[0][j].set_title(f"f{lat_idx[j]}", fontsize=4.6, pad=1.5)
            label(0, "source\n(image)")
            # Stage 1 is a 7-step run measured at its last index, so this single row is
            # BOTH the measured target and the fully denoised render -- there is no
            # separate reference row to draw.
            label(1, "target step6\nfinal (image)")

            im = None
            for r, arm in enumerate(ARM_ROWS):
                row = axes[2 + r]
                nat = natives.get(arm)
                if nat is None:
                    for ax in row:
                        ax.set_facecolor("0.92")
                    label(2 + r, f"{arm}\n(no native)")
                    continue
                lo, hi = float(nat.min()), float(nat.max())
                rng = max(hi - lo, 1e-12)
                for j in range(n_cols):
                    k = min(lat_idx[j], nat.shape[0] - 1)
                    im = row[j].imshow((nat[k] - lo) / rng, cmap=args.cmap, vmin=0, vmax=1,
                                       interpolation="nearest", aspect="auto")
                label(2 + r, f"{arm} native\n{nat.shape[1]}×{nat.shape[2]}")

            # ---------------- LATENT SPACE ----------------
            for j in range(n_cols):
                f_lat = lat_idx[j]
                if zrgb_src is not None:
                    axes[6][j].imshow(zrgb_src[min(f_lat, zrgb_src.shape[0] - 1)],
                                      interpolation="nearest", aspect="auto")
                    axes[7][j].imshow(zrgb_trg[min(f_lat, zrgb_trg.shape[0] - 1)],
                                      interpolation="nearest", aspect="auto")
                else:
                    axes[6][j].set_facecolor("0.92")
                    axes[7][j].set_facecolor("0.92")
            label(6, "z_src 30×52\n(clean encode)")
            label(7, "z_trg 30×52\n(step 7)")

            for r, arm in enumerate(ARM_ROWS):
                row = axes[8 + r]
                if arm in missing:
                    for ax in row:
                        ax.set_facecolor("0.92")
                    label(8 + r, f"{arm}\n(missing)")
                    continue
                fld = fields[arm]
                for j in range(n_cols):
                    im = row[j].imshow(fld[min(lat_idx[j], fld.shape[0] - 1)],
                                       cmap=args.cmap, vmin=0.0, vmax=1.0,
                                       interpolation="nearest", aspect="auto")
                lo, hi = spans[arm]
                label(8 + r, f"{arm} 30×52\nd∈[{lo:.3g},{hi:.3g}]")

            fig.subplots_adjust(left=0.11, right=0.94, top=0.945, bottom=0.008,
                                wspace=0.03, hspace=0.07)
            # divider between the two halves, drawn after the layout is fixed
            y = 0.5 * (axes[5][0].get_position().y0 + axes[6][0].get_position().y1)
            fig.add_artist(plt.Line2D([0.075, 0.95], [y, y], color="0.35", lw=0.8))
            fig.text(0.068, 0.5 * (axes[0][0].get_position().y1
                                   + axes[5][0].get_position().y0),
                     "IMAGE SPACE", fontsize=5.5, rotation=90, ha="center", va="center",
                     color="0.35")
            fig.text(0.068, 0.5 * (axes[6][0].get_position().y1
                                   + axes[11][0].get_position().y0),
                     "LATENT SPACE", fontsize=5.5, rotation=90, ha="center", va="center",
                     color="0.35")

            fig.suptitle(f"R31 stage 2 — {name} (edit{T}) — image space (native maps) "
                         f"vs latent space (30×52, what the blend gates)", fontsize=7,
                         y=0.997)
            if im is not None:
                cax = fig.add_axes([0.952, 0.05, 0.007, 0.45])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label("normalised divergence", fontsize=4.6)
                cb.ax.tick_params(labelsize=4)

            out = out_dir / f"{name}_edit{T}.png"
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            note = f" (missing: {','.join(missing)})" if missing else ""
            print(f"[ok] {out.name}: {n_rows}x{n_cols}{note}", flush=True)
            n_ok += 1
        except Exception as ex:
            print(f"[ERROR] {name}: {type(ex).__name__}: {ex}", flush=True)

    print(f"[r31_grid] {n_ok}/{len(cases)} pages -> {out_dir}", flush=True)
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
