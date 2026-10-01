#!/usr/bin/env python
"""R27 step 1 -- per-case depth-delta magnitude. No video generation.

Measures, once on clean pixels before any denoising, how much SHAPE CHANGE an edit
actually demands, as the mean absolute difference between the monocular depth map of the
source first frame and of the Qwen anchor, averaged over the pixels where that difference
exceeds the estimator's own noise floor:

    D = mean of |dD(p)| / IQR(D_src) over the largest (1 - top_q) fraction of pixels

WHY NOT IoU (what R25 measured). IoU responds to the COVERED SURFACE, not to demanded
geometry change: adding a small hat barely moves it, so it routes to nearly the same tau
as a pure colour edit and then fails for want of detachment. Averaging over a small top
slice -- rather than over the whole frame or the whole object -- is what makes D
independent of how much area the edit covers, which is the entire point of R27. Measured
on the 22 cases, corr(D, edited area) = +0.105 at top_q 0.995.

WHY A FIXED-SIZE TOP SLICE AND NOT A THRESHOLDED SET. The plan specified
D = mean_{p in L} |dD(p)| with L = {p : |dD(p)| > thr}. That makes D a mean CONDITIONED on
L while the threshold IS the conditioning event, so raising the threshold raises D
mechanically -- most for the cases with least real signal, which are almost entirely small
residuals. Swept over --min_effect in {0, 0.02, 0.05, 0.10}, the colour edit
0017_kid-football climbed 0.052 -> 0.423 without its depth changing at all, and the
colour/swap separation INVERTED from 1.78x to 0.61x: every setting that fixed localisation
destroyed the signal. A fixed-size slice conditions on the same number of pixels for every
case, so whatever selection bias remains is identical across cases and cross-case
comparison stays valid. L survives as a localisation diagnostic only, which is what frees
--min_effect to be raised without touching the routed number. Pass
--magnitude conditional for the original definition.

PIPELINE, per case:

  1. Depth on both images with Depth-Anything-V2-Large. Its output is a RELATIVE inverse
     depth in per-image arbitrary units (larger = nearer); neither the scale nor the shift
     is comparable between two independently-estimated maps. Steps 3 and 6 exist only
     because of that.
  2. Both images are brought onto the ANCHOR grid (480 x 832) before depth is estimated,
     and the frame-0 masks are resampled onto it with NEAREST (which keeps them binary;
     any interpolating filter would invent partial-occupancy pixels and change object
     extent). The IMAGE is resized rather than the depth map, so each map is estimated
     from the geometry it will be compared in.

     WHY THE ANCHOR GRID AND NOT THE SOURCE GRID. The plan's todo said to resample onto
     the source grid, following r25_iou.py. That is wrong here, and measurably so. The
     render pipeline builds the anchor through `transforms.Resize((480, 832))`
     (run_fivebench.py:225) -- an ANISOTROPIC squeeze of the 864x480 source, not a crop.
     The anchor's geometry therefore already IS the squeezed source geometry, so
     stretching it back out to 864 re-introduces a 3.7% horizontal warp that was never in
     the data, displacing every vertical edge by up to 32 px at the frame's right side.
     R25 was insensitive to this because it resampled BINARY masks and took an IoU; a
     pixelwise depth difference is not. Measured on a two-case smoke run under the
     source-grid convention, only 0.10 (colour edit) to 0.35 (addition) of L fell inside
     the edit region -- i.e. most of the "signal" was resize artefact along background
     edges. Pass --grid source to reproduce the literal todo wording.
  3. The anchor depth is affine-aligned to the source over the BACKGROUND ONLY:
     (a, b) = argmin ||a * D_edit + b - D_src||^2 over B = NOT(M_src | M_edit), with the
     frame-0 masks read from R25's dump. Fitting on the background is what cancels the
     per-image scale/shift ambiguity WITHOUT letting the edited region drag the fit --
     aligning on the full frame would partly absorb the very difference being measured.
  4. Residual r = D_src - (a * D_edit + b); the reported difference is |r|.
  5. Threshold on |r|, the larger of a STATISTICAL and a PHYSICAL term:

         thr = max( k * sigma ,  min_effect * IQR(D_src) )

     sigma = 1.4826 * MAD(r_B) is a robust scale of the background residual -- MAD and
     not std because depth-estimator error is spatially correlated and clusters at edges
     and textureless regions, which inflates a standard deviation. The second term is a
     minimum effect size and exists because sigma is estimated on the BACKGROUND ONLY, so
     a background with no depth structure collapses it: 0091_A_hawk (bird against open
     sky) gave MAD = 0 exactly, and 0068_planes-water (sky over flat water) gave
     sigma = 0.91, which admitted 91% of the frame into L. A depth change smaller than
     min_effect of the scene's own depth span is not a shape change, however confidently
     it clears the noise. L is the pixels whose |r| exceeds thr.
  6. D_norm = mean of |r| / IQR(D_src) over the top (1 - top_q) fraction of the frame.
     Dividing by the IQR matters because after step 3 the residual is expressed in the
     SOURCE map's arbitrary units, so without it a shallow scene reads a small D no matter
     what the edit did. `frac_in_top` -- how much of that slice fell inside the edit
     region -- is the validity gate: a low value means the anchor differs from the source
     somewhere OTHER than where it was asked to, so the number is not measuring the edit.
     D_norm is what r27_tau_map.py routes on; D_raw is kept for auditing.

DEVIATION FROM THE PLAN'S TODO TEXT, deliberate and flagged. The `write-depth` todo spells
the noise floor as the MAD of R = |r|; the Decisions table says the MAD of "the aligned
background residual". This implements the latter -- MAD of the SIGNED residual r -- because
r is zero-mean over B by construction, which is precisely the null a noise floor should be
estimated against. Taking the MAD of an already-folded |r| understates the scale by roughly
30% for Gaussian noise (1.4826 * MAD(|r|) ~ 0.70 sigma), which would silently admit more
background pixels into L. Pass --fold_mad to reproduce the literal todo wording.

WHAT THIS SCRIPT DOES NOT DECIDE. It writes no tau. The check-depth gate reads the CSV and
the panels, and the boundary condition it tests -- colour and material edits reading
D_norm at their own noise floor -- can kill R27 before anything is rendered.

Usage
-----
    HF_HUB_OFFLINE=1 python evaluation/r27_depth_delta.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --anchor_root /projects/dataggen/outputs/five_bench/anchors \\
        --mask_dir evaluation/figures/r25_iou_masks \\
        --mad_k 3.0 \\
        --viz_dir evaluation/figures/r27_depth_panels \\
        -o evaluation/csv/r27_depth.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

DEPTH_MODEL_ID: str = "depth-anything/Depth-Anything-V2-Large-hf"

# A background smaller than this cannot support a two-parameter affine fit that is
# meaningfully independent of the edited region.
MIN_BG_FRACTION: float = 0.05

# 1 / Phi^{-1}(3/4): rescales a median absolute deviation to a Gaussian-consistent sigma.
MAD_TO_SIGMA: float = 1.4826


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG that could touch the forward pass."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_depth_model(model_id: str, device: str):
    """Load Depth-Anything-V2 once. Read from the local HF cache."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return processor, model


@torch.no_grad()
def estimate_depth(
    img: Image.Image,
    processor,
    model,
    device: str,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Relative inverse depth for one image, resampled to `target_hw` with BILINEAR.

    Returns float64 (H, W). Units are per-image and arbitrary -- only differences after
    the background affine alignment are meaningful.
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    depth = outputs.predicted_depth  # (1, h', w'), model-internal resolution
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    depth = torch.nn.functional.interpolate(
        depth.float(), size=target_hw, mode="bilinear", align_corners=False
    )
    return depth[0, 0].detach().cpu().numpy().astype(np.float64)


def load_frame0_masks(mask_path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Unpack R25's bit-packed frame-0 masks. Returns (M_src, M_edit, (H, W))."""
    data = np.load(mask_path, allow_pickle=True)
    h, w = (int(x) for x in data["shape"])
    n = h * w
    m_src = np.unpackbits(data["m_src"])[:n].reshape(h, w).astype(bool)
    m_edit = np.unpackbits(data["m_edit"])[:n].reshape(h, w).astype(bool)
    return m_src, m_edit, (h, w)


def resize_nearest(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resample a bool mask onto `hw` with NEAREST, as r25_iou.py does.

    NEAREST keeps the mask binary -- any interpolating filter would invent
    partial-occupancy pixels and silently change object extent.
    """
    h, w = hw
    if mask.shape == (h, w):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    return np.array(resized) > 127


def fit_affine(
    d_edit: np.ndarray, d_src: np.ndarray, bg: np.ndarray
) -> Tuple[float, float]:
    """Least-squares (a, b) minimising ||a * d_edit + b - d_src||^2 over `bg`.

    Fitted on the background only: including the edited region would let the very
    difference being measured pull the alignment toward zero.
    """
    x = d_edit[bg]
    y = d_src[bg]
    design = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(a), float(b)


def robust_sigma(values: np.ndarray, fold: bool) -> float:
    """Gaussian-consistent scale from a median absolute deviation.

    `fold=False` (default) takes the MAD of the SIGNED residual, which is zero-mean over
    the background by construction. `fold=True` reproduces the plan todo's literal
    wording by folding first; it understates the scale by ~30% on Gaussian noise.
    """
    v = np.abs(values) if fold else values
    return float(MAD_TO_SIGMA * np.median(np.abs(v - np.median(v))))


def save_panel(
    out_path: Path,
    src_img: Image.Image,
    anchor_img: Image.Image,
    d_src: np.ndarray,
    d_edit_aligned: np.ndarray,
    resid_abs: np.ndarray,
    in_l: np.ndarray,
    cand: np.ndarray,
    row: Dict[str, object],
) -> None:
    """2x3 audit panel: the images, the two aligned depth maps, |dD| and L.

    This is the only artefact that shows WHERE the signal came from. A scalar D cannot
    distinguish a genuine silhouette change from an alignment that failed and lit up the
    whole background, which is what check-depth gate 3 reads it for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = d_src.shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.0 * h / max(w, 1) * 3 + 1.4))

    axes[0][0].imshow(np.asarray(src_img.resize((w, h), Image.BILINEAR)))
    axes[0][0].set_title("source $I_0$", fontsize=11)
    axes[0][1].imshow(np.asarray(anchor_img.resize((w, h), Image.BILINEAR)))
    axes[0][1].set_title("anchor $I_{0,\\mathrm{edit}}$", fontsize=11)

    # Shared colour range so the two depth maps are visually comparable after alignment.
    lo, hi = np.percentile(d_src, [2, 98])
    axes[0][2].imshow(d_src, cmap="magma", vmin=lo, vmax=hi)
    axes[0][2].set_title("$D_{src}$", fontsize=11)
    axes[1][0].imshow(d_edit_aligned, cmap="magma", vmin=lo, vmax=hi)
    axes[1][0].set_title(f"$a\\,D_{{edit}}+b$   (a={row['a']:.3f}, b={row['b']:.3f})",
                         fontsize=11)

    # LOG scale, not linear. On a linear ramp the edited object eats the whole dynamic
    # range and everything else flattens to black -- on 0068_planes-water the threshold
    # (4.08) sat at 1.8% of vmax and the apron's genuine 8.0 residual at 3.4%, both
    # rendering as pure black while 75% of that apron was in fact inside L. The panel
    # then reads as "L includes pixels where |dD| is zero", which is a lie told by the
    # colour map. A log norm spans threshold to peak, and the contour draws the cut.
    from matplotlib.colors import LogNorm
    thr = float(row["thr"])
    vmax = max(float(np.percentile(resid_abs, 99)), thr * 2.0)
    im = axes[1][1].imshow(np.maximum(resid_abs, thr / 8.0), cmap="viridis",
                           norm=LogNorm(vmin=thr / 8.0, vmax=vmax))
    axes[1][1].contour(resid_abs, levels=[thr], colors="red", linewidths=0.6)
    cb = fig.colorbar(im, ax=axes[1][1], fraction=0.03, pad=0.01)
    cb.ax.axhline(thr, color="red", linewidth=1.2)
    cb.ax.tick_params(labelsize=7)
    axes[1][1].set_title(
        f"$|\\Delta \\mathrm{{Depth}}|$  (log; red = thr {thr:.2f})", fontsize=11)

    # DISCARDED halo in amber, KEPT (eroded) set in green. Two display concessions,
    # both necessary and both stated in the title: the kept set is DILATED for drawing
    # only, because a surviving set can be a few dozen pixels (27 on 0017_kid-football)
    # and would otherwise be invisible at figure scale; and the counts are printed, since
    # "99.6% removed" is the fact the picture exists to convey and no amount of shading
    # conveys it when what remains is too small to see.
    from scipy.ndimage import binary_dilation, generate_binary_structure
    n_l_px, n_c_px = int(in_l.sum()), int(cand.sum())
    show = binary_dilation(cand, generate_binary_structure(2, 2), iterations=2) \
        if 0 < n_c_px < 2000 else cand
    panel = np.zeros(in_l.shape + (3,), dtype=float)
    panel[in_l & ~cand] = [0.85, 0.45, 0.05]     # removed by the erosion
    panel[show] = [0.35, 0.95, 0.45]             # kept -- what D is averaged over
    axes[1][2].imshow(panel)
    # frac_in_top is reported in the title rather than drawn: it is the gate of record
    # (tied to what D reads, untouched by any threshold), whereas frac_L_in_mask moves
    # with mad_k / min_effect and is a secondary diagnostic only.
    pct = 100.0 * (1.0 - n_c_px / n_l_px) if n_l_px else 0.0
    grew = "  (kept set dilated x2 for visibility)" if 0 < n_c_px < 2000 else ""
    axes[1][2].set_title(
        f"amber = removed by erosion r={row['erode_radius']}   green = KEPT, what $D$ uses\n"
        f"{n_l_px} px $\\rightarrow$ {n_c_px} px  ({pct:.1f}% removed)   "
        f"frac_in_top {row['frac_in_top']:.2f}{grew}", fontsize=9.5)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"{row['case_id']}  type {row['edit_type']}   "
        f"D_norm {row['d_norm']:.4f} ({row['magnitude']})   D_raw {row['d_raw']:.4f}   "
        f"sigma {row['sigma']:.4f}   thr {row['thr']:.3f} ({row['thr_by']})",
        fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# per-case measurement
# --------------------------------------------------------------------------------------
def measure_case(
    case: Dict[str, object],
    data_root: Path,
    anchor_root: Path,
    mask_dir: Path,
    processor,
    model,
    device: str,
    mad_k: float,
    min_effect: float,
    magnitude: str,
    top_q: float,
    erode_radius: int,
    fold_mad: bool,
    grid: str,
    viz_dir: Optional[Path] = None,
) -> Tuple[Dict[str, object], Optional[str]]:
    """Measure one case. Returns (csv row, hard-failure reason or None)."""
    case_id = str(case["case_id"])
    video_name = str(case["video_name"])
    edit_type = int(case["edit_type"])

    src_path = data_root / "images" / video_name / "00001.jpg"
    anchor_path = anchor_root / f"edit{edit_type}" / f"{video_name}.png"
    mask_path = mask_dir / f"{case_id}.npz"
    for p in (src_path, anchor_path, mask_path):
        if not p.exists():
            raise FileNotFoundError(f"{case_id}: missing {p}")

    src_img = Image.open(src_path).convert("RGB")
    anchor_img = Image.open(anchor_path).convert("RGB")

    m_src, m_edit, mask_hw = load_frame0_masks(mask_path)
    if (src_img.size[1], src_img.size[0]) != mask_hw:
        # R25 computed and dumped every mask on the source jpg grid; a mismatch means the
        # masks and the frame no longer describe the same image.
        raise ValueError(
            f"{case_id}: mask grid {mask_hw} != source grid "
            f"{(src_img.size[1], src_img.size[0])}")

    # The comparison grid. `anchor` (default) is the geometry the render pipeline
    # actually produced via transforms.Resize((480, 832)); `source` reproduces the plan
    # todo's wording and re-introduces the squeeze as a warp.
    hw = ((anchor_img.size[1], anchor_img.size[0]) if grid == "anchor"
          else (src_img.size[1], src_img.size[0]))
    h, w = hw

    row: Dict[str, object] = {
        "case_id": case_id,
        "video_name": video_name,
        "edit_type": edit_type,
        "grid": f"{h}x{w}",
        "aspect_src": round(src_img.size[0] / src_img.size[1], 4),
        "aspect_anchor": round(anchor_img.size[0] / anchor_img.size[1], 4),
    }

    # Resize the IMAGES onto the shared grid before estimating depth, so each map is
    # produced from the geometry it will be compared in rather than resampled after.
    src_img = src_img.resize((w, h), Image.BILINEAR)
    anchor_img = anchor_img.resize((w, h), Image.BILINEAR)
    m_src = resize_nearest(m_src, hw)
    m_edit = resize_nearest(m_edit, hw)

    d_src = estimate_depth(src_img, processor, model, device, hw)
    d_edit = estimate_depth(anchor_img, processor, model, device, hw)

    edit_region = m_src | m_edit
    bg = ~edit_region
    bg_frac = float(bg.mean())
    row["bg_frac"] = round(bg_frac, 4)
    if bg_frac < MIN_BG_FRACTION:
        row.update({k: "" for k in
                    ("a", "b", "sigma", "thr", "thr_by", "d_raw", "d_norm",
                     "l_frac", "frac_l_in_mask", "iqr_src", "frac_in_top",
                     "top_frac", "magnitude", "erode_radius")})
        return row, (f"{case_id}: background is {bg_frac:.3f} of the frame "
                     f"(< {MIN_BG_FRACTION}); the affine alignment cannot be fitted "
                     f"independently of the edited region")

    if float(np.std(d_edit[bg])) <= 0.0:
        row.update({k: "" for k in
                    ("a", "b", "sigma", "thr", "thr_by", "d_raw", "d_norm",
                     "l_frac", "frac_l_in_mask", "iqr_src", "frac_in_top",
                     "top_frac", "magnitude", "erode_radius")})
        return row, (f"{case_id}: anchor depth is constant over the background; "
                     f"the affine fit is degenerate")

    a, b = fit_affine(d_edit, d_src, bg)
    resid = d_src - (a * d_edit + b)
    resid_abs = np.abs(resid)

    sigma = robust_sigma(resid[bg], fold=fold_mad)
    row["a"] = round(a, 6)
    row["b"] = round(b, 6)
    row["sigma"] = round(sigma, 6)

    q1, q3 = np.percentile(d_src, [25, 75])
    iqr_src = float(q3 - q1)
    if iqr_src <= 0.0:
        row.update({k: "" for k in ("thr", "thr_by", "d_raw", "d_norm", "l_frac",
                                    "frac_l_in_mask", "iqr_src", "frac_in_top",
                                    "top_frac", "magnitude",
                                    "erode_radius")})
        return row, (f"{case_id}: source depth map has zero IQR; the scene spans no "
                     f"depth at all, so neither D_norm nor the effect-size floor is "
                     f"defined")
    row["iqr_src"] = round(iqr_src, 6)

    # The threshold is the larger of a STATISTICAL term and a PHYSICAL one.
    #
    #   k * sigma            -- "bigger than this frame pair's noise"
    #   min_effect * IQR     -- "bigger than a trivial fraction of the scene's own depth
    #                            range", i.e. a minimum effect size
    #
    # The statistical term alone inherits sigma's one weakness: sigma is measured only on
    # the background, so a background with no depth structure collapses it. 0091_A_hawk
    # (bird against open sky) gives MAD = 0 exactly, and 0068_planes-water (sky over flat
    # water) gives sigma = 0.91, which admitted 91% of the frame into L. The floor does
    # not care how quiet the background is: a depth change smaller than min_effect of the
    # scene's depth span is not a shape change, however confidently it exceeds the noise.
    thr_stat = mad_k * sigma
    thr_effect = min_effect * iqr_src
    thr = max(thr_stat, thr_effect)
    if thr <= 0.0:
        row.update({k: "" for k in ("thr", "thr_by", "d_raw", "d_norm", "l_frac",
                                    "frac_l_in_mask", "frac_in_top",                                     "top_frac", "magnitude", "erode_radius")})
        return row, (f"{case_id}: threshold is zero (sigma {sigma:.4g}, "
                     f"min_effect*IQR {thr_effect:.4g}); L is undefined")
    row["thr"] = round(thr, 6)
    # Which term bound. A case bound by `effect` is one where the background carried no
    # usable noise estimate -- worth knowing per case, not just in aggregate.
    row["thr_by"] = "stat" if thr_stat >= thr_effect else "effect"

    in_l = resid_abs > thr
    n_l = int(in_l.sum())
    row["l_frac"] = round(n_l / float(h * w), 6)
    # Where L landed. Mostly-background L means the alignment failed and the difference is
    # estimator drift rather than the edit -- check-depth gate 3.
    row["frac_l_in_mask"] = round(
        float(edit_region[in_l].mean()) if n_l else 0.0, 6)

    # ---------------- magnitude ----------------
    # `topq` (default) DECOUPLES the magnitude from the threshold. `conditional` is the
    # original mean-over-L.
    #
    # WHY DECOUPLE. Under `conditional`, D is a mean CONDITIONED on L while the threshold
    # IS the conditioning event, so raising the threshold raises D mechanically -- and it
    # raises it most for the cases with the least real signal, which are almost entirely
    # small residuals. Measured over --min_effect in {0, 0.02, 0.05, 0.10}, the colour
    # edit 0017_kid-football climbed 0.052 -> 0.423 without its depth changing at all, and
    # the colour/swap separation inverted from 1.78x to 0.61x. That made the threshold
    # untunable: every setting that fixed localisation destroyed the signal.
    #
    # `topq` instead averages |r|/IQR over a FIXED-SIZE top slice of the frame. The
    # conditioning set is the same size for every case, so whatever selection bias remains
    # is identical across cases and cross-case comparison stays valid. L is then free to
    # serve localisation diagnostics alone, and --min_effect can be raised without
    # touching the routed number.
    #
    # Measured at top_q 0.995 on the 22 cases: correlation between D and EDITED AREA falls
    # to +0.105 (surface-independence, R27's founding claim), the median fraction of the
    # top slice landing inside the edit region is 1.00 against 0.57 for `conditional`, and
    # among well-localised cases the colour/swap separation is 8.3x against 1.78x.
    resid_norm = resid_abs / iqr_src
    n_top = max(1, int(round((1.0 - top_q) * h * w)))

    # Candidate set: L with thin structure removed by a binary erosion. The anchor is a
    # RE-SYNTHESIS, so object OUTLINES shift by a fraction of a pixel and light up at depth
    # discontinuities, where the depth gradient is enormous. Those halos clear the
    # threshold but are one to three pixels wide, so an erosion removes them while leaving
    # region-level change untouched. Measured across radii 0-3: |L| on 0017_kid-football
    # (a red -> yellow cap, no geometry change) falls 7498 -> 2688 -> 602 -> 27 px, while
    # 0011_lucia_e2 and 0042_gym-ball are UNCHANGED to three decimals.
    if erode_radius > 0 and n_l:
        from scipy.ndimage import (binary_erosion, generate_binary_structure,
                                   iterate_structure)
        cand = binary_erosion(
            in_l, iterate_structure(generate_binary_structure(2, 2), erode_radius))
    else:
        cand = in_l

    # ZERO-PADDED fixed-size slice: the denominator is always n_top, so a case with fewer
    # survivors is scored on a mostly-empty slice and its D falls toward zero -- the
    # correct reading for an edit whose depth change was entirely boundary halo. The
    # obvious alternative, shrinking k to the survivor count, instead averages the few most
    # extreme values and keeps D HIGH (0017_kid-football reads 0.783 that way at radius 2
    # against 0.236 here) while making the conditioning set a different size per case,
    # which is precisely the variable-size bias `topq` exists to remove.
    vals = resid_norm[cand]
    if vals.size:
        k = min(n_top, vals.size)
        part = np.partition(vals, vals.size - k)[vals.size - k:]
        d_topq = float(part.sum() / n_top)
        sel = cand & (resid_norm >= part.min())
        frac_in_top = float(edit_region[sel].mean()) if sel.any() else 0.0
        n_sel = int(sel.sum())
    else:
        d_topq, frac_in_top, n_sel = 0.0, 0.0, 0

    row["top_frac"] = round(n_top / float(h * w), 6)
    row["n_selected"] = n_sel
    row["frac_in_top"] = round(frac_in_top, 6)
    row["erode_radius"] = erode_radius
    row["magnitude"] = magnitude

    if magnitude == "topq":
        d_norm = d_topq
    else:
        # An empty L means the maps agree everywhere to within noise, i.e. the edit
        # demanded no measurable geometry change. D = 0 is correct, not a failure.
        d_norm = (float(resid_abs[in_l].mean()) / iqr_src) if n_l else 0.0
    row["d_norm"] = round(d_norm, 6)
    row["d_raw"] = round(d_norm * iqr_src, 6)

    if viz_dir is not None:
        save_panel(viz_dir / f"{case_id}.png", src_img, anchor_img,
                   d_src, a * d_edit + b, resid_abs, in_l, cand, row)

    return row, None


def build_fieldnames() -> List[str]:
    return [
        "case_id", "video_name", "edit_type",
        "d_raw", "d_norm", "sigma", "a", "b",
        "magnitude", "frac_in_top", "n_selected", "top_frac", "erode_radius",
        "thr", "thr_by", "l_frac", "frac_l_in_mask", "iqr_src", "bg_frac",
        "grid", "aspect_src", "aspect_anchor",
    ]


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path, required=True,
                   help="FiVE-Bench root holding images/")
    p.add_argument("--anchor_root", type=Path, required=True,
                   help="Anchor root; the anchor is {anchor_root}/edit{T}/{video}.png")
    p.add_argument("--mask_dir", type=Path,
                   default=Path("evaluation/figures/r25_iou_masks"),
                   help="R25's dumped frame-0 masks, {mask_dir}/{case_id}.npz. Their "
                        "complement is the background the affine alignment is fitted on.")
    p.add_argument("--mad_k", type=float, default=6.0,
                   help="L is the pixels whose |dD| exceeds mad_k * sigma, with sigma a "
                        "robust scale of the aligned BACKGROUND residual. Noise-derived "
                        "rather than eyeballed: the threshold is what separates a real "
                        "geometry change from estimator error, and estimator error is "
                        "what the background measures.")
    p.add_argument("--magnitude", choices=("topq", "conditional"), default="topq",
                   help="How D is computed. 'topq' (default) averages |dD|/IQR over a "
                        "FIXED-SIZE top slice of the frame, so the magnitude does not "
                        "depend on the threshold at all. 'conditional' is the plan's "
                        "original mean-over-L, where the threshold IS the conditioning "
                        "event and therefore inflates D mechanically as it rises -- "
                        "measured, the colour edit 0017_kid-football climbed 0.052 to "
                        "0.423 across the --min_effect sweep without its depth changing, "
                        "and the colour/swap separation inverted. Under 'topq' that "
                        "coupling is gone and L serves localisation diagnostics only.")
    p.add_argument("--top_q", type=float, default=0.995,
                   help="Size of the top slice for --magnitude topq, as a quantile: "
                        "0.995 means the largest 0.5%% of pixels. Chosen by sweep over "
                        "{0.98, 0.99, 0.995, 0.999}: correlation between D and edited "
                        "AREA falls monotonically with q (+0.261 -> +0.018), which is "
                        "R27's surface-independence claim, while localisation degrades "
                        "as the slice shrinks toward noise (cases with frac_in_top < 0.5 "
                        "go 1 -> 5). 0.995 is the knee: corr +0.105, median frac_in_top "
                        "1.00, and 8.3x colour/swap separation among well-localised "
                        "cases.")
    p.add_argument("--erode_radius", type=int, default=3,
                   help="Radius of the binary erosion applied to L before the top slice is "
                        "taken. This DOES enter the routed magnitude. The anchor is a "
                        "re-synthesis, so object outlines shift sub-pixel and light up at "
                        "depth discontinuities; those halos clear the threshold but are "
                        "1-3 px wide, so an erosion removes them and leaves region-level "
                        "change intact. Combined with the zero-padded slice, a case whose "
                        "depth change was ENTIRELY halo scores near zero, which is the "
                        "correct reading. Measured over radii 0-6: colour/swap separation "
                        "0.95x (r=0) -> 2.96x (r=2) -> 56.8x (r=3) -> 255x (r=4), while "
                        "0011_lucia_e2 and 0042_gym-ball are unchanged to three decimals "
                        "and the non-colour mean moves under 1%% between r=3 and r=4. "
                        "r=3 is the knee AND the least aggressive point on the plateau; "
                        "beyond r=4 a genuine case starts degrading (0028_kite-walk_e4, "
                        "material, 1.072 -> 0.432 by r=6). Set 0 to disable.")
    p.add_argument("--min_effect", type=float, default=0.10,
                   help="Minimum effect size, as a fraction of the SOURCE depth map's "
                        "interquartile range. The threshold on |dD| is "
                        "max(mad_k * sigma, min_effect * IQR(D_src)): a depth change "
                        "smaller than this fraction of the scene's own depth span is not "
                        "a shape change, however confidently it clears the noise. Exists "
                        "because sigma is estimated on the BACKGROUND only, so a "
                        "background with no depth structure (open sky, flat water) "
                        "collapses it -- 0091_A_hawk gave MAD = 0 exactly and "
                        "0068_planes-water admitted 91%% of the frame into L. Set 0 to "
                        "disable the floor. "
                        "RAISED 0.02 -> 0.10 (with mad_k 3 -> 6) while D was briefly "
                        "independent of the threshold. ⚠ IT NO LONGER IS: since erosion "
                        "was adopted into the magnitude, the routed slice is drawn from "
                        "erode(L), so D depends on thr and therefore on this flag. "
                        "Re-measured across mad_k {3,6} x min_effect {0.02,0.10}: D is "
                        "EXACTLY invariant for most cases (median relative change 0.00%%, "
                        "because a real region-level change leaves far more than n px "
                        "after erosion) but HIGHLY sensitive for low-signal ones -- "
                        "0017_kid-football moves 0.012 -> 0.372 at min_effect 0.02. "
                        "mad_k is nearly irrelevant by comparison. "
                        "⚠ CAVEAT THAT NOW MATTERS: 0.10 was chosen by inspecting "
                        "frac_L_in_mask, and it is load-bearing for the boundary-condition "
                        "result (both colour edits reading tau 2.01). That constant was "
                        "therefore not selected independently of the outcome it supports; "
                        "the verdict must report the colour cases at min_effect in "
                        "{0.05, 0.10} so the dependence is visible.")
    p.add_argument("--fold_mad", action="store_true",
                   help="Estimate sigma from the MAD of |residual| instead of the signed "
                        "residual, reproducing the plan todo's literal wording. Off by "
                        "default: folding first understates the scale by ~30%% on "
                        "Gaussian noise and would admit more background pixels into L.")
    p.add_argument("--grid", choices=("anchor", "source"), default="anchor",
                   help="Grid both sides are compared on. 'anchor' (480x832) is the "
                        "geometry the render pipeline actually produced, via "
                        "transforms.Resize((480, 832)) at run_fivebench.py:225 -- an "
                        "anisotropic squeeze of the 864x480 source, so the anchor ALREADY "
                        "carries the squeezed geometry. 'source' reproduces the plan "
                        "todo's wording and stretches the anchor back to 864, which "
                        "re-introduces a 3.7%% horizontal warp that was never in the "
                        "data and displaces every vertical edge by up to 32 px.")
    p.add_argument("--depth_model_id", type=str, default=DEPTH_MODEL_ID)
    p.add_argument("--viz_dir", type=Path, default=None,
                   help="If given, write one 2x3 audit panel per case "
                        "({viz_dir}/{case_id}.png). Off by default; the CSV is "
                        "unaffected either way.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r27_depth.csv"))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    data_root = args.data_root.expanduser().resolve()
    anchor_root = args.anchor_root.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    viz_dir = args.viz_dir.expanduser() if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.expanduser().read_text())

    print("=== R27 step 1: depth-delta magnitude ===", flush=True)
    print(f"cases       {args.cases} ({len(cases)} cases)", flush=True)
    print(f"data_root   {data_root}", flush=True)
    print(f"anchor_root {anchor_root}", flush=True)
    print(f"mask_dir    {mask_dir}", flush=True)
    print(f"depth       {args.depth_model_id}", flush=True)
    print(f"magnitude   {args.magnitude}"
          f"{f'  top {100*(1-args.top_q):.2f}% of frame' if args.magnitude == 'topq' else '  mean over L'}",
          flush=True)
    print(f"min_effect  {args.min_effect}"
          f"{'  (floor disabled)' if args.min_effect <= 0 else '  of IQR(D_src)'}",
          flush=True)
    print(f"mad_k       {args.mad_k}   sigma from "
          f"{'MAD(|residual|)  [--fold_mad]' if args.fold_mad else 'MAD(residual)'}",
          flush=True)
    print(f"grid        {args.grid}"
          f"{'  (both sides on the anchor geometry, no warp)' if args.grid == 'anchor' else '  [--grid source: ANCHOR IS STRETCHED, see docstring]'}",
          flush=True)
    print(f"device      {args.device}   seed {args.seed}", flush=True)
    print(f"viz_dir     {viz_dir if viz_dir else '(off)'}", flush=True)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    processor, model = load_depth_model(args.depth_model_id, args.device)

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for case in cases:
        row, failure = measure_case(
            case, data_root, anchor_root, mask_dir,
            processor=processor, model=model, device=args.device,
            mad_k=args.mad_k, min_effect=args.min_effect,
            magnitude=args.magnitude, top_q=args.top_q,
            erode_radius=args.erode_radius,
            fold_mad=args.fold_mad, grid=args.grid,
            viz_dir=viz_dir,
        )
        rows.append(row)
        if failure is not None:
            failures.append(failure)
            print(f"[FAIL] {failure}", flush=True)
        else:
            print(f"[ok] {row['case_id']:22s} type {row['edit_type']} "
                  f"D_norm {row['d_norm']:.4f}  D_raw {row['d_raw']:.4f}  "
                  f"sigma {row['sigma']:.4f}  thr {row['thr']:.3f}"
                  f"({row['thr_by']})  inTop {row['frac_in_top']:.2f}"
                  f"  nsel {row['n_selected']:5d}  "
                  f"|L| {row['l_frac']:.4f}  "
                  f"inMask {row['frac_l_in_mask']:.2f}  a {row['a']:.3f}",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames()
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nwrote {args.out}  ({len(rows)} rows)", flush=True)
    if viz_dir is not None:
        print(f"wrote audit panels to {viz_dir}", flush=True)

    measured = [float(r["d_norm"]) for r in rows if r.get("d_norm") not in ("", None)]
    if measured:
        print(f"D_norm range over {len(measured)} measured cases: "
              f"[{min(measured):.4f}, {max(measured):.4f}]", flush=True)

    # Source and anchor differ in aspect by construction (864x480 vs 832x480, the
    # pipeline's anisotropic Resize). Under --grid anchor that is HARMLESS -- the source
    # is squeezed exactly as the pipeline squeezed it, so both sides share one geometry.
    # Under --grid source the same mismatch becomes a genuine warp, which is what
    # check-depth gate 2 exists to catch.
    mismatched = [r["case_id"] for r in rows
                  if abs(float(r["aspect_src"]) - float(r["aspect_anchor"]))
                  / max(float(r["aspect_src"]), 1e-9) > 0.02]
    if mismatched and args.grid == "source":
        print(f"\nWARNING: --grid source with aspect mismatch on {len(mismatched)} "
              f"case(s): {', '.join(mismatched)}", flush=True)
        print("  The anchor is STRETCHED onto the source grid, so |dD| picks up spurious "
              "structure along every vertical edge rather than at the edit. Re-run with "
              "the default --grid anchor; see check-depth gate 2.", flush=True)
    elif mismatched:
        print(f"\nnote: source and anchor differ in aspect on {len(mismatched)} case(s) "
              f"(864x480 vs 832x480, the pipeline's own Resize). Both sides were "
              f"compared on the anchor grid, so no warp was introduced.", flush=True)

    print(f"failures: {len(failures)}", flush=True)
    if failures:
        print("\nHard failures -- do NOT proceed to taumap:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
