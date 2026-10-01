#!/usr/bin/env python
"""R30 stage 1 -- eight STATIC anchor-vs-source divergences per case. No rollout.

Measures, once on clean pixels before any denoising, how much an edit demands, from the
source first frame and the Qwen anchor ALONE. Eight definitions of "how much", each of
which stage 2 turns into a blend rate `b` and renders in full:

    lpips        perceptual distance                          (spatial -> mask-reduced)
    dino_cls     1 - cos(CLS_src, CLS_anchor)                  (scalar by construction)
    dino_patch   mean over positions of 1 - cos(k_src, k_anc)  (spatial -> mask-reduced)
    clip_image   1 - cos(E_I(src), E_I(anchor))                (scalar, global)
    clip_prompt  1 - cos(E_T(P_src), E_T(P_trg))               (scalar, TEXT ONLY)
    depth        mean |dDepth| after background affine align   (spatial -> mask-reduced)
    normals      mean angular error in degrees                 (spatial -> mask-reduced)
    selfsim      ||S_src - S_anchor||_F / N, S_ij = cos(k_i,k_j)  (scalar, full image)

`miou` was dropped at planning: it needs a separate anchor-side mask, which the
union-only dump cannot provide, and standing up a segmenter for one arm was out of scope.

THE MEASUREMENT REGION IS THE UNION M^src | M^trg, not the target mask. The union covers
WHERE THE SOURCE OBJECT WAS as well as where the target content is. A target-only region
gives a removal almost nothing to measure over and would read ~0 for the largest possible
edit. R26's pass 1 already dumped exactly this union under exactly this specification
(r26_dump.sh: --blend_sched zero --union_dump_dir --vp_mode vp, same anchors, same
cases.json, --step 15 --fg_boost_factor 4 --seed 0), so this script runs NO diffusion.

WHY THE ANCHOR GRID (480x832) AND NOT THE SOURCE GRID. The render pipeline builds the
anchor through transforms.Resize((480, 832)) (run_fivebench.py:225) -- an ANISOTROPIC
squeeze of the 864x480 source, not a crop -- so the anchor's geometry ALREADY IS the
squeezed source geometry. Stretching it back to 864 re-introduces a 3.7% horizontal warp
that was never in the data and displaces every vertical edge by up to 32 px, which a
pixelwise depth or normal difference reads as signal. This is r27_depth_delta.py's
measured finding (see its docstring); it applies unchanged here.

MASKED, NOT GLOBAL, is what stage 2 routes on. Area is not the axis that sets edit demand
-- recolouring a thumbnail region and adding a thumbnail object have equal area and
opposite demand -- and, decisively, spatial selectivity is ALREADY carried by the stage-2
mask, so the scalar `b` need only encode how much per PIXEL. Loading area into `b`
double-counts a factor the mechanism already handles. The global reduction is emitted as a
second column for context (same map, zero extra cost) but is not a rollout arm.

DEPTH IS DIVIDED BY IQR(D_src), AND THAT IS NOT COSMETIC. Monocular relative depth is
scale/shift-free per image. The background affine alignment cancels the ANCHOR's scale and
shift, but the residual is then expressed in the SOURCE map's own arbitrary units, which
differ per clip. Without dividing by the source IQR a shallow scene reads a small
divergence no matter what the edit did, and the eight arms are compared across clips.
Pass --depth_norm raw for the un-normalised residual (kept as a column either way).

NORMALS COME FROM A DIRECT ESTIMATOR, NOT FROM DIFFERENTIATING THE DEPTH. Finite-
differencing a relative depth map cancels the shift but NOT the scale, which tilts every
normal by a per-clip factor and reintroduces exactly the ambiguity this arm exists to
avoid. Marigold is diffusion-based and therefore stochastic, so the seed, step count and
ensemble size are fixed and recorded in the CSV; --normals_ensemble >= 5 means the map
returned is an ensemble average rather than one sample.

WHAT THIS SCRIPT CANNOT MEASURE (check-div gate 3, DROPPED 2026-09-10). That gate
read `area(trg_fg) / area(union)` per clip to tell whether the divergence is measured
over a region the stage-2 intervention can actually reach. R26's dump is the UNION ONLY
(the npz holds a single bit-packed `M`), so the target mask is not separable from it and
that ratio is NOT computable here. `union_frac` -- the union's share of the frame -- is
emitted instead; it bounds nothing about reachability. Recovering the true ratio would
need a re-dump with separated masks, i.e. 22 rollouts and a pipeline change, both of which
the plan's Decisions table rules out. Flagged rather than faked.

Usage
-----
    HF_HUB_OFFLINE=1 python evaluation/r30_divergence.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --anchor_root /projects/dataggen/outputs/five_bench/anchors \\
        --mask_dir /projects/dataggen/outputs/five_bench/r26_masks \\
        --viz_dir evaluation/figures/r30_divergence_panels \\
        -o evaluation/csv/r30_divergence.csv
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
import torch.nn.functional as F
from PIL import Image

# ---- the eight arms, in the order they are reported ----------------------------------
ARMS: Tuple[str, ...] = ("lpips", "dino_cls", "dino_patch", "clip_image",
                         "clip_prompt", "depth", "normals", "selfsim")

# Arms whose primitive is a spatial map and therefore has a meaningful masked/global
# split. The other four are scalar by construction, and their _masked column repeats the
# scalar so downstream code (r30_b_map.py, the check-div gate) can read one column name
# for every arm.
SPATIAL_ARMS: Tuple[str, ...] = ("lpips", "dino_patch", "depth", "normals")

DEPTH_MODEL_ID: str = "depth-anything/Depth-Anything-V2-Large-hf"
DINOV2_MODEL_ID: str = "facebook/dinov2-with-registers-base"
CLIP_MODEL_ID: str = "openai/clip-vit-large-patch14"
NORMALS_MODEL_ID: str = "prs-eth/marigold-normals-v1-1"

# DINOv2 patch size. Input dims must be divisible by it, so 480x832 is trimmed to
# 476x826 -> a 34x59 token grid.
DINOV2_PATCH: int = 14

# Wan's spatial VAE stride: one latent token per 16x16 pixel block. Used to turn the
# latent-resolution union mask back into a pixel mask, and checked against the npz's own
# `frame_seq_length` so a grid change cannot pass silently.
VAE_STRIDE: int = 16

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# A background smaller than this cannot support a two-parameter affine fit that is
# meaningfully independent of the edited region (r27_depth_delta.py's constant).
MIN_BG_FRACTION: float = 0.05


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG that could touch a forward pass.

    Seven of the eight arms are deterministic; Marigold is a diffusion model and is not,
    which is the whole reason this is called before every case rather than once.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_union_frame0(mask_path: Path, hw: Tuple[int, int], open_iter: int = 1,
                      split_mask_path: Optional[Path] = None,
                      trg_degenerate_frac: float = 0.90
                      ) -> Tuple[np.ndarray, Tuple[int, int], bool]:
    """Unpack R26's frame-0 union mask, clean it, and upsample it to the pixel grid `hw`.

    The npz holds `M` bit-packed as (n_latent_frames, frame_seq_length) and `shape`.
    Row 0 is latent frame 0, which is pixel frame 0 (r26_dump.sh trims the anchor prefix
    and drops the rollout overlap, so row f is latent frame f).

    CLEANUP: morphological OPENING (erosion then dilation), `open_iter` rounds, 8-connected,
    on the LATENT grid before upsampling -- settled 2026-09-10, reversing the earlier
    "use exactly as dumped" call. THE MASK IS SPECKLE, NOT A REGION: it comes from
    cross-attention response to the trigger-word tokens (edit_causal_inference.py's
    obtain_mask path in wan/modules/model.py), thresholded at exactly `mask_soft > 0` --
    any spatial token whose attention leans even slightly toward the trigger word crosses
    threshold, so background texture (tree bark, pavement, shadows) routinely lights up as
    isolated 16x16-px blobs alongside the real object.

    WHY OPENING AND NOT DILATION. Dilation was rejected here in the same investigation:
    at 30x52 latent resolution one 8-connected step roughly DOUBLES coverage (0.50 -> 0.94
    on 0001_bus), because it grows the ALREADY-NOISY region outward. Opening does the
    opposite -- erode first (kills anything smaller than the structuring element outright,
    since dilating a fully-eroded-away region grows nothing back), THEN dilate BACK with
    the same element (restores the boundary of whatever survived erosion, i.e. any region
    large enough to have an interior). A single isolated speckle token has no interior and
    is erased for good; the real object, several tokens across, comes back close to its
    original extent. Measured on the 22 cases at open_iter=1: 0001_bus 0.50 -> 0.27 (47
    connected components -> 3, i.e. scattered speckle collapses toward one blob);
    0090_A_deer 0.56 -> 0.23 (a dense speckle field removed, one coherent deer-shaped
    region left plus one small secondary blob); 0011_lucia_e5 (already small and compact)
    0.112 -> 0.096, the removed pixels being a handful of isolated single-token dots --
    i.e. opening does NOT meaningfully erode a mask that was never noisy to begin with.
    0042_gym-ball (the depth-disqualifying case, union = 100% of frame) is UNCHANGED by
    opening: erosion peels the outermost token ring, but the subsequent dilation restores
    it exactly, since the whole interior survives erosion intact -- opening is the
    identity on a region far larger than the structuring element. So this does not rescue
    depth on that clip; it was never a speckle problem there.

    `open_iter=1` is the chosen default: at 2 the shrinkage on already-thin masks starts
    looking like real signal loss rather than noise removal (0007_guitar-violin 0.115 ->
    0.038, more than halving again), so 1 round is the smallest step that visibly
    de-speckles the noisy cases without visibly eating the honest ones. Set 0 to reproduce
    the earlier "use exactly as dumped" behaviour.

    DEGENERATE-TARGET FALLBACK (settled 2026-09-10, why 0042_gym-ball's union is 100% of
    the frame -- NOT speckle, opening cannot touch it). The union is M_src | M_trg, where
    each side grounds its own trigger phrase via cross-attention. For a REMOVAL edit
    (0042_gym-ball, edit_type 6), trg_word is a NEGATION -- "without a heavy gym ball" --
    with no visual referent anywhere in the frame, so trg grounding never localizes and
    saturates instead: confirmed by dumping M_src/M_trg SEPARATELY (a small pipeline
    patch, `edit_causal_inference.py`'s new `split_dump`/`_save_split_mask`, run once for
    this one clip via `--split_mask_dump_dir`) -- M_trg reads EXACTLY 1.0 on every one of
    its 18 frames, while M_src (the concrete "with a heavy gym ball") is a normal, compact
    0.16-0.23. If `split_mask_path` is given and exists for this clip, and M_trg's OWN
    frame-0 coverage exceeds `trg_degenerate_frac`, the measurement mask falls back to
    M_src ALONE for this clip; otherwise (no split-mask file, or trg is not degenerate)
    behaviour is unchanged -- M_src | M_trg via the plain union, same as before. Computed
    for 0042_gym-ball only so far; the other 21 clips have no split-mask file and are
    therefore byte-for-byte unaffected by this parameter.

    MEASUREMENT-ONLY, same as opening: R30's actual RENDER (stage 2) still reads R26's
    raw, un-split, un-cleaned union directly from `r26_masks` -- confirmed with the user,
    2026-09-10 -- so an arm rendered on this clip stays exactly as comparable to R26's 8
    constant-b arms on the same clip as every other clip already is; only the SCALAR b
    chosen for this clip changes, via a cleaner measurement of how much it demands.
    """
    data = np.load(mask_path, allow_pickle=True)
    n_frames, seq_len = (int(x) for x in data["shape"])
    bits = np.unpackbits(data["M"], axis=-1)[:, :seq_len].astype(bool)
    if bits.shape != (n_frames, seq_len):
        raise ValueError(f"{mask_path}: unpacked to {bits.shape}, expected "
                         f"{(n_frames, seq_len)}")

    h, w = hw
    lat_h, lat_w = h // VAE_STRIDE, w // VAE_STRIDE
    if lat_h * lat_w != seq_len:
        # The mask and the frame no longer describe the same image. Silently reshaping
        # would scramble the region and every masked arm with it.
        raise ValueError(
            f"{mask_path}: frame_seq_length {seq_len} != {lat_h}x{lat_w} = "
            f"{lat_h * lat_w} implied by the {h}x{w} grid at stride {VAE_STRIDE}")

    m_lat = bits[0].reshape(lat_h, lat_w)

    used_src_only = False
    if split_mask_path is not None and split_mask_path.exists():
        sdata = np.load(split_mask_path, allow_pickle=True)
        sn, sseq = (int(x) for x in sdata["shape"])
        if (sn, sseq) != (n_frames, seq_len):
            raise ValueError(f"{split_mask_path}: shape {(sn, sseq)} != union's "
                             f"{(n_frames, seq_len)} -- stale split dump for this clip.")
        m_src_lat = np.unpackbits(sdata["M_src"], axis=-1)[:, :sseq].astype(bool)[0] \
            .reshape(lat_h, lat_w)
        m_trg_lat = np.unpackbits(sdata["M_trg"], axis=-1)[:, :sseq].astype(bool)[0] \
            .reshape(lat_h, lat_w)
        if m_trg_lat.mean() > trg_degenerate_frac:
            m_lat = m_src_lat
            used_src_only = True

    if open_iter > 0 and m_lat.any():
        from scipy.ndimage import binary_opening, generate_binary_structure
        m_lat = binary_opening(m_lat, generate_binary_structure(2, 2), iterations=open_iter)

    # NEAREST upsampling: the mask is defined per latent token, so every pixel in a token
    # inherits that token's value exactly. An interpolating filter would invent
    # partial-occupancy pixels and change the region's extent.
    up = np.kron(m_lat, np.ones((VAE_STRIDE, VAE_STRIDE), dtype=bool))
    return up[:h, :w], (lat_h, lat_w), used_src_only


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean of `values` over `mask`. Returns nan for an empty mask rather than 0.0.

    0.0 would be indistinguishable from "measured, and the edit changed nothing", which
    is a real and different reading; nan propagates into the CSV and fails the b-map's
    coverage check loudly.
    """
    return float(values[mask].mean()) if mask.any() else float("nan")


def resize_to(img: Image.Image, hw: Tuple[int, int]) -> Image.Image:
    h, w = hw
    return img if img.size == (w, h) else img.resize((w, h), Image.BILINEAR)


def to_imagenet_tensor(img: Image.Image, hw: Tuple[int, int],
                       device: str) -> torch.Tensor:
    """(1, 3, H, W) ImageNet-normalised float32 on `device`."""
    arr = np.asarray(resize_to(img, hw), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - torch.tensor(IMAGENET_MEAN)[:, None, None]) \
        / torch.tensor(IMAGENET_STD)[:, None, None]
    return t.unsqueeze(0).to(device)


def to_unit_tensor(img: Image.Image, hw: Tuple[int, int], device: str) -> torch.Tensor:
    """(1, 3, H, W) in [0, 1] on `device`."""
    arr = np.asarray(resize_to(img, hw), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def mask_to_grid(mask: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
    """Resample a pixel mask onto a token grid by AREA MEAN, then threshold at 0.5.

    Area mean rather than nearest: a token covers a whole patch, and what matters is
    whether the majority of that patch is inside the region, not what its top-left corner
    happens to be. Matches r29_novelty.py's convention.
    """
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    pooled = F.interpolate(m, size=grid_hw, mode="area")[0, 0]
    return pooled.numpy() > 0.5


# --------------------------------------------------------------------------------------
# model loading -- every arm is optional, so nothing loads unless an arm needs it
# --------------------------------------------------------------------------------------
def load_lpips(device: str):
    """AlexNet LPIPS with spatial=True, so the forward returns per-pixel maps."""
    import lpips as lpips_pkg
    return lpips_pkg.LPIPS(net="alex", spatial=True).to(device).eval()


def load_dinov2(model_id: str, device: str):
    """DINOv2 ViT-B/14 WITH REGISTERS, from the local HF cache.

    The register variant is not a preference. Plain DINOv2 carries high-norm artifact
    patches that corrupt spatial maps -- which is precisely what registers were introduced
    to fix -- and three of the eight arms (dino_cls, dino_patch, selfsim) read either the
    patch grid or its full self-similarity. One backbone serves all three, because
    splitting them would confound the dino_patch-vs-selfsim comparison with a backbone
    change.
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model


def load_clip(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


def clip_embed(out) -> torch.Tensor:
    """Pull the projected embedding out of get_{image,text}_features.

    transformers 4.x returned a bare (B, D) tensor. transformers 5.x returns a
    BaseModelOutputWithPooling whose `pooler_output` has been REPLACED by the projected
    embedding. Both are handled so the arm does not silently change meaning with the env.
    """
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def load_depth(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return processor, model


def load_normals(model_id: str, device: str, dtype: torch.dtype):
    from diffusers import MarigoldNormalsPipeline
    pipe = MarigoldNormalsPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# --------------------------------------------------------------------------------------
# DINOv2 feature extraction
# --------------------------------------------------------------------------------------
@torch.no_grad()
def dinov2_features(model, img: Image.Image, dino_hw: Tuple[int, int],
                    device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (cls, patch_tokens, key_tokens) for one image.

    Token layout is [CLS, R registers, N patches]. THE REGISTERS ARE STRIPPED: they are
    global scratch space with no spatial position, so leaving them in and reshaping to a
    grid would shift every patch by R positions and silently scramble the map.

    `key_tokens` is the KEY facet of the LAST block, taken through a forward hook on that
    block's key projection. Keys are the standard choice for structural correspondence --
    the final output tokens are shaped by the training objective and match less cleanly
    across images. (transformers exposes separate query/key/value Linear layers, so no
    fused-qkv split is needed here, unlike the torch.hub DINO v1 layout r29_novelty.py
    works with.)

    cls / patch_tokens are L2-normalised so a cosine is a dot product; key_tokens too, so
    the self-similarity S_ij is a cosine by construction.
    """
    x = to_imagenet_tensor(img, dino_hw, device)

    last = model.encoder.layer[-1]
    store: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        store["key"] = out

    handle = last.attention.attention.key.register_forward_hook(hook)
    try:
        out = model(pixel_values=x)
    finally:
        handle.remove()

    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    hidden = out.last_hidden_state[0].float()            # (1 + R + N, C)
    cls = hidden[0]
    patches = hidden[1 + n_reg:]                         # (N, C)
    keys = store["key"][0].float()[1 + n_reg:]           # (N, C)

    return (F.normalize(cls, dim=-1),
            F.normalize(patches, dim=-1),
            F.normalize(keys, dim=-1))


# --------------------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------------------
@torch.no_grad()
def estimate_depth(img: Image.Image, processor, model, device: str,
                   hw: Tuple[int, int]) -> np.ndarray:
    """Relative inverse depth for one image, resampled to `hw`. float64 (H, W).

    Units are per-image and arbitrary; only differences after the background affine
    alignment mean anything, and only after division by IQR(D_src) do they compare across
    clips.
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    depth = model(**inputs).predicted_depth
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
    depth = F.interpolate(depth.float(), size=hw, mode="bilinear", align_corners=False)
    return depth[0, 0].detach().cpu().numpy().astype(np.float64)


def fit_affine(d_anchor: np.ndarray, d_src: np.ndarray,
               bg: np.ndarray) -> Tuple[float, float]:
    """Least-squares (a, b) minimising ||a * d_anchor + b - d_src||^2 over `bg`.

    Fitted on the union's COMPLEMENT only. Including the union would let the very
    difference being measured pull the alignment toward zero, and no pixel that changed
    in EITHER image may enter the fit -- which is the reason the measurement region is
    the union and not the target mask.
    """
    x = d_anchor[bg]
    y = d_src[bg]
    design = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(a), float(b)


# --------------------------------------------------------------------------------------
# normals
# --------------------------------------------------------------------------------------
@torch.no_grad()
def estimate_normals(pipe, img: Image.Image, hw: Tuple[int, int], steps: int,
                     ensemble: int, seed: int, device: str) -> np.ndarray:
    """Unit surface normals for one image as float64 (H, W, 3).

    Marigold is a DIFFUSION model, so this is stochastic. The generator is re-seeded per
    call with the same seed, and `ensemble` >= 5 means the returned map is an ensemble
    average rather than a single sample. Both are recorded in the CSV.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(resize_to(img, hw), num_inference_steps=steps, ensemble_size=ensemble,
               generator=gen, output_type="np", match_input_resolution=True)
    n = np.asarray(out.prediction[0], dtype=np.float64)          # (H, W, 3) in [-1, 1]
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.clip(norm, 1e-8, None)


# --------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------
def save_panel(out_path: Path, src_img: Image.Image, anchor_img: Image.Image,
               union: np.ndarray, maps: Dict[str, Optional[np.ndarray]],
               row: Dict[str, object]) -> None:
    """2x4 audit panel: the images, the union region, and all four spatial arms' maps.

    A scalar divergence cannot distinguish a genuine edit from a mask that admits half the
    background, and the difference between those two readings is the whole of check-div
    gates 1 and 3. This is the only artefact that shows WHERE each number came from.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = union.shape
    spatial = [("lpips", "LPIPS spatial"),
               ("dino_patch", "DINOv2 patch $1-\\cos$"),
               ("depth", "$|\\Delta$Depth$|$ (bg-aligned)"),
               ("normals", "normal ang. err (deg)")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 2 * 5.0 * h / max(w, 1) + 1.6))
    ax = axes.ravel()

    ax[0].imshow(np.asarray(src_img)); ax[0].set_title("source $I_0$", fontsize=11)
    ax[1].imshow(np.asarray(anchor_img))
    ax[1].set_title("anchor $I_{0,\\mathrm{edit}}$", fontsize=11)

    tint = np.asarray(anchor_img).astype(float) / 255.0
    tint[union] = 0.45 * tint[union] + 0.55 * np.array([0.20, 0.95, 0.45])
    ax[2].imshow(tint)
    ax[2].set_title(f"union $M^{{src}} \\cup M^{{trg}}$  "
                    f"({100 * float(union.mean()):.1f}% of frame)", fontsize=11)

    # The union is defined on the LATENT grid, so it is drawn there too: a speckled
    # region and a compact silhouette give the same union_frac and mean very different
    # things for every masked arm.
    ax[3].imshow(union, cmap="gray", interpolation="nearest")
    ax[3].set_title("union alone (latent-resolution)", fontsize=11)

    for k, (arm, title) in enumerate(spatial):
        a = ax[4 + k]
        m = maps.get(arm)
        if m is None:
            a.text(0.5, 0.5, f"{arm}\nnot computed", ha="center", va="center",
                   fontsize=11, transform=a.transAxes)
            continue
        lo, hi = np.percentile(m, [1, 99])
        a.imshow(m, cmap="viridis", vmin=lo, vmax=max(hi, lo + 1e-9))
        a.contour(union.astype(float), levels=[0.5], colors="red", linewidths=0.5)
        a.set_title(f"{title}\nmasked {row.get(f'd_{arm}_masked', '')}   "
                    f"global {row.get(f'd_{arm}_global', '')}", fontsize=9.5)

    for a in ax:
        a.axis("off")

    scal = "   ".join(
        f"{arm} {row.get(f'd_{arm}_masked')}" for arm in
        ("dino_cls", "clip_image", "clip_prompt", "selfsim")
        if row.get(f"d_{arm}_masked") not in ("", None))
    fig.suptitle(f"{row['case_id']}   type {row['edit_type']}   "
                 f"union {row['union_frac']} (opened x{row['mask_open']})\n{scal}",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# per-case measurement
# --------------------------------------------------------------------------------------
def measure_case(case: Dict[str, object], data_root: Path, anchor_root: Path,
                 mask_dir: Path, models: Dict[str, object], arms: Sequence[str],
                 args: argparse.Namespace,
                 viz_dir: Optional[Path]) -> Tuple[Dict[str, object], List[str]]:
    """Measure one case. Returns (csv row, list of per-arm failure notes).

    A failure is scoped to the ARM it belongs to, not to the case. One clip whose union
    mask leaves no background does not invalidate the seven arms that never look at the
    background, and aborting the case would cost 21 good clips their whole run. The arm
    is left blank in the CSV; r30_b_map.py then refuses to write THAT arm's map, naming
    the clip -- which is the correct outcome, since run_fivebench.py --tau_map demands
    coverage of all 22 pairs and an arm missing one cannot be rendered.
    """
    notes: List[str] = []
    case_id = str(case["case_id"])
    video_name = str(case["video_name"])
    edit_type = int(case["edit_type"])
    device = args.device

    src_path = data_root / "images" / video_name / "00001.jpg"
    anchor_path = anchor_root / f"edit{edit_type}" / f"{video_name}.png"
    mask_path = mask_dir / f"edit{edit_type}" / f"{video_name}.npz"
    for p in (src_path, anchor_path, mask_path):
        if not p.exists():
            raise FileNotFoundError(f"{case_id}: missing {p}")
    split_mask_path = (args.split_mask_dir / f"edit{edit_type}" / f"{video_name}.npz"
                       if args.split_mask_dir is not None else None)

    src_raw = Image.open(src_path).convert("RGB")
    anchor_raw = Image.open(anchor_path).convert("RGB")

    # The comparison grid is the ANCHOR's, which is the geometry the render pipeline
    # actually produced. See the module docstring.
    hw = (anchor_raw.size[1], anchor_raw.size[0])
    h, w = hw
    src_img = resize_to(src_raw, hw)
    anchor_img = resize_to(anchor_raw, hw)

    union, lat_hw, used_src_only = load_union_frame0(
        mask_path, hw, args.mask_open, split_mask_path, args.trg_degenerate_frac)
    bg = ~union

    row: Dict[str, object] = {
        "case_id": case_id, "video_name": video_name, "edit_type": edit_type,
        "grid": f"{h}x{w}", "latent_grid": f"{lat_hw[0]}x{lat_hw[1]}",
        "aspect_src": round(src_raw.size[0] / src_raw.size[1], 4),
        "aspect_anchor": round(anchor_raw.size[0] / anchor_raw.size[1], 4),
        "union_frac": round(float(union.mean()), 6),
        "used_src_only": int(used_src_only),
        "mask_open": args.mask_open,
        "depth_norm": args.depth_norm,
        "normals_steps": args.normals_steps,
        "normals_ensemble": args.normals_ensemble,
        "seed": args.seed,
    }
    for arm in ARMS:
        row[f"d_{arm}_masked"] = ""
        row[f"d_{arm}_global"] = ""

    if not union.any():
        return row, [f"{case_id}: the frame-0 union mask is empty; every masked arm is "
                     f"undefined and the clip cannot be routed"]

    maps: Dict[str, Optional[np.ndarray]] = {}

    # ---- lpips ------------------------------------------------------------------------
    if "lpips" in arms:
        net = models["lpips"]
        # LPIPS expects [-1, 1]. NEVER mask by multiplying the INPUT images: a black
        # rectangle creates edges the network reads as content, and the "distance" then
        # measures the rectangle. Masking is applied to the OUTPUT maps only.
        a = to_unit_tensor(src_img, hw, device) * 2.0 - 1.0
        b = to_unit_tensor(anchor_img, hw, device) * 2.0 - 1.0
        with torch.no_grad():
            total, per_layer = net(a, b, retPerLayer=True)
        # With spatial=True every returned map is already upsampled to the input HW, so
        # mask-averaging them is ALGEBRAICALLY IDENTICAL to per-layer masking with soft
        # area-weighted mask downsampling (bilinear upsampling is a partition of unity),
        # and strictly better than binarising a downsampled mask at the ~29x51 deep taps.
        m = total[0, 0].detach().cpu().numpy().astype(np.float64)
        maps["lpips"] = m
        row["d_lpips_masked"] = round(masked_mean(m, union), 6)
        row["d_lpips_global"] = round(float(m.mean()), 6)
        # Per-layer profile is a DIAGNOSTIC column only, never a sub-arm: `lins` is
        # calibrated for the summed total, so a single tap is not a calibrated distance.
        for i, layer in enumerate(per_layer):
            lm = layer[0, 0].detach().cpu().numpy().astype(np.float64)
            row[f"lpips_l{i}_masked"] = round(masked_mean(lm, union), 6)

    # ---- dinov2 arms ------------------------------------------------------------------
    if any(a in arms for a in ("dino_cls", "dino_patch", "selfsim")):
        model = models["dinov2"]
        # Trim to a multiple of the patch size: 480x832 -> 476x826 -> a 34x59 grid.
        dh = (h // DINOV2_PATCH) * DINOV2_PATCH
        dw = (w // DINOV2_PATCH) * DINOV2_PATCH
        gh, gw = dh // DINOV2_PATCH, dw // DINOV2_PATCH
        row["dino_grid"] = f"{gh}x{gw}"

        cls_s, pat_s, key_s = dinov2_features(model, src_img, (dh, dw), device)
        cls_a, pat_a, key_a = dinov2_features(model, anchor_img, (dh, dw), device)
        if pat_s.shape[0] != gh * gw:
            raise ValueError(f"{case_id}: DINOv2 returned {pat_s.shape[0]} patch tokens, "
                             f"expected {gh}x{gw} = {gh * gw}. Register stripping or the "
                             f"grid derivation is wrong.")
        union_tok = mask_to_grid(union, (gh, gw))

        if "dino_cls" in arms:
            d = float(1.0 - torch.dot(cls_s, cls_a).item())
            row["d_dino_cls_masked"] = round(d, 6)
            row["d_dino_cls_global"] = round(d, 6)

        if "dino_patch" in arms:
            # MEAN-OF-COSINES: per-position (1 - cos) FIRST, then averaged. The
            # alternative -- cosine of the mean descriptors -- is quadratic rather than
            # linear in the edited area, so it under-reads small edits and over-reads
            # large ones in a way that is purely an artefact of the reduction.
            per_pos = (1.0 - (pat_s * pat_a).sum(-1)).cpu().numpy().astype(np.float64)
            grid = per_pos.reshape(gh, gw)
            row["d_dino_patch_masked"] = round(masked_mean(grid, union_tok), 6)
            row["d_dino_patch_global"] = round(float(grid.mean()), 6)
            # Upsample to pixels for the panel only; the number above is computed on the
            # token grid, where the descriptors actually live.
            maps["dino_patch"] = np.asarray(
                Image.fromarray(grid.astype(np.float32)).resize((w, h), Image.NEAREST),
                dtype=np.float64)
            row["dino_patch_union_tokens"] = int(union_tok.sum())

        if "selfsim" in arms:
            # Splice's structure distance: S_ij = cos(k_i, k_j) WITHIN each image, then
            # the Frobenius norm of the difference. It compares ARRANGEMENT, not feature
            # values, so it is blind to appearance change that preserves layout -- which
            # is exactly what makes it a different arm from dino_patch.
            #
            # Kept FULL-IMAGE and never masked: a self-similarity restricted to the union
            # would compare a different number of tokens per clip and stop being
            # comparable across the dataset.
            #
            # WATCH-ITEM, analysis not code: S is indexed by ABSOLUTE position, so a pure
            # TRANSLATION permutes S and registers as structure change. Qwen anchors do
            # sometimes shift an object even for a recolour, which would inflate this arm
            # on exactly the clips it should score lowest. Inspect its top-scoring clips.
            s_src = key_s @ key_s.T
            s_anc = key_a @ key_a.T
            n_tok = s_src.shape[0]
            d = float(torch.linalg.norm(s_src - s_anc).item() / n_tok)
            row["d_selfsim_masked"] = round(d, 6)
            row["d_selfsim_global"] = round(d, 6)
            row["selfsim_tokens"] = n_tok
            del s_src, s_anc

    # ---- clip arms --------------------------------------------------------------------
    if any(a in arms for a in ("clip_image", "clip_prompt")):
        model, processor = models["clip"]
        if "clip_image" in arms:
            with torch.no_grad():
                inp = processor(images=[src_img, anchor_img],
                                return_tensors="pt").to(device)
                e = F.normalize(clip_embed(model.get_image_features(**inp)), dim=-1)
            row["d_clip_image_masked"] = round(float(1.0 - (e[0] @ e[1]).item()), 6)
            row["d_clip_image_global"] = row["d_clip_image_masked"]

        if "clip_prompt" in arms:
            # TEXT ONLY -- this arm never touches the anchor, which makes it a control:
            # it measures how far the prompt asks the edit to move, with no evidence that
            # anything actually moved. Directional/projected CLIP variants were dropped at
            # planning: the directional cosine is scale-normalised and measures ALIGNMENT,
            # not magnitude, which is the quantity `b` needs.
            key = "prompt" if args.clip_prompt_field == "prompt" else "word"
            p_src, p_trg = str(case[f"src_{key}"]), str(case[f"trg_{key}"])
            with torch.no_grad():
                inp = processor(text=[p_src, p_trg], return_tensors="pt",
                                padding=True, truncation=True).to(device)
                e = F.normalize(clip_embed(model.get_text_features(**inp)), dim=-1)
            row["d_clip_prompt_masked"] = round(float(1.0 - (e[0] @ e[1]).item()), 6)
            row["d_clip_prompt_global"] = row["d_clip_prompt_masked"]
            row["clip_prompt_field"] = args.clip_prompt_field

    # ---- depth ------------------------------------------------------------------------
    if "depth" in arms:
        processor, model = models["depth"]
        bg_frac = float(bg.mean())
        row["bg_frac"] = round(bg_frac, 6)
        skip: Optional[str] = None

        # ⚠ MEASURED ON THIS DATASET, and it bites: R26's grounding-mask union covers a
        # median ~38% of the frame as dumped, 76% on 0034_cows and 100% on 0042_gym-ball.
        # Without a background there is no scale/shift-free reference, so the depth arm
        # has nothing to align against. That is a property of the masks, not a bug here.
        if bg_frac < MIN_BG_FRACTION:
            skip = (f"the union's complement is {bg_frac:.3f} of the frame "
                    f"(< {MIN_BG_FRACTION}); the affine alignment cannot be fitted "
                    f"independently of the changed region")

        if skip is None:
            d_src = estimate_depth(src_img, processor, model, device, hw)
            d_anc = estimate_depth(anchor_img, processor, model, device, hw)
            if float(np.std(d_anc[bg])) <= 0.0:
                skip = ("anchor depth is constant over the union's complement; "
                        "the affine fit is degenerate")

        if skip is None:
            a, b = fit_affine(d_anc, d_src, bg)
            resid_abs = np.abs(d_src - (a * d_anc + b))
            q1, q3 = np.percentile(d_src, [25, 75])
            iqr_src = float(q3 - q1)
            row["depth_a"] = round(a, 6)
            row["depth_b"] = round(b, 6)
            row["depth_iqr_src"] = round(iqr_src, 6)
            if args.depth_norm == "iqr" and iqr_src <= 0.0:
                skip = ("source depth map has zero IQR; the scene spans no depth at all, "
                        "so the normalised divergence is undefined")

        if skip is None:
            scaled = resid_abs / iqr_src if args.depth_norm == "iqr" else resid_abs
            maps["depth"] = scaled
            row["d_depth_masked"] = round(masked_mean(scaled, union), 6)
            row["d_depth_global"] = round(float(scaled.mean()), 6)
            # Raw (un-normalised) companion, so switching --depth_norm later is a CSV read.
            row["depth_raw_masked"] = round(masked_mean(resid_abs, union), 6)
            # Robust noise floor of the background residual. Not a threshold here -- it is
            # the scale a masked mean has to beat to mean anything, and a masked mean at or
            # below it says the mask is loose rather than that the edit moved geometry.
            r_bg = (d_src - (a * d_anc + b))[bg]
            row["depth_bg_mad"] = round(
                float(1.4826 * np.median(np.abs(r_bg - np.median(r_bg)))), 6)
        else:
            notes.append(f"{case_id}: depth -- {skip}. Arm left blank for this clip, so "
                         f"r30_b_map.py will refuse the depth map.")

    # ---- normals ----------------------------------------------------------------------
    if "normals" in arms:
        pipe = models["normals"]
        n_src = estimate_normals(pipe, src_img, hw, args.normals_steps,
                                 args.normals_ensemble, args.seed, device)
        n_anc = estimate_normals(pipe, anchor_img, hw, args.normals_steps,
                                 args.normals_ensemble, args.seed, device)
        # A direct estimator needs NO alignment: both maps are already unit vectors in the
        # camera frame, so the angle between them is meaningful as it stands.
        cos = np.clip((n_src * n_anc).sum(-1), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        maps["normals"] = ang
        row["d_normals_masked"] = round(masked_mean(ang, union), 6)
        row["d_normals_global"] = round(float(ang.mean()), 6)
        row["normals_bg_median"] = round(float(np.median(ang[bg])), 6)

    if viz_dir is not None:
        save_panel(viz_dir / f"{case_id}.png", src_img, anchor_img, union, maps, row)

    return row, notes


# --------------------------------------------------------------------------------------
def build_fieldnames(arms: Sequence[str]) -> List[str]:
    head = ["case_id", "video_name", "edit_type"]
    for arm in ARMS:
        head += [f"d_{arm}_masked", f"d_{arm}_global"]
    head += ["union_frac", "used_src_only", "mask_open", "grid", "latent_grid", "dino_grid",
             "aspect_src", "aspect_anchor"]
    head += [f"lpips_l{i}_masked" for i in range(5)]
    head += ["dino_patch_union_tokens", "selfsim_tokens", "clip_prompt_field",
             "depth_a", "depth_b", "depth_iqr_src", "depth_raw_masked",
             "depth_bg_mad", "bg_frac", "depth_norm",
             "normals_bg_median", "normals_steps", "normals_ensemble", "seed"]
    return head


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path, required=True,
                   help="FiVE-Bench root holding images/")
    p.add_argument("--anchor_root", type=Path, required=True,
                   help="Anchor root; the anchor is {anchor_root}/edit{T}/{video}.png")
    p.add_argument("--mask_dir", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_masks"),
                   help="R26's frame-0 union masks, {mask_dir}/edit{T}/{video}.npz. "
                        "These are the pass-1 output of r26_dump.sh under exactly R30's "
                        "specification (--blend_sched zero, --vp_mode vp, same anchors, "
                        "same cases.json, --step 15 --fg_boost_factor 4 --seed 0), which "
                        "is why stage 1 runs no diffusion at all.")
    p.add_argument("--mask_open", type=int, default=1,
                   help="Rounds of morphological OPENING (erode then dilate, 8-connected) "
                        "applied to the union on the LATENT grid before upsampling. "
                        "DEFAULT 1 (settled 2026-09-10, reversing an earlier 'use exactly "
                        "as dumped' call): the mask is cross-attention response to the "
                        "trigger word thresholded at mask_soft > 0, so background texture "
                        "routinely lights up as isolated speckle alongside the real "
                        "object. A single erosion+dilation round erases anything smaller "
                        "than the structuring element outright while restoring the "
                        "boundary of anything large enough to have an interior -- measured "
                        "on the 22 cases: 0001_bus 0.50 -> 0.27 (47 connected components "
                        "-> 3), 0090_A_deer 0.56 -> 0.23, while an already-compact mask "
                        "(0011_lucia_e5) barely moves (0.112 -> 0.096, only isolated "
                        "single-token dots removed). Unlike DILATION (rejected: one step "
                        "nearly doubles coverage at this resolution by growing the noisy "
                        "region outward), opening only removes noise, never adds area. "
                        "Set 0 to reproduce the earlier 'exactly as dumped' behaviour.")
    p.add_argument("--split_mask_dir", type=Path, default=None,
                   help="Optional per-clip SEPARATED src/trg masks, "
                        "{split_mask_dir}/edit{T}/{video}.npz with M_src/M_trg (from "
                        "run_fivebench.py --split_mask_dump_dir). For a clip with a file "
                        "here whose M_trg covers more than --trg_degenerate_frac of the "
                        "frame, the union falls back to M_src ALONE -- settled 2026-09-10 "
                        "for 0042_gym-ball (a removal: trg_word 'without a heavy gym "
                        "ball' is a negation with no visual referent, so trg grounding "
                        "never localizes and saturates to exactly 1.0 on every frame; "
                        "opening cannot fix this, it is not speckle). Default None: no "
                        "clip is affected, byte-identical to before this flag existed. "
                        "Only 0042_gym-ball has a split-mask file so far.")
    p.add_argument("--trg_degenerate_frac", type=float, default=0.90,
                   help="Threshold on M_trg's own frame-0 coverage above which it is "
                        "treated as degenerate (see --split_mask_dir). Matches the "
                        "'union > 0.90 of the frame' threshold already used elsewhere in "
                        "this script's reporting, for one consistent reading of 'the mask "
                        "covers basically everything.'")
    p.add_argument("--arms", type=str, default=",".join(ARMS),
                   help="Comma-separated subset of the eight arms to compute. Every arm "
                        "not listed is left blank in the CSV and its model is never "
                        "loaded -- useful for re-running one arm after a model fetch.")
    p.add_argument("--depth_norm", choices=("iqr", "raw"), default="iqr",
                   help="Whether the depth residual is divided by IQR(D_src). 'iqr' "
                        "(default) is required for CROSS-CLIP comparison: the background "
                        "affine fit cancels the anchor's scale and shift, but leaves the "
                        "residual in the SOURCE map's own arbitrary units, so without it "
                        "a shallow scene reads a small divergence whatever the edit did. "
                        "'raw' keeps the un-normalised residual; both are always emitted, "
                        "this flag only chooses which one d_depth_masked carries.")
    p.add_argument("--clip_prompt_field", choices=("prompt", "word"), default="prompt",
                   help="Which cases.json text pair the clip_prompt arm embeds: the full "
                        "src_prompt/trg_prompt (default) or the trigger words "
                        "src_word/trg_word.")
    p.add_argument("--normals_steps", type=int, default=4,
                   help="Marigold denoising steps. Recorded in the CSV: this arm is the "
                        "only stochastic one, so its settings are part of its definition.")
    p.add_argument("--normals_ensemble", type=int, default=5,
                   help="Marigold ensemble size. >= 5 so the map returned is an ensemble "
                        "AVERAGE rather than one diffusion sample; at 1 the arm would "
                        "carry sampling noise into every routed b.")
    p.add_argument("--dinov2_model_id", type=str, default=DINOV2_MODEL_ID)
    p.add_argument("--clip_model_id", type=str, default=CLIP_MODEL_ID)
    p.add_argument("--depth_model_id", type=str, default=DEPTH_MODEL_ID)
    p.add_argument("--normals_model_id", type=str, default=NORMALS_MODEL_ID)
    p.add_argument("--viz_dir", type=Path, default=None,
                   help="If given, write one 2x3 audit panel per case. The CSV is "
                        "unaffected either way.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true",
                   help="Run Marigold in float16. Off by default: this arm is already "
                        "stochastic and halving its precision adds a second source of "
                        "run-to-run variation for a model that is not the bottleneck.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {list(ARMS)}")

    data_root = args.data_root.expanduser().resolve()
    anchor_root = args.anchor_root.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    viz_dir = args.viz_dir.expanduser() if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.expanduser().read_text())

    print("=== R30 stage 1: static anchor-vs-source divergences ===", flush=True)
    print(f"cases       {args.cases} ({len(cases)} cases)", flush=True)
    print(f"data_root   {data_root}", flush=True)
    print(f"anchor_root {anchor_root}", flush=True)
    print(f"mask_dir    {mask_dir}   (R26 pass-1 union, frame 0)", flush=True)
    print(f"arms        {', '.join(arms)}", flush=True)
    print(f"masks       morphological opening x{args.mask_open} (8-connected, "
          f"latent grid)   depth_norm {args.depth_norm}", flush=True)
    print(f"split_masks {args.split_mask_dir if args.split_mask_dir else '(off)'}"
          f"{f'  trg_degenerate_frac={args.trg_degenerate_frac}' if args.split_mask_dir else ''}",
          flush=True)
    print(f"normals     {args.normals_model_id}  steps {args.normals_steps}  "
          f"ensemble {args.normals_ensemble}  (STOCHASTIC, seed {args.seed})", flush=True)
    print(f"device      {args.device}   seed {args.seed}", flush=True)
    print(f"viz_dir     {viz_dir if viz_dir else '(off)'}", flush=True)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    # Load only what the requested arms need. Every load reads the local HF cache; under
    # HF_HUB_OFFLINE=1 a miss is a hard failure here, at startup, rather than 20 cases in.
    models: Dict[str, object] = {}
    if "lpips" in arms:
        models["lpips"] = load_lpips(args.device)
    if any(a in arms for a in ("dino_cls", "dino_patch", "selfsim")):
        models["dinov2"] = load_dinov2(args.dinov2_model_id, args.device)
    if any(a in arms for a in ("clip_image", "clip_prompt")):
        models["clip"] = load_clip(args.clip_model_id, args.device)
    if "depth" in arms:
        models["depth"] = load_depth(args.depth_model_id, args.device)
    if "normals" in arms:
        models["normals"] = load_normals(
            args.normals_model_id, args.device,
            torch.float16 if args.fp16 else torch.float32)
    print(f"loaded      {', '.join(sorted(models))}\n", flush=True)

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for case in cases:
        # Re-seed per case so Marigold's samples do not depend on how many cases ran
        # before this one -- otherwise the whole CSV changes when --arms changes.
        set_seed(args.seed)
        row, notes = measure_case(case, data_root, anchor_root, mask_dir,
                                  models, list(arms), args, viz_dir)
        rows.append(row)
        failures.extend(notes)
        for n in notes:
            print(f"[FAIL] {n}", flush=True)
        vals = "  ".join(
            f"{a}={row[f'd_{a}_masked']}" for a in arms
            if row.get(f"d_{a}_masked") not in ("", None))
        tag = "ok  " if not notes else "part"
        src_only = "  (src-only, trg degenerate)" if row.get("used_src_only") else ""
        print(f"[{tag}] {row['case_id']:22s} type {row['edit_type']} "
              f"union {float(row['union_frac']):.3f}  {vals}{src_only}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames(arms)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nwrote {args.out}  ({len(rows)} rows)", flush=True)
    if viz_dir is not None:
        print(f"wrote audit panels to {viz_dir}", flush=True)

    # ---- spread, the gate that decides whether any arm can route at all ---------------
    print(f"\n{'arm':<12} {'n_distinct':>10} {'min':>10} {'max':>10}  "
          f"{'argmin':<22} {'argmax':<22}", flush=True)
    degenerate: List[str] = []
    for arm in arms:
        vals = [(float(r[f"d_{arm}_masked"]), str(r["case_id"])) for r in rows
                if r.get(f"d_{arm}_masked") not in ("", None)
                and float(r[f"d_{arm}_masked"]) == float(r[f"d_{arm}_masked"])]
        if not vals:
            print(f"{arm:<12} {'-':>10} {'(no measured values)':>32}", flush=True)
            degenerate.append(arm)
            continue
        lo, hi = min(vals), max(vals)
        n_distinct = len({v for v, _ in vals})
        print(f"{arm:<12} {n_distinct:>10} {lo[0]:>10.4f} {hi[0]:>10.4f}  "
              f"{lo[1]:<22} {hi[1]:<22}", flush=True)
        if n_distinct < 3:
            degenerate.append(arm)
    if degenerate:
        # Not fatal here -- r30_b_map.py refuses to write a map with fewer than 3 distinct
        # b, which is where the decision belongs. Surfaced now so check-div sees it.
        print(f"\n** {len(degenerate)} arm(s) with fewer than 3 distinct values: "
              f"{degenerate}. r30_b_map.py will refuse to write these. **", flush=True)

    # The argmin and argmax clips are pinned to b_min and b_max by construction (the map
    # normalises against the dataset min/max), so a metric failure at either extreme drags
    # the whole map. check-div gate 5 reads this.
    print("\nNote: the argmin/argmax clips above are pinned to b_min/b_max by the "
          "min-max normalisation. Read their panels before trusting any arm.", flush=True)

    # check-div gate 2: source and anchor differ in aspect by construction (864x480 vs
    # 832x480, the pipeline's own anisotropic Resize). Harmless HERE because both sides
    # are compared on the anchor grid, but worth stating so the gate is answerable.
    mism = [r["case_id"] for r in rows
            if abs(float(r["aspect_src"]) - float(r["aspect_anchor"]))
            / max(float(r["aspect_src"]), 1e-9) > 0.02]
    if mism:
        print(f"\nnote: source and anchor differ in aspect on {len(mism)} case(s) "
              f"(the pipeline's own Resize). Both sides were compared on the ANCHOR "
              f"grid, so no warp was introduced.", flush=True)

    # check-div gate 3 (coverage) was DROPPED on 2026-09-10: area(trg_fg)/area(union)
    # is not computable from a union-only dump. `union_frac` below is the union's share of
    # the frame and says nothing about reachability -- do not read it as the old gate.
    fr = sorted((float(r["union_frac"]), str(r["case_id"])) for r in rows)
    print(f"\nunion coverage (opened x{args.mask_open}): median {fr[len(fr) // 2][0]:.3f}   "
          f"min {fr[0][0]:.3f} ({fr[0][1]})   max {fr[-1][0]:.3f} ({fr[-1][1]})",
          flush=True)
    big = [c for v, c in fr if v > 0.90]
    if big:
        print(f"  ** {len(big)} clip(s) with union > 0.90 of the frame: {big}. "
              f"There is almost no background there, so the masked reduction is the "
              f"global one and the depth arm has nothing to align against. **", flush=True)

    # An arm missing even one clip cannot be rendered: run_fivebench.py --tau_map demands
    # coverage of all 22 pairs (run_fivebench.py:260). Report that per arm, since a single
    # degenerate clip disqualifies one arm and leaves the other seven untouched.
    incomplete = {}
    for arm in arms:
        miss = [str(r["case_id"]) for r in rows
                if r.get(f"d_{arm}_masked") in ("", None)]
        if miss:
            incomplete[arm] = miss
    if incomplete:
        print(f"\n** {len(incomplete)} arm(s) do not cover all {len(rows)} clips and "
              f"therefore CANNOT be rendered: **", flush=True)
        for arm, miss in incomplete.items():
            print(f"     {arm:<12} missing {len(miss)}: {miss}", flush=True)

    print(f"\nper-arm failures: {len(failures)}", flush=True)
    if failures:
        print("\nFailures -- each is scoped to one arm on one clip:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
