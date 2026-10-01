"""R31 stage 2: per-token, per-frame divergence between the source and the UNBLENDED target.

Where R30 reduced each divergence to ONE SCALAR per clip over a mask, R31 keeps the MAP.
The output of every arm is a field shaped ``[F_lat, 1560]`` -- one value per latent token,
per latent frame -- because that is exactly what the blend indexes: ``blend_rho_tok`` in
``edit_causal_inference.py`` is cache-aligned over ``frame_seq_length = 1560`` tokens.

THE TWO REDUCTIONS
------------------
Every arm starts somewhere different and must land on the same ``[F_lat, 30, 52]`` grid.

SPATIAL, verified from the model code:
    480x832 pixels --VAE /8--> 60x104 --DiT patch_size=(1,2,2)--> 30x52 = 1560
  * lpips / normals         : maps are 480x832  -> exact 16x16 area-mean per token
  * dino_patch              : DINOv3 patch 16   -> 30x52 ALREADY. No resampling at all
                              (480/16 = 30, 832/16 = 52). This is why v3 replaces R30's
                              DINOv2: at patch 14 the grid is 34x59 and every token would
                              be a ~1.13x blend of patches with edges falling mid-token.
  * latent                  : 60x104 already    -> exact 2x2 pool (the DiT's own patchify)

TEMPORAL -- the VAE's own grouping, not a resample:
    latent frame 0       <-> pixel frame 0          (it stands ALONE)
    latent frame f >= 1  <-> pixel frames [4f-3, 4f]
  81 pixel frames -> 21 latent frames, covering each exactly once. An even 4-way split
  would offset every latent frame by one and smear the field forward in time.
  ``latent`` skips this entirely -- it is already per latent frame.
  ⚠ This is the correct nominal ALIGNMENT, not the encoder's exact support: the Wan VAE
  uses CausalConv3d with temporal receptive field > 1, so latent frame f also depends on
  pixel frames before its nominal group. If the field looks temporally smeared at group
  boundaries, suspect this first.

ARMS
----
  lpips       LPIPS(net='alex', spatial=True) per-pixel map. NEVER mask the inputs --
              black rectangles create edges the network reads as content.
  dino_patch  DINOv3 ViT-B/16, per-position 1-cos (mean-of-cosines, NOT cosine-of-means,
              which is quadratic rather than linear in edited area). ⚠ The prefix length
              is ``1 + config.num_register_tokens`` READ FROM THE LOADED MODEL: the
              checkpoint carries 4 registers while a default ``DINOv3ViTConfig()`` reports
              0, and skipping only the CLS token shifts every patch 4 slots (= 64 px) and
              silently points the gate at the wrong place.
  normals     Marigold Normals, per-pixel angular error in degrees. Stochastic -- seed and
              ensemble_size are recorded in the npz. NOTE this is a DIRECT normal
              estimator, not a finite-difference of depth, so it needs no alignment of
              any kind -- which is exactly why it survives the cut below.
  latent      mean over C of |z_trg - z_src| from stage 1's npz. NO model, NO encoder:
              it is already in the latent domain, which is the point.

WHY THERE IS NO `depth` ARM (dropped 2026-09-11, during /build-step stage2)
---------------------------------------------------------------------------
It was specced, implemented, and removed on the user's call, because monocular relative
depth is scale/shift-free per image and therefore REQUIRES an affine alignment before two
depth maps can be differenced -- and R31 has no mask to fit that alignment on. R30 could
fit over the union mask's COMPLEMENT, guaranteeing no changed pixel entered the fit; R31's
whole premise is that the field IS the output, so the edit is always inside the data.

Measured on a synthetic ramp with a known answer (a = 0.5, b = -0.25):
  * plain least-squares over all pixels returns a = 0.038 at a 13%-area edit, and the
    corrupted fit INVERTS the residual ranking -- 0 of 3200 edit pixels land in the
    top-20% residuals while 4800 honest pixels do -- so a trimmed refit seeded from it
    discards the good data and moves FURTHER from the truth (0.038 -> 0.027);
  * a robust median/IQR seed plus one trimmed refit recovers a = 0.5000 exactly at 13%,
    but still collapses to a = 0.050 by 30% edit area.
R30 measured union-mask coverage with a MEDIAN of 0.50 across these 22 clips, so 30%+
edits are typical here, not a corner case. An arm whose alignment silently fails on the
median clip cannot gate a blend. Dropped rather than patched.

NORMALISATION
-------------
Per-clip min/max to ``m`` in [0, 1] (settled with the user). Maximally spatial per clip and
robust to cross-clip scale differences. ⚠ Its one sharp edge, which ``check-stage2`` exists
to catch: a clip with NO real divergence still gets a full-range ``m`` -- pure noise
stretched over the whole tau span. ``d_min``/``d_max`` are written raw so that is visible.

OUTPUT
------
  {out_root}/{arm}/edit{T}/{video_name}.npz
      d      float32 [F_lat, 1560]   raw divergence, per latent token
      m      float32 [F_lat, 1560]   per-clip min-max normalised to [0, 1]
      d_min, d_max, grid_h, grid_w, n_latent_frames, n_pixel_frames, arm, + provenance
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# R31 reuses R30's loaders verbatim rather than copying them -- see the module docstring.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_divergence import (  # noqa: E402
    NORMALS_MODEL_ID,
    estimate_normals,
    load_lpips,
    load_normals,
    resize_to,
    set_seed,
    to_unit_tensor,
)

ARMS: Tuple[str, ...] = ("lpips", "dino_patch", "normals", "latent")
PIXEL_ARMS: Tuple[str, ...] = ("lpips", "dino_patch", "normals")

# DINOv3 replaces R30's DINOv2: patch 16 divides 480 and 832 exactly, so the patch grid
# IS the token grid. Gated on HF; downloaded 2026-09-11 and verified under HF_HUB_OFFLINE=1.
DINOV3_MODEL_ID: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# The pipeline's own transform size (run_fivebench.py / r31_stage1.py).
PIXEL_H: int = 480
PIXEL_W: int = 832
# 480/8/2 x 832/8/2 -- VAE stride 8 then the DiT's (1,2,2) patchify.
GRID_H: int = 30
GRID_W: int = 52
FRAME_SEQ_LENGTH: int = GRID_H * GRID_W          # 1560, == pipeline's self.frame_seq_length


# --------------------------------------------------------------------------------------
# reductions
# --------------------------------------------------------------------------------------
def latent_frame_groups(n_pixel_frames: int, n_latent_frames: int) -> List[List[int]]:
    """The VAE's own temporal grouping -- frame 0 alone, then blocks of 4.

    Wan's VAE has ``temperal_downsample=[False, True, True]`` (4x) with the first frame
    standing alone, so ``n_pixel = 4 * (n_latent - 1) + 1``. Refuses any other pairing
    rather than silently tiling a wrong number of frames.
    """
    expected = 4 * (n_latent_frames - 1) + 1
    if n_pixel_frames != expected:
        raise ValueError(
            f"R31: {n_pixel_frames} pixel frames does not match {n_latent_frames} latent "
            f"frames (expected {expected} = 4*(F_lat-1)+1). A mismatch here would misalign "
            "the divergence from the tokens it gates.")
    groups = [[0]] + [list(range(4 * f - 3, 4 * f + 1)) for f in range(1, n_latent_frames)]
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(n_pixel_frames)), "grouping must tile exactly once"
    return groups


def to_token_grid(maps: torch.Tensor) -> torch.Tensor:
    """Area-mean any ``[N, H, W]`` stack onto the ``[N, GRID_H, GRID_W]`` token grid.

    Area rather than bilinear: a token covers a whole patch, and what it should carry is
    the MEAN divergence over that patch, not a point sample. Same convention as
    ``r30_divergence.py:mask_to_grid``, minus its ``> 0.5`` threshold -- this field is
    continuous and is never binarised.
    """
    if maps.dim() != 3:
        raise ValueError(f"expected [N, H, W], got {tuple(maps.shape)}")
    out = F.interpolate(maps.unsqueeze(1).float(), size=(GRID_H, GRID_W), mode="area")
    return out[:, 0]


def reduce_temporal(per_pixel_frame: torch.Tensor, groups: Sequence[Sequence[int]],
                    how: str) -> torch.Tensor:
    """``[F_pix, gh, gw]`` -> ``[F_lat, gh, gw]`` by the VAE's frame grouping."""
    rows = []
    for g in groups:
        sel = per_pixel_frame[list(g)]
        rows.append(sel.amax(dim=0) if how == "max" else sel.mean(dim=0))
    return torch.stack(rows, dim=0)


# --------------------------------------------------------------------------------------
# DINOv3
# --------------------------------------------------------------------------------------
def load_dinov3(model_id: str, device: str):
    """DINOv3 ViT/16 plus the prefix length its CHECKPOINT actually uses.

    Returns ``(model, n_prefix)`` where ``n_prefix = 1 + num_register_tokens`` read from
    the LOADED config. Never hardcode this: the checkpoint carries 4 registers while a
    bare ``DINOv3ViTConfig()`` reports 0, and skipping only the CLS token shifts every
    patch 4 slots (64 px at patch 16) with no error -- a silently misplaced gate.
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    cfg = model.config
    if cfg.patch_size != 16:
        raise ValueError(
            f"R31: DINOv3 patch_size is {cfg.patch_size}, not 16. The whole reason v3 "
            f"replaces R30's v2 here is that 480/16 = {GRID_H} and 832/16 = {GRID_W} land "
            "exactly on the token grid; another patch size reintroduces a resample.")
    n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
    return model, n_prefix


@torch.no_grad()
def dinov3_patch_divergence(model, n_prefix: int, src: Image.Image, trg: Image.Image,
                            device: str) -> np.ndarray:
    """Per-position ``1 - cos`` on the native 30x52 patch grid. float64 (GRID_H, GRID_W).

    MEAN-OF-COSINES: the cosine is taken per position and only then laid out as a map.
    Cosine-of-means would be quadratic rather than linear in edited area (R30's note).
    """
    from r30_divergence import IMAGENET_MEAN, IMAGENET_STD

    def prep(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(resize_to(img, (PIXEL_H, PIXEL_W)), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - torch.tensor(IMAGENET_MEAN)[:, None, None]) \
            / torch.tensor(IMAGENET_STD)[:, None, None]
        return t.unsqueeze(0).to(device)

    fa = model(pixel_values=prep(src)).last_hidden_state
    fb = model(pixel_values=prep(trg)).last_hidden_state
    pa = fa[:, n_prefix:]
    pb = fb[:, n_prefix:]
    # The assert that catches a wrong patch size, a stray register, or a changed input
    # resolution -- all of which would otherwise land as a quietly displaced map.
    if pa.shape[1] != FRAME_SEQ_LENGTH:
        raise ValueError(
            f"R31: DINOv3 returned {pa.shape[1]} patch tokens after stripping "
            f"{n_prefix} prefix tokens, expected {FRAME_SEQ_LENGTH} "
            f"({GRID_H}x{GRID_W}). seq_len was {fa.shape[1]}.")
    d = 1.0 - F.cosine_similarity(pa, pb, dim=-1)          # [1, 1560]
    return d[0].reshape(GRID_H, GRID_W).double().cpu().numpy()


# --------------------------------------------------------------------------------------
# frame IO
# --------------------------------------------------------------------------------------
def load_frames(dir_or_video: Path, n_expected: Optional[int] = None) -> List[Image.Image]:
    """Load a directory of zero-padded PNG frames, or an mp4, as RGB PIL images."""
    if dir_or_video.is_dir():
        files = sorted(dir_or_video.glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no PNG frames in {dir_or_video}")
        frames = [Image.open(f).convert("RGB") for f in files]
    else:
        from diffusers.utils import load_video
        frames = [f.convert("RGB") for f in load_video(str(dir_or_video))]
    if n_expected is not None and len(frames) != n_expected:
        frames = frames[:n_expected]
        if len(frames) != n_expected:
            raise ValueError(f"{dir_or_video}: {len(frames)} frames, expected {n_expected}")
    return frames


# --------------------------------------------------------------------------------------
# per-case measurement
# --------------------------------------------------------------------------------------
def native_frame_indices(n_latent: int) -> List[int]:
    """The pixel frames whose NATIVE maps are persisted -- one per latent frame.

    The grid's default column set is exactly one column per latent frame, using the LAST
    pixel frame of each group as the representative (``r31_div_grid.pixel_index_for_latent``).
    Keeping only those makes the native stack 21 frames rather than 81, which is a 4x
    saving at no cost to the figure that consumes it.
    """
    return [0 if f == 0 else 4 * f for f in range(n_latent)]


def measure_case(arm: str, src_frames: List[Image.Image], trg_frames: List[Image.Image],
                 z_trg: np.ndarray, z_src: np.ndarray, models: Dict[str, object],
                 args: argparse.Namespace, device: str
                 ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Return ``(d [F_lat, 1560] float32, d_native or None, diagnostics)``.

    ``d_native`` is the arm's map BEFORE the spatial reduction, kept only for the frames
    the grid samples (see ``native_frame_indices``) and stored float16. It exists so the
    figure can show each arm at its own native resolution alongside the reduced field --
    i.e. so a reader can see what the 16x16 area-mean discarded. Only ``lpips`` and
    ``normals`` have a genuinely higher-resolution native map (480x832); ``dino_patch`` is
    natively the 30x52 token grid and ``latent`` natively 60x104, so for those the native
    array is small and nearly redundant, but is written anyway for uniformity.
    """
    n_latent = int(z_trg.shape[0])
    diag: Dict[str, object] = {}

    # ---- arm (e): no model, no encoder, already latent -------------------------------
    if arm == "latent":
        # |z_trg - z_src| averaged over channels -> [F_lat, 60, 104], then the DiT's own
        # 2x2 patchify pool -> [F_lat, 30, 52]. This is the "needs no encoding" arm.
        d = np.abs(z_trg.astype(np.float64) - z_src.astype(np.float64)).mean(axis=1)
        grid = to_token_grid(torch.from_numpy(d))
        diag["latent_hw"] = list(d.shape[1:])
        native = d.astype(np.float16) if args.save_native else None
        return grid.reshape(n_latent, -1).float().numpy(), native, diag

    # ---- pixel arms: one map per PIXEL frame, then both reductions --------------------
    n_pix = len(src_frames)
    groups = latent_frame_groups(n_pix, n_latent)
    per_frame: List[torch.Tensor] = []

    for k in range(n_pix):
        s, t = src_frames[k], trg_frames[k]
        if arm == "lpips":
            # spatial=True returns a per-pixel map. Inputs are in [-1, 1] for LPIPS.
            a = to_unit_tensor(s, (PIXEL_H, PIXEL_W), device) * 2.0 - 1.0
            b = to_unit_tensor(t, (PIXEL_H, PIXEL_W), device) * 2.0 - 1.0
            with torch.no_grad():
                m = models["lpips"](a, b)                      # [1, 1, H, W]
            per_frame.append(m[0, 0].double().cpu())
        elif arm == "dino_patch":
            # Already on the 30x52 grid -- no spatial reduction needed for this arm.
            per_frame.append(torch.from_numpy(
                dinov3_patch_divergence(models["dino"], models["dino_prefix"], s, t, device)))
        elif arm == "normals":
            ns = estimate_normals(models["normals"], s, (PIXEL_H, PIXEL_W),
                                  args.normals_steps, args.normals_ensemble,
                                  args.seed, device)
            nt = estimate_normals(models["normals"], t, (PIXEL_H, PIXEL_W),
                                  args.normals_steps, args.normals_ensemble,
                                  args.seed, device)
            cos = np.clip((ns * nt).sum(axis=-1), -1.0, 1.0)
            per_frame.append(torch.from_numpy(np.degrees(np.arccos(cos))))
        else:
            raise ValueError(f"unknown arm {arm!r}")

    stack = torch.stack(per_frame, dim=0)                      # [F_pix, H, W] or [F_pix,30,52]

    # Keep the PRE-reduction maps for the sampled frames before `stack` is pooled away.
    native = None
    if args.save_native:
        keep = [i for i in native_frame_indices(n_latent) if i < stack.shape[0]]
        native = stack[keep].numpy().astype(np.float16)

    if arm != "dino_patch":
        stack = to_token_grid(stack)                           # -> [F_pix, 30, 52]
    if tuple(stack.shape[1:]) != (GRID_H, GRID_W):
        raise ValueError(f"R31: {arm} produced {tuple(stack.shape[1:])}, expected "
                         f"({GRID_H}, {GRID_W})")
    lat = reduce_temporal(stack, groups, args.temporal_reduce)  # -> [F_lat, 30, 52]

    return lat.reshape(n_latent, -1).float().numpy(), native, diag


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edit_type", type=int, required=True, help="FiVE-Bench edit type 1..6")
    p.add_argument("--arm", type=str, required=True, choices=ARMS)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"),
                   help="FiVE-Bench root; source frames read from videos/{name}.mp4")
    p.add_argument("--target_root", type=Path, required=True,
                   help="Stage-1 unblended target frames: {root}/edit{T}/{video}/*.png "
                        "(i.e. .../r31_stage1/r31_unblended_s7/step06 -- the FINAL index of the\n"
                        "7-step stage-1 run, fully denoised)")
    p.add_argument("--latent_dir", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_latents"),
                   help="Stage-1 npz root ({dir}/edit{T}/{video}.npz)")
    p.add_argument("-o", "--out_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_div"),
                   help="Divergence npz root ({root}/{arm}/edit{T}/{video}.npz)")
    #✨ R35 2026-09-24: override the OUTPUT SUBDIRECTORY name only. R35 crosses each arm
    # with two stage-1 gating regimes (unblended / first2) and needs the four resulting
    # (arm, regime) combos in four SEPARATE trees -- e.g. dino_unblended, dino_first2 --
    # without the npz's own `arm` field losing its true meaning (lpips / dino_patch /
    # normals / latent, still validated by --arm's `choices`). Default None reproduces
    # the exact R31 layout ({out_root}/{arm}/edit{T}/) byte-for-byte.
    p.add_argument("--arm_dir_name", type=str, default=None,
                   help="Output subdirectory name, default --arm. The npz's own `arm` "
                        "field always records the true --arm value regardless of this "
                        "override.")
    p.add_argument("--temporal_reduce", choices=("mean", "max"), default="mean",
                   help="How pixel frames collapse onto a latent frame's group")
    p.add_argument("--save_native", action=argparse.BooleanOptionalAction, default=True,
                   help="Persist each arm's PRE-reduction map for the frames the grid "
                        "samples (one per latent frame), float16, so the figure can show "
                        "native resolution beside the reduced field. ~16 MB/arm/clip for "
                        "lpips and normals; negligible for dino_patch and latent.")
    p.add_argument("--normals_steps", type=int, default=4)
    p.add_argument("--normals_ensemble", type=int, default=5,
                   help=">=5 so Marigold returns an ensemble average, not one sample")
    p.add_argument("--dinov3_model_id", type=str, default=DINOV3_MODEL_ID)
    p.add_argument("--normals_model_id", type=str, default=NORMALS_MODEL_ID)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", help="Marigold in float16")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = args.device
    data_root = args.data_root.expanduser().resolve()
    target_root = args.target_root.expanduser().resolve()
    latent_dir = args.latent_dir.expanduser().resolve()
    out_dir = (args.out_root.expanduser().resolve() / (args.arm_dir_name or args.arm)
               / f"edit{args.edit_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [c for c in json.loads(args.cases.expanduser().read_text())
             if int(c["edit_type"]) == args.edit_type]
    if not cases:
        raise SystemExit(f"[r31_div] {args.cases} lists no case with "
                         f"edit_type={args.edit_type}")

    # Load ONLY the model this arm needs -- the array runs one arm per task.
    models: Dict[str, object] = {}
    if args.arm == "lpips":
        models["lpips"] = load_lpips(device)
    elif args.arm == "dino_patch":
        models["dino"], models["dino_prefix"] = load_dinov3(args.dinov3_model_id, device)
        print(f"[r31_div] DINOv3 prefix = {models['dino_prefix']} "
              f"(1 CLS + {models['dino_prefix'] - 1} registers, read from the checkpoint)",
              flush=True)
    elif args.arm == "normals":
        models["normals"] = load_normals(args.normals_model_id, device,
                                         torch.float16 if args.fp16 else torch.float32)

    print(f"[r31_div] arm={args.arm} edit_type={args.edit_type} cases={len(cases)} "
          f"temporal={args.temporal_reduce} -> {out_dir}", flush=True)

    n_ok = 0
    for c in cases:
        name = c["video_name"]
        # Marigold is stochastic; re-seed per case so a clip's map does not depend on
        # its position in the list.
        set_seed(args.seed)
        try:
            npz_path = latent_dir / f"edit{args.edit_type}" / f"{name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"stage-1 latents missing: {npz_path}")
            z = np.load(npz_path)
            z_trg, z_src = z["z_trg"], z["z_src"]
            n_latent = int(z_trg.shape[0])

            if args.arm in PIXEL_ARMS:
                n_pix = 4 * (n_latent - 1) + 1
                src_frames = load_frames(data_root / "videos" / f"{name}.mp4", n_pix)
                trg_frames = load_frames(
                    target_root / f"edit{args.edit_type}" / name, n_pix)
            else:
                src_frames = trg_frames = []

            d, d_native, diag = measure_case(args.arm, src_frames, trg_frames, z_trg,
                                             z_src, models, args, device)
            d_min, d_max = float(d.min()), float(d.max())
            span = d_max - d_min
            # Per-clip min/max. A degenerate span would divide by ~0 and amplify noise to
            # full range; emit m = 0 instead and let check-stage2 see d_min == d_max.
            m = ((d - d_min) / span).astype(np.float32) if span > 1e-12 else np.zeros_like(d)

            out = out_dir / f"{name}.npz"
            np.savez_compressed(
                out, d=d.astype(np.float32), m=m,
                d_min=np.float64(d_min), d_max=np.float64(d_max),
                grid_h=np.int32(GRID_H), grid_w=np.int32(GRID_W),
                n_latent_frames=np.int32(n_latent),
                n_pixel_frames=np.int32(len(src_frames)),
                arm=np.str_(args.arm), video_name=np.str_(name),
                edit_type=np.int32(args.edit_type),
                temporal_reduce=np.str_(args.temporal_reduce),
                seed=np.int32(args.seed),
                normals_steps=np.int32(args.normals_steps),
                normals_ensemble=np.int32(args.normals_ensemble),
                measure_index=z["measure_index"],
                #✨ R31 2026-09-15: carry stage 1's gating regime through to stage 2.
                # Stage 1 is now a 7-step run measured POST-injection, so a localised
                # map is partly mask-derived -- check-stage2 must be able to see which
                # regime produced the target without re-opening the stage-1 npz.
                measure_kind=(z["measure_kind"] if "measure_kind" in z.files
                              else np.str_("unknown__pre_2026_09_15")),
                **({} if d_native is None else {
                    "d_native": d_native,
                    "native_frame_idx": np.asarray(
                        [i for i in native_frame_indices(n_latent)
                         if i < (len(src_frames) or n_latent)], dtype=np.int32),
                    "native_hw": np.asarray(d_native.shape[1:], dtype=np.int32),
                }),
                **{f"diag_{k}": np.float64(v) if isinstance(v, float) else np.asarray(v)
                   for k, v in diag.items()},
            )
            extra = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in diag.items())
            nat = "" if d_native is None else f" native={tuple(d_native.shape)}"
            print(f"[ok] {name}: d=[{d_min:.5f}, {d_max:.5f}] "
                  f"frac(m>0.5)={float((m > 0.5).mean()):.3f}{nat} {extra}", flush=True)
            n_ok += 1
        except Exception as ex:
            print(f"[ERROR] {name}: {type(ex).__name__}: {ex}", flush=True)

    print(f"[r31_div] done arm={args.arm} edit{args.edit_type}: {n_ok}/{len(cases)} ok",
          flush=True)
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
