"""R40: draft-free divergence field -- anchor map on frame 0, propagated by optical flow.

R35 measures its divergence field on a stage-1 DRAFT render: one DINO / LPIPS map per
frame, source vs draft. R40 is the cheaper alternative a reviewer would suggest: measure
ONE map, between source frame 0 and the VP anchor (the edited first frame), and carry it
to every other frame with optical flow computed on the SOURCE video. No draft pass at all.

ORDER OF OPERATIONS (per clip) -- warp in PIXEL space, then reduce to latents
--------------------------------------------------------------------------------
  1. frame-0 map at 480x832:
       lpips_warp : LPIPS(alex, spatial=True), natively per-pixel
       dino_warp  : DINOv3 per-patch 1-cos on the 30x52 patch grid, NEAREST-upsampled
                    (each value repeated over its 16x16 patch), so step 3 pools it back
                    to exactly the native grid on frame 0
  2. warp to every pixel frame k: sample map_0 at x + C_k(x), where C_k is the composed
     backward flow k -> 0 (below) -> [F_pix, 480, 832]
  3. 16x16 area-mean onto the token grid (r31_divergence.to_token_grid) -> [F_pix, 30, 52]
  4. the VAE's temporal grouping, mean (r31_divergence.reduce_temporal) -> d [F_lat, 1560]
Steps 3-4 are R35's own stage-2 reduction, so only step 2 differs from R35.

FLOW -- chained, composed, sampled ONCE
---------------------------------------
torchvision RAFT-large (C_T_SKHT_V2), on the source frames at 480x832. Convention: for
``raft(a, b)``, ``a(x) ~ b(x + flow(x))``, channel 0 = dx, channel 1 = dy. With the
consecutive backward flow F_k = raft(src[k], src[k-1]):

    C_0 = 0
    C_k(x) = F_k(x) + C_{k-1}(x + F_k(x))        (bilinear, border padding)
    map_k(x) = map_0(x + C_k(x))

The map itself is sampled once per frame from map_0, never re-warped frame to frame, so
it does not blur over ~80 resamplings. Only the displacement field is chained.

DISOCCLUSIONS -- naive warp, diagnostics only
---------------------------------------------
No fill, no masking: border padding is all that happens at the frame edge. The forward-
backward check (Sundaram et al. 2010: |F_k + F'_k(x + F_k)|^2 > a(|F_k|^2 + |F'_k|^2) + b,
a = 0.01, b = 0.5, with F'_k = raft(src[k-1], src[k])) is computed ONLY to record how much
of each frame the warp cannot know about. It never modifies the map.

OUTPUT -- R31 stage-2 format, so r31_div_abs / r31_rho_map / r31_stage3(_grids) run
unmodified
-----------------------------------------------------------------------------------
  {out_root}/{dino_warp,lpips_warp}/edit{T}/{video_name}.npz
      d, m (per-clip min/max, overwritten by r31_div_abs), d_min, d_max, grid_h, grid_w,
      n_latent_frames, n_pixel_frames, arm (true metric name), video_name, edit_type,
      temporal_reduce, seed, measure_index = -1, measure_kind = "r40_anchor_warp"
    + R40 diagnostics, each [F_pix] with 0 at k = 0:
      occ_frac            per-step forward-backward failure, frame k vs k-1
      oob_frac            x + C_k outside the frame
      invalid_chain_frac  occluded or out-of-frame anywhere along the chain k -> 0
      photo_err_warp      mean |warp(src_0, C_k) - src_k|  (RGB in [0, 1])
      photo_err_id        mean |src_0 - src_k|              (no-warp reference)
    + flow_model, flow_iters, anchor_path

--smoke GATES (printed as ``GATE Gn PASS/FAIL``)
------------------------------------------------
  G1  frame-0 identity: row 0 of d equals the un-warped frame-0 map pooled directly
      (LPIPS) / the native DINO grid (DINO). Tests the zero-flow warp is the identity,
      i.e. the grid_sample coordinate convention, and the upsample/pool round trip.
  G2  frame count: n_latent_frames equals R35's npz for the same clip.
  G3  flow direction: photo_err_warp < photo_err_id for >= 90% of the frames that
      actually move (photo_err_id > 1e-3). A swapped flow convention fails this.
  + overlay PNGs of the warped LPIPS map on frames 0 / mid / last.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# R40 reuses R31's reductions and metric code verbatim, so the field lands on exactly
# the grid R35's stage 2 produced.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_divergence import load_lpips, set_seed, to_unit_tensor  # noqa: E402
from r31_divergence import (  # noqa: E402
    DINOV3_MODEL_ID,
    FRAME_SEQ_LENGTH,
    GRID_H,
    GRID_W,
    PIXEL_H,
    PIXEL_W,
    dinov3_patch_divergence,
    latent_frame_groups,
    load_dinov3,
    load_frames,
    reduce_temporal,
    to_token_grid,
)

# Output dir name -> the true metric name written into the npz's `arm` field.
ARMS: Dict[str, str] = {"dino_warp": "dino_patch", "lpips_warp": "lpips"}
PATCH: int = PIXEL_H // GRID_H                       # 16, == PIXEL_W // GRID_W

FLOW_MODEL: str = "torchvision.raft_large/C_T_SKHT_V2"
FB_ALPHA: float = 0.01                               # Sundaram et al. 2010
FB_BETA: float = 0.5
MEASURE_KIND: str = "r40_anchor_warp"

G1_ATOL: float = 1e-5
G3_MIN_SHARE: float = 0.90
G3_MOVE_EPS: float = 1e-3


def find_closest_num_frame(x: int, a: int = 4, b: int = 3) -> int:
    """Copied from ``inference_edit_streamedit.py`` (importing it pulls in the pipeline).

    The pipeline truncates every source video to this length, so the field must cover
    exactly these frames -- r31_stage3.py computes n_latent the same way.
    """
    max_m = (x + a - 1) // (a * b)
    while max_m > 0:
        y = a * b * max_m - a + 1
        if y <= x:
            return y
        max_m -= 1
    raise ValueError(f"R40: a {x}-frame video is too short for the pipeline")


# --------------------------------------------------------------------------------------
# warping
# --------------------------------------------------------------------------------------
def pixel_grid(device: str) -> torch.Tensor:
    """``[1, 2, H, W]`` absolute pixel coordinates, channel 0 = x, channel 1 = y."""
    ys, xs = torch.meshgrid(torch.arange(PIXEL_H, device=device, dtype=torch.float32),
                            torch.arange(PIXEL_W, device=device, dtype=torch.float32),
                            indexing="ij")
    return torch.stack([xs, ys], dim=0).unsqueeze(0)


def sample_at(field: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of ``field [B, C, H, W]`` at absolute pixel positions ``pos``.

    ``align_corners=True`` maps -1 / +1 onto the centres of the first / last pixel, which
    is what makes ``pos = pixel_grid`` the exact identity (gate G1 tests this).
    """
    gx = 2.0 * pos[:, 0] / (PIXEL_W - 1) - 1.0
    gy = 2.0 * pos[:, 1] / (PIXEL_H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


def outside(pos: torch.Tensor) -> torch.Tensor:
    """``[B, H, W]`` bool: position falls outside the frame."""
    return ((pos[:, 0] < 0) | (pos[:, 0] > PIXEL_W - 1)
            | (pos[:, 1] < 0) | (pos[:, 1] > PIXEL_H - 1))


def load_raft(device: str):
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).to(device).eval()
    return model, weights.transforms()


@torch.no_grad()
def consecutive_flows(raft, tf, frames: torch.Tensor, iters: int, batch: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``frames [N, 3, H, W]`` in [0, 1] -> (backward ``F_k = raft(f_k, f_{k-1})``,
    forward ``F'_k = raft(f_{k-1}, f_k)``), each ``[N-1, 2, H, W]``, entry k-1 <-> pair k.
    """
    later, earlier = frames[1:], frames[:-1]
    bwd, fwd = [], []
    for i in range(0, later.shape[0], batch):
        a, b = later[i:i + batch], earlier[i:i + batch]
        # Both directions in one forward: a pair and its reverse share the batch.
        x1, x2 = tf(torch.cat([a, b]), torch.cat([b, a]))
        flow = raft(x1, x2, num_flow_updates=iters)[-1]
        bwd.append(flow[: a.shape[0]])
        fwd.append(flow[a.shape[0]:])
    return torch.cat(bwd), torch.cat(fwd)


# --------------------------------------------------------------------------------------
# per-clip measurement
# --------------------------------------------------------------------------------------
@torch.no_grad()
def frame0_maps(src0: Image.Image, anchor: Image.Image, models: Dict[str, object],
                device: str) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """Both arms' frame-0 maps as ``[1, 1, 480, 832]`` float32, plus DINO's native grid."""
    a = to_unit_tensor(src0, (PIXEL_H, PIXEL_W), device) * 2.0 - 1.0
    b = to_unit_tensor(anchor, (PIXEL_H, PIXEL_W), device) * 2.0 - 1.0
    lp = models["lpips"](a, b).float()                                   # [1, 1, H, W]
    dino_grid = dinov3_patch_divergence(models["dino"], models["dino_prefix"],
                                        src0, anchor, device)            # [30, 52] f64
    dino_px = (torch.from_numpy(dino_grid).float().to(device)
               .repeat_interleave(PATCH, dim=0).repeat_interleave(PATCH, dim=1))
    return {"lpips_warp": lp, "dino_warp": dino_px[None, None]}, dino_grid


@torch.no_grad()
def measure_case(src_frames: List[Image.Image], anchor: Image.Image,
                 models: Dict[str, object], args: argparse.Namespace, device: str,
                 keep_px: Sequence[int] = ()) -> Dict[str, object]:
    n_pix = len(src_frames)
    n_lat = (n_pix - 1) // 4 + 1
    groups = latent_frame_groups(n_pix, n_lat)

    frames = torch.cat([to_unit_tensor(f, (PIXEL_H, PIXEL_W), device)
                        for f in src_frames])                            # [N, 3, H, W]
    maps0, dino_grid = frame0_maps(src_frames[0], anchor, models, device)
    fb_all, ff_all = consecutive_flows(models["raft"], models["raft_tf"], frames,
                                       args.flow_iters, args.flow_batch)

    X = pixel_grid(device)
    C = torch.zeros_like(X)
    invalid = torch.zeros((1, PIXEL_H, PIXEL_W), dtype=torch.bool, device=device)
    tokens: Dict[str, List[torch.Tensor]] = {k: [] for k in ARMS}
    kept: Dict[int, torch.Tensor] = {}
    diag = {k: np.zeros(n_pix, dtype=np.float64) for k in
            ("occ_frac", "oob_frac", "invalid_chain_frac", "photo_err_warp",
             "photo_err_id")}

    for k in range(n_pix):
        if k > 0:
            fb, ff = fb_all[k - 1:k], ff_all[k - 1:k]
            y = X + fb                                   # where x of frame k sits in k-1
            ff_at = sample_at(ff, y)
            occ = ((fb + ff_at).pow(2).sum(1)
                   > FB_ALPHA * (fb.pow(2).sum(1) + ff_at.pow(2).sum(1)) + FB_BETA)
            C = fb + sample_at(C, y)
            invalid = (occ | outside(y)
                       | (sample_at(invalid.float()[:, None], y)[:, 0] > 0.5))
            diag["occ_frac"][k] = occ.float().mean().item()
            diag["invalid_chain_frac"][k] = invalid.float().mean().item()
        pos0 = X + C
        if k > 0:
            diag["oob_frac"][k] = outside(pos0).float().mean().item()
            warped_rgb = sample_at(frames[:1], pos0)
            diag["photo_err_warp"][k] = (warped_rgb - frames[k:k + 1]).abs().mean().item()
            diag["photo_err_id"][k] = (frames[:1] - frames[k:k + 1]).abs().mean().item()
        # Frame 0 goes through the same warp (C = 0) on purpose: G1 then tests the
        # identity of the sampler, not a special case.
        for arm, m0 in maps0.items():
            w = sample_at(m0, pos0)[:, 0]                                # [1, H, W]
            tokens[arm].append(to_token_grid(w)[0].cpu())                # [30, 52]
            if arm == "lpips_warp" and k in keep_px:
                kept[k] = w[0].cpu()

    d = {arm: reduce_temporal(torch.stack(tokens[arm]), groups, args.temporal_reduce)
         .reshape(n_lat, -1).float().numpy() for arm in ARMS}
    direct = {"lpips_warp": to_token_grid(maps0["lpips_warp"][:, 0])[0].cpu()
              .reshape(-1).numpy(),
              "dino_warp": dino_grid.reshape(-1)}
    return {"d": d, "direct0": direct, "diag": diag, "n_pix": n_pix, "n_lat": n_lat,
            "kept_px": kept}


def save_npz(path: Path, arm: str, d: np.ndarray, res: Dict[str, object], name: str,
             edit_type: int, anchor_path: Path, args: argparse.Namespace) -> None:
    d_min, d_max = float(d.min()), float(d.max())
    span = d_max - d_min
    m = ((d - d_min) / span).astype(np.float32) if span > 1e-12 else np.zeros_like(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, d=d.astype(np.float32), m=m,
        d_min=np.float64(d_min), d_max=np.float64(d_max),
        grid_h=np.int32(GRID_H), grid_w=np.int32(GRID_W),
        n_latent_frames=np.int32(res["n_lat"]), n_pixel_frames=np.int32(res["n_pix"]),
        arm=np.str_(ARMS[arm]), video_name=np.str_(name), edit_type=np.int32(edit_type),
        temporal_reduce=np.str_(args.temporal_reduce), seed=np.int32(args.seed),
        measure_index=np.int32(-1), measure_kind=np.str_(MEASURE_KIND),
        flow_model=np.str_(FLOW_MODEL), flow_iters=np.int32(args.flow_iters),
        anchor_path=np.str_(str(anchor_path)),
        **{k: v.astype(np.float32) for k, v in res["diag"].items()},
    )


# --------------------------------------------------------------------------------------
# smoke gates
# --------------------------------------------------------------------------------------
def smoke_gates(res: Dict[str, object], name: str, edit_type: int,
                args: argparse.Namespace) -> List[bool]:
    ok: List[bool] = []

    errs = {arm: float(np.abs(res["d"][arm][0] - res["direct0"][arm]).max())
            for arm in ARMS}
    g1 = all(e <= G1_ATOL for e in errs.values())
    print(f"GATE G1 {'PASS' if g1 else 'FAIL'} {name}: frame-0 max|d[0] - direct| "
          + " ".join(f"{a}={e:.2e}" for a, e in errs.items()) + f" (atol {G1_ATOL:g})",
          flush=True)
    ok.append(g1)

    ref = (args.r35_div_root.expanduser() / "dino_first2" / f"edit{edit_type}"
           / f"{name}.npz")
    if ref.exists():
        n_ref = int(np.load(ref)["n_latent_frames"])
        g2 = n_ref == res["n_lat"]
        print(f"GATE G2 {'PASS' if g2 else 'FAIL'} {name}: n_latent {res['n_lat']} "
              f"vs R35 {n_ref}", flush=True)
    else:
        g2 = False
        print(f"GATE G2 FAIL {name}: R35 reference missing: {ref}", flush=True)
    ok.append(g2)

    dg = res["diag"]
    moving = dg["photo_err_id"][1:] > G3_MOVE_EPS
    better = (dg["photo_err_warp"][1:] < dg["photo_err_id"][1:])[moving]
    share = float(better.mean()) if better.size else 1.0
    g3 = share >= G3_MIN_SHARE
    print(f"GATE G3 {'PASS' if g3 else 'FAIL'} {name}: warp beats no-warp on "
          f"{share:.1%} of {int(moving.sum())} moving frames (need {G3_MIN_SHARE:.0%}); "
          f"mean err warp={dg['photo_err_warp'][1:].mean():.4f} "
          f"id={dg['photo_err_id'][1:].mean():.4f}", flush=True)
    ok.append(g3)
    return ok


def smoke_png(res: Dict[str, object], src_frames: List[Image.Image], anchor: Image.Image,
              out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(res["kept_px"])
    vmax = max(float(res["kept_px"][k].max()) for k in ks) or 1.0
    fig, axes = plt.subplots(1, 1 + len(ks), figsize=(3.2 * (1 + len(ks)), 2.2))
    axes[0].imshow(anchor.resize((PIXEL_W, PIXEL_H)))
    axes[0].set_title("anchor", fontsize=8)
    for ax, k in zip(axes[1:], ks):
        ax.imshow(src_frames[k].resize((PIXEL_W, PIXEL_H)))
        ax.imshow(res["kept_px"][k].numpy(), cmap="magma", alpha=0.55, vmin=0, vmax=vmax)
        ax.set_title(f"frame {k}: warped LPIPS map  (invalid "
                     f"{res['diag']['invalid_chain_frac'][k]:.0%})", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edit_type", type=int, default=None,
                   help="FiVE-Bench edit type 1..6 (full mode; all its manifest cases)")
    p.add_argument("--smoke", action="store_true",
                   help="Run only --case_ids and print the G1-G3 gates")
    p.add_argument("--case_ids", nargs="+", default=["edit1/0001_bus", "edit2/0034_cows"],
                   help="Smoke cases, as the manifest's case_id")
    p.add_argument("--cases", type=Path, default=Path("evaluation/r7_anchor_manifest.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--anchor_root", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/anchors"),
                   help="Same root r35_stage3.sh passes as --first_frame_edit_dir")
    p.add_argument("-o", "--out_root", type=Path, required=True)
    p.add_argument("--png_dir", type=Path, default=Path("evaluation/figures/r40_smoke"))
    p.add_argument("--r35_div_root", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/r35_div"),
                   help="Gate G2's frame-count reference")
    p.add_argument("--temporal_reduce", choices=("mean", "max"), default="mean")
    p.add_argument("--flow_iters", type=int, default=12, help="RAFT num_flow_updates")
    p.add_argument("--flow_batch", type=int, default=8,
                   help="Frame pairs per RAFT forward (x2: both directions)")
    p.add_argument("--dinov3_model_id", type=str, default=DINOV3_MODEL_ID)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.smoke and args.edit_type is None:
        p.error("--edit_type is required unless --smoke")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = args.device
    data_root = args.data_root.expanduser().resolve()
    anchor_root = args.anchor_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    manifest = json.loads(args.cases.expanduser().read_text())
    if args.smoke:
        by_id = {c["case_id"]: c for c in manifest}
        missing = [c for c in args.case_ids if c not in by_id]
        if missing:
            raise SystemExit(f"[r40_warp] case_ids not in {args.cases}: {missing}")
        cases = [by_id[c] for c in args.case_ids]
    else:
        cases = [c for c in manifest if int(c["edit_type"]) == args.edit_type]
        if not cases:
            raise SystemExit(f"[r40_warp] {args.cases} lists no case with "
                             f"edit_type={args.edit_type}")

    import torchvision
    print(f"[r40_warp] python={platform.python_version()} torch={torch.__version__} "
          f"torchvision={torchvision.__version__} device={device}", flush=True)
    print(f"[r40_warp] {vars(args)}", flush=True)

    set_seed(args.seed)
    models: Dict[str, object] = {"lpips": load_lpips(device)}
    models["dino"], models["dino_prefix"] = load_dinov3(args.dinov3_model_id, device)
    models["raft"], models["raft_tf"] = load_raft(device)
    print(f"[r40_warp] models loaded: LPIPS alex, DINOv3 (prefix {models['dino_prefix']}), "
          f"{FLOW_MODEL} x{args.flow_iters} | cases={len(cases)} -> {out_root}", flush=True)

    n_ok, gates = 0, []
    for c in cases:
        name, T = c["video_name"], int(c["edit_type"])
        set_seed(args.seed)
        try:
            frames = load_frames(data_root / "videos" / f"{name}.mp4")
            n_pix = find_closest_num_frame(len(frames))
            frames = frames[:n_pix]
            anchor_path = anchor_root / f"edit{T}" / f"{name}.png"
            if not anchor_path.exists():
                raise FileNotFoundError(f"anchor missing: {anchor_path}")
            anchor = Image.open(anchor_path).convert("RGB")

            keep = (0, n_pix // 2, n_pix - 1) if args.smoke else ()
            res = measure_case(frames, anchor, models, args, device, keep_px=keep)
            for arm in ARMS:
                save_npz(out_root / arm / f"edit{T}" / f"{name}.npz", arm, res["d"][arm],
                         res, name, T, anchor_path, args)

            dg = res["diag"]
            print(f"[ok] edit{T}/{name}: n_pix={n_pix} n_lat={res['n_lat']} "
                  + " ".join(f"{a}=[{res['d'][a].min():.4f},{res['d'][a].max():.4f}]"
                             for a in ARMS)
                  + f" occ_mean={dg['occ_frac'][1:].mean():.3f}"
                  f" invalid_last={dg['invalid_chain_frac'][-1]:.3f}"
                  f" oob_last={dg['oob_frac'][-1]:.3f}", flush=True)
            if args.smoke:
                gates += smoke_gates(res, name, T, args)
                smoke_png(res, frames, anchor, args.png_dir / f"edit{T}_{name}.png",
                          f"R40 smoke -- edit{T}/{name}")
            n_ok += 1
        except Exception as ex:
            print(f"[ERROR] edit{T}/{name}: {type(ex).__name__}: {ex}", flush=True)

    print(f"[r40_warp] done: {n_ok}/{len(cases)} ok", flush=True)
    if args.smoke:
        all_pass = bool(gates) and all(gates) and n_ok == len(cases)
        print(f"[r40_warp] GATES {'ALL PASS' if all_pass else 'FAILED'} "
              f"({sum(gates)}/{len(gates)})", flush=True)
        return 0 if all_pass else 1
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
