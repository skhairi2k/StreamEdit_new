"""R31: prove that `rho_frames=None` leaves every pre-existing render path bit-identical.

The `continuous-rho-field` patch touched the live render path in four places -- it replaced
`r26_active` with `spatial_active` at `edit_causal_inference.py` :621 / :704 / :799 / :829,
rerouted `_union_rows` through a `spatial_field` alias, and added a branch inside
`_chunk_rho`. The plan requires that passing `rho_frames=None` reproduce the previous
behaviour EXACTLY, so that R26's and R30's stored arms stay valid references.

That has been argued from the code (with `rho_frames=None` and `union_frames=None`,
`spatial_active` is False and `_chunk_rho` is never called) but never MEASURED. This script
is the measurement: render one clip on the PLAIN Eq. 4 scalar path -- no `rho_frames`, no
`union_mask_path` -- and write the final latent. Run it once against the patched tree and
once against a pre-patch copy, then compare the two latents bit-for-bit.

Anything other than an exact match means the patch perturbed a path it was supposed to
leave alone, and every stage-3 number would be measured against a moved baseline.

Usage::

    python evaluation/r31_bitparity.py --sf_root <tree> --edit_type 1 \\
        --video_name 0001_bus --out /path/latent.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_DEFAULT_SF_ROOT = Path(__file__).resolve().parent.parent / "Self-Forcing_StreamEdit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT),
                   help="Which Self-Forcing tree to render with. Point this at a "
                        "pre-patch copy for the B side of the comparison.")
    p.add_argument("--edit_type", type=int, default=1)
    p.add_argument("--video_name", type=str, default="0001_bus")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--data_root", type=str,
                   default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    p.add_argument("--first_frame_edit_dir", type=str,
                   default="/projects/dataggen/outputs/five_bench/anchors")
    p.add_argument("--vp_mode", type=str, default="vp")
    # Held at R26/R30's settings so the render exercises the real configuration.
    p.add_argument("--step", type=int, default=15)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fg_boost_factor", type=float, default=4.0)
    p.add_argument("--blend_power", type=float, default=2.0)
    p.add_argument("--rollout_chunk_size", type=int, default=21)
    p.add_argument("--rollout_overlap_block_num", type=int, default=1)
    p.add_argument("--sink_size", type=int, default=0)
    p.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    p.add_argument("--checkpoint_path", type=str,
                   default="checkpoints/self_forcing_dmd.pt")
    p.add_argument("--use_ema", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    anchors = Path(args.first_frame_edit_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame, read_json
    from utils.misc import set_seed
    from diffusers.utils import load_video

    entries = read_json(data_root / "edit_prompt" / f"edit{args.edit_type}_FiVE.json")
    e = next(x for x in entries if x["video_name"] == args.video_name)

    pipeline, low_memory, device, local_rank = load_pipe(args)
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    set_seed(args.seed)
    src_video = load_video(str(data_root / "videos" / f"{args.video_name}.mp4"))
    src_video = src_video[:find_closest_num_frame(len(src_video))]
    src_tensor = torch.stack([transform(i) for i in src_video], dim=1).unsqueeze(0)
    video_latents = pipeline.vae.encode_to_latent(
        src_tensor.to(device=device, dtype=torch.bfloat16)
    ).to(device=device, dtype=torch.bfloat16)
    pipeline.vae.model.clear_cache()

    trg_first = Image.open(anchors / f"edit{args.edit_type}"
                           / f"{args.video_name}.png").convert("RGB")
    # Encode order and the single trailing clear_cache are load bearing (VAE temporal-conv
    # state carries between calls) -- identical to run_fivebench.py / r31_stage1.py.
    src_first_latent = pipeline.vae.encode_to_latent(
        transform(src_video[0]).unsqueeze(0).unsqueeze(2).to(device=device,
                                                             dtype=torch.bfloat16)
    ).to(device=device, dtype=torch.bfloat16)
    trg_first_latent = pipeline.vae.encode_to_latent(
        transform(trg_first).unsqueeze(0).unsqueeze(2).to(device=device,
                                                          dtype=torch.bfloat16)
    ).to(device=device, dtype=torch.bfloat16)
    pipeline.vae.model.clear_cache()

    # THE POINT: no rho_frames, no union_mask_path, no tau_bg/tau_fg, no blend_sched.
    # This is the plain Eq. 4 scalar path every pre-R31 render used.
    _, final_latent = pipeline.rollout_inference(
        src_video=video_latents,
        src_prompts=e["source_prompt"],
        trg_prompts=e["target_prompt"],
        src_trigger_words=e["source_object"],
        trg_trigger_words=e["target_object"],
        return_latents=True,
        wo_video_decode=True,
        profile=False,
        low_memory=low_memory,
        independent_first_frame=True,
        src_initial_latent=src_first_latent,
        trg_initial_latent=trg_first_latent,
        fg_boost_factor=args.fg_boost_factor,
        blend_power=args.blend_power,
        rollout_chunk_size=args.rollout_chunk_size,
        rollout_overlap_block_num=args.rollout_overlap_block_num,
        vp_latent=None,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    arr = final_latent.float().cpu().numpy()
    np.savez(out, latent=arr, sf_root=np.str_(str(sf_root)),
             video_name=np.str_(args.video_name), edit_type=np.int32(args.edit_type),
             step=np.int32(args.step), seed=np.int32(args.seed))
    print(f"[r31_bitparity] sf_root={sf_root}")
    print(f"[r31_bitparity] latent {arr.shape} {arr.dtype} "
          f"sum={arr.sum():.8f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
