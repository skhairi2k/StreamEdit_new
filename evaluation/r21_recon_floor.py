"""Reconstruction-floor ladder: where is our LPIPS actually spent?

The real edit scores LPIPS 213.5 vs the jpg reference and 52.0 vs the mp4 the model
was fed, against StreamGVE Table 1's 49.84. This decomposes that into stages, so we
can tell a pipeline defect from an unreachable floor:

  L0  resize      864 -> 832 -> 864, no model at all      (already measured: 0.008)
  L1  VAE         encode -> decode, no denoising          <- this script
  L2  degenerate  full pipeline, src prompt == trg prompt <- this script
  L3  real edit   the actual arm                          (already measured)

L2 uses an identical source/target prompt AND trigger word, so the target branch has
nothing to change and its output is a reconstruction through the whole sampling +
self-attention-bridge path. That avoids modifying pipeline internals to expose the
source branch -- no experiment logic is touched, only the prompts.

Reading it: if L1 ~ L3, the loss is the VAE and no editing change recovers it (and
49.84 is unreachable for this backbone as configured). If L1 is small but L2 ~ L3,
the detail is destroyed in the editing path -- a findable bug, and the R12/R13 lever.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    p.add_argument("--out_root", type=str, default="/projects/dataggen/outputs/five_bench/r21_recon_floor")
    p.add_argument("--sf_root", type=str, default="Self-Forcing_StreamEdit")
    p.add_argument("--edit_type", type=int, default=1)
    p.add_argument("--videos", type=str, nargs="+",
                   default=["0001_bus", "0002_girl-dog", "0004_car-roundabout"])
    p.add_argument("--step", type=int, default=15)
    p.add_argument("--fg_boost_factor", type=float, default=4.0)
    p.add_argument("--blend_power", type=float, default=2.0)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--rollout_chunk_size", type=int, default=21)
    p.add_argument("--rollout_overlap_block_num", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src_frames", type=str, default="video", choices=["video", "images"],
                   help="Which rendition to feed. 'video' = videos/*.mp4 (the historical "
                        "default every render used); 'images' = images/*.jpg, the same frames "
                        "evaluate.py scores against.")
    p.add_argument("--tag", type=str, default="",
                   help="Suffix on the output dirs so the two --src_frames ladders coexist.")
    p.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    p.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    p.add_argument("--use_ema", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    sf = (repo / args.sf_root).resolve()
    sys.path.insert(0, str(sf))
    import os
    os.chdir(sf)

    from torchvision import transforms
    from diffusers.utils import load_video
    # load_pipe / find_closest_num_frame live in the SF build (importable only after
    # the chdir above); save_frames_png is inlined below to avoid importing the runner.
    from inference_edit_streamedit import load_pipe, find_closest_num_frame
    from utils.misc import set_seed

    def save_frames_png(frames, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = (frames.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
        return arr.shape[0]

    pipeline, low_memory, device, _ = load_pipe(args)
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    entries = {e["video_name"]: e for e in
               json.loads((data_root / "edit_prompt" / f"edit{args.edit_type}_FiVE.json").read_text())}

    for name in args.videos:
        e = entries[name]
        set_seed(args.seed)
        if args.src_frames == "images":
            fp = sorted(q for q in (data_root / "images" / name).iterdir()
                        if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
            src_video = [Image.open(q).convert("RGB") for q in fp]
        else:
            src_video = load_video(str(data_root / "videos" / f"{name}.mp4"))
        src_video = src_video[: find_closest_num_frame(len(src_video))]
        src_tensor = torch.stack([transform(im) for im in src_video], dim=1).unsqueeze(0)
        latents = pipeline.vae.encode_to_latent(
            src_tensor.to(device=device, dtype=torch.bfloat16)).to(device=device, dtype=torch.bfloat16)
        pipeline.vae.model.clear_cache()

        # ---- L1: VAE round-trip only, no denoising ----
        vae_px = pipeline.vae.decode_to_pixel(latents, use_cache=False)
        vae_px = (vae_px * 0.5 + 0.5).clamp(0, 1)
        n = save_frames_png(vae_px[0], out_root / f"L1_vae{args.tag}" / f"edit{args.edit_type}" / name)
        pipeline.vae.model.clear_cache()
        print(f"[recon] {name}: L1_vae{args.tag} {n} frames ({args.src_frames} input)", flush=True)

        # ---- L2: full pipeline, degenerate edit (identical prompt AND trigger) ----
        set_seed(args.seed)
        out = pipeline.rollout_inference(
            src_video=latents,
            src_prompts=e["source_prompt"], trg_prompts=e["source_prompt"],
            src_trigger_words=e["source_object"], trg_trigger_words=e["source_object"],
            return_latents=False, wo_video_decode=False, profile=False, low_memory=low_memory,
            independent_first_frame=False, src_initial_latent=None, trg_initial_latent=None,
            fg_boost_factor=args.fg_boost_factor, blend_power=args.blend_power,
            rollout_chunk_size=args.rollout_chunk_size,
            rollout_overlap_block_num=args.rollout_overlap_block_num,
            vp_latent=None, blend_sched=None,          # Eq. 4, no anchor -- the R1 default
        )
        pipeline.vae.model.clear_cache()
        n = save_frames_png(out[0], out_root / f"L2_degenerate{args.tag}" / f"edit{args.edit_type}" / name)
        print(f"[recon] {name}: L2_degenerate{args.tag} {n} frames ({args.src_frames} input)", flush=True)

    print("[recon] done", flush=True)


if __name__ == "__main__":
    main()
