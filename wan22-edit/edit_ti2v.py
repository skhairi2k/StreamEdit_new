# Wan-Edit (FiVE-Bench's FlowEdit on Wan2.1-T2V-1.3B) ported to Wan2.2-TI2V-5B.
#
# The editing loop is a line-by-line copy of wan-edit/wan/text2video.py::WanT2V.edit,
# including its quirk of noising the source latent to t_{i-1} while the model is told
# t_i, so that only the backbone changes. Deliberate deviations, all for running on a
# single GPU in a 20 GB-RAM session or for reproducibility:
#   1. The DiT is loaded in bf16 (= Wan2.2's official --convert_model_dtype).
#   2. The per-step FlowEdit noise is drawn from a seeded generator (Wan-Edit uses the
#      unseeded global RNG, so its runs are not reproducible whatever --base_seed is).
#   3. T5 encodes the three prompts once and is then freed; the DiT stays on the GPU.
import argparse
import logging
import math
import os
import sys

import cv2
import torch
from tqdm import tqdm

import types  # noqa: E402

# wan/__init__.py imports every Wan2.2 pipeline (S2V needs decord, Animate more), none of
# which this script uses. Register `wan` as a bare namespace so only the submodules below
# are executed, instead of adding those dependencies to the env.
_WAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan")
_wan_pkg = types.ModuleType("wan")
_wan_pkg.__path__ = [_WAN_DIR]
sys.modules["wan"] = _wan_pkg

from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402
from wan.modules.t5 import T5EncoderModel  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from wan.utils.utils import masks_like, save_video  # noqa: E402


def load_frames(video_path: str, num_frames: int, target_size: tuple[int, int]) -> torch.Tensor:
    """Same reader as wan-edit/edit.py::load_frames: first num_frames frames, resized, in [-1, 1].

    Returns a [3, F, H, W] float tensor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2RGB)
        frames.append(torch.tensor(frame).permute(2, 0, 1).float() / 127.5 - 1.0)
    if not frames:
        raise ValueError(f"Video {video_path} has no readable frames")
    return torch.stack(frames).permute(1, 0, 2, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Wan-Edit (FlowEdit) on Wan2.2-TI2V-5B")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--video_path", required=True)
    p.add_argument("--save_file", required=True)
    p.add_argument("--prompt", required=True, help="source prompt")
    p.add_argument("--tgt_prompt", required=True)
    p.add_argument("--size", default="1280*704", help="W*H; TI2V-5B supports 1280*704 / 704*1280")
    p.add_argument("--frame_num", type=int, default=41)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0, help="source CFG")
    p.add_argument("--tgt_guide_scale", type=float, default=10.0, help="target CFG")
    p.add_argument("--skip_timesteps", type=int, default=15)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logging.info(f"args: {vars(args)}")
    cfg = WAN_CONFIGS["ti2v-5B"]
    device = torch.device("cuda:0")
    W, H = (int(x) for x in args.size.split("*"))

    # ---- text: encode once on GPU, then free T5 (deviation 3) ----
    t5 = T5EncoderModel(
        text_len=cfg.text_len, dtype=cfg.t5_dtype, device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer))
    t5.model.to(device)
    context_src = t5([args.prompt], device)
    context_tgt = t5([args.tgt_prompt], device)
    context_null = t5([cfg.sample_neg_prompt], device)
    del t5
    torch.cuda.empty_cache()

    # ---- VAE + DiT (bf16, deviation 1) ----
    vae = Wan2_2_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=device)
    logging.info(f"Creating WanModel (bf16) from {args.ckpt_dir}")
    model = WanModel.from_pretrained(args.ckpt_dir, torch_dtype=cfg.param_dtype)
    model.eval().requires_grad_(False).to(device)

    # ---- source latents ----
    video = load_frames(args.video_path, args.frame_num, (W, H)).to(device)
    F = video.shape[1]
    latents = vae.encode([video])
    target_shape = (vae.model.z_dim, (F - 1) // cfg.vae_stride[0] + 1,
                    H // cfg.vae_stride[1], W // cfg.vae_stride[2])
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (cfg.patch_size[1] * cfg.patch_size[2]) * target_shape[1])
    logging.info(f"frames {F}, latent {tuple(latents[0].shape)}, seq_len {seq_len}")

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.base_seed)

    with torch.amp.autocast("cuda", dtype=cfg.param_dtype):
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(args.sample_steps, device=device, shift=args.sample_shift)
        timesteps = sample_scheduler.timesteps
        logging.info(f"edit starts at step {args.skip_timesteps}, "
                     f"sigma = {timesteps[args.skip_timesteps].item() / 1000:.3f}")

        arg_src = {"context": context_src, "seq_len": seq_len}
        arg_tgt = {"context": context_tgt, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}
        start_latents = latents
        mv_latent = latents
        # TI2V-5B takes a per-token timestep; in text-only mode it is uniform (mask of ones).
        _, mask2 = masks_like([latents[0]], zero=False)

        for i, t in enumerate(tqdm(timesteps)):
            # drawn every step, skipped or not, exactly as Wan-Edit -- but seeded (deviation 2)
            noise = [torch.randn(*target_shape, dtype=torch.float32, device=device,
                                 generator=seed_g)]
            if i < args.skip_timesteps:
                continue
            ts = torch.stack([t])
            temp_ts = (mask2[0][0][:, ::2, ::2] * ts).flatten()
            temp_ts = torch.cat([temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * ts])
            timestep = temp_ts.unsqueeze(0)

            # Wan-Edit's FlowEdit step, unchanged (incl. t_prev for the source noise level)
            t_prev = 1000 if i == 0 else timesteps[i - 1]
            src_latent = [t_prev / 1000.0 * noise[0] + (1000 - t_prev) / 1000.0 * start_latents[0]]
            tgt_latent = [mv_latent[0] + src_latent[0] - start_latents[0]]
            cond_src = model(src_latent, t=timestep, **arg_src)[0]
            cond_tgt = model(tgt_latent, t=timestep, **arg_tgt)[0]
            uncond_src = model(src_latent, t=timestep, **arg_null)[0]
            uncond_tgt = model(tgt_latent, t=timestep, **arg_null)[0]
            v_src = uncond_src + args.sample_guide_scale * (cond_src - uncond_src)
            v_tgt = uncond_tgt + args.tgt_guide_scale * (cond_tgt - uncond_tgt)
            temp_x0 = sample_scheduler.step(
                (v_tgt - v_src).unsqueeze(0), t, mv_latent[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            mv_latent = [temp_x0.squeeze(0)]

        out = vae.decode(mv_latent)[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.save_file)), exist_ok=True)
    save_video(tensor=out[None], save_file=args.save_file, fps=args.fps,
               nrow=1, normalize=True, value_range=(-1, 1))
    logging.info(f"saved {args.save_file}; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
