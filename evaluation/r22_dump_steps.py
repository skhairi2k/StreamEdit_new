"""R22: dump the per-denoising-step videos for a FiVE-Bench edit type.

Renders each pair exactly as ``run_fivebench.py`` would -- same per-pair seeding, same
transform, same VAE encode ordering -- but passes ``step_dump`` into the pipeline so the
target branch's x0 prediction is captured at **every denoising step of every block**, then
decodes and saves all ``S`` resulting videos instead of only the final one.

What ``V_j`` means here
----------------------
The pipeline's 15-step loop is nested *inside* a per-block loop, and ``output`` is written
once per block after that block's 15 steps finish. There is therefore no instant at which
the whole video sits at step ``j``. ``V_j`` is the assembly of **each block's own step-j x0
prediction**, collected during one unperturbed rollout: blocks still commit their step-14
value to the KV cache, so generation is untouched and ``V_14`` is the real output.

``V_j`` for ``j < 14`` is a *trajectory preview*, not what a j-step sampler would emit --
block ``b``'s step-j latent was produced while attending to blocks ``< b`` that were already
fully denoised and re-cached at clean context (Step 3.3). Read the matrix accordingly.

The bottom-right identity
-------------------------
``step_dump[-1]`` is asserted equal to the returned latent inside the pipeline (per window
and again after stitching), and again here at the driver level. That is the *latent* gate;
the substantive one is pixel-level, checked by ``r22_smoke.sh`` comparing the step-14 PNGs
against the arm's stored full-bench reference render.

Outputs zero-padded PNG frames to
``{out_root}/{method}/step{jj}/edit{T}/{video_name}/`` -- the ``step{jj}`` level sits above
``edit{T}`` on purpose, so each step directory is a valid ``--tgt_layout edit_video`` root
for ``evaluation/fivebench/evaluate.py``.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# Default location of the Self-Forcing StreamEdit build, relative to the repo root.
_DEFAULT_SF_ROOT = Path(__file__).resolve().parent.parent / "Self-Forcing_StreamEdit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # what to run
    parser.add_argument("--edit_type", type=int, required=True,
                        help="FiVE-Bench edit type 1..6 (reads edit_prompt/edit{T}_FiVE.json)")
    parser.add_argument("--method", type=str, required=True,
                        help="Arm name; frames go to {out_root}/{method}/step{jj}/edit{T}/")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark",
                        help="FiVE-Bench root (videos/, edit_prompt/)")
    parser.add_argument("--out_root", type=str,
                        default="/projects/dataggen/outputs/five_bench/r22_step_dump",
                        help="Root for produced frames (method subdir created underneath)")

    # anchor injection -- same flags and semantics as run_fivebench.py
    parser.add_argument("--first_frame_edit_dir", type=str, default=None,
                        help="Root of Qwen-edited anchor PNGs ({dir}/edit{T}/{video}.png)")
    parser.add_argument("--vp_mode", type=str, default="vp", choices=["novp", "vp", "pvp"],
                        help="Anchor injection mechanism (ignored without "
                             "--first_frame_edit_dir, which forces novp)")
    parser.add_argument("--blend_sched", type=str, default=None,
                        choices=["paper", "cos_full", "cos_half", "cos_third", "const", "zero"],
                        help="Blender-rate schedule; unset and 'paper' both mean Eq. 4")
    parser.add_argument("--cases_json", type=str, default=None,
                        help="Restrict to the video_names in this cases.json whose "
                             "edit_type matches --edit_type")

    # grounding / boosting hyper-parameters (paper Self-Forcing defaults)
    parser.add_argument("--fg_boost_factor", type=float, default=4.0, help="omega")
    parser.add_argument("--blend_power", type=float, default=2.0, help="rho")

    # rollout sampling -- 21 is what makes these renders comparable to the stored arms
    parser.add_argument("--rollout_chunk_size", type=int, default=21)
    parser.add_argument("--rollout_overlap_block_num", type=int, default=1)

    # sampling / model settings (mirrors run_fivebench.py)
    parser.add_argument("--step", type=int, default=15, help="Sampler steps => matrix columns")
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT))
    return parser.parse_args()


def save_frames_png(frames: torch.Tensor, out_dir: Path) -> int:
    """Save a ``[T, C, H, W]`` float tensor in [0, 1] as zero-padded PNG frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (frames.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
    return arr.shape[0]


def main() -> None:
    args = parse_args()

    # Resolve external paths to absolute BEFORE chdir-ing into the SF build (see
    # run_fivebench.py -- the build reads configs/checkpoints via relative paths).
    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    first_frame_edit_dir = (Path(args.first_frame_edit_dir).expanduser().resolve()
                            if args.first_frame_edit_dir else None)
    cases_json = (Path(args.cases_json).expanduser().resolve()
                  if args.cases_json else None)

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame, read_json
    from utils.misc import set_seed
    from diffusers.utils import load_video

    pipeline, low_memory, device, local_rank = load_pipe(args)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    edit_json = data_root / "edit_prompt" / f"edit{args.edit_type}_FiVE.json"
    entries = read_json(edit_json)

    # Filter on video_name AND edit_type: a video can appear under two edit types
    # (0011_lucia is in both edit2 and edit5) and only this --edit_type's pair runs here.
    if cases_json is not None:
        wanted = {c["video_name"] for c in json.loads(cases_json.read_text())
                  if int(c["edit_type"]) == args.edit_type}
        if not wanted:
            raise SystemExit(f"[r22_dump] {cases_json} lists no case with "
                             f"edit_type={args.edit_type}")
        entries = [e for e in entries if e["video_name"] in wanted]
        found = {e["video_name"] for e in entries}
        if found != wanted:
            raise SystemExit(f"[r22_dump] cases not present in {edit_json.name}: "
                             f"{sorted(wanted - found)}")

    vp_mode = args.vp_mode if first_frame_edit_dir is not None else "novp"
    n_steps = args.step

    print(f"[r22_dump] edit_type={args.edit_type} method={args.method} "
          f"pairs={len(entries)} vp_mode={vp_mode} steps={n_steps} "
          f"blend_sched={args.blend_sched or 'paper(eq4)'} out={out_root / args.method}",
          flush=True)

    rows = []
    for e in entries:
        video_name = e["video_name"]
        pair_id = e.get("id", video_name)
        src_path = data_root / "videos" / f"{video_name}.mp4"

        # Re-seed before EVERY pair -- without this a clip's noise depends on its index
        # in the edit{T} json, making subset runs incomparable to full runs. See the
        # warning in run_fivebench.py; this is what keeps the step-14 render
        # bit-identical to the stored full-bench arm.
        set_seed(args.seed)

        try:
            src_video = load_video(str(src_path))
            new_len = find_closest_num_frame(len(src_video))
            if not new_len:
                raise ValueError(f"video too short: {len(src_video)} frames")
            src_video = src_video[:new_len]

            src_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
            video_latents = pipeline.vae.encode_to_latent(
                src_tensor.to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16)
            pipeline.vae.model.clear_cache()

            if vp_mode != "novp":
                anchor_path = first_frame_edit_dir / f"edit{args.edit_type}" / f"{video_name}.png"
                if not anchor_path.exists():
                    raise FileNotFoundError(f"missing anchor {anchor_path}")
                trg_first_frame = Image.open(anchor_path).convert("RGB")

            if vp_mode == "vp":
                # Encode order (src first frame, THEN anchor) and the single trailing
                # clear_cache are load bearing: the VAE carries temporal-conv state
                # between calls, so reordering these two encodes changes the latents and
                # breaks bit-parity with R7. Do not "tidy" this.
                src_first_frame_latent = pipeline.vae.encode_to_latent(
                    transform(src_video[0]).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                trg_first_frame_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = True
                vp_latent = None
            elif vp_mode == "pvp":
                vp_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = None
            else:                                   # novp -- R1 baseline path
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = vp_latent = None

            # wo_video_decode=True: the final video IS step n_steps-1 (asserted below),
            # so decoding it here as well would be a wasted VAE pass.
            steps: list = []
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
                independent_first_frame=independent_first_frame,
                src_initial_latent=src_first_frame_latent,
                trg_initial_latent=trg_first_frame_latent,
                fg_boost_factor=args.fg_boost_factor,
                blend_power=args.blend_power,
                rollout_chunk_size=args.rollout_chunk_size,
                rollout_overlap_block_num=args.rollout_overlap_block_num,
                vp_latent=vp_latent,
                blend_sched=args.blend_sched,
                #✨ R22
                step_dump=steps,
            )

            if len(steps) != 1:
                raise RuntimeError(f"expected 1 stitched step buffer, got {len(steps)}")
            step_latents = steps[0]
            if step_latents.shape[0] != n_steps:
                raise RuntimeError(f"step buffer has {step_latents.shape[0]} steps, "
                                   f"expected {n_steps}")
            # L1 identity gate at the driver level (the pipeline already asserted it per
            # window and after stitching -- this one is what the smoke log reports).
            if not torch.equal(step_latents[-1], final_latent):
                raise RuntimeError("R22 GATE1 FAIL: last step latent != final latent")
            print(f"[r22_dump] GATE1 PASS {video_name}: step[{n_steps - 1}] == final latent "
                  f"(shape {tuple(step_latents.shape)})", flush=True)

            # Decode ONE step at a time. 15 decoded videos at [81,3,480,832] fp32 would
            # be 7.3 GB resident; this keeps peak memory at one video. clear_cache()
            # between decodes so the VAE's temporal-conv state cannot leak from step j
            # into step j+1 -- if it did, step 14 would not reproduce the stored render,
            # which is exactly what the smoke's GATE2 checks.
            n_frames = 0
            for j in range(n_steps):
                torch.cuda.empty_cache()
                video_j = pipeline.vae.decode_to_pixel(step_latents[j], use_cache=False)
                video_j = (video_j * 0.5 + 0.5).clamp(0, 1)
                pipeline.vae.model.clear_cache()
                out_dir = out_root / args.method / f"step{j:02d}" / f"edit{args.edit_type}" / video_name
                n_frames = save_frames_png(video_j[0], out_dir)
                del video_j

            del step_latents, steps
            rows.append((pair_id, video_name, "ok", n_frames, n_steps))
            print(f"[ok] {video_name}: {n_steps} steps x {n_frames} frames", flush=True)
        except Exception as ex:  # never silently swallow -- record and continue
            rows.append((pair_id, video_name, f"error:{type(ex).__name__}", 0, 0))
            print(f"[ERROR] {video_name}: {ex}", flush=True)

    manifest_dir = out_root / args.method
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"_manifest_edit{args.edit_type}.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "video_name", "status", "n_frames", "n_steps"])
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r[2] == "ok")
    print(f"[r22_dump] done edit{args.edit_type}: {n_ok}/{len(rows)} ok -> {manifest}",
          flush=True)
    if n_ok != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
