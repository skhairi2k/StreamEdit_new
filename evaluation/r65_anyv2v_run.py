"""R65: sharded AnyV2V (I2VGen-XL) driver for the FiVE-Bench-Correct baseline. Runs in the ``anyv2v`` env.

Reproduces AnyV2V's ``run_group_ddim_inversion.py`` + ``run_group_pnp_edit.py`` (commit bc540befac) in one
process, with FiVE's published setting as the defaults: 512x512 squashed source, 500 DDIM-inversion steps
(cfg 1.0, empty prompt), 50 PnP edit steps, cfg 9.0, ``ddim_init_latents_t_idx`` 0, PnP thresholds
0.2 / 0.2 / 0.5, seed 8888, AnyV2V's default negative prompt.

Per video (video j of the sorted video list goes to shard j mod K):
  1. source frames 0..n-1 (sorted, re-indexed) are squashed to 512x512 (LANCZOS) -> <D>/src512/<video>/%05d.png
  2. one DDIM inversion -> <D>/inversions/<video> (shared by all of the video's edit types; deleted once every
     pair of the video is complete; a ``.complete`` marker lets a resumed run reuse a finished inversion)
  3. per pair: PnP edit with the pair's first frame (Correct anchor, or the FiVE release frame for the repro
     gate) -> <out_root>_raw512/edit<T>/<video>/%05d.png and the same frames resized with PIL's default resize
     to the source frame size -> <out_root>_srcres/edit<T>/<video>/%05d.png
where <D> is the parent of --out_root. Inversion and editing use two pipeline instances: AnyV2V ran them as
separate processes, and PnP patches the UNet in place, so a shared pipeline would inject into the next
inversion.

Resume: a pair whose two frame dirs hold n frames is skipped. Frame dirs are written to ``<dir>.tmp`` and renamed,
so a killed job never leaves a dir that looks complete. Per-shard manifest ``<out_root>_manifest_shard<i>.csv``
(appended): case_id, n_frames, sec_inversion, sec_edit, peak_mem_gb, inversion_disk_gb, status. The inversion
time and disk are logged on the first pair rendered of each video (0 on its other pairs). Exit 1 if any pair failed.

Usage (one shard):
    python -I evaluation/r65_anyv2v_run.py --cases evaluation/r65_cases_correct.json \
        --out_root ~/Data/dataggen/outputs/five_bench/r65_anyv2v/correct --shard 0 --n_shards 10
"""
import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_NEGATIVE_PROMPT = (
    "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, "
    "disconnected limbs, Ugly faces, incomplete arms"
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
MANIFEST_FIELDS = ["case_id", "n_frames", "sec_inversion", "sec_edit", "peak_mem_gb", "inversion_disk_gb", "status"]
logger = logging.getLogger("r65_anyv2v_run")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dir_size_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 ** 3


def frames_complete(path: Path, n: int) -> bool:
    return path.is_dir() and len(list(path.glob("*.png"))) == n


def save_frames(frames: List[Image.Image], final_dir: Path) -> None:
    """Write frames 00000.png.. into <dir>.tmp, then rename, so a killed job leaves no half-filled dir."""
    tmp = final_dir.with_name(final_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for i, frame in enumerate(frames):
        frame.save(tmp / f"{i:05d}.png")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp, final_dir)


def load_cases(path: Path) -> Dict[str, List[dict]]:
    """Cases grouped by video; all pairs of a video must agree on n_frames."""
    by_video: Dict[str, List[dict]] = {}
    for case in json.loads(path.read_text()):
        by_video.setdefault(case["video_name"], []).append(case)
    for video, cases in by_video.items():
        if len({c["n_frames"] for c in cases}) != 1:
            raise ValueError(f"{video}: pairs disagree on n_frames")
    return by_video


def prepare_source(data_root: Path, video: str, n: int, size: int, src512_dir: Path) -> Tuple[List[Image.Image], Tuple[int, int]]:
    """First n source frames squashed to size x size (no crop). Returns the frames and the source (W, H)."""
    files = sorted(p for p in (data_root / "images" / video).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) < n:
        raise ValueError(f"{video}: {len(files)} source frames < n_frames {n}")
    first = Image.open(files[0]).convert("RGB")
    src_size = first.size
    if frames_complete(src512_dir, n):
        frames = [Image.open(src512_dir / f"{i:05d}.png").convert("RGB") for i in range(n)]
    else:
        frames = [Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS) for f in files[:n]]
        save_frames(frames, src512_dir)
    return frames, src_size


def load_pipe(weights: Path, device: torch.device):
    from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
    pipe = I2VGenXLPipeline.from_pretrained(str(weights), torch_dtype=torch.float16, variant="fp16")
    return pipe.to(device)


def init_pnp(pipe, scheduler, n_steps: int, pnp_f_t: float, pnp_spatial_attn_t: float, pnp_temp_attn_t: float) -> None:
    """Same as AnyV2V's run_group_pnp_edit.init_pnp."""
    from pnp_utils import register_conv_injection, register_spatial_attention_pnp, register_temp_attention_pnp
    conv_t = int(n_steps * pnp_f_t)
    spatial_t = int(n_steps * pnp_spatial_attn_t)
    temp_t = int(n_steps * pnp_temp_attn_t)
    register_conv_injection(pipe, scheduler.timesteps[:conv_t] if conv_t >= 0 else [])
    register_spatial_attention_pnp(pipe, scheduler.timesteps[:spatial_t] if spatial_t >= 0 else [])
    register_temp_attention_pnp(pipe, scheduler.timesteps[:temp_t] if temp_t >= 0 else [])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out_root", required=True, help="e.g. <D>/r65_anyv2v/correct; <D> = its parent")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--data_root", default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    ap.add_argument("--anyv2v_dir", default="~/Code/AnyV2V")
    ap.add_argument("--weights", default="~/Data/weights/i2vgen-xl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=8888)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--inv_steps", type=int, default=500)
    ap.add_argument("--inv_cfg", type=float, default=1.0)
    ap.add_argument("--edit_steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=9.0)
    ap.add_argument("--t_idx", type=int, default=0, help="ddim_init_latents_t_idx (0 = timestep 981)")
    ap.add_argument("--pnp_f_t", type=float, default=0.2)
    ap.add_argument("--pnp_spatial_attn_t", type=float, default=0.2)
    ap.add_argument("--pnp_temp_attn_t", type=float, default=0.5)
    ap.add_argument("--target_fps", type=int, default=8)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard {args.shard} outside [0, {args.n_shards})")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(args.anyv2v_dir).expanduser() / "i2vgen-xl"))
    from diffusers import DDIMInverseScheduler, DDIMScheduler
    from utils import load_ddim_latents_at_t

    out_root = Path(args.out_root).expanduser()
    base = out_root.parent
    raw_root = out_root.with_name(out_root.name + "_raw512")
    srcres_root = out_root.with_name(out_root.name + "_srcres")
    src512_root, inv_root = base / "src512", base / "inversions"
    data_root = Path(args.data_root).expanduser()
    weights = Path(args.weights).expanduser()
    manifest_path = out_root.with_name(f"{out_root.name}_manifest_shard{args.shard}.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("args: %s", json.dumps(vars(args), sort_keys=True))
    logger.info("torch %s, cuda %s, host %s", torch.__version__, torch.cuda.is_available(), os.uname().nodename)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    seed_everything(args.seed)

    by_video = load_cases(Path(args.cases).expanduser())
    videos = [v for j, v in enumerate(sorted(by_video)) if j % args.n_shards == args.shard]
    logger.info("shard %d/%d: %d of %d videos, %d pairs", args.shard, args.n_shards, len(videos), len(by_video),
                sum(len(by_video[v]) for v in videos))

    pipe_inv = pipe_edit = None
    inverse_scheduler = ddim_scheduler = None
    n_failed = 0
    new_manifest = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if new_manifest:
            writer.writeheader()

        def log_row(**row) -> None:
            writer.writerow(row)
            mf.flush()

        for video in videos:
            cases = by_video[video]
            n = cases[0]["n_frames"]
            raw_dir = lambda c: raw_root / f"edit{c['edit_type']}" / video
            res_dir = lambda c: srcres_root / f"edit{c['edit_type']}" / video
            todo = [c for c in cases if not (frames_complete(raw_dir(c), n) and frames_complete(res_dir(c), n))]
            for c in cases:
                if c not in todo:
                    log_row(case_id=c["case_id"], n_frames=n, sec_inversion=0, sec_edit=0, peak_mem_gb=0,
                            inversion_disk_gb=0, status="skipped_complete")
            inv_dir = inv_root / video
            if not todo:
                shutil.rmtree(inv_dir, ignore_errors=True)
                continue

            try:
                frames, src_size = prepare_source(data_root, video, n, args.size, src512_root / video)
            except Exception as ex:  # record and continue: one bad video must not stop the shard
                logger.error("%s: source preparation failed: %s", video, ex)
                for c in todo:
                    log_row(case_id=c["case_id"], n_frames=n, sec_inversion=0, sec_edit=0, peak_mem_gb=0,
                            inversion_disk_gb=0, status=f"error:source:{type(ex).__name__}")
                    n_failed += 1
                continue

            if pipe_inv is None:
                pipe_inv = load_pipe(weights, device)
                pipe_edit = load_pipe(weights, device)
                inverse_scheduler = DDIMInverseScheduler.from_pretrained(str(weights), subfolder="scheduler")
                ddim_scheduler = DDIMScheduler.from_pretrained(str(weights), subfolder="scheduler")

            # ---- inversion, once per video ----
            latents_dir = inv_dir / "ddim_latents"
            marker = inv_dir / ".complete"
            sec_inv, inv_disk = 0.0, 0.0
            try:
                if not (marker.is_file() and marker.read_text().strip() == str(n)):
                    shutil.rmtree(inv_dir, ignore_errors=True)
                    torch.cuda.synchronize(device)
                    t0 = time.time()
                    pipe_inv.scheduler = inverse_scheduler
                    video_latents_at_0 = pipe_inv.encode_vae_video(
                        frames, device=pipe_inv._execution_device, height=args.size, width=args.size)
                    pipe_inv.invert(
                        prompt="", image=frames[0], height=args.size, width=args.size, num_frames=n,
                        num_inference_steps=args.inv_steps, guidance_scale=args.inv_cfg, negative_prompt="",
                        target_fps=args.target_fps, latents=video_latents_at_0,
                        generator=torch.Generator(device=device).manual_seed(args.seed),
                        return_dict=False, output_dir=str(latents_dir))
                    torch.cuda.synchronize(device)
                    sec_inv = time.time() - t0
                    marker.write_text(str(n))
                    inv_disk = dir_size_gb(inv_dir)
                    logger.info("%s: inversion %d frames in %.0fs, %.2f GB", video, n, sec_inv, inv_disk)
            except Exception as ex:
                logger.error("%s: inversion failed: %s\n%s", video, ex, traceback.format_exc())
                for c in todo:
                    log_row(case_id=c["case_id"], n_frames=n, sec_inversion=0, sec_edit=0, peak_mem_gb=0,
                            inversion_disk_gb=0, status=f"error:inversion:{type(ex).__name__}")
                    n_failed += 1
                continue

            # ---- PnP edit, once per pair ----
            for k, c in enumerate(todo):
                try:
                    first_frame = Image.open(Path(c.get("anchor_image") or c["first_frame"]).expanduser()).convert("RGB")
                    first_frame = first_frame.resize((args.size, args.size), Image.LANCZOS)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    t0 = time.time()
                    ddim_scheduler.set_timesteps(args.edit_steps)
                    latents_at_t = load_ddim_latents_at_t(ddim_scheduler.timesteps[args.t_idx], str(latents_dir))
                    init_pnp(pipe_edit, ddim_scheduler, args.edit_steps, args.pnp_f_t, args.pnp_spatial_attn_t,
                             args.pnp_temp_attn_t)
                    pipe_edit.register_modules(scheduler=ddim_scheduler)
                    # random_ratio is 0 in FiVE's setting: the blended latents equal the inverted latents.
                    edited = pipe_edit.sample_with_pnp(
                        prompt=c["target_prompt"], image=first_frame, height=args.size, width=args.size,
                        num_frames=n, num_inference_steps=args.edit_steps, guidance_scale=args.cfg,
                        negative_prompt=args.negative_prompt, target_fps=args.target_fps, latents=latents_at_t,
                        generator=torch.manual_seed(args.seed), return_dict=True,
                        ddim_init_latents_t_idx=args.t_idx, ddim_inv_latents_path=str(latents_dir),
                        ddim_inv_prompt="", ddim_inv_1st_frame=frames[0]).frames[0]
                    torch.cuda.synchronize(device)
                    sec_edit = time.time() - t0
                    if len(edited) != n:
                        raise RuntimeError(f"{len(edited)} frames generated, expected {n}")
                    edited = [f.resize((args.size, args.size), resample=Image.LANCZOS) for f in edited]
                    save_frames(edited, raw_dir(c))
                    save_frames([f.resize(src_size) for f in edited], res_dir(c))  # PIL default resample
                    log_row(case_id=c["case_id"], n_frames=n, sec_inversion=round(sec_inv if k == 0 else 0, 1),
                            sec_edit=round(sec_edit, 1),
                            peak_mem_gb=round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 2),
                            inversion_disk_gb=round(inv_disk if k == 0 else 0, 2), status="ok")
                    logger.info("[ok] %s: %d frames, edit %.0fs", c["case_id"], n, sec_edit)
                except Exception as ex:
                    logger.error("%s failed: %s\n%s", c["case_id"], ex, traceback.format_exc())
                    log_row(case_id=c["case_id"], n_frames=n, sec_inversion=0, sec_edit=0, peak_mem_gb=0,
                            inversion_disk_gb=0, status=f"error:{type(ex).__name__}")
                    n_failed += 1

            if all(frames_complete(raw_dir(c), n) and frames_complete(res_dir(c), n) for c in cases):
                shutil.rmtree(inv_dir, ignore_errors=True)  # a failed pair keeps it for the resumed run

    print(f"[r65_anyv2v_run] shard {args.shard}/{args.n_shards} failures: {n_failed}")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
