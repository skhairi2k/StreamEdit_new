"""R66: render ContextFlow on FiVE-Bench-Correct pairs with the authors' 2-GPU setup (env `contextflow`, torchrun).

Launch (as inversion.sh and the demo scripts: --dit_fsdp --ulysses_size 2 --t5_cpu --offload_model True):
  ~/anaconda3/envs/contextflow/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
      evaluation/r66_contextflow_run.py --cases evaluation/r66_cases_smoke.json --frames 41 --out_root $R66_ROOT/smoke

Per video, RF-Solver inversion of the source clip (generate_reconstruction, empty prompt, source first frame) is run once and its
latents are kept under {out_root}/latents/n{F}/{video}.pt. Per pair, generate_with_pnp edits from the inverted latent at the repo's
load t with the case's mode (swap / insert / delete layers), the Correct anchor as first frame and the Correct target_prompt.

Distributed: process group and xfuser model parallel are initialised as in generate.py (nccl; sequence parallel 2 = Ulysses 2,
ring 1); WanI2V(dit_fsdp=True, use_usp=True, t5_cpu=True). Both ranks run every case in the same order with the same seed; rank 0
alone decides skips / inversions (broadcast), writes latents, frames and the manifest, and prints. generate_reconstruction saves
the latent on rank 0 without a barrier, so the driver renames it and then barriers before any rank loads it. After each case
the ranks all-reduce (MAX) the status code and peak memory, so the manifest holds the worse rank and the ranks never diverge.
After the first failed pair (OOM or error) the run STOPS: an exception inside FSDP's forward leaves the model unusable (smoke
job 1080831: every later pair failed with 'expected to be in states [IDLE] but current state is FORWARD_BACKWARD'), so the
remaining pairs are left for a resubmit, which resumes. An OOM prints its message (what was tried, free / reserved) per rank.
A failure on one rank only (the other stuck in a collective) is not recoverable in-process: the NCCL timeout (30 min) aborts the
job instead of hanging.

Case paths: anchor_image and src_first_frame are relative to --data_dir (default ~/Data; on Jean Zay ~/Data -> $WORK/Data).

Frames: the first n = case["n{F}"] source frames (F = 41 for the render, 81 only as the smoke diagnostic), resized to 832x480 with
cv2 (Wan-Edit's load_frames_path convention). The released code hard-codes 81 / 49 frames; this driver needs the clone patched with
evaluation/r66_contextflow_frames.patch and refuses to start otherwise.

Settings (repo demos): fm_new, 51 steps, CFG 3.0 for the edit, injection_step 0.5, prompt_origin '', seed 0. fm_new builds its
scheduler with shift 5.0 internally, so --shift (3.0, the CLI default at 832*480) is passed through but has no effect. The inversion
call gets the CLI default guide scale 5.0 (inversion.sh sets none); generate_reconstruction does not use it.

Outputs: {out_root}/n{F}/edit{T}/{video}/00000.png ... and {out_root}/manifest_n{F}_shard{i}.csv
(case_id, mode, n_frames, sec_inversion, sec_edit, peak_mem_gb, status). sec_inversion is filled on the pair whose run computed the
video's inversion and 0 when the latent was reused; peak_mem_gb is torch.cuda.max_memory_reserved over that pair (incl. inversion),
max over the 2 GPUs. Resume: a pair whose dir already holds n frames is skipped; an existing inversion latent is reused.

Exit code: 0 all pairs ok, 2 only OOM failures (the smoke tolerates these at one frame count), 1 any other failure. torchrun does
not pass a worker's code through, so rank 0 also writes it to {out_root}/exitcode_n{F}_shard{i}.txt (read by the sbatch scripts).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

SIZE = "832*480"
WIDTH, HEIGHT = 832, 480
SAMPLE_SOLVER = "fm_new"
SAMPLE_STEPS = 51
SHIFT = 3.0
EDIT_GUIDE_SCALE = 3.0
INVERSION_GUIDE_SCALE = 5.0
INJECTION_STEP = 0.5
LOAD_T = 995.9473266601562          # --load_intermediate_latent_t of every ContextFlow demo
ULYSSES_SIZE = 2                    # --ulysses_size 2 of inversion.sh and every demo script
PATCH_MARKER = "R66: was 81 / 49"
NCCL_TIMEOUT = timedelta(minutes=30)
MANIFEST_FIELDS = ["case_id", "mode", "n_frames", "sec_inversion", "sec_edit", "peak_mem_gb", "status"]
STATUS_OK, STATUS_OOM, STATUS_ERROR = 0, 1, 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--frames", required=True, type=int, choices=[41, 81])
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--cf_dir", type=Path, default=Path(os.environ.get("CF_DIR", "~/Code/ContextFlow")).expanduser())
    p.add_argument("--ckpt_dir", type=Path,
                   default=Path(os.environ.get("CF_WEIGHTS", "/work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P")))
    p.add_argument("--data_dir", type=Path, default=Path("~/Data").expanduser(),
                   help="root the cases' anchor_image / src_first_frame are relative to")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_source(frame_dir: Path, n: int) -> tuple[torch.Tensor, Image.Image]:
    """First n frames (sorted) resized to 832x480 -> (3, n, 480, 832) in [-1, 1], plus frame 0 as a PIL image."""
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith((".jpg", ".png")))
    if len(files) < n:
        raise RuntimeError(f"{frame_dir} has {len(files)} frames, need {n}")
    frames = []
    for name in files[:n]:
        bgr = cv2.imread(str(frame_dir / name))
        if bgr is None:
            raise RuntimeError(f"cannot read {frame_dir / name}")
        frames.append(cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB))
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div(255.0).mul(2.0).sub(1.0)
    return video, Image.fromarray(frames[0])


def load_anchor(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img if img.size == (WIDTH, HEIGHT) else img.resize((WIDTH, HEIGHT), Image.BICUBIC)


def save_frames(video: torch.Tensor, out_dir: Path) -> None:
    """(3, n, H, W) in [-1, 1] -> out_dir/00000.png ..., written to a temp dir and renamed so partial dirs never count."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    arr = video.float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    for i, frame in enumerate(arr):
        Image.fromarray(frame).save(tmp / f"{i:05d}.png")
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


def n_pngs(d: Path) -> int:
    return len(list(d.glob("*.png"))) if d.is_dir() else 0


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def broadcast(obj, rank: int):
    """rank 0's value of obj on every rank."""
    box = [obj if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def init_distributed() -> tuple[int, int, int]:
    """Process group + xfuser model parallel, as generate.py does for --ulysses_size 2."""
    rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world_size != ULYSSES_SIZE:
        raise RuntimeError(f"launch with torchrun --nproc_per_node={ULYSSES_SIZE} (got WORLD_SIZE={world_size})")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size, timeout=NCCL_TIMEOUT)
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    initialize_model_parallel(sequence_parallel_degree=dist.get_world_size(), ring_degree=1, ulysses_degree=ULYSSES_SIZE)
    return rank, world_size, local_rank


def main() -> int:
    args = parse_args()
    image2video = args.cf_dir / "wan" / "image2video.py"
    if PATCH_MARKER not in image2video.read_text():
        print(f"ERROR: {image2video} is not patched with r66_contextflow_frames.patch", file=sys.stderr)
        return 1
    rank, world_size, local_rank = init_distributed()
    sys.path.insert(0, str(args.cf_dir))
    import wan  # noqa: E402
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # noqa: E402

    nkey = f"n{args.frames}"
    cases = json.loads(args.cases.read_text())
    videos = list(dict.fromkeys(c["video_name"] for c in cases))
    mine = {v for j, v in enumerate(videos) if j % args.n_shards == args.shard}
    cases = [c for c in cases if c["video_name"] in mine]

    if rank == 0:
        commit = subprocess.run(["git", "-C", str(args.cf_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        print(f"[r66_run] args={vars(args)}")
        print(f"[r66_run] node={platform.node()} world_size={world_size} gpu={torch.cuda.get_device_name()} "
              f"torch={torch.__version__} contextflow={commit} patched=yes")
        print(f"[r66_run] data_dir={args.data_dir} dit_fsdp=True ulysses_size={ULYSSES_SIZE} t5_cpu=True offload_model=True; "
              f"shard {args.shard}/{args.n_shards}: {len(mine)} videos, {len(cases)} pairs, frames cap {args.frames}; "
              f"solver {SAMPLE_SOLVER} steps {SAMPLE_STEPS} shift {SHIFT} cfg {EDIT_GUIDE_SCALE} injection {INJECTION_STEP} "
              f"load_t {LOAD_T} seed {args.seed}")

    latent_dir = args.out_root / "latents" / nkey
    manifest = args.out_root / f"manifest_{nkey}_shard{args.shard}.csv"
    exitcode_file = args.out_root / f"exitcode_{nkey}_shard{args.shard}.txt"
    if rank == 0:
        latent_dir.mkdir(parents=True, exist_ok=True)
        exitcode_file.unlink(missing_ok=True)
    max_area = MAX_AREA_CONFIGS[SIZE]

    t0 = time.time()
    wan_i2v = wan.WanI2V(config=WAN_CONFIGS["i2v-14B"], checkpoint_dir=str(args.ckpt_dir), device_id=local_rank, rank=rank,
                         t5_fsdp=False, dit_fsdp=True, use_usp=True, t5_cpu=True)
    dist.barrier()
    if rank == 0:
        print(f"[r66_run] model loaded in {time.time() - t0:.0f}s")

    n_ok = n_skip = n_oom = n_err = 0
    stopped = False
    for video_name in [v for v in videos if v in mine]:
        if stopped:
            break
        latent_path = latent_dir / f"{video_name}.pt"
        for case in (c for c in cases if c["video_name"] == video_name):
            n = int(case[nkey])
            out_dir = args.out_root / nkey / f"edit{case['edit_type']}" / video_name
            if broadcast(n_pngs(out_dir) == n, rank):
                n_skip += 1
                continue
            row = {"case_id": case["case_id"], "mode": case["mode"], "n_frames": n,
                   "sec_inversion": 0.0, "sec_edit": 0.0, "peak_mem_gb": 0.0, "status": "ok"}
            status = STATUS_OK
            torch.cuda.reset_peak_memory_stats()
            try:
                seed_everything(args.seed)
                video, src0 = load_source((args.data_dir / case["src_first_frame"]).parent, n)
                if broadcast(not latent_path.exists(), rank):
                    t1 = time.time()
                    tmp = latent_path.with_suffix(".pt.tmp")
                    wan_i2v.generate_reconstruction(
                        "", src0, video, max_area=max_area, frame_num=n, shift=SHIFT, sample_solver=SAMPLE_SOLVER,
                        sampling_steps=SAMPLE_STEPS, guide_scale=INVERSION_GUIDE_SCALE, seed=args.seed,
                        offload_model=True, latent_name=str(tmp), is_delete=False)
                    if rank == 0:
                        os.replace(tmp, latent_path)
                    dist.barrier()
                    row["sec_inversion"] = round(time.time() - t1, 1)
                keys = torch.load(latent_path, map_location="cpu").keys()
                if LOAD_T not in keys:
                    raise RuntimeError(f"load t {LOAD_T} not among the saved inversion steps {sorted(keys)}")
                seed_everything(args.seed)
                t2 = time.time()
                _, edited = wan_i2v.generate_with_pnp(
                    case["target_prompt"], "", load_anchor(str(args.data_dir / case["anchor_image"])), src0, max_area=max_area,
                    frame_num=n, shift=SHIFT, sample_solver=SAMPLE_SOLVER, sampling_steps=SAMPLE_STEPS,
                    guide_scale=EDIT_GUIDE_SCALE, seed=args.seed, offload_model=True, pnp_layers=case["pnp_layers"],
                    load_intermediate_latent_path=str(latent_path), load_intermediate_latent_t=LOAD_T,
                    injection_step=INJECTION_STEP, latent_output_dir=None, is_delete=case["is_delete"])
                if rank == 0:
                    if edited.shape[1] != n:
                        raise RuntimeError(f"rendered {edited.shape[1]} frames, expected {n}")
                    save_frames(edited, out_dir)
                row["sec_edit"] = round(time.time() - t2, 1)
            except torch.cuda.OutOfMemoryError as exc:
                status = STATUS_OOM
                print(f"[r66_run] rank {rank}: OOM on {case['case_id']} at n={n}: {exc}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 -- logged, counted, and fails the run at the end
                status = STATUS_ERROR
                row["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
                print(f"[r66_run] rank {rank}: error on {case['case_id']}", file=sys.stderr)
                traceback.print_exc()
            finally:
                peak_gb = torch.cuda.max_memory_reserved() / 2**30
                torch.cuda.empty_cache()
            # worse status and higher peak over the ranks
            agreed = torch.tensor([float(status), peak_gb], device="cuda")
            dist.all_reduce(agreed, op=dist.ReduceOp.MAX)
            status, row["peak_mem_gb"] = int(agreed[0].item()), round(agreed[1].item(), 2)
            if status == STATUS_OK:
                n_ok += 1
            elif status == STATUS_OOM:
                row["status"] = "oom"
                n_oom += 1
            else:
                if row["status"] == "ok":
                    row["status"] = "error on another rank (see log)"
                n_err += 1
            if rank == 0:
                append_row(manifest, row)
                print(f"[r66_run] {case['case_id']} n={n} {row['status']} inv {row['sec_inversion']}s "
                      f"edit {row['sec_edit']}s peak {row['peak_mem_gb']} GB (max over GPUs)")
            if status != STATUS_OK:
                # FSDP cannot recover from an exception inside its forward: stop, a resubmit resumes
                stopped = True
                if rank == 0:
                    print(f"[r66_run] stopping after the failure on {case['case_id']}; the remaining pairs of this shard "
                          f"are left for a resubmit (resume skips finished pairs)")
                break

    code = 1 if n_err else (2 if n_oom else 0)
    if rank == 0:
        print(f"[r66_run] done: ok {n_ok}, skipped {n_skip}, oom {n_oom}, errors {n_err}; manifest {manifest}")
        print(f"[r66_run] failures: {n_oom + n_err}")
        exitcode_file.write_text(f"{code}\n")
    dist.barrier()
    dist.destroy_process_group()
    return code


if __name__ == "__main__":
    sys.exit(main())
