"""R63: UniEdit-Flow (ICLR'26) native baseline on FiVE-Bench-Correct, Wan2.1-T2V-1.3B-Diffusers.

Per pair: encode the source clip -> invert (UniInvEulerScheduler, prompt "") -> edit (UniEditEulerScheduler,
[source_prompt, target_prompt]) -> decode. Same sequence as UniEdit-Flow_diffusers/demo_wan_video_edit.py; the model is
loaded once per invocation. Frames follow the Wan-Edit convention (wan-edit/edit.py load_frames): the first min(41, n)
frames of videos/{video}.mp4, cv2 resize to 832x480, RGB, truncated to 4k+1 (r48_wan_common.expected_frames).

Both batch slots of the edit output are identical (UniEditEulerScheduler.step concatenates the fused update twice), so
edit_latents[:1] is decoded, as in the demo.

Output: {out_root}/edit{T}/{video}/00000.png ...   (the ``edit_video`` layout evaluate.py reads)
        {out_root}/_manifest_edit{T}.csv            (case_id, n_frames, seconds)
RESUMABLE: clips whose expected number of PNGs already exists are skipped.

usage: python evaluation/r63_uniedit_render.py --cases evaluation/r43_anchor_manifest_correct.json --edit_type 1 --out_root DIR
       python evaluation/r63_uniedit_render.py --cases ... --out_root DIR --verify     (no model load)
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
UNIEDIT_DIR = REPO / "UniEdit-Flow_diffusers"
sys.path.insert(0, str(UNIEDIT_DIR))
sys.path.insert(0, str(REPO / "evaluation"))
import r48_wan_common as C  # noqa: E402

WIDTH, HEIGHT = 832, 480
CORRECT_DIR = REPO / "evaluation" / "five_bench_correct"


def load_source_frames(video_name: str) -> list[Image.Image]:
    """First min(41, n) frames, cv2-resized to 832x480 (as wan-edit load_frames), truncated to 4k+1, as PIL RGB."""
    import cv2
    cap = cv2.VideoCapture(str(C.FIVE / "videos" / f"{video_name}.mp4"))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_name}.mp4")
    frames = []
    for _ in range(C.FRAME_NUM):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"{video_name}.mp4 has no frames")
    return frames[: C.expected_frames(len(frames))]


def clip_dir(out_root: Path, case: dict) -> Path:
    return out_root / f"edit{case['edit_type']}" / case["video_name"]


def expected_for(case: dict) -> int:
    return C.expected_frames(C.source_frame_count(case["video_name"]))


def is_complete(out_root: Path, case: dict) -> bool:
    d = clip_dir(out_root, case)
    return d.is_dir() and len(list(d.glob("*.png"))) == expected_for(case)


def save_frames(video: np.ndarray, out_dir: Path) -> int:
    """[F, H, W, 3] float in [0, 1] -> zero-padded PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (video.clip(0, 1) * 255).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i]).save(out_dir / f"{i:05d}.png")
    return arr.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="json with video_name / edit_type (r7 manifest format)")
    ap.add_argument("--edit_json_dir", default=str(CORRECT_DIR))
    ap.add_argument("--edit_type", type=int, default=None, help="only this edit type (1-6)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--omega", type=float, default=5.0)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    ap.add_argument("--revision", default="0fad780a534b6463e45facd96134c9f345acfa5b")
    ap.add_argument("--verify", action="store_true", help="count complete clips per type, no model load")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser()
    cases = C.load_cases(args.cases, anno_dir=args.edit_json_dir)
    if args.edit_type is not None:
        cases = [c for c in cases if c["edit_type"] == args.edit_type]
    if not cases:
        print("[r63_render] no cases selected")
        return 1

    if args.verify:
        missing = [c for c in cases if not is_complete(out_root, c)]
        for t in sorted({c["edit_type"] for c in cases}):
            n_t = sum(c["edit_type"] == t for c in cases)
            n_ok = sum(c["edit_type"] == t and is_complete(out_root, c) for c in cases)
            print(f"[r63_render] edit{t}: {n_ok}/{n_t} complete clips")
        for c in missing[:5]:
            print(f"[r63_render] incomplete: edit{c['edit_type']}/{c['video_name']}")
        print(f"[r63_render] missing: {len(missing)}")
        return int(bool(missing))

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from schedulers.UniEditEulerScheduler import UniEditEulerScheduler
    from schedulers.UniInvEulerScheduler import UniInvEulerScheduler
    from utils import decode_video, encode_video, pack_edit_input, seed_everything

    import diffusers
    import transformers
    print(f"[r63_render] node={platform.node()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"[r63_render] torch {torch.__version__} diffusers {diffusers.__version__} transformers {transformers.__version__}")
    print(f"[r63_render] model={args.model_path}@{args.revision} alpha={args.alpha} omega={args.omega} step={args.step} "
          f"flow_shift={args.flow_shift} seed={args.seed} size={WIDTH}x{HEIGHT} max_frames={C.FRAME_NUM}")
    print(f"[r63_render] cases={args.cases} edit_json_dir={args.edit_json_dir} edit_type={args.edit_type} "
          f"n={len(cases)} out_root={out_root}")

    # VAE and pipeline loaded separately (fp32 VAE, bf16 transformer), as the demo
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32,
                                           revision=args.revision)
    pipe = WanPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16, revision=args.revision)
    pipe.to("cuda")
    inv_sched = UniInvEulerScheduler(num_train_timesteps=1000, shift=args.flow_shift)
    edit_sched = UniEditEulerScheduler(num_train_timesteps=1000, shift=args.flow_shift)
    inv_sched.set_hyperparameters(alpha=args.alpha)
    edit_sched.set_hyperparameters(alpha=args.alpha, omega=args.omega)

    out_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for k, case in enumerate(cases):
        cid = f"edit{case['edit_type']}/{case['video_name']}"
        if is_complete(out_root, case):
            print(f"[r63_render] ({k + 1}/{len(cases)}) {cid}: complete, skipped")
            continue
        t0 = time.time()
        try:
            seed_everything(args.seed)    # per pair: a pair's output must not depend on its position in the list
            frames = load_source_frames(case["video_name"])
            n_frames = len(frames)
            lat = encode_video(pipe, vae, frames)
            pipe.scheduler = inv_sched
            inv_lat = pipe(prompt="", height=HEIGHT, width=WIDTH, num_frames=n_frames,
                           num_inference_steps=args.step, guidance_scale=1.0, output_type="latent",
                           latents=lat).frames
            init, prompts = pack_edit_input(inv_lat, case["source_prompt"], case["target_prompt"])
            pipe.scheduler = edit_sched
            edit_lat = pipe(prompt=prompts, height=HEIGHT, width=WIDTH, num_frames=n_frames,
                            num_inference_steps=args.step, guidance_scale=1.0, output_type="latent",
                            latents=init).frames
            video = decode_video(pipe, vae, edit_lat[:1, ...])
            n_out = save_frames(video, clip_dir(out_root, case))
            if n_out != C.expected_frames(n_frames):
                raise RuntimeError(f"wrote {n_out} frames, expected {C.expected_frames(n_frames)}")
        except Exception as e:  # noqa: BLE001 -- logged, counted, and the failure count exits non-zero
            failures += 1
            print(f"[r63_render] ({k + 1}/{len(cases)}) {cid}: FAILED {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        mf = out_root / f"_manifest_edit{case['edit_type']}.csv"
        new = not mf.exists()
        with mf.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["case_id", "n_frames", "seconds"])
            w.writerow([cid, n_out, f"{dt:.1f}"])
        print(f"[r63_render] ({k + 1}/{len(cases)}) {cid}: {n_out} frames, {dt:.1f}s")

    print(f"[r63_render] failures: {failures}")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
