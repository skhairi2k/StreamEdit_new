"""R64 reproduction check: FlowDirector's own README demos, rendered by upstream edit.py and by our driver.

Demos = the 4 ``examples_to_load_definitions`` of ~/Code/FlowDirector/app.py (the only demos with full prompts), read
verbatim; reference = the README GIF of the same edit. Everything runs with upstream's seed 42.

Modes (run in this order by slurm_scripts/five_bench/r64_repro_demo.sh):
  --upstream  upstream edit.py as a subprocess (cwd ~/Code/FlowDirector), released flags, only --ckpt_dir / --save_file
              adapted                                                        -> {out_root}/upstream/{demo}.mp4
  --driver    the same 4 demos through r64_flowdirector_render.py's functions (frame loader, trigger resolution, the
              same WanT2V.edit call), saved with upstream's cache_video exactly as edit.py does -> {out_root}/driver/{demo}.mp4
  --compare   CPU: driver vs upstream (identical?), upstream vs GIF and source vs GIF (PSNR / SSIM, the latter as a scale)
              -> evaluation/csv/r64_repro_demo.csv + evaluation/figures/r64_repro_demo_{demo}.jpg

Reading the result: driver == upstream is conclusive about our driver; upstream ~ GIF is conclusive about the env; a clear
mismatch is not (the GIFs were committed on 2025-05-30, a week before the code, from the authors' internal code/settings).

usage: python evaluation/r64_repro_demo.py --upstream | --driver | --compare
"""
from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import sys
import time
import types
from pathlib import Path

import r64_flowdirector_render as R          # puts ~/Code/FlowDirector and evaluation/ on sys.path
import r48_wan_common as C

HERE = Path(__file__).resolve().parent
OUT_ROOT = Path.home() / "Data/dataggen/outputs/five_bench/r64_flowdirector/repro_demo"
UPSTREAM_SEED = 42                            # the released script's --base_seed
FRAME_NUM = 41                                # every video_list/*.mp4 has exactly 41 frames at 832x480
# README demo tables: the GIF showing the same edit as each app.py example
GIF = {"bear_g": "bear_g_03.gif", "blackswan": "blackswan_02.gif", "jeep": "jeep_01.gif", "woman": "woman_02.gif"}
SHEET_FRAMES = (0, 20, 40)


def demos() -> list[dict]:
    tree = ast.parse((R.FD_DIR / "app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "examples_to_load_definitions" for t in node.targets):
            out = ast.literal_eval(node.value)
            assert {d["video_base_name"] for d in out} == set(GIF), f"app.py demos changed: {[d['video_base_name'] for d in out]}"
            return out
    raise SystemExit("[r64_repro_demo] examples_to_load_definitions not found in app.py")


def video_path(d: dict) -> Path:
    return R.FD_DIR / "video_list" / f"{d['video_base_name']}.mp4"


def flags42() -> list[str]:
    flags = R.released_flags()
    flags[flags.index("--base_seed") + 1] = str(UPSTREAM_SEED)
    return flags


def run_upstream(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for d in demos():
        name, dst = d["video_base_name"], out / f"{d['video_base_name']}.mp4"
        if dst.exists():
            print(f"[r64_repro_demo] upstream {name}: exists, skipped", flush=True)
            continue
        cmd = [sys.executable, "edit.py", *flags42(), "--ckpt_dir", str(C.CKPT),
               "--source_video_path", str(video_path(d)), "--source_prompt", d["src_prompt"], "--source_words", d["src_words"],
               "--prompt", d["tar_prompt"], "--target_words", d["tar_words"], "--save_file", str(dst)]
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=R.FD_DIR).returncode
        if rc != 0 or not dst.exists():
            n_fail += 1
            print(f"[r64_repro_demo] FAILED upstream {name}: exit {rc}, mp4 {'present' if dst.exists() else 'missing'}", flush=True)
        else:
            print(f"[r64_repro_demo] ok upstream {name}: {time.time() - t0:.0f}s", flush=True)
    return n_fail


def run_driver(out: Path) -> int:
    import torch
    out.mkdir(parents=True, exist_ok=True)
    cli, args = R.load_cli()
    pipe = cli.wan.WanT2V(config=cli.WAN_CONFIGS[args.task], checkpoint_dir=str(C.CKPT), device_id=0, rank=0,
                          t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    n_fail = 0
    for d in demos():
        name, dst = d["video_base_name"], out / f"{d['video_base_name']}.mp4"
        if dst.exists():
            print(f"[r64_repro_demo] driver {name}: exists, skipped", flush=True)
            continue
        try:
            t0 = time.time()
            sw, ss = R.resolve_words(d["src_prompt"], d["src_words"])
            tw, ts = R.resolve_words(d["tar_prompt"], d["tar_words"])
            pipe.load_video_frames = types.MethodType(R.first_frames_loader(FRAME_NUM), pipe)
            video = pipe.edit(
                d["tar_prompt"], size=cli.SIZE_CONFIGS[args.size], frame_num=FRAME_NUM, shift=args.sample_shift,
                sample_solver=args.sample_solver, sampling_steps=args.sample_steps, guide_scale=args.sample_guide_scale,
                tar_guide_scale=args.tar_guide_scale, seed=UPSTREAM_SEED, offload_model=False,
                source_video_path=str(video_path(d)), source_prompt=d["src_prompt"],
                nmax_step=args.n_max, nmin_step=args.n_min, n_avg=args.n_avg, worse_avg=args.worse_avg, omega=args.omega,
                source_words=sw, target_words=tw, window_size=args.window_size, decay_factor=args.decay_factor)
            if video is None:
                raise RuntimeError("WanT2V.edit returned None")
            cli.cache_video(tensor=video[None], save_file=str(dst), fps=cli.WAN_CONFIGS[args.task].sample_fps,
                            nrow=1, normalize=True, value_range=(-1, 1))           # edit.py's own save call
            if not dst.exists():
                raise RuntimeError("cache_video wrote no file")
            print(f"[r64_repro_demo] ok driver {name}: src={sw!r} ({ss}), tgt={tw!r} ({ts}), {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- report per demo, keep going
            n_fail += 1
            print(f"[r64_repro_demo] FAILED driver {name}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    return n_fail


def read_mp4(p: Path):
    import cv2
    import numpy as np
    cap, frames = cv2.VideoCapture(str(p)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def read_gif(p: Path):
    import numpy as np
    from PIL import Image, ImageSequence
    return np.stack([np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(Image.open(p))])


def mean_psnr_ssim(a, b) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    n = min(len(a), len(b))
    ps = [peak_signal_noise_ratio(a[i], b[i], data_range=255) for i in range(n)]
    ss = [structural_similarity(a[i], b[i], channel_axis=2, data_range=255) for i in range(n)]
    return sum(ps) / n, sum(ss) / n


def contact_sheet(rows: list[tuple[str, object]], title: str, dst: Path) -> None:
    from PIL import Image, ImageDraw
    w, h, top = 416, 240, 20
    sheet = Image.new("RGB", (w * (len(SHEET_FRAMES) + 1) - w // 2, top + h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title, fill="black")
    for r, (label, frames) in enumerate(rows):
        draw.text((4, top + r * h + h // 2), label, fill="black")
        for c, i in enumerate(SHEET_FRAMES):
            sheet.paste(Image.fromarray(frames[min(i, len(frames) - 1)]).resize((w, h)), (w // 2 + c * w, top + r * h))
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, quality=88)


def run_compare(out_root: Path, out_csv: Path, fig_dir: Path) -> int:
    import numpy as np
    rows, n_fail = [], 0
    for d in demos():
        name = d["video_base_name"]
        try:
            src = read_mp4(video_path(d))
            gif = read_gif(R.FD_DIR / "assets/demos" / GIF[name])
            up = read_mp4(out_root / "upstream" / f"{name}.mp4")
            drv = read_mp4(out_root / "driver" / f"{name}.mp4")
            same_shape = up.shape == drv.shape
            max_abs = int(np.abs(up.astype(np.int16) - drv.astype(np.int16)).max()) if same_shape else -1
            p_up, s_up = mean_psnr_ssim(up, gif)
            p_src, s_src = mean_psnr_ssim(src, gif)
            rows.append(dict(demo=name, source_words=d["src_words"], target_words=d["tar_words"], gif=GIF[name],
                             frames=f"src {len(src)} gif {len(gif)} up {len(up)} drv {len(drv)}",
                             driver_vs_upstream_max_abs=max_abs, identical=same_shape and max_abs == 0,
                             psnr_up_gif=round(p_up, 2), ssim_up_gif=round(s_up, 4),
                             psnr_src_gif=round(p_src, 2), ssim_src_gif=round(s_src, 4)))
            contact_sheet([("source", src), ("README GIF", gif), ("upstream", up), ("driver", drv)],
                          f"{name}: '{d['src_words']}' -> '{d['tar_words']}'  frames {SHEET_FRAMES}  (seed {UPSTREAM_SEED})",
                          fig_dir / f"r64_repro_demo_{name}.jpg")
            print(f"[r64_repro_demo] {name}: driver==upstream {rows[-1]['identical']} (max|diff| {max_abs}), "
                  f"PSNR/SSIM up-vs-GIF {p_up:.2f}/{s_up:.3f}, src-vs-GIF {p_src:.2f}/{s_src:.3f}", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- report per demo, keep going
            n_fail += 1
            print(f"[r64_repro_demo] FAILED compare {name}: {type(e).__name__}: {e}", flush=True)
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[r64_repro_demo] {len(rows)} demos -> {out_csv}")
    return n_fail


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--upstream", action="store_true")
    mode.add_argument("--driver", action="store_true")
    mode.add_argument("--compare", action="store_true")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--out_csv", default=str(HERE / "csv/r64_repro_demo.csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    a = ap.parse_args()
    out_root = Path(a.out_root).expanduser()
    if a.upstream:
        n_fail = run_upstream(out_root / "upstream")
    elif a.driver:
        n_fail = run_driver(out_root / "driver")
    else:
        n_fail = run_compare(out_root, Path(a.out_csv), Path(a.fig_dir))
    print(f"[r64_repro_demo] failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
