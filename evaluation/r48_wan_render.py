"""R48 stage 3 (Wan-Edit): one render config over a case list.

Configs (ids from r48_wan_sched.py; ONE config per invocation, the sbatch array maps array index -> config):
  adapt_{arm}      the adaptive render: sample_steps 50, loop from step 9, per-token start step from the field
                   ({field_root}/wan_{arm}/edit{T}/{video}.npy, written by r48_field.py) via --route_start_map
  native_s{sg}     global start sigma sg on Wan-Edit's own 50-step schedule (the stock behaviour at skip = {10,15,...})
  matched_s{sg}    global start sigma sg with sample_steps chosen so exactly 55 steps are active (same compute as adaptive)
Text only, seeded noise (base_seed 0), n_avg 1, up to 41 frames 832x480 (a clip with fewer source frames renders 4*((n-1)//4)+1 of them), edit.py's own guide scales -- the same settings as the
draft. The model loads once per invocation; clips whose expected number of PNGs already exists are skipped (resumable).

Output: {out_root}/{config}_{mode}/edit{T}/{video}/00000.png ...   (the ``edit_video`` layout evaluate.py reads)
usage: python evaluation/r48_wan_render.py --config native_s0.92 --mode subset --cases evaluation/r48_cases.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_wan_common as C  # noqa: E402
import r48_wan_sched as SCH  # noqa: E402

D = Path.home() / "Data/dataggen/outputs/five_bench"


def resolve(config: str):
    """-> (kind, sample_steps, skip_timesteps, arm|None)."""
    if config.startswith("adapt_"):
        arm = config[len("adapt_"):]
        if arm not in SCH.ARMS:
            raise SystemExit(f"[r48_wan_render] unknown arm {arm!r}; arms: {SCH.ARMS}")
        return "adapt", SCH.N_NATIVE, SCH.ADAPT_SKIP, arm
    cfg = {**SCH.configs(), **SCH.explore_configs()}         # explore = R48 v2a global starts above step 9
    if config not in cfg:
        raise SystemExit(f"[r48_wan_render] unknown config {config!r}; baseline/explore configs: {list(cfg)}; adapt_<arm>")
    n, skip = cfg[config]
    return "global", n, skip, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=("subset", "full"), required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--field_root", default=str(D / "r48_field"))
    ap.add_argument("--out_root", default=str(D / "r48_wan_arms"))
    a = ap.parse_args()

    SCH.selftest(verbose=False)                      # the tables are checked against the real scheduler on every run
    kind, n_steps, skip, arm = resolve(a.config)
    # Every config, baseline or adaptive, has its own directory per mode: the full run re-renders the 18 subset clips rather
    # than reuse them, so every clip of the full benchmark comes from one job with one complete guard.
    out_base = Path(a.out_root).expanduser() / f"{a.config}_{a.mode}"
    cases = C.load_cases(a.cases)
    cli = C.load_cli()
    pipe, d = C.build_pipeline(cli)
    print(f"[r48_wan_render] config={a.config} ({kind}) sample_steps={n_steps} skip={skip} "
          f"active={n_steps - skip} mode={a.mode} cases={len(cases)} -> {out_base}", flush=True)
    n_fail = n_done = 0
    for c in cases:
        vid, t = c["video_name"], c["edit_type"]
        out_dir = out_base / f"edit{t}" / vid
        want = C.expected_frames(C.source_frame_count(vid))     # 41 for most clips, fewer for the 7 short videos
        if out_dir.is_dir() and len(list(out_dir.glob("*.png"))) == want:
            n_done += 1
            continue
        try:
            t0 = time.time()
            video = cli.load_frames(str(C.FIVE / "videos" / f"{vid}.mp4"))
            kw = {}
            if kind == "adapt":
                m = Path(a.field_root).expanduser() / f"wan_{arm}" / f"edit{t}" / f"{vid}.npy"
                kw["route_start_map"] = cli.load_route_start_map(str(m), 0, skip)   # refuses a map starting before `skip`
            out = pipe.edit(
                video, c["source_prompt"], c["target_prompt"], size=cli.SIZE_CONFIGS[C.SIZE],
                frame_num=min(C.FRAME_NUM, video.shape[2]), shift=SCH.SHIFT, sample_solver=d.sample_solver,
                sampling_steps=n_steps, guide_scale=d.sample_guide_scale, tgt_guide_scale=d.tgt_guide_scale,
                skip_timesteps=skip, n_avg=1, seed_noise=True, seed=0, offload_model=False, **kw)
            nf = C.save_frames(out, out_dir)
            assert nf == want, f"{nf} frames written, expected {want}"
            n_done += 1
            print(f"[r48_wan_render] ok {t}/{vid}: {nf} frames, {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r48_wan_render] FAILED {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    print(f"[r48_wan_render] {a.config}: {n_done}/{len(cases)} clips  failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
