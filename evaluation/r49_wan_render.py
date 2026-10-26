"""R49 stage 3 (Wan-Edit): one render config over a case list.

Configs (ONE per invocation; the sbatch array maps array index -> config):
  max_s9 / max_s5          adaptive: sample_steps 50, loop from step 9 / 5 (= s_min), per-token start step from
                           {field_root}/wan_max_s{9,5}/edit{T}/{video}.npy (written by r49_field.py); s = 50 = never edited
  matched59_s{0.95,0.92,0.85,0.978}  global start sigma with sample_steps chosen so exactly 59 steps are active (the max_s5
                           budget: 14 draft + 45 final), N/skip = r48_wan_sched.matched_n(sg, 59) = 75/16, 85/26, 111/52, 66/7
                           (0.978 is appended last: indices 0-2 are the original curve; its real start sigma is 0.9766)
Text only, seeded noise (base_seed 0), n_avg 1, up to 41 frames 832x480, edit.py's own guide scales -- the same settings as
r48_wan_render.py. The model loads once per invocation; clips whose expected number of PNGs already exists are skipped.

--mode subset (default) = the 18 clips, --mode full = the 419 pairs (R49 full stage; --cases evaluation/r7_anchor_manifest.json, adaptive
start maps from --field_root $D/r49_field_full).
Output: {out_root}/{config}_{mode}/edit{T}/{video}/00000.png ...   (the ``edit_video`` layout evaluate.py reads)
usage: python evaluation/r49_wan_render.py --config max_s9 --cases evaluation/r48_cases.json [--mode subset|full]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_wan_common as C  # noqa: E402
import r48_wan_sched as SCH  # noqa: E402

D = Path.home() / "Data/dataggen/outputs/five_bench"
ADAPT = {"max_s9": 9, "max_s5": 5, "nosupp_s5": 5}      # arm -> loop start (= the arm's s_min); nosupp_s5 = R51's A4
A_ACTIVE_59 = 59                                         # max_s5's worst case: 14 draft + 45 final active steps
MATCHED59 = (0.95, 0.92, 0.85, 0.978)                 # 0.978 appended: subset indices 0-2 are unchanged


def matched59_configs() -> dict:
    out = {}
    for sg in MATCHED59:
        n, skip = SCH.matched_n(sg, A_ACTIVE_59)
        real = float(SCH.real_sigmas(n)[skip])
        assert n - skip == A_ACTIVE_59 and abs(real - sg) < 0.015, f"matched59_s{sg:g}: N {n} skip {skip} starts at {real:.4f}"
        out[f"matched59_s{sg:g}"] = (n, skip)
    return out


def resolve(config: str):
    """-> (kind, sample_steps, skip_timesteps, arm|None)."""
    if config in ADAPT:
        return "adapt", SCH.N_NATIVE, ADAPT[config], config
    m59 = matched59_configs()
    if config in m59:
        n, skip = m59[config]
        return "global", n, skip, None
    raise SystemExit(f"[r49_wan_render] unknown config {config!r}; adaptive: {list(ADAPT)}; matched-59: {list(m59)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--anno_dir", default=None,
                    help="annotation folder with edit{T}_FiVE.json (default: the released FiVE edit_prompt)")
    ap.add_argument("--field_root", default=str(D / "r49_field"))
    ap.add_argument("--out_root", default=str(D / "r49_wan_arms"))
    ap.add_argument("--mode", choices=("subset", "full"), default="subset")
    a = ap.parse_args()

    kind, n_steps, skip, arm = resolve(a.config)
    out_base = Path(a.out_root).expanduser() / f"{a.config}_{a.mode}"
    cases = C.load_cases(a.cases, a.anno_dir)
    cli = C.load_cli()
    pipe, d = C.build_pipeline(cli)
    print(f"[r49_wan_render] config={a.config} ({kind}) sample_steps={n_steps} skip={skip} "
          f"active={n_steps - skip} mode={a.mode} cases={len(cases)} anno={a.anno_dir or 'released FiVE'} -> {out_base}", flush=True)
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
            print(f"[r49_wan_render] ok {t}/{vid}: {nf} frames, {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r49_wan_render] FAILED {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    print(f"[r49_wan_render] {a.config}: {n_done}/{len(cases)} clips  failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
