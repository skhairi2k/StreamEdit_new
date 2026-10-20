"""R49 side experiment: the Wan-Edit draft (17-step grid, shift 5, seeded noise, text only) started at a different step.

The R49 draft starts at step 3 (sigma 0.958, 14 active steps) -- the same noise level as skip 9 on the 50-step grid.
Skip 5 on the 50-step grid is sigma 0.978; the nearest start on the 17-step grid is step 2 (sigma 0.974, 15 active steps).
This script renders ONLY the draft video (no attention recording) with a chosen --skip, with every other setting identical
to r48_wan_draft.py (the recorder is read-only, so frames match), into
  {out_frames}/skip{S}/edit{T}/{video}/00000.png ...
usage: python evaluation/r49_wan_draft_skip.py --skip 2 --cases evaluation/r49_cases.json --out_frames DIR
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r48_wan_common as C  # noqa: E402
import r48_wan_sched as SCH  # noqa: E402

SAMPLE_STEPS, SHIFT = 17, 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out_frames", required=True)
    a = ap.parse_args()
    out_root = Path(a.out_frames).expanduser() / f"skip{a.skip}"
    cases = C.load_cases(a.cases)
    cli = C.load_cli()
    pipe, d = C.build_pipeline(cli)
    print(f"[r49_draft_skip] 17-step grid skip {a.skip}: sigma {SCH.sigma_at(a.skip, SAMPLE_STEPS):.4f}, "
          f"{SAMPLE_STEPS - a.skip} active steps, {len(cases)} clips -> {out_root}", flush=True)
    n_fail = 0
    for c in cases:
        vid, t = c["video_name"], c["edit_type"]
        out_dir = out_root / f"edit{t}" / vid
        want = C.expected_frames(C.source_frame_count(vid))
        if out_dir.is_dir() and len(list(out_dir.glob("*.png"))) == want:
            continue
        try:
            t0 = time.time()
            video = cli.load_frames(str(C.FIVE / "videos" / f"{vid}.mp4"))
            out = pipe.edit(
                video, c["source_prompt"], c["target_prompt"], size=cli.SIZE_CONFIGS[C.SIZE],
                frame_num=min(C.FRAME_NUM, video.shape[2]), shift=SHIFT, sample_solver=d.sample_solver,
                sampling_steps=SAMPLE_STEPS, guide_scale=d.sample_guide_scale,
                tgt_guide_scale=d.tgt_guide_scale, skip_timesteps=a.skip, n_avg=1, seed_noise=True,
                seed=0, offload_model=False)
            nf = C.save_frames(out, out_dir)
            assert nf == want, f"{nf} frames written, expected {want}"
            print(f"[r49_draft_skip] ok {t}/{vid}: {nf} frames, {time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r49_draft_skip] FAILED {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    print(f"[r49_draft_skip] failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
