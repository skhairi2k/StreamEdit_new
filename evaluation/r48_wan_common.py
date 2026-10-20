"""R48 shared Wan-Edit helpers: one model load per process, case lookup, frame dump.

Used by r48_wan_draft.py and r48_wan_render.py. Runs against ``wan-edit/`` unmodified except for the
opt-in ``route_start_map`` (R48). Everything here is host plumbing -- no R48 maths.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
WAN_DIR = REPO / "wan-edit"
FIVE = Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
CKPT = REPO / "Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B/"
SIZE = "832*480"
FRAME_NUM = 41
GRID = (30, 52)                       # DiT token grid per latent frame at 832x480


def load_cli():
    """``wan-edit/edit.py`` as a module (its CLI is guarded by __main__)."""
    sys.path.insert(0, str(WAN_DIR))
    spec = importlib.util.spec_from_file_location("wan_edit_cli", WAN_DIR / "edit.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def load_se_utils():
    """StreamEdit's pipeline/utils.py (torch/numpy/PIL only), for find_phrase_token_indices --
    the SAME phrase matcher both hosts use."""
    spec = importlib.util.spec_from_file_location(
        "se_utils", REPO / "Self-Forcing_StreamEdit/pipeline/utils.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_pipeline(cli):
    """(WanT2V, defaults) -- ``defaults`` are edit.py's own CLI defaults (guide scales, solver, shift)."""
    sys.argv = ["edit.py", "--task", "t2v-1.3B", "--size", SIZE, "--ckpt_dir", str(CKPT)]
    args = cli._parse_args()
    cfg = cli.WAN_CONFIGS["t2v-1.3B"]
    pipe = cli.wan.WanT2V(config=cfg, checkpoint_dir=str(CKPT), device_id=0, rank=0,
                          t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    return pipe, args


def load_cases(cases_path: str) -> List[Dict]:
    """Cases json (cases.json or r7_anchor_manifest.json: only video_name/edit_type are read) ->
    prompts and trigger words from edit{T}_FiVE.json, the same source r31_stage1.py uses."""
    cases = json.loads(Path(cases_path).read_text())
    by_type: Dict[int, Dict[str, dict]] = {}
    out = []
    for c in cases:
        t = int(c["edit_type"])
        if t not in by_type:
            by_type[t] = {e["video_name"]: e for e in json.loads(
                (FIVE / "edit_prompt" / f"edit{t}_FiVE.json").read_text())}
        e = by_type[t][c["video_name"]]
        out.append(dict(video_name=c["video_name"], edit_type=t,
                        source_prompt=e["source_prompt"], target_prompt=e["target_prompt"],
                        source_object=e["source_object"], target_object=e["target_object"]))
    return out


def save_frames(video: torch.Tensor, out_dir: Path) -> int:
    """[C, F, H, W] in [-1, 1] -> zero-padded PNGs (the ``edit_video`` layout evaluate.py reads)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = ((video.clamp(-1, 1).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    for i in range(arr.shape[1]):
        Image.fromarray(arr[:, i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
    return arr.shape[1]


def n_latent_frames(frame_num: int = FRAME_NUM) -> int:
    return (frame_num - 1) // 4 + 1


def source_frame_count(video_name: str) -> int:
    """Number of frames in the source mp4 (7 of the 100 benchmark videos have fewer than 41: 25..40)."""
    import cv2
    return int(cv2.VideoCapture(str(FIVE / "videos" / f"{video_name}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))


def expected_frames(n_source: int) -> int:
    """Frames Wan-Edit renders for a source of ``n_source`` frames: it reads min(41, n_source) and decodes 4*((n-1)//4)+1
    (a 41-frame clip -> 41, but 35 -> 33, 40 -> 37, 25 -> 25)."""
    n = min(FRAME_NUM, n_source)
    return 4 * ((n - 1) // 4) + 1
