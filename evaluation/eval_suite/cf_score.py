"""Content Fidelity (IVEBench, Chen et al., ICLR 2026) -- R57 unified metric suite.

A Qwen2.5-VL-72B-Instruct judge sees the source video, the edited video and the edit instruction and rates 1-5 whether
the edited video preserves everything except the edit itself (1 = a small part preserved ... 5 = identical in every
detail). It measures PRESERVATION ONLY: an unedited copy of the source scores 5, so CF is always read next to an
editability metric (EditReward, PSC, FiVE-Acc).

IVEBench's own code does the work, from the clone pinned by fetch_weights.sh: ``QwenVLContentFidelityEvaluator.
evaluate_video`` with its prompt, its video preparation (frames -> 25-fps MP4 -> at most 20 frames, longest side 426 px,
5 fps, via ffmpeg + libx264), its bundled qwen_vl_utils and the checkpoint's generation_config. Only the package
``__init__`` of ``fidelity`` is bypassed (a stub package with the real ``__path__``): it would import IVEBench's semantic
and motion fidelity code.

Run in five-bench (transformers 5.3.0), with two changes that restore what IVEBench saw on transformers 4.49:
  * the processor's call gets ``video_metadata`` with fps=2.0 and ``do_sample_frames=False``. 4.49's default fps is 2.0
    (second_per_grid_ts = 1.0, rope time interval 2); 5.3.0 assumes 24 fps, second_per_grid_ts becomes 0.083 and
    ``int()`` turns the rope time interval into 0, i.e. every frame of a video gets the same temporal position. Every
    call asserts second_per_grid_ts == 1.0;
  * ``use_fast=False`` for the image processor. CF feeds videos only and 5.3.0 has a single (fast) video processor, so
    this changes nothing here; the small numeric difference of that video processor versus 4.49's remains.
ffmpeg must have libx264 (IVEBench hard-codes it): the job puts imageio-ffmpeg's static build first on PATH; checked at
start.

Frames of both videos are first symlinked into temp directories as ``%05d.<ext>``, the pattern IVEBench's ffmpeg call
reads (renders are 0-indexed, FiVE sources 1-indexed; a gap in the numbering would otherwise cut the video short).

Columns: cf (1-5), cf_raw (the judge's full answer). IVEBench's parse failure (score -1) is stored as NaN with the raw
answer kept; exit 1 when any pair is missing or NaN (the CSV is still written).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite_common as sc  # noqa: E402

METRICS = ("cf",)
EXTS = (".jpg", ".jpeg", ".png")


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("no ffmpeg on PATH")
    enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if " libx264 " not in enc:
        raise RuntimeError(f"{exe} has no libx264 encoder (IVEBench's video preparation needs it); put imageio-ffmpeg's "
                           "static binary first on PATH")
    return exe


def link_frames(frames, dest: str) -> str:
    """Symlink frames as dest/00001.<ext>, 00002.<ext>, ... (one extension per video)."""
    exts = {os.path.splitext(f)[1].lower() for f in frames}
    if len(exts) != 1:
        raise ValueError(f"mixed frame extensions {sorted(exts)}")
    os.makedirs(dest, exist_ok=True)
    ext = exts.pop()
    for i, f in enumerate(frames, start=1):
        os.symlink(os.path.abspath(f), os.path.join(dest, f"{i:05d}{ext}"))
    return dest


def make_evaluator(ivebench_dir: Path, model_path: Path):
    metrics_dir = ivebench_dir / "metrics"
    sys.path.insert(0, str(metrics_dir))
    stub = types.ModuleType("fidelity")
    stub.__path__ = [str(metrics_dir / "fidelity")]
    sys.modules["fidelity"] = stub
    from fidelity.content_fidelity import QwenVLContentFidelityEvaluator as Base
    from transformers.video_utils import VideoMetadata

    class FpsFixedProcessor:
        """Delegates to the real processor; every call with videos carries fps=2.0 metadata and is checked."""

        def __init__(self, processor):
            self._p = processor

        def __call__(self, *args, videos=None, **kw):
            if videos:
                kw["video_metadata"] = [VideoMetadata(total_num_frames=len(v), fps=2.0) for v in videos]
                kw["do_sample_frames"] = False
            out = self._p(*args, videos=videos, **kw)
            if videos:
                s = out["second_per_grid_ts"].tolist()
                if any(abs(x - 1.0) > 1e-6 for x in s):
                    raise RuntimeError(f"second_per_grid_ts {s} != 1.0 (IVEBench's temporal positions)")
            return out

        def __getattr__(self, name):
            return getattr(self._p, name)

    class Evaluator(Base):
        def _load_model(self):
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from fidelity.qwen_vl_utils import process_vision_info
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, dtype="auto",
                                                                            device_map="auto")
            self.processor = FpsFixedProcessor(AutoProcessor.from_pretrained(self.model_path, use_fast=False))
            self.process_vision_info = process_vision_info
            print(f"[cf] judge loaded from {self.model_path}; device map spans "
                  f"{sorted(set(map(str, getattr(self.model, 'hf_device_map', {}).values())))}", flush=True)

    return Evaluator(str(model_path))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(p)
    p.add_argument("--ivebench_dir", default="~/Code/IVEBench")
    p.add_argument("--model_path", default="~/Data/weights/Qwen2.5-VL-72B-Instruct")
    a = p.parse_args()

    import torch
    import transformers
    print(f"[cf] transformers {transformers.__version__}, torch {torch.__version__}, gpus {torch.cuda.device_count()}, "
          f"ffmpeg {check_ffmpeg()}", flush=True)

    pairs = sc.load_pairs(a.edit_json_dir, a.cases)
    store = sc.ResultStore(a.out, ("edit_type", "video_name", "cf_raw"), METRICS)

    todo = [q for q in pairs if not store.done(q)]
    print(f"[cf] {len(pairs)} pairs, {len(pairs) - len(todo)} already scored (resume), {len(todo)} to do", flush=True)
    if todo:
        ev = make_evaluator(Path(a.ivebench_dir).expanduser(), Path(a.model_path).expanduser())
        for q in todo:
            t = int(q["edit_type"])
            name = f"edit{t}/{q['video_name']}"
            t0 = time.time()
            row = {"edit_type": t, "video_name": q["video_name"], "cf_raw": "", "cf": float("nan")}
            try:
                src = sc.list_images(sc.source_dir(a.data_root, str(q["video_name"])))
                tgt = sc.ensure_render_frames(sc.render_dir(a.root, a.layout, t, str(q["video_name"])))
                if not src or not tgt:
                    raise FileNotFoundError(f"{len(src)} source / {len(tgt)} render frames")
                with tempfile.TemporaryDirectory(prefix="cf_frames_") as tmp:
                    score, answer = ev.evaluate_video(link_frames(src, os.path.join(tmp, "source")),
                                                      link_frames(tgt, os.path.join(tmp, "edited")),
                                                      str(q["instruction"]))
                row["cf_raw"] = " ".join(str(answer).split())
                if score == -1.0:
                    raise ValueError(f"unparseable judge answer: {row['cf_raw'][:200]}")
                row["cf"] = float(score)
                print(f"[cf] {name:<32} cf={score:.0f} ({time.time() - t0:.1f}s) :: {row['cf_raw'][:160]}", flush=True)
            except Exception as e:  # noqa: BLE001 -- recorded as NaN, makes the run exit 1, never swallowed silently
                print(f"Error: cf {name}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            store.add(row)
    return sc.finish(store, len(pairs), "cf")


if __name__ == "__main__":
    sys.exit(main())
