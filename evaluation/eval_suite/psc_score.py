"""Phrase Semantic Consistency (IVEBench, Chen et al., ICLR 2026) with VideoCLIP-XL-v2 -- R57 unified metric suite.

Runs in five-bench. IVEBench's own code does the work: its ``VideoCLIPEvaluator.compute_similarity(frames_dir, phrase)``
(8 frames evenly spaced over the whole clip, 224x224, ImageNet normalisation, VideoCLIP-XL-v2 video-text cosine) is used
unchanged from the clone pinned by fetch_weights.sh. Only the ``compliance`` package's ``__init__`` is bypassed (a stub
package with the real ``__path__``), because it imports IVEBench's other metrics (GroundingDINO, ...).

Frames: IVEBench's sampling on the clip's full frame list, not the harness's every-8th-frame list -- VideoCLIP-XL-v2 is
validated on 8 frames. Both videos are scored from their frame directories (source: FiVE ``images/<video>``).

Per pair, with ``phrase`` = target_object (edit types 1-5) or source_object (edit type 6, which has no target phrase):
  s_edit = s(V_edit, phrase)        s_src = s(V_src, phrase)
  types 1-5 : psc = s_edit                                     psc_rule = target
  type 6    : psc = s_src - s_edit   (drop of the removed object's score versus the source video; ~0 when nothing was
              removed)                                          psc_rule = removal_drop
The type-6 score is never averaged with types 1-5 (merge_suite.py reports it as psc_removal).

Exit 1 when any pair is missing or NaN (the CSV is still written).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite_common as sc  # noqa: E402

METRICS = ("psc", "s_edit", "s_src")


def load_evaluator(ivebench_dir: Path, model_file: Path, device: str):
    """IVEBench's VideoCLIPEvaluator without running compliance/__init__.py."""
    metrics_dir = ivebench_dir / "metrics"
    sys.path.insert(0, str(metrics_dir))
    stub = types.ModuleType("compliance")
    stub.__path__ = [str(metrics_dir / "compliance")]
    sys.modules["compliance"] = stub
    from compliance.phrase_semantic_consistency import VideoCLIPEvaluator
    return VideoCLIPEvaluator(str(model_file), device)


def phrase_of(pair) -> str:
    phrase = str(pair["source_object"] if int(pair["edit_type"]) == 6 else pair["target_object"]).strip()
    if not phrase:
        raise ValueError(f"edit{pair['edit_type']}/{pair['video_name']}: empty "
                         f"{'source' if int(pair['edit_type']) == 6 else 'target'}_object")
    return phrase


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(p)
    p.add_argument("--ivebench_dir", default="~/Code/IVEBench")
    p.add_argument("--model_file", default="~/Data/weights/VideoCLIP-XL-v2/VideoCLIP-XL-v2.bin")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    pairs = sc.load_pairs(a.edit_json_dir, a.cases)
    store = sc.ResultStore(a.out, ("edit_type", "video_name", "psc_rule"), METRICS)
    todo = [q for q in pairs if not store.done(q)]
    print(f"[psc] {len(pairs)} pairs, {len(pairs) - len(todo)} already scored (resume), {len(todo)} to do", flush=True)
    if todo:
        ev = load_evaluator(Path(a.ivebench_dir).expanduser(), Path(a.model_file).expanduser(), a.device)
        for q in todo:
            t = int(q["edit_type"])
            name = f"edit{t}/{q['video_name']}"
            t0 = time.time()
            row = {"edit_type": t, "video_name": q["video_name"],
                   "psc_rule": "removal_drop" if t == 6 else "target", **{m: float("nan") for m in METRICS}}
            try:
                phrase = phrase_of(q)
                edit_dir = sc.render_dir(a.root, a.layout, t, str(q["video_name"]))
                if not sc.ensure_render_frames(edit_dir):
                    raise FileNotFoundError(f"no render frames under {edit_dir}")
                s_edit = ev.compute_similarity(edit_dir, phrase)
                s_src = ev.compute_similarity(sc.source_dir(a.data_root, str(q["video_name"])), phrase)
                row.update(s_edit=s_edit, s_src=s_src, psc=(s_src - s_edit) if t == 6 else s_edit)
                print(f"[psc] {name:<32} {row['psc_rule']:<12} phrase='{phrase}' s_edit={s_edit:+.4f} "
                      f"s_src={s_src:+.4f} psc={row['psc']:+.4f} ({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001 -- recorded as NaN, makes the run exit 1, never swallowed silently
                print(f"Error: psc {name}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            store.add(row)
    return sc.finish(store, len(pairs), "psc")


if __name__ == "__main__":
    sys.exit(main())
