"""EditReward (Wu et al., ICLR 2026) on the frames the FiVE harness scores -- R57 unified metric suite.

Run with the ``editreward`` env's python (transformers 4.57.0, trl 0.8.6), NOT five-bench: EditReward does not import
under transformers 5.x (see the R57 plan).

Per pair: for every strided frame t of the clip, EditReward scores (source frame t, instruction, edited frame t) with the
MiMo-VL-7B checkpoint; the clip score is the mean over frames, as FiVE averages CLIP over the sampled frames. The
edited frame is resized to the source frame's size first, like the harness does.

Columns
  editreward     the repo's reported score: ``pooling_strategy: mean`` of the config = mean over the two heads of
                 logits[:, 0] (what ``EditRewardInferencer.reward(...)[i][0]`` returns)
  editreward_if  head 1 alone, instruction following
  editreward_vq  head 2 alone, visual quality
No fixed range: compare across methods only. The heads come from one forward pass with ``pooling_strategy = None``; the
fused score is recomputed from them and cross-checked once against the repo's own fused output (hard error on mismatch).

The inferencer builds the model and processor from ``XiaomiMiMo/MiMo-VL-7B-SFT-2508`` (the config's model_name_or_path)
and loads the EditReward checkpoint over it; offline that name is replaced by the local base-model directory in a copy of
the config.

Exit 1 when any pair is missing or NaN (the CSV is still written).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Sequence

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite_common as sc  # noqa: E402

CONFIG_REL = "EditReward/config/EditReward-MiMo-VL-7B-SFT-2508.yaml"
METRICS = ("editreward", "editreward_if", "editreward_vq")


def build_inferencer(editreward_dir: Path, checkpoint: Path, base_model: Path, device: str):
    """EditRewardInferencer on the MiMo config, base model path rewritten to the local copy."""
    cfg_src = editreward_dir / CONFIG_REL
    cfg = yaml.safe_load(cfg_src.read_text())
    if cfg.get("pooling_strategy") != "mean" or cfg.get("rm_head_type") != "ranknet_multi_head":
        raise RuntimeError(f"unexpected EditReward config {cfg_src}: pooling_strategy={cfg.get('pooling_strategy')}, "
                           f"rm_head_type={cfg.get('rm_head_type')} (the fused score below assumes mean over 2 heads)")
    cfg["model_name_or_path"] = str(base_model)
    tmp_cfg = Path(tempfile.mkdtemp(prefix="editreward_cfg_")) / cfg_src.name
    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    sys.path.insert(0, str(editreward_dir))
    cwd = os.getcwd()
    os.chdir(editreward_dir)           # the config carries repo-relative paths (e.g. the deepspeed json)
    try:
        from EditReward import EditRewardInferencer
        return EditRewardInferencer(config_path=str(tmp_cfg), checkpoint_path=str(checkpoint), device=device,
                                    reward_dim="overall_detail", rm_head_type="ranknet_multi_head")
    finally:
        os.chdir(cwd)


@torch.no_grad()
def head_scores(inf, instruction: str, src: Sequence[str], tgt: Sequence[str], check_fused: bool = False):
    """(head1, head2) logits[:, 0] per frame, each a list of floats; optional cross-check against the fused output."""
    batch = inf.prepare_batch(list(src), list(tgt), [instruction] * len(src))
    model = inf.model
    saved = model.pooling_strategy
    try:
        model.pooling_strategy = None
        heads = model(return_dict=True, **batch)
        if not isinstance(heads, (list, tuple)) or len(heads) != 2:
            raise RuntimeError(f"expected 2 per-head logits with pooling_strategy=None, got {type(heads)}")
        h1, h2 = (h[:, 0].float().cpu() for h in heads)
        if check_fused:
            model.pooling_strategy = saved
            fused = model(return_dict=True, **batch)["logits"][:, 0].float().cpu()
            if not torch.allclose(fused, (h1 + h2) / 2, rtol=2e-2, atol=2e-2):
                raise RuntimeError(f"fused score != mean of heads: {fused.tolist()} vs {((h1 + h2) / 2).tolist()}")
            print(f"[editreward] fused-score cross-check ok ({len(src)} frames)", flush=True)
    finally:
        model.pooling_strategy = saved
    return h1.tolist(), h2.tolist()


def mean(x: List[float]) -> float:
    return sum(x) / len(x)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(p)
    p.add_argument("--frame_stride", type=int, default=8)
    p.add_argument("--editreward_dir", default="~/Code/EditReward")
    p.add_argument("--weights_dir", default="~/Data/weights")
    p.add_argument("--batch_size", type=int, default=8, help="frames per forward pass")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    import transformers
    print(f"[editreward] transformers {transformers.__version__}, torch {torch.__version__}", flush=True)
    if transformers.__version__ != "4.57.0":
        raise RuntimeError(f"EditReward needs transformers 4.57.0, this env has {transformers.__version__} "
                           "(run it with the `editreward` env's python)")

    pairs = sc.load_pairs(a.edit_json_dir, a.cases)
    store = sc.ResultStore(a.out, ("edit_type", "video_name", "n_frames"), METRICS)
    todo = [q for q in pairs if not store.done(q)]
    print(f"[editreward] {len(pairs)} pairs, {len(pairs) - len(todo)} already scored (resume), {len(todo)} to do", flush=True)
    if todo:
        w = Path(a.weights_dir).expanduser()
        inf = build_inferencer(Path(a.editreward_dir).expanduser(), w / "EditReward-MiMo-VL-7B-SFT-2508",
                               w / "MiMo-VL-7B-SFT-2508", a.device)
        checked = False
        for q in todo:
            name = f"edit{q['edit_type']}/{q['video_name']}"
            t0 = time.time()
            row = {"edit_type": q["edit_type"], "video_name": q["video_name"], "n_frames": 0,
                   **{m: float("nan") for m in METRICS}}
            try:
                src, tgt = sc.paired_frames(a.data_root, a.root, a.layout, q, a.frame_stride)
                with tempfile.TemporaryDirectory(prefix="editreward_frames_") as tmp:
                    resized = []
                    for i, (s, t) in enumerate(zip(src, tgt)):
                        size = Image.open(s).size
                        out = os.path.join(tmp, f"{i:05d}.png")
                        Image.open(t).convert("RGB").resize(size).save(out)
                        resized.append(out)
                    h1: List[float] = []
                    h2: List[float] = []
                    for i in range(0, len(src), a.batch_size):
                        b1, b2 = head_scores(inf, str(q["instruction"]), src[i:i + a.batch_size],
                                             resized[i:i + a.batch_size], check_fused=not checked)
                        checked = True
                        h1 += b1
                        h2 += b2
                row.update(n_frames=len(src), editreward_if=mean(h1), editreward_vq=mean(h2),
                           editreward=mean([(x + y) / 2 for x, y in zip(h1, h2)]))
                print(f"[editreward] {name:<32} n={len(src):<3} reward={row['editreward']:+.4f} "
                      f"if={row['editreward_if']:+.4f} vq={row['editreward_vq']:+.4f} ({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001 -- recorded as NaN, makes the run exit 1, never swallowed silently
                print(f"Error: editreward {name}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            store.add(row)
    return sc.finish(store, len(pairs), "editreward")


if __name__ == "__main__":
    sys.exit(main())
