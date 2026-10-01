"""Anchor generation -- oracle T2I-edited frame-0 images via Qwen-Image-Edit.

For each case in ``evaluation/cases.json`` this loads the source first frame
(``src_first_frame`` under ``--data_root``), applies the case ``instruction`` with
Qwen-Image-Edit, and saves the edited frame resized to 832x480 (matching the StreamEdit
pipeline transform ``transforms.Resize((480, 832))``) to ``{--out}/{case_id}.png``. These
are the *oracle* appearance anchors -- curate them (the ``check-anchors`` step)
before inference; ``--first_frame_edit evaluation/anchors/<case>.png`` consumes them.

Run this from a DEDICATED environment: it needs ``diffusers`` new enough to expose
``QwenImageEditPlusPipeline`` plus the ``Qwen/Qwen-Image-Edit-2511`` weights. It is NOT
runnable in the ``streamgve`` env (diffusers 0.31 has no such pipeline), which is
intentional -- the whole StreamEdit pipeline pins that env, so anchor generation is kept
out of it.

Examples
--------
    # all missing anchors
    python evaluation/gen_anchors.py --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark --out evaluation/anchors

    # regenerate one case during the quality gate
    python evaluation/gen_anchors.py --only 0034_cows
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# Target size matches the pipeline transform (Resize((480, 832)) -> H=480, W=832).
TARGET_W, TARGET_H = 832, 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=str, default="evaluation/cases.json",
                        help="Case manifest (list of dicts with case_id/src_first_frame/instruction)")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark",
                        help="FiVE-Bench root; src_first_frame paths are resolved under it")
    parser.add_argument("--out", type=str, default="evaluation/anchors",
                        help="Output dir for <case_id>.png anchors")
    parser.add_argument("--only", type=str, default=None,
                        help="Regenerate only this case_id (forces overwrite for it)")
    parser.add_argument("--edit_type", type=int, default=None,
                        help="Only generate cases whose edit_type == T (for sbatch-array split)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Regenerate anchors even if the PNG already exists")

    # Qwen-Image-Edit-2511 (Plus pipeline) settings
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--true_cfg_scale", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true", default=False,
                        help="Enable model CPU offload (lower VRAM, slower)")
    return parser.parse_args()


def load_pipeline(model_id: str, cpu_offload: bool):
    """Load Qwen-Image-Edit, failing loudly if the env can't provide it."""
    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:  # wrong env (e.g. streamgve's diffusers 0.31)
        import diffusers
        sys.exit(
            f"[gen_anchors] QwenImageEditPlusPipeline unavailable (diffusers "
            f"{diffusers.__version__}): {exc}\n"
            f"  Run this in a dedicated env with a recent diffusers and the "
            f"'{model_id}' weights -- NOT the streamgve env."
        )
    if not torch.cuda.is_available():
        sys.exit("[gen_anchors] CUDA not available; Qwen-Image-Edit needs a GPU.")

    pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(Path(args.cases).expanduser().read_text())
    if args.only is not None:
        cases = [c for c in cases if c["case_id"] == args.only]
        if not cases:
            sys.exit(f"[gen_anchors] --only {args.only!r} matched no case in {args.cases}")
    if args.edit_type is not None:
        cases = [c for c in cases if c.get("edit_type") == args.edit_type]

    # Decide what actually needs generating before paying the model-load cost.
    todo = []
    for c in cases:
        out_path = out_dir / f"{c['case_id']}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        force = args.overwrite or (args.only == c["case_id"])
        if out_path.exists() and not force:
            print(f"[skip] {c['case_id']}: {out_path} exists (use --overwrite / --only)")
            continue
        src = data_root / c["src_first_frame"]
        if not src.exists():
            print(f"[ERROR] {c['case_id']}: missing src_first_frame {src}")
            continue
        todo.append((c, src, out_path))

    if not todo:
        print("[gen_anchors] nothing to generate.")
        return

    pipe = load_pipeline(args.model_id, args.cpu_offload)

    n_ok = 0
    for c, src, out_path in todo:
        instruction = c["first_frame_edit_prompt"] or c["instruction"]
        image = Image.open(src).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        try:
            result = pipe(
                image=[image],  # Plus pipeline takes a list of reference images
                prompt=instruction,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
            )
            edited = result.images[0].convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
            edited.save(out_path)
            n_ok += 1
            print(f"[ok] {c['case_id']}: {instruction!r} -> {out_path}")
        except Exception as exc:  # never silently swallow -- report and continue
            print(f"[ERROR] {c['case_id']}: {type(exc).__name__}: {exc}")

    print(f"[gen_anchors] done: {n_ok}/{len(todo)} anchors -> {out_dir}")


if __name__ == "__main__":
    main()
