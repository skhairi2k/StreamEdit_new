"""R32 driver: head-wise re-balancing of the target branch's KV segments.

Each arm applies a per-head additive bias to the softmax logits of the target
branch's PREVIOUS-FRAMES key segment, leaving both current-frame segments
(the source's current frame and the target's current frame) at 0. That single
scalar per head spans the whole past-vs-current axis, because softmax is
shift-invariant per query row.

This is anchor-free by design. R10 gated an ADDED segment (a visual-prompt
bank) per head; R32 REDISTRIBUTES mass among the segments StreamEdit already
has. The bridge refuses the two together.

ARMS -- ``{table}_b{b_max x 10}``, plus the bare ``zero``:

  Stage 1 (the default; establishes whether an effect exists and at what scale)
    * ``flash``           -- no table at all, so the FlashAttention path. NOT an
                             experimental arm: flash-vs-zero is the noise floor
                             the kernel change alone contributes, and no arm
                             delta below it is readable. Measured on every clip,
                             because a floor from one clip cannot bound effects
                             from 22.
    * ``zero``            -- b == 0, but still through SDPA. THE REFERENCE for
                             every comparison.
    * ``label_b05/10/20`` -- R19's validated labels: -1 SPATIAL (117 heads),
                             +1 TEMPORAL (49), 0 MIXED/DENSE (the abstain).
    * ``cont_b05/10/20``  -- tanh(margin/T) attenuated by reconstruction
                             quality; no threshold, 177 heads nonzero.
    * ``label_rev_b10`` / ``cont_rev_b10`` -- negated. If routed and reversed
                             degrade SYMMETRICALLY, the heads are already at
                             their preferred allocation and the margin has no
                             headroom. That is SATURATION, which is a different
                             finding from "routing does not work", and this is
                             the cheapest way to tell them apart.

  Stage 2 (run ONLY if Stage 1 shows an effect -- do not spend GPU on controls
  for a null)
    * ``*_unif_b10`` -- every head gets mean(s). Same global shift, zero
                        per-head variation: kills "you just globally
                        up-weighted the past".
    * ``*_shuf_b10`` -- the exact multiset of s permuted across heads at a
                        fixed seed. Same distribution, wrong assignment: kills
                        "any per-head variation would have done". ``unif`` and
                        ``shuf`` rule out DIFFERENT things; both are needed.

WHY EVERY ARM PASSES A TABLE, INCLUDING ``zero``
------------------------------------------------
FlashAttention accepts no additive bias, so a biased arm necessarily runs on
SDPA. Zeros-bias SDPA is not bitwise equal to ``attn_mask=None`` SDPA
(measured 2.4e-4 at production shapes), so ``zero`` must ALSO carry an explicit
zeros table and take the identical code path. Otherwise every arm delta would
be confounded with the change of kernel. For the same reason the stored
FlashAttention baseline is NOT a valid reference here -- the ``zero`` arm is.
The flash-vs-zero difference is measured once by the smoke gate and is the
noise floor below which no arm delta is readable.

Tables come from ``evaluation/r32_seg_bias.pt`` (built by
``evaluation/r32_build_seg_bias.py`` from R19's equal-budget disjoint key
sets). The driver refuses a table file without R32 provenance rather than
running on it silently.

The same seed is reset before every arm, so a difference between arms is
attributable to the bias rather than to sampling noise.

Outputs PNG frames to ``{out_root}/{arm}/edit{T}/{video_name}/`` plus
``_manifest.csv``. The ``edit{T}/`` level is required, not cosmetic: the clip
set contains the same video under two edit types (0011_lucia in edit2 and
edit5), so a flat ``{arm}/{video_name}/`` would have one run silently
overwrite the other. It is also the layout R7/R10 already use, so
``evaluate.py --tgt_layout edit_video`` scores it without a new code path.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SF_ROOT = _REPO_ROOT / "Self-Forcing_StreamEdit"

# Single-window inference() holds at most 21 latent frames (max_attention_size
# 32760 / 1560 tokens per frame); 21 latent frames == 81 pixel frames. Longer
# clips overflow the cache, so they are truncated -- same treatment as R10, so
# the two are directly comparable.
MAX_PIXEL_FRAMES = 81

STAGE1_ARMS = ("flash", "zero",
               "label_b05", "label_b10", "label_b20",
               "cont_b05", "cont_b10", "cont_b20",
               "label_rev_b10", "cont_rev_b10")
STAGE2_ARMS = ("label_unif_b10", "cont_unif_b10",
               "label_shuf_b10", "cont_shuf_b10")
ARMS = STAGE1_ARMS + STAGE2_ARMS

# `flash` passes NO table, so it takes the FlashAttention path -- the only arm
# that does. It is not an experimental arm: flash-vs-zero is the NOISE FLOOR
# contributed by the kernel change alone, and no arm delta below it is
# readable. It is in Stage 1 because a floor measured on one clip cannot bound
# effects measured on 22.
_NO_TABLE_ARM = "flash"


def parse_arm(arm: str) -> tuple[str, float]:
    """``'label_rev_b10'`` -> ``('label_rev', 1.0)``; ``'zero'`` -> ``('zero', 0.0)``."""
    if arm in (_NO_TABLE_ARM, "zero"):
        return arm, 0.0
    table, sep, suffix = arm.rpartition("_b")
    if not sep or not suffix.isdigit():
        raise ValueError(f"cannot parse arm '{arm}': expected '{{table}}_b{{NN}}'")
    return table, int(suffix) / 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_json", type=str, default=str(_REPO_ROOT / "evaluation" / "cases.json"))
    parser.add_argument("--cases", type=str, nargs="*", default=None,
                        help="case_ids to run (default: every case in cases.json)")
    # Default is Stage 1 ONLY: the *_unif / *_shuf controls exist to explain an
    # effect, so running them before one is established just burns GPU time.
    parser.add_argument("--arms", type=str, nargs="*", default=list(STAGE1_ARMS),
                        choices=list(ARMS))
    parser.add_argument("--bias", type=str,
                        default=str(_REPO_ROOT / "evaluation" / "r32_seg_bias.pt"),
                        help="per-head bias tables from evaluation/r32_build_seg_bias.py")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--out_root", type=str,
                        default="/projects/dataggen/outputs/five_bench/r32_seg_bias")
    # The manifest name carries the arm set. r32_infer.sh is an ARRAY over arms,
    # so several processes share out_root: a single fixed `_manifest.csv` would
    # have the last task to finish silently clobber every other task's record.
    parser.add_argument("--tag", type=str, default=None,
                        help="manifest suffix (default: the arm name for a "
                             "single-arm run, else '{N}arms')")

    parser.add_argument("--fg_boost_factor", type=float, default=4.0, help="omega, CrossAttn boosting")
    parser.add_argument("--blend_power", type=float, default=2.0, help="rho in Eq. 4")
    # R32 asks about the cache StreamEdit ACTUALLY runs with, so the self-attn
    # blending stays ON (blend_off=False, the paper's Eq. 4 path) -- unlike R10,
    # which turned it off to isolate an added anchor. Opting out gives the
    # isolated variant: pure target keys and an unblended query, cleaner to
    # attribute but no longer the deployed configuration.
    parser.add_argument("--blend_off", action="store_true", default=False,
                        help="Debug: disable self-attn query/key blending "
                             "(default: ON, the deployed StreamEdit recipe)")
    parser.add_argument("--rollout_chunk_size", type=int, default=-1,
                        help="-1 = single-window inference() path (clips truncated to "
                             "--max_pixel_frames). >0 enables the multi-window rollout; "
                             "the bias is re-stamped per window, so that path works too.")
    parser.add_argument("--max_pixel_frames", type=int, default=MAX_PIXEL_FRAMES)

    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT))
    return parser.parse_args()


def save_frames_png(frames: torch.Tensor, out_dir: Path) -> int:
    """Save a ``[T, C, H, W]`` float tensor in [0, 1] as zero-padded PNG frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (frames.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
    return arr.shape[0]


def main() -> None:
    args = parse_args()

    # Resolve every external path BEFORE chdir-ing into the SF build, so relative
    # config/checkpoint paths resolve there while data/output paths stay put.
    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    cases_json = Path(args.cases_json).expanduser().resolve()
    bias_path = Path(args.bias).expanduser().resolve()

    all_cases = json.loads(cases_json.read_text())
    wanted = args.cases
    if wanted:
        by_id = {c["case_id"]: c for c in all_cases}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise SystemExit(f"[r32] case_ids not in {cases_json}: {missing}")
        cases = [by_id[c] for c in wanted]
    else:
        cases = all_cases

    bias_payload = torch.load(bias_path, weights_only=False)
    if "provenance" not in bias_payload:
        raise SystemExit(
            f"[r32] {bias_path} has no provenance field. Rebuild with:\n"
            "  python evaluation/r32_build_seg_bias.py")
    arm_spec = {}
    for arm in args.arms:
        table, b_max = parse_arm(arm)
        if table != _NO_TABLE_ARM and table not in bias_payload:
            raise SystemExit(f"[r32] table '{table}' (for arm '{arm}') not in {bias_path}")
        arm_spec[arm] = (table, b_max)
    print(f"[r32] bias: {bias_payload['provenance']}")
    print(f"[r32]   thresholds {bias_payload['thresholds']} "
          f"T={bias_payload['temperature']} normalized={bias_payload['normalized']}")
    print(f"[r32]   R19 census {bias_payload['census']}")

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame
    from diffusers.utils import load_video

    pipeline, low_memory, device, local_rank = load_pipe(args)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print(f"[r32] cases={len(cases)} arms={args.arms} blend_off={args.blend_off} "
          f"seed={args.seed} out={out_root}", flush=True)
    for arm, (table, b_max) in arm_spec.items():
        if table == _NO_TABLE_ARM:
            print(f"[r32]   arm {arm:<16} NO table -> FlashAttention path "
                  f"(noise-floor reference)", flush=True)
            continue
        s = bias_payload[table].float() * b_max
        nz = int((s.abs() > 1e-9).sum())
        print(f"[r32]   arm {arm:<16} table={table:<12} b_max={b_max:<4} "
              f"nonzero={nz:3d}/{s.numel()} "
              f"b in [{s.min():+.3f}, {s.max():+.3f}]", flush=True)

    rows = []
    for case in cases:
        video_name = case["video_name"]
        src_path = data_root / "videos" / f"{video_name}.mp4"

        # ---- encode once per case, reused by every arm ----
        try:
            src_video = load_video(str(src_path))
            new_len = find_closest_num_frame(len(src_video))
            if not new_len:
                raise ValueError(f"video too short: {len(src_video)} frames")
            if args.rollout_chunk_size <= 0:
                new_len = min(new_len, args.max_pixel_frames)
            src_video = src_video[:new_len]

            src_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
            video_latents = pipeline.vae.encode_to_latent(
                src_tensor.to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16)
            pipeline.vae.model.clear_cache()
            print(f"[r32] {video_name}: {new_len} frames -> latents "
                  f"{tuple(video_latents.shape)}", flush=True)
        except Exception as ex:  # encoding failed -> record for every arm and move on
            for arm in args.arms:
                rows.append((case["case_id"], video_name, arm,
                             f"encode_error:{type(ex).__name__}", 0, 0.0))
            print(f"[ERROR] {video_name}: encode failed: {ex}", flush=True)
            continue

        # ---- one run per arm, identical seed ----
        for arm in args.arms:
            table, b_max = arm_spec[arm]
            out_dir = out_root / arm / f"edit{case['edit_type']}" / video_name
            try:
                # Reset before every arm so arms differ only by the bias, not noise.
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                # A real table for every arm EXCEPT `flash`, `zero` included:
                # the zero arm has to take the same SDPA path as the biased ones
                # or the comparison is confounded with the flash->SDPA kernel
                # change. `flash` deliberately passes None to measure exactly
                # that confound. See the module docstring.
                seg_bias_table = None if table == _NO_TABLE_ARM else \
                    (bias_payload[table].float() * b_max).to(device)

                t0 = time.time()
                edit_video = pipeline.rollout_inference(
                    src_video=video_latents,
                    src_prompts=case["src_prompt"],
                    trg_prompts=case["trg_prompt"],
                    src_trigger_words=case["src_word"],
                    trg_trigger_words=case["trg_word"],
                    return_latents=False,
                    wo_video_decode=False,
                    profile=False,
                    low_memory=low_memory,
                    independent_first_frame=False,
                    src_initial_latent=None,
                    trg_initial_latent=None,
                    fg_boost_factor=args.fg_boost_factor,
                    blend_power=args.blend_power,
                    rollout_chunk_size=args.rollout_chunk_size,
                    blend_off=args.blend_off,
                    # Anchor-free by design; the bridge refuses VP + bias together.
                    vp_latent=None,
                    vp_head_gate=None,
                    seg_bias_table=seg_bias_table,
                )
                pipeline.vae.model.clear_cache()
                elapsed = time.time() - t0

                n_frames = save_frames_png(edit_video[0], out_dir)
                rows.append((case["case_id"], video_name, arm, "ok", n_frames, round(elapsed, 1)))
                print(f"[ok] {video_name} / {arm}: {n_frames} frames in {elapsed:.1f}s", flush=True)
            except Exception as ex:  # never silently swallow -- record and continue
                rows.append((case["case_id"], video_name, arm, f"error:{type(ex).__name__}", 0, 0.0))
                print(f"[ERROR] {video_name} / {arm}: {ex}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("-".join(args.arms) if len(args.arms) <= 2
                       else f"{len(args.arms)}arms")
    manifest = out_root / f"_manifest_{tag}.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "video_name", "arm", "status", "n_frames", "seconds"])
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r[3] == "ok")
    print(f"[r32] done: {n_ok}/{len(rows)} ok -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
