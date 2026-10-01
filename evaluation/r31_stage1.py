"""R31 stage 1: the UNBLENDED target render, plus the latents stage 2 measures on.

Renders each pair exactly as ``run_fivebench.py`` would -- same per-pair seeding, same
transform, same VAE encode ordering -- but with the StreamGVE Q/K blend switched OFF and
R22's ``step_dump`` hook on, so one rollout yields both the mid-denoising target and the
final render without a second pass.

WHY A 7-STEP RUN, AND WHY THE MEASURED INDEX IS THE LAST ONE
------------------------------------------------------------
CHANGED 2026-09-15 on the user's call. This used to be a 15-step rollout whose MIDDLE x0
prediction (index 7 == len//2) was the measured target. It is now a genuine ``--step 7``
rollout whose FINAL x0 prediction (index 6) is measured, so stage 2 reads a fully
denoised image rather than a mid-trajectory estimate -- LPIPS / DINOv3 / Marigold all
expect a natural image, and on a half-denoised x0 they partly measure denoising artefacts
rather than the edit.

⚠️ THIS IS NOT THE 15-STEP RUN TRUNCATED. ``denoising_step_list`` is built as
``np.arange(1000, 0, -1000/step)`` (``inference_edit_streamedit.py:68``), so 7 steps give
``[1000, 857, 714, 571, 428, 285, 142]`` -- a COARSER grid, not a subset of the 15-step
one. This is a different sampler trajectory, not a prefix of the old one.

⚠️ AND IT IS NOT "PRE-INJECTION". Both gates are ``len(denoising_step_list) // 2``,
which at 7 steps is 3, not 7:

  * the grounding-mask union is injected into the KV cache at
    ``edit_causal_inference.py:896`` -- at index 3;
  * the background source-KV injection is gated on
    ``current_timestep_index > total_timestep // 2`` (``causal_model.py:492``), so it
    fires at indices 4, 5 and 6.

The measured index 6 therefore sits THREE STEPS AFTER both channels engage, and the R26
grounding-mask union has already shaped it. Accepted deliberately -- the fully denoised
image was judged the more useful measurement target -- but it MUST be read that way at
``check-stage2``: the injection suppresses target-vs-source difference in the BACKGROUND
specifically, so a divergence map may localise on the edit partly BECAUSE the mask made
it localise, not purely because the divergence measure is informative. A map that
localises here is weaker evidence than a pre-injection one would have been. The npz
records which regime produced the target in ``measure_kind``.

WHAT "UNBLENDED" MEANS HERE
---------------------------
⚠️ Since R35 (2026-09-24) this script no longer runs ONE regime. ``--blend_sched`` selects
from a closed set: ``zero`` (the default, everything below) or ``first1``/``first2``/
``first3`` (Q/K blend fully ON for the first K steps, then OFF for the rest). R35 uses the
two of them as ALTERNATIVE divergence-measurement targets. The paragraph below describes
``zero`` only; a ``firstK`` run is NOT unblended and its npz says so in ``blend_sched``.

``--blend_sched zero`` => ``s(p) = 0`` => ``blender_rate = 1.0``, which makes all three
Q/K blend sites the identity (pure target Q/K). *Unblended* is unqualified and survives
the step change: the schedule applies to every step of every block. It is only
*pre-injection* that is lost. The bg source-KV injection and the mask gathering are
separate channels and stay ON, deliberately (see the R10 comment in ``causal_model.py``:
without the injection the target branch decouples from the source video entirely). Stage
3 runs with those channels on as well, which is what makes the divergence transfer.

THE SOURCE LATENT IS NOT A DENOISED QUANTITY
--------------------------------------------
``z_src`` is the clean VAE encode of the source clip, full stop. The source branch
re-noises the SAME ``src_input`` at every denoising step
(``edit_causal_inference.py:827``) and never evolves an x0 of its own, so "the source
latent at the final denoising step" IS the encode. Stage 2's ``latent`` arm is therefore
``z_trg - z_src`` = (unblended target x0 at index 6, the final step) - (clean source
encode), NOT a
difference of two trajectories. The npz records this in ``z_src_kind`` so it cannot be
misread downstream.

OUTPUTS
-------
  {out_root}/{method}/step{jj}/edit{T}/{video_name}/00000.png ...
      one frame dir per dumped step; the ``step{jj}`` level sits above ``edit{T}`` on
      purpose, so each step dir is a valid ``--tgt_layout edit_video`` root for
      ``evaluation/fivebench/evaluate.py`` (R22's convention, kept).
  {latent_dir}/edit{T}/{video_name}.npz
      z_trg [F_lat, C, h, w] float32, z_src same shape, plus provenance metadata.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# Default location of the Self-Forcing StreamEdit build, relative to the repo root.
_DEFAULT_SF_ROOT = Path(__file__).resolve().parent.parent / "Self-Forcing_StreamEdit"

# The denoising index stage 2 measures on: the LAST one, i.e. the fully denoised render
# of a 7-step rollout. Since 2026-09-15 this is `step - 1`, not the fractional len//2
# boundary the 15-step version used -- see the module docstring.
_DEFAULT_DUMP_STEPS = "6"
_MEASURE_INDEX = 6
_CALIBRATED_STEP = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # what to run
    parser.add_argument("--edit_type", type=int, required=True,
                        help="FiVE-Bench edit type 1..6 (reads edit_prompt/edit{T}_FiVE.json)")
    parser.add_argument("--method", type=str, default="r31_unblended",
                        help="Arm name; frames go to {out_root}/{method}/step{jj}/edit{T}/")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark",
                        help="FiVE-Bench root (videos/, edit_prompt/)")
    parser.add_argument("--out_root", type=str,
                        default="/projects/dataggen/outputs/five_bench/r31_stage1",
                        help="Root for produced frames (method subdir created underneath)")
    parser.add_argument("--latent_dir", type=str,
                        default="/projects/dataggen/outputs/five_bench/r31_latents",
                        help="Root for the per-clip z_trg/z_src npz ({dir}/edit{T}/{video}.npz)")

    # anchor injection -- same flags and semantics as run_fivebench.py / r22_dump_steps.py
    parser.add_argument("--first_frame_edit_dir", type=str, default=None,
                        help="Root of Qwen-edited anchor PNGs ({dir}/edit{T}/{video}.png)")
    parser.add_argument("--vp_mode", type=str, default="vp", choices=["novp", "vp", "pvp"],
                        help="Anchor injection mechanism (ignored without "
                             "--first_frame_edit_dir, which forces novp)")
    parser.add_argument("--cases_json", type=str, default=None,
                        help="Restrict to the video_names in this cases.json whose "
                             "edit_type matches --edit_type")

    # R31: which denoising indices to decode and save
    parser.add_argument("--dump_steps", type=str, default=_DEFAULT_DUMP_STEPS,
                        help="Comma-separated denoising indices to decode, or 'all'. "
                             f"Default '{_DEFAULT_DUMP_STEPS}': at --step "
                             f"{_CALIBRATED_STEP} index {_MEASURE_INDEX} is BOTH the "
                             "unblended target stage 2 measures AND the final render, "
                             "so one VAE pass per clip covers both.")
    parser.add_argument("--measure_index", type=int, default=_MEASURE_INDEX,
                        help="Denoising index whose latent is written to the npz as "
                             "z_trg. Must be one of --dump_steps.")

    # grounding / boosting hyper-parameters (paper Self-Forcing defaults)
    parser.add_argument("--src_kv", choices=("bg_half", "full"), default="bg_half",
                        help="Source-KV injection into the target branch. 'bg_half' "
                             "(default, and what every R26/R30/R31 arm uses) injects "
                             "BACKGROUND tokens only, and only for denoising indices "
                             "> len//2. 'full' is an R31 PROBE: inject every source "
                             "token, foreground included, at every step -- maximum "
                             "source anchoring with no spatial selectivity anywhere. "
                             "Use a distinct --method so probe output cannot mix with "
                             "a real stage-1 run.")
    #✨ R35 2026-09-24: the stage-1 gating regime, previously PINNED to "zero".
    # A CLOSED set, not a free string -- see the note where it used to be pinned.
    parser.add_argument("--blend_sched", type=str, default="zero",
                        #✨ R38 2026-10-01: + tgt0.9 (still a closed set).
                        choices=("zero", "first1", "first2", "first3", "tgt0.9"),
                        help="Stage-1 gating regime. 'zero' (default, R31's regime) "
                             "holds s(p) = 0 for every step, i.e. blender_rate = 1 and "
                             "the Q/K blend is OFF for the whole rollout. 'firstK' holds "
                             "s(p) = 1 for the first K steps -- Q/K blend fully ON, "
                             "source-anchored -- then 0 for the rest. 'tgt0.9' holds "
                             "s(p) = 1 for every call whose warped input t > 0.9 (R38; "
                             "== first2 at 15 steps, first1 at 5). The resolved "
                             "per-step rate is echoed at startup and the name is written "
                             "into every npz.")
    parser.add_argument("--fg_boost_factor", type=float, default=4.0, help="omega")
    parser.add_argument("--blend_power", type=float, default=2.0,
                        help="rho -- inert here: any --blend_sched value overrides Eq. 4 "
                             "entirely. Kept so the invocation matches R26/R30 verbatim.")

    # rollout sampling -- 21 is what makes these renders comparable to the stored arms
    parser.add_argument("--rollout_chunk_size", type=int, default=21)
    parser.add_argument("--rollout_overlap_block_num", type=int, default=1)

    # sampling / model settings (mirrors run_fivebench.py)
    parser.add_argument("--step", type=int, default=7,
                        help="Sampler steps for STAGE 1 ONLY (stage 3 stays at 15, which "
                             "is what keeps R30's A_disc endpoints and the overlay on "
                             "R26's curve valid). 7 so the measured index 6 is a fully "
                             "denoised render rather than a mid-trajectory x0.")
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT))

    args = parser.parse_args()

    #✨ R35 2026-09-24: this used to read `args.blend_sched = "zero"`, pinned with the
    # note "the Q/K blend is the POINT of this run, not an option ... so no caller can
    # produce a stage 1 output that was silently blended". R35 needs a SECOND regime
    # (first2) as a divergence-measurement target, so the pin becomes argparse `choices`.
    # The guarantee weakens from "stage 1 is never blended" to "stage 1 can only run a
    # schedule that is explicitly named, echoed at startup, and recorded in the npz" --
    # which is what actually prevents a silently-blended output being mistaken for an
    # unblended one. `zero` remains the default, so every pre-R35 caller is unchanged.

    # Resolve --dump_steps against --step, and refuse a measure index that is not dumped
    # (the npz would reference a latent that was never decoded, and the mismatch would
    # only surface in stage 2 as a silently wrong divergence).
    if args.dump_steps.strip().lower() == "all":
        args.dump_step_list = list(range(args.step))
    else:
        try:
            args.dump_step_list = sorted({int(x) for x in args.dump_steps.split(",") if x.strip()})
        except ValueError:
            raise SystemExit(f"[r31_stage1] --dump_steps must be 'all' or a comma-separated "
                             f"list of integers, got {args.dump_steps!r}")
    bad = [j for j in args.dump_step_list if not 0 <= j < args.step]
    if bad:
        raise SystemExit(f"[r31_stage1] --dump_steps {bad} outside [0, {args.step})")
    if args.measure_index not in args.dump_step_list:
        raise SystemExit(f"[r31_stage1] --measure_index {args.measure_index} is not in "
                         f"--dump_steps {args.dump_step_list}; the npz would name a latent "
                         f"that was never decoded.")

    # The property stage 2 now relies on is "the measured latent is the FINAL x0 of the
    # run", not a fractional boundary. Record which gating regime produced it -- both
    # gates are len//2, so a measure index above that saw the mask union and the bg
    # source-KV injection, and a divergence map built on it inherits the mask.
    half = args.step // 2
    args.measure_kind = ("post_injection__final_render"
                         if args.measure_index > half
                         else "pre_injection__mid_trajectory")
    # Not fatal -- a deliberate probe at another index is legitimate -- but stage 2's
    # image-space arms (LPIPS / DINOv3 / Marigold) assume a natural image.
    if args.measure_index != args.step - 1:
        print(f"[r31_stage1] WARNING: --measure_index {args.measure_index} is not the "
              f"final index {args.step - 1}; the npz will hold a MID-TRAJECTORY x0, "
              f"which stage 2's image-space arms are not calibrated for.", flush=True)
    if args.step != _CALIBRATED_STEP:
        print(f"[r31_stage1] WARNING: --step {args.step} != {_CALIBRATED_STEP}, R31's "
              f"stage-1 setting since 2026-09-15.", flush=True)
    print(f"[r31_stage1] measure_kind={args.measure_kind}: len//2 = {half}, so the mask "
          f"union is injected at index {half} and the bg source-KV at indices > {half}.",
          flush=True)
    return args


def save_frames_png(frames: torch.Tensor, out_dir: Path) -> int:
    """Save a ``[T, C, H, W]`` float tensor in [0, 1] as zero-padded PNG frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (frames.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
    return arr.shape[0]


def main() -> None:
    args = parse_args()

    # Resolve external paths to absolute BEFORE chdir-ing into the SF build (see
    # run_fivebench.py -- the build reads configs/checkpoints via relative paths).
    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    latent_dir = Path(args.latent_dir).expanduser().resolve()
    first_frame_edit_dir = (Path(args.first_frame_edit_dir).expanduser().resolve()
                            if args.first_frame_edit_dir else None)
    cases_json = (Path(args.cases_json).expanduser().resolve()
                  if args.cases_json else None)

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame, read_json
    from utils.misc import set_seed
    from diffusers.utils import load_video

    pipeline, low_memory, device, local_rank = load_pipe(args)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    edit_json = data_root / "edit_prompt" / f"edit{args.edit_type}_FiVE.json"
    entries = read_json(edit_json)

    # Filter on video_name AND edit_type: a video can appear under two edit types
    # (0011_lucia is in both edit2 and edit5) and only this --edit_type's pair runs here.
    if cases_json is not None:
        wanted = {c["video_name"] for c in json.loads(cases_json.read_text())
                  if int(c["edit_type"]) == args.edit_type}
        if not wanted:
            raise SystemExit(f"[r31_stage1] {cases_json} lists no case with "
                             f"edit_type={args.edit_type}")
        entries = [e for e in entries if e["video_name"] in wanted]
        found = {e["video_name"] for e in entries}
        if found != wanted:
            raise SystemExit(f"[r31_stage1] cases not present in {edit_json.name}: "
                             f"{sorted(wanted - found)}")

    vp_mode = args.vp_mode if first_frame_edit_dir is not None else "novp"
    n_steps = args.step

    #✨ R35: resolve the regime through the SAME function the bridge calls, and print the
    # per-step rate it will actually see. A wrong regime still produces a complete,
    # plausible stage-1 tree -- nothing downstream can tell -- so this echo is the only
    # cheap guard that the run measured what it claims to have measured.
    from pipeline.utils import _schedule_blend_rate      # sf_root is on sys.path by here
    #✨ R38: t_cur from the pipeline's ACTUAL warped grid, the same value the bridge passes
    # -- so a tgt<tau> echo is computed on what the run sees, not on a re-derived grid.
    _grid = [float(t) / 1000 for t in pipeline.denoising_step_list]
    if len(_grid) != n_steps:
        raise SystemExit(f"[r31_stage1] pipeline grid has {len(_grid)} steps, --step is {n_steps}")
    _s = [_schedule_blend_rate(args.blend_sched, i, n_steps, t_cur=_grid[i])
          for i in range(n_steps)]
    _n_on = sum(int(x) for x in _s)
    _regime = ("Q/K blend OFF for every step" if _n_on == 0 else
               f"Q/K blend ON for the first {_n_on} of {n_steps} steps, then OFF")

    print(f"[r31_stage1] edit_type={args.edit_type} method={args.method} "
          f"pairs={len(entries)} vp_mode={vp_mode} steps={n_steps} "
          f"blend_sched={args.blend_sched} ({_regime}) src_kv={args.src_kv} "
          f"dump_steps={args.dump_step_list} measure_index={args.measure_index}",
          flush=True)
    print(f"[r31_stage1] blender_rate per step = {[int(1.0 - x) for x in _s]}"
          f"   (0 = blend ON / source-anchored, 1 = blend OFF)", flush=True)
    print(f"[r31_stage1] input t per step     = {[round(t, 4) for t in _grid]}", flush=True)
    print(f"[r31_stage1] frames -> {out_root / args.method}", flush=True)
    print(f"[r31_stage1] latents -> {latent_dir / f'edit{args.edit_type}'}", flush=True)

    rows = []
    for e in entries:
        video_name = e["video_name"]
        pair_id = e.get("id", video_name)
        src_path = data_root / "videos" / f"{video_name}.mp4"

        # Re-seed before EVERY pair -- without this a clip's noise depends on its index
        # in the edit{T} json, making subset runs incomparable to full runs. See the
        # warning in run_fivebench.py; this is what keeps a 22-pair subset comparable to
        # R26/R30's stored arms at all.
        set_seed(args.seed)

        try:
            src_video = load_video(str(src_path))
            new_len = find_closest_num_frame(len(src_video))
            if not new_len:
                raise ValueError(f"video too short: {len(src_video)} frames")
            src_video = src_video[:new_len]

            src_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
            video_latents = pipeline.vae.encode_to_latent(
                src_tensor.to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16)
            pipeline.vae.model.clear_cache()

            if vp_mode != "novp":
                anchor_path = first_frame_edit_dir / f"edit{args.edit_type}" / f"{video_name}.png"
                if not anchor_path.exists():
                    raise FileNotFoundError(f"missing anchor {anchor_path}")
                trg_first_frame = Image.open(anchor_path).convert("RGB")

            if vp_mode == "vp":
                # Encode order (src first frame, THEN anchor) and the single trailing
                # clear_cache are load bearing: the VAE carries temporal-conv state
                # between calls, so reordering these two encodes changes the latents and
                # breaks bit-parity with R7/R26/R30. Do not "tidy" this.
                src_first_frame_latent = pipeline.vae.encode_to_latent(
                    transform(src_video[0]).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                trg_first_frame_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = True
                vp_latent = None
            elif vp_mode == "pvp":
                vp_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = None
            else:                                   # novp -- R1 baseline path
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = vp_latent = None

            # wo_video_decode=True: the final video IS step n_steps-1 (asserted below),
            # so decoding it here as well would be a wasted VAE pass.
            steps: list = []
            _, final_latent = pipeline.rollout_inference(
                src_video=video_latents,
                src_prompts=e["source_prompt"],
                trg_prompts=e["target_prompt"],
                src_trigger_words=e["source_object"],
                trg_trigger_words=e["target_object"],
                return_latents=True,
                wo_video_decode=True,
                profile=False,
                low_memory=low_memory,
                independent_first_frame=independent_first_frame,
                src_initial_latent=src_first_frame_latent,
                trg_initial_latent=trg_first_frame_latent,
                fg_boost_factor=args.fg_boost_factor,
                blend_power=args.blend_power,
                rollout_chunk_size=args.rollout_chunk_size,
                rollout_overlap_block_num=args.rollout_overlap_block_num,
                vp_latent=vp_latent,
                #✨ R31: s(p) = 0 => blender_rate = 1.0 => all three Q/K blend sites
                # are the identity. The bg source-KV injection is a separate channel and
                # is deliberately NOT disabled.
                blend_sched=args.blend_sched,
                #✨ R31 probe: whole source K/V at every step (see --src_kv).
                src_kv_full=(args.src_kv == "full"),
                #✨ R22 hook, reused: one rollout, every step's x0 prediction.
                step_dump=steps,
            )

            if len(steps) != 1:
                raise RuntimeError(f"expected 1 stitched step buffer, got {len(steps)}")
            step_latents = steps[0]
            if step_latents.shape[0] != n_steps:
                raise RuntimeError(f"step buffer has {step_latents.shape[0]} steps, "
                                   f"expected {n_steps}")
            # L1 identity gate at the driver level (the pipeline already asserted it per
            # window and after stitching). This is what proves the dump is aligned with
            # the render -- without it, every divergence in stage 2 could be measured on
            # a misaligned tensor and nothing downstream would notice.
            if not torch.equal(step_latents[-1], final_latent):
                raise RuntimeError("R31 GATE1 FAIL: last step latent != final latent")
            print(f"[r31_stage1] GATE1 PASS {video_name}: step[{n_steps - 1}] == final "
                  f"latent (shape {tuple(step_latents.shape)})", flush=True)

            # --- the latents stage 2 measures on -------------------------------------
            # z_src is the CLEAN VAE ENCODE, not a denoised quantity: the source branch
            # re-noises the same src_input at every step and never evolves an x0. See the
            # module docstring; `z_src_kind` records it so stage 2 cannot misread this as
            # a difference of two trajectories.
            z_trg = step_latents[args.measure_index]
            if z_trg.shape[0] != 1:
                raise RuntimeError(f"expected batch 1, got {z_trg.shape[0]}")
            if z_trg.shape != video_latents.shape:
                raise RuntimeError(
                    f"R31: z_trg {tuple(z_trg.shape)} != z_src {tuple(video_latents.shape)}; "
                    "stage 2's `latent` arm differences these elementwise and a shape "
                    "mismatch here would be a silent misalignment.")
            npz_path = latent_dir / f"edit{args.edit_type}" / f"{video_name}.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                z_trg=z_trg[0].float().cpu().numpy(),
                z_src=video_latents[0].float().cpu().numpy(),
                measure_index=np.int32(args.measure_index),
                n_steps=np.int32(n_steps),
                blend_sched=np.str_(args.blend_sched),
                vp_mode=np.str_(vp_mode),
                seed=np.int32(args.seed),
                flow_shift=np.float32(args.flow_shift),
                fg_boost_factor=np.float32(args.fg_boost_factor),
                edit_type=np.int32(args.edit_type),
                video_name=np.str_(video_name),
                z_src_kind=np.str_("clean_vae_encode__never_denoised"),
                #✨ R31 2026-09-15: which gating regime produced z_trg. At --step 7
                # the measured index 6 is POST-injection (both gates are len//2 = 3),
                # so the map partly inherits the R26 grounding mask -- check-stage2
                # must weigh a localised map accordingly.
                measure_kind=np.str_(args.measure_kind),
            )

            # --- decode only the requested steps -------------------------------------
            # clear_cache() between decodes so the VAE's temporal-conv state cannot leak
            # from one step into the next -- if it did, step 14 would not reproduce the
            # stored render.
            n_frames = 0
            for j in args.dump_step_list:
                torch.cuda.empty_cache()
                video_j = pipeline.vae.decode_to_pixel(step_latents[j], use_cache=False)
                video_j = (video_j * 0.5 + 0.5).clamp(0, 1)
                pipeline.vae.model.clear_cache()
                out_dir = out_root / args.method / f"step{j:02d}" / f"edit{args.edit_type}" / video_name
                n_frames = save_frames_png(video_j[0], out_dir)
                del video_j

            del step_latents, steps
            rows.append((pair_id, video_name, "ok", n_frames, len(args.dump_step_list)))
            print(f"[ok] {video_name}: {len(args.dump_step_list)} steps x {n_frames} frames "
                  f"+ latents -> {npz_path.name}", flush=True)
        except Exception as ex:  # never silently swallow -- record and continue
            rows.append((pair_id, video_name, f"error:{type(ex).__name__}", 0, 0))
            print(f"[ERROR] {video_name}: {ex}", flush=True)

    manifest_dir = out_root / args.method
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"_manifest_edit{args.edit_type}.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "video_name", "status", "n_frames", "n_dumped_steps"])
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r[2] == "ok")
    print(f"[r31_stage1] done edit{args.edit_type}: {n_ok}/{len(rows)} ok -> {manifest}",
          flush=True)
    if n_ok != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
