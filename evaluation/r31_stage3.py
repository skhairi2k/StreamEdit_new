"""R31 stage 3: the SPATIALLY-GATED render.

Full 15-step rollout per clip with StreamGVE Eq. 4's scalar release exponent replaced by a
CONTINUOUS PER-TOKEN field ``rho(f, p)``, built offline by ``evaluation/r31_rho_map.py``
from stage 2's divergence maps. One rollout per clip; R22's ``step_dump`` rides along so
the final AND every intermediate video come out of the SAME run.

WHY 15 STEPS HERE WHEN STAGE 1 IS 7
-----------------------------------
Stage 1's step count is free -- it is never compared to an R26/R30 arm, and it was dropped
to 7 on 2026-09-15. Stage 3's is NOT. Everything about the comparison depends on this run
matching R26/R30 exactly:

  * R30's ``A_disc`` endpoints (``A_disc(2) = 0.300``, ``A_disc(50) = 0.0021``) are defined
    over the 15-step ``t_next`` grid, and ``r31_rho_map.py`` inverted the budget on that
    grid. A different ``--step`` silently invalidates every exponent in the field.
  * the (CLIP-target, lpips_unedit_part) points only drop onto ``r26_tradeoff_figure.py``'s
    stored 52-arm cloud because seed / anchors / cases / sampler are byte-identical.

So ``--step 15`` is pinned, and the driver REFUSES a rho field built for another schedule
(the npz records ``step`` and ``flow_shift`` precisely so this can be checked).

PREFLIGHT BEFORE THE MODEL LOADS
--------------------------------
Every clip's rho npz is validated BEFORE ``load_pipe`` -- existence, ``frame_seq_length ==
1560``, latent-frame count matching the clip, schedule match, finite non-negative
exponents. ``rollout_inference`` raises on a bad field too, but doing it here costs seconds
instead of failing after a ~20-minute model load, and it reports EVERY bad clip at once
rather than dying on the first.

⚠️ A SHORT FIELD IS THE DANGEROUS FAILURE. ``rho_frames`` is indexed in ABSOLUTE
latent-frame coordinates, so a field with too few frames would misalign every frame after
the first chunk. R26's plumbing refuses it; this refuses it earlier.

OUTPUTS
-------
  {out_root}/{method}/step{jj}/edit{T}/{video_name}/00000.png ...
      one frame dir per denoising index (00..14 with --dump_steps all). The ``step{jj}``
      level sits ABOVE ``edit{T}`` so each step dir is a valid ``--tgt_layout edit_video``
      root for ``evaluation/fivebench/evaluate.py`` (R22's convention, kept). ``step14`` is
      what ``r31_eval.sh`` scores.
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

_DEFAULT_SF_ROOT = Path(__file__).resolve().parent.parent / "Self-Forcing_StreamEdit"

FRAME_SEQ_LENGTH = 1560          # == pipeline's self.frame_seq_length (30 x 52)
CALIBRATED_STEP = 15             # stage 3 is pinned here; see the module docstring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edit_type", type=int, required=True,
                        help="FiVE-Bench edit type 1..6")
    parser.add_argument("--method", type=str, required=True,
                        help="Arm name, e.g. r31_lpips; frames go to "
                             "{out_root}/{method}/step{jj}/edit{T}/")
    parser.add_argument("--rho_dir", type=str, required=True,
                        help="One arm's exponent fields from r31_rho_map.py, i.e. "
                             ".../r31_rho/{arm}, containing edit{T}/{video}.npz")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--out_root", type=str,
                        default="/projects/dataggen/outputs/five_bench/r31_arms")

    parser.add_argument("--first_frame_edit_dir", type=str, default=None,
                        help="Root of Qwen-edited anchor PNGs ({dir}/edit{T}/{video}.png)")
    parser.add_argument("--vp_mode", type=str, default="vp",
                        choices=["novp", "vp", "pvp"])
    parser.add_argument("--cases_json", type=str, default=None)

    parser.add_argument("--dump_steps", type=str, default="all",
                        help="Comma-separated denoising indices to decode, or 'all' "
                             "(default). The user asked for the final AND mid-step "
                             "videos, and step_dump yields every intermediate from the "
                             "SAME rollout -- so 'all' is 1 rollout + 15 VAE decodes per "
                             "clip, NOT 15 rollouts.")

    parser.add_argument("--fg_boost_factor", type=float, default=4.0)
    parser.add_argument("--blend_power", type=float, default=2.0,
                        help="Scalar Eq. 4 exponent -- INERT once rho_frames is supplied "
                             "(the per-token field replaces it). Kept so the invocation "
                             "matches R26/R30 verbatim.")
    parser.add_argument("--rollout_chunk_size", type=int, default=21)
    parser.add_argument("--rollout_overlap_block_num", type=int, default=1)

    parser.add_argument("--step", type=int, default=CALIBRATED_STEP,
                        help="PINNED at 15. r31_rho_map.py inverted the source budget on "
                             "the 15-step t_next grid, and the overlay onto R26's curve "
                             "requires it. Stage 1's 7 steps are unrelated.")
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT))
    parser.add_argument("--allow_schedule_mismatch", action="store_true",
                        help="Proceed even if a rho npz was built for a different "
                             "step/flow_shift. Diagnostic only -- the exponents would be "
                             "wrong for this schedule.")

    args = parser.parse_args()

    # rho_frames and R26's blend_sched overrides are mutually exclusive in the pipeline
    # (the schedule would replace the per-step scalar the field is built on), so this
    # driver never sets one.
    if args.dump_steps.strip().lower() == "all":
        args.dump_step_list = list(range(args.step))
    else:
        try:
            args.dump_step_list = sorted({int(x) for x in args.dump_steps.split(",")
                                          if x.strip()})
        except ValueError:
            raise SystemExit(f"[r31_stage3] --dump_steps must be 'all' or integers, "
                             f"got {args.dump_steps!r}")
    bad = [j for j in args.dump_step_list if not 0 <= j < args.step]
    if bad:
        raise SystemExit(f"[r31_stage3] --dump_steps {bad} outside [0, {args.step})")
    if args.step != CALIBRATED_STEP:
        print(f"[r31_stage3] WARNING: --step {args.step} != {CALIBRATED_STEP}. The rho "
              f"field's budget inversion and R26's A_disc endpoints are both defined on "
              f"the {CALIBRATED_STEP}-step grid; this run is NOT comparable to R26/R30.",
              flush=True)
    return args


def load_rho(npz_path: Path, n_latent: int, step: int, flow_shift: float,
             allow_schedule_mismatch: bool) -> torch.Tensor:
    """Read and fully validate one clip's exponent field -> float32 [F_lat, 1560]."""
    z = np.load(npz_path)
    rho = z["rho"]
    if rho.ndim != 2:
        raise ValueError(f"rho has shape {rho.shape}, expected [F_lat, {FRAME_SEQ_LENGTH}]")
    if rho.shape[1] != FRAME_SEQ_LENGTH:
        raise ValueError(f"frame_seq_length {rho.shape[1]} != {FRAME_SEQ_LENGTH}")
    if rho.shape[0] != n_latent:
        raise ValueError(
            f"rho holds {rho.shape[0]} latent frames but this clip has {n_latent}. "
            "rho_frames is indexed in ABSOLUTE latent-frame coordinates, so a short "
            "field would misalign every frame after the first chunk.")
    if not np.isfinite(rho).all():
        raise ValueError("rho holds a non-finite exponent")
    if (rho < 0).any():
        raise ValueError("rho holds a negative exponent; W_src = t ** rho diverges as t->0")
    # A_disc is schedule-dependent: a field inverted on another grid encodes a different
    # source budget than this rollout will actually deliver.
    if "step" in z.files:
        f_step, f_shift = int(z["step"]), float(z["flow_shift"])
        if (f_step, f_shift) != (step, flow_shift):
            msg = (f"rho was built for step={f_step} flow_shift={f_shift} but this run "
                   f"uses step={step} flow_shift={flow_shift}; the budget inversion does "
                   f"not transfer.")
            if not allow_schedule_mismatch:
                raise ValueError(msg)
            print(f"[r31_stage3] WARNING (--allow_schedule_mismatch): {msg}", flush=True)
    return torch.from_numpy(rho.astype(np.float32))


def save_frames_png(frames: torch.Tensor, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (frames.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
    return arr.shape[0]


def main() -> None:
    args = parse_args()

    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    rho_dir = Path(args.rho_dir).expanduser().resolve()
    first_frame_edit_dir = (Path(args.first_frame_edit_dir).expanduser().resolve()
                            if args.first_frame_edit_dir else None)
    cases_json = (Path(args.cases_json).expanduser().resolve()
                  if args.cases_json else None)

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame, read_json
    from utils.misc import set_seed
    from diffusers.utils import load_video

    edit_json = data_root / "edit_prompt" / f"edit{args.edit_type}_FiVE.json"
    entries = read_json(edit_json)
    if cases_json is not None:
        wanted = {c["video_name"] for c in json.loads(cases_json.read_text())
                  if int(c["edit_type"]) == args.edit_type}
        if not wanted:
            raise SystemExit(f"[r31_stage3] {cases_json} lists no case with "
                             f"edit_type={args.edit_type}")
        entries = [e for e in entries if e["video_name"] in wanted]
        found = {e["video_name"] for e in entries}
        if found != wanted:
            raise SystemExit(f"[r31_stage3] cases not present in {edit_json.name}: "
                             f"{sorted(wanted - found)}")

    # ---- PREFLIGHT: every rho field validated BEFORE the model loads -----------------
    # Reports every bad clip at once rather than dying on the first, and costs seconds
    # instead of a ~20-minute load followed by a crash.
    print(f"[r31_stage3] preflight: validating {len(entries)} rho fields in {rho_dir}",
          flush=True)
    rho_cache = {}
    problems = []
    for e in entries:
        name = e["video_name"]
        npz = rho_dir / f"edit{args.edit_type}" / f"{name}.npz"
        if not npz.exists():
            problems.append(f"{name}: missing {npz}")
            continue
        try:
            src_video = load_video(str(data_root / "videos" / f"{name}.mp4"))
            new_len = find_closest_num_frame(len(src_video))
            if not new_len:
                raise ValueError(f"video too short: {len(src_video)} frames")
            n_latent = (new_len - 1) // 4 + 1
            rho_cache[name] = load_rho(npz, n_latent, args.step, args.flow_shift,
                                       args.allow_schedule_mismatch)
            print(f"[r31_stage3]   ok {name}: rho {tuple(rho_cache[name].shape)} "
                  f"range [{rho_cache[name].min():.2f}, {rho_cache[name].max():.2f}]",
                  flush=True)
        except Exception as ex:
            problems.append(f"{name}: {type(ex).__name__}: {ex}")
    if problems:
        print(f"[r31_stage3] PREFLIGHT FAILED ({len(problems)}):", flush=True)
        for p in problems:
            print(f"    {p}", flush=True)
        raise SystemExit(1)
    print(f"[r31_stage3] preflight PASS for all {len(entries)} clips", flush=True)

    pipeline, low_memory, device, local_rank = load_pipe(args)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    vp_mode = args.vp_mode if first_frame_edit_dir is not None else "novp"
    n_steps = args.step
    print(f"[r31_stage3] edit_type={args.edit_type} method={args.method} "
          f"pairs={len(entries)} vp_mode={vp_mode} steps={n_steps} "
          f"dump_steps={len(args.dump_step_list)} indices "
          f"rho_dir={rho_dir}", flush=True)

    rows = []
    for e in entries:
        video_name = e["video_name"]
        pair_id = e.get("id", video_name)

        # Re-seed before EVERY pair, or a clip's noise depends on its index in the
        # edit{T} json and a subset run stops being comparable to R26/R30's stored arms.
        set_seed(args.seed)

        try:
            src_video = load_video(str(data_root / "videos" / f"{video_name}.mp4"))
            new_len = find_closest_num_frame(len(src_video))
            src_video = src_video[:new_len]

            src_tensor = torch.stack([transform(img) for img in src_video],
                                     dim=1).unsqueeze(0)
            video_latents = pipeline.vae.encode_to_latent(
                src_tensor.to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16)
            pipeline.vae.model.clear_cache()

            rho_frames = rho_cache[video_name].to(device)
            if rho_frames.shape[0] != video_latents.shape[1]:
                raise RuntimeError(
                    f"rho has {rho_frames.shape[0]} latent frames but the encode produced "
                    f"{video_latents.shape[1]} -- preflight and the encode disagree.")

            if vp_mode != "novp":
                anchor_path = (first_frame_edit_dir / f"edit{args.edit_type}"
                               / f"{video_name}.png")
                if not anchor_path.exists():
                    raise FileNotFoundError(f"missing anchor {anchor_path}")
                trg_first_frame = Image.open(anchor_path).convert("RGB")

            if vp_mode == "vp":
                # Encode order (src first frame, THEN anchor) and the single trailing
                # clear_cache are load bearing: the VAE carries temporal-conv state
                # between calls, so reordering breaks bit-parity with R7/R26/R30.
                src_first_frame_latent = pipeline.vae.encode_to_latent(
                    transform(src_video[0]).unsqueeze(0).unsqueeze(2).to(
                        device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                trg_first_frame_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(
                        device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = True
                vp_latent = None
            elif vp_mode == "pvp":
                vp_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(
                        device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = None
            else:
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = vp_latent = None

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
                #✨ R31: THE POINT OF THIS RUN. A continuous per-token exponent field
                # replaces Eq. 4's scalar `blend_power`. No blend_sched is passed -- it
                # would override the per-step scalar rate the field is built on, and the
                # pipeline refuses the combination.
                rho_frames=rho_frames,
                #✨ R22 hook: one rollout, every step's x0 prediction.
                step_dump=steps,
            )

            if len(steps) != 1:
                raise RuntimeError(f"expected 1 stitched step buffer, got {len(steps)}")
            step_latents = steps[0]
            if step_latents.shape[0] != n_steps:
                raise RuntimeError(f"step buffer has {step_latents.shape[0]} steps, "
                                   f"expected {n_steps}")
            # The alignment gate R22 established: without it every dumped step could be
            # misaligned with the render and nothing downstream would notice.
            if not torch.equal(step_latents[-1], final_latent):
                raise RuntimeError("R31 GATE1 FAIL: last step latent != final latent")
            print(f"[r31_stage3] GATE1 PASS {video_name}: step[{n_steps - 1}] == final "
                  f"latent (shape {tuple(step_latents.shape)})", flush=True)

            n_frames = 0
            for j in args.dump_step_list:
                torch.cuda.empty_cache()
                video_j = pipeline.vae.decode_to_pixel(step_latents[j], use_cache=False)
                video_j = (video_j * 0.5 + 0.5).clamp(0, 1)
                pipeline.vae.model.clear_cache()
                out_dir = (out_root / args.method / f"step{j:02d}"
                           / f"edit{args.edit_type}" / video_name)
                n_frames = save_frames_png(video_j[0], out_dir)
                del video_j

            del step_latents, steps
            rows.append((pair_id, video_name, "ok", n_frames, len(args.dump_step_list)))
            print(f"[ok] {video_name}: {len(args.dump_step_list)} steps x {n_frames} "
                  f"frames", flush=True)
        except Exception as ex:
            rows.append((pair_id, video_name, f"error:{type(ex).__name__}", 0, 0))
            print(f"[ERROR] {video_name}: {ex}", flush=True)

    manifest_dir = out_root / args.method
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"_manifest_edit{args.edit_type}.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "video_name", "status", "n_frames", "n_dumped_steps"])
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r[2] == "ok")
    print(f"[r31_stage3] done edit{args.edit_type}: {n_ok}/{len(rows)} ok -> {manifest}",
          flush=True)
    if n_ok != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
