"""Shared FiVE-Bench inference runner for the StreamEdit (Self-Forcing) pipeline.

Loads ``EditCausalInferencePipeline`` **once** and runs a **single** FiVE-Bench edit
type per invocation, so the six edit types can be parallelised with a SLURM array.
It reuses ``load_pipe`` / ``find_closest_num_frame`` / ``read_json`` from
``Self-Forcing_StreamEdit/inference_edit_streamedit.py`` and drives the pipeline over
every prompt pair in ``edit_prompt/edit{T}_FiVE.json``.

Shared across tasks (no task-id prefix):
  * R1 runs it with ``--method baseline`` (selectivity off) to produce the baseline arm.
  * R7 adds ``--first_frame_edit_dir`` for the paper's SS4.5 visual prompting.
  * R20 adds ``--vp_mode`` (how the anchor enters), ``--blend_sched`` (which blender-rate
    schedule replaces Eq. 4) and ``--cases_json`` (run a named subset of pairs).
  * R25 adds ``--tau_map`` (a per-pair scalar rho).
  * R26 adds ``--union_dump_dir`` (pass 1: dump the grounding-mask union M_f) and
    ``--union_mask_dir`` / ``--tau_bg`` / ``--tau_fg`` (pass 2: a SPATIAL rho field
    built from that mask).

Every ``--vp_mode`` / ``--blend_sched`` default reproduces the R1/R7 code path, so
those flags do not change what an existing call produces.

.. warning::

   **Seeding changed on 2026-07-22; outputs rendered before that date are a
   different generation.** The RNG is now re-seeded before every pair. It used to be
   seeded once per process, so a pair's noise depended on how many pairs preceded
   it -- meaning a ``--cases_json`` subset silently produced a *different video* for
   the same pair than a full run did (measured on ``0042_gym-ball``, index 4 of
   edit6: 93% of pixels differed, max 237/255). Pairs at index 0 of their edit{T}
   json are unaffected and stay bit-identical; every later pair is not.

   Consequence: subset runs are now comparable to full runs, which they were not
   before. But ``five_bench/baseline`` (R1) and ``five_bench/r7_visual_prompting``
   (R7) as rendered on 2026-07-18/19 predate this change -- they are being
   re-rendered. Until that lands, do not byte-compare against them and do not put
   their metrics in the same table as fresh runs.

Outputs zero-padded PNG frames to ``{out_root}/{method}/edit{T}/{video_name}/`` and a
per-type ``_manifest.csv`` (pair id, video, status, n_frames, empty-trigger flag).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # what to run
    parser.add_argument("--edit_type", type=int, required=True,
                        help="FiVE-Bench edit type 1..6 (reads edit_prompt/edit{T}_FiVE.json)")
    parser.add_argument("--method", type=str, default="baseline",
                        help="Output arm name; frames go to {out_root}/{method}/edit{T}/")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark",
                        help="FiVE-Bench root (videos/, edit_prompt/)")
    parser.add_argument("--out_root", type=str, default="outputs/five_bench",
                        help="Root for produced frames (method subdir created underneath)")

    # R7: visual prompting -- Qwen-edited oracle first frame per pair
    parser.add_argument("--first_frame_edit_dir", type=str, default=None,
                        help="Root of Qwen-edited anchor PNGs; enables visual prompting "
                             "via {dir}/edit{T}/{video_name}.png")

    # R20: how the anchor enters, and which blender-rate schedule replaces Eq. 4.
    parser.add_argument("--vp_mode", type=str, default="vp", choices=["novp", "vp", "pvp"],
                        help="Anchor injection mechanism. 'vp' = paper SS4.5, cached as an "
                             "independent first frame (fades as the window slides) -- the "
                             "R7 behaviour and the default, so --first_frame_edit_dir alone "
                             "reproduces R7. 'pvp' = R10 persistent K/V bank, re-roped per "
                             "block, never fades. 'novp' ignores --first_frame_edit_dir.")
    parser.add_argument("--blend_sched", type=str, default=None,
                        choices=["paper", "cos_full", "cos_half", "cos_third", "const",
                                 "zero", "first1", "first2", "first3"],
                        help="Blender-rate schedule over the denoising step index. Default "
                             "(unset) and 'paper' both evaluate StreamGVE Eq. 4 unchanged.")
    # R26: two-pass spatial tau. Pass 1 dumps the grounding-mask union M_f; pass 2
    # reads it back and turns Eq. 4's scalar rho into a per-token field.
    parser.add_argument("--union_dump_dir", type=str, default=None,
                        help="R26 pass 1: write M_f = M^src_f | M^trg_f per pair to "
                             "{dir}/edit{T}/{video_name}.npz (packed bool, one row per "
                             "latent frame). Absent => byte-identical to today.")
    parser.add_argument("--split_mask_dump_dir", type=str, default=None,
                        help="R30 diagnostic: like --union_dump_dir, but writes M_src and "
                             "M_trg SEPARATELY (before the OR) to {dir}/edit{T}/"
                             "{video_name}.npz as M_src/M_trg. Investigates why "
                             "0042_gym-ball's (a removal case) union saturates to 100% of "
                             "the frame -- its trg_word is a negation phrase ('without a "
                             "heavy gym ball') with no visual referent. Can be combined "
                             "with --union_dump_dir in the same call (both are pure reads "
                             "of the same computation, no interaction). Absent => not a "
                             "single extra op.")
    parser.add_argument("--union_mask_dir", type=str, default=None,
                        help="R26 pass 2: read {dir}/edit{T}/{video_name}.npz and drive a "
                             "SPATIAL exponent tau(f,p) = tau_bg + (tau_fg - tau_bg)*M_f(p) "
                             "instead of the scalar --blend_power. Requires --tau_bg/--tau_fg.")
    parser.add_argument("--tau_bg", type=float, default=None,
                        help="R26: release exponent OUTSIDE the mask (0 pins the "
                             "background to the source for the whole rollout)")
    parser.add_argument("--tau_fg", type=float, default=None,
                        help="R26: release exponent INSIDE the mask")

    parser.add_argument("--cases_json", type=str, default=None,
                        help="Restrict to the video_names listed in this cases.json whose "
                             "edit_type matches --edit_type (default: every pair in the "
                             "edit{T} json)")

    # grounding / boosting hyper-parameters (paper Self-Forcing defaults; pipeline default omega is 2.0)
    parser.add_argument("--fg_boost_factor", type=float, default=4.0, help="omega, CrossAttn boosting")
    parser.add_argument("--blend_power", type=float, default=2.0, help="rho, self-attn blend strength")
    parser.add_argument("--tau_map", type=str, default=None,
                        help="R25: CSV of per-pair release exponents "
                             "(video_name, edit_type, iou, tau) from r25_tau_map.py. "
                             "When given, tau replaces --blend_power PER PAIR; tau IS "
                             "Eq. 4's rho. Absent => byte-identical to today.")

    # Self-Forcing rollout sampling. Defaults match inference_edit_streamedit.py: a clip
    # that fits in one window takes a single inference() call (identical to the old
    # hardcoded -1), while longer clips are split into overlapping windows instead of
    # overflowing the 21-latent-frame KV cache. -1 forces the single-window path.
    parser.add_argument("--rollout_chunk_size", type=int, default=21,
                        help="Latent frames per rollout window (-1 = single-window inference)")
    parser.add_argument("--rollout_overlap_block_num", type=int, default=1,
                        help="Blocks of overlap between consecutive rollout windows")

    # sampling / model settings (mirrors inference_edit_streamedit.py)
    parser.add_argument("--step", type=int, default=15, help="Sampler steps")
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--src_frames", type=str, default="video",
                        choices=["video", "images"],
                        help="Which rendition of the source to FEED THE MODEL. "
                             "'video' = data_root/videos/{name}.mp4 (historical default, "
                             "what R1/R3/R5/R7/R10/R20/R21 used). 'images' = "
                             "data_root/images/{name}/*.jpg -- the SAME frames "
                             "evaluate.py scores against. The two renditions are NOT "
                             "pixel-identical: measured 2026-08-14, mp4-vs-jpg is "
                             "LPIPS 0.173 / ~30 dB apart, and since the eval reference "
                             "is the jpgs, feeding the mp4 charges that entire gap to "
                             "the method (reported LPIPS 213 vs 53 measured against the "
                             "model's own input). Default left at 'video' so existing "
                             "runs stay reproducible; use 'images' for paper-comparable "
                             "numbers.")
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml",
                        help="Pipeline config (resolved relative to --sf_root)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt",
                        help="Generator checkpoint (resolved relative to --sf_root)")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT),
                        help="Path to the Self-Forcing_StreamEdit build")
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

    # Resolve all external paths to absolute BEFORE chdir-ing into the SF build,
    # so relative config/checkpoint paths (configs/..., checkpoints/...) resolve there
    # while data/output paths stay anchored where the user meant them.
    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    # Resolve the anchor root before chdir-ing into the SF build so a relative
    # --first_frame_edit_dir (e.g. evaluation/anchors) stays anchored at the repo.
    first_frame_edit_dir = (Path(args.first_frame_edit_dir).expanduser().resolve()
                            if args.first_frame_edit_dir else None)
    cases_json = (Path(args.cases_json).expanduser().resolve()
                  if args.cases_json else None)

    # R26 cross-validation, BEFORE the pipeline loads: a misconfigured arm must fail in
    # seconds, not after a GPU-hour of sampling that silently produced the wrong thing.
    if (args.tau_bg is not None or args.tau_fg is not None) and args.union_mask_dir is None:
        raise SystemExit("[run_fivebench] --tau_bg/--tau_fg require --union_mask_dir: "
                         "without the mask there is no field to build, and the taus "
                         "would be accepted and then silently ignored.")
    if args.union_mask_dir is not None and args.tau_bg is None:
        raise SystemExit("[run_fivebench] --union_mask_dir requires --tau_bg: the "
                         "background exponent is always a fixed scalar, whether the "
                         "foreground comes from --tau_fg or per-pair from --tau_map "
                         "(R30).")
    if (args.union_mask_dir is not None and args.tau_fg is None
            and args.tau_map is None):
        raise SystemExit("[run_fivebench] --union_mask_dir requires --tau_fg, unless "
                         "--tau_map is given instead to supply a per-pair foreground "
                         "exponent (R30: SPATIAL exponent, tau_fg varies by pair, "
                         "tau_bg does not).")
    # R30: --union_mask_dir + --tau_map together means "route the SPATIAL foreground
    # exponent per pair" -- the map's `tau` column feeds tau_fg instead of the scalar
    # --tau_fg (R26's fixed-per-arm mode, preserved above when --tau_map is absent).
    # A --tau_fg given alongside would be accepted and then silently ignored, which is
    # exactly the failure class every guard in this block exists to prevent.
    if (args.union_mask_dir is not None and args.tau_fg is not None
            and args.tau_map is not None):
        raise SystemExit("[run_fivebench] --union_mask_dir + --tau_map already drives "
                         "tau_fg PER PAIR from the map; a separate --tau_fg would be "
                         "accepted and then silently ignored. Drop --tau_fg.")
    if args.union_mask_dir is not None and args.blend_sched is not None:
        raise SystemExit("[run_fivebench] --union_mask_dir and --blend_sched are mutually "
                         "exclusive: the R20 schedule overrides the per-step scalar rate "
                         "the spatial field is built on, so combining them would silently "
                         "discard one of the two.")
    if args.union_mask_dir is not None and args.union_dump_dir is not None:
        raise SystemExit("[run_fivebench] --union_dump_dir (pass 1) and --union_mask_dir "
                         "(pass 2) are separate renders; dumping a mask-driven run would "
                         "overwrite the oracle with a second-generation mask.")
    if args.union_mask_dir is not None and args.split_mask_dump_dir is not None:
        raise SystemExit("[run_fivebench] --split_mask_dump_dir has nothing to capture "
                         "during --union_mask_dir (pass 2): that pass reads a STORED "
                         "field and never computes a grounding mask at all.")

    union_dump_dir = (Path(args.union_dump_dir).expanduser().resolve()
                      if args.union_dump_dir else None)
    union_mask_dir = (Path(args.union_mask_dir).expanduser().resolve()
                      if args.union_mask_dir else None)
    split_mask_dump_dir = (Path(args.split_mask_dump_dir).expanduser().resolve()
                           if args.split_mask_dump_dir else None)

    # The Self-Forcing build imports as a top-level package (`from pipeline import ...`)
    # and reads configs via relative paths, so put it on sys.path and make it the cwd.
    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    # Imports that depend on the SF build living on sys.path.
    from inference_edit_streamedit import load_pipe, find_closest_num_frame, read_json
    from pipeline.utils import find_phrase_token_indices
    from utils.misc import set_seed
    from diffusers.utils import load_video

    pipeline, low_memory, device, local_rank = load_pipe(args)
    # tokenizer used only to flag pairs whose trigger words don't tokenize into the prompt
    trans_tokenizer = pipeline.text_encoder.tokenizer.tokenizer

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    edit_json = data_root / "edit_prompt" / f"edit{args.edit_type}_FiVE.json"
    entries = read_json(edit_json)

    # R20: optional subset. Filter on video_name AND edit_type, because a video can
    # appear under two edit types (0011_lucia is in both edit2 and edit5) and only the
    # pair belonging to THIS --edit_type may be run here.
    if cases_json is not None:
        wanted = {c["video_name"] for c in json.loads(cases_json.read_text())
                  if int(c["edit_type"]) == args.edit_type}
        if not wanted:
            raise SystemExit(f"[run_fivebench] {cases_json} lists no case with "
                             f"edit_type={args.edit_type}")
        entries = [e for e in entries if e["video_name"] in wanted]
        found = {e["video_name"] for e in entries}
        if found != wanted:
            raise SystemExit(f"[run_fivebench] cases not present in {edit_json.name}: "
                             f"{sorted(wanted - found)}")

    # R25: per-pair release exponent. Loaded AFTER the --cases_json filter so coverage
    # is checked against the pairs that will actually render. A missing pair is a hard
    # error, never a silent fall back to --blend_power: a half-adaptive arm would still
    # produce 22 clips and a full metrics table, and nothing downstream could tell.
    tau_map = None
    if args.tau_map is not None:
        tau_map = {}
        with open(Path(args.tau_map).expanduser()) as fh:
            for r in csv.DictReader(fh):
                tau_map[(r["video_name"], int(r["edit_type"]))] = float(r["tau"])
        missing = [e["video_name"] for e in entries
                   if (e["video_name"], args.edit_type) not in tau_map]
        if missing:
            raise SystemExit(f"[run_fivebench] --tau_map {args.tau_map} does not cover "
                             f"edit_type={args.edit_type} pairs: {sorted(missing)}")
        print(f"[run_fivebench] tau_map: {len(tau_map)} pairs loaded, "
              f"{len(entries)} needed for edit{args.edit_type}", flush=True)

    # `vp_mode` only means anything when there is an anchor to inject; with no
    # --first_frame_edit_dir this stays the R1 baseline path regardless of the flag.
    vp_mode = args.vp_mode if first_frame_edit_dir is not None else "novp"

    method_dir = out_root / args.method / f"edit{args.edit_type}"
    method_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_fivebench] edit_type={args.edit_type} method={args.method} "
          f"pairs={len(entries)} vp_mode={vp_mode} "
          f"blend_sched={args.blend_sched or 'paper(eq4)'} out={method_dir}", flush=True)
    if union_dump_dir is not None:
        print(f"[run_fivebench] R26 pass 1: dumping M_f -> "
              f"{union_dump_dir}/edit{args.edit_type}/", flush=True)
    if split_mask_dump_dir is not None:
        print(f"[run_fivebench] R30 diagnostic: dumping M_src/M_trg SEPARATELY -> "
              f"{split_mask_dump_dir}/edit{args.edit_type}/", flush=True)
    if union_mask_dir is not None:
        fg_desc = (f"per-pair from {args.tau_map}" if args.tau_map is not None
                  else f"tau_fg={args.tau_fg}")
        print(f"[run_fivebench] R26/R30 pass 2: spatial tau from "
              f"{union_mask_dir}/edit{args.edit_type}/ "
              f"tau_bg={args.tau_bg} {fg_desc}", flush=True)

    rows = []
    for e in entries:
        video_name = e["video_name"]
        pair_id = e.get("id", video_name)
        out_dir = method_dir / video_name
        src_path = data_root / "videos" / f"{video_name}.mp4"
        # Re-seed before EVERY pair. `load_pipe` seeds once at process start, so
        # without this a clip's sampled noise depends on how many clips were
        # rendered before it in the same process -- i.e. on its position in the
        # edit{T} json. That makes any subset run incomparable to a full run: with
        # --cases_json restricted to 2 of edit6's 10 pairs, 0042_gym-ball (index 4)
        # drew different noise and 93% of its pixels differed from the stored
        # baseline, while the index-0 pairs were bit-identical. Seeding per pair
        # makes each pair's result depend only on the pair and the seed.
        #
        # This CHANGES what run_fivebench produces for every pair after index 0,
        # so outputs predating 2026-07-22 (R1 baseline, R7, R3, R5) are NOT
        # reproducible with this code and must not be used as byte references or
        # mixed into a comparison with fresh runs. Re-render what you need.
        set_seed(args.seed)

        try:
            if args.src_frames == "images":
                # Read the SAME jpgs evaluate.py uses as its reference, in the same
                # sorted order it lists them (list_images -> sorted). Frame counts
                # match the mp4 exactly (checked on 0001_bus 80/80, 0002 86/86,
                # 0004 75/75), so this is a drop-in for load_video.
                frame_dir = data_root / "images" / video_name
                frame_paths = sorted(
                    p for p in frame_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
                if not frame_paths:
                    raise ValueError(f"no source frames in {frame_dir}")
                src_video = [Image.open(p).convert("RGB") for p in frame_paths]
            else:
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

            # R7 visual prompting: encode the Qwen-edited oracle first frame (the
            # anchor) alongside the source first frame and feed them as independent
            # first-frame conditions (paper §4.5). Missing anchor -> raise, caught by
            # the per-pair try/except below and logged as an error row.
            # R20 adds a second injection mechanism for the same anchor file:
            #   vp  -> paper SS4.5: anchor cached as an independent first frame, so it
            #          enters kv_cache_trg and fades as the window slides (R7).
            #   pvp -> R10: anchor held in a private K/V bank, re-roped to the current
            #          chunk at every block, so it never fades.
            # The two are mutually exclusive by construction -- pvp must NOT also set
            # trg_initial_latent, or the pipeline raises.
            if vp_mode != "novp":
                anchor_path = first_frame_edit_dir / f"edit{args.edit_type}" / f"{video_name}.png"
                if not anchor_path.exists():
                    raise FileNotFoundError(f"missing anchor {anchor_path}")
                trg_first_frame = Image.open(anchor_path).convert("RGB")

            if vp_mode == "vp":
                # Encode order (src first frame, THEN anchor) and the single trailing
                # clear_cache are load bearing: the VAE carries temporal-conv state
                # between calls, so reordering these two encodes would change the
                # latents and break bit-parity with R7. Do not "tidy" this.
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
                # Only the anchor is encoded -- there is no independent first frame.
                vp_latent = pipeline.vae.encode_to_latent(
                    transform(trg_first_frame).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
                ).to(device=device, dtype=torch.bfloat16)
                pipeline.vae.model.clear_cache()
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = None
            else:                                   # novp -- R1 baseline path, unchanged
                independent_first_frame = False
                src_first_frame_latent = trg_first_frame_latent = vp_latent = None

            tok = find_phrase_token_indices(
                trans_tokenizer, [e["target_prompt"]], [e["target_object"]]
            )
            empty = len(tok[0]) == 0

            pair_tau = (tau_map[(video_name, args.edit_type)]
                        if tau_map is not None else args.blend_power)

            # R30: when a spatial run also carries --tau_map, the map's `tau` column IS
            # the per-pair FOREGROUND exponent -- `pair_tau` above is already exactly
            # that value (same dict, same key), so no second lookup is needed. tau_bg
            # stays the fixed CLI scalar; only tau_fg varies by pair. Without --tau_map
            # this is unchanged from R26: the fixed --tau_fg scalar for the whole arm.
            pair_tau_fg = pair_tau if (union_mask_dir is not None and tau_map is not None) \
                else args.tau_fg

            # R26: the pipeline has no notion of case identity, so the per-pair npz path
            # is composed here. A missing pass-1 mask is a hard error -- falling back to
            # the scalar path would produce a full 22-clip arm that is silently NOT the
            # spatial arm, and nothing downstream could tell.
            union_dump_path = (str(union_dump_dir / f"edit{args.edit_type}" / f"{video_name}.npz")
                               if union_dump_dir is not None else None)
            split_dump_path = (str(split_mask_dump_dir / f"edit{args.edit_type}" / f"{video_name}.npz")
                               if split_mask_dump_dir is not None else None)
            union_mask_path = None
            if union_mask_dir is not None:
                union_mask_path = union_mask_dir / f"edit{args.edit_type}" / f"{video_name}.npz"
                if not union_mask_path.exists():
                    raise FileNotFoundError(f"missing R26 union mask {union_mask_path}")
                union_mask_path = str(union_mask_path)

            edit_video = pipeline.rollout_inference(
                src_video=video_latents,
                src_prompts=e["source_prompt"],
                trg_prompts=e["target_prompt"],
                src_trigger_words=e["source_object"],
                trg_trigger_words=e["target_object"],
                return_latents=False,
                wo_video_decode=False,
                profile=False,
                low_memory=low_memory,
                independent_first_frame=independent_first_frame,
                src_initial_latent=src_first_frame_latent,
                trg_initial_latent=trg_first_frame_latent,
                fg_boost_factor=args.fg_boost_factor,
                #✨ R25: `blend_power` IS Eq. 4's rho at causal_model.py:334, so a
                # per-pair value here is exactly a per-pair W_src = t ** tau.
                blend_power=pair_tau,
                rollout_chunk_size=args.rollout_chunk_size,
                rollout_overlap_block_num=args.rollout_overlap_block_num,
                #✨ R20: both default to the R1/R7 behaviour (no bank, Eq. 4).
                vp_latent=vp_latent,
                blend_sched=args.blend_sched,
                #✨ R26: all four default to None => the call is byte-identical to today.
                #✨ R30: tau_fg is pair_tau_fg -- the map's per-pair value when
                # --tau_map rides along with --union_mask_dir, else R26's fixed scalar.
                union_dump_path=union_dump_path,
                split_dump_path=split_dump_path,
                union_mask_path=union_mask_path,
                tau_bg=args.tau_bg,
                tau_fg=pair_tau_fg,
            )
            pipeline.vae.model.clear_cache()

            n_frames = save_frames_png(edit_video[0], out_dir)
            rows.append((pair_id, video_name, "ok", n_frames, int(empty)))
            print(f"[ok] {video_name}: {n_frames} frames  tau={pair_tau}"
                  f"{' (empty-trigger)' if empty else ''}", flush=True)
        except Exception as ex:  # never silently swallow -- record and continue
            rows.append((pair_id, video_name, f"error:{type(ex).__name__}", 0, ""))
            print(f"[ERROR] {video_name}: {ex}", flush=True)

    manifest = method_dir / "_manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "video_name", "status", "n_frames", "empty_trigger"])
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r[2] == "ok")
    print(f"[run_fivebench] done edit{args.edit_type}: {n_ok}/{len(rows)} ok "
          f"-> {manifest}", flush=True)


if __name__ == "__main__":
    main()
