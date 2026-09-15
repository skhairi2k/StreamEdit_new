from typing import List, Optional, Iterable
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm 

import os
import numpy as np
from PIL import Image

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from .utils import (find_phrase_token_indices, _schedule_blend_rate,
                    build_rho_field_from_mask, blender_rate_from_rho)

class EditCausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _check_prompts(self, *args):
        ret_list = []
        for itm in args:
            if isinstance(itm, str):
                itm = [itm]
            ret_list.append(itm)
        return ret_list


    def rollout_inference(
        self,
        src_video: torch.Tensor,
        src_prompts: str | List[str],
        trg_prompts: str | List[str],
        src_trigger_words: str | List[str],
        trg_trigger_words: str | List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        independent_first_frame: bool = False,
        triple_first_frame: bool = False,
        src_initial_latent: Optional[torch.Tensor] = None,  
        trg_initial_latent: Optional[torch.Tensor] = None,

        fg_boost_factor=2.0,
        blend_power=2.0,

        mask_layers: Iterable = range(20),
        enhance_layers: Iterable = range(30),

        fg_scale=1.0,
        reuse_noise_temporal_mean=True,

        rollout_chunk_size: int = 21,
        rollout_overlap_block_num: int = 1,

        #✨ R10: head-gated persistent visual-prompt injection
        blend_off: bool = False,
        #✨ R31 probe: inject the WHOLE source K/V (fg + bg) at EVERY denoising step
        # instead of background-only over the second half. Diagnostic; False =>
        # every path below is bit-for-bit what it was.
        src_kv_full: bool = False,
        vp_latent: Optional[torch.Tensor] = None,
        vp_head_gate: Optional[torch.Tensor] = None,

        #✨ R32: [num_layers, num_heads] per-head additive logit bias on the
        # target branch's previous-frames key segment. None => untouched.
        seg_bias_table: Optional[torch.Tensor] = None,

        #✨ R20: named blend-rate schedule (None / "paper" => Eq. 4 unchanged)
        blend_sched: Optional[str] = None,

        #✨ R22: per-denoising-step latent dump. Pass a list to have this call append ONE
        # [S, B, F, C, H, W] tensor covering the whole video: each window's per-step
        # buffers are stitched with exactly the same overlap rule the final latents use.
        step_dump: Optional[list] = None,

        #✨ R26 pass 1: write the grounding-mask union M_f = M^src_f | M^trg_f -- the
        # tensor the pipeline ALREADY computes at denoising index len//2 (t_inj = 0.5) --
        # to this .npz path, one row per latent frame of the finished video. A full path
        # rather than the plan's `union_dump_dir`, because the pipeline has no notion of
        # case identity: the caller (run_fivebench.py) composes {dir}/edit{T}/{video}.npz.
        # None => not a single extra op, and the call is bit-for-bit what it was.
        union_dump_path: Optional[str] = None,

        #✨ R30 diagnostic: like union_dump_path, but writes M_src and M_trg SEPARATELY
        # (see `inference`'s split_dump / `_save_split_mask`). None (default, every
        # normal render) => not a single extra op.
        split_dump_path: Optional[str] = None,

        #✨ R26 pass 2: read that .npz back and drive a SPATIAL release exponent
        # tau(f, p) = tau_bg + (tau_fg - tau_bg) * M_f(p) instead of the scalar
        # `blend_power`. tau_bg == tau_fg is the degenerate control and must reproduce
        # Eq. 4 exactly. All three are required together.
        union_mask_path: Optional[str] = None,
        tau_bg: Optional[float] = None,
        tau_fg: Optional[float] = None,

        #✨ R31: a CONTINUOUS per-token exponent field -- the spatial generalisation of
        # R26's two-level (tau_bg, tau_fg). float32 [F_video, frame_seq_length] in
        # ABSOLUTE latent-frame coordinates, and ALREADY exponents: the d -> tau mapping
        # happens offline in evaluation/r31_rho_map.py, so nothing here needs a schedule
        # or an A_disc inversion. Mutually exclusive with the R26 mask path.
        # None => every R26 and baseline path is bit-identical.
        rho_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #✨ R26: resolve the two passes before anything is rendered, so a misconfigured
        # arm fails at call time rather than after a GPU-hour of sampling.
        if (union_mask_path is not None) != (tau_bg is not None and tau_fg is not None):
            raise ValueError(
                "R26: union_mask_path, tau_bg and tau_fg must be given together "
                f"(got union_mask_path={union_mask_path!r}, tau_bg={tau_bg}, tau_fg={tau_fg})."
            )
        if union_mask_path is not None and union_dump_path is not None:
            raise ValueError(
                "R26: pass 1 (union_dump_path) and pass 2 (union_mask_path) are separate "
                "renders -- dumping the union of a run that is itself driven by a mask "
                "would silently overwrite the oracle with a second-generation mask."
            )
        if union_mask_path is not None and blend_sched not in (None, "paper"):
            raise ValueError(
                f"R26: --blend_sched {blend_sched!r} overrides the per-step scalar rate "
                "that the spatial field is built on, so combining them would discard one "
                "of the two. Run the field against Eq. 4 only."
            )

        #✨ R31: refuse the two spatial paths together, and validate the field BEFORE
        # the model runs. R26 builds tau(f,p) from a bool mask + two scalars; R31 hands
        # tau(f,p) over directly. Accepting both would leave it ambiguous which field
        # reached the attention kernel.
        if rho_frames is not None:
            if union_mask_path is not None or tau_bg is not None or tau_fg is not None:
                raise ValueError(
                    "R31: rho_frames is mutually exclusive with R26's union_mask_path / "
                    f"tau_bg / tau_fg (got union_mask_path={union_mask_path!r}, "
                    f"tau_bg={tau_bg}, tau_fg={tau_fg})."
                )
            if blend_sched not in (None, "paper"):
                raise ValueError(
                    f"R31: --blend_sched {blend_sched!r} overrides the per-step scalar "
                    "rate the spatial field is built on, so combining them would discard "
                    "one of the two. Run the field against Eq. 4 only."
                )
            if rho_frames.dim() != 2:
                raise ValueError(
                    f"R31: rho_frames must be [F_video, frame_seq_length], got "
                    f"{tuple(rho_frames.shape)}."
                )
            # Mirrors R26's frame-count refusal: a short field must fail LOUDLY. Silent
            # truncation would misalign every frame after the first chunk.
            if rho_frames.shape[0] != src_video.shape[1]:
                raise ValueError(
                    f"R31: rho_frames holds {rho_frames.shape[0]} latent frames but this "
                    f"clip has {src_video.shape[1]}. Rebuild the field for this pair."
                )
            if rho_frames.shape[1] != self.frame_seq_length:
                raise ValueError(
                    f"R31: rho_frames has frame_seq_length {rho_frames.shape[1]}, but "
                    f"this model uses {self.frame_seq_length}."
                )
            if not bool(torch.isfinite(rho_frames).all()):
                raise ValueError("R31: rho_frames holds a non-finite exponent.")
            if bool((rho_frames < 0).any()):
                raise ValueError(
                    "R31: rho_frames holds a negative exponent; W_src = t ** rho would "
                    "diverge as t -> 0."
                )
            rho_frames = rho_frames.to(device=src_video.device, dtype=torch.float32)

        union_frames = None
        if union_mask_path is not None:
            union_frames = self._load_union_mask(union_mask_path, src_video.device)
            if union_frames.shape[0] != src_video.shape[1]:
                raise ValueError(
                    f"R26: {union_mask_path} holds {union_frames.shape[0]} latent frames but "
                    f"this clip has {src_video.shape[1]}. A silent truncation would misalign "
                    "every frame after the first chunk -- re-dump the mask for this pair."
                )
            if union_frames.shape[1] != self.frame_seq_length:
                raise ValueError(
                    f"R26: {union_mask_path} has frame_seq_length {union_frames.shape[1]}, "
                    f"but this model uses {self.frame_seq_length}."
                )

        #✨ R26: per-window union buffers, stitched below under the same overlap rule as
        # `ret_latent_list`. `inference` appends one entry per call, so this list stays
        # index-aligned with the windows.
        ret_union_list = [] if union_dump_path is not None else None
        #✨ R30 diagnostic: per-window (src, trg) pairs, same overlap rule.
        ret_split_list = [] if split_dump_path is not None else None

        if rollout_chunk_size < 0:
            # for testing local attn
            single_window_out = self.inference(
                src_video=src_video,
                src_prompts=src_prompts,
                trg_prompts=trg_prompts,
                src_trigger_words=src_trigger_words,
                trg_trigger_words=trg_trigger_words,
                return_latents=return_latents,
                wo_video_decode=wo_video_decode,
                profile=profile,
                low_memory=low_memory,

                independent_first_frame=independent_first_frame,
                triple_first_frame=triple_first_frame,
                src_initial_latent=src_initial_latent,
                trg_initial_latent=trg_initial_latent,

                mask_layers=mask_layers,
                enhance_layers=enhance_layers,
                reuse_noise_temporal_mean=reuse_noise_temporal_mean,

                fg_scale=fg_scale,
                fg_boost_factor=fg_boost_factor,

                blend_power=blend_power,

                blend_off=blend_off,
                src_kv_full=src_kv_full,
                vp_latent=vp_latent,
                vp_head_gate=vp_head_gate,
                seg_bias_table=seg_bias_table,

                blend_sched=blend_sched,

                step_dump=step_dump,

                #✨ R26: the single-window path is one call, so its window IS the whole
                # video: abs_frame_offset 0 and no overlap to drop.
                union_dump=ret_union_list,
                split_dump=ret_split_list,
                union_frames=union_frames,
                abs_frame_offset=0,
                tau_bg=tau_bg,
                tau_fg=tau_fg,
                rho_frames=rho_frames,          #✨ R31
            )
            #✨ R26: the single-window path returns early, so it has to save the dump
            # itself. `ret_union_list` holds exactly one entry, already trimmed the same
            # way the returned latents are.
            if union_dump_path is not None:
                self._save_union_mask(union_dump_path, torch.cat(ret_union_list, dim=1))
            #✨ R30 diagnostic: same early-return save, for the (src, trg) pair.
            if split_dump_path is not None:
                src0, trg0 = ret_split_list[0]
                self._save_split_mask(split_dump_path, src0, trg0)
            return single_window_out

        rollout_overlap = rollout_overlap_block_num * self.num_frame_per_block

        total_frame_num = src_video.shape[1]
        ret_latent_list = []
        #✨ R22: per-window step buffers, stitched below under the same overlap rule as
        # `ret_latent_list`. `inference` appends one entry per call, so this list stays
        # index-aligned with the windows.
        ret_step_list = [] if step_dump is not None else None
        start_idx = 0

        while True:
            chunk_right_idx = start_idx + rollout_chunk_size
            if (start_idx == 0) and (independent_first_frame or triple_first_frame):
                # provide kv_cache space for image condition
                chunk_right_idx -= self.num_frame_per_block

            if start_idx == 0:
                rollout_src_video = src_video[:, start_idx: chunk_right_idx]
            else:
                rollout_src_video = src_video[:, start_idx + rollout_overlap: chunk_right_idx]

            # inference
            _, rollout_latent = self.inference(
                src_video=rollout_src_video,
                src_prompts=src_prompts,
                trg_prompts=trg_prompts,
                src_trigger_words=src_trigger_words,
                trg_trigger_words=trg_trigger_words,

                return_latents=True,
                wo_video_decode=True,

                profile=profile,
                low_memory=low_memory,

                independent_first_frame=independent_first_frame if start_idx == 0 else False,
                triple_first_frame=triple_first_frame if start_idx == 0 else False,
                
                src_initial_latent=src_initial_latent,
                trg_initial_latent=trg_initial_latent,

                mask_layers=mask_layers,
                enhance_layers=enhance_layers,
                reuse_noise_temporal_mean=reuse_noise_temporal_mean,

                fg_scale=fg_scale,
                fg_boost_factor=fg_boost_factor,

                blend_power=blend_power,

                #✨ R20: forward the R10 anchor flags into EVERY window. The bank is
                # rebuilt and re-stamped per window, so the persistent VP stays
                # "present" for the whole video instead of being dropped after
                # window 1 (the R8 finding about the SS4.5 path).
                blend_off=blend_off,
                src_kv_full=src_kv_full,
                vp_latent=vp_latent,
                vp_head_gate=vp_head_gate,
                seg_bias_table=seg_bias_table,
                blend_sched=blend_sched,
                #✨ R22: `inference` appends this window's [S, B, F, C, H, W] buffer.
                step_dump=ret_step_list,

                #✨ R26: `start_idx` IS the absolute latent-frame index of this window's
                # first RETURNED (post-trim) frame -- window 1 trims the SS4.5 anchor
                # prefix, later windows return their overlap and rollout drops it below.
                # `inference` turns that offset into per-frame M_f rows itself, so no
                # position arithmetic is duplicated here.
                union_dump=ret_union_list,
                split_dump=ret_split_list,
                union_frames=union_frames,
                abs_frame_offset=start_idx,
                tau_bg=tau_bg,
                tau_fg=tau_fg,
                rho_frames=rho_frames,          #✨ R31
            )

            # store results
            if start_idx == 0:
                ret_latent_list.append(rollout_latent)
            else:
                ret_latent_list.append(rollout_latent[:, rollout_overlap: ])

            #✨ R22: drop the overlap from this window's step buffer exactly as the line
            # above drops it from the latents -- but on dim 2, since the step buffer
            # carries a leading step axis. Slicing the wrong axis here would silently
            # misalign frames on multi-window clips ONLY (>21 latent frames, e.g.
            # 0034_cows), which no single-window smoke test can catch.
            if step_dump is not None and start_idx != 0:
                ret_step_list[-1] = ret_step_list[-1][:, :, rollout_overlap: ]

            #✨ R26: same overlap rule again, on the union buffer's frame axis (dim 1).
            if union_dump_path is not None and start_idx != 0:
                ret_union_list[-1] = ret_union_list[-1][:, rollout_overlap: ]

            #✨ R30 diagnostic: same overlap rule, on both the src and trg buffers.
            if split_dump_path is not None and start_idx != 0:
                s, t = ret_split_list[-1]
                ret_split_list[-1] = (s[:, rollout_overlap:], t[:, rollout_overlap:])


            # finish, end loop
            if chunk_right_idx >= total_frame_num:
                break

            # index update
            start_idx = chunk_right_idx - rollout_overlap

            # prepare prev_cond
            src_initial_latent = rollout_src_video[:, -rollout_overlap: ]
            trg_initial_latent = rollout_latent[:, -rollout_overlap: ]

        output = torch.cat(ret_latent_list, dim=1)
        assert src_video.shape == output.shape, 'noise shape: %s, but output: %s.' % (str(src_video.shape), str(output.shape))

        #✨ R22: stitch the windows into one whole-video step buffer and re-assert the L1
        # identity at rollout level. `inference` already checked it per window; this
        # catches a stitching mistake (wrong axis, wrong overlap) that per-window checks
        # cannot see.
        if step_dump is not None:
            full_step = torch.cat(ret_step_list, dim=2)
            assert torch.equal(full_step[-1], output), (
                "R22: stitched step-dump final step != rollout output -- overlap slicing "
                "or concat axis is wrong (check multi-window clips)."
            )
            step_dump.append(full_step)

        #✨ R26: stitch the windows into one whole-video union and re-assert the frame
        # count against the latents. A mismatch here means the overlap slicing above is
        # wrong, which would misalign every pass-2 field row -- exactly the failure this
        # task cannot afford to miss.
        if union_dump_path is not None:
            full_union = torch.cat(ret_union_list, dim=1)
            if full_union.shape[1] != output.shape[1]:
                raise ValueError(
                    f"R26: stitched union has {full_union.shape[1]} latent frames but the "
                    f"rollout produced {output.shape[1]} -- overlap slicing or concat axis "
                    "is wrong (check multi-window clips)."
                )
            self._save_union_mask(union_dump_path, full_union)

        #✨ R30 diagnostic: same stitch-and-check, for the (src, trg) pair.
        if split_dump_path is not None:
            full_src = torch.cat([s for s, _ in ret_split_list], dim=1)
            full_trg = torch.cat([t for _, t in ret_split_list], dim=1)
            if full_src.shape[1] != output.shape[1]:
                raise ValueError(
                    f"R30: stitched split mask has {full_src.shape[1]} latent frames but "
                    f"the rollout produced {output.shape[1]} -- overlap slicing or concat "
                    "axis is wrong (check multi-window clips)."
                )
            self._save_split_mask(split_dump_path, full_src, full_trg)

        # clean cache before decode to avoid OOM
        torch.cuda.empty_cache()

        if wo_video_decode:
            video = None
        else:
            dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            torch.cuda.synchronize()

        if return_latents:
            return video, output
        else:
            return video

    def inference(
        self,
        src_video: torch.Tensor,
        src_prompts: str | List[str],
        trg_prompts: str | List[str],
        src_trigger_words: str | List[str],
        trg_trigger_words: str | List[str],
        return_latents: bool = False,
        wo_video_decode: bool = False,
        profile: bool = False,
        low_memory: bool = False,

        independent_first_frame: bool = False,
        triple_first_frame: bool = False,
        src_initial_latent: Optional[torch.Tensor] = None,  
        trg_initial_latent: Optional[torch.Tensor] = None,

        fg_boost_factor=2.0,
        blend_power=2.0,

        mask_layers: Iterable = range(20),
        enhance_layers: Iterable = range(30),

        fg_scale=1.0,
        reuse_noise_temporal_mean=True,

        #✨ R10: head-gated persistent visual-prompt injection
        blend_off: bool = False,
        #✨ R31 probe: inject the WHOLE source K/V (fg + bg) at EVERY denoising step
        # instead of background-only over the second half. Diagnostic; False =>
        # every path below is bit-for-bit what it was.
        src_kv_full: bool = False,
        vp_latent: Optional[torch.Tensor] = None,
        vp_head_gate: Optional[torch.Tensor] = None,

        #✨ R32: [num_layers, num_heads] per-head additive logit bias on the
        # target branch's previous-frames key segment. None => untouched.
        seg_bias_table: Optional[torch.Tensor] = None,

        #✨ R20: named blend-rate schedule (None / "paper" => Eq. 4 unchanged)
        blend_sched: Optional[str] = None,

        #✨ R22: per-denoising-step latent dump. Pass a list to have this call append
        # one [S, B, F, C, H, W] tensor holding, for every step index s, the target
        # branch's x0 prediction at step s of EVERY block. Default None => not a
        # single extra op, and the call is bit-for-bit what it was before.
        step_dump: Optional[list] = None,

        #✨ R26 pass 1: pass a list to have this call append ONE [B, F, frame_seq_length]
        # bool tensor holding the grounding-mask union M_f for this window, trimmed
        # exactly like the returned latents. `rollout_inference` stitches and saves.
        union_dump: Optional[list] = None,

        #✨ R30 diagnostic: pass a list to have this call append ONE (src, trg) pair of
        # [B, F, frame_seq_length] bool tensors -- the SAME src_fg_mask_bin / trg-only
        # mask that union_dump's M_f = src | trg is built from, captured separately
        # BEFORE the OR. Investigates why 0042_gym-ball's (a removal case) union
        # saturates to 100% of the frame: src_word is a concrete noun phrase ("with a
        # heavy gym ball") but trg_word is a negation ("without a heavy gym ball") with
        # no visual referent, so trg grounding plausibly never localizes. Diagnostic
        # only -- normal renders (run_fivebench.py, every prior task) never pass this,
        # so the call stays bit-for-bit what it was when split_dump is None.
        split_dump: Optional[list] = None,

        #✨ R26 pass 2: [F_video, frame_seq_length] bool M_f for the WHOLE clip, in
        # absolute latent-frame coordinates, plus this window's absolute offset. The
        # release exponent becomes tau(f, p) = tau_bg + (tau_fg - tau_bg) * M_f(p)
        # instead of the scalar `blend_power`. None => untouched scalar path.
        union_frames: Optional[torch.Tensor] = None,
        abs_frame_offset: int = 0,
        tau_bg: Optional[float] = None,
        tau_fg: Optional[float] = None,

        #✨ R31: continuous per-token exponents, float32 [F_video, frame_seq_length], in
        # the SAME absolute latent-frame coordinates as `union_frames` and consumed by the
        # same `_union_rows` indexing. Already tau -- no mapping happens here.
        # Validated in `rollout_inference`. None => untouched.
        rho_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not (independent_first_frame and triple_first_frame)
        #✨ R26
        r26_active = union_frames is not None
        if r26_active and (tau_bg is None or tau_fg is None):
            raise ValueError("R26: union_frames given without tau_bg / tau_fg.")
        #✨ R31: the continuous twin. `spatial_active` is what every downstream site
        # gates on, so R26 and R31 light up EXACTLY the same machinery -- the cache-aligned
        # blend_rho_tok buffer, the per-step rate injection, and the rolling under
        # eviction. The only difference is how `_chunk_rho` produces its exponents.
        r31_active = rho_frames is not None
        if r31_active and r26_active:
            raise ValueError(
                "R31: rho_frames and union_frames are mutually exclusive -- two spatial "
                "exponent fields would race for the same blend_rho_tok buffer."
            )
        spatial_active = r26_active or r31_active
        #✨ R31: whichever field is live supplies the rows. Both are
        # [F_video, frame_seq_length] in absolute latent-frame coordinates, so the
        # indexing, the bounds check and the clamp(min=0) prefix borrow are identical.
        spatial_field = union_frames if r26_active else rho_frames
        #✨ R20: narrowed from "vp_latent xor trg_initial_latent". The rule being
        # enforced is that the ANCHOR must not also be cached as an initial latent
        # (the fading SS4.5 path) -- that misuse always arrives as
        # independent_first_frame=True. Rollout windows >= 2 also pass a
        # trg_initial_latent, but that is overlap seeding from the previous window
        # (rollout_inference), which is unrelated and must be allowed through or
        # persistent VP cannot survive a window boundary.
        if vp_latent is not None and trg_initial_latent is not None and independent_first_frame:
            raise ValueError(
                "R10: vp_latent and an independent-first-frame trg_initial_latent are "
                "mutually exclusive. The persistent VP must not also be cached as an "
                "initial latent (paper SS4.5 path), or it would enter the main KV "
                "cache and fade."
            )
        independent_first_frame = independent_first_frame or self.independent_first_frame

        #✨ R26: `output` is trimmed by exactly this many leading frames at the end of
        # the call (see Step 4), so LOCAL output row j corresponds to ABSOLUTE latent
        # frame `abs_frame_offset + j - num_trim`. Deriving it from the same two flags
        # the trim uses is what keeps the dumped mask and the pass-2 field in the same
        # coordinate system as the rendered video.
        num_trim = (1 if independent_first_frame else 0) + (3 if triple_first_frame else 0)

        def _union_rows(local_left: int, local_right: int) -> torch.Tensor:
            '''✨ R26: M_f rows [n, frame_seq_length] for local output frames
            [local_left, local_right).'''
            idx = torch.arange(local_left, local_right, device=spatial_field.device) \
                + abs_frame_offset - num_trim
            hi = int(idx.max().item())
            if hi >= spatial_field.shape[0]:
                raise ValueError(
                    f"R26/R31: the field has {spatial_field.shape[0]} latent frames but this "
                    f"window asked for absolute frame {hi} (local [{local_left}, {local_right}), "
                    f"offset {abs_frame_offset}, trim {num_trim}). A silent truncation would "
                    "misalign every frame after this chunk."
                )
            # Only the trimmed prefix (the SS4.5 anchor frame, j < num_trim) can index
            # before frame 0. It is never part of the rendered video, but its key/value
            # pair does sit in the cache and gets blended as a PREVIOUS key, so it needs
            # an exponent: it borrows absolute frame 0's, the frame it anchors.
            return spatial_field[idx.clamp(min=0)]

        def _chunk_rho(local_left: int, local_right: int) -> torch.Tensor:
            '''✨ R26: [B, (local_right - local_left) * frame_seq_length] float32 exponents.'''
            rows = _union_rows(local_left, local_right).reshape(-1)
            #✨ R31: `rows` ARE the exponents already -- r31_rho_map.py did the d -> tau
            # mapping offline -- so there is no two-level field to build, just a widen to
            # batch. Everything downstream treats the result as opaque float32.
            if r31_active:
                return rows.unsqueeze(0).expand(batch_size, -1).to(torch.float32)
            return build_rho_field_from_mask(
                rows.unsqueeze(0).expand(batch_size, -1), tau_bg, tau_fg
            )
        
        #✨ R10: defensive clear. The driver runs many cases/arms in one process; a
        # crash mid-run must not leave a stale VP bank stamped on the modules and
        # silently contaminate the next arm.
        self._clear_vp_stamps()
        #✨ R32: same hazard, same remedy -- a stale bias table would silently
        # make the next arm a different experiment.
        self._clear_seg_bias_stamps()

        batch_size, num_frames, num_channels, height, width = src_video.shape
        if not independent_first_frame or (independent_first_frame and trg_initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = trg_initial_latent.shape[1] if trg_initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        src_prompts, trg_prompts, src_trigger_words, trg_trigger_words = self._check_prompts(
            src_prompts, trg_prompts, src_trigger_words, trg_trigger_words
        )
        conditional_dict = self.text_encoder(
            text_prompts=src_prompts + trg_prompts
        )
        src_conditional_dict = self.text_encoder(
            text_prompts=src_prompts
        )
        trg_conditional_dict = self.text_encoder(
            text_prompts=trg_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=src_video.device,
            dtype=src_video.dtype
        )

        #✨ R26 pass 1: one row per latent frame of this window, in the SAME local
        # coordinates as `output`, so the trims in Step 4 apply to both identically.
        union_out = (
            torch.zeros([batch_size, num_output_frames, self.frame_seq_length],
                        dtype=torch.bool, device=src_video.device)
            if union_dump is not None else None
        )
        #✨ R30 diagnostic: same shape and coordinate system as union_out, captured
        # in parallel wherever union_out is written.
        src_mask_out = (
            torch.zeros([batch_size, num_output_frames, self.frame_seq_length],
                        dtype=torch.bool, device=src_video.device)
            if split_dump is not None else None
        )
        trg_mask_out = (
            torch.zeros([batch_size, num_output_frames, self.frame_seq_length],
                        dtype=torch.bool, device=src_video.device)
            if split_dump is not None else None
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache, trg_fg_mask cache, and crossattn cache
        kv_cache_src = self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        kv_cache_trg = self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        trg_fg_mask_cache = self._initialize_trg_fg_mask_cache(
            batch_size=batch_size,
            device=src_video.device,
            #✨ R26: allocates the cache-aligned exponent buffer only when pass 2 runs.
            #✨ R31: same buffer, but the fill value for never-read cache slots is the
            # field's MINIMUM (its most source-preserving exponent); `tau_bg` does not
            # exist on this path.
            blend_rho_init=(tau_bg if r26_active else
                            (float(rho_frames.min()) if r31_active else None)),
        )
        crossattn_cache_src = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        crossattn_cache_trg = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=src_video.dtype,
            device=src_video.device
        )
        crossattn_cache_dual = self._initialize_crossattn_cache(
            batch_size=batch_size * 2,
            dtype=src_video.dtype,
            device=src_video.device
        )
        # Initialize some helper
        self._initialize_noise_statistics(reuse_noise_temporal_mean)

        # get trigger token indices
        trans_tokenizer = self.text_encoder.tokenizer.tokenizer
        tok_src = find_phrase_token_indices(trans_tokenizer, src_prompts, src_trigger_words)
        tok_trg = find_phrase_token_indices(trans_tokenizer, trg_prompts, trg_trigger_words)
        print(tok_src, tok_trg)

        # Step 2: Cache context feature
        current_start_frame = 0
        if trg_initial_latent is not None:
            # obtain both kv_cache and mask of both src and trg
            timestep = torch.zeros([batch_size, 1], device=src_video.device, dtype=torch.int64)
            if independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                slice_list = [(0, 1)] + [
                    (1 + idx * self.num_frame_per_block, 1 + (idx + 1) * self.num_frame_per_block)
                    for idx in range(num_input_blocks)
                ]
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block
                slice_list = [
                    (idx * self.num_frame_per_block, (idx + 1) * self.num_frame_per_block)
                    for idx in range(num_input_blocks)
                ]

            for left, right in slice_list:
                context_timestep = torch.ones(
                    [batch_size, right - left], device=src_video.device, dtype=torch.float32
                ) * self.args.context_noise
                #✨ src and src mask
                current_src_ref_latents = src_initial_latent[:, left: right]
                self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
                self.generator(
                    noisy_image_or_video=current_src_ref_latents,
                    conditional_dict=src_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_src,
                    crossattn_cache=crossattn_cache_src,
                    current_start=left * self.frame_seq_length,
                )
                _, src_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_src)
                #✨ trg and trg mask
                current_trg_ref_latents = trg_initial_latent[:, left: right]
                self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
                self.generator(
                    noisy_image_or_video=current_trg_ref_latents,
                    conditional_dict=trg_conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache_trg,
                    crossattn_cache=crossattn_cache_trg,
                    current_start=left * self.frame_seq_length,
                )
                _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)

                #✨ src & trg union
                current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
                #✨ R26: the context prefix's keys sit in the cache and are blended as
                # PREVIOUS keys during the first block, so they need an exponent too.
                self._update_trg_fg_mask_cache(
                    trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg,
                    current_rho=(_chunk_rho(left, right) if spatial_active else None),
                )

                output[:, left: right] = current_trg_ref_latents
                if union_out is not None:
                    union_out[:, left: right] = current_trg_fg_mask.view(
                        batch_size, right - left, self.frame_seq_length)
                #✨ R30 diagnostic: src_fg_mask_bin / trg_fg_mask_bin are the exact two
                # operands of `current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin`
                # a few lines above -- captured here, before the OR, unchanged.
                if src_mask_out is not None:
                    src_mask_out[:, left: right] = src_fg_mask_bin.view(
                        batch_size, right - left, self.frame_seq_length)
                    trg_mask_out[:, left: right] = trg_fg_mask_bin.view(
                        batch_size, right - left, self.frame_seq_length)
                current_start_frame = right

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        #✨ R10: build the persistent visual-prompt bank and stamp it on the
        # self-attention modules. The bank lives outside kv_cache_src/trg, so it is
        # never touched by the sliding-window write path and never evicts; the bridge
        # re-ropes it to the current chunk at every block.
        if vp_latent is not None:
            vp_bank = self.build_vp_bank(vp_latent, trg_conditional_dict)
            self._stamp_vp(vp_bank, vp_head_gate)

        #✨ R32: stamp the per-head segment bias. Anchor-free by design -- the
        # bridge refuses the combination, so this is stamped only when no VP
        # bank was.
        if seg_bias_table is not None:
            self._stamp_seg_bias(seg_bias_table)

        # Step 3: Temporal denoising loop
        denoising_step_list = self.denoising_step_list
        #✨ R22: one full-video buffer per denoising step. Cloned from `output` HERE --
        # after Step 2 has written the initial-latent prefix (the SS4.5 anchor frame
        # under vp, the overlap seeding on rollout windows >= 2) -- so every step
        # slice carries that prefix exactly as the final `output` does. The block loop
        # below then overwrites only the frames it generates, which is what makes
        # step_out[-1] identical to `output` rather than merely close to it.
        step_out = (
            output.unsqueeze(0).repeat(len(denoising_step_list), *([1] * output.ndim))
            if step_dump is not None else None
        )
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if independent_first_frame and trg_initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in tqdm(all_num_frames):
            if profile:
                block_start.record()

            src_input = src_video[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            denoised_pred = src_input

            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=src_video.device,
                dtype=torch.float32
            ) * self.args.context_noise
            
            # obtain currently inprocessed kv_cache for dual branch
            shared_dict_dual = dict()
            shared_dict_dual['blend_off'] = blend_off   #✨ R10 (keeps bg source-KV injection)
            shared_dict_dual['src_kv_full'] = src_kv_full   #✨ R31 probe
            kv_cache_dual = self._concat_kv_cache(kv_cache_src, kv_cache_trg, shared_dict=shared_dict_dual)

            #✨ forward clean source video to get source mask, and store into kv_cache
            self._register_crossattn_mask_gatherer(crossattn_cache_src, tok_src, layers=mask_layers, fg_scale=fg_scale)
            self.generator(
                noisy_image_or_video=src_input,
                conditional_dict=src_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_src,
                crossattn_cache=crossattn_cache_src,
                current_start=current_start_frame * self.frame_seq_length,
            )
            _, src_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_src)
            # inject to kv_cache
            self._inject_masks_to_kv_cache(
                kv_cache_dual, trg_fg_mask_cache, src_fg_mask_bin, 
            )
            src_fg_mask_map = self._mask_reshape(
                src_fg_mask_bin, size=(current_num_frames, height, width)
            )
            inloop_trg_fg_mask = src_fg_mask_bin

            #✨ R26: this chunk's exponents. Timestep-independent, so built once per
            # block; the per-step rate is derived from it inside the loop below.
            chunk_rho = (
                _chunk_rho(current_start_frame, current_start_frame + current_num_frames)
                if spatial_active else None
            )

            # Step 3.1: Spatial denoising loop
            noisy_pred_input = None
            for index, current_timestep in tqdm(enumerate(denoising_step_list), total=len(denoising_step_list), leave=False):
                
                # set current timestep
                timestep = torch.ones(
                    [batch_size * 2, current_num_frames],
                    device=src_video.device,
                    dtype=torch.float32
                ) * current_timestep
                timestep_next = denoising_step_list[index + 1] / 1000 if (index < len(denoising_step_list) - 1) else 0
                shared_dict_dual['current_timestep_next'] = float(timestep_next)
                shared_dict_dual['current_timestep'] = float(current_timestep / 1000)
                shared_dict_dual['current_timestep_index'] = index
                shared_dict_dual['total_timestep'] = len(denoising_step_list)
                shared_dict_dual['blend_power'] = blend_power
                #✨ R20: install the scheduled rate for this step, or None to leave
                # the bridge on Eq. 4. `_schedule_blend_rate` returns the SOURCE
                # weight s(p), so the rate handed to the bridge is 1 - s(p).
                shared_dict_dual['blender_rate'] = (
                    None if blend_sched in (None, "paper")
                    else 1.0 - _schedule_blend_rate(blend_sched, index, len(denoising_step_list))
                )
                #✨ R26: the spatial twin of the line above -- W^src(t) = t ** tau(p)
                # evaluated per token instead of once. Published every step because the
                # rate depends on t; the EXPONENT field it is built from does not and is
                # rolled with the mask cache.
                if spatial_active:
                    rate_prev, comp_prev = blender_rate_from_rho(
                        trg_fg_mask_cache['blend_rho_tok'], timestep_next)
                    rate_cur, comp_cur = blender_rate_from_rho(chunk_rho, timestep_next)
                    self._inject_blend_rates_to_kv_cache(
                        kv_cache_dual, rate_prev, comp_prev, rate_cur, comp_cur)
                # use previous statistics on noise
                fwd_noise = torch.randn_like(src_input)
                fwd_noise = self._reuse_noise_statistics(fwd_noise, index, fg_mask=src_fg_mask_map)
                fwd_trg_noise = fwd_noise

                # update mask with trg mask at t^inj=0.5
                if index == len(denoising_step_list) // 2:
                    self._register_crossattn_mask_gatherer(crossattn_cache_dual, tok_src + tok_trg, layers=mask_layers, fg_scale=fg_scale)

                if fg_boost_factor != 1.0:
                    self._register_crossattn_enhancement(
                        crossattn_cache_dual, tok_src + tok_trg, 
                        layers=enhance_layers, fg_boost_factor=fg_boost_factor,
                        current_src_fg_mask=inloop_trg_fg_mask,
                    )

                # add noise to both source video and generating video
                noisy_src_input = self.scheduler.add_noise(
                    src_input.flatten(0, 1),
                    fwd_noise.flatten(0, 1),
                    timestep[: batch_size],
                ).unflatten(0, src_input.shape[:2])
                noisy_pred_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    fwd_trg_noise.flatten(0, 1),
                    timestep[batch_size: ],
                ).unflatten(0, denoised_pred.shape[:2])
                noisy_input = torch.cat([noisy_src_input, noisy_pred_input], dim=0)

                # model forward
                velocity_pred, _ = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache_dual,
                    crossattn_cache=crossattn_cache_dual,
                    current_start=current_start_frame * self.frame_seq_length
                )
                # for getting real output
                t_i = current_timestep / 1000
                v_src, v_trg = velocity_pred.chunk(2, dim=0)
                v_gt = fwd_noise - src_input
                
                #✨ source-oriented guidance
                fg_mask = (v_trg - v_src).abs().mean(dim=2, keepdim=True)     # [B, F, 1, H, W]
                data_dims = list(range(fg_mask.ndim))[1: ]
                fg_mask = (fg_mask - fg_mask.amin(dim=data_dims, keepdim=True)) / \
                    (fg_mask.amax(dim=data_dims, keepdim=True) - fg_mask.amin(dim=data_dims, keepdim=True) + 1e-7)
                bg_mask = 1 - fg_mask
                v_t = v_trg + bg_mask * (v_gt - v_src)
                denoised_pred = noisy_pred_input - t_i * v_t

                #✨ R22: record this step's x0 prediction for the current block. Written
                # AFTER the S.O.G. update, so index 0 already holds the first real model
                # estimate rather than the `denoised_pred = src_input` initialisation.
                # At index == len(denoising_step_list) - 1 this is the same tensor Step
                # 3.2 below writes into `output`.
                if step_dump is not None:
                    step_out[index][:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                #✨ target mask grounding
                if index == len(denoising_step_list) // 2:
                    _, inloop_src_trg_fg_mask_bin, mask_soft_vis, mask_bin_vis = self._aggregate_crossattn_mask(
                        crossattn_cache_dual, size=(current_num_frames, height, width), scale_factor=16
                    )
                    inloop_trg_fg_mask_bin = inloop_src_trg_fg_mask_bin.chunk(2, dim=0)[1]
                    # inject union of origin src and in-processing trg masks to kv_cache
                    inloop_trg_fg_mask = inloop_trg_fg_mask_bin | src_fg_mask_bin
                    #✨ R26 pass 1: this IS M_f -- no new mask maths, just a copy out.
                    if union_out is not None:
                        union_out[:, current_start_frame: current_start_frame + current_num_frames] = \
                            inloop_trg_fg_mask.view(
                                batch_size, current_num_frames, self.frame_seq_length)
                    #✨ R30 diagnostic: src_fg_mask_bin / inloop_trg_fg_mask_bin are the
                    # exact two operands of `inloop_trg_fg_mask = inloop_trg_fg_mask_bin |
                    # src_fg_mask_bin` a few lines above -- captured before the OR. This
                    # is the site that fires for a single-window clip (0042_gym-ball:
                    # 18 latent frames, under the 21-frame rollout_chunk_size).
                    if src_mask_out is not None:
                        src_mask_out[:, current_start_frame: current_start_frame + current_num_frames] = \
                            src_fg_mask_bin.view(
                                batch_size, current_num_frames, self.frame_seq_length)
                        trg_mask_out[:, current_start_frame: current_start_frame + current_num_frames] = \
                            inloop_trg_fg_mask_bin.view(
                                batch_size, current_num_frames, self.frame_seq_length)
                    self._inject_masks_to_kv_cache(
                        kv_cache_dual, trg_fg_mask_cache, inloop_trg_fg_mask, 
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            del kv_cache_dual
            self._kv_cache_to(kv_cache_trg, 'cuda', low_memory)
            self._register_crossattn_mask_gatherer(crossattn_cache_trg, tok_trg, layers=mask_layers, fg_scale=fg_scale)
            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=trg_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache_trg,
                crossattn_cache=crossattn_cache_trg,
                current_start=current_start_frame * self.frame_seq_length,
            )
            #✨ store clean target kv cache, and obtain clean target mask
            _, trg_fg_mask_bin, _, _ = self._aggregate_crossattn_mask(crossattn_cache_trg)
            current_trg_fg_mask = trg_fg_mask_bin | src_fg_mask_bin
            self._update_trg_fg_mask_cache(
                trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg,
                #✨ R26: roll this chunk's exponents alongside its mask, so the next
                # block reads them as previous-token rates at the right offsets.
                current_rho=chunk_rho,
            )
            self._kv_cache_to(kv_cache_trg, 'cpu', low_memory)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        if independent_first_frame:
            output = output[:, 1: ]
        if triple_first_frame:
            output = output[:, 3: ]

        #✨ R22: mirror the same trims on the step buffer (its frame axis is dim 2, not
        # dim 1, because of the leading step axis) so the dumped videos are frame-aligned
        # with `output`. The assert is the L1 identity gate: the last denoising step must
        # BE the returned latent, not merely resemble it. It costs one comparison per
        # window and turns any future reordering of Step 3.2 into a loud failure.
        if step_dump is not None:
            if independent_first_frame:
                step_out = step_out[:, :, 1: ]
            if triple_first_frame:
                step_out = step_out[:, :, 3: ]
            assert torch.equal(step_out[-1], output), (
                "R22: step-dump final step diverged from `output` -- the per-step buffer "
                "is no longer being written from the same tensor Step 3.2 records."
            )
            step_dump.append(step_out)

        #✨ R26: the same trims on the union buffer (frame axis is dim 1 here), so the
        # dumped M_f rows are in the same coordinate system as the returned latents --
        # which is what `abs_frame_offset` assumes on the pass-2 side.
        if union_out is not None:
            if independent_first_frame:
                union_out = union_out[:, 1: ]
            if triple_first_frame:
                union_out = union_out[:, 3: ]
            if union_out.shape[1] != output.shape[1]:
                raise ValueError(
                    f"R26: union buffer has {union_out.shape[1]} frames but `output` has "
                    f"{output.shape[1]} -- the trims have drifted apart."
                )
            union_dump.append(union_out.cpu())

        #✨ R30 diagnostic: same trims, on the same frame axis, as the union buffer above.
        if src_mask_out is not None:
            if independent_first_frame:
                src_mask_out = src_mask_out[:, 1: ]
                trg_mask_out = trg_mask_out[:, 1: ]
            if triple_first_frame:
                src_mask_out = src_mask_out[:, 3: ]
                trg_mask_out = trg_mask_out[:, 3: ]
            if src_mask_out.shape[1] != output.shape[1]:
                raise ValueError(
                    f"R30: split-mask buffer has {src_mask_out.shape[1]} frames but "
                    f"`output` has {output.shape[1]} -- the trims have drifted apart."
                )
            split_dump.append((src_mask_out.cpu(), trg_mask_out.cpu()))

        if wo_video_decode:
            video = None
        else:
            dec_latent = output
            video = self.vae.decode_to_pixel(dec_latent, use_cache=False)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video


    def build_vp_bank(self, vp_latent, conditional_dict):
        """Build the persistent visual-prompt K/V bank (R10).

        Runs ONE clean forward of the anchor latent through a third, private KV
        cache -- parallel to ``kv_cache_src``/``kv_cache_trg`` but never handed to
        the generator again. Because it is not in the sliding-window write path it
        never evicts, unlike the paper's SS4.5 initial-latent caching which writes
        the anchor into ``kv_cache_trg`` and lets it fade as the window advances.

        The stored keys are UNROPED (the ``sink_size == 0`` path writes raw ``k``),
        which is what makes the bridge's per-block re-roping to
        ``last_chunk_start_frame`` legal -- that is the "always present" behaviour.

        Returns a list of ``(k, v)`` tensors, one per transformer block, each
        ``[B, num_vp_frames * frame_seq_length, num_heads, head_dim]``.
        """
        batch_size, num_vp_frames = vp_latent.shape[0], vp_latent.shape[1]
        device, dtype = vp_latent.device, vp_latent.dtype

        anchor_cache = self._initialize_kv_cache(batch_size, dtype, device)
        anchor_crossattn_cache = self._initialize_crossattn_cache(batch_size, dtype, device)

        context_timestep = torch.ones(
            [batch_size, num_vp_frames], device=device, dtype=torch.float32
        ) * self.args.context_noise

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=vp_latent,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=anchor_cache,
                crossattn_cache=anchor_crossattn_cache,
                current_start=0,
            )

        num_vp_tokens = num_vp_frames * self.frame_seq_length
        # Copy the written slice out so the ~6 GB of unused preallocated cache
        # (32760 tokens/layer) can be released; the bank keeps only what was filled.
        vp_bank = [
            (layer["k"][:, :num_vp_tokens].clone(), layer["v"][:, :num_vp_tokens].clone())
            for layer in anchor_cache
        ]
        del anchor_cache, anchor_crossattn_cache

        print(f"[r10] vp_bank built: {len(vp_bank)} layers x {num_vp_tokens} tokens "
              f"({num_vp_frames} latent frame(s))")
        return vp_bank

    def _stamp_vp(self, vp_bank, vp_head_gate):
        """Attach the VP bank + per-head gate to each self-attention module."""
        blocks = self.generator.model.blocks
        if len(vp_bank) != len(blocks):
            raise ValueError(f"vp_bank has {len(vp_bank)} layers but model has {len(blocks)} blocks")
        for layer_idx, blk in enumerate(blocks):
            gate = None
            if vp_head_gate is not None:
                gate = vp_head_gate[layer_idx].to(device=vp_bank[layer_idx][0].device)
            blk.self_attn.layer_idx = layer_idx
            blk.self_attn.vp_kv = vp_bank[layer_idx]
            blk.self_attn.vp_gate = gate

    def _clear_vp_stamps(self):
        """Remove any VP bank left on the modules (see defensive clear in inference)."""
        for blk in self.generator.model.blocks:
            blk.self_attn.vp_kv = None
            blk.self_attn.vp_gate = None

    def _stamp_seg_bias(self, seg_bias_table):
        """Attach one [num_heads] bias row per self-attention module (R32).

        The module has no intrinsic layer index, so -- as with the VP bank --
        the pipeline is what supplies the per-layer row. The bias applies to
        the target branch's previous-frames key segment; the current-frame
        segments stay at 0, which spans the whole past-vs-current axis because
        softmax is shift-invariant per query row.
        """
        blocks = self.generator.model.blocks
        if seg_bias_table.shape[0] != len(blocks):
            raise ValueError(
                f"seg_bias_table has {seg_bias_table.shape[0]} layers but model "
                f"has {len(blocks)} blocks"
            )
        n_heads = blocks[0].self_attn.num_heads
        if seg_bias_table.shape[1] != n_heads:
            raise ValueError(
                f"seg_bias_table has {seg_bias_table.shape[1]} heads but model "
                f"has {n_heads} heads per layer"
            )
        device = next(self.generator.parameters()).device
        for layer_idx, blk in enumerate(blocks):
            blk.self_attn.layer_idx = layer_idx
            blk.self_attn.seg_bias = seg_bias_table[layer_idx].to(device=device)
        print(f"[r32] seg_bias stamped: {seg_bias_table.shape[0]} layers x "
              f"{n_heads} heads, "
              f"range [{seg_bias_table.min():+.3f}, {seg_bias_table.max():+.3f}], "
              f"{int((seg_bias_table != 0).sum())}/{seg_bias_table.numel()} nonzero")

    def _clear_seg_bias_stamps(self):
        """Remove any R32 bias left on the modules."""
        for blk in self.generator.model.blocks:
            blk.self_attn.seg_bias = None

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        return kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        return crossattn_cache


    def _initialize_trg_fg_mask_cache(self, batch_size, device, blend_rho_init=None):
        '''
        ✨ initialize target mask as ones

        ✨ R26: `blend_rho_init` (a float) additionally allocates a CACHE-ALIGNED
        per-token release exponent buffer that is rolled by exactly the same eviction
        arithmetic as `trg_fg_mask` -- see `_update_trg_fg_mask_cache`. None => the
        buffer does not exist and nothing downstream reads it.
        '''
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
        trg_fg_mask_cache = {
            "trg_fg_mask": torch.ones([batch_size, kv_cache_size], dtype=torch.bool, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        }
        if blend_rho_init is not None:
            # Filled with tau_bg: slots past `local_end_index` are never read as
            # previous keys, but a background-valued default keeps any future misuse
            # from silently behaving like tau = 0 (source pinned forever).
            trg_fg_mask_cache["blend_rho_tok"] = torch.full(
                [batch_size, kv_cache_size], float(blend_rho_init),
                dtype=torch.float32, device=device
            )
        return trg_fg_mask_cache

    def _update_trg_fg_mask_cache(self, trg_fg_mask_cache, current_trg_fg_mask, kv_cache_trg,
                                  current_rho=None):
        '''
        ✨ update trg_fg_mask similar to kv cache update

        ✨ R26: `current_rho` ([B, num_new_tokens] float32) is rolled and written with
        the SAME indices computed for the mask, in this one place, so the exponent field
        cannot drift out of alignment with the mask under KV-cache eviction.
        '''
        current_end = kv_cache_trg[0]["global_end_index"].item()
        sink_tokens = kv_cache_trg[0]["sink_tokens"]
        kv_cache_size = trg_fg_mask_cache["trg_fg_mask"].shape[1]
        num_new_tokens = current_trg_fg_mask.shape[1]
        assert num_new_tokens == kv_cache_trg[0]["num_new_tokens"], '%d != %d' % (num_new_tokens, kv_cache_trg[0]["num_new_tokens"])
        if self.local_attn_size != -1 and (current_end > trg_fg_mask_cache["global_end_index"].item()) and (
                num_new_tokens + trg_fg_mask_cache["local_end_index"].item() > kv_cache_size):
            num_evicted_tokens = num_new_tokens + trg_fg_mask_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = trg_fg_mask_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            trg_fg_mask_cache["trg_fg_mask"][:, sink_tokens: sink_tokens + num_rolled_tokens] = \
                trg_fg_mask_cache["trg_fg_mask"][:, sink_tokens + num_evicted_tokens: sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            if current_rho is not None:
                trg_fg_mask_cache["blend_rho_tok"][:, sink_tokens: sink_tokens + num_rolled_tokens] = \
                    trg_fg_mask_cache["blend_rho_tok"][:, sink_tokens + num_evicted_tokens: sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            # Insert the new keys/values at the end
            local_end_index = trg_fg_mask_cache["local_end_index"].item() + current_end - \
                trg_fg_mask_cache["global_end_index"].item() - num_evicted_tokens
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = trg_fg_mask_cache["local_end_index"].item() + current_end - trg_fg_mask_cache["global_end_index"].item()
        local_start_index = local_end_index - num_new_tokens
        trg_fg_mask_cache["trg_fg_mask"][:, local_start_index:local_end_index] = current_trg_fg_mask
        if current_rho is not None:
            trg_fg_mask_cache["blend_rho_tok"][:, local_start_index:local_end_index] = current_rho
        trg_fg_mask_cache["global_end_index"].fill_(current_end)
        trg_fg_mask_cache["local_end_index"].fill_(local_end_index)


    def _concat_kv_cache(self, kvc_1, kvc_2, index_select=-1, shared_dict=None):
        '''
        ✨ concat source and target kv cache at batch dim for dual branch sampling
        '''
        kv_cache1 = []
        if index_select == -1:
            index_kvc = kvc_2
        else:
            index_kvc = kvc_1
        for b_idx in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.cat((kvc_1[b_idx]["k"], kvc_2[b_idx]["k"]), dim=0).clone(),
                "v": torch.cat((kvc_1[b_idx]["v"], kvc_2[b_idx]["v"]), dim=0).clone(),
                "global_end_index": index_kvc[b_idx]["global_end_index"].clone(),
                "local_end_index": index_kvc[b_idx]["local_end_index"].clone(),
                "shared_dict": shared_dict,
            })
        return kv_cache1

    def _append_clean_src_kv_cache(self, kvc_dual, kvc_src):
        '''
        ✨ add clean src kv cache to dual cache dict
        '''
        for b_idx in range(self.num_transformer_blocks):
            kvc_dual[b_idx].update({
                'k_src_clean': kvc_src[b_idx]['k'],
                'v_src_clean': kvc_src[b_idx]['v'],
            })

    def _inject_masks_to_kv_cache(
        self, kv_cache, 
        trg_fg_mask_cache=None, current_src_fg_mask=None,
    ):
        '''
        ✨
        trg_fg_mask: [B, kv_cache_size], previous chunks' foreground mask.
        current_src_fg_mask: [B, lq], current chunk's foreground mask.
        '''
        for b_idx in range(self.num_transformer_blocks):
            kv_cache[b_idx].update({
                "trg_fg_mask": trg_fg_mask_cache['trg_fg_mask'],
                "current_src_fg_mask": current_src_fg_mask,
            })

    def _inject_blend_rates_to_kv_cache(self, kv_cache, rate_prev, comp_prev, rate_cur, comp_cur):
        '''
        ✨ R26: publish this denoising step's spatial blender rate to every block.

        Split the same way the masks are: `blend_rate_tok` is CACHE-ALIGNED
        ([B, kv_cache_size]) and read through the block's own `attn_seq_slice`, while
        `current_blend_rate` is [B, Lq] for the chunk being denoised -- whose cache slots
        `_update_trg_fg_mask_cache` has not written yet at this point in the block.
        Complements travel alongside so `causal_model.py` never has to recover `1 - rate`
        from a rounded float32 (see `blender_rate_from_rho`).
        '''
        for b_idx in range(self.num_transformer_blocks):
            kv_cache[b_idx].update({
                "blend_rate_tok": rate_prev,
                "blend_comp_tok": comp_prev,
                "current_blend_rate": rate_cur,
                "current_blend_comp": comp_cur,
            })

    def _save_union_mask(self, path, union):
        '''
        ✨ R26 pass 1: write M_f as a packed bool array, one row per latent frame.
        `union` is [B, F, frame_seq_length]; only batch element 0 is stored (the driver
        renders one clip per call).
        '''
        M = union[0].to(torch.bool).cpu().numpy()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            M=np.packbits(M, axis=-1),
            shape=np.array(M.shape, dtype=np.int64),
            frame_seq_length=np.array(M.shape[1], dtype=np.int64),
        )

    def _load_union_mask(self, path, device):
        '''
        ✨ R26 pass 2: read back what `_save_union_mask` wrote -> [F, frame_seq_length] bool.
        '''
        with np.load(path) as z:
            shape = tuple(int(v) for v in z["shape"])
            M = np.unpackbits(z["M"], axis=-1, count=shape[1])
        if M.shape != shape:
            raise ValueError(f"R26: {path} unpacked to {M.shape}, expected {shape}")
        return torch.from_numpy(M.astype(bool)).to(device)

    def _save_split_mask(self, path, src, trg):
        '''
        ✨ R30 diagnostic: write M_src and M_trg SEPARATELY -- the two operands of the OR
        that produces M_f = M_src | M_trg (`_save_union_mask`) -- one row per latent
        frame, packed bool arrays. `src`/`trg` are [B, F, frame_seq_length]; only batch
        element 0 is stored (one clip per call). Same schema as `_save_union_mask`'s `M`,
        duplicated under `M_src`/`M_trg`, so a reader can unpack either with the same
        `shape`/`frame_seq_length` it already uses for `M`.
        '''
        M_src = src[0].to(torch.bool).cpu().numpy()
        M_trg = trg[0].to(torch.bool).cpu().numpy()
        if M_src.shape != M_trg.shape:
            raise ValueError(f"R30: src {M_src.shape} != trg {M_trg.shape}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            M_src=np.packbits(M_src, axis=-1),
            M_trg=np.packbits(M_trg, axis=-1),
            shape=np.array(M_src.shape, dtype=np.int64),
            frame_seq_length=np.array(M_src.shape[1], dtype=np.int64),
        )

    
    def _kv_cache_to(self, kvc, device, low_memory):
        if not low_memory:
            return
        for itm in kvc:
            for k, v in itm.items():
                if isinstance(v, torch.Tensor):
                    v.to(device)


    def _register_crossattn_enhancement(self, crossattn_cache, fg_indices, fg_boost_factor=1.0, layers=range(30), current_src_fg_mask=None):
        '''
        ✨ default [src, trg] for multiple batches
        '''
        if layers is None:
            # all layers
            layers = range(self.num_transformer_blocks)
        for l_idx in layers:
            crossattn_cache[l_idx]["fg_indices"] = fg_indices
            crossattn_cache[l_idx]["fg_boost_factor"] = fg_boost_factor
            crossattn_cache[l_idx]["current_src_fg_mask"] = current_src_fg_mask
            crossattn_cache[l_idx]["apply_enhance"] = True
        
    def _register_crossattn_mask_gatherer(self, crossattn_cache, fg_indices, fg_scale=1.0, layers=range(20)):
        '''
        ✨ fg_indices will be poped in blocks to avoid repeating
        '''
        if layers is None:
            # all layers
            layers = range(self.num_transformer_blocks)
        for l_idx in layers:
            crossattn_cache[l_idx]["fg_indices"] = fg_indices
            crossattn_cache[l_idx]["fg_scale"] = fg_scale
            crossattn_cache[l_idx]["obtain_mask"] = True

    def _aggregate_crossattn_mask(self, crossattn_cache, size=None, patch=(1, 2, 2), scale_factor=1):
        '''
        ✨
        size: (Ttok, Htok, Wtok), for visualization only. \\
        patch: patchify kernel size. \\
        
        crossattn_cache[l_idx]["fg_mask_soft"]: [B, Lq, 1, 1] \\
        return:
            mask_soft, mask_bin: [B, Lq]
            mask_soft_vis, mask_bin_vis: [B, Ttok, Htok, Wtok]
        '''
        total_mask = 0
        account = 0
        for l_idx in range(self.num_transformer_blocks):
            if "fg_mask_soft" in crossattn_cache[l_idx]:
                total_mask += crossattn_cache[l_idx]["fg_mask_soft"].squeeze(-1).squeeze(-1)
                account += 1
        mask_soft = total_mask / account
        mask_bin = mask_soft > 0
        if size is None:
            mask_soft_vis = None
            mask_bin_vis = None
        else:
            view_size = (total_mask.size(0), *map(lambda s, p: s // p, size, patch))
            mask_soft_vis = mask_soft.view(view_size)
            mask_bin_vis = mask_bin.view(view_size)
            if scale_factor != 1:
                mask_soft_vis = F.interpolate(mask_soft_vis, scale_factor=scale_factor).to(mask_soft_vis)
                mask_bin_vis = F.interpolate(mask_bin_vis.float(), scale_factor=scale_factor) > 0.5
        return mask_soft, mask_bin, mask_soft_vis, mask_bin_vis

    def _mask_reshape(self, mask_seq, size, patch=(1, 2, 2), scale_factor=2):
        '''
        ✨
        mask_seq: [B, Lq]
        mask_map: [B, Ttok, Htok, Wtok]
        '''
        view_size = (mask_seq.size(0), *map(lambda s, p: s // p, size, patch))
        mask_map = mask_seq.view(view_size)
        if scale_factor != 1:
            mask_map = F.interpolate(mask_map.float(), scale_factor=scale_factor) > 0.5
        return mask_map


    def _initialize_noise_statistics(
        self, reuse_noise_temporal_mean=False
    ):
        if reuse_noise_temporal_mean:
            self.noise_temporal_mean = dict()
            self.noise_temporal_mean_fg = dict()
            self.noise_temporal_mean_bg = dict()
        else:
            self.noise_temporal_mean = None
            self.noise_temporal_mean_fg = None
            self.noise_temporal_mean_bg = None

    def _reuse_noise_statistics(
        self, noise: torch.Tensor, step_idx: int, 
        ema_factor: float = 0.5, fg_mask=None,
        alpha_prog=2, alpha_mixed=1,
    ):
        if self.noise_temporal_mean is not None:
            if step_idx not in self.noise_temporal_mean.keys():
                self.noise_temporal_mean[step_idx] = noise
            else:
                noise = self.noise_temporal_mean[step_idx].flip(1) * alpha_prog / (1 + alpha_prog ** 2) ** 0.5 + \
                    noise * 1 / (1 + alpha_prog ** 2) ** 0.5
                self.noise_temporal_mean[step_idx] = noise
        
        return noise