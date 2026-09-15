# StreamEdit / StreamGVE — paper summary & code map

> **Naming.** The PDF (`StreamGVE_ Training-Free Video Editing via-1-15.pdf`, arXiv:2605.21466) titles the method
> **StreamGVE** (*Streaming-Generation-based Video Editing*). The public repo and [README](../README.md) rebrand it
> **StreamEdit**. Same authors, same method — treat the two names as synonyms below. ECCV 2026.

---

## 1. Problem & core idea

Training-free video editing has mostly followed a **data → data** paradigm:

- **Inversion-based** methods do `source → noise → target`: invert the source to reconstructive noise, then denoise
  under the target prompt. They pay for extra iterations and accumulate inversion error, distorting editing-irrelevant
  regions (the classic **inversion drift** failure).
- **Inversion-free** methods do direct `source → target` transfer with noisy auxiliaries. Because early samples stay
  close to the source, visible edits only appear after many iterations, and their irregular trajectories force small,
  careful step sizes → slow and unstable.

Neither exploits the **few-step, noise → data** generators (consistency models, rectified flow, DMD-distilled models)
that now dominate fast generation.

**Key reformulation.** StreamEdit recasts editing as **source-conditioned `noise → target` generation**. It keeps the
few-step sampler of a *streaming* (autoregressive, KV-cached) video generator and injects the source video as a
condition through training-free attention manipulation. Built on two streaming backbones — **Self Forcing** and
**LongLive** (both distilled from Wan2.1-T2V-1.3B) — it needs no fine-tuning.

Reported: ~0.6 s/frame at 15 steps, **< 0.32 s/frame at 5 steps** on a single A100; length-unrestricted via
autoregressive rollout; first on all axes of the FiVE-Bench comparison and the long-video user study.

## 2. Method → code map

The whole method lives in **two files per backbone**: the *pipeline* (sampling, guidance, mask/KV bookkeeping) and the
*attention modules* (the actual blending/boosting inside the DiT). References below are to the Self-Forcing build; the
LongLive build (`LongLive_StreamEdit/`) mirrors it.

| # | Paper component (Sec.) | What it does | Code |
|---|---|---|---|
| 4.1 | **Dual-branch fast sampling** (Eq. 3) | Denoise a *source* and a *target* branch in parallel under **shared noise**; source is fixed-endpoint (constant clean `x₀ˢʳᶜ`), target is the usual few-step CM update | [`EditCausalInferencePipeline.inference`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L203) |
| 4.2 | **Self-attention bridge** (Eq. 4–6) | Query blending + key blending + masked previous-key blending + delayed source-KV injection between branches | [`CausalWanSelfAttention.forward`](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L315-L375) |
| 4.3 | **Cross-attn grounding** (Eq. 7) | Adaptive editing-region mask from fg−bg attention difference (no extra model) | [`WanT2VCrossAttention.forward`](../Self-Forcing_StreamEdit/wan/modules/model.py#L263-L289) |
| 4.3 | **Cross-attn boosting** (Eq. 8) | Additive, mask-gated enhancement of trigger-word attention (edit-strength valve `ω`) | [`WanT2VCrossAttention.forward`](../Self-Forcing_StreamEdit/wan/modules/model.py#L189-L251) |
| 4.4 | **Source-oriented guidance** (Eq. 9–10) | CFG-like correction of editing-*irrelevant* regions using the source-branch velocity error | [`inference` loop](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L484-L493) |
| 4.5 | **Visual prompting** | Treat a user/model-edited first frame as a "previous chunk" for fine-grained control | [`inference` context caching](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L322-L376) + [driver](../Self-Forcing_StreamEdit/inference_edit_streamedit.py#L161-L177) |
| 5.5 | **Autoregressive long-video rollout** | Chunk-wise sliding editing with overlap, feeding prev latents as condition | [`rollout_inference`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L62-L201) |

### 4.1 Dual-branch fast sampling
Both branches share the same forward noise `fwd_noise` and are stacked along the batch dim, so a **single** model
forward produces `v_src, v_trg`:
[edit_causal_inference.py:444–483](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L444-L483).
The source/target KV caches are concatenated into a "dual" cache by
[`_concat_kv_cache`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L659). This realizes Eq. 3:
`xᵗ = (1−t)·z + t·ε` for the target, with the source pinned to its clean latent.

### 4.2 Self-attention bridge
Implemented entirely in [causal_model.py:315–375](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L315-L375).
The time-dependent blend ratio `rₜ = 1 − t_{i−1}^ρ` (Eq. 4) is `blender_rate` at
[line 320](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L320); `ρ` is the CLI `--blend_power`.

- **Query blending** (Eq. 4) — structure/motion preservation: [line 371](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L371).
- **Key blending** of current keys (Eq. 5): [line 362](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L362).
- **Masked-blended previous keys** `M_prev ⊙ (…)` (Eq. 5): [lines 342–344](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L342-L344) (only foreground positions from `trg_fg_mask` are blended).
- **Delayed source-KV injection** `[t < t_inj]·(M_curr ⊙ Kˢʳᶜ)` (Eq. 6), with **t_inj = 0.5**: gated by
  `current_timestep_index > total_timestep // 2` at [lines 353–357](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L353-L357) — only background source KV is appended in the low-noise second half.

### 4.3 Cross-attention grounding & boosting
- **Grounding** (Eq. 7): the fg−bg attention-difference mask (Heaviside) is computed at
  [model.py:263–289](../Self-Forcing_StreamEdit/wan/modules/model.py#L263-L289) via a signed `mask_value`
  (`+fg_scale/|T|` on trigger tokens, `−1/(L−|T|)` elsewhere) pushed through the attention and thresholded at 0. Masks
  are gathered/aggregated in the pipeline by
  [`_register_crossattn_mask_gatherer`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L725) /
  [`_aggregate_crossattn_mask`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L737); grounding runs on the
  first 20 of 30 cross-attn layers (`mask_layers=range(20)`).
- **Boosting** (Eq. 8): **additive** log-domain enhancement — `w_{p,q}=ω` for trigger tokens inside the mask, else 1 —
  implemented as the numerator/denominator flash-attention trick at
  [model.py:189–251](../Self-Forcing_StreamEdit/wan/modules/model.py#L189-L251). `ω` is the CLI `--fg_boost_factor`
  (paper: 4 for Self-Forcing, 2 otherwise). Trigger-word token indices come from
  [`find_phrase_token_indices`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L316).

### 4.4 Source-oriented guidance
CFG-analogue that only corrects **editing-irrelevant** regions
([edit_causal_inference.py:484–493](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L484-L493)):
- `v_gt = ε − x₀ˢʳᶜ`, observable error `g = v_gt − v_src` (Eq. 9).
- A soft **Abs-Mean-Norm** mask from `|v_trg − v_src|` selects background: `bg_mask`; corrected velocity
  `ṽ_trg = v_trg + bg_mask ⊙ g` (Eq. 10). The inter-branch velocity difference doubles as a latent-level cue of the
  edit region.

### 4.5 Visual prompting
When `--first_frame_edit` is given, the driver VAE-encodes both the source first frame and a (Qwen-Image-Edit) edited
first frame ([inference_edit_streamedit.py:161–177](../Self-Forcing_StreamEdit/inference_edit_streamedit.py#L161-L177))
and passes them as `src_initial_latent` / `trg_initial_latent`. The pipeline caches them as a "previous chunk" (KV +
masks) before the denoising loop
([edit_causal_inference.py:322–376](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L322-L376)), costing
one extra forward. `--triple_first_frame` repeats the frame ×3 (LongLive convention).

### Mask lifecycle (threefold update, Sec. 5.4)
Worth calling out because it's spread across the loop:
1. **Before denoising** — forward *clean source* to get `M_src_curr`, cache source KV
   ([lines 407–423](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L407-L423)).
2. **At t = t_inj** (`index == len//2`) — extract target mask, form union `M_curr = M_trg ∪ M_src`, inject to KV cache
   ([lines 496–505](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L496-L505)).
3. **After denoising** — forward *clean target* to refresh KV and produce `M_prev` for subsequent chunks
   ([lines 512–526](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L512-L526)).
Masks therefore add **no extra NFEs** beyond the two clean-context forwards the streaming backbone already does.

## 3. Sampling budget / reproducibility notes

- CFG is **disabled**; the dual branch means **NFE = 2 × (steps + 1)** — flagged in the paper as the fair-comparison
  budget. Standard runs use **15 steps**, long-video **5 steps**, uniform timestep scheduler.
- Denoising steps are set as `np.arange(1000, 0, -1000/step)` in
  [inference_edit_streamedit.py:68](../Self-Forcing_StreamEdit/inference_edit_streamedit.py#L68); `guidance_scale`
  forced to 1.0 ([line 64](../Self-Forcing_StreamEdit/inference_edit_streamedit.py#L64)).
- Seeds set via [`set_seed`](../Self-Forcing_StreamEdit/inference_edit_streamedit.py#L47) (`--seed`, default 0).
- Key hyper-parameters and their code names:

  | Paper | Meaning | CLI flag | Default |
  |---|---|---|---|
  | `ρ` | self-attn blend strength (Eq. 4) | `--blend_power` | 2 |
  | `ω` | cross-attn boost / edit strength (Eq. 8) | `--fg_boost_factor` | 4 (SF), 2 (LL/§) |
  | `t_inj` | source-KV injection start | fixed | 0.5 (`index == len//2`) |
  | steps | sampler steps | `--step` | 15 (short) / 5 (long) |

- Runnable examples (color / remove / add / porcelain-with-visual-prompt / long tiger→elephant) in
  [inference_edit_streamedit.sh](../Self-Forcing_StreamEdit/inference_edit_streamedit.sh). For **removal**, set
  `src_word` = object, `trg_word` = background; for **addition**, set `src_word` = "" or the add location.

## 4. Evaluation (paper)

- **Benchmark:** FiVE-Bench — 100 videos, 420 prompt pairs, 6 edit types (color, material, object-sub ±non-rigid, add,
  remove).
- **Metrics:** structure/background preservation (structure-dist, PSNR, LPIPS, MSE, SSIM), text alignment (CLIP, edit-
  region CLIP), IQA (NIQE), temporal (motion fidelity), VLM-based success (FiVE-Acc).
- **Result:** best/2nd-best on nearly every column at the lowest time-per-frame (0.60 s SF, 0.69 s LL). Self-Forcing
  favors preservation; LongLive favors edit strength.
- **Ablations:** removing S.O.G. hurts background alignment; removing the S.A.B. lets generation drift to the target
  prompt (loses source). `ρ` trades preservation ↔ edit success; `ω` scales edit intensity.

### ⚠️ Reading-the-code flags (for our own use)
- **`--use_ema` defaults to `True`** ([line 119](../Self-Forcing_StreamEdit/inference_edit_streamedit.py#L119)) with
  `action="store_true"` — so it's *always* True and cannot be disabled from the CLI. If we ever want the non-EMA
  generator, this needs a `BooleanOptionalAction` or a `--no_ema` flag.
- **In-place KV blend**: [causal_model.py:343](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L343) mutates
  `b_trg_prev_fg_key` (a view into `trg_prev_key`) in place. Fine here since the dual cache is a fresh `.clone()` each
  block ([`_concat_kv_cache`](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L670-L671)), but worth
  remembering before any refactor that removes that clone.
- **`_reuse_noise_statistics`** ([lines 793–805](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L793-L805))
  correlates each step's noise with a *time-flipped* copy of the first-seen noise (`alpha_prog=2`). This is an
  undocumented temporal-coherence trick not described in the 15-page PDF — check the full paper/appendix before relying
  on it, and note it breaks strict per-step noise independence.
</content>
