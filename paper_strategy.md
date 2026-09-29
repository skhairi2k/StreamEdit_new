# Paper strategy — pushing an image edit through a video, per token

**Date:** 2026-09-29 · **Target:** CVPR 2027 (deadline ≈ mid-Nov 2026 — verify) · **Status:** pre-gate; nothing below is a result unless marked *evidence*.

---

## 1. Story

**Premise.** Image editors (Qwen-Image-Edit, FLUX Kontext) are ahead of video editors. Editing frame 0 is largely solved. The open problem is **how hard to push that edit through the rest of the video**. That is an edit-strength problem.
*Needs one motivating measurement (T9): frame-0 quality of text-driven video editors vs the image editor's anchor.*

**Why the current answer fails.** Training-free propagators control edit strength with **one global knob**: StreamEdit's ρ (Q/K blend, source weight $t^{\rho}$), FlowEdit/FlowDirector's $n_{\max}$, AnyV2V's injection thresholds $\tau$. One strength cannot serve a **shape-changing edit**:
- tokens that must take a new outline need early release, at high noise where structure forms;
- the rest must stay anchored.

A strong global anchor blocks the new shape; a weak one damages the background.

**Contribution — "when, not where".** Masks decide *where* a token may change. We set *how hard / when* per token: a **per-token release time**, set from where a cheap draft actually diverges from the source.

| | Content |
|---|---|
| C1 · Method | Per-token release: $t^{\rho_j}$ in StreamEdit (Self-Forcing, LongLive). Optional: $\mathbb{1}[t\le\tau_j]\,V^{\Delta}$ in FlowDirector (= per-token $n_{\max}$) |
| C2 · Evaluation | FiVE's bg metrics penalise valid deformation → distance-weighted PSNR/SSIM/LPIPS (R37) + FiVE-Shape subset (anchor spill) |
| C3 · Evidence | Beats the *tuned* global-strength curve at equal NFE and beats where-only masks; gain concentrated on shape-changing edits; holds across two image editors |

**Evidence so far** (*evidence*, R35 vs R36, full FiVE, gain at matched bg LPIPS):
- `lpips_first2` beats the ρ curve on full FiVE-Acc in all 4 main edit types: +0.019 / +0.028 / +0.008 / +0.023 (types 1–4). CLIP-D is also positive in all four.
- The gain is largest on type 2 (non-rigid) and smallest on type 3 (colour).
- Small effect; 2× NFE; no comparison with R26 on the full bench yet.
- **Committed arm: `lpips_first2`.** It routes on LPIPS and is scored on LPIPS, so the claim must also hold on weighted SSIM/PSNR.

---

## 2. Tables

**Table 1 — matched first frames (main).** Every method gets the same fixed Qwen-Image-Edit-2511 frame 0.

| Group | Methods | Note |
|---|---|---|
| Training-free | **Ours** (SF, LL) · StreamEdit§ (SF, LL) at ρ=2 *and* best global ρ · AnyV2V (TMLR 2024) | StreamEdit§ rows rerun with *our* anchors (the paper says only "Qwen-Image-Edit") |
| Per-video optimisation, marked | LoRA-Edit (ICLR 2026) | FiVE-Shape / subset only (tuning cost per video) |
| Trained, shaded reference | FreeProp / FFP-300K (CVPR 2026) · VACE (ICCV 2025, Wan2.1-VACE-1.3B) | "Close at a fraction of the cost" is a result; no need to beat them |

Wording: *"to our knowledge, StreamEdit§ is the only training-free first-frame propagation method on a modern video DiT"* — only after T0; AnyV2V and Videoshop (ECCV 2024) are training-free but UNet-era. Frame Guidance (ICLR 2026) steers generation, not editing, so it is not a competitor (§6).

**Table 2 — text to video, end to end.** Rows that use an image editor are marked ✎.

| Rows | Source of numbers |
|---|---|
| **Ours ✎** · StreamEdit§ ✎ at ρ=2 (image editor + naive propagation) · StreamEdit§ ✎ at best global ρ | own runs |
| StreamEdit (text) · FlowDirector (CVPR 2026) · DynaEdit (arXiv 2603.17989) | own runs |
| TokenFlow · DMT · VidToMe · VideoGrain · Pyramid-Edit · Wan-Edit · UniEdit-Flow (ICLR 2026) | reuse StreamEdit's FiVE table if the metric code matches; check 420 vs our 419 pairs |
| *(optional)* FlowDirector + per-token $n_{\max}$ | own runs — the same rule on a text-only method |

Whatever we gain over "StreamEdit§ ✎ at best global ρ" is propagation, not better editing.

**Table 3 — equal NFE (the core evidence; FiVE-Shape / Control / all).** Budget $B = k + n$ (draft + final), counting both StreamEdit branches; wall-clock also reported.

| Arm | Spends $B$ as |
|---|---|
| Global ρ, more steps | one pass, $B$ steps |
| Global ρ chosen per clip | draft $k$ → one ρ per clip → final $n$ |
| Where-only mask (R26 fg/bg gate) | one pass, $B$ steps |
| **Ours** | draft $k$ → per-token ρ field → final $n$ |

**Ablations:**
- draft length $k \in \{4, 7, 15\}$;
- source of the field (draft / anchor-only / attention);
- divergence network (LPIPS / DINO) and stage-1 regime (unblended / first2);
- hard vs soft gate (FlowDirector, optional).

**Fairness controls:**
1. Release the fixed first frames (and the converted prompts for EditVerseBench).
2. Editor swap: Qwen-Image-Edit-2511 vs FLUX.1 Kontext. The gain over StreamEdit§ at best global ρ must hold with both.

---

## 3. Benchmarks and metrics

| Benchmark | Role | Caveats |
|---|---|---|
| **FiVE-Bench** (419 pairs) + fixed Qwen anchors | Primary; carries the claim. Subsets: **FiVE-Shape** (anchor spill above a threshold fixed in advance), **FiVE-Control** (types 3–4) | Types 5 (9 clips) / 6 (10 clips) too small alone; the StreamEdit family fails removal (FiVE-Acc ≈ 0) |
| **EditVerseBench-FFP** (125 videos, filtered by FFP-300K from EditVerse, ICLR 2026) | Secondary. Most recent top-venue first-frame propagation protocol; non-square videos, 20 edit types; comparable to FreeProp's published numbers | Instructions → source/target prompts must be converted (VLM) and released · Self-Forcing/LongLive trained at 832×480: smoke-test portrait/square · no source masks (no bg metrics / anchor spill without SAM2) · use their metrics unchanged, and only if their eval code/VLM prompt is released |

**Anchor spill** (input-only): fraction of pixels outside the slightly dilated source mask that differ between source frame 0 and the anchor.

**Metrics — primary per axis declared in advance; everything else secondary:**

| Axis | Primary | Secondary |
|---|---|---|
| Edit alignment | FiVE-Acc (full) on FiVE · their VLM score on EditVerseBench | CLIP-T, CLIP-D · EditReward (ICLR 2026; image reward model → per-frame mean, blind to time) · **PSC-Δ**: VideoCLIP-XL2 similarity to target phrase minus source phrase (*our variant* of IVEBench's PSC, which is target-only, ICLR 2026) |
| Preservation | R37 α-averaged LPIPS (α ∈ [0, 0.5]) **and** FiVE-official LPIPS | weighted/official PSNR, SSIM |
| Temporal | warp error | CLIP-F, DINO temporal consistency |

Aggregation: plain per-clip mean (never `_avg.csv`); gain sign per edit type; paired bootstrap resampled over source videos if the gain is borderline.

---

## 4. Tasks, ranked by priority, then effort

Effort: S ≤ 1 day · M 2–4 days · L ≥ 1 week. Unless stated, runs are on FiVE-Shape + Control first, full bench for final numbers.

### P0 — gates (weeks 1–2, Self-Forcing VP only)

| ID | Task | Effort | Output |
|---|---|---|---|
| T0 | `/litsearch`: training-free first-frame propagation on DiT video models (backs the "only method" sentence); per-token/region-wise $n_{\max}$ or edit strength; shape-changing video editing; boundary-tolerant bg metrics | S | novelty, wording |
| T1 | Anchor spill for 419 clips → fix threshold → FiVE-Shape / FiVE-Control manifests | S (CPU) | subsets |
| T2 | NFE accounting (both branches) + wall-clock of draft / divergence / final | S | cost table |
| T3 | Baseline step scaling: global ρ at 15 vs 19 / 22 / 30 steps | S–M | equal-NFE reference |
| T4 | Ablation: draft length $k \in \{4, 7, 15\}$ (field agreement + render) | M | choice of $k$ |
| T5 | Where-only: R26 gate at equal NFE (3–4 values of b) | M | "when vs where" |
| T6 | Per-clip ρ choice from the draft at equal NFE | M | is "per-token" needed? |
| T7 | R37 metric (planned) + ρ × edit-type gap + hard-dilation ablation + check of type-5 masks | M | C2 |

**Gate A** (end of week 2), on FiVE-Shape at equal NFE:
1. Ours > best global ρ.
2. Ours > R26.
3. Gain on Shape clearly > gain on Control.

If (1) or (2) fails, the paper reduces to the evaluation contribution (C2) plus tuning: weak.

### P1 — after Gate A (weeks 3–5)

| ID | Task | Effort |
|---|---|---|
| T8 | FLUX Kontext anchors for 419 clips → rerun Ours + StreamEdit§ at best global ρ (editor swap) · release both anchor sets | S–M |
| T9 | Motivating measurement: frame-0 alignment/preservation of text-driven video editors vs the image editor's anchor | S |
| T10 | Self-Forcing text mode: ρ curve + Ours + equal-NFE arms (Table 2 rows) | S–M |
| T11 | LongLive port (`rho_frames`, vectorised blend, ρ for sink tokens) → bit-parity gate → VP + text runs | M–L |
| T12 | Table 1 externals with fixed anchors: VACE-1.3B, AnyV2V (smoke 1 clip first: 16-frame native), FreeProp (if weights are public) | M |
| T13 | Table 2 externals: FlowDirector, DynaEdit (if code is public); reuse StreamEdit's reported baselines after checking the metric code and the 420/419 pair difference | M |
| T14 | Metrics: EditReward (per frame), PSC-Δ (VideoCLIP-XL2), warp error on all arms | M |
| T15 | EditVerseBench-FFP: get the 125-video list + protocol, convert prompts, aspect-ratio smoke test, run Table 1 rows | M |

### P2 — if time allows (week 6+)

| ID | Task | Effort |
|---|---|---|
| T16 | LoRA-Edit on FiVE-Shape (per-video tuning; FiVE masks as input) | L |
| T17 | FlowDirector + per-token $n_{\max}$ (optional row; upgrades the claim to "across mechanisms") | M |
| T18 | 2AFC study: ~100 pairs where FiVE-official and R37 disagree + method preference | M |
| T19 | Figures: FiVE penalising a correct outline; ρ fields; per-type trade-off curves; editor-swap examples | S |

**Suggested order:** T0 ∥ T1 → T2 → T3 ∥ T4 → T5 ∥ T6 → T7 → **Gate A** → T8 ∥ T9 ∥ T10 → T11 → T12 ∥ T13 → T14 → T15 → P2.

---

## 5. Risks

- **Small effect** (+0.02 FiVE-Acc on full FiVE): needs a clearly larger gain on FiVE-Shape.
- **Equal NFE may erase the gain.** T3/T4 decide this early.
- **Tuned vs default baseline:** claims must be against the best global ρ, not ρ=2 (ρ=2 → ρ=50 alone moves full FiVE-Acc 0.697 → 0.811).
- **Metric circularity** (LPIPS-routed, LPIPS-scored) and **leakage forgiven by distance weighting**: fix α in advance, always report FiVE-official, add the 2AFC study.
- **Too many metrics:** primaries declared in §3 before the final runs.
- **EditVerseBench portability** (prompts, aspect ratios, protocol availability) could cost more than expected: it stays secondary.
- **Compute:** a full-bench curve is ~85 L40S GPU-h per host (8 points). Use 5 points on Shape + Control, full bench for final tables only.

## 6. References

arXiv IDs and venues checked 2026-09-29. A venue is given only where it was checked; otherwise only the arXiv year.

**Base method and backbones**
- **StreamEdit** (arXiv 2026) — [arXiv:2605.21466](https://arxiv.org/abs/2605.21466). Training-free editing as source-conditioned generation on streaming few-step video models, with Q/K blending controlled by one global ρ and optional first-frame visual prompting (§). Our base and main baseline: we make its global ρ per-token.
- **Self Forcing** (arXiv 2025) — [arXiv:2506.08009](https://arxiv.org/abs/2506.08009). Autoregressive few-step video diffusion on Wan2.1-1.3B, trained on its own rollouts with a KV cache. Backbone of StreamEdit (SF) and of our main host.
- **LongLive** (ICLR 2026) — [arXiv:2509.22622](https://arxiv.org/abs/2509.22622). Real-time long video generation: a Wan-1.3B model fine-tuned with a frame-level attention sink and KV re-cache. Our second backbone, StreamEdit (LL).

**Image editors** (produce the fixed first frames)
- **Qwen-Image** (arXiv 2025) — [arXiv:2508.02324](https://arxiv.org/abs/2508.02324). Technical report of the Qwen-Image family, which includes Qwen-Image-Edit. Our main anchors come from the Qwen-Image-Edit-2511 checkpoint.
- **FLUX.1 Kontext** (arXiv 2025) — [arXiv:2506.15742](https://arxiv.org/abs/2506.15742). Flow-matching model for in-context image generation and editing. The second editor for the editor-swap control.

**First-frame propagation — Table 1**
- **AnyV2V** (TMLR 2024) — [arXiv:2403.14468](https://arxiv.org/abs/2403.14468). Training-free: edit frame 0 with any image editor, then regenerate with an image-to-video model plus source feature injection gated by global thresholds τ. Training-free competitor on an older UNet backbone (I2VGen-XL, 16 frames).
- **Videoshop** (ECCV 2024) — [arXiv:2403.14617](https://arxiv.org/abs/2403.14617). Training-free propagation of a first-frame edit through noise-extrapolated inversion of an image-to-video model. The other UNet-era training-free propagator; cited, not run.
- **LoRA-Edit** (ICLR 2026) — [arXiv:2506.10082](https://arxiv.org/abs/2506.10082). First-frame-guided editing by per-video, mask-aware LoRA fine-tuning of an image-to-video model. Per-video-optimisation reference, run on a subset only.
- **FFP-300K / FreeProp** (CVPR 2026) — [arXiv:2601.01720](https://arxiv.org/abs/2601.01720). A 300K-pair first-frame propagation dataset, plus FreeProp, a LoRA fine-tune of Fun-Control (Wan 2.1), evaluated on a 125-video filter of EditVerseBench with Qwen-Edit first frames. Trained reference (shaded) and source of our second benchmark.
- **VACE** (ICCV 2025) — [arXiv:2503.07598](https://arxiv.org/abs/2503.07598). All-in-one trained video creation and editing model on Wan2.1, including reference- and mask-conditioned editing (1.3B checkpoint public). Trained reference at our backbone size.

**Text-driven editing — Table 2**
- **FlowEdit** (arXiv 2024) — [arXiv:2412.08629](https://arxiv.org/abs/2412.08629). Inversion-free editing: an ODE driven by the target–source velocity difference, with edit strength set by the start step $n_{\max}$. Defines the $n_{\max}$ knob we make per-token (optional host via FlowDirector).
- **FlowDirector** (CVPR 2026) — [arXiv:2506.05046](https://arxiv.org/abs/2506.05046). FlowEdit-style video editing on Wan2.1-1.3B, with an attention-derived spatial mask on the edit velocity and averaged-guidance steering. Text-driven competitor. Its mask is a "where"-only baseline, and per-token $n_{\max}$ can be added with one gated line.
- **UniEdit-Flow** (ICLR 2026) — [arXiv:2504.13109](https://arxiv.org/abs/2504.13109). Predictor-corrector inversion and region-aware "delayed injection" editing for flow models. Text-driven baseline, reported on FiVE by StreamEdit.
- **DynaEdit** (arXiv 2026) — [arXiv:2603.17989](https://arxiv.org/abs/2603.17989). Training-free, inversion-free editing with text-to-video flow models, aimed at action, interaction and dynamics edits. Recent training-free text-driven competitor (run only if code is public).
- **TokenFlow** (arXiv 2023) — [arXiv:2307.10373](https://arxiv.org/abs/2307.10373). Training-free consistent video editing by propagating diffusion features along inter-frame correspondences. Diffusion-era baseline; numbers reused from StreamEdit's FiVE table.
- **DMT — Space-Time Diffusion Features** (arXiv 2023) — [arXiv:2311.17009](https://arxiv.org/abs/2311.17009). Zero-shot motion transfer guided by a space-time feature loss from a text-to-video model, including large shape changes. Diffusion-era baseline, reused numbers; relevant to shape change.
- **VidToMe** (arXiv 2023) — [arXiv:2312.10656](https://arxiv.org/abs/2312.10656). Zero-shot video editing that merges self-attention tokens across frames for temporal consistency. Diffusion-era baseline, reused numbers.
- **VideoGrain** (ICLR 2025) — [arXiv:2502.17258](https://arxiv.org/abs/2502.17258). Zero-shot multi-grained (class, instance, part) editing by modulating space-time cross- and self-attention. Diffusion-era baseline, reused numbers.

**Related, not a competitor**
- **Frame Guidance** (ICLR 2026) — [arXiv:2506.07177](https://arxiv.org/abs/2506.07177). Training-free steering of video *generation* from a few frame-level signals (keyframes, style image, sketch, depth, colour blocks, looping): at each step it decodes a predicted clean frame, computes a loss against the target, and updates the latent by gradient. It generates rather than edits a source video, so it is not a competitor; cite it as related training-free frame-level control.

**Benchmarks and metrics**
- **FiVE-Bench** (ICCV 2025) — [arXiv:2503.13684](https://arxiv.org/abs/2503.13684). 100 videos, 420 object-level prompt pairs in 6 edit types, with masks, the FiVE-Acc VLM metric, and the FlowEdit adaptations Pyramid-Edit and Wan-Edit. Primary benchmark; its mask-based bg metrics are what our distance-weighted metric corrects.
- **EditVerse / EditVerseBench** (ICLR 2026) — [arXiv:2509.20360](https://arxiv.org/abs/2509.20360). Unified in-context image/video editing model plus EditVerseBench, an instruction-based video editing benchmark. Source of the 125-video FFP subset, our secondary benchmark.
- **IVEBench** (ICLR 2026) — [arXiv:2510.11647](https://arxiv.org/abs/2510.11647). Instruction-guided video editing benchmark (600 videos, 8 task categories) with quality, instruction-compliance and fidelity metrics, including PSC (VideoCLIP-XL2 video–target-phrase similarity). Our PSC-Δ is a source-relative variant of its PSC.
- **VideoCLIP-XL** (EMNLP 2024) — [arXiv:2410.00741](https://arxiv.org/abs/2410.00741). Video CLIP trained for long-description understanding; IVEBench uses its v2 checkpoint (VideoCLIP-XL2). The embedding model behind PSC-Δ.
- **EditReward** (ICLR 2026) — [arXiv:2509.26346](https://arxiv.org/abs/2509.26346). Human-aligned reward model for instruction-guided *image* editing, trained on 200K+ expert preference pairs. Secondary alignment metric, applied per frame (blind to time).
