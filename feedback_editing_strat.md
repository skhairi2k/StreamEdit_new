# Closed-Loop Streaming Editing via Realized-Decode Feedback — Research Brief

**Status:** idea development / positioning (pre-plan)
**Date:** 2026-08-12
**Setting:** training-free, source/target-prompt (NOT instruction) video editing on **streaming/causal** generators (StreamEdit / StreamGVE on Self-Forcing + LongLive, Wan2.1-1.3B distilled).
**Core question being explored:** can we exploit the causal generator's ability to produce the *realized state of decoded frames rapidly and cheaply* to improve editing **online**, and is it necessarily training-based? (Answer: yes it's exploitable; no, a training-free instantiation exists.)

---

## 0. One-paragraph thesis

Recast training-free streaming editing as a **closed-loop controller** in which the **realized decoded state** (the cheap, near-clean `x0` decode a few-step causal model already produces) is used as a **feedback signal at two timescales**: an **intra-block fast loop** that fixes a block before it is committed (and is the *only* lever available on block 0), and an **inter-block slow loop** that reads committed chunks to steer subsequent chunks against drift and preservation loss. The unifying principle is one signal (the realized decode / internal state), two timescales, steering an actuator inside the editing pipeline, under an explicit **edit-vs-preservation** objective, aiming to approach the offline per-video oracle (R23) *without* ground truth. It is publishable as a **new problem framing + non-obvious combination**, not a new mechanism — so the burden is (a) the combination yields a non-obvious result and (b) the *causal cheap feedback* is shown to be what drives it.

---

## 1. The two axes of "improving with feedback"

Feedback in a few-step causal editor can enter at two fundamentally different granularities. They correct **different error classes at different timescales** and are complementary, not redundant.

### Axis 1 — intra-block / denoising-step-wise (the "fast loop")
- **What it is:** within a single causal block's 5–15 denoising steps, act on the cheap intermediate `x0` prediction (the R22 step-dump literally materializes this) to correct *this* block before committing it.
- **Corrects:** edit not applied, artifacts, wrong edit region, block-0 errors.
- **Why it matters here specifically:** errors are *already present within the first causal block* (usual case), and block 0 has no committed history — so inter-block feedback structurally cannot touch it. Intra is the block-0 workhorse and seeds the KV cache every later block conditions on.
- **Caveat (crowded + unreliable early):** step-wise reward-guided sampling is a saturated literature and exists identically in *bidirectional* models — causality gives no special advantage here. Verifiers are also least reliable on noisy early-step decodes. → scope Axis 1 as a **special case / supporting component**, not the headline.

### Axis 2 — inter-block / frame-wise (the "slow loop")
- **What it is:** across committed blocks, read the realized decoded chunk *k* and steer chunk *k+1* (guidance schedule, correction term, re-injection, compute).
- **Corrects:** temporal drift, error accumulation, background/preservation loss over the horizon, cross-block consistency.
- **Why it matters here specifically:** this is the **only axis where "causal" is the differentiator** — the realized frame is available *and still actionable* before the next chunk. In a bidirectional editor the same idea requires regenerating the whole clip (outer-loop, offline). Both trained competitors (LiveEdit, JoyAI) name accumulated drift / background instability as *the* unsolved problem, and it is a frame-wise/rollout phenomenon. Training-free frame-wise feedback under an **editing preservation objective** is an open lane (existing frame-wise work is generation-only).

### Verdict on scope
**Both, unified under one principle** — but Axis 2 is the spine (carries novelty + the causal argument), Axis 1 is the subordinate special case (covers block 0 and within-block quality). Presenting them as "two tricks" invites the reviewer question *"is this just two stapled tricks / which one does the work?"* → a clean per-component ablation attributing which loop fixes which error class is **owed from day one** and is also the best figure.

---

## 2. Related work to READ (verified 2026-08-12; grouped by relevance)

### A. Closest to the *task* (training-free reward/feedback editing) — read first
- **ITOC — Training-Free Reward-Guided Image *Editing* via Trajectory Optimal Control** (ICLR 2026), arXiv:2509.25845 — https://arxiv.org/html/2509.25845v2 . Reward-guided editing as optimal control of the whole reverse trajectory (adjoint/PMP), training-free, explicitly balances reward vs source preservation and avoids reward hacking. *Closest to our objective; but images, inversion-based, whole-trajectory (opposite of cheap/causal). This is our preservation-vs-reward-hacking reference.*
- **SVDD — Derivative-Free Guidance via Soft Value-based Decoding** (NeurIPS 2025), arXiv:2408.08252 — https://arxiv.org/html/2408.08252v3 . Sample several candidates per step, keep highest soft-value, using `x_t→x0` expectation. *Our training-free, no-backprop route for the intra-block loop (candidate selection). Non-differentiable feedback ok.*

### B. Frame-wise / autoregressive drift (Axis 2 mechanism) — read for the slow loop
- **Pathwise Test-Time Correction (TTC)**, arXiv:2602.05871 — https://arxiv.org/html/2602.05871v2 . Training-free; anchor to the initial frame to calibrate intermediate stochastic states along the rollout; re-noise a corrected clean prediction back to the current level. *Purest training-free frame-wise correction — but T2V drift, single anchor, no edit/preservation objective. The re-noise-and-re-run actuator is directly reusable.*
- **FreqForcing — Spectral Self-Anchoring**, arXiv:2607.27110 — https://arxiv.org/html/2607.27110v1 . Training-free; low-freq anchor branch + high-freq local branch fused in frequency domain, **on Self-Forcing** (our backbone), 24× extrapolation. *Same model family; mechanism template for stabilizing without training.*
- **BAgger — Backwards Aggregation** (CVPR 2026), arXiv:2512.12080 — https://arxiv.org/abs/2512.12080 . *Training-based* corrective-rollout counterpart. *Read as the "you'd need training if you did it their way" contrast.*
- (context) **Causal-rCM**, arXiv:2606.25473 — unified TF/SF distillation recipe; background on exposure bias in streaming.

### C. Step-wise inference-time scaling / verifiers (Axis 1 context) — skim, know the landscape
- **Video-T1 / Tree-of-Frames** (ICCV 2025), arXiv:2503.18942 — https://arxiv.org/html/2503.18942v1 . Test-time search over trajectories with verifier; ToF expands/prunes *autoregressively* (overlaps our streaming setting).
- **Verifier Matters** (BMVC 2025) — https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1006/paper.pdf . **Key negative result:** verifiers are unreliable on noisy early-step samples; fine-tuning on partially-denoised samples helps. *Directly governs how early our cheap decodes can be trusted.*
- **LatSearch — Latent Reward-Guided Search**, arXiv:2603.14526 — https://www.alphaxiv.org/abs/2603.14526 . Trained *latent* reward model scores partially-denoised latents; reward-guided resampling/pruning, ~79% cheaper search. *Closest to "step-wise feedback" but needs a trained latent reward model, T2V quality not editing.*
- **EvoSearch**, arXiv:2505.17618 — https://arxiv.org/html/2505.17618 . Evolutionary test-time search over the denoising trajectory, training-free.
- (context) **DAS — Diffusion Alignment as Sampling** (SMC reward alignment), OpenReview `vi3DjUhFVm`.

### D. Competitive set — streaming editing (all TRAINED; our niche is training-free)
- **LiveEdit** (ECCV 2026), arXiv:2606.26740 — https://arxiv.org/html/2606.26740 . 3-stage distillation streaming editor; AR mask cache; 12.66 FPS; names background-preservation + latency as the two core issues.
- **JoyAI-Video-Edit** (Aug 2026), arXiv:2608.03974 — https://arxiv.org/html/2608.03974v1 . 16B trained AR streaming editor, 30 FPS 720p; explicitly fights accumulated temporal drift (instruction-based).
- **SANA-Streaming** — https://github.com/NVlabs/Sana/blob/78be97ae/docs/sana_streaming.md . Trained streaming v2v, cycle-reverse regularizer for consistency.

### E. Outer-loop self-refinement — read to DIFFERENTIATE AGAINST (offline / bidirectional / prompt- or weight-level)
- **VISTA** (CVPR 2026) — multi-agent test-time prompt refinement via pairwise tournaments; regenerates.
- **ReViSE** (Reason-Informed Video Editing, self-reflective learning), arXiv:2512.09924 — internal-VLM differentiable feedback *during training*.
- **JarvisEvo** (CVPR 2026) — self-evolving photo-editing agent, editor-evaluator RL.
- *These use output-as-feedback but only via full regeneration / training — exactly the contrast that makes the causal "cheap + actionable in-stream" argument land.*

---

## 3. Strategy to adopt — realized decoded state as a feedback signal at two timescales

**Principle:** one feedback source (the realized decode / cheap internal state) → two loops.

- **Intra-block fast loop (Axis 1, subordinate):** within a block, act on the cheap `x0` decode (or a free internal proxy) to fix the block before commit. Covers block 0 and within-block edit quality. Preferred instantiation: **derivative-free selection** (SVDD-style) or scaling the correction — avoid gradients (distilled few-step params are hypersensitive; TTC finding).
- **Inter-block slow loop (Axis 2, spine):** read committed chunk *k*, steer chunk *k+1*. Preferred instantiation: feedback-driven **guidance schedule** (R20/R23 already parameterize this) + a **decoded-background re-injection** (TTC-style) against source drift.

**Upper/lower bounds already in hand:** R23's per-video oracle over blend schedules = the *static* version of the inter-block controller → use as **upper bound**; best fixed schedule = **lower bound**. Headline claim: online feedback approaches the oracle *without* ground truth, and fixes drift no fixed schedule can.

**Training-free question:** an instantiation is training-free (SVDD/TTC/ITOC are). Caveat: "training-free" refers to *our pipeline* — the *scorer* may be off-the-shelf (CLIP/VLM/reward model) or, better, a **free internal signal** (see §4) that needs no decode at all.

---

## 4. What the feedback acts on (signal vs actuator)

Separate the two — reviewers attack the hand-wave here.

### Signal (what we measure)
- **Decode-based:** edit-region CLIP (edit success); background LPIPS/PSNR-to-source on non-edit region (preservation); VLM yes/no (expensive, sparing use).
- **Near-free internal (no decode):** the inter-branch velocity gap `|v_trg − v_src|` (already the S.O.G. cue) and the cross-attn foreground mask. *These may recover most of the gain with zero decode latency — a clean scientific ablation and the answer to the latency objection.*

### Actuator (what we change) — StreamEdit exposes an unusually large control surface, all already in `Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py`:
- **Guidance strengths:** `blend_power` (ρ), `blend_sched`/`blender_rate` (`_schedule_blend_rate`, R20/R23), `fg_boost_factor` (ω), `fg_scale`.
- **Injection timing:** `t_inj` — currently hard-fixed at `index == len//2`; make it feedback-driven.
- **The S.O.G. term itself:** `v_t = v_trg + bg_mask ⊙ (v_gt − v_src)` — scale / re-target from realized-background error, not only velocity error.
- **Discrete / compute:** SVDD candidate selection among sampled `x0`; early-exit a good block; **adaptive NFE** (extra steps for a bad block).
- **Re-injection (TTC-style):** re-noise a corrected decoded background to the current level and re-run — most direct "realized state" actuator, inter-block-friendly.
- **Rollout seeding:** `trg_initial_latent` / overlap handed to the next window in `rollout_inference`.

**Suggested first actuators:** intra → candidate selection or S.O.G. scaling on the cheap decode; inter → next-block blend/boost schedule (R23 oracle bound) + decoded-background re-injection. Continuous knobs first (low-risk, derivative-free).

### Pipeline dependence — **UNDECIDED, flagged as a design fork**
- **Signal is pipeline-agnostic** (on decoded frames / source comparison) — portable to LiveEdit, SANA-Streaming, any editor.
- **Actuator is pipeline-specific.** StreamEdit's rich knobs come *from* being training-free attention manipulation. A *trained* streaming editor exposes far fewer; there only **pipeline-agnostic actuators** port: candidate selection, adaptive NFE, latent/velocity correction (any flow/diffusion sampler). Prompt- vs instruction-based only changes conditioning, not the loop.
- **Fork to resolve:**
  - **(broad)** anchor the method on a pipeline-agnostic actuator (velocity/latent correction + selection); StreamEdit's knobs = "even more when the editor is training-free" bonus; demonstrate portability on one other editor.
  - **(deep, recommended for first submission)** lean fully into StreamEdit's actuators for max effect; claim generality only for the *principle*.

---

## 5. Why this is relevant in the streaming editing setting

1. **Causal = cheap AND actionable feedback.** The realized frame exists *and* the next chunk hasn't been generated yet — feedback is in-stream, unlike bidirectional editors that must regenerate the whole clip (VISTA/ReViSE/JarvisEvo are all offline outer loops). This single sentence is the memorable contribution.
2. **Targets the field's agreed failure mode.** LiveEdit and JoyAI both name accumulated drift / background instability as the core streaming-editing problem — a frame-wise phenomenon our slow loop attacks **without their 3-stage distillation**.
3. **Defends the training-free niche.** StreamEdit is the training-free point among trained competitors; a training-free feedback controller widens that moat.
4. **Favorable economics.** Frame-wise decodes once per chunk (≈ what we already pay) and scores a near-clean realized frame where verifiers are trustworthy (contra early-step unreliability) — better signal, lower marginal cost, defuses the "decode-in-the-loop kills streaming" objection.
5. **Infra is 80% built.** `rollout_inference` is the chunk loop; S.O.G. is already a background-preservation correction (in velocity space); R22 gives per-step decoded prefixes; R23 gives the oracle upper bound.

**Strongest objection to pre-empt:** decode-in-the-loop latency erodes the streaming pitch (<0.32 s/frame). Must show one of: (a) feedback on the already-required final decode → negligible overhead; (b) free internal signal recovers the gain (§4); (c) an explicit quality-vs-latency Pareto that still beats fixed-schedule at equal wall-clock. Secondary: reward hacking vs preservation (ITOC reference); distilled-param hypersensitivity → prefer **selection/search over gradient** guidance (TTC finding).

---

## 6. Upcoming tasks

### Reasoning / reading (do first, cheap)
- **T-R1.** Read §2.A (ITOC, SVDD) + §2.B (TTC, FreqForcing) in full; extract each one's *actuator* and *preservation-handling*. Deliver a half-page "actuator inventory" table (theirs vs ours).
- **T-R2.** Read §2.C Verifier Matters carefully → decide the earliest denoising step at which our cheap decode is trustworthy (governs the intra loop's entry point).
- **T-R3.** Write the differentiation paragraph vs §2.E (why in-stream causal feedback ≠ offline self-refinement) — this becomes the related-work spine.

### Code / analysis (de-risk before building)
- **T-C1 (gating experiment — DO THIS FIRST).** On existing **R22** dumps, decompose error into **(a) within-block** (is block *k* bad at step 1?) vs **(b) cross-block accumulation** (monotonic degradation over *k*?), using background LPIPS-to-source + edit-region CLIP. The (a):(b) ratio sizes the intra:inter budget and the owed ablation. *Also produces the motivation figure (drift curve) for free.*
- **T-C2 (signal validity).** From R22, correlate the **cheap early-step decode** metric and the **free internal signal** (`|v_trg − v_src|`, cross-attn mask) against the **final** edit/preservation metric. Establishes whether feedback is even trustworthy on this distilled model, and whether the *free* signal can replace decode.
- **T-C3 (inter-block controller v0).** Turn R23's static oracle into a per-chunk online policy: pick `blend_sched`/`blend_power`/`t_inj`/`fg_boost` for chunk *k+1* from the chunk-*k* signal. Baselines: R23 oracle (upper), best fixed schedule (lower).
- **T-C4 (intra-block loop v0, incl. block 0).** SVDD-style derivative-free candidate selection on the cheap `x0` decode within a block; or feedback-scaled S.O.G. Measure block-0 error reduction specifically.
- **T-C5 (frame-wise re-injection).** TTC-style: re-noise a corrected decoded background to current level, re-run for the next chunk; measure drift/preservation over the horizon.
- **T-C6 (component ablation — owed).** Intra-only vs inter-only vs both; attribute which loop fixes which error class (from T-C1's taxonomy).
- **T-C7 (latency Pareto).** Wall-clock vs quality for decode-based vs free-internal-signal variants; the answer to the main objection.
- **T-C8 (pipeline-dependence fork).** Port the pipeline-agnostic actuator (selection / velocity correction) to one other editor to decide broad-vs-deep framing (§4 fork).

**Recommended order:** T-C1 → T-C2 (gating + signal validity) → T-R1/T-R2 (reading in parallel) → T-C3/T-C4 (two loops v0) → T-C6 (ablation) → T-C5/T-C7/T-C8.

---

## 7. Honest contribution classification (novelty-check)
- **Category:** new problem framing + non-obvious combination (NOT a new mechanism — the step-wise and frame-wise tools exist).
- **Becomes publishable iff:** (a) the online loop yields a non-obvious result (online ≈ oracle, or fixes a failure fixed schedules can't), and (b) an ablation isolates that the *causal cheap feedback* drives it (not just "more compute").
- **Falsifiable core claims:** (1) streaming-editing error is dominated by cross-block accumulation (T-C1); (2) a cheap realized-decode (or free internal) signal predicts final quality early enough to act (T-C2); (3) online feedback closes most of the fixed-schedule→oracle gap (T-C3); (4) both timescales are needed — inter fixes drift, intra fixes block 0 (T-C6).
