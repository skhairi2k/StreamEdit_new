---
name: R32 — Head-wise provenance routing (source-vs-target at the current frame)
overview: >-
  Resolve the appearance/motion trade-off by routing, per head, between the two
  candidates that sit at the CURRENT frame's RoPE indices: the source's current
  frame (copies the layout, preserves motion, but leaks source appearance back
  and under-edits) and the target's current frame (lands the edit, but loses the
  layout). Today a single global blend rate trades these off for every head at
  once; R32 makes it a per-head decision. Three phases: fix the edit metric
  (already implemented, never scored), measure which heads prefer which
  candidate (a target-branch probe -- R19 cannot answer this), then route.
  Redirected from Axis 1 (past-vs-current) on 2026-09-12; see Why-not-Axis-1.
task_id: R32
todos:
  - id: attn-bias-path
    content: "Add `attn_bias=None` to `attention()` in wan/modules/attention.py.
      Non-None => an explicit SDPA path wrapped in
      `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`; None => the FlashAttention
      dispatch is untouched. Bias arrives ALREADY in SDPA layout
      ([B, Nh, Lq|1, Lk], head at dim 1) and is never transposed. Validate the
      head axis loudly and REFUSE a head axis of 1 rather than broadcasting it."
    status: completed
    notes: >-
      AXIS-AGNOSTIC -- it biases whichever segment it is pointed at, so the
      redirect to Axis 2 changes the table and the segment index, not this.
      Verified on real shapes (Lq=4680, Lk=28080, Nh=12, D=128, bf16): biasing
      head 3 alone moves head 3 and leaves the other 11 BITWISE unchanged;
      restricting to a segment matches true restricted attention to 4.88e-04;
      all three malformed-bias refusals fire. 8.60 ms vs flash 4.01 ms (2.15x),
      peak 0.31 GiB -- the cutlass memory-efficient backend, not math.
  - id: bridge-call-site
    content: "wan/modules/causal_model.py: read `getattr(self, 'seg_bias', None)`;
      take segment lengths from the ACTUAL b_key_list entries; assert they sum to
      the concatenated key length; build the bias via a module-level
      `build_segment_logit_bias`. Guard the untested VP+bias combination and
      sink_tokens != 0."
    status: completed
    notes: >-
      A latent bug was caught and fixed here: the helper assigned a [Nh] row into
      an [Nh, len_prev] slice, which broadcasts along the KEY axis and simply
      raises for len_prev != Nh. Needs `.view(-1, 1)`. Verified for
      len_prev in {Lk-Lq, 0, Lk} -- 0 matters, it is block 0 where no past exists
      yet. R32_TRACE=1 logs (layer, step, seg_lens, Lk) per biased capture.
  - id: pipeline-plumbing
    content: "pipeline/edit_causal_inference.py: `seg_bias_table` kwarg on
      rollout_inference + inference, forwarded at both inner call sites;
      `_stamp_seg_bias` / `_clear_seg_bias_stamps` mirroring the VP pair, with
      layer- AND head-count guards; defensive clear alongside the VP clear."
    status: completed
  - id: edit-metric
    content: "PHASE 1. The `*_edit_part` family (lpips/ssim/psnr/structure_distance
      between SOURCE and TARGET restricted to the edit mask) is ALREADY
      implemented in evaluation/fivebench/evaluate.py:130-154 and is simply
      ABSENT from the default --metrics list, which is why no run has ever
      scored it. `lpips_edit_part` is the under-editing measure: high = the
      region diverged from the source = edited; low = still looks like the
      source. Re-score the EXISTING R20 blend sweep (const=no blending ...
      zero=max blending, all on disk) plus R10's arms. No new evaluator code."
    status: in_progress
  - id: provenance-probe
    content: "PHASE 2. An R19-style equal-budget DISJOINT probe run in the TARGET
      branch, with the two candidates that share the current frame's RoPE
      indices: `src_current` vs `trg_current`. Budget-matched by construction,
      SVG Algorithm 1 on sampled query rows, two synthetic null gates before any
      model number is read. Emits a per-head label: layout-preferring vs
      edit-preferring. This is the measurement R19 structurally cannot provide."
    status: pending
  - id: routed-arms
    content: "PHASE 3. Bias the `src_current` segment per head from the probe's
      labels, against zero / reversed / uniform / shuffled controls. Likely
      splits into its own task once the probe returns, since its design depends
      on what the probe finds."
    status: pending
steps:
  - id: rescore-edit-metric
    type: manual
    command: >-
      python evaluation/r10_metrics.py --in_root
      /projects/dataggen/outputs/five_bench/r20_blend_sched --arms const_novp
      zero_novp cos_third_novp cos_full_novp --metrics lpips_edit_part
      ssim_edit_part psnr_edit_part structure_distance_edit_part
      clip_similarity_target_image_edit_part clip_similarity_target_image
    check_hint: >-
      Does lpips_edit_part SEPARATE const_novp (no blending) from zero_novp (max
      blending)? If yes, the under-editing claim is measurable and Phase 2 can be
      read. If lpips_edit_part is as flat as CLIP was (21.8-22.1 across the whole
      sweep), the metric problem is NOT solved and no routing result will be
      readable either -- stop and find a metric that works first.
    status: in_progress
  - id: probe-gates
    type: manual
    check_hint: "synthetic nulls PASS -- null margin ~0 with src_current/trg_current budget-matched; a planted layout-head reads as such at every N"
    wait_for: rescore-edit-metric
    status: pending
  - id: launch-probe
    type: sbatch
    wait_for: probe-gates
    sets_status: running
    status: pending
  - id: probe-verdict
    type: manual
    wait_for: launch-probe
    check_hint: >-
      GO if the src-vs-trg margin separates heads with low flip rates across
      videos/steps/blocks AND beats a count-matched random control. NO-GO if
      margins are unimodal near 0: spatial heads do not differentiate provenance,
      and per-head provenance routing has nothing to route on -- which would be a
      real negative result worth recording, not a failure.
    sets_status: analyzed
    status: pending
isProject: true
---

# R32: Head-wise provenance routing

## Context

The goal is to adapt a streaming generator's KV cache to editing, training-free,
by resolving the appearance/motion trade-off. This plan locates that trade-off
precisely and then routes it per head.

### Where the trade-off actually is

RoPE decides which head can read which key. `causal_rope_apply_multi_chunk(...,
start_frame=0)` (`causal_model.py:316`) ropes the whole cached key sequence at
its true absolute frame indices, and `src_key, trg_key = attn_key.chunk(2)` are
both slices of it. So the cache is a 2x2:

| | **past** RoPE → **temporal** heads read here | **current** RoPE → **spatial** heads read here |
|---|---|---|
| **source** | `src_prev` — exists in code, never appended as a segment | `src_current` — appended, background-only, late steps only |
| **target** | `trg_prev` — appended (FG-blended with `src_prev`) | `trg_current` — appended (blended with `src_current`) |

The target needs the **layout of the frame it is generating**. That information
exists only in `src_current`, which sits at **current** RoPE. So:

- **Current motion is delivered by SPATIAL heads, not temporal ones.** A spatial
  head does not have to *compute* motion (which would need two time points) --
  the source video already did, and the head reads the layout straight off
  `src_current`.
- **`src_prev` cannot substitute.** It is *past* motion, so it cannot supply the
  layout of the frame being generated.

Therefore both `src_current` and `trg_current` are spatial-head territory, at the
same RoPE indices, and they want opposite things:

- `src_current` → copy the source layout ⇒ motion preserved, source appearance
  leaks back, **under-editing**
- `trg_current` → apply the edited appearance ⇒ edit lands, **layout drifts**

**That is the appearance/motion trade-off, and it is entirely inside the spatial
heads at the current frame.** It is a *provenance* question, not a geometry one.

### Why not Axis 1 (past-vs-current), which this plan originally proposed

Axis 1 is well-posed and RoPE-aligned -- past-vs-current is exactly the
temporal-vs-spatial split, so R19's labels match the geometry rather than
fighting it. But the conflict lives *inside* one of those two groups, so Axis 1
would measure something real and answer a different question. The mechanism
built for it is axis-agnostic and carries over unchanged; only the table and the
biased segment index change. The Axis-1 artifacts
(`evaluation/r32_build_seg_bias.py`, `r32_seg_bias_arms.py`,
`slurm_scripts/five_bench/r32_smoke.sh`) are **parked, not wrong** -- they are a
valid cheap side-experiment if ever wanted.

### Why R19 cannot route this

R19 labels heads spatial vs temporal, i.e. own-frame vs other-frames. Both
provenance candidates are own-frame. A spatial/temporal label therefore carries
**no information** about which of them a head prefers. Phase 2 exists because
this measurement does not exist yet -- and it can only be made in the **target**
branch, since both candidates only co-exist under a real edit. (That is the
non-transferability point: R19 profiles the source branch, where provenance is
degenerate.)

### Why the current mechanism is the wrong lever

Motion preservation today is a **global blend rate**: `k = trg*r + src*(1-r)`
(`causal_model.py:471`, `:502`), applied to every head at once. Two properties
make it a poor instrument:

1. **It blends queries and keys only -- values stay pure target.** It reshapes
   *where* attention looks without changing *what* it retrieves. At `r≡0` that is
   a Q/K-vs-V mismatch, which plausibly explains why the `zero` arm degrades
   everything at once rather than simply looking more like the source.
2. **It is not per-head.** Every head pays the same appearance dilution to buy
   motion, which is exactly the trade-off routing is meant to escape.

R21's full sweep: blend rate moves `motion_fidelity` by **14 points** and
`structure_distance` by **26**, while every CLIP edit metric moves by ~0.1-0.3.
On the metrics as scored, blend rate is a motion/structure knob and edit strength
is invisible -- which is a **measurement failure**, not evidence that blending
does not under-edit. Hence Phase 1 comes first.

Done when: the edit metric demonstrably separates the blend extremes, the
provenance probe returns per-head labels with null gates passed and a
count-matched random control, and a GO/NO-GO on per-head provenance routing is
recorded.

## Phases

| # | phase | what | gate to the next |
|---|-------|------|------------------|
| 1 | **edit metric** | score the `*_edit_part` family on the existing R20 sweep | `lpips_edit_part` must separate `const_novp` from `zero_novp` |
| 2 | **provenance probe** | R19-style equal-budget disjoint probe, target branch, `src_current` vs `trg_current` | null gates + random control; margins not unimodal at 0 |
| 3 | **routed arms** | bias `src_current` per head vs zero/reversed/uniform/shuffled | — |

## Decisions

| Decision | Choice |
|---|---|
| **Phase 1 needs no new evaluator code** | `psnr/lpips/mse/ssim/structure_distance_edit_part` are implemented at `evaluate.py:130-154` and are merely missing from the default `--metrics` list (`:723-737`). `lpips_edit_part` = LPIPS(source, target) inside the edit mask = the under-editing measure. Re-score, do not rebuild |
| Phase 1 uses existing renders | The whole R20 blend sweep is on disk, including both extremes: `const_*` (r≡1, **no** blending) and `zero_*` (r≡0, **max** blending). The `novp` family isolates blending from the visual prompt. Zero new renders |
| Phase 1 is a hard gate | If `lpips_edit_part` is as flat as CLIP was, no routing result is readable either. Find a working metric before spending GPU on Phase 2 |
| Phase 2 probe design | SVG Algorithm 1 unchanged; the departure is the key sets, as in R19. Candidates `src_current` vs `trg_current` are **naturally budget-matched** (both are the current chunk, `Lq` tokens each) and **naturally disjoint** (different provenance, no shared token) -- so the two problems R19 had to engineer around are free here. Still gate on synthetic nulls |
| Phase 2 must run in the target branch | Both candidates only co-exist under a real edit. Run with blending OFF so the keys are pure and the winner is attributable, then again at the deployed rate to measure how much the label moves |
| Mechanism | **SDPA with a float additive bias**, `[1, Nh, 1, Lk]` bf16. Benchmarked: 8.66 ms vs flash's 3.93 ms (2.15x), 0.01 GiB extra -- torch 2.8 routes a non-None bf16 mask at these shapes to the **cutlass memory-efficient** backend, honouring the stride-0 query-axis broadcast without materializing logits |
| Rejected: flex `score_mod` | `Lk` is data-dependent, so with `dynamic=False` the **9th distinct shape** hits `recompile_limit(8)` and silently falls back to eager flex: **208 ms/call (47x) and +18.4 GiB**, correct numbers, one warning line. `dynamic=True` needs a separate compiled handle, and **Triton is broken in this env** (`FileNotFoundError: /usr/bin/gcc-13`; only 14.3.1) -- which also means the repo's existing flex training path is broken here |
| Rejected: head-dim padding to keep flash | FA2 rejects `v` with `head_dim != q/k`, and the default `softmax_scale` silently becomes `1/sqrt(136)`, rescaling every logit by 0.970 -- a non-crashing, plausible-looking, globally-wrong temperature. Three silent-bug surfaces for 25% speed |
| **The `zero` arm is the reference -- not the flash baseline** | A biased arm cannot run on flash, so every arm including the control passes an explicit bias through the identical SDPA path. Zeros-bias SDPA is **not** bitwise equal to `attn_mask=None` SDPA (measured 2.4e-4). A `flash` arm measures that gap as the **noise floor**; no arm delta below it is readable |
| Source branch stays on flash | It is arm-invariant, so it needs no bias and should not pay 2.15x |
| Out of scope | Axis 1 (parked); per-step bias schedules; reading the bias off the live source branch (the natural follow-on -- the source branch is recomputed every block anyway, so the per-video adaptation FFP-300K trains a module to predict could be measured for free) |

## Findings recorded along the way

- **R10's temporal null has a third explanation, and it is mechanical.** The VP
  bank was re-roped to **zero temporal offset** -- "reads as present in every
  block" (`causal_model.py:509-511`) -- i.e. to the *current* frame index. So the
  `temporal` arm injected a current-frame key into heads that, by construction,
  attend to *other* frame indices. Temporal heads structurally could not see it.
  This is distinct from "temporal heads do not carry motion" and from "the arm
  got 49 of 88 heads", and it **predicts the spatial arm should work** -- which
  it did: 8/8 metrics over its count-matched random control, and motion fidelity
  80.93 vs 77.10 for `all`. Any future injection must be roped where its intended
  readers actually look.
- **The blend is Q/K-only; values are pure target** (`b_trg_*_value` untouched).
  So "more blending" cannot add source *content*, only source attention geometry.
- **`*_edit_part` has existed all along and was never scored.** Every
  under-editing question asked of R10/R20/R21 was asked of a metric set that
  could not answer it.
- **R10 still has no outcome entry** (workboard `implemented`; last entry
  2026-07-20 is the analyze job launching). Its result exists only as seven
  unread CSVs. Close it out alongside R32.

## Code to touch

Landed and axis-agnostic:
- **`Self-Forcing_StreamEdit/wan/modules/attention.py`** — `_sdpa_attention_with_bias` + `attn_bias=`.
- **`Self-Forcing_StreamEdit/wan/modules/causal_model.py`** — `build_segment_logit_bias`, `_R32_TRACE`, the `seg_bias` read + guards, the third branch at the target attention call.
- **`Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py`** — `seg_bias_table` plumbing, `_stamp_seg_bias`, `_clear_seg_bias_stamps`.

Parked (Axis 1, valid but not the current question):
- `evaluation/r32_build_seg_bias.py`, `evaluation/r32_seg_bias_arms.py`, `slurm_scripts/five_bench/r32_smoke.sh`.

Phase 2 (new, not yet written):
- a target-branch provenance profiler + its null gates, following `evaluation/r19_gates.py` / `r19_analyze.py`.

## Risks

- **The probe may find no separation.** If spatial heads do not differentiate
  provenance, per-head provenance routing has nothing to route on. That is a
  real negative result and should be recorded as one, not treated as a failure.
- **Appending or re-weighting `src_current` changes the softmax denominator** for
  every head, so R19's budget-matching discipline reappears at the mechanism
  level. The `zero` arm alone does not control for it.
- **Uncertain layer band.** R19 found `best_rel` worst at layers 15, 18-19,
  21-24 in 21/21 videos -- mid-network, where the edit signal most needs to land.
- **Phase 1 could fail.** `lpips_edit_part` is masked-region LPIPS on a
  multiply-by-mask image, which is somewhat out of distribution. If it does not
  separate the blend extremes, the fallback is a DINO/CLIP image-image delta in
  the mask, or a small VLM/human preference count on ~10 clips.
