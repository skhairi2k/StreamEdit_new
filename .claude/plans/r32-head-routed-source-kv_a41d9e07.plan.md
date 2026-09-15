---
name: R32 — Head-Routed Source-KV Injection
overview: >-
  Delete StreamEdit's global Q/K blend outright and reintroduce the source as a
  single appended K/V segment whose influence is reduced two independent ways:
  by how TEMPORAL the head is, and by how far denoising has progressed. The
  blend anchors every head identically, which is why its rate traces one
  edit-vs-motion trade-off curve that no global setting escapes. Routing makes
  anchoring a per-head, per-step decision so the curve can be left behind.
  Additive logit bias b_src[h,i] = -(alpha*w(h) + beta*g(i)), with alpha=0 and
  beta=0 ablating the two halves apart and a size-matched head permutation as
  the control that makes the result mean anything.
task_id: R32
todos:
  - id: head-weights
    content: "evaluation/r32_build_head_weights.py -> evaluation/r32_head_weights.pt.
      Read margin_flat [30,12] from evaluation/r19_tau_flat.pt and emit w(h) =
      (1 + tanh(margin/T))/2, T=0.5, so w=0 is strongly spatial and w=1 strongly
      temporal. Also emit the two controls: w_shuf (exact multiset permuted
      across heads at a fixed seed -- claim-2's falsifier) and w_rev (1-w).
      ASSERT R19's sign convention against the labels rather than trusting a
      comment: every SPATIAL head must have margin < 0 and every TEMPORAL head
      margin > 0. The experiment inverts silently if that is wrong and would
      still produce plausible numbers."
    status: pending
  - id: bridge-source-bias
    content: "wan/modules/causal_model.py -- retarget the already-landed per-head
      bias from the trg_prev segment to the SOURCE segment, and make it per-step.
      Record each segment's index as b_key_list is built (never recompute: the
      source segment's presence and length are config- and data-dependent).
      Read w from `self.seg_bias_w`, alpha/beta from `self.seg_bias_ab`, and g
      from kv_cache['shared_dict']['r32_g']; form row = -(alpha*w + beta*g) and
      pass it through the existing build_segment_logit_bias. `seg_bias_w is
      None` must reach the untouched baseline call."
    status: pending
  - id: pipeline-schedule
    content: "pipeline/edit_causal_inference.py -- install g(i) into
      shared_dict_dual each denoising step, right beside blender_rate, as
      g = 1 - _schedule_blend_rate(g_sched, index, len(denoising_step_list)).
      Reusing that helper verbatim gives the Eq. 4 shape (anchor hard at high
      noise, release as the sample resolves) plus cos_half/cos_third/const/zero
      for free, all already validated by R20. Add `_stamp_seg_bias_w` mirroring
      `_stamp_vp`, and its defensive clear -- the driver runs many arms in one
      process and a stale table would silently make the next arm a different
      experiment."
    status: pending
  - id: driver
    content: "evaluation/r32_arms.py following evaluation/r10_vp_arms.py exactly:
      module-level ARMS tuple as --arms default and choices; provenance refusal
      on the weights file; case-outer/arm-inner so the VAE encode is reused;
      seed reset before EVERY arm; out_root/{arm}/edit{T}/{video}/ (the edit{T}
      level is mandatory -- 0011_lucia appears under edit2 and edit5); per-run
      try/except recording a status row; manifest named per arm-set, since
      r32_infer.sh is an array over arms sharing one out_root."
    status: pending
  - id: smoke
    content: "slurm_scripts/five_bench/r32_smoke.sh -- four blocking gates in the
      r28_smoke.sh shape (numbered GATEs naming the bug each catches,
      GATEn-OK/GATEn-FAIL, non-zero exit). G1 head axis: bias one head, assert
      only that head's output moves and the other 11 are BITWISE unchanged.
      G2 segment: assert the bias lands on the source segment's token range and
      that the range is non-empty at every step under src_kv_full. G3 fires:
      alpha=4 must differ from alpha=0 above a floor. G4 floor: render one clip
      at alpha=beta=0 against the FlashAttention path and RECORD the delta --
      a biased arm cannot use flash, so every arm pays a kernel change, and no
      arm delta below that floor is readable."
    status: pending
  - id: slurm
    content: "slurm_scripts/five_bench/r32_infer.sh (array over the 12 arms, L40S,
      --gres=gpu:1, --mem=64G, --exclude=node52, --time=02:00:00, CASES
      overridable by \"$@\", trailing frame-dir self-check) and
      slurm_scripts/five_bench/r32_eval.sh (r10_metrics.py over all arms with
      the metric list in Decisions)."
    status: pending
  - id: summarize
    content: "evaluation/r32_summarize.py -- place every arm as a point in the
      (edit strength, motion fidelity) plane against the blend-sweep trade-off
      curve already measured in evaluation/csv/r32_editmetric/, and emit
      evaluation/figures/r32_tradeoff.pdf. The curve is the reference the whole
      claim is read against, so it is drawn from stored R20 numbers, not
      re-rendered."
    status: pending
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r32_smoke.sh
    status: pending

  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "tail -40 logs/r32_smoke_{job_id}.out -- all four gates print -OK. RECORD GATE4's flash-vs-zero delta: it is the noise floor, and no arm difference below it is readable."
    status: pending

  - id: launch-infer
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r32_infer.sh
    wait_for: wait-smoke
    sets_status: running
    status: pending

  - id: wait-infer
    type: manual
    wait_for: launch-infer
    check_hint: "sacct -j {job_id} --format=JobID,State; ls -d /projects/dataggen/outputs/five_bench/r32_head_routed/*/edit*/*/ | wc -l  (expect 12 x 22); grep -c 'failures: 0' logs/r32_infer_*.out"
    sets_status: finished
    status: pending

  - id: launch-eval
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r32_eval.sh
    wait_for: wait-infer
    status: pending

  - id: wait-eval
    type: manual
    wait_for: launch-eval
    check_hint: "ls evaluation/csv/r32_arms/r32_*_avg.csv | wc -l (expect 12); no nan in lpips_edit_part or motion_fidelity_score"
    status: pending

  - id: summarize
    type: local
    wait_for: wait-eval
    command: |
      python evaluation/r32_summarize.py \
        --arms_csv evaluation/csv/r32_arms \
        --blend_curve evaluation/csv/r32_editmetric \
        --out_csv evaluation/csv/r32_summary.csv \
        --out_fig evaluation/figures/r32_tradeoff.pdf
    output_paths:
      - evaluation/csv/r32_summary.csv
      - evaluation/figures/r32_tradeoff.pdf
    status: pending

  - id: verdict
    type: manual
    wait_for: summarize
    check_hint: "GO if `full` beats the blend-sweep curve -- more edit at equal motion, or equal edit at better motion -- by more than GATE4's floor, AND `full` beats `full_shuf` (claim 2: head type is a routing variable, not a proxy for injecting less). NO-GO if `full` lands ON the curve: routing then buys nothing a global scalar could not. SATURATION, a distinct outcome, if `full` and `full_rev` degrade symmetrically -- the heads are already at their preferred allocation and the margin has no headroom; that is not the same finding as 'routing does not work'. Record verdict + outcome bullet in daily.md."
    sets_status: analyzed
    status: pending
isProject: true
---

# R32: Head-Routed Source-KV Injection

## Context

StreamEdit anchors the target branch to the source with a **global Q/K blend**
applied identically to every head. Rescoring the R20 blend sweep on 2026-09-15
showed what that costs: across the full rate range, edit-region LPIPS moves
0.025 -> 0.049 and edit-region PSNR 30.4 -> 21.3 dB (**18/18 clips agree on the
direction**), while motion fidelity falls 83.88 -> 69.66. At maximum anchoring
the edit region is SSIM 0.970 to the source -- the edit substantially does not
happen. Every setting of that one scalar lands somewhere on a single
edit-vs-motion trade-off curve.

R32 removes the blend entirely and reintroduces the source as **one appended
K/V segment**, then reduces its influence per head and per step. Spatial heads
early see the source whole; temporal heads late see none of it. Goal: land
**off** the trade-off curve rather than somewhere on it.

**Done when:** all 12 arms are rendered on the 22 `cases.json` clips, scored on
edit strength and motion separately, and a GO / NO-GO / SATURATION verdict is
recorded against the stored blend-sweep curve, with the claim-2 permutation
control reported alongside.

## Execution steps

| # | id | type | what | sets_status |
|---|----|------|------|-------------|
| — | *(prep)* | — | `todos`: head weights, bridge, schedule, driver, smoke, slurm, summarize | — |
| 0 | smoke | sbatch | four blocking gates; G4 fixes the noise floor | — |
| 1 | wait-smoke | manual | all gates `-OK`, floor recorded | — |
| 2 | launch-infer | sbatch | array over 12 arms x 22 clips | `running` |
| 3 | wait-infer | manual | 12 x 22 frame dirs, failures 0 | `finished` |
| 4 | launch-eval | sbatch | `r10_metrics.py` over all arms | — |
| 5 | wait-eval | manual | 12 per-arm CSVs, no nan | — |
| 6 | summarize | local | trade-off figure vs the stored blend curve | — |
| 7 | verdict | manual | GO / NO-GO / SATURATION | `analyzed` |

```
/build-step head-weights R32     # then bridge-source-bias, pipeline-schedule,
                                 # driver, smoke, slurm, summarize
/run-step R32 smoke
/run-step R32 launch-infer
/run-step R32 summarize
```

## Decisions

| Decision | Choice |
|---|---|
| Base configuration | `blend_off=True` **and** `src_kv_full=True`. Both flags already exist. Together they give key set `[trg_prev \| src_current \| trg_current]` with pure-target prev/current and the **full** source current chunk appended at **every** step -- source enters the target branch at exactly one place |
| Why the existing injection gate is not inherited | The default source-KV path fires only on `current_timestep_index > total_timestep // 2`, i.e. the **low-noise** half. That is the opposite shape to Eq. 4's documented rationale (*"anchor hard at high noise and release as the sample resolves"*). Inheriting it would supply source exactly when structure is already fixed and appearance is being painted. `src_kv_full=True` bypasses the gate; `g(i)` then reinstates the correct shape |
| Bias form | **Additive, two independent factors**: `b_src[h,i] = -(alpha*w(h) + beta*g(i))`. Additive in logits = multiplicative on attention weight, and the two causes compose without interacting. `alpha=0` isolates the schedule, `beta=0` isolates head routing, both `0` is uniform full injection |
| Sign of the knob | One-sided: `b <= 0` always, so the bias only ever **reduces** source below its natural level, never amplifies. Worst case degrades toward "no source", which is a measured arm (`noblend_nosrc`), not an unknown |
| `w(h)` | `w = (1 + tanh(margin_flat/T))/2`, `T = 0.5`, from `evaluation/r19_tau_flat.pt`. `w=0` strongly spatial, `w=1` strongly temporal. Continuous, not thresholded: R19's margin histogram is **not** bimodal (~20% of mass within \|margin\|<0.2), so any cut is arbitrary, and the 183 DENSE + 11 MIXED heads land mid-range rather than being forced to a side |
| Sign convention, asserted not assumed | Negative margin = spatial, positive = temporal. Verified in `r19_tau_flat.pt`: all 117 SPATIAL heads in [-0.997, -0.159], all 49 TEMPORAL in [+0.184, +1.000]. The builder **re-asserts** this against the labels, because an inverted experiment still runs and still produces plausible numbers |
| `g(i)` | `g = 1 - _schedule_blend_rate(g_sched, i, n_steps)`, default `g_sched = cos_full`, so `g` rises 0 -> 1 over the schedule. Reuses the R20-validated helper verbatim, which also makes `cos_half`/`cos_third` free variants and gives two degenerate references: `const` => `g == 0` (beta inert), `zero` => `g == 1` (uniform `-beta`) |
| Source token set | Source **current chunk only**, foreground included. No `src_prev`: with blending off it would be the only past-side source signal, doubling the past key count and moving every head's softmax denominator -- a second mechanism, out of scope here |
| Arms (12) | `paper` (StreamEdit reference) · `noblend_nosrc` (motion floor) · `uniform` (a=0,b=0: mechanism only) · `head_a1/a2/a4` (b=0) · `time_b1/b2/b4` (a=0) · `full` (a*,b*) · `full_shuf` · `full_rev` |
| Controls, both required | `full_shuf` = the exact `w` multiset permuted across heads at a fixed seed -- **ideas.tex claim 2's stated falsifier** (head type must not be a proxy for injecting less). `full_rev` = `1-w`; symmetric degradation of `full` and `full_rev` means saturation, which is a different finding from "routing does not work" |
| Reference arm is `uniform`, not `paper` | A biased arm cannot run on FlashAttention, so every routed arm pays an SDPA kernel change; zeros-bias SDPA is not bitwise equal to `attn_mask=None` SDPA. `uniform` carries an explicit zero bias through the identical path. `paper` is the external reference, `uniform` the internal one, and GATE4 measures the gap between kernels as the readability floor |
| Coverage | The 22 `evaluation/cases.json` clips, edit types 1/2/5/6 -- the same set as R20/R21/R26/R30/R31, so the stored blend-sweep curve is directly comparable. Single-window path, clips truncated to 81 pixel frames |
| Edit-strength metric | `lpips_edit_part`, `psnr_edit_part`, `ssim_edit_part`, `structure_distance_edit_part` -- LPIPS/PSNR/SSIM between **source and target inside the edit mask**. Already implemented in `evaluate.py` and merely absent from the default `--metrics` list, which is why no prior run scored it. Demonstrated 18/18 clip agreement on the blend sweep; CLIP was only 4/18 and is demoted to a secondary column |
| Motion + preservation metrics | `motion_fidelity_score`; `structure_distance`, `psnr/lpips/mse/ssim_unedit_part`; `clip_similarity_target_image(_edit_part)` secondary |
| Sweep protocol | Two stages in one array: `head_a*` and `time_b*` fix the scales independently, then `full` runs at the (`a*`,`b*`) pair with the largest single-axis effect. Controls run at that same pair |
| Out of scope | The reference KV bank / `I^ref` routing (ideas.tex L3); `src_prev` injection; head-typed **query** routing (ideas.tex L1's `r^T=0`, subsumed here by removing the blend globally); per-token spatial gating (R26/R30/R31 own that axis); R11's general `patched_attn` refactor, explicitly dropped |

## Step commands

### smoke

```bash
cd ~/Code/StreamEdit_bigchantier
sbatch slurm_scripts/five_bench/r32_smoke.sh
```

### wait-smoke

```bash
tail -40 logs/r32_smoke_{job_id}.out    # four GATEn-OK lines; record GATE4's floor
```

### launch-infer

```bash
sbatch slurm_scripts/five_bench/r32_infer.sh
```

### wait-infer

```bash
sacct -j {job_id} --format=JobID,State
ls -d /projects/dataggen/outputs/five_bench/r32_head_routed/*/edit*/*/ | wc -l   # expect 264
grep -h "done:" logs/r32_infer_*.out
```

### launch-eval

```bash
sbatch slurm_scripts/five_bench/r32_eval.sh
```

### wait-eval

```bash
ls evaluation/csv/r32_arms/r32_*_avg.csv | wc -l      # expect 12
grep -l nan evaluation/csv/r32_arms/r32_*_avg.csv     # expect none
```

### summarize

```bash
python evaluation/r32_summarize.py \
  --arms_csv evaluation/csv/r32_arms \
  --blend_curve evaluation/csv/r32_editmetric \
  --out_csv evaluation/csv/r32_summary.csv \
  --out_fig evaluation/figures/r32_tradeoff.pdf
```

### verdict

```bash
# Read evaluation/figures/r32_tradeoff.pdf FIRST.
#  GO         : `full` sits OFF the blend curve by more than GATE4's floor,
#               AND beats `full_shuf` (ideas.tex claim 2)
#  NO-GO      : `full` lands ON the curve -- routing buys nothing a global
#               scalar could not
#  SATURATION : `full` and `full_rev` degrade symmetrically -- no headroom in
#               the margin; NOT the same finding as "routing does not work"
# Record verdict + outcome bullet in daily.md; update ideas.tex claim-1/2 status.
```

## Pipeline

```mermaid
flowchart LR
  TAU[r19_tau_flat.pt margin_flat] --> BW[r32_build_head_weights.py]
  BW --> W[r32_head_weights.pt: w, w_shuf, w_rev]
  W --> G{r32_smoke.sh: 4 gates}
  G -->|all OK| D[r32_arms.py]
  SCHED[_schedule_blend_rate] --> D
  D --> P[_stamp_seg_bias_w + shared_dict r32_g]
  P --> BR[causal_model: build_segment_logit_bias on the SOURCE segment]
  BR --> A["attention(attn_bias=...) -> SDPA"]
  A --> OUT[r32_head_routed/arm/editT/video/]
  OUT --> E[r32_eval.sh -> r10_metrics.py]
  E --> CSV[csv/r32_arms/]
  CSV --> S[r32_summarize.py]
  CURVE[csv/r32_editmetric blend sweep] --> S
  S --> FIG[figures/r32_tradeoff.pdf]
  FIG --> V{verdict}
```

## Code to touch

**Already landed, needs retargeting** — a per-head additive-bias SDPA path
exists and is verified (head isolation exact, segment restriction 4.88e-04,
8.60 ms vs flash 4.01 ms on the cutlass memory-efficient backend, 0.31 GiB).
It currently biases the `trg_prev` segment; R32 needs it on the **source**
segment and per-step.

- `Self-Forcing_StreamEdit/wan/modules/attention.py` — no change. `attn_bias`
  already refuses a head axis of 1 rather than broadcasting it, which is the
  failure mode GATE1 exists to catch.
- `Self-Forcing_StreamEdit/wan/modules/causal_model.py` — record segment indices
  as `b_key_list` is built; read `seg_bias_w` / `seg_bias_ab` / `shared_dict['r32_g']`;
  form `row = -(alpha*w + beta*g)` and target the **source** segment.
  Keep `build_segment_logit_bias` (its `.view(-1, 1)` is load-bearing: a bare
  `[Nh]` row broadcasts along the key axis and raises unless `len == Nh`).
  `seg_bias_w is None` must reach the untouched baseline call.
- `Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py` — add `g` to
  `shared_dict_dual` beside `blender_rate`; `_stamp_seg_bias_w` mirroring
  `_stamp_vp` with layer- and head-count guards, plus its defensive clear.

**New**

- `evaluation/r32_build_head_weights.py` — w / w_shuf / w_rev + the sign assert.
- `evaluation/r32_arms.py` — driver, per `r10_vp_arms.py`.
- `evaluation/r32_summarize.py` — arms as points against the stored blend curve.
- `slurm_scripts/five_bench/r32_smoke.sh` — the four gates.
- `slurm_scripts/five_bench/r32_infer.sh` — array over 12 arms.
- `slurm_scripts/five_bench/r32_eval.sh` — `r10_metrics.py`, metric list per Decisions.
