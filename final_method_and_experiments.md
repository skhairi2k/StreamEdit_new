# Final Method and Experimental Plan

**Project:** Draft-derived adaptive control for propagating video edits  
**Status:** Working plan as of 12 October 2026. Distinguish completed experiments from hypotheses that still require validation.

## 1. Paper question and claim

### Main question

Can a cheap, high-editability draft reveal **where additional editability is needed**, so a video-editing method can use different release times or edit strengths for different spatial tokens instead of one global control value?

### Intended core claim

> A draft-derived, per-token edit-demand field improves the editability–preservation trade-off over a tuned global-control baseline at equal total NFE, especially for shape-changing edits.

The main method is **StreamEdit with visual prompt/anchor conditioning on the Self-Forcing backbone (VP+SF)**. Text-only StreamEdit and a second backbone are generalization experiments, not prerequisites for the core claim.

### Terminology

- **VP:** visual prompt / edited image anchor.
- **TXT:** text-only conditioning.
- **SF:** Self-Forcing backbone.
- **LL:** LongLive backbone.
- **Draft length** `k`: number of model steps used to produce the diagnostic draft.
- **Final length** `n`: number of steps in the final edited video.
- **Total budget:** count the draft and final computations, including both branches where applicable. For the main 5-step-draft experiment, `k=5`, `n=15`, so nominal total is `B=20` NFE.

## 2. Proposed method: attention-supported draft divergence

### Motivation

Raw DINO/LPIPS divergence detects **where the draft changed**, not whether each change is intended. On some hosts, relaxing edit constraints reveals the desired object shape but also introduces background artifacts. Those artifacts then appear in the divergence map and can be mistaken for additional edit demand.

The proposed solution separates two signals:

1. **Where is editing allowed?** Estimate object support from source-branch and target-branch spatial-to-text attention.
2. **Where is more editability needed?** Estimate draft–source divergence with DINO or LPIPS.

The support estimator is external to each method's native editing mechanism. It should not use StreamEdit's Q/K blend, UniEdit-Flow's velocity-difference mask, FlowDirector's native mask, or benchmark ground-truth masks.

### Two-run procedure

**Run 1 — high-editability draft**

1. Run the selected host in a draft configuration that lets the requested edit express itself strongly. The draft may contain unwanted background changes; it is diagnostic, not the final output.
2. On the final draft denoising step, extract spatial-to-text attention from both available branches:
   - source branch, attending to the source-object phrase;
   - target branch, attending to the target-object phrase.
3. Build source and target soft support maps, `S_src` and `S_tgt`, from those attention maps. Aggregate attention over heads/layers/tokens using one fixed rule across hosts. A contrast between attention to the object phrase and attention to other prompt words is a candidate way to reduce diffuse scene-context attention; validate and freeze the rule in the pilot.
4. Combine supports with a soft union:

   `S(p) = 1 - (1 - S_src(p)) * (1 - S_tgt(p))`.

   The union matters for shape changes because the target silhouette may extend beyond the source silhouette.
5. Compute a spatial draft–source divergence map `D(p)` (initially use the project's established LPIPS-first2 signal; evaluate DINO as a controlled alternative, ideally from the same draft outputs).
6. Normalize divergence with a robust **clip-level** scale, not independent framewise min–max normalization, which could amplify small artifacts in an otherwise quiet frame. Denote the result `D_norm(p)`.
7. Build the attention-supported edit-demand field:

   `E(p) = S(p) * D_norm(p)`.

   Outside estimated support, assign the most source-preserving control. Inside support, let the divergence determine the local edit demand.

**Run 2 — adaptive final edit**

Map `E(p)` monotonically to the host's native per-token control:

- high supported divergence → earlier release / higher editability;
- low supported divergence → later release / stronger source preservation;
- outside the support → minimum-editability / maximum-preservation setting.

The controller should preserve the same interpretation of “more control” across hosts even though the parameterization differs. For StreamEdit this is a per-token release/blending value; for UniEdit-Flow it is a token-dependent delay/release schedule. A FlowDirector implementation would instead naturally test a per-token edit-amplification parameter and thus supports a broader adaptive-strength claim.

### Method-agnostic scope and caveat

The proposed estimator is **architecture-interface-agnostic**, not guaranteed to work unchanged in every DiT or U-Net. It requires source/target branches and accessible spatial-to-text attention. Some models use joint attention, fused attention kernels, or do not expose meaningful attention matrices; these may require a small extraction adapter. The intended invariant is the same support/divergence estimator and mapping policy—not zero code changes in every host.

Use the final draft step first to avoid extra denoising compute. If attention maps are unstable, test averaging the last few draft steps as a predefined fallback. Do not add a new model run solely to extract attention if the existing draft execution can expose it.

### Important limitation

Attention support is an estimate of *where the prompt refers*, not a guarantee of semantic correctness. It may miss an object in a poor draft, diffuse onto the surroundings, or confuse similar objects. The pilot must inspect these maps before committing to a large experiment. The support map is not ground truth.

## 3. Host-specific adapter: keep it as small as possible

The common controller should produce the same `E(p)` field. A small adapter maps it to each host's control mechanism; do not redesign each method's full algorithm.

### StreamEdit

Use `E(p)` to set the per-token release / Q-K blending control. Keep the host's other settings fixed. This is the primary and most mature implementation.

### UniEdit-Flow: first generalization candidate

UniEdit-Flow's global delay parameter `alpha` is the closest match to a release-time hypothesis; keep guidance strength `omega` global and fixed during the first adaptive comparison. Validate/tune the global baseline sufficiently before freezing it. Use a fixed global maximum window to define the computation budget, and compare at matched actual NFE.

The observed high-alpha outputs show that stronger object editing can coincide with increased background changes. Therefore, **do not merely substitute per-token alpha values and assume the problem is solved**. The adaptive gate must keep out-of-support tokens on the source-preserving path throughout the relevant update, including any correction/fusion path through which background changes can otherwise accumulate. Implement this as a small control adapter, document the exact insertion points, and keep the rest of UniEdit-Flow unchanged. If this requires substantial algorithm redesign, stop rather than present it as a clean transfer.

### FlowDirector: fallback / alternative

If UniEdit-Flow proves too invasive or its support maps do not yield a clean result, pilot FlowDirector. A reasonable token-dependent choice is its DA-FC edit-amplification factor, while keeping the native cross-attention mask and other hyperparameters fixed. This tests **per-token adaptive edit strength**, not exactly the same per-token release-time mechanism. Do not run both second hosts at full scale unless the first experiment produces a compelling result and compute allows it.

### Wan-Edit

Do not make unmodified Wan-Edit the primary generalization experiment at this stage. Its high-editability draft produced desired object changes together with substantial background artifacts, making raw divergence ambiguous. Keep this as a diagnostic failure case unless a later pilot shows the shared attention-supported field addresses the problem.

## 4. Experiment plan and priorities

## Phase A — core StreamEdit result (VP+SF)

### Already completed

- **A0.** `STREAMEDIT (VP+SF)` global-`rho` sweep.
- **A1.** `STREAMEDIT** (VP+SF)` with full draft `k=15`.

These should not be rerun without a concrete implementation or evaluation change.

### Next required runs

1. **A2.** `STREAMEDIT** (VP+SF)`, `k=5`, followed by the 15-step final pass.
2. **A3.** Global-`rho` VP+SF baseline at **20 NFE** (one 20-step pass), tuned sufficiently to establish a strong global baseline.
3. **A4.** Compare `k=5` adaptive against the 20-NFE tuned global baseline on FiVE-Shape + FiVE-Control first.
4. **A5.** If the existing `k=15` result is reported as a budget-matched result, add the 30-NFE global reference. Otherwise label `k=15` as the proof-of-concept only and do not imply an equal-compute win from it.
5. **A6.** Track actual NFE and wall-clock, including draft, divergence, support extraction, and final generation. Report model evaluations and any external vision computation separately.

**Decision:** Initially skip `k=7`. Keep it as a fallback only if `k=5` is inconclusive or clearly loses too much performance. Do not claim `k=5` is empirically optimal without comparing other values; say it is the selected short-draft setting if it passes the comparison.

## Phase B — essential causal comparators

Run on the same FiVE-Shape + FiVE-Control subset and at the same total NFE as the main adaptive method (`B=20` for `k=5`, `n=15`).

1. **B1. Anchor-only field:** derive a control field from anchor frame 0 versus source frame 0, without the diagnostic draft. Test both:
   - **broadcast:** reuse the exact frame-0 spatial field at the same pixel/token coordinates in later frames;
   - **flow-warped:** move the field using source optical flow.
2. **B2. Region/mask timing:** a two-region comparator (inside support/mask versus outside) without continuous per-token divergence-based timing. Do not use FiVE ground-truth masks to construct it; use the allowed method-native or independently generated mask and clearly state its source.
3. **B3. Compare:** tuned global control, anchor-only, region timing, and proposed attention-supported divergence routing.

These runs answer the two principal objections: “Why is the draft needed?” and “Why is per-token control better than a region-wise mask?”

## Phase C — test the new support-conditioned field cheaply

Before expanding to more backbones or a second method, run a small, fixed pilot set covering a non-rigid shape change, an object replacement, and a low-shape-change/control edit. Use exactly the same clips for each variant.

Compare:

| Variant | Field / control | Purpose |
|---|---|---|
| C1. Global baseline | Native global control | Reference |
| C2. Raw divergence | `D_norm` without attention support | Reproduce the failure mode observed in Wan-Edit / UniEdit-Flow |
| C3. Support only | `S` without divergence weighting | Test what localization alone contributes |
| C4. Support-conditioned divergence | `S * D_norm` | Test whether divergence adds useful local edit demand after suppressing out-of-support changes |

Do this first on StreamEdit VP+SF and UniEdit-Flow if the attention maps are accessible. Inspect source support, target support, the union, divergence, and final `E` before rendering. Evaluate intended-object editing and collateral background changes separately.

**Gate C:** only proceed to a larger second-host benchmark if C4 reduces unwanted background change without suppressing the intended object/shape transformation, and it improves on C2 and C3. If support itself is unreliable, fix or reject the estimator before another expensive campaign.

## Phase D — conditioning portability (TXT+SF)

This is a deliberate part of the story: does the adaptive controller work without an image anchor?

1. **D1.** `STREAMEDIT (TXT+SF)` global-`rho` baseline at official/default **15 NFE**.
2. **D2.** TXT+SF global-`rho` at **20 NFE**, tuned sufficiently for the equal-NFE comparison.
3. **D3.** `STREAMEDIT** (TXT+SF)` with `k=5` draft + 15-step final run (**20 NFE**).
4. Compare adaptive versus tuned global control at 20 NFE.

Do not run TXT+LL. Do not use a comparison only against the default `rho` if a tuned global baseline is available.

## Phase E — backbone portability (VP+LL)

1. **E1.** `STREAMEDIT (VP+LL)` global-control baseline at **15 NFE**.
2. **E2.** VP+LL global control at **20 NFE**, tuned sufficiently for equal-budget comparison.
3. **E3.** `STREAMEDIT** (VP+LL)` using the selected `k=5` controller and 15-step final run.
4. Compare at 20 NFE.

Do not run TXT+LL. This tests the backbone axis without building the full VP/TXT × SF/LL factorial matrix.

## Phase F — second editing method beyond StreamEdit

**First choice:** UniEdit-Flow pilot with the same attention-supported divergence field and a minimal adapter to its delay/release mechanism.

1. Establish its native global baseline and a small, defensible global-`alpha` tuning range on the pilot set.
2. Keep `omega` global; do not make both delay and guidance strength adaptive in the same experiment.
3. Compare raw divergence, support-only, and support-conditioned divergence routing.
4. Keep its native velocity-based editing mechanism intact except for the narrowly defined adaptive gate needed to enforce local release/preservation.
5. Compare against the tuned global baseline at equal **actual** NFE, accounting for inversion, correction, draft, and final passes as appropriate.
6. If the pilot succeeds, expand to a fixed FiVE-Shape + FiVE-Control subset and report both object-edit success and background preservation. Full-FiVE evaluation is optional, not an initial requirement.

If UniEdit-Flow is too invasive or fails the pilot, evaluate FlowDirector instead, changing only its DA-FC amplification factor token-wise and freezing other knobs. Do not implement both at full scale by default.

## Phase G — benchmark integrity and metrics

### FiVE annotation audit

There are known instruction/target-prompt inconsistencies that may affect results. Do not silently replace official annotations.

1. Audit and classify defects before expensive full-benchmark runs.
2. Create a frozen correction mapping: sample ID, original annotation, corrected annotation, defect class, rationale, and whether the instruction or any evaluation field changes.
3. Restrict FiVE-C corrections to objectively demonstrable textual inconsistencies; do not add content or change intended edits. If intent cannot be recovered unambiguously, flag rather than invent a correction.
4. Preserve **official FiVE** results for comparability and report **FiVE-C** as an explicitly named corrected annotation variant.
5. Rerun the methods used for any headline comparison on the same FiVE-C inputs. Do not compare our FiVE-C result against another method's official-FiVE number.

### Evaluation

- Primary benchmark: official FiVE, 419 pairs; focus early experiments on the frozen FiVE-Shape and FiVE-Control subsets.
- Keep full FiVE per-edit-type reporting, including removal/type 6 as a limitation rather than silently dropping it from the full benchmark.
- Main axes: edit alignment, source preservation, temporal consistency.
- Use the tuned global-control curve as the main comparator. For matched-metric analysis, match alignment at preservation (LPIPS and PSNR separately) and preservation at alignment (FiVE-Acc); do not extrapolate outside the baseline curve.
- Keep the anchor-zone background metric as a secondary, input-derived evaluation region with explicit limitations.
- Optional metric investigation: a boundary-tolerant LPIPS sensitivity curve derived from a soft distance-to-source-mask weighting. Do not choose a different tolerance per edit after seeing outcomes, and do not replace the official hard-mask metrics. Treat this as exploratory unless validated.
- Use per-video paired statistics/bootstrap for borderline gains; report losses as well as wins.

## Phase H — external baselines (after the core gate)

Run external baselines only after the main equal-NFE and causal comparisons are promising. Use the same fixed Qwen anchors and one consistent benchmark protocol; do not introduce FLUX anchors as a second baseline universe.

1. **AnyV2V:** one-clip smoke test, then a larger/full evaluation if feasible.
2. **ContextFlow:** one-clip smoke test first; it is comparatively expensive. Tune a small global-`tau` set on Shape + Control, then run the selected configuration more broadly if worthwhile.
3. **VACE:** trained/reference baseline.
4. **FreeProp/FFP-300K:** run if public code/weights and reproduction cost permit.
5. **I2VEdit / LoRA-Edit:** subset-only references because of per-video optimization/training cost.

Use the same anchors and corrected annotations for methods compared on FiVE-C. Do not require all external baselines to be rerun with another image editor.

## 5. Explicit cuts / non-goals for this iteration

- **Drop FLUX Kontext / VP2** from the main plan; it would create a second anchor and baseline campaign.
- **Drop the frame-0 premise experiment** comparing video-editor output against the image-editor anchor.
- **Do not run TXT+LL** unless a new result creates a specific reason.
- **Skip `k=7` initially**; run only if `k=5` is inconclusive.
- **Do not tune multiple host-specific parameters simultaneously** in the second-method experiment.
- **Do not run UniEdit-Flow and FlowDirector at full scale by default**; pilot one, use the other only as a fallback.
- **Do not start full-benchmark external-baseline runs before the core equal-NFE gate passes.**

## 6. Execution order at a glance

1. Implement attention extraction on the existing VP+SF draft; inspect source/target support and form `E = S * D_norm`.
2. Run VP+SF adaptive `k=5`; run global baseline at 20 NFE; compare against already completed `k=15` only with a 30-NFE reference if making an equal-budget claim.
3. Run anchor-only (broadcast and flow-warped) and region-timing comparators at 20 NFE.
4. Decide whether the main StreamEdit result is convincing at equal NFE.
5. Run small support-conditioned pilot: global / raw divergence / support-only / support-conditioned divergence.
6. If the field is reliable, run TXT+SF and VP+LL baseline + adaptive pairs at 15 and 20 NFE as specified.
7. Pilot UniEdit-Flow as the second method; retain FlowDirector as a fallback.
8. Freeze FiVE-C annotation corrections and evaluation manifests before full-benchmark reruns.
9. Run selected external baselines, full FiVE final evaluations, metric matching/statistics, figures, and supplementary material.

## 7. Success criteria and interpretation

The central result succeeds only if the adaptive method improves the editability–preservation trade-off against a **tuned global baseline at equal total NFE**, with the benefit concentrated on shape-changing edits and no unacceptable background degradation.

The attention-supported controller adds a separate hypothesis: the same externally derived field should transfer across hosts because its support comes from source/target spatial-to-text attention and its edit-demand magnitude comes from draft divergence—not from any host's native mask. Demonstrate this through ablations; do not claim it merely because the code runs on a second model.

If the second host fails, report that the reliability of draft-derived edit demand depends on the host/draft behavior. Do not weaken the primary StreamEdit story or spend unlimited compute trying to force portability.
