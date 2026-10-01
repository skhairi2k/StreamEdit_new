# Paper strategy — pushing an image edit through a video, per token

**Date:** 2026-09-29 · **Target:** CVPR 2027 (checked 2026-09-29 on cvpr.thecvf.com: registration **Tue Nov 10**, paper **Mon Nov 16**, supplementary **Mon Nov 23**, all AoE) · **Status:** pre-gate; nothing below is a result unless marked *evidence*.

---

## 1. Story

**Premise.** Image editors (Qwen-Image-Edit, FLUX Kontext) are ahead of video editors. Editing frame 0 is largely solved. The open problem is **how hard to push that edit through the rest of the video**. That is an edit-strength problem.
*Measured, not assumed: T2 compares frame 0 from video editors with the image editor's anchor.*

**Why the current answer fails.** Training-free propagators set edit strength with **one global knob**: StreamEdit's ρ (Q/K blend, source weight $t^{\rho}$), FlowEdit/FlowDirector's $n_{\max}$, AnyV2V's injection thresholds $\tau$, ContextFlow's enrichment cut-off $\tau$ (first 50% of steps, top-k layers). One strength cannot serve a **shape-changing edit**:

- tokens that must take a new outline need early release, at high noise where structure forms;
- the rest must stay anchored.

A strong global anchor blocks the new shape; a weak one damages the background.

**Contribution.** A **per-token release time, measured from where a cheap draft actually diverges from the source**, on a few-step streaming video model. This replaces one global time, or one time per mask region.

- Varying *where* or *when* is not new. Differential Diffusion sets a continuous per-pixel release time, but from a user-given map. Follow-Your-Shape builds a mask from measured velocity divergence, inside a global schedule, on images. Uni-Edit masks the velocity by region under one global delay. AnyV2V gates injection by global timestep thresholds. Stable Flow chooses layers. VM-Edit (preprint) uses two start times, foreground and background, from GT masks.
- Our claim is the combination: the release time is **set automatically per token from measured draft–source divergence**, it releases an **attention anchor** (Q/K blend), and it runs in **video on a few-step streaming model**. (Context, not a pillar of the claim: Follow-Your-Shape's appendix, "Limitations and Future Work → Extending to Video Editing", Fig. 8 of arXiv v2, reports qualitatively that on Wan2.1 its divergence map "often fluctuate[s] across frames". The evidence is a single-timestep visualisation, with no quantitative result.)


|                                           | Content                                                                                                                                                                                                                                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| C1 · Method                               | Per-token release $t^{\rho_j}$ in StreamEdit (Self-Forcing, LongLive), with $\rho_j$ from draft–source divergence. Optional: $\mathbb{1}[t\le\tau_j]V^{\Delta}$ in FlowDirector (= per-token $n_{\max}$)                                                                                               |
| C2 · Evaluation (*reported, not claimed*) | FiVE's bg metrics penalise valid deformation as much as far damage (synthetic check, motivation only) → an anchor-zone bg metric reported next to FiVE-official, with stated limitations; FiVE-Shape subset (anchor spill). It becomes a contribution again only if the T20 human study agrees with it |
| C3 · Evidence                             | At equal NFE, beats: the *tuned* global-strength curve, mask-region timing (R26), and the zero-draft anchor-only field. Gain concentrated on shape-changing edits; holds with two image editors                                                                                                        |


**Evidence so far** (*evidence*, R35 vs R36, full FiVE; figures `evaluation/figures/r35_*_vs_lpips.pdf`; numbers recomputed 2026-09-29 with the §3 definition of "matched" from the per-clip CSVs, plain per-clip mean):

- `lpips_first2` beats the ρ curve on full FiVE-Acc in all 4 main edit types.
  - FiVE-official LPIPS-matched: +0.019 / +0.028 / +0.008 / +0.023 (types 1–4).
  - FiVE-official PSNR-matched: +0.019 / +0.030 / +0.006 / +0.024.
  - Full bench: +0.021 under both.
  - CLIP-D is also positive in all four (LPIPS-matched; full bench +0.003 under both).
  - No significance test yet: the type 3 gain (+0.006/+0.008) may be noise.
  - Type 5 (n=9) +0.15 and type 6 (n=10) +0.000: too small to read.
- The gain is largest on type 2 (non-rigid) and smallest on type 3 (colour).
- Small effect; 2× NFE; no comparison with R26 or the anchor-only field on the full bench yet.
- **Committed arm:** `lpips_first2`**.** It routes on LPIPS and is scored on LPIPS. The gain holds when matched on FiVE-official PSNR (above). Still to check: SSIM-matched, and the anchor-zone metric.

---



## 2. Tables

**Rule:** competitors must be peer-reviewed. Image editors and backbones are tools, not competitors.

**Table 1 — matched first frames (main).** Every method gets the same fixed Qwen-Image-Edit-2511 frame 0.


| Group                          | Methods                                                                                                                                                                                  | Note                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Training-free                  | **Ours** (SF, LL) · StreamEdit§ (ECCV 2026; SF, LL) at ρ=2 *and* best global ρ · ContextFlow (AAAI 2026; Wan2.1-I2V-14B, 50 steps — report NFE/backbone next to it) · AnyV2V (TMLR 2024) | StreamEdit§ rows rerun with *our* anchors (the paper says only "Qwen-Image-Edit") |
| Per-video optimisation, marked | LoRA-Edit (ICLR 2026) · I2VEdit (SIGGRAPH Asia 2024; trains a motion LoRA per source video)                                                                                              | FiVE-Shape / subset only (tuning cost per video)                                  |
| Trained, shaded reference      | FreeProp / FFP-300K (CVPR 2026) · VACE (ICCV 2025, Wan2.1-VACE-1.3B)                                                                                                                     | "Close at a fraction of the cost" is a result; no need to beat them               |


Wording: the "only method" claim is **dropped**. ContextFlow (AAAI 2026) is training-free first-frame propagation on a Wan2.1 DiT. The narrower claim "only one on a few-step streaming model" was not searched specifically: check before using. AnyV2V and Videoshop (ECCV 2024) are training-free but UNet-era. FREE-Edit (LTX/Wan) is a preprint: cited, not a competitor. Frame Guidance (ICLR 2026) steers generation, not editing, so it is not a competitor (§6).

**Table 2 — text to video, end to end.** Rows that use an image editor are marked ✎.


| Rows                                                                                                  | Source of numbers                                                                    |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Ours ✎** · StreamEdit§ ✎ at ρ=2 (image editor + naive propagation) · StreamEdit§ ✎ at best global ρ | own runs                                                                             |
| StreamEdit (text) · FlowDirector (CVPR 2026)                                                          | own runs                                                                             |
| TokenFlow · DMT · VidToMe · VideoGrain · Pyramid-Edit · Wan-Edit · UniEdit-Flow (ICLR 2026)           | reuse StreamEdit's FiVE table if the metric code matches; check 420 vs our 419 pairs |
| *(optional)* FlowDirector + per-token $n_{\max}$                                                      | own runs — the same rule on a text-only method                                       |


Whatever we gain over "StreamEdit§ ✎ at best global ρ" is propagation, not better editing.

**Table 3 — equal NFE (the core evidence; FiVE-Shape / Control / all).** Budget $B = k + n$ (draft + final), counting both StreamEdit branches. Wall-clock is also reported. No arm may use FiVE's ground-truth masks (privileged).


| Arm                                                                                     | Spends $B$ as                                                                                                                                                         |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Global ρ, more steps                                                                    | one pass, $B$ steps                                                                                                                                                   |
| Global ρ chosen per clip                                                                | draft $k$ → one ρ per clip → final $n$                                                                                                                                |
| **Anchor-only field** (no draft)                                                        | divergence between anchor and source frame 0 → field on frame 0, extended to later frames (broadcast, or warped with the source's optical flow) → one pass, $B$ steps |
| Mask-region timing (R26: two ρ values inside/outside StreamEdit's grounding-mask union) | the mask needs a render → draft $k$ → mask → final $n$                                                                                                                |
| **Ours**                                                                                | draft $k$ → per-token divergence field → final $n$                                                                                                                    |


The anchor-only field is the draft's free alternative; reviewers will ask why a draft is paid for when the anchor is free. If it matches Ours, it *becomes* the method (cheaper). That is a good outcome, not a failure.
Precedent: FREE-Edit (preprint) uses a binary, all-timestep version of this field (pixel-difference threshold + optical-flow warp → Q/K weight). Ours differs by continuous per-token *timing*.

**Ablations:**

- draft length $k \in 4, 7, 15$;
- divergence network (LPIPS / DINO) and stage-1 regime (unblended / first2);
- anchor-only extension (broadcast vs flow-warped);
- hard vs soft gate (FlowDirector, optional);
- field temporal stability: frame-to-frame variation of $\rho_j$ (and its effect on warp error).

**Fairness controls:**

1. Release the fixed first frames (and the converted prompts for EditVerseBench).
2. Editor swap: Qwen-Image-Edit-2511 vs FLUX.1 Kontext. The gain over StreamEdit§ at best global ρ must hold with both.

---



## 3. Benchmarks, metrics, sanity check


| Benchmark                                       | Role                                                                                                                                                                   | Caveats                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FiVE-Bench** (419 pairs) + fixed Qwen anchors | Primary; carries the claim; comparable to StreamEdit's § rows. Subsets: **FiVE-Shape** (anchor spill above a threshold fixed in advance), **FiVE-Control** (types 3–4) | Types 5 (9 clips) / 6 (10 clips) too small alone. **Removal (type 6) is excluded from FiVE-Shape** (FiVE-Control is types 3–4 only), decided before T1. It stays in the full bench (419, comparable with StreamEdit's table) with its own per-type row, and is stated as a limitation. Why (*evidence*, n=10): |


- FiVE-Acc is 0.05 for every global ρ in 2–20 and for all four R35 arms.
- Only ρ=50 moves it (0.20), at a cost of bg PSNR 28.7 → 22.5 dB.
- So no timing strategy we test separates on removal, and including it would only dilute the Shape subset.
- Removal changes pixels *inside* the source mask, while Shape is defined by spill *outside* it.
- T1 reports how many type-6 clips would have passed the spill threshold, so the effect of the exclusion is visible. |
| **EditVerseBench-FFP** (125 videos, filtered by FFP-300K from EditVerse, ICLR 2026) | Secondary. Most recent top-venue protocol for first-frame propagation; non-square videos, 20 edit types; comparable to FreeProp's published numbers | Instructions → source/target prompts must be converted (VLM) and released · Self-Forcing/LongLive trained at 832×480: smoke-test portrait/square · no source masks (no bg metrics / anchor spill without SAM2) · use their metrics unchanged, and only if their eval code/VLM prompt is released |

**Anchor spill** (inputs only): fraction of pixels outside the slightly dilated source mask that differ between source frame 0 and the anchor.

**Metrics — report all of them.**

- **The claim is sign-consistency within each axis.** Against the tuned baseline, alignment must improve *at matched preservation*, and preservation *at matched alignment*. Reading axes separately would count a move along the trade-off curve as a gain.
- **What "matched" means.** The baseline is the tuned global-ρ curve: one point per ρ, same host, same subset. At our arm's value of the matching metric, the baseline's value is **linearly interpolated between the two neighbouring ρ points**. The gain is ours minus the interpolated value. If the curve is not monotonic, take the baseline value most favourable to the baseline. If our point is outside the curve's range, report "outside curve" with no gain (**no extrapolation**).
  - *Alignment at matched preservation:* match **separately** on **FiVE-official LPIPS** and on **FiVE-official PSNR**. The gain must hold **under both**. LPIPS alone is circular, because `lpips_first2` routes on LPIPS.
  - *Preservation at matched alignment:* match on **FiVE-Acc (full)**.
- **Primary vs secondary metrics (fixed now).** The claim requires sign agreement on the **primary** metrics only: alignment = FiVE-Acc (full), CLIP-D; preservation = FiVE-official PSNR/SSIM/LPIPS.
  - **Secondary:** CLIP-T (already considered weak), EditReward (per frame, blind to time), PSC-Δ (our variant, not externally validated), EditVerseBench VLM score, anchor-zone bg metric (ours, not human-validated).
  - If a **secondary** metric disagrees, it stays in the main table and gets a one-line explanation tied to its known blind spot. The claim is then worded "on the primary metrics".
  - If a **primary** metric disagrees, there is **no claim** on that axis for that subset.
  - On FiVE-Shape, FiVE-official and the anchor-zone metric may disagree. Report it, and point to synthetic cases (a)/(b), where FiVE-official cannot tell a valid deformation from far damage.
- **Losses are reported, not hidden.**
- **FiVE-official bg metrics and FiVE-Acc are always in the main table**, for comparability with StreamEdit.


| Axis           | Metrics                                                                                                                                                                                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Edit alignment | FiVE-Acc (full) · CLIP-T · CLIP-D · EditReward (ICLR 2026; image reward model → per-frame mean, blind to time) · **PSC-Δ**: VideoCLIP-XL2 similarity to target phrase minus source phrase (*our variant* of IVEBench's PSC, which is target-only) · EditVerseBench: their VLM score |
| Preservation   | FiVE-official PSNR / SSIM / LPIPS · anchor-zone PSNR / SSIM / LPIPS (secondary; see below)                                                                                                                                                                                          |
| Temporal       | warp error · CLIP-F · DINO temporal consistency                                                                                                                                                                                                                                     |


**Anchor-zone bg metric** (replaces R37's distance weighting).

- *Why distance weighting was dropped:* its synthetic result is fixed by its own definition, and it cannot tell valid deformation (a) from near leakage (c).
- *Definition:* the allowed zone is $Z_0 = M_{src}(0) \cup \text{anchor} \neq \text{source frame 0}$, i.e. the source mask plus T1's anchor spill. It is carried to frame $t$ with the **source** optical flow (T6's warp), plus a small fixed dilation. The background is its complement.
- *Same region for every method:* the zone depends only on the inputs.
- *Why not grounded target masks:* R28 grounded the target object on each method's own frames, and that is confounded. Spearman(per-arm background area, `lpips_unedit_union`) = +0.972, and the per-arm "wins" disappear under a fixed union.
- *Limitations, stated:*
  - Leakage already present in the anchor is forgiven. This is fair here, since all methods get the same anchor.
  - The flow warp is approximate where the new shape moves unlike the source object (type 2).
  - It is not validated against humans unless T20 runs.

Aggregation: plain per-clip mean (never `_avg.csv`); sign per edit type; paired bootstrap resampled over source videos if a gain is borderline.

**Synthetic check (motivation only, not validation).** Perturb real FiVE source videos in known ways, with the same number of changed pixels in each case. It shows that FiVE-official cannot tell (a) from (b). It does **not** validate any replacement metric: synthetic cases are built to match that metric's definition, so only T20 can validate one.


| Case                  | Construction                                                              | FiVE-official   |
| --------------------- | ------------------------------------------------------------------------- | --------------- |
| (a) Valid deformation | source object scaled up / dilated and pasted back: changes only next to S | penalised       |
| (b) Far damage        | same pixel budget of colour shift / blur far from S                       | penalised ≈ (a) |
| (c) Near leakage      | colour bleed in a ring around S (invalid, but near)                       | penalised       |


For editability metrics, three cases: unedited source (floor) · anchor on frame 0 only, source elsewhere (partial) · anchor repeated as a static video (high alignment, no motion). This shows which alignment metrics are blind to time or motion.

---



## 4. Tasks, ranked by priority, then effort

Effort: S ≤ 1 day · M 2–4 days · L ≥ 1 week. Unless stated, runs are on FiVE-Shape + Control first, full bench for final numbers.

### Calendar


| Phase                | Dates                                | Content                                                                                                                                                   |
| -------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 — gate            | Wed Sep 30 → **Tue Oct 13** (Gate A) | only what Gate A needs: T1, T2, T3, T4, T6, T7                                                                                                            |
| P1 — build the paper | Wed Oct 14 → Thu Nov 5               | T5, T8, T9, T10 (moved from P0), T11, T12, T14, T15, T16; **T13, T17 first to cut** (decide Wed Oct 28)                                                   |
| W — writing          | Fri Nov 6 → **Mon Nov 16**           | ~10 days; title/abstract registered **Tue Nov 10**; **tables frozen Wed Nov 11** (runs still going then are dropped or go to the supplement); T22 figures |
| S — supplement       | Nov 17 → **Mon Nov 23**              | videos, extra per-type tables, ablations that missed the freeze; no new claims                                                                            |




### P0 — gate (Sep 30 → Oct 13, Self-Forcing VP only)


| ID  | Task                                                                                                                                                                                                                                                                                                                | Effort  | Output                  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------- |
| T0  | ✅ done 2026-09-29 (see §6). `/litsearch`: training-free first-frame propagation on DiT video models (backs the "only method" sentence); per-token/region-wise $n_{\max}$ or edit strength; check whether Uni-Edit's delayed injection varies per region; shape-changing video editing; boundary-tolerant bg metrics | S       | novelty, wording        |
| T1  | Anchor spill for 419 clips → fix threshold → FiVE-Shape / FiVE-Control manifests (type 6 removal excluded from Shape, pre-registered)                                                                                                                                                                               | S (CPU) | subsets                 |
| T2  | **Premise measurement:** frame 0 of StreamEdit text-mode renders (R1 baseline, full bench, already on disk) vs the Qwen anchor — EditReward, FiVE-Acc-style VLM questions and CLIP-D on single frames; add FlowDirector's frame 0 when T15 runs                                                                     | S       | premise holds?          |
| T3  | NFE accounting (both branches) + wall-clock of draft / divergence / final                                                                                                                                                                                                                                           | S       | cost table              |
| T4  | Baseline step scaling: global ρ at 15 vs 19 / 22 / 30 steps                                                                                                                                                                                                                                                         | S–M     | equal-NFE reference     |
| T6  | **Anchor-only field** arm (anchor vs source frame 0; broadcast and flow-warped), equal NFE                                                                                                                                                                                                                          | S–M     | is the draft needed?    |
| T7  | Mask-region timing: R26 at equal NFE (3–4 values of b). VM-Edit (preprint) is the video precedent for region timing → per-token vs per-region is the key test                                                                                                                                                       | M       | per-token vs per-region |


**Gate A** (Tue Oct 13), on FiVE-Shape at equal NFE:

1. Ours (or the anchor-only field) > best global ρ.
2. It > R26.
3. Gain on Shape clearly > gain on Control.
4. T2 shows a real frame-0 gap.

If (1) or (2) fails, there is no CVPR method paper: C2 is now a reported metric, not a contribution. If (4) fails, drop the premise and lead with the shape-change failure instead.

### P1 — after Gate A (Oct 14 → Nov 5)


| ID    | Task                                                                                                                                                                                                                                                                                                                                                                      | Effort |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| T8    | Per-clip ρ choice from the draft at equal NFE                                                                                                                                                                                                                                                                                                                             | M      |
| T5    | Ablation: draft length $k \in 4, 7, 15$ (field agreement + render)                                                                                                                                                                                                                                                                                                        | M      |
| T11   | FLUX Kontext anchors for 419 clips → rerun Ours + StreamEdit§ at best global ρ (editor swap) · release both anchor sets                                                                                                                                                                                                                                                   | S–M    |
| T14   | Table 1 externals with fixed anchors: VACE-1.3B, AnyV2V (smoke 1 clip first: 16-frame native), ContextFlow (smoke 1 clip first; 14B at 50 steps is expensive on the full bench; sweep its global τ ∈ {0.3, 0.5, 0.7} on Shape + Control, full bench at best τ; check whether it needs object masks → if FiVE GT masks, mark privileged), FreeProp (if weights are public) | M      |
| T9    | Anchor-zone bg metric (§3): $Z_0$ from T1's spill + T6's flow warp; `evaluate.py` reads it via R28's precomputed-mask path; + ρ × edit-type gap + check of type-5 masks. Rescope the R37 plan accordingly                                                                                                                                                                 | S–M    |
| T10   | Synthetic check (§3, motivation only): FiVE-official (a) ≈ (b); editability cases                                                                                                                                                                                                                                                                                         | S–M    |
| T16   | Metrics on all arms: EditReward (per frame), PSC-Δ (VideoCLIP-XL2), warp error                                                                                                                                                                                                                                                                                            | M      |
| T12   | Self-Forcing text mode: ρ curve + Ours + equal-NFE arms (Table 2 rows)                                                                                                                                                                                                                                                                                                    | S–M    |
| T15   | Table 2 externals: FlowDirector; reuse StreamEdit's reported baselines after checking the metric code and the 420/419 pair difference                                                                                                                                                                                                                                     | M      |
| T13 ✂ | LongLive port (`rho_frames`, vectorised blend, ρ for sink tokens) → bit-parity gate → VP + text runs                                                                                                                                                                                                                                                                      | M–L    |
| T17 ✂ | EditVerseBench-FFP: get the 125-video list + protocol, convert prompts, aspect-ratio smoke test, run Table 1 rows                                                                                                                                                                                                                                                         | M      |


✂ = first to cut. Decide on **Wed Oct 28**: if T8, T5, T11 and T14 are not done, drop both. Without T13 the paper uses one host (Self-Forcing), so remove "(SF, LL)" claims. Without T17, FiVE is the only benchmark.

### Writing phase (Nov 6 → Nov 16)


| ID  | Task                                                                                                  | Effort |
| --- | ----------------------------------------------------------------------------------------------------- | ------ |
| T22 | Figures: FiVE penalising a correct outline; ρ fields; per-type trade-off curves; editor-swap examples | S      |




### P2 — not scheduled (only if something above finishes early; otherwise cut)


| ID  | Task                                                                                                                                     | Effort |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| T18 | LoRA-Edit and I2VEdit on FiVE-Shape (per-video tuning; LoRA-Edit takes FiVE masks as input — mark it)                                    | L      |
| T19 | FlowDirector + per-token $n_{\max}$ (optional row; upgrades the claim to "across mechanisms")                                            | M      |
| T20 | 2AFC metric validation: ~100 pairs where FiVE-official and the anchor-zone metric disagree. The only way C2 becomes a contribution again | M      |
| T21 | Method-preference user study (optional; some reviewers expect one, not required)                                                         | M      |


**Order:** T1 ∥ T2 ∥ T3 → T4 → T6 ∥ T7 → **Gate A (Oct 13)** → T8 ∥ T5 ∥ T11 → T14 ∥ T9 ∥ T10 → T16 → T12 ∥ T15 → **cut decision (Oct 28)** → T13 / T17 or nothing → **freeze (Nov 11)** → writing.

---



## 5. Risks

- **Small effect** (+0.02 FiVE-Acc on full FiVE): needs a clearly larger gain on FiVE-Shape.
- **Equal NFE may erase the gain.** T4/T5 decide this early.
- **The anchor-only field may match the draft.** It then becomes the method, and the story shifts from "measure the draft" to "measure the anchor".
- **Tuned vs default baseline:** claims must be against the best global ρ, not ρ=2 (ρ=2 → ρ=50 alone moves full FiVE-Acc 0.697 → 0.811).
- **Novelty is a combination** (T0). Differential Diffusion (continuous per-pixel time), Follow-Your-Shape (divergence-derived mask) and, as concurrent preprints, Constrained Edit Fields (proposal → continuous strength field) and FREE-Edit (anchor-derived Q/K weight) each cover one part. Gate A (> R26 and > anchor-only at equal NFE) carries the paper.
- **Removal:** the StreamEdit family (and so Ours) scores FiVE-Acc ≈ 0 on type 6; ContextFlow does deletion and is expected to win that row. Report it in the per-type table and state removal as a limitation. Do not drop the row.
- **Field flicker across frames** (Follow-Your-Shape observed it qualitatively for its divergence map on Wan2.1, appendix Fig. 8): measure the field's temporal stability.
- **Metric circularity** (LPIPS-routed, LPIPS-scored): gains are also matched on FiVE-official PSNR (they hold on full FiVE, §1). **Anchor-zone metric limits**: it forgives leakage already in the anchor, and its flow warp is approximate for non-rigid shapes. Both are stated; it stays secondary.
- **Many metrics:** all are reported; the claim is sign-consistency per axis at matched other axis; losses are shown.
- **EditVerseBench portability** (prompts, aspect ratios, protocol availability) could cost more than expected: it stays secondary.
- **Compute:** a full-bench curve is ~85 L40S GPU-h per host (8 points). Use 5 points on Shape + Control, full bench for final tables only.

---



## 6. References

arXiv IDs and venues checked 2026-09-29. Qwen-Image and FLUX.1 Kontext are technical reports (arXiv only).

**Base method and backbones**

- **StreamEdit** (ECCV 2026) — [arXiv:2605.21466](https://arxiv.org/abs/2605.21466). Training-free editing as source-conditioned generation on streaming few-step video models, with Q/K blending controlled by one global ρ and optional first-frame visual prompting (§). Our base and main baseline: we make its global ρ per-token.
- **Self Forcing** (NeurIPS 2025) — [arXiv:2506.08009](https://arxiv.org/abs/2506.08009). Autoregressive few-step video diffusion on Wan2.1-1.3B, trained on its own rollouts with a KV cache. Backbone of StreamEdit (SF) and of our main host.
- **LongLive** (ICLR 2026) — [arXiv:2509.22622](https://arxiv.org/abs/2509.22622). Real-time long video generation: a Wan-1.3B model fine-tuned with a frame-level attention sink and KV re-cache. Our second backbone, StreamEdit (LL).

**Image editors** (produce the fixed first frames)

- **Qwen-Image** (arXiv 2025) — [arXiv:2508.02324](https://arxiv.org/abs/2508.02324). Technical report of the Qwen-Image family, which includes Qwen-Image-Edit. Our main anchors come from the Qwen-Image-Edit-2511 checkpoint.
- **FLUX.1 Kontext** (arXiv 2025) — [arXiv:2506.15742](https://arxiv.org/abs/2506.15742). Flow-matching model for in-context image generation and editing. The second editor for the editor-swap control.

**First-frame propagation — Table 1**

- **AnyV2V** (TMLR 2024) — [arXiv:2403.14468](https://arxiv.org/abs/2403.14468). Training-free: edit frame 0 with any image editor, then regenerate with an image-to-video model plus source feature injection gated by global timestep thresholds τ. Training-free competitor on an older UNet backbone (I2VGen-XL, 16 frames).
- **Videoshop** (ECCV 2024) — [arXiv:2403.14617](https://arxiv.org/abs/2403.14617). Training-free propagation of a first-frame edit through noise-extrapolated inversion of an image-to-video model. The other UNet-era training-free propagator; cited, not run.
- **I2VEdit** (SIGGRAPH Asia 2024) — [arXiv:2405.16537](https://arxiv.org/abs/2405.16537). First-frame-guided editing that trains a motion LoRA on each source video, then refines appearance and motion by attention matching whose strength adapts to the amount of structural change. Per-video-optimisation reference, and prior art for adapting strength to shape change (per video, not per token).
- **LoRA-Edit** (ICLR 2026) — [arXiv:2506.10082](https://arxiv.org/abs/2506.10082). First-frame-guided editing by per-video, mask-aware LoRA fine-tuning of an image-to-video model. Per-video-optimisation reference, run on a subset only.
- **FFP-300K / FreeProp** (CVPR 2026) — [arXiv:2601.01720](https://arxiv.org/abs/2601.01720). A 300K-pair first-frame propagation dataset, plus FreeProp, a LoRA fine-tune of Fun-Control (Wan 2.1), evaluated on a 125-video filter of EditVerseBench with Qwen-Edit first frames. Trained reference (shaded) and source of our second benchmark.
- **ContextFlow** (AAAI 2026) — [arXiv:2509.17818](https://arxiv.org/abs/2509.17818). Training-free first-frame propagation on Wan2.1-I2V-14B (DiT, 50 RF-Solver steps): K/V from reconstruction and editing paths concatenated in self-attention, only in top-k "vital layers" and the first 50% of steps (τ=0.5). Compares with AnyV2V, AnyV2V-DiT, I2VEdit, VACE. Training-free competitor on a modern DiT; refutes the old "only method" sentence.
- **VACE** (ICCV 2025) — [arXiv:2503.07598](https://arxiv.org/abs/2503.07598). All-in-one trained video creation and editing model on Wan2.1, including reference- and mask-conditioned editing (1.3B checkpoint public). Trained reference at our backbone size.

**Text-driven editing — Table 2**

- **FlowEdit** (ICCV 2025) — [arXiv:2412.08629](https://arxiv.org/abs/2412.08629). Inversion-free editing: an ODE driven by the target–source velocity difference, with edit strength set by the start step $n_{\max}$. Defines the $n_{\max}$ knob we make per-token (optional host via FlowDirector).
- **FlowDirector** (CVPR 2026) — [arXiv:2506.05046](https://arxiv.org/abs/2506.05046). FlowEdit-style video editing on Wan2.1-1.3B, with an attention-derived spatial mask on the edit velocity and averaged-guidance steering. Text-driven competitor; per-token $n_{\max}$ can be added with one gated line.
- **UniEdit-Flow** (ICLR 2026) — [arXiv:2504.13109](https://arxiv.org/abs/2504.13109). Predictor-corrector inversion and Uni-Edit: one global delay α plus a velocity-difference mask for region-aware velocity fusion. Delay is not per region. Text-driven baseline reported on FiVE by StreamEdit.
- **TokenFlow** (ICLR 2024) — [arXiv:2307.10373](https://arxiv.org/abs/2307.10373). Training-free consistent video editing by propagating diffusion features along inter-frame correspondences. Diffusion-era baseline; numbers reused from StreamEdit's FiVE table.
- **DMT — Space-Time Diffusion Features** (CVPR 2024) — [arXiv:2311.17009](https://arxiv.org/abs/2311.17009). Zero-shot motion transfer guided by a space-time feature loss from a text-to-video model, including large shape changes. Diffusion-era baseline, reused numbers; relevant to shape change.
- **VidToMe** (CVPR 2024) — [arXiv:2312.10656](https://arxiv.org/abs/2312.10656). Zero-shot video editing that merges self-attention tokens across frames for temporal consistency. Diffusion-era baseline, reused numbers.
- **VideoGrain** (ICLR 2025) — [arXiv:2502.17258](https://arxiv.org/abs/2502.17258). Zero-shot multi-grained (class, instance, part) editing by modulating space-time cross- and self-attention. Diffusion-era baseline, reused numbers.

**Related, not competitors**

- **Frame Guidance** (ICLR 2026) — [arXiv:2506.07177](https://arxiv.org/abs/2506.07177). Training-free steering of video *generation* from a few frame-level signals (keyframes, style image, sketch, depth, colour blocks, looping): at each step it decodes a predicted clean frame, computes a loss against the target, and updates the latent by gradient. It generates rather than edits a source video, so it is not a competitor; cite it as related training-free frame-level control.
- **Differential Diffusion** (Computer Graphics Forum 2025 — verify author list/venue; Wiley page not reachable) — [arXiv:2306.00950](https://arxiv.org/abs/2306.00950). Training-free per-pixel edit strength from a user-given change map: each region is released at its own timestep. Prior art for continuous per-pixel release time; ours sets it automatically from measured divergence.
- **Follow-Your-Shape** (ICLR 2026) — [arXiv:2508.08134](https://arxiv.org/abs/2508.08134). Shape-aware image editing on FLUX: a Trajectory Divergence Map (token-wise ‖v_tgt − v_src‖, fused over steps, thresholded) gates KV injection inside a global 3-stage schedule; introduces ReShapeBench. Prior art for divergence-derived spatial control. Its appendix ("Limitations and Future Work → Extending to Video Editing", Fig. 8, arXiv v2) notes qualitatively that the map fluctuates across frames on Wan2.1; no quantitative video results.
- **VideoSwap** (CVPR 2024) — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Gu_VideoSwap_Customized_Video_Subject_Swapping_with_Interactive_Semantic_Point_Correspondence_CVPR_2024_paper.html). Shape-changing subject swap via sparse semantic-point correspondences (interactive, customised). Motivates that dense-correspondence editing fails on shape change.
- **StableV2V** — [arXiv:2411.11045](https://arxiv.org/abs/2411.11045) (journal version on IEEE Xplore; venue unconfirmed). Shape-consistent first-frame propagation with simulated flow/depth (trained modules); introduces DAVIS-Edit.
- **Stable Flow** (CVPR 2025) — [arXiv:2411.14430](https://arxiv.org/abs/2411.14430). Training-free image editing in flow DiTs that injects source features only into automatically found "vital layers". Context for choosing *where in the network* to inject; our knob is *when, per token*.

**Benchmarks and metrics**

- **FiVE-Bench** (ICCV 2025) — [arXiv:2503.13684](https://arxiv.org/abs/2503.13684). 100 videos, 420 object-level prompt pairs in 6 edit types, with masks, the FiVE-Acc VLM metric, and the FlowEdit adaptations Pyramid-Edit and Wan-Edit. Primary benchmark; its mask-based bg metrics are what our distance-weighted metric corrects.
- **EditVerse / EditVerseBench** (ICLR 2026) — [arXiv:2509.20360](https://arxiv.org/abs/2509.20360). Unified in-context image/video editing model plus EditVerseBench, an instruction-based video editing benchmark. Source of the 125-video FFP subset, our secondary benchmark.
- **IVEBench** (ICLR 2026) — [arXiv:2510.11647](https://arxiv.org/abs/2510.11647). Instruction-guided video editing benchmark (600 videos, 8 task categories) with quality, instruction-compliance and fidelity metrics, including PSC (VideoCLIP-XL2 video–target-phrase similarity). Our PSC-Δ is a source-relative variant of its PSC.
- **VideoCLIP-XL** (EMNLP 2024) — [arXiv:2410.00741](https://arxiv.org/abs/2410.00741). Video CLIP trained for long-description understanding; IVEBench uses its v2 checkpoint (VideoCLIP-XL2). The embedding model behind PSC-Δ.
- **EditReward** (ICLR 2026) — [arXiv:2509.26346](https://arxiv.org/abs/2509.26346). Human-aligned reward model for instruction-guided *image* editing, trained on 200K+ expert preference pairs. Alignment metric, applied per frame (blind to time), and used on single frames for the premise measurement (T2).



**Concurrent preprints (cited, not competitors)**

- **FREE-Edit** — [arXiv:2603.01164](https://arxiv.org/abs/2603.01164). Training-free first-frame propagation on LTX-Video-2B / Wan2.1-14B: Q/K blend with a binary token weight from the anchor–source pixel difference, warped by optical flow, at all timesteps. Binary, untimed version of our anchor-only field.
- **Constrained Edit Fields** — [arXiv:2609.33735](https://arxiv.org/abs/2609.33735). Image editing (SD3.5/FLUX): a full FlowEdit proposal, localised with CLIPSeg, gives a continuous per-location strength field on the velocity. Same draft → field structure; strength rather than time, semantic rather than divergence, images only.
- **NRVBench / VM-Edit** — [arXiv:2601.18340](https://arxiv.org/abs/2601.18340). Non-rigid video editing benchmark (180 videos); VM-Edit baseline uses two start timesteps (foreground/background) from GT masks. Video precedent for region timing.
- **DynaEdit** — [arXiv:2603.17989](https://arxiv.org/abs/2603.17989). Training-free text-driven editing of actions and dynamics on T2V flow models.

