---
name: R10 — Claim-4 pre-test, head-gated persistent anchor injection
overview: >-
  Route the appearance anchor (the Qwen-edited first frame) to head subsets
  chosen by the R19 taxonomy, and ask whether head identity decides where an
  injected appearance takes effect. Six arms — none / all / spatial / temporal
  and two count-matched random controls — on 20 FiVE clips, all sharing one
  base (blending off, background source-KV on). Inference is COMPLETE (job
  906099, 120/120). What is being rebuilt is the evaluation: the original
  metric set was whole-frame and could not see any of the predictions, so R10
  is re-scored on four masked metrics, two of which need new code. R10 tests
  the APPEARANCE half only; motion is deliberately un-bridged here and its
  absolute level is not a finding.
task_id: R10
todos:
  - id: gates-builder
    content: Gates from R19 (equal-budget disjoint key sets) via
      evaluation/r19_build_head_gates.py — all(360) / spatial(117) /
      temporal(49) / rand_spatial(117) / rand_temporal(49) →
      evaluation/r19_head_gates.pt. Supersedes r10_build_head_gates.py, whose
      margins came from the defective 1560-vs-18 comparison.
    status: completed
  - id: pipeline-flags
    content: "blend_off (blender_rate:=1 ⇒ pure target Q/K, masked prev-blend
      skipped; bg source-KV concat KEPT) and vp_persistent threaded through
      rollout_inference→inference→shared_dict; layer_idx + vp attrs set on each
      self_attn at setup."
    status: completed
  - id: vp-bank
    content: EditCausalInferencePipeline.build_vp_bank — a third, never-evicted
      anchor_cache holding unroped anchor K/V, re-roped at every block with
      start_frame=last_chunk_start_frame so the relative offset is zero
      ("always present"). Contrast §4.5, which writes the anchor into
      kv_cache_trg and lets it fade as the window slides.
    status: completed
  - id: driver
    content: evaluation/r10_vp_arms.py — 20 cases × 6 arms, one pipe load,
      seed reset per arm, frames → {out_root}/{arm}/edit{T}/{video}/.
    status: completed
  - id: slurm-infer
    content: slurm_scripts/five_bench/r10_infer.sh (L40S, --mem=64G, single job).
    status: completed
  - id: anchor-sim
    content: >-
      NEW — evaluation/r10_anchor_sim.py. The two metrics that decide this
      experiment do not exist in the benchmark: calculate_clip_similarity is
      image↔TEXT only, and there is no image↔image similarity anywhere in
      metrics_calculator. Per (arm, clip, frame): crop BOTH the anchor PNG and
      the frame to the edit-region bbox (the same bbox construction MFS uses at
      metrics_calculator.py:336-338 — crop, do not zero-fill, or the black
      surround dominates the ViT CLS token), embed with the DINO ViT already
      loaded for structure_distance (dino_vitb8 via VitExtractor), take CLS
      cosine similarity. Emit per frame, then reduce to anchor_sim_f0
      (adherence), anchor_sim_slope_per100f (OLS fit ×100, reusing
      r10_metrics.slope) and retention = s_T/s_0. Must also score the R7 §4.5
      arm — the fading reference is the whole point of the slope. Keys on
      (video_name, editing_type_id). → evaluation/csv/r10_anchor_sim.csv
    status: pending
  - id: eval-clean
    content: >-
      NEW — rewrite slurm_scripts/five_bench/r10_eval.sh around the four
      surviving metrics and drop everything else. Stage 1 r10_metrics.py with
      --metrics motion_fidelity_score_edit_part
      clip_similarity_target_image_edit_part lpips_unedit_part (was: eight
      metrics, five of them whole-frame). Stage 2 r10_anchor_sim.py. Stage 3
      r10_make_grids.py. All six arms + --r7_root. The earlier CSVs under
      evaluation/csv/ are from the superseded metric set — archive, do not merge.
    status: pending
  - id: grids
    content: evaluation/r10_make_grids.py — rows source / appearance anchor /
      VP baseline (R7 §4.5) / none / all / spatial / temporal / rand_spatial /
      rand_temporal. The anchor is a single image and is excluded from the
      shared column count (_SINGLE_ROWS) or it collapses every grid to one column.
    status: completed
steps:
  - id: launch-infer
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r10_infer.sh
    sets_status: running
    status: completed
    notes: >-
      DONE — job 906099, 2026-07-21 21:00, 120/120 ok, 6 arms × 20 clips,
      gates r19_head_gates.pt (spatial 117 / temporal 49). Frames under
      /projects/dataggen/outputs/five_bench/r10_vp_arms/{arm}/edit{T}/{video}/.
      No re-run is needed for the metric rebuild — the pixels are fine, only
      the scoring was wrong.
  - id: build-anchor-sim
    type: manual
    command: write evaluation/r10_anchor_sim.py
    wait_for: launch-infer
    status: pending
    check_hint: >-
      Smoke-test on one clip before scheduling the panel. Sanity: the `all` arm
      must score anchor_sim_f0 well above `none` on 0057_dog — if it does not,
      the mask, the bbox crop or the anchor path is wrong and every downstream
      number is noise. Also confirm the R7 row is present, since the slope
      comparison is meaningless without it.
  - id: analyze
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r10_eval.sh
    wait_for: build-anchor-sim
    status: pending
    notes: >-
      Walltime was a risk with eight metrics over seven arms; with three
      benchmark metrics it is much smaller, but CoTracker still dominates.
      r10_metrics.py --arms takes a subset if it needs splitting, then
      --skip_eval to join and replot.
  - id: verdict
    type: manual
    wait_for: analyze
    status: pending
    check_hint: >-
      Read the four metrics against the prediction table in this plan. PRIMARY
      DECISION is spatial vs rand_spatial on anchor_sim_f0 — if a count-matched
      random set adheres as well as the labelled one, head identity is not doing
      the work and the routing program stops. Do NOT compare
      motion_fidelity_score_edit_part against the r7_vp row: R10 runs blend_off
      and R7 does not, so that gap is the ablation, not the gate. Record verdict
      + outcome bullet in daily.md and update the ideas.tex claim-4 status.
    sets_status: analyzed
isProject: true
---

# R10: Claim-4 pre-test — head-gated persistent anchor injection

## Context

**Decision gate, not a method demo.** The out-of-repo experiment showed
persistent anchor injection ⇒ frozen video, but not why. Two explanations
survive: **(A)** the anchor occupies the *temporal* heads specifically ⇒ routing
is the fix; **(B)** an always-present, motion-free key set freezes any head it
lands in ⇒ head identity is irrelevant. R10 varies exactly one thing — **where
the anchor lands** — and separates them.

**Inference is done; the evaluation was wrong.** Job 906099 rendered all 120
videos correctly. The original metric set then scored them whole-frame, and
whole-frame cannot see this experiment: every arm injects source background
keys, so the background is pinned to the source *by construction* and scores
near-perfectly regardless of the gate. Averaging that fixed region together with
the small region actually being manipulated destroys the signal — on the earlier
run `all` and `none` scored **0.775 vs 0.773** (p = 0.45) on whole-frame motion
fidelity while differing significantly on masked background measures. Every
prediction R10 makes is about the **edit region**, so every metric must be
masked to it.

**Scope limit (deliberate, and it governs how the numbers may be read).** The
full design routes *two* signals: the appearance anchor to spatial heads **and**
source attention-map transfer to temporal heads. R10 routes only the first. The
second needs the R11 patched-attention harness and is tested in R12.

More sharply: R10 runs `blend_off = True`, which disables **query blending** —
the mechanism the paper (§4.2) credits with *structure and motion preservation* —
and key blending with it. What remains of the source is background keys, current
chunk only, low-noise half only (`t < 0.5`). So R10's motion is expected to be
poor in absolute terms, and **poorer than the §4.5 VP baseline, which keeps
blending on**. That gap is a property of the ablation, not a finding about heads.
Edit-part motion is read **between R10 arms only**.

What R10 can establish is therefore narrow, and worth stating plainly: whether
routing the anchor by head identity improves **appearance adherence** and
**reduces vanishing**. Nothing about motion transfer.

## Execution steps

| # | id | type | what | status |
|---|----|------|------|--------|
| 1 | launch-infer | sbatch | 20 clips × 6 arms | **done** (906099, 120/120) |
| 2 | build-anchor-sim | manual | write `r10_anchor_sim.py`, smoke-test 1 clip | pending |
| 3 | analyze | sbatch | rewritten `r10_eval.sh` — 3 benchmark metrics + anchor sim + grids | pending |
| 4 | verdict | manual | prediction table below; GO/STOP | pending |

## Metrics — the complete set, and why nothing else is here

| metric | question | reads | new? |
|---|---|---|---|
| `anchor_sim_f0` | did the anchor's appearance land at all? | **primary arm discriminator** | **yes** |
| `anchor_sim_slope_per100f` | does it fade across the clip? | R10's persistent bank **vs the §4.5 baseline** — the one metric where that comparison is legitimate | **yes** |
| `motion_fidelity_score_edit_part` | did the subject freeze? | `all`/`temporal` lowest; **R10 arms only** | exists |
| `clip_similarity_target_image_edit_part` | did *an* edit happen? | guards "never adopted ⇒ nothing to lose" | exists |
| `lpips_unedit_part` | is the background untouched? | plumbing; should be flat across arms | exists |

**Dropped, whole-frame dilution:** `motion_fidelity_score`, `structure_distance`,
`clip_similarity_target_image`.

**Dropped, no readable direction:** `lpips_edit_part` and
`structure_distance_edit_part` score the output against the **source** inside the
one region we deliberately change, so a successful edit must score badly on them.

**Dropped, redundant:** `psnr/mse/ssim_unedit_part` — three restatements of
`lpips_unedit_part`.

**Held in reserve:** mean foreground track displacement (output ÷ source).
`motion_fidelity_score_edit_part` scores "frozen" and "wrong motion" alike, so if
it floors out across every arm it cannot separate `none` (wrong) from `all`
(frozen). `get_tracklets` (metrics_calculator.py:301) already returns per-point
trajectories and accepts a `segm_mask`, so this is ~30 lines of reuse. Build only
if needed.

## Predictions

| arm | `anchor_sim_f0` | `anchor_sim_slope` | `MF_edit_part` | `clip_edit_part` |
|---|---|---|---|---|
| `none` | **low** — no anchor, and no query blending either | n/a | poor | prompt-only edit |
| `all` | **high** | ≈ 0 | **lowest** — the static failure | high |
| `spatial` | **high** | ≈ 0 | **preserved** | high |
| `temporal` | **low** — never adheres; the anchor is absent from the appearance heads | n/a | low (freeze) | low |
| `rand_spatial` | ? — **this is the experiment** | | | |
| `rand_temporal` | ? | | | |
| `r7_vp` (reference) | high | **strongly negative** — it fades | *not comparable* | decays |

`temporal` is predicted to **never adhere**, not to adhere-then-lose-it. There is
nothing to fade from, so its slope is uninterpretable; the same applies to
`none`. A slope is meaningful only where `anchor_sim_f0` is already high.

## Decisions

| Decision | Choice |
|---|---|
| Gates | **R19**, not R9 — equal-budget disjoint key sets. `spatial` 117, `temporal` 49, from `evaluation/r19_head_gates.pt` (flat band). Wrong-arm rate on the labelled sets is **1.3%** across videos/steps/blocks vs **14.4%/26.7%** for the count-matched random sets, so the mislabelling channel is bounded *before* the outcome is read: a null here means the taxonomy does not control appearance, not that the labels were too noisy |
| Band shape | `flat` — SVG's own construction and the more stable of the two on wrong-arm. Disk agrees on head TYPE (zero spatial↔temporal confusions) but labels 88 temporal heads to flat's 49, 37 of them heads flat abstains on. If the `temporal` arm comes back null, "the gate was missing a third of its candidates" is a live alternative and the disk gate separates it — as a **follow-up**, pre-committed here so it is not a post-hoc second try |
| Base config | `blend_off = True` ⇒ blender_rate := 1 (pure target Q/K), masked prev-blend skipped. Background source-KV concat **stays on**, gated at `t < 0.5`, identical in all six arms. It is the source-anchoring channel; without it the target branch decouples entirely |
| `none` arm | Not an unconditioned baseline — it still receives source background KV on 7 of 15 steps. It is "everything except the anchor", and additionally lacks query blending, so it is the deficit reference. Also the plumbing check: broken (as opposed to merely un-edited) frames mean the flags are wrong and every other arm is uninterpretable |
| Arms are not size-matched | 117 vs 49, so a `spatial`-vs-`temporal` gap confounds head type with head count. `rand_spatial`/`rand_temporal` draw the same counts at seed 0 over all 360 heads. **If a random set of the same size adheres as well, the labels are not doing the work** |
| §4.5 reference (`r7_vp`) | The **fading** baseline the persistent bank is argued against. Comparable to R10 on `anchor_sim_slope` (both inject the same anchor image; only the persistence mechanism differs) and **not comparable on motion** (R7 runs with blending on, R10 with it off). Quote it as a reference on appearance, never as an ablation delta on motion. If `r7_vp` does *not* fade, the persistence premise itself is wrong and the predictions above are moot — report that rather than burying it |
| Anchors | `/projects/dataggen/outputs/five_bench/anchors/edit{T}/{video}.png` — the same files the §4.5/R7 arm reads, so the two differ only by injection mechanism, never by anchor image. Also the reference image for `anchor_sim` |
| Clips (20) | **e1 (1):** 0001_bus. **e2 (14):** 0028_kite-walk · 0040_tennis · 0057_dog · 0011_lucia · 0014_burnout · 0016_horsejump-high · 0068_planes-water · 0072_dog-agility · 0074_rhino · 0075_A_bicycle · 0076_A_rabbit · 0079_A_bus · 0090_A_deer · 0091_A_hawk. **e5 (3):** 0007_guitar-violin · 0069_car-turn · 0011_lucia. **e6 (2):** 0042_gym-ball · 0002_girl-dog. All verified to have a shared anchor, source video, bmasks and an R7 render |
| Duplicate video | 0011_lucia is scored **twice**, as e2 and as e5 — same source, two target prompts, therefore two clips. `case_id`s suffixed (`_e2`, `_e5`); frames at `{arm}/edit{T}/{video}`; scoring via `--tgt_layout edit_video`; every summary, curve and grid keys on `(video_name, editing_type_id)` |
| Short clips | 0072_dog-agility (21 frames) and 0068_planes-water (33) give ~11 and ~17 curve points at stride 2, so their **OLS slopes are the noisiest in the panel** — read the slope verdict panel-wide, never from these two |
| Statistics | Every clip appears under every arm ⇒ the arms are **paired**. Wilcoxon signed-rank on per-clip deltas, reporting median delta and per-clip win counts, not an unpaired test on arm means. Pre-specified primary: `anchor_sim_f0` for adherence, `motion_fidelity_score_edit_part` for freeze. 20 cases over **19 videos** — do not describe them as 20 independent clips |
| Seeding | `torch.manual_seed` + `np.random.seed` reset immediately before every arm, so arm differences are attributable to gating, not sampling |
| Sampler | `--step 15`, seed 0, `rollout_chunk_size = -1` (single window), ω = 4, 81-frame truncation (21 latent frames = one window) |
| Outputs (bulk) | `/projects/dataggen/outputs/five_bench/r10_vp_arms/{arm}/edit{T}/{video}/` + `_manifest.csv` — already written by 906099 |
| Outputs (repo) | `evaluation/csv/r10_arms.csv` (one row per arm×clip), `evaluation/csv/r10_per_frame.csv`, `evaluation/csv/r10_anchor_sim.csv`, `evaluation/figures/r10_fading_curves.pdf`, `evaluation/figures/r10_grids/{video}_edit{T}.png` |
| Complementary arm | **Rejected for R10** — a seventh arm (anchor→spatial *and* bg-KV→temporal) would test two changes at once, and gated bg-KV is only a proxy for the real Q^s,K^s→V^t map transfer, so a null there would not cleanly indict the method. Complementary design belongs to R12, after R11's harness |
| Out of scope | Source-side head routing (R12), mask-gated anchor (R13), scaled stability (R18), rollout long-video, and **anything about motion transfer** |

## Verdict rules

**GO** — `spatial` adheres (high `anchor_sim_f0`) *and* keeps edit-part motion,
while `all` adheres but freezes, **and** `spatial` beats `rand_spatial` on
adherence. Head identity decides where an injected appearance takes effect ⇒ R11
and R12 are worth their cost.

**STOP** — `spatial` ≈ `rand_spatial` on `anchor_sim_f0`. Only the head *count*
matters, and the routing program is built on sand. This reading is available
precisely because the wrong-arm rate was bounded at 1.3% beforehand.

**Mechanism claim, independent of the above** — `r7_vp` slope ≪ 0 while
`all`/`spatial` slopes ≈ 0 ⇒ the persistent re-roped bank holds appearance where
cached-latent injection fades. This is the argument for the design and it
survives even a STOP on routing.

**Not concluded here, in any outcome:** anything about motion *transfer*, the
absolute motion level, or the complementary appearance/motion design. The motion
anchor is unbuilt.

## Step commands

### build-anchor-sim

Needs a GPU (DINO), so it runs as a batch job. **Never write outputs to `/tmp`** —
it is node-local here, so the CSVs land on the compute node and are lost when the
job ends (this happened on job 907281; only stdout survived). Write into the repo.

```bash
cd ~/Code/StreamEdit_bigchantier
python evaluation/r10_anchor_sim.py \
  --in_root /projects/dataggen/outputs/five_bench/r10_vp_arms \
  --arms none all spatial temporal \
  --r7_root /projects/dataggen/outputs/five_bench/r7_visual_prompting \
  --cases 0057_dog \
  --out_csv evaluation/csv/smoke/r10_anchor_sim.csv
# PASS if anchor_sim_f0(all) >> anchor_sim_f0(none)
```

Result on 907281: `all` 0.949 vs `none` 0.598 — the metric sees the anchor, so
the mask, bbox crop and anchor path are all correct.

### analyze

Needs a GPU + the `five-bench` env (CLIP / LPIPS / DINO / CoTracker), so it runs
as a batch job rather than on the login node:

```bash
cd ~/Code/StreamEdit_bigchantier
mkdir -p evaluation/csv/_superseded && mv evaluation/csv/r10_*.csv evaluation/csv/_superseded/
mv evaluation/csv/edit*_FiVE_r10_*.csv evaluation/csv/_superseded/ 2>/dev/null
sbatch slurm_scripts/five_bench/r10_eval.sh
```

Archiving first matters: the existing `r10_arms.csv` and the per-arm
`edit{T}_FiVE_r10_{arm}_*.csv` files come from the eight-metric whole-frame run
(job 907209) and share filenames with the new output. Merging the two silently
mixes metric sets.
