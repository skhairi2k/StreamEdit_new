---
name: R26 — Spatial Tau via Mask Union
overview: Oracle test of whether a spatially-varying blend exponent tau(p) beats StreamGVE's single global blend_power. Pass 1 renders the 22-clip R26 case set with --blend_sched zero and dumps the per-latent-frame grounding-mask union M_f = M^src_f | M^trg_f already computed at denoising index 7 (t_inj = 0.5). Pass 2 re-renders the same cases with a per-frame, cache-aligned two-level field built from scratch (the earlier R25 per-token scaffolding has been deleted from the tree) tau(f,p) = tau_bg + (tau_fg - tau_bg) * M_f(p), sweeping tau_bg in {0,1,2} x tau_fg in {2,6,10,50} under vp anchoring (tau_fg=50 added 2026-08-28 as an extreme endpoint, tasks 9-11). The (2,2) cell is a degenerate control that must reproduce the Eq. 4 baseline bit-for-bit. Not a proposed method — the mask is privileged (it costs a full extra render), so this measures the ceiling a spatial tau could reach, and a null result closes the direction.
task_id: R26
todos:
  - id: make-cases
    content: "DONE - evaluation/cases.json was extended IN PLACE from 18 to 22 entries (no separate per-task manifest). Appended: 0017_kid-football, 0058_boat at edit_type 3 (colour change); 0089_A_dog, 0028_kite-walk at edit_type 4 (material change), all four copied from the FiVE edit3/edit4 jsons. 0028_kite-walk now appears twice, at type 2 and type 4, under distinct case_ids (the second is 0028_kite-walk_e4, following the existing 0011_lucia_e2/_e5 convention); run_fivebench.py filters on video_name AND edit_type (:184-194) so the two never collide. The 18 pre-existing entries are byte-identical and their case_ids untouched, because r9/r10/r19 key stored output dirs on case_id. Composition: type1 x1, type2 x13, type3 x2, type4 x2, type5 x3, type6 x1 = 22."
    status: completed
  - id: dump-union
    content: "Port a mask dump into the ACTIVE tree (the R4 --dump_masks only ever landed in old_streamedit/). Add `union_dump_dir: Optional[str] = None` to rollout_inference (:62) and inference (:260) in pipeline/edit_causal_inference.py, forwarded at both window-loop call sites (:135, :197). At the existing t_inj union site (:618-628) — the line `inloop_trg_fg_mask = inloop_trg_fg_mask_bin | src_fg_mask_bin` IS M_f, no new mask maths — write the bool tensor for this chunk into a per-case accumulator, and at the end of the rollout save one npz per case: key 'M' packed via np.packbits, shape [num_latent_frames, frame_seq_length], plus 'shape' and 'frame_seq_length' as plain arrays. `union_dump_dir=None` must leave the code path byte-identical."
    status: completed
  - id: cache-aligned-field
    content: "Add a per-token blender rate to wan/modules/causal_model.py from scratch. Today the three blend sites all multiply by the SCALAR `blender_rate` (:371-372 masked prev key, :390 current key, :399 query), computed at :330-334 with the R10 blend_off and R20 scheduled-rate branches ahead of it. Introduce `rate_tok = None if blend_off else kv_cache.get('blend_rho_tok', None)` of shape [B, L_cache_size], CACHE-ALIGNED so it is sliced by exactly the same `attn_seq_slice` (:277) that `b_trg_attn_fg_mask` already uses at :365 -- that is what makes the field correct under KV-cache eviction and across chunk boundaries without any position arithmetic. Substitute `br_prev`/`br_cur` for the scalar at the three sites, keeping each expression's original shape so `rate_tok is None` stays the untouched scalar path. CRITICAL numerics: hold the rate in float32, never the key dtype. The scalar path multiplies a bfloat16 tensor by a PYTHON FLOAT, which PyTorch applies at full precision; casting the rate to bfloat16 rounds it to 8 mantissa bits (0.360000 -> 0.359375, 0.17% error) which compounds over 15 steps x 6 blocks of an autoregressive rollout and makes the tau_bg == tau_fg control FAIL to reproduce the global schedule. bf16 * fp32 promotes to fp32, so round each PRODUCT back to the key dtype before adding, matching the scalar path's rounding structure exactly."
    status: completed
  - id: rho-from-mask
    content: "pipeline/utils.py (which now holds only `_schedule_blend_rate` at :24 and the tokenizer helpers from :39): add `build_rho_field_from_mask(mask, tau_bg, tau_fg)` returning tau_bg + (tau_fg - tau_bg) * mask.float() for a bool mask [B, L], and `blender_rate_from_rho(rho, timestep_next)` returning 1 - t_next ** rho elementwise -- the tensor twin of the scalar `1 - t ** blend_power` at causal_model.py:334. Hard binary, no dilation, no percentile rescale: the input is already a binary mask, not a continuous field. tau == 0 must be handled EXPLICITLY rather than leaning on torch.pow(0., 0.) == 1.: at tau_bg = 0 the source weight is 1 at every step INCLUDING the final t = 0, so background queries/keys stay pinned to the source for the whole rollout and never release. Document that as an extreme oracle arm, not a schedule, and reject tau < 0 outright."
    status: completed
  - id: plumb-pass2
    content: "Thread the pass-2 field end to end: `union_mask_dir`, `tau_bg`, `tau_fg` onto rollout_inference/inference; per chunk, load the case's npz, slice the rows for this chunk's absolute latent frames (current_start_frame .. +current_num_frames), build the rho field, and write it into kv_cache['blend_rho_tok'] through the same helper that maintains trg_fg_mask so the rolling alignment is maintained in ONE place. Refuse loudly on a frame-count mismatch between the npz and the rollout — a silent truncation would misalign every frame after the first chunk and is exactly the failure this task cannot afford to miss."
    status: completed
  - id: extend-run-fivebench
    content: "evaluation/run_fivebench.py: add --union_dump_dir (pass 1), --union_mask_dir/--tau_bg/--tau_fg (pass 2). Cross-validate: --tau_bg/--tau_fg without --union_mask_dir is a SystemExit, and --union_mask_dir with --blend_sched is a SystemExit (the R20 schedule overrides the per-step scalar the field is built on, so combining them silently discards one). Shared file with R1/R7/R20/R21/R24 — every addition defaults to today's behaviour."
    status: completed
  - id: write-smoke
    content: "slurm_scripts/five_bench/r26_smoke.sh — four gates on 2 clips. (1) Degenerate control: tau_bg=tau_fg=2 sha256-identical to the plain Eq. 4 vp run (blend_power=2). This is THE gate — it proves the cache-aligned field reaches the bridge AND that the refactor of the modulo path did not perturb the scalar case. (2) Non-regression: defaults still bit-identical to five_bench/baseline. (3) Field reaches the bridge: tau_bg=1,tau_fg=1 (uniform but != 2) must DIFFER from the baseline — without this, gate 1 alone is also satisfied by a field that is silently ignored. (4) Per-frame check: assert the dumped M_f differs across latent frames on a moving-object clip (0069_car-turn), so a collapsed frame-0-only field cannot masquerade as per-frame."
    status: completed
  - id: launch-dump
    content: "slurm_scripts/five_bench/r26_dump.sh — pass 1. `#SBATCH --array=0-5` (one edit type per task), L40S, --mem=64G, --time=04:00:00. --blend_sched zero, --vp_mode vp, --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors (verified: all 6 edit types populated, all 22 R26 pairs covered), --cases_json evaluation/cases.json, --seed 0, --union_dump_dir. Masks to /projects/dataggen/outputs/five_bench/r26_masks/edit{T}/{video}.npz."
    status: completed
  - id: launch-infer
    content: "slurm_scripts/five_bench/r26_infer.sh — pass 2. `#SBATCH --array=0-11`, ONE (tau_bg, tau_fg) cell per task, each looping all 6 edit types internally so the model loads once. L40S, --mem=64G, --time=06:00:00 (22 pairs is ~5% of the 419-pair full bench that took ~10h/arm in R21, so ~30 min/arm with wide margin). ARMS = tau_bg {0,1,2} x tau_fg {2,6,10} (tasks 0-8, bg-major), then tau_fg=50 at each tau_bg (tasks 9-11, APPENDED 2026-08-28 so indices 0-8 keep their existing arm mapping and the completed renders stay valid). arm dir taubg{X}_taufg{Y}_vp under OUT_ROOT r26_spatial_tau. Same seed, same anchors, same cases_json as the dump — per-pair reseeding (run_fivebench.py:213-226) is what makes a 22-pair subset comparable at all, so none of the three may drift."
    status: completed
  - id: write-eval
    content: "slurm_scripts/five_bench/r26_eval.sh — `#SBATCH --array=0-11`, one arm per task, **L40S,A100** (retargeted 2026-08-27: H100 unusable; A100 added — R21's "40GB A100 is worse than L40S" objection judged the full metric set, and no longer binds once Qwen2.5-VL + CoTracker are dropped), --exclude=node52, --mem=64G, --time=06:00:00, --config_path evaluation/fivebench/config_l40s.yaml (drops five_acc + motion_fidelity_score{,_edit_part} — the Qwen2.5-VL and CoTracker models that forced H100). ⚠️ motion_fidelity is the only TEMPORAL metric; the tau_bg=0 arms pin the background to the source for the whole rollout, so losing it removes the measure most likely to expose seam/freeze artefacts. Provisional on the temporal axis; re-score on H100 before publication, `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the R20/R21 metric-crash fix). 16-metric evaluate.py, --cases_json evaluation/cases.json across all 6 edit{T}_FiVE.json. csv stems r26_taubg{X}_taufg{Y}_vp."
    status: completed
  - id: write-summarize
    content: "evaluation/r26_summarize.py — join the 12 arm CSVs into evaluation/csv/r26_spatial_tau.csv, one row per (tau_bg, tau_fg), metrics plus deltas against the (2,2) control row. Cross-check the control against the stored per-video vp reference evaluation/csv/edit{T}_FiVE_r21_ref_vp_frame_stride8.csv (all 6 edit types present, verified) restricted to the 22 cases.json pairs; a mismatch there means the subset is NOT reproducing the full run and every delta in the table is suspect. Guard the arm glob so the surviving r21_* reference CSVs are never mistaken for experiment arms (the R21 bug)."
    status: completed
  - id: write-figures
    content: "evaluation/r26_grid_figure.py — (a) two 3x4 heatmaps over (tau_bg, tau_fg) — 3x3 before the tau_fg=50 column was added 2026-08-28; the code derives the axes from the summary CSV, so no change was needed — one for clip_similarity_target_image and one for lpips_unedit_part, with the (2,2) control cell marked; (b) a qualitative grid, 22 clips x {source, baseline vp (2,2), best tau (overall), best tau (per video), M_f (pass 1)} x 5 time-sampled frames, all rows TRIMMED to `find_closest_num_frame` before normalized-time sampling. FIXED 2026-08-28 after the user spotted drift in `edit5_0007_guitar-violin`: the source dir holds MORE frames than were ever edited (55 on disk, 45 rendered), so sampling normalized time over the untrimmed source put its t=1.0 ten frames past the end of the render and the rows drifted apart at the tail. The VAE round-trips n -> n; it is the TRIM, not the VAE, that shortens the render. After trimming, all rows are the same length and normalized time is exact. REVISED 2026-08-28 at the user's request: the `tau_bg=0` row was dropped and replaced by the pass-1 mask itself, rendered by encoding the source, zeroing the latents outside M_f and VAE-decoding -- the mask is a latent-space object, so this shows it at the resolution and alignment the blend actually applies it. This makes the `figures` step GPU-dependent (`--mask_row none` restores the CPU-only path). REVISED 2026-09-01 at the user's request: row 3 is `best tau (overall)`, chosen by argmax of (normalized clip_similarity_target_image + normalized ssim_unedit_union) across the twelve tau cells' DATASET-WIDE means — ssim_unedit_union is R28's per-arm union-mask preservation metric (`1 - (M_src | M_tgt)`); both metrics are min-max normalized and summed. On the current CSVs this selects (tau_bg=0, tau_fg=6). This replaces the earlier `max tau_fg (2,10)` controlled-comparison row. It reads the R26 summary for CLIP and the R28 `r28_union_vs_part.csv` for the union SSIM (`--r28_summary`), both at subset=all. REVISED AGAIN 2026-09-01, same session, after the user clarified they meant the selection PER VIDEO, not instead of the overall one: added row 4, `best tau (per video)` — the SAME criterion (normalized CLIP + normalized union-SSIM, argmax) but with the min-max normalization taken across EACH clip's own twelve arm values rather than the dataset mean, so the winning cell is an INDEPENDENT argmax per clip and genuinely varies page to page (on the current data: 8 distinct cells across 22 clips, `(0,6)` most-picked at 6/22). This needed a new per-clip data source — R28's per-video CSVs (`evaluation/csv/edit{T}_FiVE_r28_{arm}_frame_stride8.csv`, all 12 arms x 6 edit types) joined back to `cases.json` via the FiVE annotation json's `enumerate()` index (`evaluate.py`'s `file_id` is that index, NOT a video name), since the aggregated summaries have already averaged the per-clip signal away. Row 4 is additive to row 3, not a replacement, per the user's explicit instruction. ⚠️ ssim_unedit_union is PER-ARM and R28 measured Spearman(bg-area, union score)=+0.972, so NEITHER row is a like-for-like cross-arm win (that verdict stays with the heatmaps and the *_unedit_union_fixed family); both illustrate the user's criterion. The criterion also favours tau_bg=0 (background pinned to source, the arm most prone to seam/freeze) at both granularities, so read rows 3-4 against the M_f row; (c) a mask-sanity overlay, M_f drawn over 5 frames of 4 clips, which is the only thing that catches a mask that is empty, inverted, or frame-misaligned. -> evaluation/figures/r26_tau_heatmap.pdf, r26_grids/, r26_mask_sanity.pdf. REVISED AGAIN 2026-09-01, same session, before the figures step was re-run: the user asked for a SECOND selection criterion alongside the existing `union` one (rows 3-4), run at both granularities the same way — `part`: normalized clip_similarity_target_image + normalized `ssim_unedit_part` (the FiVE-Bench background metric `1 - M_src`, present directly in the R26 summary, so its `overall` row needs no R28 input at all). The grid is now 7 rows: source, baseline vp (2,2), best tau (overall, union), best tau (overall, part), best tau (per video, union), best tau (per video, part), M_f (pass 1) — rows that dedupe against the (2,2) control are dropped. `part`'s per-video row reads R26's OWN per-video CSVs (`evaluation/csv/edit{T}_FiVE_r26_{arm}_frame_stride8.csv`), which already carry both `clip_similarity_target_image` and `ssim_unedit_part`, joined back to `cases.json` the same way as `union`'s (`file_id` -> `enumerate()` index into `edit{T}_FiVE.json`). On the current CSVs both criteria's `overall` row pick the SAME cell, (tau_bg=0, tau_fg=6), but their `per video` distributions differ (union: 8 distinct cells across 22 clips, `(0,6)` most-picked at 6/22; part: also 8 distinct cells, `(0,6)` most-picked at 7/22) — the two criteria agreeing on `overall` but disagreeing per-clip is itself a data point on how selection-sensitive this picture is. `union` and `ssim_unedit_part` carry DIFFERENT caveats (see the module docstring): union's per-arm background-area confound vs part's original `1 - M_src` motivating flaw (penalizing correctly-changed pixels) — so all four selection rows remain qualitative illustrations, never a Pareto verdict. `load_per_video_r28_metrics`/`_PER_VIDEO_R28`-only plumbing was generalized to `load_per_video_metrics(..., csv_tmpl, ssim_metric)` and `draw_grids(..., per_video_rows=[(label, cell_map), ...])` so both criteria share one code path instead of duplicating it. REVISED AGAIN 2026-09-01, same session, after the figures step was re-run once with the 7-row layout: the user asked for one more FIXED (non-selected) row, `extreme tau (tau_bg=0, tau_fg=50)` — the two per-region extremes shown together (bg pinned to source, fg pushed to the most aggressive edit-side arm in the sweep), independent of any selection criterion. Inserted right after `baseline vp (2,2)`, so the grid is now up to 8 rows: `{source, baseline vp (2,2), extreme tau (0,50), best tau (overall, union), best tau (overall, part), best tau (per video, union), best tau (per video, part), M_f (pass 1)}`, deduped against `(2,2)` and against each other same as before. New module constant `EXTREME = (0, 50)` and CLI flag `--extreme_cell` (default `'0,50'`, empty string omits the row) so the cell is configurable rather than hardcoded, consistent with `--select_ssim_metric`/`--select_ssim_metric_part`. Confirmed via dry-run (`--skip_grids`) that the `(0,50)` arm has rendered frames on disk — it already appears as a legitimate per-video pick for 5/22 clips under the `union` criterion, so the arm is populated end to end."
    status: completed
  - id: tradeoff-figures
    content: "evaluation/r26_tradeoff_figure.py (new, 2026-09-01 user request) -- two trade-off figures, CLIP (x) vs a background-preservation metric (y), each with THREE series: BLUE = the degenerate diagonal tau_bg==tau_fg (plain scalar Eq.4 baseline swept over blend_power in {2,3,4,6,8,10,20,50}, no spatial signal); RED = every (tau_bg,tau_fg) with tau_bg<tau_fg, one point per cell at its DATASET-WIDE mean (a single global field, fixed for every video); GREEN = a PER-VIDEO oracle frontier -- for alpha swept over [0,1] (`--n_alpha`, default 21), each clip independently argmaxes `alpha*CLIP_norm + (1-alpha)*SSIM_norm` over its OWN tau_bg<tau_fg cells, the winning cell's RAW (CLIP,SSIM) is recorded, and the frontier point for that alpha is the mean of those raw values across clips -- a ceiling requiring both the pass-1 mask AND per-clip supervision, not a candidate setting. fig1 (`--out_part`) uses `ssim_unedit_part` (R26's own summary, no R28 dependency); fig2 (`--out_union_fixed`) uses `ssim_unedit_union_fixed` (R28's region-controlled family, needs the 54-arm R28 rebuild -- see `extend-to-52-arms` in the R28 plan). TWO NORMALIZATIONS, deliberately different: (1) DISPLAY axes are GLOBAL min-max, fit once per figure across every plotted arm's dataset-wide mean, so blue/red/green share one coordinate system where 0/1 = worst/best arm in the sweep; (2) the ORACLE'S SELECTION criterion is PER-VIDEO min-max, computed independently within each clip's own tau_bg<tau_fg cells -- reusing the global scale here would let one clip's naturally wider raw metric range dominate the argmax for every clip and every alpha, which is exactly why `r26_grid_figure.py`'s existing `pick_best_tau_per_video` also normalizes per-video rather than globally. The oracle's averaged raw point is then mapped through the SAME global (lo,hi) affine as the display axes (`_global_affine`/`_apply_affine`) so it lands on the same plot, and is free to fall outside [0,1] since it need not equal any single arm's mean. fig2's dataset-wide CLIP is read from `r26_spatial_tau.csv` rather than `r28_union_vs_part.csv`, because R28's summary carries NO CLIP column (its `--metrics` list is the four background bases x three families only) -- but fig2's ORACLE side reads CLIP and `ssim_unedit_union_fixed` TOGETHER from the same R28 per-video CSV row, avoiding a cross-run join for the value that decides the argmax. Reuses `load_summary`/`load_r28_union_metric`/`load_per_video_metrics`/`_minmax`/`_PER_VIDEO_R26`/`_PER_VIDEO_R28` from `r26_grid_figure.py` rather than re-deriving them. Smoke-tested against the CURRENT partial data (12 cells, 11 off-diagonal, from before the 40-arm extension finished) -- both figures render end to end, the oracle frontier visibly dominates the blue/red points as expected, and the script warns explicitly (`fewer than 52 cells`) rather than silently treating a partial grid as final. **Not yet run on final data** -- blocked on the R26 40-arm render+eval extension (tasks 12-51) and, for fig2, the R28 54-arm rebuild (`extend-to-52-arms`, itself blocked on the same R26 renders).

RUN ON FINAL DATA 2026-09-01, same session (`tradeoff-figures-run` step): both blockers cleared, both figures produced on the full 52-cell grid. Oracle frontier dominates blue/red in both; fig2's dominance margin is visibly smaller than fig1's, consistent with `union_fixed` correcting the per-arm `union` background-area confound R28 documented.

GENERALIZED TO LPIPS 2026-09-01, same session, user request: LPIPS is LOWER-is-better, which the original `alpha*CLIP_norm + (1-alpha)*Y_norm` criterion silently assumed was higher-is-better (correct for SSIM, wrong for LPIPS -- summing a maximize term with an un-corrected minimize term biases alpha's low end toward WORSE LPIPS). Fixed by adding `higher_is_better: bool` to `oracle_frontier`: a lower-is-better y is normalized as `1 - minmax(raw)` instead of `minmax(raw)` before entering the weighted sum, so 'higher normalized = more preferred' holds uniformly regardless of the metric's own native direction; the rest of the criterion (weighted sum, argmax, tie-break) is unchanged. DISPLAY axes are untouched by this -- they already plot true/raw values (see the normalization revision above), so a lower-is-better metric simply reads low-is-good on the y-axis, same as its raw numbers read everywhere else in this codebase; no axis flip. Added a `_Y_CONFIGS` table (`part`, `union_fixed`, `lpips_part`, `lpips_union_fixed`) so `--which` now accepts any subset of the four, plus `both` (original two SSIM figures, default) and `all`; each config carries its own `higher_is_better`, source CSV, per-video template and default output path, so `main()` no longer hardcodes SSIM. New default outputs `r26_tradeoff_lpips_part.pdf` / `r26_tradeoff_lpips_union_fixed.pdf`. Verified with `--which all`: all four render, exit 0; both LPIPS figures show the CORRECT oracle shape -- alpha=0 sits at the LOWEST (best) LPIPS on the curve, alpha=1 trades toward the highest CLIP, and the oracle tracks below (better than) the cloud/baseline at every comparable CLIP level, confirming the direction fix rather than just checking it runs without erroring."
    status: completed
  - id: verdict
    content: "CAVEAT from the dump (job 960715, 2026-08-27): 0042_gym-ball, the only edit6 pair, grounded to an ALL-ONES M_f, so its tau field is uniformly tau_fg -- a global rho with NO spatial signal. It is 1 of 22 and must not be counted as evidence for or against spatial tau. Read heatmaps, grids and mask-sanity together, then record in daily.md whether ANY cell Pareto-beats the (2,2) control on (edit-region CLIP, background LPIPS). The direction is only alive if a cell wins with the mask overlay confirming the masks are sound; a win with visibly broken masks is a bug, and a loss with sound masks closes spatial tau as an idea. Note explicitly whether tau_bg=0 (background pinned to source throughout) helps or produces seam/freeze artefacts — it is the arm most likely to score well on background metrics while looking wrong. VERDICT 2026-09-01 (see daily.md Progress/outcomes for full writeup): NO cell Pareto-beats (2,2) on (clip_similarity_target_image, lpips_unedit_part) in either subset -- every CLIP gain trades off against lpips_unedit_part and vice versa (monotone trade-off curve, not a win). Masks are sound except the known 0042_gym-ball degenerate (1/22); 0034_cows' visually broad mask is EXPECTED per R28 (union of source-cow + target-dragon-wings mask), not a defect. tau_bg=0 shows no seam/freeze/ghosting in the reviewed clips (bus, horsejump-high, gym-ball, kite-walk), but the `extreme (0,50)` arm shows a mild, real background/overall SOFTENING (loss of fine texture, e.g. sand grain in kite-walk) relative to (2,2) -- consistent with `structure_distance` getting monotonically worse with tau_fg, and explains why `niqe_target_image` improves at high tau_fg (NIQE rewards smoothness, not fidelity). Direction CLOSED at the single-global-cell level; an oracle-style per-video pick (as R23/R25 did) was NOT tested here and remains open if pursued."
    status: completed
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r26_smoke.sh
    status: completed
    completed_at: 2026-08-26
    job_id: "960461"

  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "grep -E '^(IDENTICAL|DIFFERS|PERFRAME-OK|MULTIWINDOW-OK|CROSSVAL-OK|GATE[0-9]*B?-(FAIL|SKIP))' logs/r26_smoke_{job_id}.out  # the 6 gates as written in r26_smoke.sh: IDENTICAL gate1 (tau(2,2)==eq4 vp on both edit5 clips -- THE gate), IDENTICAL gate2 (defaults==five_bench/baseline; GATE2-SKIP if the reference dir is absent, and a GATE2-FAIL may be the stale pre-2026-07-22 reseeding reference rather than a regression -- only conclusive if gate1 passed), DIFFERS gate3 (tau(1,1) != eq4 -- proves the field is not silently ignored; gate1 alone does NOT rule that out), PERFRAME-OK gate4 (M_f varies across latent frames on 0069_car-turn), MULTIWINDOW-OK gate5 + IDENTICAL gate5b (0034_cows dumps 24 latent frames across the window boundary AND the control still holds there -- the only gate that exercises the rollout overlap stitching), CROSSVAL-OK gate6 (--union_mask_dir+--blend_sched and bare --tau_* both SystemExit). Expect 3 IDENTICAL, 1 DIFFERS, and no *-FAIL line. Any FAIL means the field is not correctly wired to the bridge -- stop, do not launch 9 arms."
    status: completed
    completed_at: 2026-08-26

  - id: dump-union
    type: sbatch
    wait_for: wait-smoke
    command: sbatch slurm_scripts/five_bench/r26_dump.sh
    sets_status: running
    status: completed
    completed_at: 2026-08-27
    job_id: "960715"

  - id: wait-dump
    type: manual
    wait_for: dump-union
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; ls /projects/dataggen/outputs/five_bench/r26_masks/edit*/*.npz | wc -l  # expect 22; python -c \"import numpy as np,glob; [print(f.split('/')[-1], np.load(f)['shape']) for f in sorted(glob.glob('/projects/dataggen/outputs/five_bench/r26_masks/edit*/*.npz'))]\"  # every row [num_latent_frames, frame_seq_length], no zero-frame case; grep -h 'failures:' logs/r26_dump_*.out  # all 0"
    status: completed
    completed_at: 2026-08-27

  - id: infer
    type: sbatch
    wait_for: wait-dump
    command: sbatch slurm_scripts/five_bench/r26_infer.sh
    status: completed
    completed_at: 2026-08-27
    job_id: "960999"

  - id: wait-infer
    type: manual
    wait_for: infer
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; OUT=/projects/dataggen/outputs/five_bench/r26_spatial_tau; for a in $OUT/taubg*_taufg*_vp; do echo $(basename $a) $(ls -d $a/*/*/ 2>/dev/null | grep -vc '_resize/$'); done  # 12 arms, each 22; grep -h 'failures:' logs/r26_infer_*.out  # all 0"
    sets_status: finished
    status: completed
    completed_at: 2026-08-28

  - id: evaluate
    type: sbatch
    wait_for: wait-infer
    command: sbatch slurm_scripts/five_bench/r26_eval.sh
    status: completed
    completed_at: 2026-08-27
    job_id: "961492"

  - id: wait-eval
    type: manual
    wait_for: evaluate
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; ls evaluation/csv/r26_taubg*_taufg*_vp.csv | wc -l  # expect 9; python -c \"import glob,csv; [print(f, sum(1 for _ in open(f))) for f in sorted(glob.glob('evaluation/csv/r26_taubg*_vp.csv'))]\"  # 17 columns, no nan in any overall mean"
    status: completed
    completed_at: 2026-08-31

  - id: summarize
    type: local
    wait_for: wait-eval
    command: |
      python evaluation/r26_summarize.py \
        --arm_glob 'evaluation/csv/r26_taubg*_taufg*_vp_avg.csv' \
        --control taubg2_taufg2_vp \
        --cases evaluation/cases.json \
        -o evaluation/csv/r26_spatial_tau.csv
    output_paths:
      - evaluation/csv/r26_spatial_tau.csv
    note: "SUPERSEDED-PENDING 2026-09-01: computed on the original 12-arm grid. summarize-52 below will overwrite evaluation/csv/r26_spatial_tau.csv with all 52 arms once evaluate-52 finishes -- re-run figures/verdict against the new file rather than treating this run as final."

  - id: infer-52
    type: sbatch
    wait_for: summarize
    command: "sbatch --array=12-51 slurm_scripts/five_bench/r26_infer.sh   # the 40-arm grid extension (see Decisions: tau grid); submitted in capacity-chunked pieces (jobs 974780/974782/974783/974784/974918/975191/975232/975564/975623/975911/976020/976094/976172/976194/976215/976230) because the array exceeded the per-user submit cap in one shot"
    sets_status: finished
    status: completed
    completed_at: 2026-09-01
    job_id: "multiple chunked jobs, see above; last task (976230_51) finished 2026-09-01 ~16:21"

  - id: wait-infer-52
    type: manual
    wait_for: infer-52
    check_hint: "squeue -u \\$USER | grep r26_infe  # expect empty once done; ls -d /projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg*_vp | wc -l  # expect 52; grep -h 'failures:' logs/r26_infer_*.out | grep -v 'failures: 0'  # expect no output"
    status: completed
    completed_at: 2026-09-01
    note: "Verified 2026-09-01: 52/52 arm directories present under r26_spatial_tau/. Full per-manifest failure-count check not yet re-run across all 52 arms' logs."

  - id: evaluate-52
    type: sbatch
    wait_for: wait-infer-52
    command: "sbatch --array=12-51 slurm_scripts/five_bench/r26_eval.sh   # NO --dependency; independent of R28's dump/fixed_union/eval chain -- R26's own eval never touches R28's masks"
    sets_status: finished
    status: completed
    completed_at: 2026-09-01
    job_id: "976470 (12-26), 976550 (27-28), 976569 (29-39), 976586 (40-51) -- all completed"
    note: "BUG FOUND AND FIXED 2026-09-01 ~16:31: the original watcher (r26_eval_remainder.sh) used `--dependency=afterok:976215:976230` (the last 2 R26 render tasks). Its FIRST chunk (12-26, job 976470) succeeded because those render jobs were still in live squeue at submission time, but the SECOND chunk (27-40), submitted ~10 minutes later after both render jobs had completed and left squeue, failed with `sbatch: error: Batch job submission failed: Job dependency problem` -- referencing an already-evicted job apparently needs the slurmdbd accounting DB, which is unreachable on this cluster (see R28 plan's `dump-masks-52` note for the full diagnosis; R28's chain hit the identical failure at the same time). Killed the broken watcher (TaskStop) and replaced it with r26_eval_remainder2.sh (scratchpad-only), which submitted the remaining tasks (27-51) with NO dependency at all. All 40 new arms scored cleanly: 0 non-ok manifest rows, 0 error_lines in any metrics log."

  - id: wait-eval-52
    type: manual
    wait_for: evaluate-52
    check_hint: "ls evaluation/csv/r26_taubg*_taufg*_vp.csv | wc -l  # expect 52; grep -h 'error_lines=' logs/r26_eval_*.metrics.log 2>/dev/null; python -c \"import glob; [print(f) for f in sorted(glob.glob('evaluation/csv/r26_taubg*_vp_avg.csv'))]\" | wc -l  # expect 52"
    status: completed
    completed_at: 2026-09-01
    note: "Verified 2026-09-01: 52/52 r26_taubg*_taufg*_vp_avg.csv present, 0 files with 'nan', 0 non-'failures: 0' lines across logs/r26_eval_976*.out."

  - id: summarize-52
    type: local
    wait_for: wait-eval-52
    command: |
      python evaluation/r26_summarize.py \
        --arm_glob 'evaluation/csv/r26_taubg*_taufg*_vp_avg.csv' \
        --control taubg2_taufg2_vp \
        --cases evaluation/cases.json \
        -o evaluation/csv/r26_spatial_tau.csv
    output_paths:
      - evaluation/csv/r26_spatial_tau.csv
    status: completed
    completed_at: 2026-09-01
    note: "104 rows (52 arms x 2 subsets), exit 0 -- BUT the first attempt crashed (exit 1) after writing the CSV: the console heatmap printer assumed a full RECTANGULAR (tau_bg, tau_fg) grid and did `next(a for b,f,a,_ in arms if b==bg and f==fg)` with no default, which raises StopIteration on a missing cross-product cell -- the extended grid is TRIANGULAR (tau_bg<=tau_fg only), so e.g. (tau_bg=3, tau_fg=2) genuinely does not exist. Fixed in evaluation/r26_summarize.py: the lookup now defaults to None and prints '--' for a missing cell instead of crashing. Control cross-check still OK (9.43e-05, unchanged from the 12-arm run). Console table shows CLIP-T peaking around (tau_bg=20, tau_fg=20) (28.3064 all / 28.5562 excl_degenerate) rather than at tau_bg=0 as the original 12-arm grid suggested -- worth checking against LPIPS at the same cell in verdict-52, since LPIPS-bg degrades monotonically with both tau_bg and tau_fg in this table."

  - id: tradeoff-figures-run
    type: local
    wait_for: summarize-52
    command: |
      python evaluation/r26_tradeoff_figure.py \
        --summary evaluation/csv/r26_spatial_tau.csv \
        --r28_summary evaluation/csv/r28_union_vs_part.csv \
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
        --cases evaluation/cases.json \
        --which all
    output_paths:
      - evaluation/figures/r26_tradeoff_part.pdf
      - evaluation/figures/r26_tradeoff_union_fixed.pdf
      - evaluation/figures/r26_tradeoff_lpips_part.pdf
      - evaluation/figures/r26_tradeoff_lpips_union_fixed.pdf
    status: completed
    completed_at: 2026-09-01
    note: "Both figures produced on the FULL 52-cell grid (44 off-diagonal + 8 diagonal), 21 alpha points each, exit 0. Visually verified: in BOTH figures the green per-video oracle frontier dominates the blue baseline curve and red single-global-cell cloud everywhere. fig2's dominance is visibly LESS extreme than fig1's -- consistent with union_fixed correcting the per-arm union's background-area confound that inflated apparent gains in the uncontrolled metric. CROSS-PLAN DEPENDENCY resolved: --r28_summary read from evaluation/csv/r28_union_vs_part.csv, produced by the R28 plan's summarize-52 (completed 2026-09-01) and re-verified by verdict-52 (completed 2026-09-01, same session).

REVISED 2026-09-01, same session, at the user's request: axes switched from min-max NORMALIZED to TRUE/raw values. Normalization is retained ONLY where it is actually needed -- balancing CLIP and SSIM's unrelated raw scales inside the oracle's alpha-weighted per-video selection criterion (`oracle_frontier`'s internal `_minmax` calls, unchanged) -- and removed everywhere a value is actually PLOTTED. `_global_affine`/`_apply_affine` (the display-axis min-max mapping) deleted from `r26_tradeoff_figure.py` as now-dead code; `draw_tradeoff` plots `cells_clip[c]`/`cells_ssim[c]` and the oracle's raw `(mean_clip, mean_ssim)` tuples directly, axis labels dropped the 'normalized ' prefix. Re-run, same command, exit 0: fig1 now reads CLIP-T in [26.9, 29.4] vs ssim_unedit_part in [0.775, 0.857]; fig2 reads the same CLIP-T range vs ssim_unedit_union_fixed in [0.845, 0.882] -- both directly comparable to the per-cell numbers in the underlying CSVs, e.g. the (2,2) control at clip=27.03/part=0.8542 is now readable straight off the plot rather than requiring a lookup against a normalized position.

EXTENDED TO LPIPS 2026-09-01, same session, at the user's request (see the `tradeoff-figures` todo for the `higher_is_better` mechanism). Command changed to `--which all` so this step now produces all four figures in one run. Re-run, exit 0: fig3 (CLIP vs lpips_unedit_part) reads lpips_unedit_part in [0.208, 0.276]; fig4 (CLIP vs lpips_unedit_union_fixed) reads lpips_unedit_union_fixed in [0.170, 0.192]. Both LPIPS oracle curves confirmed correctly shaped (alpha=0 at the lowest/best LPIPS, tracking below the cloud/baseline at every comparable CLIP)."

  - id: figures
    type: local
    wait_for: summarize
    command: |
      python evaluation/r26_grid_figure.py \
        --summary evaluation/csv/r26_spatial_tau.csv \
        --r28_summary evaluation/csv/r28_union_vs_part.csv \
        --masks /projects/dataggen/outputs/five_bench/r26_masks \
        --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \
        --cases evaluation/cases.json
    output_paths:
      - evaluation/figures/r26_tau_heatmap.pdf
      - evaluation/figures/r26_mask_sanity.pdf
    status: completed
    completed_at: 2026-09-01

  - id: verdict
    type: manual
    wait_for: figures
    check_hint: "Read evaluation/figures/r26_tau_heatmap.pdf, r26_mask_sanity.pdf and figures/r26_grids/ together. Does any cell Pareto-beat taubg2_taufg2 on (clip_similarity_target_image, lpips_unedit_part) WITH sound masks? Record the verdict and the tau_bg=0 artefact check in daily.md."
    sets_status: analyzed
    status: completed
    completed_at: 2026-09-01
isProject: true
---

# R26: Spatial Tau via Mask Union

## Context

Test whether making the blend exponent `tau` **spatial** (per latent token, per frame) rather than a single global scalar improves editing. StreamGVE Eq. 4 gives every token the same release schedule `W^src(t_i) = t_i^tau`; this task gives edit-region tokens a large `tau` (release the source fast) and background tokens a small one (stay anchored).

This is an **oracle test, not a method**. The mask that drives the field is obtained from a full extra render of the same clip, so the two-pass cost is deliberate and irrelevant to any speed claim: the question is only whether a spatial `tau` has a ceiling worth chasing.

One piece already exists and is reused rather than rebuilt: the union `M_f = M^src_f ∪ M^trg_f` is literally computed today at [edit_causal_inference.py:625](../../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L625) (`inloop_trg_fg_mask = inloop_trg_fg_mask_bin | src_fg_mask_bin`) at denoising index `len//2` = 7 of 15, i.e. `t_inj = 0.5`; pass 1 only has to write it out.

The per-token blend field, by contrast, is built from scratch. An earlier per-token scaffolding (R25 V0) existed in this tree but has been deleted, so the three blend sites in [causal_model.py:371-399](../../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L371-L399) are back to the pristine scalar `blender_rate` form and `pipeline/utils.py` holds only the R20 schedule helper. That is a cleaner starting point — the field is designed for a per-frame mask from the outset rather than widened from a one-frame assumption — but it does mean the "uniform field reproduces the scalar path bit-for-bit" property has to be re-established by this task rather than inherited, which is what the smoke gates exist to prove.

**Done when:** the `(tau_bg, tau_fg)` heatmap and qualitative grids are read together with the mask-sanity overlay, and `daily.md` records whether any cell Pareto-beats the `(2,2)` control on (edit-region CLIP, background LPIPS) with masks confirmed sound.

## Execution steps

| # | Step id | Type | What | Sets status |
|---|---------|------|------|-------------|
| — | *(prep)* | — | All `todos` — mask dump, cache-aligned field, pass-2 plumbing, scripts | `implemented` |
| 1 | `smoke` | sbatch | 4 gates on 2 clips — control identity, non-regression, field-reaches-bridge, per-frame | |
| 2 | `wait-smoke` | manual | Verify all 4 gates before committing the arms | |
| 3 | `dump-union` | sbatch | **Pass 1** — `--blend_sched zero`, dump `M_f` per case | `running` |
| 4 | `wait-dump` | manual | 22 npz present, shapes sane | |
| 5 | `infer` | sbatch | **Pass 2** — 12-cell `(tau_bg, tau_fg)` array | |
| 6 | `wait-infer` | manual | 12 arms × 22 renders, 0 failures | `finished` |
| 7 | `evaluate` | sbatch | 16-metric eval, one task per arm | |
| 8 | `wait-eval` | manual | 9 CSVs, 17 columns, no nan | |
| 9 | `summarize` | local | Join arms, deltas vs `(2,2)` control | |
| 10 | `figures` | local | Heatmaps, qualitative grids, mask sanity | |
| 11 | `verdict` | manual | Record the go/no-go in `daily.md` | `analyzed` |

```
/run-step R26                 # next pending step
/run-step R26 dump-union      # a named step
```

## Decisions

| Question | Choice | Note |
|---|---|---|
| Task id | **R26** | The earlier off-board `r26_*` / `r27_*` propagation artefacts (scripts, CSVs, case manifests) have been deleted, so the `r26_` namespace is clean and exclusively this task's |
| Case set | `evaluation/cases.json`, extended in place 18 → 22 | 18 from `cases.json` + `0017_kid-football`,`0058_boat` @ type 3 (colour) + `0089_A_dog`,`0028_kite-walk` @ type 4 (material); all 4 verified present in the FiVE jsons |
| Pass 1 blending | `--blend_sched zero` | `W^src ≡ 0`, `blender_rate = 1`, pure target Q/K. Background source-KV injection at `t > t_inj` is NOT disabled by this — it is a separate channel |
| Mask extracted at | denoising index 7 of 15 (`len//2`, `t_inj = 0.5`) | The existing union site; no new mask computation |
| `tau` grid | `tau_bg ∈ {0,1,2,3,4,6,8,10,20,50}` × `tau_fg ∈ {2,3,4,6,8,10,20,50}`, **constrained to `tau_bg ≤ tau_fg`** = 52 arms | `(2,2)` is the degenerate control = global `blend_power=2`. `tau_fg=50` added 2026-08-28 (tasks 9-11): `tau_fg` 6→10 already flattened and a paired test cannot separate any `tau_fg≥6` cell, so this is an extreme endpoint to see whether the axis saturates or breaks. `W^src = t^50` is 3.1e-2 at the first blend and <1e-3 after, i.e. the edit region is effectively released from the source for the whole rollout — an endpoint, not a candidate setting. **Extended 2026-09-01** (tasks 12-51, +40 arms) to a finer sweep after the original 12-cell verdict closed the direction at coarse resolution — the finer grid asks whether a non-monotonic sweet spot was hiding between the sampled points, not whether to reopen the closed conclusion. The full 10×8=80 cross product is deliberately NOT used: `tau_bg` and `tau_fg`'s ranges overlap (both reach into `{2,3,4,6,8,10,20,50}`), so an unconstrained product would include inverted pairs like `(tau_bg=50, tau_fg=2)` — background released faster than the edit region, the opposite of the method's intended semantics (`tau_bg` anchors, `tau_fg` releases). Restricting to `tau_bg ≤ tau_fg` keeps every arm semantically meaningful and holds the grid to 52 arms instead of 80. `run_fivebench.py --tau_bg/--tau_fg` are `type=float` with no range validation, so no code change was needed — only `r26_infer.sh`/`r26_eval.sh`'s `ARMS` arrays and `--array` bounds (`0-11` → `0-51`), appending the 40 new pairs after the existing 12 (same convention as the `tau_fg=50` append) so the already-rendered 12 arms' output directories, keyed by `METHOD` name rather than array index, are untouched. `r26_summarize.py`'s arm glob/regex and `r26_grid_figure.py`'s heatmap axes are both derived from the matched CSVs, not hardcoded to 12, so neither needed changes. **Not yet launched** — `infer`/`evaluate`/`summarize`/`figures`/`verdict` below still reflect the original 12-arm analysis; re-running tasks 12-51 of `r26_infer.sh` and `r26_eval.sh` (then `summarize`/`figures`/`verdict`) is pending explicit go-ahead given the compute cost (40 arms × ~2h infer + eval each). **Degenerate-cell shortcut** (user request, 2026-09-01): every `tau_bg == tau_fg` cell — `(2,2),(3,3),(4,4),(6,6),(8,8),(10,10),(20,20),(50,50)`, 8 of the 52 arms — is mathematically identical to the plain scalar Eq. 4 baseline at `blend_power=tau`, since a uniform two-level field collapses to the constant it interpolates between. `r26_infer.sh` now detects `BG == FG` and calls `run_fivebench.py --blend_power "$BG"` directly (no `--union_mask_dir`/`--tau_bg`/`--tau_fg`, no pass-1 mask dependency), skipping the mask-coverage preflight for those arms; `r26_smoke.sh` job 960461 gate1 already proved the two code paths agree sha256-identically at `(2,2)`, so there is nothing left to re-derive at the other 7 diagonal cells through the two-pass field. `r26_eval.sh` is unaffected — it scores `$OUT_ROOT/$METHOD` by directory layout, which is identical either way |
| `tau_bg = 0` semantics | Background pinned to source for the entire rollout | `W^src = t^0 ≡ 1` at every step incl. `t=0`; requires relaxing the `rho_min > 0` guard, and must be made explicit rather than relying on `torch.pow(0.,0.)==1.` |
| Field shape | Hard binary two-level | `tau(f,p) = tau_bg + (tau_fg - tau_bg)·M_f(p)`; no dilation, no soft rescale |
| Field extent | Per-frame, **cache-aligned** `[B, L_cache_size]` | Sliced by the same `attn_seq_slice` as `trg_fg_mask`, so eviction and chunk boundaries are handled in one place and no position arithmetic is needed |
| Anchoring | `vp` only | `--first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors` — verified all 6 types populated, all 22 pairs covered |
| Seed / steps | `--seed 0`, 15 steps, `guidance_scale 1.0` | Identical across pass 1, pass 2 and smoke; per-pair reseeding is what makes the 22-pair subset comparable |
| Mask store | `/projects/dataggen/outputs/five_bench/r26_masks/edit{T}/{video}.npz` | `np.packbits`, `[num_latent_frames, frame_seq_length]` bool |
| Render store | `/projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg{X}_taufg{Y}_vp/` | |
| Out of scope | dilated/feathered masks, soft `mask_soft` field, `novp`/`pvp`, single-pass online variant, full-bench scale | Follow-ups only if a cell wins |

## Step commands

### smoke
```bash
sbatch slurm_scripts/five_bench/r26_smoke.sh
```

### dump-union
```bash
sbatch slurm_scripts/five_bench/r26_dump.sh
# inside, per array task T = SLURM_ARRAY_TASK_ID + 1:
#   python evaluation/run_fivebench.py \
#     --edit_type "$T" --method r26_dump \
#     --out_root /projects/dataggen/outputs/five_bench/r26_masks_render \
#     --cases_json evaluation/cases.json \
#     --blend_sched zero \
#     --vp_mode vp \
#     --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors \
#     --union_dump_dir /projects/dataggen/outputs/five_bench/r26_masks \
#     --seed 0
```

### infer
```bash
sbatch slurm_scripts/five_bench/r26_infer.sh
# ARMS[SLURM_ARRAY_TASK_ID] = "<tau_bg> <tau_fg>" over {0,1,2} x {2,6,10} then {0,1,2} x {50}; per arm, loop T=1..6:
#   python evaluation/run_fivebench.py \
#     --edit_type "$T" --method "taubg${BG}_taufg${FG}_vp" \
#     --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \
#     --cases_json evaluation/cases.json \
#     --union_mask_dir /projects/dataggen/outputs/five_bench/r26_masks \
#     --tau_bg "$BG" --tau_fg "$FG" \
#     --vp_mode vp \
#     --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors \
#     --seed 0
```

### evaluate
```bash
sbatch slurm_scripts/five_bench/r26_eval.sh
# per array task: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   python evaluation/fivebench/evaluate.py \
#     --result_dir /projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg${BG}_taufg${FG}_vp \
#     --cases_json evaluation/cases.json \
#     --out evaluation/csv/r26_taubg${BG}_taufg${FG}_vp.csv
```

### summarize
```bash
python evaluation/r26_summarize.py \
  --arm_glob 'evaluation/csv/r26_taubg*_taufg*_vp_avg.csv' \
  --control taubg2_taufg2_vp \
  --cases evaluation/cases.json \
  -o evaluation/csv/r26_spatial_tau.csv
```

### figures
```bash
python evaluation/r26_grid_figure.py \
  --summary evaluation/csv/r26_spatial_tau.csv \
  --r28_summary evaluation/csv/r28_union_vs_part.csv \
  --masks /projects/dataggen/outputs/five_bench/r26_masks \
  --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \
  --cases evaluation/cases.json
```

## Pipeline

```mermaid
flowchart TD
    C[cases.json<br/>22 pairs] --> D[r26_dump.sh<br/>PASS 1 --blend_sched zero]
    ANC[/projects .../anchors/edit1-6/] --> D
    ANC --> F
    D --> E[r26_masks/edit_T/video.npz<br/>M_f packed bool]
    E --> F[r26_infer.sh<br/>PASS 2, 12 arms]
    C --> F
    F --> G[r26_spatial_tau/<br/>taubgX_taufgY_vp]
    G --> H[r26_eval.sh<br/>16 metrics]
    H --> I[csv/r26_taubgX_taufgY_vp.csv]
    I --> J[r26_summarize.py]
    J --> K[csv/r26_spatial_tau.csv]
    K --> L[r26_grid_figure.py]
    E --> L
    G --> L
    R28[(r28_union_vs_part.csv<br/>ssim_unedit_union per arm)] --> L
    R28PV[(edit{T}_FiVE_r28_{arm}_frame_stride8.csv<br/>per-video CLIP + union SSIM, 12 arms x 6 types)] --> L
    R26PV[(edit{T}_FiVE_r26_{arm}_frame_stride8.csv<br/>per-video CLIP + ssim_unedit_part, 12 arms x 6 types)] --> L
    L --> M[figures/r26_tau_heatmap.pdf<br/>r26_mask_sanity.pdf<br/>r26_grids/]
```

## Code to touch

| File | Change |
|---|---|
| `Self-Forcing_StreamEdit/pipeline/utils.py` | **new** `build_rho_field_from_mask(mask, tau_bg, tau_fg)` and `blender_rate_from_rho(rho, t_next)`, with `tau == 0` handled explicitly |
| `Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py` | Add `union_dump_dir` / `union_mask_dir` / `tau_bg` / `tau_fg` to `rollout_inference` (:62) and `inference` (:260), forward at :135 and :197; dump `inloop_trg_fg_mask` at :625; per chunk build the rho field and install it via the `trg_fg_mask` cache helper |
| `Self-Forcing_StreamEdit/wan/modules/causal_model.py` | Add a cache-aligned `rate_tok` sliced by `attn_seq_slice` (:277), substituting `br_prev`/`br_cur` for the scalar `blender_rate` at the three blend sites (:371-372, :390, :399); float32 rate, per-product rounding |
| `evaluation/run_fivebench.py` | Add `--union_dump_dir`, `--union_mask_dir`, `--tau_bg`, `--tau_fg`; SystemExit on `tau_*` without `--union_mask_dir`, and on `--union_mask_dir` with `--blend_sched` |
| `evaluation/r26_summarize.py` | **new** — join 12 arm CSVs, deltas vs `(2,2)`, cross-check control against R21 vp per-video rows on the 22 pairs |
| `evaluation/r26_grid_figure.py` | **new** — 3×3 heatmaps, qualitative grids, mask-sanity overlay. Grid rows are `{source, baseline vp (2,2), extreme tau (0,50), best tau (overall, union), best tau (overall, part), best tau (per video, union), best tau (per video, part), M_f (pass 1)}` (8 rows; dedupes against `(2,2)` and against each other). `extreme tau` is a FIXED arm (`EXTREME = (0,50)`, `--extreme_cell` to change/omit), not a selection — the two per-region extremes (bg pinned to source, most aggressive fg push) shown together. TWO selection criteria, each an argmax of (min-max-normalized `clip_similarity_target_image` + min-max-normalized SSIM), run at two granularities: `union` uses R28's per-arm union-mask preservation metric `ssim_unedit_union` (`--r28_summary` = `r28_union_vs_part.csv`, subset=all); `part` uses the FiVE-Bench background metric `ssim_unedit_part` straight from the R26 summary (no R28 input). `overall` normalizes once across the twelve cells' dataset-wide means (both currently select (0,6)); `per video` normalizes independently within each clip's own twelve arm values, read from each criterion's own PER-VIDEO CSVs — `union` from `evaluation/csv/edit{T}_FiVE_r28_{arm}_frame_stride8.csv`, `part` from `evaluation/csv/edit{T}_FiVE_r26_{arm}_frame_stride8.csv` — joined to `cases.json` via the FiVE annotation json's `enumerate()` index (`load_edit_id_to_name`/`load_per_video_metrics`/`pick_best_tau_per_video`; new `--per_video_row {on,off}`, `--select_ssim_metric_part`). Both per-video distributions genuinely vary per clip (8 distinct cells across 22 clips each on current data). ⚠️ `ssim_unedit_union` is PER-ARM and ~0.97 rank-correlated with background area (R28); `ssim_unedit_part` carries R28's ORIGINAL flaw instead (`1 - M_src` penalizes correctly-changed pixels) — so NONE of the four rows is a like-for-like cross-arm win; the Pareto verdict stays with the heatmaps and the `*_unedit_union_fixed` family. Both tend to favour `tau_bg=0` (background pinned to source), so read these rows against the M_f row. The `M_f` row is rendered via `MaskDecoder` (encode source → zero latents outside `M_f` → VAE-decode), which makes the step GPU-dependent (`--mask_row none` restores CPU-only). Every row is trimmed to `find_closest_num_frame` before normalized-time sampling, or source and render drift apart at the tail. |
| `slurm_scripts/five_bench/r26_smoke.sh` | **new** — L40S, 4 gates on 2 clips |
| `slurm_scripts/five_bench/r26_dump.sh` | **new** — `--array=0-5`, L40S, `--mem=64G`, `--time=04:00:00` |
| `slurm_scripts/five_bench/r26_infer.sh` | **new** — `--array=0-11`, L40S, `--mem=64G`, `--time=06:00:00`. Tasks 9-11 (`tau_fg=50`) appended 2026-08-28; 0-8 keep their mapping |
| `slurm_scripts/five_bench/r26_eval.sh` | **new** — `--array=0-11`, L40S/A100, `--mem=64G`, `--time=06:00:00`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. ARMS block byte-identical to r26_infer.sh |

Cache-aligned field, replacing the modulo mapping in `causal_model.py`:

```python
# The field lives in the KV cache, aligned exactly like kv_cache["trg_fg_mask"],
# so eviction and chunk boundaries are handled in ONE place instead of two.
rate_tok = None if blend_off else kv_cache.get('blend_rho_tok', None)   # [B, L_cache_size]
...
if rate_tok is None:
    br_prev = br_cur = blender_rate
else:
    # Same slice as b_trg_attn_fg_mask below -- float32 on purpose (see the
    # todo `cache-aligned-field`): a bf16 rate rounds 0.36 -> 0.359375 and compounds
    # over 15 steps x 6 blocks, breaking the tau_bg == tau_fg control.
    b_rate = rate_tok[b_idx][attn_seq_slice].view(-1, 1, 1).float()
    br_prev = b_rate[: -num_new_tokens]
    br_cur  = b_rate[-num_new_tokens:]
```

Pass-1 dump at the existing union site (`edit_causal_inference.py:618-628`):

```python
if index == len(denoising_step_list) // 2:
    ...
    inloop_trg_fg_mask = inloop_trg_fg_mask_bin | src_fg_mask_bin   # == M_f, unchanged
    if union_dump_dir is not None:
        # [B, current_num_frames * frame_seq_length] -> per-frame rows
        union_chunks.append(
            inloop_trg_fg_mask.view(batch_size, current_num_frames, self.frame_seq_length)
                              .cpu()
        )
    self._inject_masks_to_kv_cache(kv_cache_dual, trg_fg_mask_cache, inloop_trg_fg_mask)
```

Pass-2 field build, per chunk, refusing a frame-count mismatch outright:

```python
if union_mask_dir is not None:
    M = union_frames[current_start_frame: current_start_frame + current_num_frames]
    if M.shape[0] != current_num_frames:
        raise ValueError(
            f"R26: mask npz has {union_frames.shape[0]} latent frames but the rollout "
            f"asked for frames [{current_start_frame}, {current_start_frame + current_num_frames}); "
            f"a silent truncation would misalign every frame after this chunk."
        )
    rho = build_rho_field_from_mask(M.flatten(0, 1).unsqueeze(0), tau_bg, tau_fg)
```
