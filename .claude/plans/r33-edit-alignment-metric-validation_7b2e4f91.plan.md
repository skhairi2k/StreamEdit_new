---
name: "R33: Edit-Alignment Metric Validation"
overview: "Replace FiVE-Bench's inherited whole-frame CLIP achievement axis, which does not rank renders by edit strength on this case set (Spearman(b, clip_target) = +0.098, positive on 11/22 clips), with two axes that do: directional CLIP over R26/R30/R31's renders, and comprehensive FiVE-Acc for R26/R31."
task_id: R33
report_url: "https://claude.ai/artifact/PuAmCbhopy1r8fwjYCnMkG"
todos:
  - id: clip-d-script
    content: "Rename evaluation/r31_clip_directional.py -> evaluation/r33_clip_directional.py and slurm_scripts/five_bench/r31_clip_directional.sh -> r33_clip_directional.sh (task-specific files take the r33_ prefix; they were drafted during R31's debugging session before R33 existed). Then extend: (a) add --r30_root for R30's 8 routed arms under r30_arms/, (b) add a delta_img_norm output column -- ||E_img(render) - E_img(source)|| BEFORE normalisation -- because CLIP-D's direction is poorly determined when the render barely differs from the source, and the low-b end of the sweep is exactly that regime; without this column a degenerate reading is indistinguishable from a real one."
    status: completed
    note: "Done 2026-09-18. Both files renamed with plain `mv` (not `git mv` -- `evaluation/`
      is gitignored and the .sh was untracked). Registry verified: 28 methods, R26 16 /
      R30 8 / R31 4, all names unique. ⚠️ R30's tree shape differs from R31's and had to be
      handled separately: r30_stage2.sh:124 sets METHOD=\"$ARM\", so renders sit at
      r30_arms/{arm}/edit{T}/{video}/ with a BARE arm name and NO step dir, while R31 nests
      under r31_{arm}/step14/. Three R30 arm names (lpips, dino_patch, normals) collide with
      R31's, hence the r26_/r30_/r31_ prefix on every method label. delta_img_norm
      unit-tested: a step 1000x smaller gives a BIT-IDENTICAL cos of +1.0000 while the norm
      separates them 0.4580 vs 0.0005 -- which is the whole reason the column exists. Caught
      a real bug while validating: the .py's default output path was still
      r31_clip_directional.csv while the .sh checked for r33_..., so the job would have
      reported FAILED on a successful run; both now agree. Wrapper also prints the 10
      smallest |d_img| rows post-run. Deliberately NOT arrayed -- see the Decisions row."
  - id: fiveacc-script
    content: "slurm_scripts/five_bench/r33_fiveacc.sh -- array over 20 arms (R26's 16 constant-b + R31's 4), calling evaluation/fivebench/evaluate.py with --metrics five_acc ALONE. ⚠️ The arm->path shape differs per task: R26 is {r26_spatial_tau}/{arm}/edit{T}/{video}, R31 is {r31_arms}/r31_{arm}/step14/edit{T}/{video} -- the array registry must carry the full dir per arm, not a root plus a formula. Running five_acc alone is what makes this safe: R20's GPU exhaustion came from Qwen2.5-VL-7B and CoTracker being co-resident with the other metric models, not from five_acc itself."
    status: completed
    note: "Built 2026-09-21 via /build-step. Modeled closely on r26_eval.sh/r31_eval.sh
      (same conda-env dance, --exclude=node52, 22-real-frame-dir guard excluding
      _resize siblings, teed metrics log grepped for 'Error:'), narrowed to
      --metrics five_acc alone. --array=0-19: TID 0-7 -> R26 SPATIAL
      taubg0_taufg{b}_vp (b in CONST_BS = 2,3,4,6,8,10,20,50, same list as
      r30_score.py), TID 8-15 -> R26 UNIFORM taubg{b}_taufg{b}_vp, TID 16-19 -> R31's
      4 arms (lpips/dino_patch/normals/latent) at r31_arms/r31_{arm}/step14/. STEM =
      r33_fiveacc_{arm-label}, following the naming already used by the (now-removed)
      discrete-oracle framework's r30_fiveacc_taubg0_taufg{b}_vp_avg.csv files still on
      disk, and matching wait-fiveacc's check_hint glob
      (edit*_FiVE_r33_fiveacc_*_frame_stride8.csv). bash -n clean; dry-ran the 20-task
      TID->STEM/DIR dispatch locally outside Slurm and confirmed all 20 resolve to the
      arm directories the clip-d job (1002662) just confirmed populated. Not yet
      submitted."
  - id: axis-compare-script
    content: "evaluation/r33_axis_compare.py -- for each candidate achievement axis (raw whole-frame CLIP, edit-part CLIP, CLIP-D prompt, CLIP-D word, FiVE-Acc yn/mc), compute per-clip Spearman(b, axis) over R26's 8-b SPATIAL sweep, the mean, and the count of clips with rho > 0. This is the pass/fail test for every axis on the same footing, including the incumbent. Reads the per-edit-type per-clip CSVs directly, never the top-level _avg.csv."
    status: completed
    note: "Built + run 2026-09-22, ahead of the axis-compare STEP (which /run-step will
      still execute formally) because wait-fiveacc finished sooner than expected and
      the user asked to get moving on the CLIP side. Reuses r30_score.build_join_tables
      / load_r26_metric verbatim for the two stored R26 CLIP columns; CLIP-D read
      directly from r33_clip_directional.csv (case_id already a column, no join
      needed); FiVE-Acc read via a small local analogue of load_r26_metric against
      R33's own edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv files and
      evaluate.py's raw column names (five_acc_yes_no / five_acc_multi_choice, not
      yn_acc/mc_acc -- those are this script's OWN axis labels). One real bug caught
      and fixed while running: scipy.spearmanr returns nan (ConstantInputWarning) for
      a clip whose axis value is identical across all 8 b -- naively averaging that
      into the mean would silently nan out the whole axis. Fixed by excluding flat
      clips from the mean/n_positive denominator and reporting their count as n_flat
      instead. RESULTS (evaluation/csv/r33_axis_comparison.csv): incumbent
      clip_similarity_target_image +0.1145 (11/21, tennis excluded, matches the plan's
      stated +0.098 closely) and its edit-part variant +0.1485 (13/21) both weak, same
      failure mode as the plan's original finding. clip_d_prompt +0.4080 (17/22, no
      flat clips) and clip_d_word +0.3258 (15/22) both clear the incumbent by a wide
      margin and are defined for every clip. yn_acc +0.7113 and mc_acc +0.4638 are
      HIGH but only computable on 6/22 and 4/22 clips respectively (16 and 18 flat) --
      confirms the plan's own coarseness warning: high correlation where FiVE-Acc
      varies at all, but it mostly doesn't vary across this 8-b range, so it is closer
      to a vertical band than a continuous ranking axis."
  - id: figures-script
    content: "evaluation/r33_report_figures.py -- per-video Pareto panels + HTML on the CHOSEN axis, importing the row-strip helpers (crop_arm_rows, extract_baseline_row, label_strip) from evaluation/r31_html_report.py rather than reimplementing them. Writes r33_-prefixed outputs so R31's existing report is not overwritten."
    status: completed
    note: "Built + smoke-tested 2026-09-22 via /build-step (script written and run to
      validate; the FIGURES STEP itself is left pending for /run-step per the skill's
      own rule not to mark steps entries here). Imports build_strips (which itself
      calls crop_arm_rows/extract_baseline_row/label_strip) and write_html from
      r31_html_report.py unchanged -- row-strip imagery never depended on the
      achievement axis, only the Pareto panel's x-VALUES do. --x_metric dispatches to
      three loader families: r30_score.load_r26_metric (stored CLIP columns),
      r33_axis_compare.load_clipd_metric (CLIP-D, method_fmt prefixed r26_/r31_ to
      match r33_clip_directional.csv's method column), and
      r33_axis_compare.load_fiveacc_metric (FiVE-Acc, against R33's own
      edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv files) -- both curve families
      (SPATIAL/UNIFORM) and R31's 4 single-point arms. Scope deliberately matches
      r31_html_report.py exactly (same 4 R31 arms, not R30's 8) -- this regenerates
      R31's existing report on a new axis, not a new R30+R31 combined one. write_html
      is reused verbatim, then its R31-titled header/title is string-patched to name
      R33 and the actual --x_metric used (2 sentinel strings, both confirmed present
      pre-patch and absent post-patch -- avoids duplicating the whole function for a
      2-line difference). Smoke test with --x_metric clip_d_prompt: clean run, 22
      grids + 22 pareto points, 0 stray 'R31' strings left in the HTML, 110 strip PNGs
      (22 x 5 rows). Visually spot-checked 0001_bus's panel: monotone-ish SPATIAL curve
      rising in CLIP-D with b, R31's 4 arms landing inside the achieved range, oracle
      frontier below everything -- a coherent, sane figure, not a fluke of the numbers."
  - id: tennis-note
    content: "Document the 0040_tennis EOT-truncation bug and exclude that clip from every headline mean. NO rescore, NO CSV patching (decided). Its prompts are 85/84 tokens; torchmetrics 1.9.0 clip_score.py:145-146 raw-slices input_ids[:77], dropping the EOT token CLIP pools its text embedding at, so the lookup falls through to position 0 (BOS) and the embedding stops depending on the prompt -- src_prompt and trg_prompt return a bit-identical 5.528 (correct truncation: 24.165/21.727). Only the three clip_similarity_* columns are affected; LPIPS/SSIM/PSNR/MSE/NIQE/structure_distance for that clip are fine. R33's own axes are immune: CLIP-D tokenises with truncation=True, FiVE-Acc is a VLM asking short questions."
    status: pending
  - id: aggregate-score-script
    content: "evaluation/r33_score.py -- ADDED 2026-09-22, post-verdict: the plan never scoped an AGGREGATE (mean-over-22-clip) CLIP-D-vs-LPIPS/SSIM trade-off figure analogous to r30_clip_vs_lpips.pdf / r31_clip_vs_lpips.pdf -- only the per-video panels (r33_report_figures.py) were built. Modeled on r30_score.py's and r31_score.py's draw_curve_scatter, merged into ONE script/figure pair covering ALL 28 CLIP-D arms together (user's explicit scope choice over a narrower R31-only remake): R26's SPATIAL curve + UNIFORM baseline + oracle frontier (unchanged, x swapped from clip_target to clip_d_prompt), R30's 8 arms as SQUARE markers (r30_score.py's convention, s=70), R31's 4 arms as STAR markers (r31_score.py's convention, s=260, visually distinct from R30's). x-values for every point come from evaluation/csv/r33_clip_directional.csv (clip_d_prompt column; method labels carry the r26_/r30_/r31_ prefix per the clip-d-script todo). y-values (lpips_unedit_part / ssim_unedit_part): R26's curves from R26's own per-edit-type CSVs (r30_score.load_r26_metric, unchanged); R30 arms from evaluation/csv/r30_arms.csv (already has per-arm mean lpips/ssim -- just re-pair with clip_d_prompt instead of its own stored clip_target column); R31 arms via r31_score.load_r31_percli_metric (unchanged). Writes evaluation/csv/r33_arms.csv (28 rows: arm, source{r26|r30|r31}, clip_d_prompt, lpips, ssim, n_clips) + evaluation/figures/r33_clip_vs_lpips.pdf + r33_clip_vs_ssim.pdf."
    status: completed
    note: "Built + smoke-tested 2026-09-22 via /build-step (the FIGURES STEP itself is
      left pending for /run-step per the skill's own rule). Also added load_r26_arms
      (R26's own 16 constant-b arms as rows in r33_arms.csv, not just curve points --
      needed since the CSV's own stated spec is '28 rows', all three sources on equal
      footing) alongside load_r30_arms/load_r31_arms, which the todo's content did not
      spell out as a separate function but follows directly from the 28-row CSV spec.
      R31 star markers colour-offset by +4 (cmap index) from R30 squares so an
      unrelated square/star pair is never drawn in the exact same tab10 colour.
      Smoke test: clean run, exactly 28/28 arms (8 r26_spatial + 8 r26_uniform + 8 r30
      + 4 r31, verified by Counter), all n_clips=22, R26 spatial numbers cross-checked
      against the verdict step's manual computation (bit-identical). Both PDFs visually
      inspected (converted to PNG via PyMuPDF, no pdftoppm on this node): SPATIAL curve
      rises monotonically in clip_d_prompt as expected, UNIFORM baseline sits above it,
      oracle frontier dominates below, and R30/R31 arms cluster near R26's b=3-6 region
      -- consistent with the verdict step's 'all four R31 arms tied' finding, now shown
      for all 28 arms at once. One new arm visible for the first time in this
      combined view: r30_dino_cls sits noticeably worse (higher LPIPS at its CLIP-D)
      than every other arm on both panels -- not previously singled out in R30's or
      R33's own text, worth a look if R30's arm-selection story is revisited."

steps:
  - id: clip-d
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r33_clip_directional.sh
    sets_status: running
    output_paths:
      - evaluation/csv/r33_clip_directional.csv
    status: completed
    completed_at: 2026-09-21
    job_id: "1002662"

  - id: wait-clip-d
    type: manual
    wait_for: clip-d
    check_hint: |
      sacct -j {job_id} --format=JobID,State,Elapsed
      # 28 arms x 22 clips = 616 rows expected
      python -c "import csv;r=list(csv.DictReader(open('evaluation/csv/r33_clip_directional.csv')));print(len(r),'rows',len({x['method'] for x in r}),'methods')"
      # ⚠️ degeneracy check: if delta_img_norm is near 0 at low b, CLIP-D's direction
      # there is undefined and those points must not be read as real.
      grep -c MISSING logs/r33_clipd_{job_id}.out
    status: completed
    completed_at: 2026-09-21
    note: "job 1002662 (A100/node56) COMPLETED. 616 rows, 28 methods, 0 MISSING lines.
      delta_img_norm guard: 0/616 rows below 0.05 (smallest 0.4396) -- no degenerate
      low-b readings. Took three submissions to get here: 1002579/1002591 both failed
      instantly on node57 (L40S) with /projects/dataggen missing; user independently hit
      the same gap on node51; treated as an L40S-partition-wide mount regression (2/4
      L40S nodes bad in one sitting) rather than a single flaky node, so
      r33_clip_directional.sh was narrowed from --partition=L40S,A100 to A100-only
      (dated comment left in the script; revisit once L40S's mount is confirmed fixed)."

  - id: fiveacc
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r33_fiveacc.sh
    wait_for: clip-d
    sets_status: running
    output_paths:
      - evaluation/csv/r33_fiveacc_arms.csv
    status: completed
    completed_at: 2026-09-21
    job_id: "1002751,1002959,1004302"
    note: "Five submissions total, root cause now fixed at the source. Three retried
      against /projects/dataggen with a growing node exclude list (see below); each
      round the previously-excluded node's slot got absorbed by a NEW idle node with
      the same stale mount (node57/node51 -> node01 -> node02), while nodes already
      holding warm jobs (node55, node56, interactive V100/node12) saw it fine --
      exclude-and-retry was not converging. User pointed to a local copy at
      ~/Data/dataggen (home dir, reliably mounted everywhere, unlike the /projects
      share) -- R26_ROOT/R31_ROOT in r33_fiveacc.sh repointed there. Resubmitted the
      17 still-failed indices as job 1004302; first 4 tasks landed on 4 DIFFERENT
      nodes (node55, node03, node04, node05) and all passed the 22-real-frame-dir
      check this time, one already loading Qwen2.5-VL-7B for real five_acc scoring --
      confirms the fix. Full breakdown of the /projects/dataggen chase below.
      (1) job 1002742, --partition=L40S,A100: cancelled after its first 8 tasks (0-7)
      all landed on node57 and failed the missing-/projects/dataggen guard, same as
      clip-d's earlier failures. (2) job 1002751, narrowed to A100-only (mirroring
      clip-d's fix): 19/20 tasks landed on node01 and failed the SAME guard; only the
      one task on node55 succeeded (taubg0_taufg20_vp) -- proved the A100-only
      restriction was not the fix, since node01 is just as bad as node51/node57.
      (3) job 1002959: reopened --partition=L40S,A100 and instead excluded the four
      then-confirmed-bad nodes by name (node01, node51, node52, node57); resubmitted
      only the 19 still-failed array indices (--array=0-5,7-19), skipping the one that
      already succeeded -- 17/19 landed on a FIFTH bad node (node02) and failed the
      same guard, 2 succeeded (node05, node55). Root cause across all three: a
      shifting set of compute nodes cannot see /projects/dataggen at job start while
      others (node55, node56, an interactive V100/node12 session) see it fine --
      exclude-and-retry was chasing a moving target, not converging. (4) job 1004302:
      switched data root to ~/Data/dataggen (see above) instead of excluding more
      nodes; resubmitted the 17 still-failed indices -- ALL 17 SUCCEEDED, spread
      across node55/node03/node04/node05/node39. Combined with the 3 that already
      succeeded earlier (taubg0_taufg2_vp, taubg0_taufg8_vp, taubg0_taufg20_vp), all
      20/20 arms done."

  - id: wait-fiveacc
    type: manual
    wait_for: fiveacc
    check_hint: |
      sacct -j {job_id} --format=JobID,State,Elapsed | head -25
      # 20 array tasks, one per arm; each writes per-edit-type CSVs
      ls evaluation/csv/edit*_FiVE_r33_fiveacc_*_frame_stride8.csv | wc -l
      # ⚠️ evaluate.py swallows per-metric exceptions and still exits 0 -- a clean exit
      # code is NOT evidence the table is clean. Any 'Error:' line means a dropped metric.
      grep -c 'Error:' logs/r33_fiveacc_{job_id}_*.out
    sets_status: finished
    status: completed
    completed_at: 2026-09-22
    note: "All 20/20 arms COMPLETED across the three eventually-successful submissions
      (1002751 task 6; 1002959 tasks 0, 4; 1004302 all 17 remaining). Verified
      directly (sacct unreachable -- 'Connection refused' to slurmdbd from this
      session, same as earlier in R33; used log/CSV evidence instead): 120/120
      per-edit-type CSVs (ls evaluation/csv/edit*_FiVE_r33_fiveacc_*_frame_stride8.csv,
      20 arms x 6 edit types), 0 'Error:' lines summed across every *.metrics.log from
      all three job ids, 20/20 r33_fiveacc_*_avg.csv files present. Finished far under
      the 6h budget -- five_acc alone (Qwen2.5-VL-7B, a handful of yes/no questions per
      22-clip arm) is cheap next to R26/R31's usual 9/16-metric multi-model passes."

  - id: axis-compare
    type: local
    wait_for: wait-fiveacc
    command: python evaluation/r33_axis_compare.py
    output_paths:
      - evaluation/csv/r33_axis_comparison.csv
    status: completed
    completed_at: 2026-09-22
    note: "Ran clean, exit 0. See axis-compare-script todo's note for the full
      methodology and results table. Headline: clip_d_prompt (+0.4080, 17/22) and
      clip_d_word (+0.3258, 15/22) both clear the incumbent clip_similarity_target_image
      (+0.1145, 11/21) by a wide margin and are defined for every clip; yn_acc/mc_acc
      score higher where defined (+0.71, +0.46) but are flat (undefined Spearman) on
      16/22 and 18/22 clips respectively, matching the plan's predicted FiVE-Acc
      coarseness. Axis decision deferred to the verdict step, per plan -- this step
      only produces the comparison table."

  - id: figures
    type: local
    wait_for: axis-compare
    command: python evaluation/r33_report_figures.py --x_metric clip_d_prompt
    output_paths:
      - evaluation/figures/r33_video_pareto
      - evaluation/figures/r33_report.html
    status: completed
    completed_at: 2026-09-22
    note: "Ran clean, exit 0: 22 grids, 22 pareto points, 0 stray 'R31' strings in the
      HTML. Already built + smoke-tested during /build-step; this run is the formal
      /run-step execution (idempotent, same script/flags, same outputs). See the
      figures-script todo's note for the full implementation details and the
      0001_bus visual spot-check."

  - id: verdict
    type: manual
    wait_for: figures
    check_hint: |
      Read evaluation/csv/r33_axis_comparison.csv. For each axis: mean Spearman(b, axis)
      and #clips with rho > 0 (incumbent raw CLIP = +0.098, 11/22 -- the bar to clear).
      (1) Does any axis rank edit strength reliably? An axis near 0.1 has FAILED the same
          way the incumbent did, and a high mean with a low clip count is not a pass.
      (2) Record WHICH axis the paper uses, and why.
      (3) Re-read R31's "clean negative result" verdict on that axis in
          .claude/plans/r31-spatial-divergence-routing_5e1c7a94.plan.md -- amend or
          confirm it. That verdict was read off the axis R33 is testing, so it cannot
          stand unexamined either way.
      ⚠️ If CLIP-D ALSO lands near zero, the problem is not the metric family and the
      conclusion is that this 22-clip set cannot resolve edit strength at all -- which is
      a finding about the experiment, not about R31's arms.
    sets_status: analyzed
    status: completed
    completed_at: 2026-09-22
    note: "(1) Does any axis rank reliably? YES: clip_d_prompt (+0.4080 mean, 17/22
      positive, 0 flat clips) and clip_d_word (+0.3258, 15/22) both clear the
      incumbent's +0.1145 (11/21, tennis excluded, matches the plan's original +0.098
      closely) by a wide margin. yn_acc/mc_acc score HIGHER where defined (+0.7113,
      +0.4638) but are flat/undefined on 16/22 and 18/22 clips -- a high mean with a
      tiny denominator is not a pass, per the check_hint's own warning; too coarse to
      use as a continuous Pareto x-axis, exactly as the plan's Decisions table
      predicted. (2) AXIS CHOSEN: clip_d_prompt. Highest mean Spearman AND highest
      clip count of any candidate; defined for all 22 clips (0 flat, unlike FiVE-Acc);
      immune to the 0040_tennis EOT-truncation bug by construction (truncation=True);
      scored for all 28 arms across R26/R30/R31 on one shared footing;
      clip_d_word was a close second but more separable-by-construction word pairs are
      a slightly narrower signal than full prompt pairs, and the plan's own
      clip-d-script todo positions prompt as primary, word as the free second column.
      (3) R31'S VERDICT RE-READ AND AMENDED (not confirmed as-is): re-scored R31's 4
      arms against R26's SPATIAL curve interpolated in clip_d_prompt instead of
      clip_target. All 4 arms now land within +/-0.0013 LPIPS of the curve (TIED) --
      lpips and dino_patch, previously called 'strictly dominated', were only
      worse-looking because clip_target (Spearman +0.098 with b) placed them at the
      WRONG x-position, not because they were actually worse-edited-for-their-LPIPS.
      Top-line conclusion UNCHANGED (no R31 arm Pareto-improves on R26's curve) but
      SEVERITY weakens from '2 dominated + 2 tied' to '4 tied' -- a more defensible,
      not reversed, negative result. Full table + amendment appended to
      .claude/plans/r31-spatial-divergence-routing_5e1c7a94.plan.md's Verdict section,
      original text left intact per the plan's own no-rescore/no-CSV-patch convention
      for amendments. CLIP-D did NOT land near zero, so this is a metric-family fix,
      not a finding that the 22-clip set is unresolvable."

  - id: aggregate-figures
    type: local
    wait_for: verdict
    command: python evaluation/r33_score.py -o evaluation/csv/r33_arms.csv --fig_dir evaluation/figures
    output_paths:
      - evaluation/csv/r33_arms.csv
      - evaluation/figures/r33_clip_vs_lpips.pdf
      - evaluation/figures/r33_clip_vs_ssim.pdf
    status: completed
    completed_at: 2026-09-22
    note: "Ran clean, exit 0: 28/28 arms. Already built + smoke-tested during
      /build-step (see the aggregate-score-script todo's note for full
      implementation details, the visual spot-check, and the r30_dino_cls outlier
      finding); this run is the formal /run-step execution (idempotent, same
      script/flags, same outputs)."
isProject: true
---

# R33: Edit-Alignment Metric Validation

## Context

FiVE-Bench's `clip_similarity_target_image` is the achievement axis every R26/R30/R31
trade-off figure is plotted against, and on this case set it does not rank renders by edit
strength. Measured per clip over R26's own 8-`b` SPATIAL sweep: `Spearman(b, clip_target)`
= **+0.098** mean, positive on only **11/22** clips — a coin flip — while
`Spearman(b, niqe_target)` = −0.530 (quality *improves* with `b`, on 18/22) and
`Spearman(niqe, clip)` = −0.093 (CLIP tracks quality no better than it tracks `b`).

The cause is measured, not assumed: `cos(E_txt(src_prompt), E_txt(trg_prompt))` = **0.846**
mean, up to 0.970. Source and target prompts differ by one noun inside a ~35-token scene
description, and CLIP is source-agnostic — a regenerated tree is still a tree — so the
shared majority is satisfied identically at every `b`. It dominates the embedding while
contributing no gradient. The confound is **render-level, not frame-level**, so the
harness's averaging over strided frames does not reduce it.

This is a metric-design problem, not a harness bug: CLIPScore was reimplemented
independently and reproduces the stored `clip_similarity_source_image` exactly (6/8 probe
clips to 0.000).

**Done when:** CLIP-D and FiVE-Acc CSVs exist for all arms; `Spearman(b, axis)` is reported
for every candidate axis on the same footing; a decision is recorded on which axis the paper
uses; R31's verdict is re-read on it; the per-video Pareto panels and HTML report are
regenerated on the chosen axis; and the aggregate (mean-over-22-clip) CLIP-D-vs-LPIPS/SSIM
trade-off figures exist for all 28 arms, superseding `r30_clip_vs_lpips.pdf` /
`r31_clip_vs_lpips.pdf` on the corrected axis.

## Execution steps

| # | Step id | Type | What | wait_for | sets_status |
|---|---------|------|------|----------|-------------|
| — | *(prep)* | — | `/build-step` the six `todos` — scripts do not all exist yet | — | `implemented` |
| 1 | `clip-d` | sbatch | Directional CLIP over 28 arms (R26 16 + R30 8 + R31 4) | — | `running` |
| 2 | `wait-clip-d` | manual | 616 rows, 28 methods, no MISSING; check `delta_img_norm` at low `b` | `clip-d` | — |
| 3 | `fiveacc` | sbatch | FiVE-Acc array, 20 arms × 22 clips = 440 videos | `clip-d` | `running` |
| 4 | `wait-fiveacc` | manual | 20 tasks COMPLETED, zero `Error:` lines | `fiveacc` | `finished` |
| 5 | `axis-compare` | local | `Spearman(b, axis)` per candidate axis, incumbent included | `wait-fiveacc` | — |
| 6 | `figures` | local | Per-video Pareto panels + HTML on the chosen axis | `axis-compare` | — |
| 7 | `verdict` | manual | Record the axis decision; re-read R31's verdict on it | `figures` | `analyzed` |
| 8 | `aggregate-figures` | local | Aggregate CLIP-D-vs-LPIPS/SSIM figures, all 28 arms, superseding `r30_`/`r31_clip_vs_*.pdf` | `verdict` | — |

```
/run-step R33 clip-d
/run-step R33 wait-clip-d
/run-step R33 fiveacc
/run-step R33 wait-fiveacc
/run-step R33 axis-compare
/run-step R33 figures
/run-step R33 verdict
/run-step R33 aggregate-figures
```

## Decisions

| Question | Choice |
|---|---|
| CLIP-D scope | **28 arms**: R26's 16 constant-`b` (8 spatial `taubg0_taufg{b}` + 8 uniform `taubg{b}_taufg{b}`) + R30's 8 routed arms + R31's 4 divergence arms. R31's `lin04` branch excluded. |
| FiVE-Acc scope | **Full**: R26's 16 + R31's 4 = 20 arms × 22 clips = **440 videos**. R30 already has `r30_fiveacc_arms.csv` and is not re-run. |
| Why R30's FiVE-Acc cannot substitute | Its 8 arms each use a **per-clip routed `b`** (161 distinct values over 176 rows), so it contains no constant-`b` arm and cannot draw a reference curve. |
| `0040_tennis` EOT bug | **Document + exclude from aggregates.** No rescore, no CSV patch. See the `tennis-note` todo for the mechanism. |
| CLIP-D formula | `cos(E_img(render) − E_img(src), E_txt(trg_prompt) − E_txt(src_prompt))`, embeddings L2-normalised **before** the subtraction. |
| CLIP-D sign | **Kept, not clamped.** Absolute CLIPScore uses `max(cos, 0)`; here a negative value means the edit moved the wrong way, which is the most informative thing an over-released arm can report. |
| CLIP-D second column | `clip_d_word` from `src_word`→`trg_word` alongside `clip_d_prompt`. The word pair is more separable (mean cos 0.71 vs 0.85) and costs nothing. |
| Degeneracy guard | `delta_img_norm` column: at low `b` the render barely differs from the source, so `Δ_img` is short and its direction ill-conditioned. Must be inspected before trusting the low-`b` end. |
| `clip-d` job shape | **Single task, not arrayed.** The script loops clips OUTER / methods INNER, so each clip's source frames are embedded once (211 total) and reused by all 28 methods: 6,119 embeds for the whole job. Arraying by arm re-embeds the sources in every task → 11,816 embeds, **+93% GPU**, plus 28 model loads, to parallelise a job that needs one GPU for well under an hour. Documented fallback if it ever runs long: split by task group (R26-spatial / R26-uniform / R30 / R31, `--array=0-3`), which holds source redundancy at 4×. |
| `frame_stride` | **8**, matching every R26/R30/R31 eval. CLIP-D is only comparable with the stored LPIPS/SSIM/CLIP columns if it describes the same frames. |
| CSV reading | Per-edit-type per-clip files (`edit{T}_FiVE_{stem}_frame_stride8.csv`). **Never** the top-level `{stem}_avg.csv` — evaluate.py's own averaging step is broken (off by ~100× on some columns; see R31's plan). |
| FiVE-Acc isolation | Run with `--metrics five_acc` **alone**. R20's GPU exhaustion was Qwen2.5-VL-7B + CoTracker co-residency, not five_acc's own cost. |
| FiVE-Acc coarseness | ⚠️ Expected to tie: 6 of R30's 8 arms share `mc_acc` = 0.818 (18/22 clips). At n=22 it is an accuracy, so it yields vertical bands, **not** a continuous Pareto x-axis. |
| Conda env | `five-bench` (not `streamgve` — the R2 env bug: `.bashrc` auto-activates `streamgve` and `conda activate five-bench` alone does not pop it). |
| Naming | All task files take the `r33_` prefix. The two drafts written during R31's debugging (`r31_clip_directional.{py,sh}`) are **renamed**, not copied. |
| Out of scope | Re-running the 9-metric harness; `motion_fidelity_score`; R31's `lin04` branch; any change to R31's renders. |

## Step commands

### clip-d

```bash
sbatch slurm_scripts/five_bench/r33_clip_directional.sh
```

### wait-clip-d

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
python -c "import csv;r=list(csv.DictReader(open('evaluation/csv/r33_clip_directional.csv')));print(len(r),'rows',len({x['method'] for x in r}),'methods')"
grep -c MISSING logs/r33_clipd_{job_id}.out
```

### fiveacc

```bash
sbatch slurm_scripts/five_bench/r33_fiveacc.sh
```

### wait-fiveacc

```bash
sacct -j {job_id} --format=JobID,State,Elapsed | head -25
ls evaluation/csv/edit*_FiVE_r33_fiveacc_*_frame_stride8.csv | wc -l
grep -c 'Error:' logs/r33_fiveacc_{job_id}_*.out
```

### axis-compare

```bash
python evaluation/r33_axis_compare.py
```

### figures

```bash
python evaluation/r33_report_figures.py --x_metric clip_d_prompt
```

### verdict

```bash
column -s, -t < evaluation/csv/r33_axis_comparison.csv
```

### aggregate-figures

```bash
python evaluation/r33_score.py -o evaluation/csv/r33_arms.csv --fig_dir evaluation/figures
```

## Pipeline

```mermaid
flowchart LR
  R26[(r26_spatial_tau\n16 constant-b arms)] --> CD[r33_clip_directional.py]
  R30[(r30_arms\n8 routed arms)] --> CD
  R31[(r31_arms\n4 divergence arms)] --> CD
  SRC[(FiVE source frames\n+ src/trg prompts)] --> CD
  CD --> CDCSV[/r33_clip_directional.csv/]

  R26 --> FA[r33_fiveacc.sh\nevaluate.py --metrics five_acc]
  R31 --> FA
  FA --> FACSV[/r33_fiveacc_arms.csv/]

  CDCSV --> CMP[r33_axis_compare.py]
  FACSV --> CMP
  OLD[(stored clip_target\nand clip_edit_part)] --> CMP
  CMP --> AX[/r33_axis_comparison.csv/]

  AX --> FIG[r33_report_figures.py]
  CDCSV --> FIG
  FIG --> HTML[/r33_report.html\n+ r33_video_pareto//]
  AX --> V{verdict\naxis decision +\nR31 re-read}

  V --> AGG[r33_score.py]
  CDCSV --> AGG
  R26 --> AGG
  R30 --> AGG
  R31 --> AGG
  AGG --> AGGCSV[/r33_arms.csv/]
  AGG --> AGGFIG[/r33_clip_vs_lpips.pdf\n+ r33_clip_vs_ssim.pdf/]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r31_clip_directional.py` | **Rename** to `evaluation/r33_clip_directional.py` (`git mv`). Already written and sanity-checked: reverse direction scores exactly −1.0, two distinct edits are orthogonal (−0.018), source-vs-itself returns `nan`. |
| `evaluation/r33_clip_directional.py` | Add `--r30_root` (default `/projects/dataggen/outputs/five_bench/r30_arms`) and extend `method_specs()` with R30's 8 arms. Add a `delta_img_norm` column: mean over paired frames of `‖E_img(render) − E_img(src)‖` measured **before** the per-frame normalisation in `clip_d()`, so the degeneracy guard reflects the real magnitude. |
| `slurm_scripts/five_bench/r31_clip_directional.sh` | **Rename** to `r33_clip_directional.sh`; update `--which`/root flags for the 28-arm scope, the output path to `r33_clip_directional.csv`, the log stem to `r33_clipd_%j`, and the row assertion from 440 to **616** (28 × 22). Keep the existing guard that fails loudly when `/projects/dataggen` is not mounted on the assigned node. |
| `slurm_scripts/five_bench/r33_fiveacc.sh` | **New.** `#SBATCH --array=0-19`, partition `L40S,A100`, `--exclude=node52`, 1 GPU, `--mem=64G`, `--time=06:00:00`. A bash array maps task id → `(stem, full_dir)`; R26 entries are `r26_spatial_tau/{arm}/`, R31 entries are `r31_arms/r31_{arm}/step14/`. Calls `evaluate.py --metrics five_acc --tgt_layout edit_video --result_path evaluation/csv/r33_fiveacc_{stem}.csv`. Reuse r31_eval.sh's post-run guards verbatim: assert 22 real frame dirs (excluding the `_resize` siblings evaluate.py writes), and `grep -c 'Error:'` the teed metrics log, since evaluate.py swallows per-metric exceptions and still exits 0. |
| `evaluation/r33_axis_compare.py` | **New.** Reuses `r30_score.build_join_tables` / `load_r26_metric` for the joins. For each axis in {`clip_similarity_target_image`, `clip_similarity_target_image_edit_part`, `clip_d_prompt`, `clip_d_word`, `yn_acc`, `mc_acc`}: per-clip `Spearman(b, axis)` over `CONST_BS`, then mean and `#rho > 0`. Drops `0040_tennis` from the CLIP-family rows only, and states the exclusion in the output. Writes `evaluation/csv/r33_axis_comparison.csv`. |
| `evaluation/r33_report_figures.py` | **New.** `--x_metric` selects the achievement axis; imports `crop_arm_rows`, `extract_baseline_row`, `label_strip` from `evaluation/r31_html_report.py` instead of duplicating the strip logic. Writes `evaluation/figures/r33_video_pareto/` and `evaluation/figures/r33_report.html`. |
| `evaluation/r31_html_report.py` | Export the three strip helpers unchanged (they are already module-level functions — no edit expected; confirm no import-time side effects before relying on it). |
| `evaluation/r33_score.py` | **New**, added 2026-09-22 post-verdict. Aggregate (mean-over-22-clip) CLIP-D-vs-LPIPS/SSIM trade-off figures for **all 28 CLIP-D arms**, superseding `r30_clip_vs_lpips.pdf`/`r31_clip_vs_lpips.pdf` on the corrected axis (the plan originally scoped only the per-video panels in `r33_report_figures.py`, not this aggregate figure). `draw_curve_scatter` merged from `r30_score.py` and `r31_score.py`: R26 SPATIAL curve + UNIFORM baseline + oracle frontier unchanged (x swapped `clip_target` → `clip_d_prompt`); R30's 8 arms as SQUARE markers (`r30_score.py`'s convention); R31's 4 arms as STAR markers (`r31_score.py`'s convention). x-values from `r33_clip_directional.csv`'s `clip_d_prompt` column (via `r33_axis_compare.load_clipd_metric`); y-values (`lpips_unedit_part`/`ssim_unedit_part`) from R26's own per-edit-type CSVs for the curves, `evaluation/csv/r30_arms.csv` for R30's 8 arms, and `r31_score.load_r31_percli_metric` for R31's 4 — none of these y-sources change, only the x pairing does. Writes `evaluation/csv/r33_arms.csv` (28 rows: `arm, source, clip_d_prompt, lpips, ssim, n_clips`) + `evaluation/figures/r33_clip_vs_lpips.pdf` + `r33_clip_vs_ssim.pdf`. |
