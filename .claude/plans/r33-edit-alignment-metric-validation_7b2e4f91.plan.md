---
name: "R33: Edit-Alignment Metric Validation"
overview: "Replace FiVE-Bench's inherited whole-frame CLIP achievement axis, which does not rank renders by edit strength on this case set (Spearman(b, clip_target) = +0.098, positive on 11/22 clips), with two axes that do: directional CLIP over R26/R30/R31's renders, and comprehensive FiVE-Acc for R26/R31."
task_id: R33
report_url: "https://claude.ai/artifact/PuAmCbhopy1r8fwjYCnMkG"
video_report_url: "https://claude.ai/artifact/79dK2mNkHDzC8BPyDRPMcq"
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
  - id: fiveacc-aggregate-script
    content: "ADDED 2026-09-22, after aggregate-figures: the same aggregate trade-off pair as r33_score.py already draws, but on the FiVE-Acc achievement axis instead of CLIP-D -- r33_fiveacc_vs_lpips.pdf / r33_fiveacc_vs_ssim.pdf (yn_acc) plus an r33_fiveacc_mc_* pair (mc_acc). No new GPU work: every input is already on disk from this task's own fiveacc step (R26's 16 + R31's 4) and from r30_fiveacc_arms.csv (R30's 8). evaluation/r33_score.py is already parametrized with --x_metric / --fig_stem, so this WIDENS existing dispatch rather than adding structure: add yn_acc/mc_acc to the choices and to XSHORT/XLABEL/XNOTE, give _x_r26 and load_r31_arms a FiVE-Acc branch, and leave load_r30_arms alone (it already falls through to the else branch that reads the column by name). Also promotes r34_score.load_stem_metric into r30_score.py so both this script and r31_score.py can read the single-arm FiVE-Acc CSVs through the shared base. See Code to touch for the column-name-differs-by-source trap."
    status: completed
    note: "BUILT 2026-09-22 and smoke-tested. Exactly the five specified edits, no new
      structure: XM_FIVEACC label->column map + yn_acc/mc_acc added to XSHORT/XNOTE/XLABEL
      and to the --x_metric choices; a FiVE-Acc branch in _x_r26 (calling
      load_fiveacc_metric with the BARE method_fmt -- the 'r26_' prefix the CLIP-D branch
      prepends would have been wrong, since that loader builds the r33_fiveacc_{method}
      stem itself); an elif branch in load_r31_arms calling the promoted load_stem_metric
      with stem=r33_fiveacc_r31_{arm}, stride=8. load_r30_arms confirmed untouched -- it
      really does fall through to the existing else at line 115 and read the literal
      yn_acc / mc_acc columns of r30_fiveacc_arms.csv.
      LOADER PROMOTION DONE: load_stem_metric now lives in r30_score.py; r34_score.py
      re-exports it from there (verified `r34_score.load_stem_metric is
      r30_score.load_stem_metric` -> True, and r34_report_figures still imports cleanly).
      Re-verified after the move: 22 clips / mean 0.7273 on five_acc_yes_no, 22 / 0.7727
      on five_acc_multi_choice, and bit-identical to load_r31_percli_metric on LPIPS --
      the same three checks the plan recorded before the move.
      ✅ REGRESSION CLEAN: re-running the default `r33_score.py -o r33_arms.csv` after the
      edit reproduces r33_arms.csv byte-for-byte (diff empty), so aggregate-figures is
      unaffected.
      SMOKE TEST: both new axes ran to /tmp, exit 0, 28/28 arms x 22 clips on all three
      sources. The R26 columns reproduce the plan's independently-measured values EXACTLY
      -- yn_acc SPATIAL 0.545/0.636/0.727x3/0.773/0.818x2 and UNIFORM 0.500/0.682/0.727x2/
      0.773/0.818x2/0.864; mc_acc SPATIAL saturating at 0.818 from b=4 and UNIFORM
      non-monotone (0.727 0.773 0.818 0.773 0.864 0.818 0.818 0.818). One panel rendered
      and visually checked: curves/markers/frontier all correct and the parametrized
      alpha=1 annotation reads '(FiVE-Acc(yes/no))'.
      ⚠️ The quantisation caveat in Decisions is worse than predicted for R31: ALL FOUR
      R31 arms sit at exactly yn_acc=0.727273, so their four stars overplot at a single x
      and the arms are separated only on y. Worth stating in the write-up rather than
      leaving a reader to infer a ranking that is not there."

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
      only produces the comparison table.
      SUPERSEDED on disk by `axis-compare-fiveacc` below: `AXES` gained a 7th entry
      (`five_acc`) and this same command was re-run, overwriting
      `r33_axis_comparison.csv` in place. This entry is the historical record of the
      original 6-axis run."

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
      0001_bus visual spot-check.
      SUPERSEDED on disk by the `video-pareto-split` step below: the 22 PNGs this step
      wrote flat into `r33_video_pareto/` now live under `r33_video_pareto/clipd/`
      instead, and this step's own `r33_report.html` was regenerated in place (same
      command, `--pareto_out` pointed at the new subfolder) to match. This entry is left
      as the historical record of the original run; it does not reflect today's file
      layout."

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
      not a finding that the 22-clip set is unresolvable.
      RE-CONFIRMED, not superseded, by `verdict-fiveacc-reread` below: a 7th axis
      (`five_acc`) was added to `r33_axis_comparison.csv` after this step ran, and the
      choice recorded here still holds against it -- see that step for the numbers."

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

  - id: fiveacc-figures
    type: local
    wait_for: aggregate-figures
    command: |
      python evaluation/r33_score.py --x_metric yn_acc \
        --fig_stem r33_fiveacc -o evaluation/csv/r33_arms_fiveacc.csv \
        --fig_dir evaluation/figures
      python evaluation/r33_score.py --x_metric mc_acc \
        --fig_stem r33_fiveacc_mc -o evaluation/csv/r33_arms_fiveacc_mc.csv \
        --fig_dir evaluation/figures
    output_paths:
      - evaluation/csv/r33_arms_fiveacc.csv
      - evaluation/csv/r33_arms_fiveacc_mc.csv
      - evaluation/figures/r33_fiveacc_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_vs_ssim.pdf
      - evaluation/figures/r33_fiveacc_mc_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_mc_vs_ssim.pdf
    status: completed
    completed_at: 2026-09-22
    note: "Both invocations exit 0, 28/28 arms x 22 clips, all 6 output paths on disk.
      Outputs are byte-identical to the /build-step smoke run, and r33_arms.csv (the
      CLIP-D table) is confirmed unchanged -- the distinct -o / --fig_stem did their job.
      yn_acc PANELS ARE SOUND: SPATIAL curve monotone 0.545 -> 0.818, UNIFORM 0.500 ->
      0.864, oracle frontier above both, R30 squares / R31 stars placed as on the CLIP-D
      panels. Both LPIPS and SSIM panels visually checked.
      ⚠️ mc_acc PANELS ARE A NULL RESULT, and should be read as one -- the figure is
      visibly degenerate, not merely tied: all 28 arms collapse onto THREE distinct x
      values (0.727 / 0.818 / 0.864), the spatial curve is a near-vertical stack at
      0.818, and the uniform baseline runs BACKWARDS between b=8 (0.864) and b=10
      (0.818), so the dashed line crosses itself. The XNOTE title states this on the
      figure itself. Keep them as the robustness variant they were commissioned as; do
      not put them next to the yn_acc pair as if they were a second measurement.
      ⚠️ Also true on the yn_acc panels: ALL FOUR R31 arms sit at exactly 0.727273, so
      their stars overplot at one x and differ only in y. No achievement ranking exists
      among R31's arms on this axis.
      ⚠️ NAMING SUPERSEDED by `fiveacc-figures-relabel` below: the bare
      `r33_fiveacc_vs_*.pdf`/`r33_arms_fiveacc.csv` this step wrote were actually the
      yn_acc run's output -- misleading, since 'FiVE-Acc' names the whole 4-column
      family (yn/mc/union/inter), not just its yes/no column. That data now lives under
      `r33_fiveacc_yn_vs_*.pdf`/`r33_arms_fiveacc_yn.csv`; the bare name was freed for a
      new combined `five_acc` axis. The mc_acc pair this step wrote
      (`r33_fiveacc_mc_vs_*.pdf`/`r33_arms_fiveacc_mc.csv`) was correctly named already
      and is unchanged."

  - id: video-pareto-split
    type: local
    wait_for: figures
    command: python evaluation/r33_report_figures.py --x_metric clip_d_prompt --pareto_out evaluation/figures/r33_video_pareto/clipd
    output_paths:
      - evaluation/figures/r33_video_pareto/clipd
      - evaluation/figures/r33_report.html
    status: completed
    completed_at: 2026-09-22
    note: "Per-video panels are now split one subfolder per achievement axis, at the
      user's request, mirroring the CLIP-D per-video treatment for the FiVE-Acc one added
      right after (video-pareto-fiveacc). Re-ran the SAME command the `figures` step used,
      only `--pareto_out` changed (`r33_video_pareto` -> `r33_video_pareto/clipd`) --
      deliberately a re-run, not a manual `mv` + hand-patched HTML, because
      `r31_html_report.write_html` computes every `<img src>` as
      `p.relative_to(root)` (root = `r33_report.html`'s own directory), so a re-run
      regenerates BOTH the PNGs and the correct nested references in one step, with no
      risk of an img src silently drifting from the file it names. Confirmed: 22/22 PNGs
      written under `clipd/`, `r33_report.html`'s 22 `<img src=\"r33_video_pareto/clipd/...\">`
      references all resolve, 0 remaining references to the old flat path. The 22
      original flat-folder PNGs (byte-identical content, same script/flags) were then
      deleted, leaving `clipd/` as the only copy."

  - id: video-pareto-fiveacc
    type: local
    wait_for: video-pareto-split
    command: python evaluation/r33_report_figures.py --x_metric yn_acc --pareto_out evaluation/figures/r33_video_pareto/fiveacc --out evaluation/figures/r33_report_fiveacc.html
    output_paths:
      - evaluation/figures/r33_video_pareto/fiveacc
      - evaluation/figures/r33_report_fiveacc.html
    status: completed
    completed_at: 2026-09-22
    note: "Per-video Pareto panels for FiVE-Acc, previously built only as the 28-arm
      AGGREGATE (fiveacc-figures) -- this is the per-clip analogue, one panel per video,
      same treatment CLIP-D already got via the `figures` step. SCOPE: `yn_acc` only, not
      `mc_acc` -- yn_acc is the Decisions table's primary FiVE-Acc axis and mc_acc is
      already established as a null result (fiveacc-figures' note), so a per-video mc_acc
      panel would add a second copy of a result already known to be uninformative rather
      than new evidence. `--out` pointed at a NEW file (`r33_report_fiveacc.html`),
      distinct from `r33_report.html`, so this run does not overwrite the CLIP-D report
      -- `--strips_out` left at its shared default since the row-strip imagery is
      axis-independent (unchanged from the figures-script todo's own reasoning).
      Ran clean, exit 0: 22 grids, 22 pareto points, 0 'R31' strings left in the HTML,
      22/22 PNGs under `fiveacc/`.
      ⚠️ REAL FINDING, not just plumbing: spot-checked 0001_bus's panel and found every
      one of its 8 `b` points collapsed onto a single x = 1.0 (yn_acc answers 'yes' at
      every b for this clip) -- a visibly degenerate vertical line, not a curve. Cross-checked
      against axis-compare's own count: 16/22 clips are flagged `n_flat` for yn_acc in
      `r33_axis_comparison.csv`, so 0001_bus is not a one-off -- most of these 22 per-video
      panels are expected to render as a single vertical stack rather than a spread. This
      is the per-clip mechanism BEHIND the aggregate coarseness already reported
      (fiveacc-figures' note, section 4.3 of the report): FiVE-Acc's 8-point vertical
      spread only appears once results are averaged over many clips; at the single-clip
      level, a yes/no question usually doesn't flip across the whole b-sweep. Not yet
      spot-checked which of the 6 non-flat clips looks like on this axis -- worth a look
      before using any single-clip FiVE-Acc panel as a qualitative example in the paper.
      SUPERSEDED on disk by the `report-dual` step below: `r33_report_fiveacc.html`, the
      standalone file this step wrote, was deleted once `report-dual` merged its content
      (same 22 `fiveacc/` PNGs, same data) into `r33_report.html` alongside CLIP-D. The
      22 PNGs under `fiveacc/` are unaffected and still exactly what this step produced."

  - id: report-dual
    type: local
    wait_for: video-pareto-fiveacc
    command: |
      python evaluation/r33_report_figures.py --x_metric clip_d_prompt --x_metric2 yn_acc \
        --pareto_out evaluation/figures/r33_video_pareto/clipd \
        --pareto_out2 evaluation/figures/r33_video_pareto/fiveacc \
        --out evaluation/figures/r33_report.html
    output_paths:
      - evaluation/figures/r33_report.html
    status: completed
    completed_at: 2026-09-22
    note: "User's request: r33_report.html has to show BOTH per-video CLIP-D and
      per-video FiVE-Acc, not one axis per file. Required a script change first (see
      the Code to touch row on `r33_report_figures.py`, 2026-09-22): a new
      `--x_metric2`/`--pareto_out2` pair and a `write_html_dual` function that puts two
      `<figure>` panels, each captioned with its own AXIS_LABEL, side by side per video
      section, instead of the existing single-panel `write_html`/sentinel-patch path.
      Ran clean, exit 0: 22 grids, 22+22 pareto points (`x_metric=clip_d_prompt
      x_metric2=yn_acc`). Verified: 22/22 sections in `r33_report.html` carry a
      `pareto-wrap dual` block with both a `r33_video_pareto/clipd/...` and a
      `r33_video_pareto/fiveacc/...` <img>, `<title>` reads 'R33 qualitative report',
      spot-checked 0001_bus's section renders both images with correct captions
      ('CLIP-D, prompt pair (achieved edit)' / 'FiVE-Acc yes/no accuracy (achieved
      edit)'). This run wrote into the SAME `clipd/`/`fiveacc/` subfolders
      `video-pareto-split`/`video-pareto-fiveacc` already populated (idempotent,
      byte-identical PNGs) -- only `r33_report.html`'s structure is new.
      `r33_report_fiveacc.html` (the standalone FiVE-Acc-only report from
      `video-pareto-fiveacc`) is now fully redundant with this merged file and was
      deleted to avoid two reports going stale independently.
      ⚠️ No `localhost:8899` file server was reachable from this session (`curl` to
      `http://localhost:8899/r33_report.html` got no response, no local listener on that
      port) -- the file on disk at `evaluation/figures/r33_report.html` is current and
      correct; if a server elsewhere serves that directory, it needs no restart for a
      static file, just a browser refresh, but this session could not confirm it directly.
      SUPERSEDED on disk by the `report-multi` step below: `write_html_dual` (fixed at
      exactly two panels) was generalised to `write_html_multi` (any number), and
      `r33_report.html` was regenerated again with four panels, not two. This entry is
      left as the historical record of the dual-axis run; `--x_metric2`/`--pareto_out2`,
      the flags this run used, no longer exist on the script (see report-multi's Code to
      touch row) -- use `--x_metric_more`/`--pareto_out_more` if replaying this command."

  - id: report-multi
    type: local
    wait_for: report-dual
    command: |
      python evaluation/r33_report_figures.py --x_metric clip_d_prompt \
        --pareto_out evaluation/figures/r33_video_pareto/clipd \
        --x_metric_more yn_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc \
        --x_metric_more mc_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_mc \
        --x_metric_more five_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_combined \
        --out evaluation/figures/r33_report.html
    output_paths:
      - evaluation/figures/r33_video_pareto/fiveacc_mc
      - evaluation/figures/r33_video_pareto/fiveacc_combined
      - evaluation/figures/r33_report.html
    status: completed
    completed_at: 2026-09-22
    note: "User's request: also want mc_acc per-video, plus a new combined axis
      five_acc = (yn_acc + mc_acc + union + inter) / 4, both added to r33_report.html
      alongside CLIP-D and yn_acc (4 panels per clip now, not 2).
      FIVE_ACC PROVENANCE VERIFIED FIRST, no new scoring needed: read
      evaluation/fivebench/evaluate.py directly (~line 519-527) rather than assuming --
      given yn_acc/mc_acc in {0,1} it computes union = int(yn_acc or mc_acc),
      inter = int(yn_acc and mc_acc), five_acc = mean([yn_acc, mc_acc, union, inter]),
      i.e. exactly (yn+mc+union+inter)/4 as the user specified. Already present as a raw
      five_acc column in every R26/R31 per-clip CSV this task's `fiveacc` step already
      produced (confirmed: edit1_FiVE_r33_fiveacc_taubg0_taufg2_vp_frame_stride8.csv row
      0 has yn=1,mc=1,union=1,inter=1,five_acc=1.0) -- R30's r30_fiveacc_arms.csv has
      union/inter but no precomputed five_acc column, moot here since this script never
      reads R30's per-video data.
      SCRIPT GENERALISED from 2 axes to N (see the r33_report_figures.py Code to touch
      row, 2nd pass): mc_acc was already a valid --x_metric (built for the aggregate
      fiveacc-figures step); five_acc is newly added to AXIS_COLUMN/AXIS_LABEL/AXIS_STEM
      and the two FiVE-Acc-family dispatch tuples in load_axis_curve/load_axis_r31_point
      (routes through the existing generic column-suffix-match loader, since evaluate.py
      already writes the five_acc column with the same *|<metric> shape as yn/mc). The
      single-pair --x_metric2/--pareto_out2 from report-dual was replaced by repeatable
      --x_metric_more/--pareto_out_more (paired positionally, count-checked), and
      write_html_dual by write_html_multi (N panels, not 2) -- see that row for why.
      Ran clean, exit 0: 22 grids, 22+22+22+22 pareto points
      (axes=[clip_d_prompt, yn_acc, mc_acc, five_acc]). Verified: 4 new/reused pareto
      subfolders (clipd/, fiveacc/, fiveacc_mc/, fiveacc_combined/, 22 PNGs each),
      22/22 sections in r33_report.html carry a pareto-wrap with FOUR <figure> panels in
      that order, correctly captioned; spot-checked 0001_bus's raw section HTML directly.
      ⚠️ REAL FINDING: five_acc is structurally at least as coarse as yn_acc alone, not
      finer -- with yn_acc/mc_acc each binary, union/inter are deterministic functions of
      them, so five_acc can only take 3 distinct per-clip values: 0, 0.5, or 1.0 (yn=mc=0
      -> 0; exactly one of yn/mc -> union=1,inter=0 -> 0.5; yn=mc=1 -> 1.0). Visually
      confirmed on both 0001_bus (flat at 1.0 across all 8 b, matching yn_acc's own
      flatness there) and 0069_car-turn (flat at 0.5). Out of scope for this step: this
      five_acc axis was NOT added to r33_axis_compare.py's AXES table, so it has no
      Spearman(b, axis) entry in r33_axis_comparison.csv and did not participate in the
      verdict step's axis choice -- only its per-video panel was requested."

  - id: fiveacc-figures-relabel
    type: local
    wait_for: report-multi
    command: |
      python evaluation/r33_score.py --x_metric yn_acc \
        --fig_stem r33_fiveacc_yn -o evaluation/csv/r33_arms_fiveacc_yn.csv \
        --fig_dir evaluation/figures
      python evaluation/r33_score.py --x_metric five_acc \
        --fig_stem r33_fiveacc -o evaluation/csv/r33_arms_fiveacc.csv \
        --fig_dir evaluation/figures
    output_paths:
      - evaluation/csv/r33_arms_fiveacc.csv
      - evaluation/csv/r33_arms_fiveacc_yn.csv
      - evaluation/figures/r33_fiveacc_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_vs_ssim.pdf
      - evaluation/figures/r33_fiveacc_yn_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_yn_vs_ssim.pdf
    status: completed
    completed_at: 2026-09-22
    note: "User's complaint, verified correct: `fiveacc-figures`' bare
      `r33_fiveacc_vs_lpips.pdf`/`_vs_ssim.pdf` were misleading -- they only ever
      plotted `yn_acc`, not FiVE-Acc's actual combined verdict, but the filename gave no
      hint only one of its four columns was used. Fix has two parts, both on the AGGREGATE
      (28-arm mean) figures -- the per-video panels report-multi just added were already
      correctly disambiguated (fiveacc/fiveacc_mc/fiveacc_combined subfolders).
      (1) SCRIPT: `r33_score.py` gained `five_acc` as a real `--x_metric` choice (see
      Code to touch) -- XM_FIVEACC/XSHORT/XNOTE/XLABEL extended, and `load_r30_arms`
      gained a dedicated `five_acc` branch computing `(yn_acc+mc_acc+union+inter)/4`
      per-clip from R30's existing `r30_fiveacc_arms.csv` columns (that file has no
      precomputed `five_acc` column, unlike R26/R31's CSVs where evaluate.py writes one
      directly -- confirmed by inspecting r30_fiveacc_arms.csv's header before writing
      the branch, not assumed). R26/R31 route through the already-generic
      column-suffix-match loader, same mechanism yn_acc/mc_acc already used.
      (2) RENAME: two runs. First re-ran the yn_acc pair under `--fig_stem
      r33_fiveacc_yn` / `-o r33_arms_fiveacc_yn.csv` (freeing the bare name). Then ran
      the NEW five_acc axis under the now-available bare `--fig_stem r33_fiveacc` /
      `-o r33_arms_fiveacc.csv` -- this run's output legitimately IS what the bare name
      should mean, so it correctly overwrites the old (mislabeled) file at that path
      rather than leaving an orphan. `r33_fiveacc_mc_vs_*`/`r33_arms_fiveacc_mc.csv`
      were already correctly named and untouched.
      Ran clean, both invocations exit 0, 28/28 arms each. Verified on disk: exactly
      three unambiguous pairs exist (`r33_fiveacc_vs_*` combined,
      `r33_fiveacc_yn_vs_*`, `r33_fiveacc_mc_vs_*`), no stale ambiguously-named file
      left over. Visually inspected the new combined figure (PyMuPDF, no pdftoppm on
      this node): title states the formula and the coarseness caveat on the figure
      itself, SPATIAL curve shape close to the yn_acc panel's (expected, since five_acc
      is yn_acc-dominated: union=yn_acc whenever mc_acc=0), values range ~0.61-0.84
      across the 28 arms, R30/R31 markers land in the same visual cluster as on every
      other R33 aggregate panel -- coherent, not a fluke."

  - id: axis-compare-fiveacc
    type: local
    wait_for: fiveacc-figures-relabel
    command: python evaluation/r33_axis_compare.py
    output_paths:
      - evaluation/csv/r33_axis_comparison.csv
    status: completed
    completed_at: 2026-09-22
    note: "User's request: add five_acc to the Spearman(b, axis) validation table too --
      it had only been added to the aggregate/per-video FIGURES so far, never to the
      per-clip axis-CHOICE test itself. One-line change: `AXES` in
      `r33_axis_compare.py` gained `(\"five_acc\", \"fiveacc\", \"five_acc\", True)`,
      routing through the same generic `load_fiveacc_metric` every other FiVE-Acc axis
      already uses (no new loader). Ran clean, exit 0, 7 axes now (was 6). RESULT:
      five_acc mean_rho=+0.5916, n_pos=6/7, n_flat=15/22 -- high mean, same coarseness
      failure mode as yn_acc (+0.7113, 16/22 flat) and mc_acc (+0.4638, 18/22 flat), one
      clip less flat than yn_acc's 16 (five_acc's 3rd achievable value per clip, 0.5,
      gives it marginally more resolution) but nowhere near clip_d_prompt's 0/22 flat.
      Does NOT change which axis clears the bar."

  - id: verdict-fiveacc-reread
    type: manual
    wait_for: axis-compare-fiveacc
    check_hint: |
      Re-read evaluation/csv/r33_axis_comparison.csv now that five_acc is a 7th row.
      Same test as the original verdict step: does it clear the incumbent AND avoid the
      high-mean-tiny-denominator trap the check_hint originally warned about?
    status: completed
    completed_at: 2026-09-22
    note: "AXIS CHOICE UNCHANGED: clip_d_prompt remains chosen. five_acc scores
      +0.5916 mean, higher than clip_d_prompt's +0.4080 -- but only on 7/22 clips
      (15 flat), the same failure mode the original verdict step already named for
      yn_acc/mc_acc: 'a high mean with a low clip count is not a pass.' This is the
      expected outcome, not a surprise -- five_acc is a deterministic function of
      yn_acc/mc_acc (see r33_report_figures.py's per-video note), so it inherits their
      coarseness rather than escaping it; its one extra achievable value (0.5, from
      union without inter) narrows the flat count from 16 to 15 but does not come close
      to clip_d_prompt's 0. No amendment needed to the original verdict's reasoning or
      to R31's re-read on clip_d_prompt -- this step exists to make that explicit for
      five_acc specifically, not to reopen the choice."

  - id: fiveacc-figures-titlewrap
    type: local
    wait_for: verdict-fiveacc-reread
    command: |
      python evaluation/r33_score.py -o evaluation/csv/r33_arms.csv --fig_dir evaluation/figures
      python evaluation/r33_score.py --x_metric yn_acc \
        --fig_stem r33_fiveacc_yn -o evaluation/csv/r33_arms_fiveacc_yn.csv \
        --fig_dir evaluation/figures
      python evaluation/r33_score.py --x_metric mc_acc \
        --fig_stem r33_fiveacc_mc -o evaluation/csv/r33_arms_fiveacc_mc.csv \
        --fig_dir evaluation/figures
      python evaluation/r33_score.py --x_metric five_acc \
        --fig_stem r33_fiveacc -o evaluation/csv/r33_arms_fiveacc.csv \
        --fig_dir evaluation/figures
    output_paths:
      - evaluation/figures/r33_clip_vs_lpips.pdf
      - evaluation/figures/r33_clip_vs_ssim.pdf
      - evaluation/figures/r33_fiveacc_yn_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_yn_vs_ssim.pdf
      - evaluation/figures/r33_fiveacc_mc_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_mc_vs_ssim.pdf
      - evaluation/figures/r33_fiveacc_vs_lpips.pdf
      - evaluation/figures/r33_fiveacc_vs_ssim.pdf
    status: completed
    completed_at: 2026-09-22
    note: "User noticed the five_acc panels in the report's 4.1 rendered visibly
      SHORTER than the CLIP-D ones at the same display width -- a real bug, not a
      subjective read. ROOT CAUSE, found by comparing PDF page dimensions directly
      (not guessed): `draw_curve_scatter`'s `ax.set_title(title, fontsize=10)` renders
      `f\"R33: {XSHORT} vs. {Y} -- all 28 arms on {XNOTE}\"` as ONE unwrapped line, and
      `fig.savefig(out_path, bbox_inches=\"tight\")` expands the saved PAGE to fit
      whatever that line needs. `XNOTE[\"five_acc\"]` is by far the longest XNOTE
      string (the full 3-value coarseness explanation) so its PDF page came out at
      959x424pt (ratio 2.26) against CLIP-D's 528x424pt (ratio 1.25) -- same height,
      ~1.8x the width. At matching CSS width in the HTML grid, that reads as shorter.
      Measured every axis before fixing, not just five_acc: yn_acc 626x424 (1.48),
      mc_acc 875x424 (2.06) -- same bug, smaller because their XNOTE strings are
      shorter, not a five_acc-only issue.
      FIX: `ax.set_title(textwrap.fill(title, width=72), fontsize=10)` -- width=72
      chosen because CLIP-D's own (short) title is ~72 chars, so it stays on one line
      (no regression) while every longer XNOTE wraps to 2-3 lines instead of widening
      the canvas. Re-ran all four `--x_metric` variants (not just five_acc) so every
      aggregate figure is on the same fixed footing. VERIFIED: all four PDFs now
      528-530 x 423-424pt, ratio 1.25 on every one (was 1.25/1.48/2.06/2.26). CLIP-D's
      own output is pixel-equivalent to before (its title was already one line, so
      wrapping is a no-op for it) -- confirmed by re-embedding only the three FiVE-Acc
      figures' images in the report, not re-touching the CLIP-D ones."

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
`r31_clip_vs_lpips.pdf` on the corrected axis; and — added 2026-09-22 — the same aggregate
pair exists on the FiVE-Acc axis (originally `r33_fiveacc_vs_lpips.pdf` / `_vs_ssim.pdf`
on `yn_acc`, plus the `r33_fiveacc_mc_*` variant); and — added 2026-09-22, later the same
day — the per-video Pareto panels exist per achievement axis in their own subfolder
(`r33_video_pareto/clipd/`, `r33_video_pareto/fiveacc/`), not just as an aggregate; and
— added 2026-09-22, later still — `r33_report.html` shows every one of those per-video
panels together, one section per clip, not split across files: CLIP-D, `yn_acc`,
`mc_acc`, and the combined `five_acc = (yn+mc+union+inter)/4` (own subfolder each,
`fiveacc/`/`fiveacc_mc/`/`fiveacc_combined/`), four `<figure>` panels per section; and —
added 2026-09-22, later still again — the AGGREGATE FiVE-Acc figures got the same
combined `five_acc` axis and an unambiguous rename: the bare `r33_fiveacc_vs_*.pdf` /
`r33_arms_fiveacc.csv` now mean the combined axis (not silently `yn_acc`), that former
`yn_acc` run moved to `r33_fiveacc_yn_vs_*.pdf` / `r33_arms_fiveacc_yn.csv`, and
`r33_fiveacc_mc_*` is unchanged — three unambiguous aggregate pairs total, matching the
per-video subfolder split; and — added 2026-09-22, finally — `five_acc` is a full 7th
row in `r33_axis_comparison.csv` too (not just the figures), and the axis choice was
formally re-checked against it: unchanged, `clip_d_prompt` remains chosen.

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
| 9 | `fiveacc-figures` | local | The same aggregate pair on the **FiVE-Acc** axis instead of CLIP-D — `yn_acc` then `mc_acc`, 4 PDFs + 2 CSVs | `aggregate-figures` | — |
| 10 | `video-pareto-split` | local | Re-run `figures` with `--pareto_out` pointed at a `clipd/` subfolder, so CLIP-D's per-video panels are no longer flat in `r33_video_pareto/` | `figures` | — |
| 11 | `video-pareto-fiveacc` | local | Per-video Pareto panels for FiVE-Acc's `yn_acc` axis, one PNG per clip, written to a sibling `fiveacc/` subfolder — the per-clip analogue of `fiveacc-figures`' aggregate | `video-pareto-split` | — |
| 12 | `report-dual` | local | Regenerate `r33_report.html` with BOTH axes' per-video panels in one section per clip (`--x_metric2`), superseding the split single-axis reports | `video-pareto-fiveacc` | — |
| 13 | `report-multi` | local | Add `mc_acc` and a new combined `five_acc` axis, per-video: `r33_report.html` now carries FOUR panels per clip, not two | `report-dual` | — |
| 14 | `fiveacc-figures-relabel` | local | Fix `fiveacc-figures`' misleading AGGREGATE filenames: bare `r33_fiveacc_vs_*` now means the combined `five_acc` axis (was silently `yn_acc`); that data moves to `r33_fiveacc_yn_vs_*` | `report-multi` | — |
| 15 | `axis-compare-fiveacc` | local | Add `five_acc` to the Spearman(b, axis) validation table (`AXES`), not just the figures — re-run `axis-compare` | `fiveacc-figures-relabel` | — |
| 16 | `verdict-fiveacc-reread` | manual | Confirm `five_acc` doesn't change the axis choice — high mean (+0.59) but 15/22 flat, same failure mode as `yn_acc`/`mc_acc` | `axis-compare-fiveacc` | — |
| 17 | `fiveacc-figures-titlewrap` | local | Fix a real bug the user spotted: `five_acc`'s aggregate panels rendered visibly shorter than CLIP-D's — long `XNOTE` strings were silently widening the saved PDF page | `verdict-fiveacc-reread` | — |

Step 9 was **added 2026-09-22, after `aggregate-figures`**, at the user's request for a
FiVE-Acc-vs-LPIPS/SSIM aggregate pair alongside the CLIP-D one. It needs no new jobs —
this task's own `fiveacc` step already scored R26's 16 and R31's 4 arms, and R30's 8 were
never re-run because `r30_fiveacc_arms.csv` already holds them (see the Decisions row on
why R30's FiVE-Acc cannot draw a reference curve — it still cannot; R30's arms remain
scatter points here, exactly as on the CLIP-D panels).

Steps 10–11 were **added 2026-09-22, after `fiveacc-figures`**, at the user's request to
give FiVE-Acc the same per-video (not just aggregate) treatment CLIP-D already had, and to
stop the two axes' per-video PNGs sharing one flat folder under generic-looking names.
Both are `local` and need no new jobs — `r33_report_figures.py` was already parametrized
over `--x_metric`/`--pareto_out`/`--out`, so this is two more invocations of an existing
script, not new code (see the `video-pareto-split`/`video-pareto-fiveacc` step notes for
why a re-run was used over a manual file move, and for the real finding step 11 surfaced:
most single clips' `yn_acc` is flat across the whole `b`-sweep, which is the per-clip
mechanism behind the coarseness already reported at the aggregate level).

Step 12 was **added 2026-09-22, after `video-pareto-fiveacc`**, at the user's request:
`r33_report.html` has to show both per-video axes, not send a reader to two separate
files. Needed one script change first (`--x_metric2`/`--pareto_out2` + `write_html_dual`
on `r33_report_figures.py`, see Code to touch) but no new jobs or CSVs — steps 10–11 had
already produced every PNG this step's HTML links to. `r33_report_fiveacc.html` (step
11's standalone file) is now redundant and was deleted.

Step 13 was **added 2026-09-22, after `report-dual`**, at the user's request for two more
per-video axes: `mc_acc` (FiVE-Acc's other accuracy column, already usable as
`--x_metric`) and a new combined `five_acc = (yn_acc+mc_acc+union+inter)/4` — evaluate.py's
own verdict, verified against its source rather than recomputed (see Decisions and the
step's own note). No new GPU work: every column this step reads was already on disk from
the `fiveacc` step. Required generalising `report-dual`'s 2-axis-only script interface to
N axes first (`--x_metric_more`/`--pareto_out_more`, `write_html_multi`; see Code to touch,
2nd pass) since a third and fourth panel per clip needed adding, not just swapping two.

Step 14 was **added 2026-09-22, after `report-multi`**, when the user pointed out that
`fiveacc-figures`' AGGREGATE `r33_fiveacc_vs_lpips.pdf`/`_vs_ssim.pdf` were misleading:
the bare, unqualified name implied "FiVE-Acc" broadly but the figure only ever plotted
`yn_acc`. Fixed by making `five_acc` (the combined axis `report-multi` just added for
per-video) a real `--x_metric` on `r33_score.py` too, computing it for R30's arms from
existing columns since `r30_fiveacc_arms.csv` has no precomputed `five_acc` column (see
Code to touch and the step's own note), then re-running under corrected names: the bare
name now means the combined axis, `yn_acc`'s figures moved to an explicit `_yn` suffix,
and `mc_acc`'s were already correct. No new GPU work — same inputs `fiveacc-figures`
already had on disk.

Steps 15–16 were **added 2026-09-22, after `fiveacc-figures-relabel`**, at the user's
request: `five_acc` had been added to the aggregate and per-video FIGURES (steps
13–14) but never to `r33_axis_compare.py`'s `AXES` table, so it had no
`Spearman(b, axis)` row and never went through the actual axis-CHOICE test the other
five candidates did. One-line script addition, then a re-read against the same bar the
original `verdict` step used. RESULT: no change — `five_acc` fails the same
high-mean/low-coverage test `yn_acc`/`mc_acc` already failed (expected, since it's a
deterministic function of them), and `clip_d_prompt` remains the chosen axis.

Step 17 was **added 2026-09-22, after `verdict-fiveacc-reread`**, when the user noticed
`five_acc`'s panels in the report's 4.1 rendered visibly shorter than CLIP-D's at the
same display width. Traced to a real bug in `draw_curve_scatter` (see Code to touch and
the step's own note): an unwrapped title plus `bbox_inches="tight"` let a long `XNOTE`
string silently widen the saved PDF page, not just for `five_acc` — every FiVE-Acc axis
had a wider-than-CLIP-D page, in proportion to how long its `XNOTE` string was. Fixed
with `textwrap.fill`, then all four `--x_metric` variants re-run so nothing is left
inconsistent.

```
/run-step R33 clip-d
/run-step R33 wait-clip-d
/run-step R33 fiveacc
/run-step R33 wait-fiveacc
/run-step R33 axis-compare
/run-step R33 figures
/run-step R33 verdict
/run-step R33 aggregate-figures
/build-step R33 fiveacc-figures
/run-step R33 fiveacc-figures
/run-step R33 video-pareto-split
/run-step R33 video-pareto-fiveacc
/build-step R33 report-dual
/run-step R33 report-dual
/build-step R33 report-multi
/run-step R33 report-multi
/build-step R33 fiveacc-figures-relabel
/run-step R33 fiveacc-figures-relabel
/build-step R33 axis-compare-fiveacc
/run-step R33 axis-compare-fiveacc
/run-step R33 verdict-fiveacc-reread
/build-step R33 fiveacc-figures-titlewrap
/run-step R33 fiveacc-figures-titlewrap
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
| FiVE-Acc aggregate axis (2026-09-22) | **`yn_acc` primary, `mc_acc` secondary.** Measured over the 22 clips, ordered by `b`: `yn_acc` SPATIAL 0.545 → 0.818 (spread 0.273, monotone) and UNIFORM 0.500 → 0.864 (spread 0.364, monotone), while `mc_acc` SPATIAL is saturated from `b`=4 (spread 0.136) and UNIFORM is outright **non-monotone** (0.727 0.773 0.818 0.773 0.864 0.818 0.818 0.818). So `yn_acc` takes the requested `r33_fiveacc_vs_*.pdf` filenames; `mc_acc` is drawn too, under `r33_fiveacc_mc_vs_*.pdf`, as a robustness variant that should be read as a null result rather than a curve. |
| Aggregate curves vs. the per-clip coarseness row | **Not a contradiction.** The FiVE-Acc-coarseness row above ("6 of 8 R30 arms share `mc_acc` = 0.818") is a *per-clip* statement about a 0/1 metric; averaging 22 clips per arm smooths it into the monotone aggregate curves quoted above. |
| Residual quantisation caveat | Averaging 22 binary clips quantises x to multiples of 1/22, so **arms still collide exactly**: on `yn_acc` SPATIAL, `b` = 4, 6, 8 all sit at 0.727. Points overplot and the figure reads as a vertical band, not a clean ranking — the same failure mode as the coarseness row, one order of magnitude weaker. Do not read arm order off the x-axis at ties; read it off the `b` labels. |
| No new GPU work for these figures | Every x-value already exists on disk: `edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv` (R26's 16, this task's `fiveacc` step, job 1004302), `edit{T}_FiVE_r33_fiveacc_r31_{arm}_frame_stride8.csv` (R31's 4, same job), `evaluation/csv/r30_fiveacc_arms.csv` (R30's 8). y-values (`lpips_unedit_part` / `ssim_unedit_part`) are unchanged from the CLIP-D panels — only the x pairing differs. |
| Distinct `-o` and `--fig_stem` are mandatory | `r33_score.py` names its CSV column after the x-metric, so reusing the default `-o` would silently overwrite `r33_arms.csv`'s `clip_d_prompt` table with a `yn_acc` one. Hence four output paths, four figure stems. |
| `0040_tennis` on the FiVE-Acc panels | **Kept, all 22 clips.** The exclusion above applies to the CLIP family only: the EOT-truncation bug corrupts the text embedding, and FiVE-Acc never embeds the prompt (`r33_axis_compare.py`'s `AXES` table already marks these axes `tennis_immune=True`). The FiVE-Acc panels therefore carry 22 clips per arm where the CLIP-D panels carry 21. |
| Per-video panels: one subfolder per axis (2026-09-22) | **`r33_video_pareto/clipd/` and `r33_video_pareto/fiveacc/`, not one flat folder.** `r33_report_figures.py` is parametrized over `--x_metric`, so a second axis run into the same flat directory would sit under filenames distinguished only by the `AXIS_STEM` suffix (`_clipd_vs_lpips.png` vs `_fiveacc_yn_vs_lpips.png`) — technically non-colliding, but easy to misread as one mixed set when browsing the folder. Split at the user's request; regenerated via a re-run with `--pareto_out` changed, not a manual file move (see `video-pareto-split`'s step note for why that matters: `write_html`'s `<img src>` is computed from the actual returned path, so a re-run keeps `r33_report.html` correct automatically). |
| Per-video FiVE-Acc axis scope | **`yn_acc` only, not `mc_acc`.** Mirrors the aggregate-level call (`FiVE-Acc aggregate axis` row above): `mc_acc` is already an established null result, so a per-video panel for it would not be new evidence. If `mc_acc` per-video panels are ever wanted, `video-pareto-fiveacc`'s command is the template — swap `--x_metric yn_acc` for `mc_acc` and give it its own `--pareto_out`/`--out`. |
| One merged `r33_report.html`, not two single-axis reports (2026-09-22) | **User's explicit request.** `video-pareto-split`/`video-pareto-fiveacc` had already produced a correct CLIP-D report (`r33_report.html`) and a correct but separate FiVE-Acc-only report (`r33_report_fiveacc.html`). `report-dual` merges them into one: every clip's section now shows both per-video panels side by side, captioned by axis. The standalone FiVE-Acc report was deleted once the merge was verified, so there is exactly one canonical per-video report on disk, not two that could drift out of sync on a future re-run. |
| `write_html_dual` lives in `r33_report_figures.py`, not `r31_html_report.py` | `r31_html_report.write_html` accepts one `pareto_by_case` dict and renders one `<img>` per section — extending it to two would change its signature for R31/R34's own (single-axis) call sites too. A local dual-panel writer in R33's own script keeps that shared function untouched, consistent with this task's standing rule of reusing R31's code unmodified (see the figures-script todo). |
| `five_acc` formula (2026-09-22) | **`(yn_acc + mc_acc + union + inter) / 4`, exactly evaluate.py's own combined column, not a new derived metric.** Verified against `evaluation/fivebench/evaluate.py` (~line 519-527), not assumed: `union = int(yn_acc or mc_acc)`, `inter = int(yn_acc and mc_acc)`, `five_acc = mean([yn_acc, mc_acc, union, inter])`. Already computed at job time and sitting in every R26/R31 per-clip CSV from this task's own `fiveacc` step (`*\|five_acc` column) — reading it needed zero new scoring. |
| `five_acc` per-video scope | **R26/R31 only, same as `yn_acc`/`mc_acc`'s existing per-video scope — R30 out of scope.** `r30_fiveacc_arms.csv` has `union`/`inter` columns (so a `five_acc` value is *computable* for R30's 8 arms) but no precomputed `five_acc` column, and `r33_report_figures.py` never reads R30's per-video data for any axis (R30 only appears as scatter points on the aggregate `r33_score.py` panels) — no gap here, just noting why R30 isn't part of this. |
| `five_acc` NOT added to the axis-comparison / verdict test | **Per-video panel only, at the user's explicit request — not wired into `r33_axis_compare.py`'s `AXES` table.** It has no `Spearman(b, axis)` row in `r33_axis_comparison.csv` and played no part in the `verdict` step's axis choice. If it's ever wanted there too, `AXES` just needs one more `("five_acc", "fiveacc", "five_acc", True)` tuple — `load_fiveacc_metric` already reads arbitrary `*\|<metric>` columns generically, no new loader needed. |
| CLI generalised to N per-video axes (2026-09-22) | **`--x_metric_more`/`--pareto_out_more` (repeatable, paired positionally), replacing the same-day `--x_metric2`/`--pareto_out2`.** Two flags rather than a packed `"metric:path"` string, to match the script's existing plain-flag style. Chosen once a *third* axis (`five_acc`, alongside `yn_acc` and `mc_acc`) was requested for the same report — a fixed two-slot interface would have needed a second special-cased rename instead of one generalisation. |
| `r33_fiveacc_vs_*`/`r33_arms_fiveacc.csv` naming (2026-09-22, corrected) | **Bare "fiveacc" means the combined `five_acc` axis, not `yn_acc`.** `fiveacc-figures` originally used the bare name for the `yn_acc` run — reasonable in isolation, but misleading once a genuinely combined axis existed, since "FiVE-Acc" is the name of the whole 4-column family. `fiveacc-figures-relabel` renamed that run to an explicit `_yn` suffix and put the combined `five_acc` axis under the bare name instead, matching the per-video subfolder naming `report-multi` already used (`fiveacc_combined` for per-video is arguably the odd one out now — not renamed here since the per-video user request didn't ask for it, but worth aligning if this comes up again). `r33_fiveacc_mc_*` was already correctly qualified and untouched. |
| R30's `five_acc` is computed, not read (2026-09-22) | **`r30_fiveacc_arms.csv` has `yn_acc`/`mc_acc`/`union`/`inter` columns but no precomputed `five_acc` one** (unlike R26/R31's CSVs, where `evaluate.py` writes a `five_acc` column directly at scoring time — confirmed by reading `r30_fiveacc_arms.csv`'s header before writing code, not assumed). `load_r30_arms`'s new `five_acc` branch computes `(yn_acc+mc_acc+union+inter)/4` per clip from those four existing columns — same formula, no rescore, no new file. |
| `five_acc` added to the axis-CHOICE test too (2026-09-22) | **Not just the figures — `r33_axis_compare.py`'s `AXES` table too, per the user's explicit request.** Confirms rather than changes the earlier verdict: `five_acc` mean_rho=+0.5916 (high) but only 7/22 clips non-flat (15 flat) — the same high-mean/low-coverage failure `yn_acc` (+0.7113, 16 flat) and `mc_acc` (+0.4638, 18 flat) already showed, expected since `five_acc` is a deterministic function of them (see `r33_report_figures.py`'s per-video note on the 3-value coarseness: 0, 0.5, 1.0). `clip_d_prompt` (+0.4080, 0 flat, full 22/22 coverage) remains the only axis that both clears the incumbent and avoids this trap — unchanged axis choice, formally re-confirmed rather than left implicit. |

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

### fiveacc-figures

```bash
python evaluation/r33_score.py --x_metric yn_acc \
  --fig_stem r33_fiveacc -o evaluation/csv/r33_arms_fiveacc.csv --fig_dir evaluation/figures
python evaluation/r33_score.py --x_metric mc_acc \
  --fig_stem r33_fiveacc_mc -o evaluation/csv/r33_arms_fiveacc_mc.csv --fig_dir evaluation/figures
```

### video-pareto-split

```bash
python evaluation/r33_report_figures.py --x_metric clip_d_prompt \
  --pareto_out evaluation/figures/r33_video_pareto/clipd
```

### video-pareto-fiveacc

```bash
python evaluation/r33_report_figures.py --x_metric yn_acc \
  --pareto_out evaluation/figures/r33_video_pareto/fiveacc \
  --out evaluation/figures/r33_report_fiveacc.html
```

### report-dual

```bash
python evaluation/r33_report_figures.py --x_metric clip_d_prompt --x_metric2 yn_acc \
  --pareto_out evaluation/figures/r33_video_pareto/clipd \
  --pareto_out2 evaluation/figures/r33_video_pareto/fiveacc \
  --out evaluation/figures/r33_report.html
```

### report-multi

```bash
python evaluation/r33_report_figures.py --x_metric clip_d_prompt \
  --pareto_out evaluation/figures/r33_video_pareto/clipd \
  --x_metric_more yn_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc \
  --x_metric_more mc_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_mc \
  --x_metric_more five_acc --pareto_out_more evaluation/figures/r33_video_pareto/fiveacc_combined \
  --out evaluation/figures/r33_report.html
```

### fiveacc-figures-relabel

```bash
python evaluation/r33_score.py --x_metric yn_acc \
  --fig_stem r33_fiveacc_yn -o evaluation/csv/r33_arms_fiveacc_yn.csv \
  --fig_dir evaluation/figures
python evaluation/r33_score.py --x_metric five_acc \
  --fig_stem r33_fiveacc -o evaluation/csv/r33_arms_fiveacc.csv \
  --fig_dir evaluation/figures
```

### axis-compare-fiveacc

```bash
python evaluation/r33_axis_compare.py
```

### verdict-fiveacc-reread

```bash
column -s, -t < evaluation/csv/r33_axis_comparison.csv
```

### fiveacc-figures-titlewrap

```bash
python evaluation/r33_score.py -o evaluation/csv/r33_arms.csv --fig_dir evaluation/figures
python evaluation/r33_score.py --x_metric yn_acc \
  --fig_stem r33_fiveacc_yn -o evaluation/csv/r33_arms_fiveacc_yn.csv --fig_dir evaluation/figures
python evaluation/r33_score.py --x_metric mc_acc \
  --fig_stem r33_fiveacc_mc -o evaluation/csv/r33_arms_fiveacc_mc.csv --fig_dir evaluation/figures
python evaluation/r33_score.py --x_metric five_acc \
  --fig_stem r33_fiveacc -o evaluation/csv/r33_arms_fiveacc.csv --fig_dir evaluation/figures
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

FiVE-Acc axis (step 9). Same script, same y-sources, x swapped — the only structural
difference is that R30's arms come from a CSV that already carries `yn_acc` / `mc_acc`
columns, so they need no loader at all:

```mermaid
flowchart LR
  F26[(edit{T}_FiVE_r33_fiveacc_{method}\n_frame_stride8.csv\nR26 16 arms)] --> L1[load_fiveacc_metric\nb-swept families, exists]
  F31[(edit{T}_FiVE_r33_fiveacc_r31_{arm}\n_frame_stride8.csv\nR31 4 arms)] --> L2[load_stem_metric\nsingle arm, exists\nmoves to r30_score.py]
  F30[(r30_fiveacc_arms.csv\nyn_acc / mc_acc columns\nR30 8 arms)] --> L3[existing else-branch\nreads the column directly]
  L1 --> FAGG[r33_score.py\n--x_metric yn_acc / mc_acc]
  L2 --> FAGG
  L3 --> FAGG
  FAGG --> FCSV[/r33_arms_fiveacc.csv\n+ r33_arms_fiveacc_mc.csv/]
  FAGG --> FFIG[/r33_fiveacc_vs_lpips.pdf\n+ r33_fiveacc_vs_ssim.pdf\n+ the two _mc_ variants/]
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
| `evaluation/r33_score.py` (2nd pass) | **Widen `--x_metric` to the FiVE-Acc axes**, for the `fiveacc-aggregate-script` todo. This is dispatch-widening, not new structure — the script was already parametrized. Five edits: (1) add `yn_acc` / `mc_acc` to the `--x_metric` `choices` (line 404) and to `XSHORT` / `XNOTE` / `XLABEL` (lines 62-73); (2) `_x_r26` (line 76) gains a FiVE-Acc branch returning `r33_axis_compare.load_fiveacc_metric(r33_csv_dir, idx2vid, name2case, CONST_BS, col, method_fmt=method_fmt)` — note it takes the **bare** `method_fmt`, not the `"r26_" + method_fmt` the CLIP-D branch prepends, because it builds the stem `edit{T}_FiVE_r33_fiveacc_{method}_frame_stride8.csv` itself; (3) `load_r31_arms`'s `else` branch (lines 168-173) gains a FiVE-Acc branch calling the promoted `load_stem_metric` with `stem=f"r33_fiveacc_r31_{arm}"`, `stride=8` — the current `load_r31_percli_metric(..., arm, "", x_metric)` call cannot be reused here because it globs the `r31_{arm}` stem, and R33's FiVE-Acc lives under the `r33_fiveacc_r31_{arm}` stem; (4) nothing else. |
| `load_r30_arms` — **no change needed for `yn_acc`/`mc_acc`** | Because `yn_acc` / `mc_acc` are not in `XM_CLIPD`, x already falls through to the `else` branch at lines 113-118, which reads `r30_perclip_csv` (= `evaluation/csv/r30_fiveacc_arms.csv`) and does `float(r[x_metric])`. That file's columns are *literally* named `yn_acc` and `mc_acc`, and it is the same file the lpips/ssim loop below it already reads, so both axes stay filtered by the identical `case_id` set. Verify before editing — do not "fix" this branch. **UPDATE, 2026-09-22: `five_acc` DID need a change** — see the `r33_score.py` (2nd pass) row below. This original row's claim still holds exactly as written for `yn_acc`/`mc_acc`; it just isn't the whole function's story anymore. |
| ⚠️ The one real trap: **the column name differs by source** | R26's and R31's FiVE-Acc CSVs are written by `evaluate.py` and carry its raw column names `five_acc_yes_no` / `five_acc_multi_choice` (matched suffix-wise, as `step14\|five_acc_yes_no`). `r30_fiveacc_arms.csv` carries the short labels `yn_acc` / `mc_acc`. So a label→column map (`{"yn_acc": "five_acc_yes_no", "mc_acc": "five_acc_multi_choice"}`) is needed in the `_x_r26` and `load_r31_arms` branches **and must not be applied in `load_r30_arms`**, which wants the label verbatim. Applying it uniformly is the failure mode to guard against: it raises `no *\|five_acc_yes_no column` on R30 rather than failing silently, but only after the other 20 arms have already loaded. |
| `evaluation/r34_score.py` → `evaluation/r30_score.py` | **Promote `load_stem_metric` (r34_score.py lines 89-130) into `r30_score.py`; leave a re-export in `r34_score.py`.** No behaviour change, pure relocation. Rationale: `r30_score.py` is already the shared base that `r31_score.py`, `r33_score.py`, `r33_axis_compare.py`, `r34_score.py` and `r34_report_figures.py` all import from (`CONST_BS`, `build_join_tables`, `load_r26_metric`, `oracle_frontier`), so both this task's `load_r31_arms` branch and R31's `fiveacc-score-script` can reach it without either script importing the other or R33/R31 taking a backward dependency on R34. The re-export keeps R34's existing call sites and `r34_report_figures.py`'s import untouched. **Verified read-only that the function already reads the FiVE-Acc single-arm files with no change**, because its column lookup is `endswith(f"\|{metric}")` and so tolerates R31's rows being keyed `step14\|five_acc_yes_no`: `load_stem_metric(..., "r33_fiveacc_r31_lpips", 8, "five_acc_yes_no")` returns 22 clips, mean 0.7273; with `five_acc_multi_choice`, 22 clips, mean 0.7727; and on `("r31_lpips", 8, "lpips_unedit_part")` it is bit-identical to `load_r31_percli_metric`. Keep its docstring's naming-collision warning verbatim on the move — it is the reason the function takes a fully-resolved stem rather than a format string, and the `r31_` / `r34_r31_` collision it guards against is exactly the one the `r33_fiveacc_r31_` stem would otherwise walk into. Update the two `[r34_score]` strings in its error messages to a neutral prefix. |
| `evaluation/r33_report_figures.py` (2026-09-22) | **Added `--x_metric2`/`--pareto_out2` and a new `write_html_dual` function**, so `r33_report.html` can show both achievement axes' per-video panels in one section instead of one axis per report. `main()` refactored: a local `load_axis(x_metric)` closure (curve + R31 points) is now called once for `--x_metric` and, if given, again for `--x_metric2`; `r31_lpips` (axis-independent) is loaded once and shared by both `build_video_pareto` calls. When `--x_metric2` is set, `write_html_dual` (defined in this file, not `r31_html_report.py` — that module's `write_html` only accepts one pareto image per section) replaces the existing `_r31_write_html` + sentinel-string-patch path; that path is otherwise unchanged and still what the single-axis `--pareto_out` runs (`video-pareto-split`, `video-pareto-fiveacc`) use. `SystemExit` guard added: `--x_metric2` without `--pareto_out2` fails loudly rather than silently mixing the second axis's PNGs into `--pareto_out`. |
| `evaluation/r33_report_figures.py` (2nd pass, 2026-09-22, same day) | **Generalised the dual-axis pass above to N axes, and added `five_acc` as a candidate axis.** (1) `AXIS_COLUMN`/`AXIS_LABEL`/`AXIS_STEM` gain a `"five_acc": "five_acc"` entry each (label equals column name — no renaming needed, unlike `yn_acc`/`mc_acc`); `"five_acc"` added to both FiVE-Acc-family dispatch tuples in `load_axis_curve`/`load_axis_r31_point` (routes through the already-generic `*\|<metric>` suffix-matching loader — no new loader function). (2) `--x_metric2`/`--pareto_out2` **removed**, replaced by repeatable `action="append"` `--x_metric_more`/`--pareto_out_more`, paired positionally with a count-mismatch `SystemExit` guard. (3) `write_html_dual` **removed**, replaced by `write_html_multi(cases, strips_by_case, panels, out_path, root)` where `panels` is an ordered `[(label, pareto_by_case), ...]` of any length — the 2-panel case is now just a 1-element `panels` list, not a separate code path. (4) `main()`'s per-axis data (curve + R31 points + `pareto_out`) collected into one `axis_data` list (primary axis first, then each `--x_metric_more` in order) and the per-case loop iterates it, writing into `pareto_by_axis[x_metric]` — replaces the hand-duplicated `pareto_by_case`/`pareto2_by_case` pair from the first pass. `r31_lpips` still loaded once, shared by every axis, unchanged from the first pass. |
| `evaluation/r33_score.py` (2nd pass, 2026-09-22) | **Added `five_acc` as a third `--x_metric` choice, for the AGGREGATE (28-arm) figures — the counterpart of the per-video change above.** `XM_FIVEACC` gains `"five_acc": "five_acc"` (R26/R31 route through the already-generic `_x_r26`/`load_r31_arms` FiVE-Acc branches unchanged, same suffix-matching loader as `yn_acc`/`mc_acc`); `XSHORT`/`XNOTE`/`XLABEL` gain matching entries, `XNOTE["five_acc"]` stating the coarseness caveat directly on the rendered figure. `load_r30_arms` gains a new `elif x_metric == "five_acc":` branch (before the existing generic `else`, which stays untouched for every other axis) computing `(float(r["yn_acc"])+float(r["mc_acc"])+float(r["union"])+float(r["inter"]))/4` per clip from `r30_fiveacc_arms.csv`'s existing columns — that file has no precomputed `five_acc` column, unlike R26/R31's CSVs. `--x_metric`'s `choices` tuple extended to include it. |
| `evaluation/r33_axis_compare.py` (2026-09-22) | **Added `five_acc` to the `AXES` tuple: `("five_acc", "fiveacc", "five_acc", True)`.** One line — `kind="fiveacc"` already dispatches to `load_fiveacc_metric` with whatever column string is given, and `"five_acc"` is a real column on every R26/R31 per-clip CSV (evaluate.py writes it directly), so no new loader or dispatch branch was needed, unlike `r33_score.py`'s `load_r30_arms` (this script never reads R30's per-clip data at all — R30 has no row in `r33_axis_comparison.csv`, same as before). `tennis_immune=True` since FiVE-Acc never embeds the prompt. |
| `evaluation/r33_score.py` (3rd pass, 2026-09-22) | **Bug fix, not a feature addition: title text was silently controlling output image size.** `import textwrap` added; `draw_curve_scatter`'s `ax.set_title(title, fontsize=10)` changed to `ax.set_title(textwrap.fill(title, width=72), fontsize=10)`. Root cause: `title` embeds `XNOTE[x_metric]` as one unwrapped line, and `fig.savefig(out_path, bbox_inches="tight")` sizes the saved PAGE to fit it — so an axis with a long `XNOTE` (five_acc's coarseness caveat, the longest) got a much wider PDF page than CLIP-D's short one, same height, and read as visibly shorter once both were displayed at the same CSS width in the report. `width=72` matches CLIP-D's own title length so its output is unchanged; measured before and after on all four `--x_metric` variants (clip_d_prompt/yn_acc/mc_acc/five_acc): page aspect ratio went from {1.25, 1.48, 2.06, 2.26} to {1.25, 1.25, 1.25, 1.25}. |
