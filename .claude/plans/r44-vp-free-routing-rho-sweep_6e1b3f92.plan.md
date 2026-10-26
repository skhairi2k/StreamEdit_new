---
name: "R44 plan — VP-free routing + text-only rho sweep"
overview: "Text-only (novp) counterpart of R35/R38: an 8-rho uniform sweep at N=15 as the reference curve, plus the divergence-routing arms for whichever of the 8 (N, metric, regime) combos the user selects at a gate. Routing gain at matched bg LPIPS, text-only vs VP. **Equal NFE is mandatory**: a second text-only sweep at N = 15 + N_d (N=20 or N=30, chosen later at a gate, like the combo) is the headline reference."
task_id: R44
todos:
  - id: combos-lib
    content: "slurm_scripts/five_bench/r44_combos.sh -- sourceable table of the 8 combos {dino,lpips} x {n5,n15} x {unblended,tgt09} + r44_parse (combo -> ARM, N, REGIME, SCHED, RUN) + r44_selected (reads evaluation/r44_selected_combos.txt, hard-fails if missing/empty/unknown) + r44_runs (unique stage-1 RUNs of the selection). Run directly: `--combos` / `--runs` print one per line, so sbatch --array sizes are computed."
    status: completed
  - id: smoke
    content: "slurm_scripts/five_bench/r44_smoke.sh -- 2-clip GPU smoke with gates G1-G3 (see Code to touch). Blocks step `smoke`."
    status: completed
  - id: infer-script
    content: "slurm_scripts/five_bench/r44_infer.sh -- r36_infer.sh with --vp_mode novp and no --first_frame_edit_dir; positional N in {15,20,30} (default 15); METHOD r44_n{N}_rho{R}_novp; out r44_rho_sweep. Array 0-7 over RHOS."
    status: completed
  - id: eval-scripts
    content: "slurm_scripts/five_bench/r44_eval.sh, r44_fiveacc.sh, r44_clipd.sh -- mode `sweep <N>` (array 0-7 over rho; clipd single job) or `arms` (array over the selected combos; clipd single job). Stems r44_n{N}_rho{R}_novp / r44_fiveacc_n{N}_rho{R}_novp / r44_{combo} / r44_fiveacc_{combo}; clipd -> r44_clip_directional_sweep_n{N}.csv / r44_clip_directional_arms.csv. Guards as in the R36/R38 copies."
    status: completed
  - id: stage1-script
    content: "slurm_scripts/five_bench/r44_stage1.sh -- r38_stage1.sh with --vp_mode novp and no anchors; array over r44_runs (n5_unblended, n5_tgt09, n15_unblended, n15_tgt09 -- only those the selection needs); --step N --dump_steps N-1 --measure_index N-1; out r44_stage1/{run}, r44_latents/{run}."
    status: completed
  - id: stage2-script
    content: "slurm_scripts/five_bench/r44_stage2.sh -- r38_stage2.sh, array over the selected combos; TARGET_ROOT r44_stage1/{run}/step{N-1:02d}, LATENT_DIR r44_latents/{run}, out r44_div/{combo}."
    status: completed
  - id: divstats
    content: "evaluation/r44_div_stats.py -- per selected combo, raw-divergence stats (mean, p50/p90/p99, sat_frac at div_max) for text-only (r44_div) next to its VP counterpart (r35_div for N=15, r38_div for N=5). Imports pooled_d / DIV_MAX from r38_div_stats.py."
    status: completed
  - id: absnorm-script
    content: "slurm_scripts/five_bench/r44_absnorm.sh -- r38_absnorm.sh over the selected combos; div_max by metric prefix (0.65 / 1.0); r44_div -> r44_div_abs -> r44_rho (budget_linear, tau 2..50, --step 15)."
    status: completed
  - id: stage3-script
    content: "slurm_scripts/five_bench/r44_stage3.sh -- r38_stage3.sh with --vp_mode novp, no anchors; array over the selected combos; out r44_arms/r44_{combo}/step14; per-file manifest guard."
    status: completed
  - id: score-script
    content: "evaluation/r44_score.py -- three figures (CLIP / CLIP-D / FiVE-Acc vs bg LPIPS), each two panels VP | text-only with shared axes; routing gain at matched LPIPS per combo x modality -> r44_routing_gain.csv; r44_arms.csv. --matched_n {20,30} is REQUIRED for the final run: the equal-NFE curves (R39 for VP, R44 for text) are the headline reference, N=15 secondary. Only --interim (N=15 only) may omit it, and then writes *_interim outputs."
    status: completed
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r44_smoke.sh
    status: completed
    completed_at: 2026-10-06
    job_id: "1023699"
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "grep -h GATE logs/r44_smoke_{job_id}.out -- G1 IDENTICAL (or user accepts the reported max|diff|), G2 echoes match, G3 PASS"
    status: completed
    completed_at: 2026-10-07
    note: "logs/r44_smoke_1023699.out: GATE G1/G2/G3 all PASS. G1 routed(const rho=2)==uniform(blend_power=2) sha256-identical on both smoke clips. G2 echoed blender_rate vectors correct for n15_tgt09/n15_unblended/n5_tgt09/n5_unblended. G3: vp_mode=novp x2 on every run+npz, 0 other modes; n15_tgt09 draft frames/z_trg DIFFER from R35 VP first2 (0/69, 0/93 frames identical) -- no anchor leak."
  - id: infer-sweep
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=0-7 slurm_scripts/five_bench/r44_infer.sh 15
    sets_status: running
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r44_rho_sweep
    status: completed
    completed_at: 2026-10-07
    job_id: "1027756"
  - id: wait-sweep
    type: manual
    wait_for: infer-sweep
    check_hint: "8 tasks exit 0; each r44_n15_rho{R}_novp has 100/100/100/100/9/10 real dirs and 0 non-ok manifest rows; logs say vp_mode=novp"
    status: completed
    completed_at: 2026-10-09
    note: "verified on disk 2026-10-09: all 8 r44_n15_rho{R}_novp arms 419/419 dirs, 0 per-type mismatches, 0 real non-ok manifest rows (first check used `cat *.csv | awk NR>1` across 6 files and wrongly counted 5 other headers as errors -- redone per-file, genuinely clean)."
  - id: eval-sweep
    type: sbatch
    wait_for: wait-sweep
    command: |
      sbatch --array=0-7 slurm_scripts/five_bench/r44_eval.sh sweep 15
      sbatch --array=0-7 slurm_scripts/five_bench/r44_fiveacc.sh sweep 15
      sbatch slurm_scripts/five_bench/r44_clipd.sh sweep 15
    status: completed
    completed_at: 2026-10-09
    note: "job_ids: eval=1031039 fiveacc=1031040 clipd=1031041, submitted with NO --dependency: render pre-verified clean on disk, and afterok:1027756 was REJECTED by sbatch (Job dependency problem) -- the prior poller instance (chain2) had already submitted a first attempt (jobs 1031009-1031012, seen briefly in squeue) but it never produced any log output and silently left the queue, most likely killed by the same context-compaction/session boundary that stopped the poller process itself (background-task notification: stopped, no completion record). Lesson: once a prerequisite job has fully left squeue, afterok on its id can be rejected as aged-out of dependency tracking -- verify via disk instead and submit without --dependency when already confirmed done."

  - id: wait-eval-sweep
    type: manual
    wait_for: eval-sweep
    check_hint: "48 harness + 48 five_acc per-type CSVs (8 rho x 6 types); r44_clip_directional_sweep_n15.csv = 8*419 rows; 0 Error:/OOM in logs/r44_*sweep*"
    status: completed
    completed_at: 2026-10-12
    note: "verified 2026-10-12: r44_rho_sweep_n15.csv exists (summarize-sweep succeeded, which hard-fails unless all 48 harness + 48 fiveacc CSVs + the clip_directional sweep are complete), so N=15 eval-sweep is confirmed fully done."
  - id: summarize-sweep
    type: local
    wait_for: wait-eval-sweep
    command: |
      python evaluation/r36_summarize.py \
        --harness_fmt 'r44_n15_rho{rho}_novp' \
        --fiveacc_fmt 'r44_fiveacc_n15_rho{rho}_novp' \
        --clipd_csv evaluation/csv/r44_clip_directional_sweep_n15.csv \
        --clipd_method_fmt 'r44_n15_rho{rho}_novp' \
        --out evaluation/csv/r44_rho_sweep_n15.csv
    output_paths:
      - evaluation/csv/r44_rho_sweep_n15.csv
    status: completed
    completed_at: 2026-10-09
    job_id: "1031042"
    note: "sbatch --wrap --dependency=afterok:1031039:1031040:1031041 (the fresh eval-sweep trio)"

  - id: choose-config
    type: manual
    wait_for: wait-smoke
    check_hint: "USER DECISION. Write the chosen combos (one per line, from the 8 in r44_combos.sh) to evaluation/r44_selected_combos.txt, then check: bash slurm_scripts/five_bench/r44_combos.sh --combos ; --runs. N=5 combos need R38's eval/fiveacc/clipd done before `figures`."
    status: completed
    completed_at: 2026-10-07
    note: "USER DECISION (memory project_routing_config_final.md, 2026-10-07): FINAL routing config = DINO x tau=0.9 x N=5 (combo dino_n5_tgt09), single combo. Reason: cost (N=5 draft+render ~1.54x vs 2x at N=15); DINO.tau0.9 vs LPIPS.tau0.9 at N=5 statistically indistinguishable overall but DINO significantly better on non-rigid substitutions (edit2 CLIP-D +0.0077) and better bg preservation than DINO.unblended. Written to evaluation/r44_selected_combos.txt."
  - id: stage1
    type: sbatch
    wait_for: choose-config
    command: sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --runs | wc -l) - 1 )) slurm_scripts/five_bench/r44_stage1.sh
    sets_status: running
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r44_stage1
      - ~/Data/dataggen/outputs/five_bench/r44_latents
    status: completed
    completed_at: 2026-10-07
    job_id: "1027757"
  - id: wait-stage1
    type: manual
    wait_for: stage1
    check_hint: "every task exit 0; 419 npz per run (100/100/100/100/9/10); npz vp_mode == novp; echoed blender_rate vectors as in G2"
    status: completed
    completed_at: 2026-10-09
    note: "verified on disk 2026-10-09: r44_latents/n5_tgt09 has 419 npz, r44_stage1/n5_tgt09 has 28509 frame files."
  - id: stage2
    type: sbatch
    wait_for: wait-stage1
    command: sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --combos | wc -l) - 1 )) slurm_scripts/five_bench/r44_stage2.sh
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r44_div
    status: completed
    completed_at: 2026-10-07
    job_id: "1027778"
    note: "sbatch --dependency=afterok:1027757 (stage1), automatic chain per user request"
  - id: wait-stage2
    type: manual
    wait_for: stage2
    check_hint: "419 npz per selected combo, exact per-type breakdown; 0 [ERROR]/Traceback"
    status: completed
    completed_at: 2026-10-09
    note: "verified on disk 2026-10-09: r44_div/dino_n5_tgt09 has 419 npz."
  - id: divstats
    type: local
    wait_for: wait-stage2
    command: python evaluation/r44_div_stats.py
    output_paths:
      - evaluation/csv/r44_div_stats.csv
    status: completed
    completed_at: 2026-10-07
    job_id: "1027779"
    note: "submitted as sbatch --wrap --dependency=afterok:1027778 (stage2) instead of running locally, per user request to chain stage2->divstats/absnorm->stage3->eval-arms automatically; CPU partition, streamgve env"
  - id: absnorm
    type: sbatch
    wait_for: wait-stage2
    command: sbatch slurm_scripts/five_bench/r44_absnorm.sh
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r44_div_abs
      - ~/Data/dataggen/outputs/five_bench/r44_rho
    status: completed
    completed_at: 2026-10-07
    job_id: "1027780"
    note: "sbatch --dependency=afterok:1027778 (stage2), automatic chain per user request"
  - id: stage3
    type: sbatch
    wait_for: absnorm
    command: sbatch --dependency=afterok:1027780 --array=0-0 slurm_scripts/five_bench/r44_stage3.sh
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r44_arms
    status: completed
    completed_at: 2026-10-07
    job_id: "1027805"
    note: "automatic chain per user request; /tmp/r44_chain_poll.sh waited ~6 min for the cluster's 30-job submit cap to clear (R45's sweep left the queue), then submitted this + eval-arms"
  - id: wait-stage3
    type: manual
    wait_for: stage3
    check_hint: "419 step14 dirs per selected combo; 0 non-ok manifest rows (counted per file)"
    sets_status: finished
    status: completed
    completed_at: 2026-10-09
    note: "verified on disk 2026-10-09: r44_arms/r44_dino_n5_tgt09 has 28509 files (step14 renders); r44_div_abs and r44_rho each 419 npz (absnorm complete)."
  - id: eval-arms
    type: sbatch
    wait_for: wait-stage3
    command: |
      sbatch --dependency=afterok:1027805 --array=0-0 slurm_scripts/five_bench/r44_eval.sh arms
      sbatch --dependency=afterok:1027805 --array=0-0 slurm_scripts/five_bench/r44_fiveacc.sh arms
      sbatch --dependency=afterok:1027805 slurm_scripts/five_bench/r44_clipd.sh arms
    status: completed
    completed_at: 2026-10-07
    note: "job_ids: eval=1027806 fiveacc=1027807 clipd=1027808, all --dependency=afterok:1027805 (stage3); automatic chain per user request -- fiveacc and clipd finished clean (6/6 edit types, r44_clip_directional_arms.csv written); eval still running 2026-10-09, 3/6 edit types done so far."
  - id: wait-eval-arms
    type: manual
    wait_for: eval-arms
    check_hint: "6 harness + 6 five_acc CSVs per selected combo, 419 rows each, 0 NaN; r44_clip_directional_arms.csv = n_combos*419 rows; 0 Error:/OOM"
    status: completed
    completed_at: 2026-10-12
    note: "verified 2026-10-12: all 6 edit-type harness CSVs exist for r44_dino_n5_tgt09 (eval job 1027806 finished, up from 3/6 last check); fiveacc and clipd were already done."
  - id: figures-interim
    type: local
    wait_for: [wait-eval-arms, summarize-sweep]
    command: python evaluation/r44_score.py --all --interim
    output_paths:
      - evaluation/csv/r44_routing_gain_interim.csv
    status: pending
  - id: choose-matched
    type: manual
    wait_for: wait-smoke
    check_hint: "USER DECISION, REQUIRED (not skippable): pick the equal-NFE N, N = 15 + N_d of the chosen config -- N=20 for n5 combos, N=30 for n15 combos. It must match every combo in evaluation/r44_selected_combos.txt (bash slurm_scripts/five_bench/r44_combos.sh --runs); a selection mixing n5 and n15 needs both sweeps"
    status: completed
    completed_at: 2026-10-07
    note: "USER DECISION (memory project_routing_config_final.md): N=20 (=15+5, matches the only selected combo's draft N). bash r44_combos.sh --runs -> n5_tgt09 only, so 15+5=20 is consistent for every selected combo."
  - id: infer-matched
    type: sbatch
    wait_for: choose-matched
    command: sbatch --array=0-7 slurm_scripts/five_bench/r44_infer.sh 20
    status: completed
    completed_at: 2026-10-07
    job_id: "1027773"
    note: "REDUCED 2026-10-08 (user decision): cancelled array indices 1,3,4,6 (rho 3,6,8,20) via scancel -- only rho 2,4,10,50 now run. Reasoning: user judged higher NFE unlikely to change the picture, and this directly frees 4 queue slots under the clusters 30-job submit cap. Kept BOTH grid endpoints (2, 50) rather than the originally-proposed {2,4,10,20}: interp_at() never extrapolates, so dropping an endpoint risks outside_range for any point beyond the remaining span; dropping an interior point only coarsens that stretch. Required code changes before eval-matched/figures can use this reduced curve: r35_score.r36_points() gained an `expected_rhos` override (default unchanged, full 8, every other caller unaffected); r44_score.py gained --text_matched_rhos (VP side / R39 curve untouched, still full 8) and two zip(curve, RHOS) call sites that would have silently mislabelled a reduced curve were fixed to read each points own rho from its label; r44_clipd.sh sweep mode gained an optional 3rd arg (comma-separated rho subset). All tested on a synthetic 4-row fixture before use (r36_points correctly loads/rejects, CSV export and figure end-labels correctly show rho 2/4/10/50, not the old hardcoded RHOS[0]/[-1])."
  - id: wait-matched
    type: manual
    wait_for: infer-matched
    check_hint: "8 tasks exit 0; each r44_n<N>_rho{R}_novp 419 real dirs, 0 non-ok manifest rows"
    status: completed
    completed_at: 2026-10-09
    note: "verified on disk 2026-10-09: all 4 surviving r44_n20_rho{R}_novp arms (2,4,10,50) 419/419 dirs, 0 mismatches, 0 real non-ok rows."
  - id: eval-matched
    type: sbatch
    wait_for: wait-matched
    command: |
      sbatch --array=0,2,5,7 slurm_scripts/five_bench/r44_eval.sh sweep 20
      sbatch --array=0,2,5,7 slurm_scripts/five_bench/r44_fiveacc.sh sweep 20
      sbatch slurm_scripts/five_bench/r44_clipd.sh sweep 20 "2,4,10,50"
    status: completed
    completed_at: 2026-10-12
    note: "chain3 poller (b14ap80wh) also died across a session-compaction boundary before firing (same pattern as the N=15 attempt) -- submitted directly instead once room existed: job_ids eval=1035818 fiveacc=1035819 clipd=1035820, array 0,2,5,7, no --dependency (render pre-verified clean on disk). This filled the queue to exactly 30/30, leaving no room for the figures wrap -- see figures step note."
  - id: wait-eval-matched
    type: manual
    wait_for: eval-matched
    check_hint: "as wait-eval-sweep, for n<N>"
    status: completed
    completed_at: 2026-10-25
    note: "verified on disk: 24/24 harness CSVs, 24/24 fiveacc CSVs (4 rho x 6 edit types), r44_clip_directional_sweep_n20.csv present, 0 Error:/OOM in the 1035818/1035819 metrics logs."
  - id: figures
    type: local
    wait_for: [wait-eval-matched, wait-eval-arms, summarize-sweep]
    command: |
      python evaluation/r36_summarize.py \
        --harness_fmt 'r44_n<N>_rho{rho}_novp' \
        --fiveacc_fmt 'r44_fiveacc_n<N>_rho{rho}_novp' \
        --clipd_csv evaluation/csv/r44_clip_directional_sweep_n<N>.csv \
        --clipd_method_fmt 'r44_n<N>_rho{rho}_novp' \
        --out evaluation/csv/r44_rho_sweep_n<N>.csv
      python evaluation/r44_score.py --all --matched_n 20 --text_matched_rhos 2 4 10 50
    output_paths:
      - evaluation/csv/r44_rho_sweep_n<N>.csv
      - evaluation/csv/r44_arms.csv
      - evaluation/csv/r44_routing_gain.csv
      - evaluation/figures/r44_clip_vs_lpips.pdf
      - evaluation/figures/r44_clipd_vs_lpips.pdf
      - evaluation/figures/r44_fiveacc_vs_lpips.pdf
    sets_status: analyzed
    status: completed
    completed_at: 2026-10-25
    job_id: "1064405"
    note: "every poller attempt (b1mqj02fq/b14ap80wh/bo2l0wukq/bk7zoeis2) died across session-compaction boundaries before firing, across a 16-day gap -- but the underlying eval-matched work (1035818/19/20) had actually completed cleanly in the background regardless (24/24 harness + 24/24 fiveacc CSVs, clipd table, 0 errors), confirmed on disk 2026-10-25. Submitted this step directly (queue had 12 free slots, no poller needed) and it ran clean, exit 0. RESULT: dino_n5_tgt09 vs its own-modality N=20 equal-NFE uniform curve -- CLIP-T gain_text +0.1395 vs gain_vp +0.1264 (diff +0.0131, comparable); CLIP-D (R33 axis of record) gain_text +0.0025 vs gain_vp +0.0052 (both near zero -- routing matches uniform rho, does not clearly beat it, in EITHER modality); FiVE-Acc gain_text -0.0083 vs gain_vp +0.0168 (diff -0.0251, the one real divergence -- VP-routing has a small edge over text-only-routing on this metric). Same pattern holds at N=15 (secondary): CLIP-T diff +0.0398, CLIP-D diff -0.0031, FiVE-Acc diff -0.0314. All gain flags ok (no outside_range), so the reduced 4-point N=20 curve (keeping both grid endpoints) interpolated cleanly as designed. **Headline: the routing gain is NOT strongly tied to Visual Prompting** -- CLIP-T and CLIP-D transfer to text-only at comparable magnitude; only FiVE-Acc shows a modest, reproducible VP edge."
isProject: true
---

# R44: VP-Free Routing + Rho Sweep

## Context

R35/R38 show divergence routing against R36's VP uniform-rho curve, but every stage there runs with visual prompting. R44 asks whether the routing gain survives **without VP**: the same routed pipeline with a text-only instruction (`--vp_mode novp`, no anchor), compared against a **text-only uniform-rho curve**. The method config (draft N, divergence metric, anchoring regime) is not fixed yet. So all 8 combos are built, and only the ones chosen at `choose-config` run. The uniform sweep does not depend on the config and can run right after the smoke. **Equal NFE is mandatory**: a routed arm costs draft + render = 15 + N_d steps, so its headline reference is a text-only uniform curve at N = 15 + N_d (N=20 or N=30), chosen later at `choose-matched`, like the combo.

**Done when:** the text-only 8-rho curves at N=15 **and at the equal-NFE N** and the selected text-only arms are scored on all 419 pairs, and the three two-panel figures plus `r44_routing_gain.csv` put each combo's routing gain over its own modality's **equal-NFE** curve (headline; N=15 secondary) next to the VP gain. Without the equal-NFE curve the task is not done.

## Execution steps

| # | Step id | Type | What it does | Sets status |
|---|---------|------|--------------|-------------|
| — | *(prep)* | — | `/build-step` the 10 `todos` | `implemented` |
| 1 | `smoke` | sbatch | 2 clips, gates G1-G3 | |
| 2 | `wait-smoke` | manual | All gates pass | |
| 3 | `infer-sweep` | sbatch | Text-only rho sweep, N=15, array 0-7 | `running` |
| 4 | `wait-sweep` | manual | 419 dirs per rho | |
| 5 | `eval-sweep` | sbatch | Harness + FiVE-Acc arrays, CLIP-D job | |
| 6 | `wait-eval-sweep` | manual | CSV coverage, 0 errors | |
| 7 | `summarize-sweep` | local | `r44_rho_sweep_n15.csv` | |
| 8 | `choose-config` | manual | **User** writes `r44_selected_combos.txt` | |
| 9 | `stage1` | sbatch | Text-only drafts, one task per needed (N, regime) | `running` |
| 10 | `wait-stage1` | manual | 419 npz per run, `vp_mode == novp` | |
| 11 | `stage2` | sbatch | Divergence npz per selected combo | |
| 12 | `wait-stage2` | manual | 419 npz per combo | |
| 13 | `divstats` | local | Text-only vs VP divergence distributions | |
| 14 | `absnorm` | sbatch | CPU: absolute norm + rho maps | |
| 15 | `stage3` | sbatch | Gated 15-step text-only render | |
| 16 | `wait-stage3` | manual | 419 step14 dirs per combo | `finished` |
| 17 | `eval-arms` | sbatch | Harness + FiVE-Acc arrays, CLIP-D job | |
| 18 | `wait-eval-arms` | manual | CSV coverage, 0 errors | |
| 19 | `figures-interim` | local | Early look vs the N=15 curve only, `*_interim` outputs (not the result) | |
| 20 | `choose-matched` | manual | **User** picks the equal-NFE N: 20 (n5) or 30 (n15). Required | |
| 21 | `infer-matched` | sbatch | Text-only sweep at the equal-NFE N | |
| 22 | `wait-matched` | manual | 419 dirs per rho | |
| 23 | `eval-matched` | sbatch | Harness, FiVE-Acc, CLIP-D | |
| 24 | `wait-eval-matched` | manual | CSV coverage | |
| 25 | `figures` | local | Summarize N + two-panel figures + routing gain vs equal-NFE curves | `analyzed` |

```
/run-step R44 smoke            /run-step R44 choose-config    /run-step R44 stage3
/run-step R44 wait-smoke       /run-step R44 stage1           /run-step R44 wait-stage3
/run-step R44 infer-sweep      /run-step R44 wait-stage1      /run-step R44 eval-arms
/run-step R44 wait-sweep       /run-step R44 stage2           /run-step R44 wait-eval-arms
/run-step R44 eval-sweep       /run-step R44 wait-stage2      /run-step R44 figures-interim
/run-step R44 wait-eval-sweep  /run-step R44 divstats         /run-step R44 choose-matched … wait-eval-matched
/run-step R44 summarize-sweep  /run-step R44 absnorm          /run-step R44 figures
```

## Decisions

| Question | Choice |
|---|---|
| Config management | All 8 combos built: `{dino, lpips}` × `{n5, n15}` × `{unblended, tgt09}`. **Gate before launch**: nothing combo-dependent runs until the user writes `evaluation/r44_selected_combos.txt`. Every combo-dependent script reads that file, so no step command changes with the choice |
| Combo names | `{metric}_n{N}_{regime}`, method `r44_{combo}`; stage-1 runs `n{N}_{regime}`, shared by both metrics |
| VP counterpart per combo | `n15_tgt09` ↔ R35 `{metric}_first2` (τ = 0.9 is the same regime at N=15, R38 gate G1); `n15_unblended` ↔ R35 `{metric}_unblended`; `n5_*` ↔ R38 `{metric}_n5_*`. N=5 combos need R38's eval finished before `figures` |
| Text-only definition | `--vp_mode novp` and **no** `--first_frame_edit_dir`, in sweep, stage 1 and stage 3. Every other flag identical to R35/R36: `--step 15` (stage 3, sweep), `fg_boost 4`, `flow_shift 1.0`, `rollout_chunk_size 21`, `seed 0` |
| Uniform text-only curve | **Full R36 grid**, rho ∈ {2, 3, 4, 6, 8, 10, 20, 50}, N=15, 419 pairs. **rho=2 re-rendered**, not reused from R1 (old code, on `/projects`), as R36 did for R7 |
| Equal NFE (**mandatory**) | A routed arm costs draft + render = 15 + N_d model steps, so the headline reference is a uniform text-only curve at **N = 15 + N_d**: N=20 for n5 combos, N=30 for n15 combos. Which one is chosen later at `choose-matched`, like the combo; it is not skippable. The N=15 curve is kept as the secondary reference. VP side = R39's curve at the same N. A selection mixing n5 and n15 combos would need both sweeps |
| Divergence → rho | Unchanged from R35/R38: stage 2 `r31_divergence.py` (draft vs source, no anchor input), absolute norm `div_min 0`, `div_max` 0.65 DINO / 1.0 LPIPS, `budget_linear`, tau 2..50, `--step 15`. Not recalibrated for text-only; `r44_div_stats.csv` reports how far the text-only maps shift |
| Headline metric | **Routing gain** = arm's x − its own modality's **equal-NFE** uniform curve (N = 15 + N_d; N=15 gain reported as secondary), linearly interpolated at the arm's bg LPIPS (never extrapolated; `outside_range` flagged). Per combo: gain_text, gain_VP, gain_text − gain_VP, for x ∈ {CLIP-target, `clip_d_prompt`, FiVE-Acc yes-no} |
| Figures | `r44_{clip,clipd,fiveacc}_vs_lpips.pdf`: two panels, **VP \| text-only**, shared axes. Each panel shows its uniform curve (gray, ρ-labelled) plus the selected combos (hue = metric, marker = regime, fill = N). y = `lpips_unedit_part`. Plain per-clip mean over 419, never `_avg.csv` |
| Confound check | Without VP the draft itself changes, so the divergence maps do too. `divstats` compares text-only vs VP raw divergence per combo. Read it before interpreting a null gain |
| Smoke clips | `0001_bus` / edit1, `0034_cows` / edit2 (multi-window), as in R38 |
| Cost (L40S) | Sweep 8 × ~10.5 h ≈ **84 GPU-h**. Per N=15 combo: draft ~10.5 h (shared per run) + render ~10.5 h; N=5 draft ~4 h. One N=15 combo ≈ 21 GPU-h; all 8 ≈ 110 GPU-h. Equal-NFE sweep (required): ~110 h (N=20) / ~140–150 h (N=30), from R39's per-task estimates |
| Storage | `~/Data/dataggen/outputs/five_bench/r44_*`. Roughly 15 GB per full-bench render: sweep ≈ 120 GB, plus ~15 GB per draft run and ~15 GB per arm. R35 already hit `Errno 122` (quota) once |
| Node / partition | `--partition=L40S`, `--exclude=node01,node51,node52,node57`, `--mem=64G`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Out of scope | Recalibrated `div_max` for text-only; tau grids; per-category breakdown; `pvp`; LongLive backbone |

## Step commands

### smoke

```bash
sbatch slurm_scripts/five_bench/r44_smoke.sh          # 1 GPU, ~1.5 h
```

### wait-smoke

```bash
grep -h "GATE\|blender_rate per step\|vp_mode=" logs/r44_smoke_*.out
```

### infer-sweep

```bash
sbatch --array=0-7 slurm_scripts/five_bench/r44_infer.sh 15
```

### wait-sweep

```bash
D=~/Data/dataggen/outputs/five_bench/r44_rho_sweep
for m in $D/r44_n15_rho*_novp; do
  echo -n "$(basename $m): $(ls -d $m/edit*/*/ | grep -vc '_resize/$') (expect 419), non-ok "
  for f in $m/edit*/_manifest.csv; do awk -F, 'NR>1 && $3!="ok"' "$f"; done | wc -l
done
grep -L "vp_mode=novp" logs/r44_infer_*.out
```

### eval-sweep

```bash
sbatch --array=0-7 slurm_scripts/five_bench/r44_eval.sh sweep 15
sbatch --array=0-7 slurm_scripts/five_bench/r44_fiveacc.sh sweep 15
sbatch slurm_scripts/five_bench/r44_clipd.sh sweep 15
```

### wait-eval-sweep

```bash
ls evaluation/csv/edit?_FiVE_r44_n15_rho*_novp_frame_stride8.csv | wc -l           # expect 48
ls evaluation/csv/edit?_FiVE_r44_fiveacc_n15_rho*_novp_frame_stride8.csv | wc -l   # expect 48
wc -l evaluation/csv/r44_clip_directional_sweep_n15.csv                            # expect 3353
grep -c "Error:\|out of memory" logs/r44_eval_*.metrics.log logs/r44_fiveacc_*.metrics.log
```

### summarize-sweep

```bash
python evaluation/r36_summarize.py \
  --harness_fmt 'r44_n15_rho{rho}_novp' \
  --fiveacc_fmt 'r44_fiveacc_n15_rho{rho}_novp' \
  --clipd_csv evaluation/csv/r44_clip_directional_sweep_n15.csv \
  --clipd_method_fmt 'r44_n15_rho{rho}_novp' \
  --out evaluation/csv/r44_rho_sweep_n15.csv
```

### choose-config

```bash
# e.g. one combo:
printf '%s\n' dino_n15_tgt09 > evaluation/r44_selected_combos.txt
bash slurm_scripts/five_bench/r44_combos.sh --combos
bash slurm_scripts/five_bench/r44_combos.sh --runs
```

### stage1

```bash
sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --runs | wc -l) - 1 )) \
  slurm_scripts/five_bench/r44_stage1.sh
```

### wait-stage1

```bash
D=~/Data/dataggen/outputs/five_bench
for r in $(bash slurm_scripts/five_bench/r44_combos.sh --runs); do
  echo -n "$r: $(ls $D/r44_latents/$r/edit*/*.npz | wc -l) (expect 419) "
  for T in 1 2 3 4 5 6; do echo -n " e$T=$(ls $D/r44_latents/$r/edit$T/*.npz | wc -l)"; done; echo
done
grep -h "blender_rate per step\|vp_mode=" logs/r44_stage1_*.out | sort | uniq -c
```

### stage2

```bash
sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --combos | wc -l) - 1 )) \
  slurm_scripts/five_bench/r44_stage2.sh
```

### wait-stage2

```bash
D=~/Data/dataggen/outputs/five_bench
for c in $(bash slurm_scripts/five_bench/r44_combos.sh --combos); do
  echo "$c: $(ls $D/r44_div/$c/edit*/*.npz | wc -l) (expect 419)"; done
grep -c "\[ERROR\]\|Traceback" logs/r44_stage2_*.out
```

### divstats

```bash
python evaluation/r44_div_stats.py       # -> evaluation/csv/r44_div_stats.csv
```

### absnorm

```bash
sbatch slurm_scripts/five_bench/r44_absnorm.sh        # selected combos, CPU node
```

### stage3

```bash
sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --combos | wc -l) - 1 )) \
  slurm_scripts/five_bench/r44_stage3.sh
```

### wait-stage3

```bash
D=~/Data/dataggen/outputs/five_bench/r44_arms
for c in $(bash slurm_scripts/five_bench/r44_combos.sh --combos); do
  echo -n "r44_$c: $(ls -d $D/r44_$c/step14/edit*/*/ | wc -l) step14 dirs (expect 419), non-ok "
  for f in $D/r44_$c/_manifest_edit*.csv; do awk -F, 'NR>1 && $3!="ok"' "$f"; done | wc -l
done
```

### eval-arms

```bash
K=$(( $(bash slurm_scripts/five_bench/r44_combos.sh --combos | wc -l) - 1 ))
sbatch --array=0-$K slurm_scripts/five_bench/r44_eval.sh arms
sbatch --array=0-$K slurm_scripts/five_bench/r44_fiveacc.sh arms
sbatch slurm_scripts/five_bench/r44_clipd.sh arms
```

### wait-eval-arms

```bash
for c in $(bash slurm_scripts/five_bench/r44_combos.sh --combos); do
  echo "$c: $(ls evaluation/csv/edit?_FiVE_r44_${c}_frame_stride8.csv | wc -l) harness, $(ls evaluation/csv/edit?_FiVE_r44_fiveacc_${c}_frame_stride8.csv | wc -l) five_acc (expect 6/6)"
done
wc -l evaluation/csv/r44_clip_directional_arms.csv     # expect n_combos*419 + 1
grep -c "Error:\|out of memory" logs/r44_eval_*.metrics.log logs/r44_fiveacc_*.metrics.log
```

### figures-interim

```bash
python evaluation/r44_score.py --all --interim     # N=15 curve only -> *_interim outputs, not the result
```

### choose-matched

```bash
column -s, -t evaluation/csv/r38_arms.csv     # decide N=20 (n5 config) or N=30 (n15 config) -- required
bash slurm_scripts/five_bench/r44_combos.sh --runs   # every run's n{N_d} must give 15 + N_d == N
```

### infer-matched

```bash
sbatch --array=0-7 slurm_scripts/five_bench/r44_infer.sh <N>     # N = 20 or 30
```

### wait-matched

```bash
D=~/Data/dataggen/outputs/five_bench/r44_rho_sweep
for m in $D/r44_n<N>_rho*_novp; do echo "$(basename $m): $(ls -d $m/edit*/*/ | grep -vc '_resize/$') (expect 419)"; done
```

### eval-matched

```bash
sbatch --array=0-7 slurm_scripts/five_bench/r44_eval.sh sweep <N>
sbatch --array=0-7 slurm_scripts/five_bench/r44_fiveacc.sh sweep <N>
sbatch slurm_scripts/five_bench/r44_clipd.sh sweep <N>
```

### wait-eval-matched

```bash
ls evaluation/csv/edit?_FiVE_r44_n<N>_rho*_novp_frame_stride8.csv | wc -l           # expect 48
ls evaluation/csv/edit?_FiVE_r44_fiveacc_n<N>_rho*_novp_frame_stride8.csv | wc -l   # expect 48
wc -l evaluation/csv/r44_clip_directional_sweep_n<N>.csv                            # expect 3353
```

### figures

```bash
python evaluation/r36_summarize.py \
  --harness_fmt 'r44_n<N>_rho{rho}_novp' \
  --fiveacc_fmt 'r44_fiveacc_n<N>_rho{rho}_novp' \
  --clipd_csv evaluation/csv/r44_clip_directional_sweep_n<N>.csv \
  --clipd_method_fmt 'r44_n<N>_rho{rho}_novp' \
  --out evaluation/csv/r44_rho_sweep_n<N>.csv
python evaluation/r44_score.py --all --matched_n <N>
```

## Pipeline

```mermaid
flowchart TD
  BENCH["FiVE-Bench 419 pairs (text only, no anchors)"] --> SM["r44_smoke.sh<br/>G1-G3"]
  SM --> INF["r44_infer.sh 15 (array 0-7)<br/>uniform rho, novp"]
  INF --> EVS["r44_eval · r44_fiveacc · r44_clipd (sweep 15)"]
  EVS --> SUM["r36_summarize.py"]
  SUM --> CURVE["r44_rho_sweep_n15.csv"]
  SM --> GATE{"choose-config<br/>r44_selected_combos.txt"}
  GATE --> S1["r44_stage1.sh<br/>drafts n{5,15}_{unblended,tgt09}"]
  S1 --> S2["r44_stage2.sh<br/>r31_divergence.py"]
  S2 --> DIV["r44_div/{combo}"]
  DIV --> DS["r44_div_stats.py"]
  VPDIV["R35 r35_div · R38 r38_div"] --> DS
  DIV --> ABS["r44_absnorm.sh"]
  ABS --> S3["r44_stage3.sh<br/>gated 15-step novp render"]
  S3 --> EVA["r44_eval · r44_fiveacc · r44_clipd (arms)"]
  EVA --> SC["r44_score.py"]
  CURVE --> SC
  VPC["R36 r36_rho_sweep.csv"] --> SC
  VPA["R35 / R38 per-clip CSVs"] --> SC
  SC --> FIG["r44_{clip,clipd,fiveacc}_vs_lpips.pdf<br/>r44_arms.csv · r44_routing_gain.csv"]
  SM --> GM{"choose-matched (required)<br/>equal NFE: N=20 or N=30"}
  GM --> INFM["r44_infer.sh N + eval + summarize"]
  INFM --> SC
  R39["R39 VP curve at N"] --> SC
```

## Code to touch

No pipeline code changes: `r31_stage1.py`, `r31_stage3.py` and `run_fivebench.py` already accept `--vp_mode novp` (forced when `--first_frame_edit_dir` is absent). `tgt0.9` already exists (R38). `r31_divergence.py` reads only the draft and the source.

**`slurm_scripts/five_bench/r44_combos.sh`** *(new, sourceable)*:

```bash
R44_ALL_COMBOS=(dino_n15_tgt09 dino_n15_unblended lpips_n15_tgt09 lpips_n15_unblended
                dino_n5_tgt09  dino_n5_unblended  lpips_n5_tgt09  lpips_n5_unblended)
R44_SELECTION=$REPO/evaluation/r44_selected_combos.txt
r44_parse() {   # $1 combo -> METRIC ARM N REGIME SCHED RUN
  IFS=_ read -r METRIC NTAG REGIME <<< "$1"
  N=${NTAG#n}; RUN="${NTAG}_${REGIME}"
  case $METRIC in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; *) return 1 ;; esac
  case $REGIME in tgt09) SCHED=tgt0.9 ;; unblended) SCHED=zero ;; *) return 1 ;; esac
  case $N in 5|15) ;; *) return 1 ;; esac
}
r44_selected() { ... }   # non-empty, no comments/blank lines, every entry in R44_ALL_COMBOS, no duplicates; else exit 1
r44_runs()     { ... }   # unique RUN over r44_selected, stable order
# executed directly: --combos | --runs print one per line
```

**`slurm_scripts/five_bench/r44_smoke.sh`** *(new)* — 1 GPU, L40S, 3 h. Writes a 2-clip cases json to `$TMPDIR`; out root `~/Data/dataggen/outputs/five_bench/r44_smoke/`. Gates print `GATE<k> PASS|FAIL`:
- **G1** routed render = uniform render in text-only. `r31_stage3.py --vp_mode novp` on a **constant rho = 2 field** (copy the clip's `r35_rho/dino_first2` npz, fill with 2.0; same shape and frame count) vs `run_fivebench.py --vp_mode novp --blend_power 2`. sha256 of every final PNG. If they are not identical, print max |Δ| and mean LPIPS and FAIL, so the user decides. This licenses comparing the arms to the R44 curve.
- **G2** `r31_stage1.py --vp_mode novp` echoes: N=15 `zero` all 1s, `tgt0.9` `[0,0,1,…]`; N=5 `zero` all 1s, `tgt0.9` `[0,1,1,1,1]`.
- **G3** no anchor leak: stage-1 npz `vp_mode == "novp"`, log line `vp_mode=novp`, and the N=15 `tgt0.9` frames differ (sha256) from R35's VP `first2` frames for the same clip.

**`slurm_scripts/five_bench/r44_infer.sh`** *(new)* — copy of `r36_infer.sh`: `N=${1:-15}`, refuse N ∉ {15, 20, 30}; drop `--first_frame_edit_dir`, pass `--vp_mode novp`; `--step $N`; `METHOD=r44_n${N}_rho${R}_novp`; `OUT_ROOT=$FIVE_ROOT/r44_rho_sweep`; keep the skip-complete logic, count and manifest guards; `--exclude=node01,node51,node52,node57` (missing from r36_infer.sh). `--time=24:00:00` (partition max; R36 measured 11.5–12.8 h at N=15, R39 estimates ~14 h at N=20 and ~17.5–19 h at N=30), with skip-complete on resubmission.

**`slurm_scripts/five_bench/r44_eval.sh`, `r44_fiveacc.sh`** *(new)* — copies of the R36 two. `$1` = `sweep` (needs `$2` = N, array 0-7 over RHOS, dir `r44_rho_sweep/r44_n${N}_rho${R}_novp`) or `arms` (source `r44_combos.sh`, array over `r44_selected`, dir `r44_arms/r44_${COMBO}/step14`). Stem = method name (fiveacc: `r44_fiveacc_…`). Same metric lists, preflight and post-run guards.

**`slurm_scripts/five_bench/r44_clipd.sh`** *(new)* — merge of `r36_clipd.sh` / `r38_clipd.sh` with the same `sweep N | arms` modes. One `--methods name=dir` per target; out `evaluation/csv/r44_clip_directional_sweep_n${N}.csv` or `r44_clip_directional_arms.csv`; per-method 100/100/100/100/9/10 guard.

**`slurm_scripts/five_bench/r44_stage1.sh`** *(new)* — copy of `r38_stage1.sh`: source `r44_combos.sh`, `RUN=$(r44_runs | sed -n "$((TID+1))p")`, parse N and SCHED from it; drop `--first_frame_edit_dir`, pass `--vp_mode novp`; `--step $N --dump_steps $((N-1)) --measure_index $((N-1))`; out `r44_stage1/$RUN`, latents `r44_latents/$RUN`; same per-type guard plus a check that every npz says `vp_mode == novp`.

**`slurm_scripts/five_bench/r44_stage2.sh`** *(new)* — copy of `r38_stage2.sh`: combo from `r44_selected` by `TID`, `r44_parse` replaces the case table; `TARGET_ROOT=r44_stage1/$RUN/step$(printf %02d $((N-1)))`, `LATENT_DIR=r44_latents/$RUN`, `-o r44_div --arm_dir_name $COMBO`, five-bench env.

**`evaluation/r44_div_stats.py`** *(new)* — imports `pooled_d`, `DIV_MAX` from `r38_div_stats.py`. Reads the selection file. For each combo, one row for text-only (`r44_div/{combo}`) and one for the VP counterpart (`r35_div/{metric}_{first2|unblended}` for n15, `r38_div/{combo}` for n5): `modality, combo, N, regime, metric, mean, p50, p90, p99, sat_frac`. Writes `evaluation/csv/r44_div_stats.csv`.

**`slurm_scripts/five_bench/r44_absnorm.sh`** *(new)* — copy of `r38_absnorm.sh`; combos from `r44_selected` (positional args still override); roots `r44_div` → `r44_div_abs` → `r44_rho`; div_max by prefix.

**`slurm_scripts/five_bench/r44_stage3.sh`** *(new)* — copy of `r38_stage3.sh`: combo from `r44_selected` by `TID`; `METHOD=r44_${COMBO}`; drop `--first_frame_edit_dir`, pass `--vp_mode novp`; roots `r44_div_abs` / `r44_rho` → `r44_arms`; per-file manifest guard.

**`evaluation/r44_score.py`** *(new)* — imports `clip_index`, `load_harness`, `load_clipd`, `_check_coverage`, `r36_points` from `r35_score.py` and `interp_at` from `r39_pareto.py` (no copies).

```python
combos = read_selection()                            # hard-fail if missing
vp_curve   = r36_points("evaluation/csv/r36_rho_sweep.csv")
text_curve = r36_points("evaluation/csv/r44_rho_sweep_n15.csv")   # same schema (r36_summarize)
for c in combos:
    text_pt = arm_point(stem=f"r44_{c}", fiveacc=f"r44_fiveacc_{c}", clipd=("r44_clip_directional_arms.csv", f"r44_{c}"))
    vp_pt   = arm_point(**vp_counterpart(c))          # r35_{metric}_{first2|unblended} or r38_{c}; hard-fail if absent
    for x in X_METRICS:
        g_t, f_t = gain(text_pt, text_curve, x); g_v, f_v = gain(vp_pt, vp_curve, x)
        rows.append(dict(combo=c, x_metric=x, gain_text=g_t, flag_text=f_t,
                         gain_vp=g_v, flag_vp=f_v, gain_text_minus_vp=g_t - g_v, ...))
# Equal NFE is MANDATORY: --matched_n {20,30} is required unless --interim.
# Final run: add curves r44_rho_sweep_n{N}.csv (text) and R39's n{N} curve (VP); the
# gain_*_eqnfe columns (vs those curves) are the HEADLINE, the N=15 gains secondary.
# Hard-fail if either equal-NFE curve is missing, or if N != 15 + N_d for any selected combo.
# --interim: N=15 curves only, outputs r44_routing_gain_interim.csv (+ *_interim PDFs if
# plotted), titles marked "INTERIM -- no equal-NFE reference"; never writes the final names.
```

Plot per x-metric: `plt.subplots(1, 2, sharex=True, sharey=True)`, left VP, right text-only; uniform curve gray and ρ-labelled; combos with hue = metric, marker = regime, filled = N=15 / hollow = N=5. Writes `r44_arms.csv` (modality, combo, n_clips, three x metrics, `lpips_unedit_part`) and `r44_routing_gain.csv`.
