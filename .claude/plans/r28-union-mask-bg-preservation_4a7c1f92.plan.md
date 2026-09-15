---
name: R28 — Union-Mask Background Preservation
overview: >-
  Background-preservation metrics in FiVE-Bench are currently scored on the complement of the SOURCE mask alone -- evaluate.py:462 passes `mask, mask`, so `src_mask` doubles as `tgt_mask` and the background region is `1 - M_src`. Any edit that moves, grows or reshapes the object therefore has its new pixels counted as "background that should have been preserved", penalising correct edits and rewarding under-editing. R28 adds parallel `*_unedit_union` metrics scored on `1 - (M_src | M_tgt)`. `M_src` is the FiVE-Bench GT bmask (shipped, per-frame); `M_tgt` is grounded from the METHOD'S OWN rendered frames with the GroundingDINO+SAM2 machinery R25 already validated (median IoU 0.974 against the GT source masks). Masks are precomputed to disk and evaluate.py only reads them, so the shared evaluator gains no new model residency. Scored on R26's 12 arms x 22 cases to answer a live question: does the union mask change R26's "no cell Pareto-beats the control" verdict?
task_id: R28
todos:
  - id: mask-dumper
    content: "evaluation/r28_target_masks.py (new) -- per-arm, per-clip target masks written as npz. Imports `load_models`, `phrase_mask`, `ground_one`, `resize_nearest` from evaluation/r25_iou.py rather than re-implementing them; that module is already validated (iou_gt_src median 0.974). Grounds `trg_word` on the METHOD'S rendered frames at the eval stride, and applies R25's per-type construction rules by edit_type. Run over 14 arms: the 12 R26 cells plus `baseline` (novp Eq.4) and `r7_visual_prompting` (vp Eq.4), both verified 22/22 on the same edit{T}/{video} layout -- the two extra arms exist only to widen the FIXED union. Stores the stride in the npz and refuses a mismatch at read time. Records `fired` and `score` per frame -- GroundingDINO NEVER abstains, so a fired flag is not evidence the phrase is present, only that a box was returned."
    status: completed
  - id: evaluate-union-metrics
    content: "evaluation/fivebench/evaluate.py -- add `psnr_unedit_union`, `lpips_unedit_union`, `mse_unedit_union`, `ssim_unedit_union`, `structure_distance_unedit_union` to `calculate_metric`, plus the same five with a `_fixed` suffix, and `--tgt_mask_dir` / `--fixed_union_dir` arguments. In the per-frame loop build `union = src_mask | tgt_mask` (per-arm) and read the precomputed fixed union, passing `1-union` as BOTH mask_pred and mask_gt in each case, matching the existing `1-src_mask, 1-tgt_mask` call shape. `--tgt_mask_dir` absent => every existing metric is byte-identical and the union metrics return 'nan'. Shared file (R1/R3/R7/R20-R26 all use it): additive only."
    status: completed
  - id: fixed-union
    content: "evaluation/r28_fixed_union.py (new) -- pure reduction, no grounding: OR every arm's M_tgt with M_src into one FIXED union per (edit_type, video), written to r28_tgt_masks/_fixed_union/. Records the contributing arm list inside the npz, because adding an arm can only SHRINK the resulting background -- so fixed-union numbers are comparable only within an identical arm set, and a later run with a different set must fail loudly rather than be silently compared. Also asserts the set-inclusion invariant `fixed_background subset-of every per-arm background`, which must hold by construction."
    status: completed
  - id: mask-sanity-figure
    content: "evaluation/r28_mask_sanity.py (new) -- overlay M_src (GT), M_tgt (grounded) and their union on rendered frames for a spread of clips, one panel row per clip. This is the only artifact that can separate 'the union metric changed the ranking' from 'the target masks are grounding the wrong object'. Mirrors R26's r26_mask_sanity.pdf, which is what caught R26's loose-mask caveat. EXTENDED 2026-09-01 at the user's request ('can you also save this fixed union mask for each video') -- the raw npz was already saved per (edit_type, video) by r28_fixed_union.py's main loop, but nothing visualized it, so a new optional `--fixed_union_dir` argument draws a 5th overlay category (purple): pixels inside the FIXED cross-arm union but outside THIS arm's own `M_src | M_tgt`, i.e. area contributed by another arm. `arm_union subset-of fixed_union` always (asserted in r28_fixed_union.py), so this can only ever ADD a category, never remove one -- omitting the flag reproduces the original 4-colour figure byte-for-byte. This is exactly the quantity that matters for trusting `*_unedit_union_fixed`: how much MORE background the fixed family excludes for an arm versus what that arm's own grounding would have excluded alone."
    status: completed
  - id: extend-to-52-arms
    content: "2026-09-01, user request: R26 extended its (tau_bg, tau_fg) grid from 12 to 52 cells (see the R26 plan). R28's fixed-union family is defined over the ENTIRE arm set (`U = M_src | union_over_arms(M_tgt)`), so adding 40 arms changes the shared background region for EVERY arm, old and new alike -- it is not additive the way the R26 renders were. `slurm_scripts/five_bench/r28_dump.sh` and `r28_eval.sh` ARMS arrays extended from 14 to 54 entries (all 52 R26 cells + `baseline` + `r7_visual_prompting`), `--array` bounds 0-13 -> 0-53, ARMS kept byte-identical in order between the two scripts (verified via diff). `r28_fixed_union.py` needed NO code change -- it discovers arms by listing directories under `--mask_root`, so it will pick up all 54 once dumped. Consequence that must be honoured downstream: the OLD `r28_taubg*_avg.csv` / `r28_union_vs_part.csv` files, scored against the 14-arm union, are STALE the moment the 54-arm union is rebuilt and must not be read after `build-union` and `evaluate` are re-run. Motivated by R26's new fig2 (CLIP vs `ssim_unedit_union_fixed` trade-off across the full 52-cell grid), which needs this metric on arms R28 has never scored. Blocked on the R26 40-arm render extension finishing first (r26_infer.sh tasks 12-51) -- R28 grounds the RENDERED frames, which do not exist yet for the new arms.

CORRECTED 2026-09-01, same session, after the user questioned re-running everything: `r28_dump.sh` and `r28_eval.sh` do NOT need the same scope. Grounding is per-arm and independent -- arm X's `M_tgt` depends only on arm X's own rendered frames, never on any other arm -- so the original 14 arms' masks (already on disk, correct and unchanged) do not need re-grounding, and re-running GroundingDINO+SAM2 on them would be pure waste (it is the expensive step). `r28_eval.sh`, by contrast, DOES need to re-score all 54: `_part` and per-arm `_union` are unaffected by the arm-set size, but `_union_fixed` is scored against the shared fixed-union region, which is rebuilt from all 54 masks and can only grow, so the old 14 arms' `_union_fixed` numbers from the 14-arm run are stale. `evaluate.py` computes all three families in one pass per arm, so there is no cheap way to recompute only `_union_fixed` without a separate reduced-metrics rerun -- not worth the added complexity given eval (no grounding, just precomputed-mask reads + CLIP/LPIPS/etc.) is far cheaper than dump.

Rather than hand-computing which of the 54 array indices are new against R28's differently-sorted ARMS list (bg-major over ALL 52 R26 cells, vs r26_infer.sh's list which kept the original 12 at indices 0-11 and appended the 40 new ones after -- the two orderings do NOT correspond index-for-index), `r28_dump.sh` gained an idempotent skip: if an arm's mask dir already has 22 npz, the task exits immediately with no GPU/model load. This makes a blanket `sbatch --array=0-53 r28_dump.sh` safe and cheap -- it does real grounding only for the 40 arms that actually need it -- rather than a fragile explicit index list. `r28_eval.sh` keeps the full `--array=0-53` with no skip, since every arm genuinely needs rescoring.

NEW SCRIPT, same session: `slurm_scripts/five_bench/r28_fixed_union.sh` -- wraps `r28_fixed_union.py` (previously run as a manual `type: local` step, see the original `build-union` step) as its OWN sbatch job. CPU-only (`--partition=CPU`, no `--gres`), 8 cores, 32G, 1h. Needed because the 54-arm rebuild is chained end-to-end via `sbatch --dependency=afterok:...` (user request, see below) rather than run by hand between dump and eval -- a job in the middle of a dependency chain needs a job ID of its own for `r28_eval.sh` to depend on. Asserts exactly 54 arm dirs under `$MASK_ROOT` before running the reduction (catches a dependency-chain edge case -- e.g. a task that exits 0 without writing all 22 npz -- before it wastes the downstream eval run on a wrong union), duplicating the check `evaluation/r28_fixed_union.py` itself does not make at the shell level.

LAUNCH MECHANISM, same session, user request ('launch both of them and make them dependent on the two last job being ended... how to do that with sbatch in practice'): `slurm_scripts/five_bench/r28_dump.sh` (0-53) -> `r28_fixed_union.sh` -> `slurm_scripts/five_bench/r28_eval.sh` (0-53), chained via `sbatch --dependency=afterok:<jobid>[:<jobid>...]` rather than a polling watcher -- SLURM enforces the ordering natively (a job with an unsatisfied `afterok` dependency sits `PD (Dependency)` until every listed job exits 0; depending on an ARRAY job's base ID waits for the WHOLE array, and if any task fails the dependency is never satisfied, which is exactly the fail-closed behaviour wanted here). Because each of the three stages can itself exceed the ~30-job per-user submit cap (see R26's `extend-grid-2026-09-01`), a throwaway driver script (`r28_chain.sh`, scratchpad-only, not committed to the repo) submits each stage in whatever chunks currently fit under the cap, attaching the SAME dependency string to every chunk of a stage, then collects all of that stage's job IDs (colon-joined) as the `afterok` dependency for the next stage. This means the driver only ever has to solve SUBMIT CAPACITY, never job COMPLETION -- no polling for actual run state is needed anywhere in the chain; SLURM's dependency resolution does that. Launched depending on `afterok:976215:976230` (the last 2 r26_infer render tasks, arms 50-51) -- stage 1 (dump) begins as soon as those finish; stage 2 (fixed_union) begins as soon as ALL of stage 1's chunks succeed; stage 3 (eval) begins as soon as stage 2 succeeds. First dump chunk submitted as job 976426 (array 0-26), sitting `PD (Dependency)` on the two render tasks; remaining chunks submit automatically as queue capacity frees up.

SUPERSEDED, same session: this `afterok`-chain approach broke partway through -- see `dump-masks-52`'s note for the full diagnosis (a dependency referencing an already-completed, already-evicted job fails outright on this cluster, apparently because slurmdbd is unreachable) and its fix (`r28_chain2.sh`, no-dependency submission for preconditions already verified true, polling for completion of this script's own prior stages). Data collection for the full 54-arm extension is now COMPLETE end to end: dump (1188/1188 npz), fixed_union rebuild, eval (54/54 CSVs, 0 errors/nan), and `summarize-52` (108-row `r28_union_vs_part.csv`, `Spearman(part, union_fixed) = +0.9894`, ranking essentially unchanged from the 14-arm run). Only `verdict-52` (re-reading the verdict against the new CSV) remains."
    status: completed
    completed_at: 2026-09-01
  - id: summarize
    content: "evaluation/r28_summarize.py (new) -- join the 14 arm CSVs and emit, per arm, all three of `*_unedit_part`, `*_unedit_union` (per-arm) and `*_unedit_union_fixed` with their deltas, plus the rank correlation between the two orderings, AND a per-arm background-area column (mean fraction of pixels outside the union) -- without it a grounding failure looks identical to a preservation win. The headline output is whether the union metric changes R26's verdict; a Spearman near 1.0 means the metric is a refinement with no scientific consequence, which is itself a publishable negative."
    status: completed
  - id: slurm-scripts
    content: "slurm_scripts/five_bench/r28_smoke.sh, r28_dump.sh (array 0-13, one arm per task: 12 R26 cells + baseline + r7_visual_prompting), r28_eval.sh (array 0-13, ALL 14 arms: the 12 R26 cells plus both Eq.4 baselines). r28_eval.sh's ARMS list is now byte-identical in ORDER to r28_dump.sh's, so an arm cannot get masks without a score or a score without masks. L40S, --mem=64G, --exclude=node52 (that node advertises gpu:8 but exposes no device to batch jobs). r28_eval.sh MUST pass the metric list on the command line: the `metrics:` key in config.yaml is VESTIGIAL (evaluate.py:200 reads `metrics = args.metrics`), a trap that already cost R26 a full 9-arm eval run. EXTENDED 2026-09-01, see `extend-to-52-arms`: r28_dump.sh/r28_eval.sh grown to array 0-53 (54 arms), r28_dump.sh gained an idempotent per-arm skip, and a NEW script `slurm_scripts/five_bench/r28_fixed_union.sh` wraps the fixed-union reduction as its own sbatch job (CPU partition) so it can sit in a `--dependency=afterok` chain between dump and eval."
    status: completed
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r28_smoke.sh
    status: completed
    completed_at: 2026-08-31
    job_id: "965912"

  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "grep -E '^(GATE[0-9]-(OK|FAIL))' logs/r28_smoke_{job_id}.out # gate1 NONREG: evaluate.py without --tgt_mask_dir matches a stored r26 arm CSV within 1e-3 RELATIVE (bit-equality is untestable across nodes; additivity was proven directly by a reverted-copy run); gate2 FIRES: union metrics are non-nan with --tgt_mask_dir; gate3 DIFFERS: *_unedit_union != *_unedit_part on a clip whose object moves (0034_cows) -- without this, gate2 is also satisfied by a union that silently equals M_src; gate4 TYPERULES: edit5 unions with M_src and edit6 target mask is empty; gate5 SUBSET: the fixed-union background is a strict subset of every per-arm background, which must hold by construction and catches a wrong reduction axis. Expect 5 GATE*-OK and no FAIL."
    status: completed
    completed_at: 2026-08-31

  - id: dump-masks
    type: sbatch
    wait_for: wait-smoke
    command: sbatch slurm_scripts/five_bench/r28_dump.sh
    sets_status: running
    status: completed
    completed_at: 2026-08-31
    job_id: "966088"

  - id: wait-dump
    type: manual
    wait_for: dump-masks
    check_hint: "ls /projects/dataggen/outputs/five_bench/r28_tgt_masks/*/edit*/*.npz | wc -l # expect 308 = 14 arms x 22; grep -h 'failures:' logs/r28_dump_*.out # all 0; check the per-clip fg fraction table in the logs for empty (0.0) or all-ones (1.0) masks -- both are grounding failures, not valid data"
    status: completed
    completed_at: 2026-08-31

  - id: build-union
    type: local
    wait_for: wait-dump
    command: |
      python evaluation/r28_fixed_union.py \
        --mask_root /projects/dataggen/outputs/five_bench/r28_tgt_masks \
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
        --cases evaluation/cases.json \
        -o /projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union
    output_paths:
      - /projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union
    status: completed
    completed_at: 2026-08-31

  - id: evaluate
    type: sbatch
    wait_for: build-union
    command: sbatch slurm_scripts/five_bench/r28_eval.sh
    status: completed
    completed_at: 2026-08-31
    job_id: "966177"

  - id: wait-eval
    type: manual
    wait_for: evaluate
    check_hint: "ls evaluation/csv/r28_*_avg.csv | wc -l # expect 14 (12 R26 cells + baseline + r7_visual_prompting); grep -h 'error_lines=' logs/r28_eval_*.out # all 0; confirm every CSV carries ALL THREE families -- *_unedit_part, *_unedit_union, *_unedit_union_fixed -- and no nan in any"
    sets_status: finished
    status: completed
    completed_at: 2026-08-31

  - id: summarize
    type: local
    wait_for: wait-eval
    command: |
      ~/anaconda3/envs/streamgve/bin/python evaluation/r28_summarize.py \
        --arm_glob 'evaluation/csv/r28_*_avg.csv' \
        --control taubg2_taufg2_vp \
        --novp_baseline baseline \
        --cases evaluation/cases.json \
        -o evaluation/csv/r28_union_vs_part.csv
    output_paths:
      - evaluation/csv/r28_union_vs_part.csv
    status: completed
    completed_at: 2026-08-31

  - id: mask-sanity
    type: local
    wait_for: wait-dump
    command: |
      python evaluation/r28_mask_sanity.py \
        --tgt_mask_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/taubg2_taufg2_vp \
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
        --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg2_taufg2_vp \
        --cases evaluation/cases.json \
        -o evaluation/figures/r28_mask_sanity.pdf
    output_paths:
      - evaluation/figures/r28_mask_sanity.pdf
    status: completed
    completed_at: 2026-09-01

  - id: verdict
    type: manual
    wait_for: summarize
    check_hint: "Read evaluation/csv/r28_union_vs_part.csv together with evaluation/figures/r28_mask_sanity.pdf. Does scoring on the union change R26's ranking, and specifically does any cell now Pareto-beat the (2,2) control on (CLIP-T, lpips_unedit_union) and on (CLIP-T, lpips_unedit_union_fixed)? Record in daily.md, and state the Spearman between the two orderings -- a value near 1.0 means the fix is a refinement with no scientific consequence, which must be reported as such rather than buried. ⚠️ MANDATORY CAVEAT: masks are PER-ARM, so every arm is scored over a different background region. Before calling any ranking change real, check the per-arm background-area column: an arm whose grounding under-fired has a LARGER background region and an easier score, which is a measurement artifact and not a preservation win."
    sets_status: analyzed
    status: completed
    completed_at: 2026-09-01
    note: "SUPERSEDED-PENDING 2026-09-01: this verdict was computed on the 14-arm run (12 R26 cells + 2 baselines). The 54-arm rebuild below (dump-52/build-union-52/evaluate-52/summarize-52) will overwrite evaluation/csv/r28_union_vs_part.csv with numbers scored against a DIFFERENT (larger) fixed-union region -- re-read this verdict against the new CSV once that lands rather than treating this entry as final."

  - id: dump-masks-52
    type: sbatch
    wait_for: verdict
    command: "sbatch --dependency=afterok:<last 2 r26_infer render tasks> --array=0-53 slurm_scripts/five_bench/r28_dump.sh   # chunked under the submit cap by scratchpad-only r28_chain.sh; idempotent skip makes this cheap for the 14 already-dumped arms"
    sets_status: finished
    status: completed
    completed_at: 2026-09-01
    job_id: "976426 (0-26), 976446 (27), 976503 (28-41), 976517 (42-44), 976529 (45), 976537 (46-48), 976548 (49-53) -- all completed"
    note: "BUG FOUND AND FIXED 2026-09-01 ~16:29: the original r28_chain.sh chained every stage via `sbatch --dependency=afterok:<jobid>`, including repeated NEW submissions depending on the SAME 2 render job ids (976215, 976230) across multiple capacity-limited retries over ~20 minutes. Once those render jobs finished and left the live squeue table, a NEW dependency submission referencing them started failing outright with `sbatch: error: Batch job submission failed: Job dependency problem` -- apparently resolving a dependency on an already-evicted job needs the slurmdbd accounting DB, which is unreachable on this cluster (`sacct`/`sacctmgr` return `Connection refused`, same symptom seen earlier this session). r26_eval_remainder.sh hit the identical failure on ITS second chunk, referencing the same 2 job ids. Both watchers were stuck retrying every 60s against a dependency that could never resolve. FIX: killed both broken watchers (TaskStop); replaced with r28_chain2.sh (scratchpad-only), which (a) submitted the remaining dump chunks with NO dependency at all, since the render precondition was independently verified true (52/52 R26 arm dirs on disk), and (b) for fixed_union and eval -- which had to wait for THIS script's own prior stage to actually finish -- POLLED for completion itself (squeue absence by job name, plus an npz-count / directory-existence check) rather than depending on a job old enough to have left live squeue. All 54 arms dumped cleanly: 1188/1188 npz, 0 failures across every log."

  - id: wait-dump-52
    type: manual
    wait_for: dump-masks-52
    check_hint: "find /projects/dataggen/outputs/five_bench/r28_tgt_masks -mindepth 1 -maxdepth 1 -type d ! -name _fixed_union | wc -l  # expect 54; ls /projects/dataggen/outputs/five_bench/r28_tgt_masks/*/edit*/*.npz | wc -l  # expect 54*22=1188; grep -h 'failures:' logs/r28_dump_*.out  # all 0"
    status: completed
    completed_at: 2026-09-01
    note: "Verified 2026-09-01: 54 arm dirs, 1188/1188 npz, 0 failures."

  - id: build-union-52
    type: sbatch
    wait_for: wait-dump-52
    command: "sbatch slurm_scripts/five_bench/r28_fixed_union.sh   # NO --dependency (see dump-masks-52's note): r28_chain2.sh polled for all r28_dump jobs to clear squeue AND verified 1188 npz on disk before submitting this"
    status: completed
    completed_at: 2026-09-01
    job_id: "976585"

  - id: wait-union-52
    type: manual
    wait_for: build-union-52
    check_hint: "grep -h 'OK ->' logs/r28_fixed_union_*.out; python -c \"import numpy as np; d=np.load('/projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union/edit1/<a_video>.npz'); print(list(d['arms']), len(d['arms']))\"  # expect 54 arms listed"
    status: completed
    completed_at: 2026-09-01
    note: "logs/r28_fixed_union_976585.out ends 'exit=0 / OK -> .../_fixed_union', marginal-contribution table printed with no dominant arm flagged."

  - id: evaluate-52
    type: sbatch
    wait_for: wait-union-52
    command: "sbatch --array=0-53 slurm_scripts/five_bench/r28_eval.sh   # NO --dependency, same reason as build-union-52; r28_chain2.sh polled for the fixed_union job to clear squeue AND verified _fixed_union/ exists before submitting these, chunked under the submit cap"
    sets_status: finished
    status: completed
    completed_at: 2026-09-01
    job_id: "976589, 976599, 976603, 976616, 976625, 976640, 976649, 976667, 976679, 976695, 976698, 976702, 976707, 976718, 976735 -- all completed"

  - id: wait-eval-52
    type: manual
    wait_for: evaluate-52
    check_hint: "ls evaluation/csv/r28_*_avg.csv | wc -l  # expect 54; grep -h 'error_lines=' logs/r28_eval_*.out  # all 0; confirm ALL THREE families populated, no nan"
    status: completed
    completed_at: 2026-09-01
    note: "Verified 2026-09-01: 54/54 r28_*_avg.csv present, 0 files with 'error_lines=' != 0 or oom, 0 files containing 'nan'."

  - id: summarize-52
    type: local
    wait_for: wait-eval-52
    command: |
      ~/anaconda3/envs/streamgve/bin/python evaluation/r28_summarize.py \
        --arm_glob 'evaluation/csv/r28_*_avg.csv' \
        --control taubg2_taufg2_vp \
        --novp_baseline baseline \
        --cases evaluation/cases.json \
        -o evaluation/csv/r28_union_vs_part.csv
    output_paths:
      - evaluation/csv/r28_union_vs_part.csv
    status: completed
    completed_at: 2026-09-01
    note: "108 rows (54 arms x 2 subsets), exit 0. best by fixed unchanged: taubg0_taufg2_vp. Spearman(part, union_fixed) = +0.9894 (all), +0.9808 (excl_edit6) -- ordering essentially unchanged from the original 14-arm run. Spearman(part, union) (per-arm, NOT the ranking family) is now -0.8159/-0.8769, more negative than the 14-arm run's earlier reading -- expected, since union remains confounded with background area and the grid now spans a much wider area range (per-arm background area 0.83-0.89 across 52 cells vs the tighter 12-cell spread); does not affect the fixed-family conclusion."

  - id: verdict-52
    type: manual
    wait_for: summarize-52
    check_hint: "Re-read evaluation/csv/r28_union_vs_part.csv (now 54 arms) against the ORIGINAL verdict above. Does the larger fixed-union region change which arms Pareto-beat (2,2)? This also unblocks R26's fig2 (evaluation/r26_tradeoff_figure.py --which union_fixed), which needs ssim_unedit_union_fixed on the full 52-cell grid."
    status: completed
    completed_at: 2026-09-01
    note: "VERDICT: UNCHANGED at 52/54-arm scale, same conclusion as the original 14-arm run. Joined evaluation/csv/r26_spatial_tau.csv (CLIP-T, lpips_unedit_part) with evaluation/csv/r28_union_vs_part.csv (lpips_unedit_union_fixed, lpips_unedit_union) by (tau_bg, tau_fg), checked every non-control cell against the (2,2) control for Pareto dominance on (CLIP-T higher, LPIPS lower): subset=all -- `part` 0/51 wins, `union_fixed` 0/51 wins, `union` (per-arm, confounded) 45/51 'wins' -- MORE than the original 9/11, as expected since the confound (Spearman(area, union) now -0.8159, more negative on the wider grid) scales with how many cells have a smaller-than-control per-arm background. subset=excl_edit6 -- `part` 0/51; `union_fixed` shows 2 near-ties, (0,3) 0.172900 vs control 0.173029 (delta -0.000129) and (1,3) 0.172974 (delta -0.000055) -- BOTH deltas are smaller than the 9.43e-05/1e-3-relative noise floor this session has repeatedly measured for cross-node scoring drift (R26's own control cross-check, R28's smoke gate1), so these are NOT robust Pareto wins, they are indistinguishable from measurement noise. **Conclusion: no cell Pareto-beats (2,2) under either properly-controlled family, at 52/54-arm resolution just as at 12/14.** R26's fig2 is now unblocked -- ssim_unedit_union_fixed is populated across the full 52-cell grid."
isProject: true
---

# R28: Union-Mask Background Preservation

## Context

FiVE-Bench scores background preservation on `1 - M_src`. [evaluate.py:462](../../evaluation/fivebench/evaluate.py#L462) calls `calculate_metric(..., mask, mask, ...)`, passing the source mask as both `src_mask` and `tgt_mask`, so every `*_unedit_part` metric treats "background" as everything the source object did not occupy. An edit that moves, grows or reshapes the object puts its new pixels inside that region, where they are scored as failed preservation. The metric therefore penalises correct edits and rewards under-editing — the opposite of what it is meant to measure.

R28 adds parallel `*_unedit_union` metrics scored on `1 - (M_src | M_tgt)`, the region that legitimately changed in neither video. `M_src` ships with FiVE-Bench (`bmasks/{video}/`, 101 videos, per-frame). `M_tgt` does not exist and must be produced: it is grounded from the method's own rendered frames using the GroundingDINO+SAM2 pipeline `evaluation/r25_iou.py` already implements and validated at median IoU 0.974 against the GT source masks.

**Done when:** `evaluation/csv/r28_union_vs_part.csv` and `evaluation/figures/r28_mask_sanity.pdf` are read together, and `daily.md` records whether scoring on the union changes R26's "no cell Pareto-beats the (2,2) control" verdict, with the masks confirmed sound.

## Execution steps

| # | Step id | Type | What | Sets status |
|---|---------|------|------|-------------|
| — | *(prep)* | — | All `todos` — mask dumper, evaluate.py metrics, sanity figure, summarize, Slurm scripts | `implemented` |
| 1 | `smoke` | sbatch | 4 gates on 2 clips — non-regression, union fires, union differs, per-type rules | |
| 2 | `wait-smoke` | manual | All 4 gates before committing 12 arms | |
| 3 | `dump-masks` | sbatch | Target masks for 14 arms × 22 clips | `running` |
| 4 | `wait-dump` | manual | 308 npz, no empty/all-ones masks | |
| 5 | `build-union` | local | Reduce all arms into one fixed union | |
| 6 | `evaluate` | sbatch | Score all 14 arms, all three families | |
| 7 | `wait-eval` | manual | 14 CSVs, three column families, no nan | `finished` |
| 8 | `summarize` | local | part vs union vs union_fixed, areas, rank correlation | |
| 9 | `mask-sanity` | local | M_src / M_tgt / union overlay | |
| 10 | `verdict` | manual | Does either union change R26's verdict? | `analyzed` |

```
/run-step R28                 # next pending step
/run-step R28 dump-masks      # a named step
```

## Decisions

| Question | Choice | Note |
|---|---|---|
| Source mask | FiVE-Bench GT `bmasks/{video}/{frame}.jpg` | Shipped, per-frame, already loaded by evaluate.py:302-316. Never grounded — R25 grounded it only to validate the grounder |
| Target mask | GroundingDINO + SAM2 on the **method's own rendered frames** | Reuses `evaluation/r25_iou.py`; grounding on *generated* frames is untested, which is what smoke gate 3 exists for |
| Mask scope across arms | **Per-arm** — every arm grounds `M_tgt` on its OWN output (user decision, 2026-08-31) | Each arm is judged on what it actually changed, which is the faithful per-method reading. ⚠️ CONSEQUENCE: arms are scored over DIFFERENT background regions, so cross-arm deltas are not a like-for-like pixel comparison. The rejected alternative was a fixed union (`M_src ∪ ⋃ M_tgt` over all arms), which is commensurable but over-excludes for conservative arms |
| Reporting that consequence | `r28_summarize.py` emits per-arm background AREA alongside every metric | An arm whose grounding failed gets a larger background region and an easier score; without the area column that is invisible and would read as a genuine preservation win |
| 🔴 The area column EARNED ITS PLACE (measured 2026-08-31) | `Spearman(per-arm background AREA, lpips_unedit_union) = +0.972` | The per-arm union ordering is almost entirely explained by HOW MUCH background each arm was scored on, not by preservation. Aggressive arms (`tau_fg=50`) produce a larger `M_tgt`, hence a smaller background (area 0.839 vs 0.895), and score best — an artifact. **`*_unedit_union` must NOT be used to rank arms against each other**; it answers only "did THIS arm preserve what IT left alone" |
| Which family ranks arms | **`*_unedit_union_fixed`** — identical region for every arm (0.8234 on all 14, invariant verified) | `Spearman(part, union_fixed) = +0.993`, so the properly-controlled comparison PRESERVES R26's ordering. What changes is magnitude, not rank: the between-arm LPIPS spread falls from **0.0455 (part) to 0.0067 (fixed)**, ~7x. R26's ranking conclusion survives; the size of the `tau_fg` penalty was inflated ~7x by the metric |
| **Second family: fixed union** | `*_unedit_union_fixed` on `1 - (M_src ∪ ⋃_arms M_tgt)` over **54 arms** (extended 2026-09-01 from 14) — all 52 R26 cells + `baseline` (novp) + `r7_visual_prompting` (vp) | Identical background region for every arm, so cross-arm deltas ARE like-for-like. Costs one extra reduction pass and 2 extra arms of grounding; no extra eval. Growing the arm set can only GROW the union (never shrink it), so the 54-arm region is a superset of the 14-arm one and every arm's `_fixed` score can only move in the more-conservative (smaller-background) direction |
| ⚠️ Fixed-union failure mode | A single OVER-firing arm shrinks the background for **every** arm | Exact mirror of the per-arm mode (under-firing inflates one arm's score). `r28_summarize.py` reports each arm's marginal contribution to the union so one bad arm is attributable rather than diffuse |
| Fixed-union reproducibility | The contributing arm list is stored **inside** the npz; a read with a different set is a hard failure | Adding an arm can only SHRINK the background, so fixed-union numbers are comparable only within an identical arm set. Silent recomposition would invalidate every delta |
| Which arms get scored | **All 54** — all 52 R26 `(tau_bg, tau_fg)` cells PLUS both Eq.4 baselines | Extended 2026-08-31 (12→14, both baselines) then AGAIN 2026-09-01 (14→54) once R26 grew its grid from 12 to 52 cells. **`r7_visual_prompting` (vp) is the designated "StreamGVE baseline"** (user-confirmed 2026-08-31, correcting an earlier reading of `baseline`/novp). `baseline` (novp) is scored alongside at no extra grounding cost and can be ignored. The fixed union must be rebuilt over all 54 before ANY arm — including the original 14 — is re-scored; see `extend-to-52-arms` |
| ✅ No vp confound | Designated baseline `r7_visual_prompting` is **vp**, and all 12 R26 cells are **vp** | The comparison moves ONE variable (the tau field), which is what makes an R26-vs-baseline difference attributable to tau. Had `baseline` (novp) been the comparator it would have moved vp as well — kept in the run as the separate "what does the full pipeline buy over the paper method" number, not as the tau comparator |
| Baseline is duplicated | `r7_visual_prompting` and the R26 **(2,2)** cell are **bit-identical renders** (sha256-verified on 3 clips), not merely the same configuration | Measured 2026-08-31. They score EXACTLY equal on all six union/part columns (rel 0.00e+00), so this is a metric-DETERMINISM check (same input, same output), not the independent-render reproducibility check first assumed. It also explains R26's 9.43e-05 control cross-check: that gap was cross-node SCORING drift against a stored CSV, not any difference between the renders || ⚠️ Conda environment | **`streamgve` for everything** — grounding AND evaluation | GroundingDINO + SAM2 were INSTALLED INTO streamgve on 2026-08-31 at the user's request so the task does not straddle two envs: `SAM-2 1.0` (facebookresearch/sam2 @2b90b9f5, `--no-deps`) + `hydra-core` + `iopath` + `portalocker` + `qwen-vl-utils`. streamgve already carried **transformers 5.12.0**, which exposes the `threshold` kwarg `r25_iou.py` calls — 4.44 (addit, DGE) calls it `box_threshold` and raises TypeError, which killed job 965789 |
| Install safety | 4 packages added, **zero pre-existing versions changed**; WAN pipeline re-imported to confirm | `pip freeze` diffed before/after: only additions. `WanVAEWrapper`, `wan.modules.causal_model` and `pipeline.edit_causal_inference` all still import, with transformers 5.12.0 / torch 2.8.0+cu128 unchanged — so no R-task's rendering is affected. Snapshot kept for revert |
| Interpreter form | Absolute path `~/anaconda3/envs/streamgve/bin/python`, not `conda activate` | Removes activation stacking as a failure mode entirely (the R2 job-880402 class of failure) |
| Phrase grounded | `trg_word` from `cases.json` | One phrase per side. R25 proved a symmetric multi-phrase union adds false positives on every case |
| Per-type rules | 1-4: `M_tgt = m(trg_word)` · 5: `m(trg_word) \| M_src` · 6: `M_tgt = ∅` | Copied from R25. Type 5 grows the region, type 6 removes it; both by construction, not detection |
| **GroundingDINO never abstains** | Record `fired` + `score`, never threshold on them alone | It returns its best box unconditionally — R25 measured 0.94 for "A dragon" on an unedited cow. A fired flag is not evidence of presence |
| New columns | `{psnr,lpips,mse,ssim,structure_distance}_unedit_union` | **Additive.** `*_unedit_part` untouched, so every stored R20-R26 reference CSV stays comparable |
| Masking convention | Multiply image by `1-union`, do not crop | Matches the existing `_unedit_part` convention exactly, so the two families are comparable; a cropped variant would differ for reasons unrelated to the mask |
| Where grounding runs | **Precomputed to disk**, evaluate.py only reads npz | evaluate.py is shared by R1/R3/R7/R20-R26; adding two more resident models is exactly what caused the R20/R21/R26 eval OOMs (Qwen2.5-VL + CoTracker at 44 GB) |
| Mask store | `/projects/dataggen/outputs/five_bench/r28_tgt_masks/{arm}/edit{T}/{video}.npz` | `np.packbits`, `[n_frames, H, W]` bool + `stride` + per-frame `fired`/`score` |
| Frame stride | Dump at the **eval stride (8)**, recorded in the npz | Stride 1 would be 12x the grounding calls for frames evaluate.py never scores. A stride mismatch at read time is a hard failure |
| Scope | R26's **52 arms × 22 cases** (extended 2026-09-01 from 12) | Smallest set answering a live question, later widened to feed R26's fig2 trade-off plot across its full grid. Renders under `r26_spatial_tau/`, dependent on the R26 40-arm render extension finishing first |
| Validation | `iou_tgt_src` diagnostic + empty/all-ones gate + overlay figure | IoU is a *diagnostic*, not a gate: types 5 and 6 legitimately diverge from `M_src` |
| ⚠️ `config.yaml` metrics | **Vestigial** — pass `--metrics` on the command line | `evaluate.py:200` reads `metrics = args.metrics`. Editing the config does nothing; this already cost R26 a full 9-arm eval run |
| Out of scope | Full 419-pair bench, other methods' outputs, retiring `*_unedit_part`, temporal metrics | Follow-ups once the union metric is trusted |

## Step commands

### smoke
```bash
sbatch slurm_scripts/five_bench/r28_smoke.sh
```

### dump-masks
```bash
sbatch slurm_scripts/five_bench/r28_dump.sh
# inside, per array task = one of 14 arms (12 r26 cells + baseline + r7_visual_prompting):
#   HF_HUB_OFFLINE=1 python evaluation/r28_target_masks.py \
#     --cases evaluation/cases.json \
#     --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
#     --tgt_root /projects/dataggen/outputs/five_bench/r26_spatial_tau/${ARM} \
#     --out_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/${ARM} \
#     --frame_stride 8 --seed 0
```

### build-union
```bash
python evaluation/r28_fixed_union.py \
  --mask_root /projects/dataggen/outputs/five_bench/r28_tgt_masks \
  --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
  --cases evaluation/cases.json \
  -o /projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union
```

### evaluate
```bash
sbatch slurm_scripts/five_bench/r28_eval.sh
# inside, per array task = one arm; METRICS PASSED ON THE COMMAND LINE:
#   python evaluation/fivebench/evaluate.py \
#     --config_path evaluation/fivebench/config.yaml \
#     --metrics structure_distance psnr_unedit_part lpips_unedit_part \
#               mse_unedit_part ssim_unedit_part \
#               psnr_unedit_union lpips_unedit_union \
#               mse_unedit_union ssim_unedit_union \
#               psnr_unedit_union_fixed lpips_unedit_union_fixed \
#               mse_unedit_union_fixed ssim_unedit_union_fixed \
#               clip_similarity_target_image \
#     --tgt_mask_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/${ARM} \
#     --fixed_union_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union \
#     --tgt_methods /projects/dataggen/outputs/five_bench/r26_spatial_tau/${ARM} \
#     --tgt_layout edit_video --cases_json evaluation/cases.json \
#     --result_path evaluation/csv/r28_${ARM}.csv
```

### summarize
```bash
~/anaconda3/envs/streamgve/bin/python evaluation/r28_summarize.py \
  --arm_glob 'evaluation/csv/r28_*_avg.csv' \
  --control taubg2_taufg2_vp \
  --novp_baseline baseline \
  --cases evaluation/cases.json \
  -o evaluation/csv/r28_union_vs_part.csv
```

### mask-sanity
```bash
python evaluation/r28_mask_sanity.py \
  --tgt_mask_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/taubg2_taufg2_vp \
  --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
  --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg2_taufg2_vp \
  --cases evaluation/cases.json \
  -o evaluation/figures/r28_mask_sanity.pdf
```

### verdict
```bash
# manual: read evaluation/csv/r28_union_vs_part.csv with evaluation/figures/r28_mask_sanity.pdf
```

## Pipeline

```mermaid
flowchart TD
    A[FiVE-Bench bmasks/<br/>GT source masks M_src] --> D
    B[r26_spatial_tau/{arm}/<br/>12 arms x 22 rendered clips] --> C
    C[r28_target_masks.py<br/>GroundingDINO + SAM2 on trg_word<br/>14 arms] --> D[r28_tgt_masks/{arm}/edit{T}/{video}.npz<br/>M_tgt + fired/score + stride]
    B2[baseline + r7_visual_prompting<br/>2 extra arms, union only] --> C
    D --> U[r28_fixed_union.py<br/>OR over all 14 arms] --> V[_fixed_union/edit{T}/{video}.npz<br/>+ contributing arm list]
    D --> E[evaluate.py --tgt_mask_dir --fixed_union_dir<br/>*_unedit_union and *_unedit_union_fixed]
    V --> E
    B --> E
    E --> F[evaluation/csv/r28_{arm}_avg.csv<br/>three metric families]
    F --> G[r28_summarize.py]
    G --> H[evaluation/csv/r28_union_vs_part.csv]
    D --> I[r28_mask_sanity.py]
    I --> J[evaluation/figures/r28_mask_sanity.pdf]
    H --> K[verdict: does the union change R26's ranking?]
    J --> K
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r28_target_masks.py` | **new** — per-arm target-mask dumper; imports `load_models` / `phrase_mask` / `ground_one` / `resize_nearest` from `evaluation/r25_iou.py`; per-type rules; npz with `M`, `stride`, `fired`, `score` |
| `evaluation/fivebench/evaluate.py` | Add 5 `*_unedit_union` + 5 `*_unedit_union_fixed` branches to `calculate_metric` (:44), `--tgt_mask_dir` and `--fixed_union_dir` args (:667), and per-clip mask loading in the frame loop (:442). Both `None` => byte-identical to today |
| `evaluation/r28_fixed_union.py` | **new** — pure reduction over the 14 dumped arms into one fixed union per clip; stores the contributing arm list; asserts `fixed_background ⊆ every per-arm background` |
| `evaluation/r28_summarize.py` | **new** — per-arm `part` vs `union` with deltas and the Spearman between the two arm orderings |
| `evaluation/r28_mask_sanity.py` | **new** — M_src / M_tgt / union overlay on rendered frames, one row per clip |
| `slurm_scripts/five_bench/r28_smoke.sh` | **new** — L40S, `--mem=64G`, `--exclude=node52`, 5 gates on 2 clips (incl. the fixed-union subset invariant) |
| `slurm_scripts/five_bench/r28_dump.sh` | **new** — `--array=0-13` (12 r26 arms + baseline + r7_visual_prompting), L40S, `--mem=64G`, `--time=04:00:00`, `HF_HUB_OFFLINE=1` |
| `slurm_scripts/five_bench/r28_eval.sh` | **new** — `--array=0-11`, L40S, `--mem=64G`, `--time=06:00:00`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, explicit `--metrics` |

New metric branches in `calculate_metric`, mirroring the existing `_unedit_part` shape exactly:

```python
# `tgt_mask` is now a REAL target mask when --tgt_mask_dir is set, not a copy of src_mask.
# union = the region that changed in EITHER video; its complement is the only region where
# "preserved" is a fair ask.
if metric.endswith("_unedit_union") or metric.endswith("_unedit_union_fixed"):
    # per-arm: this arm's own M_tgt. fixed: the OR over all 14 arms, identical for every
    # arm, so cross-arm deltas are like-for-like. Both are supersets of M_src, so both
    # backgrounds are subsets of today's `1 - M_src`.
    union = fixed_union if metric.endswith("_fixed") else np.clip(src_mask + tgt_mask, 0, 1)
    if (1 - union).sum() == 0:
        return "nan"                      # union covers the frame; no background to score
    base = metric[: -len("_unedit_union")]
    fn = {"psnr": metrics_calculator.calculate_psnr,
          "lpips": metrics_calculator.calculate_lpips,
          "mse": metrics_calculator.calculate_mse,
          "ssim": metrics_calculator.calculate_ssim,
          "structure_distance": metrics_calculator.calculate_structure_distance}[base]
    # Same call shape as `_unedit_part`: multiply-by-mask, not crop, so the two families
    # differ ONLY by which region is excluded.
    return fn(src_image, tgt_image, 1 - union, 1 - union)
```

Per-type target-mask construction in `r28_target_masks.py`, copied from R25 rather than re-derived:

```python
# GroundingDINO NEVER abstains -- it returns its best box unconditionally, so a phrase
# naming something absent still "fires" (R25 measured 0.94 for 'A dragon' on an unedited
# cow). Types 5 and 6 are therefore handled by CONSTRUCTION, never by detection.
if edit_type in (1, 2, 3, 4):          # swap / colour / material
    m_tgt = phrase_mask(trg_word, tgt_frame)
elif edit_type == 5:                    # addition: the region GREW
    m_tgt = phrase_mask(trg_word, tgt_frame) | m_src
elif edit_type == 6:                    # removal: nothing to ground on the target
    m_tgt = np.zeros_like(m_src)        # => union == m_src, i.e. today's behaviour
```
