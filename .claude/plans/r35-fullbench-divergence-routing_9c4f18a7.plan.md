---
name: "R35: Full-Bench Spatial Divergence Routing"
overview: "R31's spatial divergence routing taken to the whole 419-pair FiVE-Bench, with ABSOLUTE (not per-clip) divergence normalisation, restricted to the DINO and LPIPS arms, and crossed with two stage-1 regimes (UNBLENDED / FIRST2) to give four arms."
task_id: R35
isProject: true

todos:
  - id: stage1-blend-flag
    content: "evaluation/r31_stage1.py -- expose --blend_sched. Line 172 currently PINS `args.blend_sched = \"zero\"` with a deliberate comment ('the Q/K blend is the POINT of this run, not an option ... so no caller can produce a stage 1 output that was silently blended'). FIRST2 cannot be produced at all until this opens up. Open it to the {zero, first1, first2, first3} set ONLY, keep `zero` as the default so every existing caller is byte-identical, and keep the guard's intent by echoing the regime at startup and writing `blend_sched` into the npz (it already does, line 405). Print the resolved s(p)/blender_rate vector at startup, as r35_blendfirst.sh's gate does."
    status: completed
    note: "BUILT 2026-09-24. Added --blend_sched with choices (zero|first1|first2|first3), default zero; removed the pin at the old line 172; startup now resolves the regime through pipeline.utils._schedule_blend_rate and echoes the per-step blender_rate. VERIFIED: default is still zero (pre-R35 callers unchanged); paper/cos_third/first9/zzz all rejected, so the set is closed; resolved vectors correct at step 15. Reminder for r35_stage1.sh: pass --measure_index 14 --dump_steps 14 (already in Decisions) -- the script's own default of 6 warns but does not refuse."
  - id: fullbench-cases
    content: "Confirm evaluation/r7_anchor_manifest.json is usable as the full-bench case list for stage 2 and the tau grids: {video_name, edit_type} only -- the two fields r31_divergence.py and r31_stage3_grids.py actually read. Stage 1 and stage 3 take NO --cases_json (absent = full bench via run_fivebench's own edit{T}_FiVE.json), so only stage 2 and the grids need this file. Assert 419 == 100/100/100/100/9/10 by edit type, and 419 unique (video_name, edit_type) pairs, before use."
    status: completed
    note: "BUILT 2026-09-24, in two passes. First pass: verifying the manifest as it stood on disk found it was NOT safe -- for edit1-4 it had 100 rows but only 99 UNIQUE video names each, because 0086_A_red_sports_car appeared TWICE per edit type (once correctly, once carrying 0087_A_seagull's first_frame_edit_prompt under 0086's video_name/case_id) while 0087_A_seagull was entirely absent -- confirmed by diffing against the 6 real edit{T}_FiVE.json files. Root-caused to the MANIFEST FILE ITSELF being stale (mtime 2026-07-20, four days after the 2026-07-16 anchor-generation run that daily.md records as having verified 419/419 correct anchor PNGs on disk at the time) -- evaluation/r7_build_anchor_manifest.py, the script that builds this file, was confirmed to have no indexing bug itself. Fixed by rerunning it against the real edit_prompt/edit{T}_FiVE.json files (user-requested 2026-09-24, independent of this task) and re-verifying: 419 entries, exact set match against all 6 source jsons, 0086/0087 each present once per edit type 1-4 with correct prompts. An intermediate second file (cases_r35_fullbench.json) had been built as a workaround before the manifest itself was fixed; once it was fixed the two files were confirmed to carry byte-identical (video_name, edit_type) content and the extra file was dropped as redundant -- this plan's references now point at the corrected evaluation/r7_anchor_manifest.json directly."
  - id: stage1-script
    content: "slurm_scripts/five_bench/r35_stage1.sh -- array 0-1, one stage-1 REGIME per task (0 = unblended, 1 = first2), each looping edit types 1-6 with no --cases_json. Model loads once per task. --step 15 (NOT R31's 7). Writes frames to r35_stage1/{regime}/ and latents to r35_latents/{regime}/edit{T}/{video}.npz."
    status: completed
    note: "BUILT 2026-09-24. array 0-1 over REGIMES=(unblended first2) / SCHEDS=(zero first2); each task loops T=1..6 with no --cases_json. --step 15 --dump_steps 14 --measure_index 14 (the fully-denoised final x0, matching r31_stage1.py's --measure_index 14 -> measure_kind=post_injection__final_render). Storage under ~/Data/dataggen (not /projects), --exclude=node01,node51,node52,node57, --partition=L40S,A100 at build time -- changed to --partition=L40S plus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on 2026-09-24 after job 1007492 hit CUDA OOM on a 40 GB A100 --, --time=20:00:00 per Decisions. Post-run guard checks the exact per-type breakdown (100/100/100/100/9/10) per regime, not just a total, after the cases_r35_fullbench.json episode showed a total-only count can hide one dropped pair. VERIFIED (no GPU): bash -n clean; both task ids resolve to the correct REGIME/SCHED/METHOD/LATENT_DIR; bad array id (2) rejected; every flag in the python invocation is a real r31_stage1.py flag (16/16, diffed against its own --help); both --blend_sched zero and --blend_sched first2 parse cleanly at step=15/dump_steps=14/measure_index=14 and resolve to measure_kind=post_injection__final_render, confirming this is NOT the mid-trajectory default."
  - id: stage2-script
    content: "slurm_scripts/five_bench/r35_stage2.sh -- array 0-3 over the four COMBOS (dino_unblended, dino_first2, lpips_unblended, lpips_first2). Each task runs evaluation/r31_divergence.py for its (arm, regime) pair over edit types 1-6, reading that regime's latents + target frames, writing to r35_div/{combo}/edit{T}/{video}.npz. MUST pass --no-save_native (see Decisions)."
    status: completed
    note: "BUILT 2026-09-24, the preferred route. evaluation/r31_divergence.py: added --arm_dir_name (default None -- byte-identical to pre-R35 behaviour), which redirects ONLY the output subdirectory; --arm stays the true metric name and is written into the npz's own `arm` field unaffected. slurm_scripts/five_bench/r35_stage2.sh: array 0-3, one combo per task, explicit case-statement lookup (not string-splitting, since dino_patch itself contains an underscore) to ARM+REGIME; TARGET_ROOT=.../r35_stage1/{regime}/step14, LATENT_DIR=.../r35_latents/{regime}, -o .../r35_div --arm_dir_name {combo}; --no-save_native passed. ⚠️ ENV IS five-bench, NOT streamgve -- copied verbatim from r31_stage2.sh's own hard-won fix (lpips is not installed in streamgve at all). Preflight checks the regime's stage-1 output exists AND has the exact 100/100/100/100/9/10 breakdown before the model loads. Post-run guard checks the same exact breakdown per combo, not a bare total (same discipline as stage1-script, after the manifest episode showed a total-only count insufficient). VERIFIED (no GPU): both files syntax-clean; regression -- no --arm_dir_name reproduces the exact pre-R35 out_dir; with an override, out_dir changes but args.arm (the metric identity) does not; all 4 combos resolve to the correct arm/regime/paths; bad combo id (4) rejected; --no-save_native confirmed to resolve save_native=False; all 11 flags in the python invocation cross-checked as real, current r31_divergence.py flags."
  - id: absnorm-script
    content: "slurm_scripts/five_bench/r35_absnorm.sh -- the `absnorm` step's script (steps.absnorm: sbatch slurm_scripts/five_bench/r35_absnorm.sh, a CPU batch job). This todo did not exist in the original plan even though the step and its Code-to-touch spec did -- added 2026-09-24 while building stage3-script, once it became a blocking gap (see stage3-script's note). CPU only, no sbatch: for each of the 4 combos, r31_div_abs.py (absolute normalisation, div_max 1.0 lpips / 0.65 dino_patch, EXPLICIT per Decisions; r31_div_abs.py's DEFAULT_DIV_MAX was corrected from a stale dino_patch=0.6 to 0.65 on 2026-09-24) then r31_rho_map.py (budget-linear, tau_min=2 tau_max=50, --step 15) into r35_div_abs/{combo} and r35_rho/{combo}."
    status: completed
    note: "BUILT 2026-09-24. Discovered while starting stage3-script: the plan's `steps` block already had `absnorm` as stage3's wait_for dependency and Code-to-touch already specced r35_absnorm.sh, but no `todos` entry tracked building it -- a genuine gap, now closed. Neither r31_div_abs.py nor r31_rho_map.py needed changes (both already support the needed flags). Preflight checks stage 2's output is complete (419, exact per-type breakdown) before normalising; post-run guard checks both r35_div_abs and r35_rho land at 419/combo with the same exact breakdown. No `set -e` / no conda block -- matches this repo's convention for a bare local script (r31_div_abs.py and r31_rho_map.py are numpy-only, no GPU/env dependency). VERIFIED: bash -n clean; --arms confirmed to have no `choices=` restriction in r31_div_abs.py, so it works as a free-form directory-name selector; full END-TO-END smoke test on a synthetic combo-keyed npz through BOTH tools in sequence (r31_div_abs.py -> r31_rho_map.py) -- output landed at the combo-named path, the npz's own `arm` field stayed the true metric name (dino_patch) unaffected by the combo directory name, div_max=0.65 applied correctly, and the resulting rho landed in [2,50] as expected. CONVERTED 2026-09-24 to a CPU batch job for the overnight dependency chains (user): SBATCH header on --partition=CPU, cd + conda activate, positional combo subset, root overrides for testing. VERIFIED on synthetic stage-2 data (2 DINO combos x 419 clips, 18 latent frames each): local subset run exit 0, 419/419 in both trees per combo, 239 s wall; m == clip(d/0.65, 0, 1) to 2.8e-8 on all 838 clips, rho within [2,50], every above-cap token at rho=50; no-args run exits 1 on the missing LPIPS combos (why a chain must pass its subset); unknown combo rejected. Also submitted as a real sbatch job on a CPU node with an empty environment (job 1007647)."
  - id: stage3-script
    content: "slurm_scripts/five_bench/r35_stage3.sh -- array 0-3, one COMBO per task in the user's priority order, each looping edit types 1-6 with no --cases_json. Folds the CPU-only rho build (r31_rho_map.py) into the head of the task as r31_stage3.sh does, then calls evaluation/r31_stage3.py. Preflight: that combo's r35_div_abs tree must hold 419 npz before the model loads."
    status: completed
    note: "BUILT 2026-09-24, with one correction to its own content above: rho is NOT re-built inside this script. The plan already had a separate `absnorm` step as this one's `wait_for` dependency (which itself already specced building rho, per Code-to-touch) -- re-building it here too would have been silent redundant work duplicating the one place tau_min/tau_max/step/flow_shift live. r35_stage3.sh instead preflights that BOTH r35_div_abs/{combo} AND r35_rho/{combo} are already complete (419, exact per-type breakdown) before the model loads, and only renders. ⚠ SCOPE CHANGE FROM THE Decisions TABLE, confirmed with the user before building: --dump_steps 14 (final only), not `all` as r31_stage3.sh uses and as the Decisions row 'Render config ... identical to r31_stage3.sh' literally implied. Measured R31's actual disk cost (12 GB/arm at 22 clips for --dump_steps all vs 0.976 GB/arm final-only) and extrapolated to R35's 419 clips x 4 combos: ~914 GB (all) vs ~74 GB (final-only) -- and nothing in R35's plan (grids, eval, fiveacc, clipd, the Pareto figures) ever reads an intermediate step. User confirmed final-only. The Decisions table's Render config row has been corrected to state this explicitly. array 0-3, combos in the same priority order as stage2 (dino_unblended, dino_first2, lpips_unblended, lpips_first2); METHOD=r35_{combo}; --step 15 pinned; conda streamgve (render env, distinct from absnorm's bare-numpy and stage2's five-bench). VERIFIED (no GPU): both files syntax-clean; all 4 combos resolve to the correct method/paths; bad array id (4) rejected; every flag in the r31_div_abs.py / r31_rho_map.py / r31_stage3.py invocations cross-checked as real, current flags against each tool's own --help; manifest CSV's column 3 confirmed to be `status` (matching the post-run guard's awk)."
  - id: eval-script
    content: "slurm_scripts/five_bench/r35_eval.sh -- array 0-3 over the combos, the 9-metric FiVE-Bench harness at stride 8, same metric list as r26/r31_eval.sh, on r35_arms/r35_{combo}/step14, no --cases_json. Blocks step `eval`."
    status: completed
    note: "BUILT 2026-09-24. Modelled on r31_eval.sh (9 metrics passed explicitly -- config.yaml's metrics key is vestigial; teed log grepped for Error:/OOM) and r21_eval.sh (full bench, no --cases_json). Scores the 4 R35 arms ONLY -- the VP baseline is a separate task (user, 2026-09-24). L40S,A100: without five_acc/motion_fidelity a 40 GB A100 suffices (R26 job 961490). --time=12:00:00 (R20: ~45 s/clip with 16 metrics -> ~5.2 h at 419). Preflight hard-fails unless the render has exactly 100/100/100/100/9/10 REAL pairs (excluding evaluate.py's _resize siblings); post-run guard checks each edit{T}_FiVE_{stem}_frame_stride8.csv has the same per-type row count, since a clean exit and a written _avg.csv do not prove every clip was scored. VERIFIED: bash -n clean; all 7 flags exist in evaluate.py's argparse; edit_video is an accepted --tgt_layout; the 9 metric names identical to r31_eval.sh's; the CSV naming the guard reads matches an existing stem; stage 3 writes step{jj:02d}, i.e. step14. NOTE for score-script: evaluate.py's top-level _avg.csv is a mean over six per-edit-type means, not over 419 clips -- weight per clip from the per-type CSVs."
  - id: fiveacc-script
    content: "slurm_scripts/five_bench/r35_fiveacc.sh -- array 0-3 over the combos, evaluate.py --metrics five_acc alone, modelled on r33_fiveacc.sh with ~/Data roots and the node-exclusion set. Blocks step `fiveacc`."
    status: completed
    note: "BUILT 2026-09-24. five_acc alone (no co-resident models), five-bench env, teed log grepped for Error:/OOM; array 0-3 over the combos on r35_arms/r35_{combo}/step14, stem r35_fiveacc_{combo}, no --cases_json. Scores the 4 R35 arms ONLY -- the VP baseline is a separate task. --time=04:00:00, not R33's 6 h: R33's 22-clip five_acc task took 1 min 50 s end to end (tqdm in its metrics log). Preflight hard-fails unless 100/100/100/100/9/10 real pairs; post-run guard checks each per-type CSV for the full row count AND both FiVE-Acc columns present, NaN-free and binary, since evaluate.py drops a failed metric's column silently. VERIFIED: bash -n clean; all 7 flags in evaluate.py; post-run guard run on real CSVs -- passes on a 419-row full-bench table, fails with exit 1 on a 22-clip one."
  - id: clipd-script
    content: "slurm_scripts/five_bench/r35_clipd.sh -- single job (per-clip source embeddings are cached and reused across the 4 combos), CLIP-D via evaluation/r33_clip_directional.py over the full bench. Blocks step `clipd`."
    status: completed
    note: "BUILT 2026-09-24. r33_clip_directional.py could not score R35 as it stood: its method roster is hard-coded per task (--which), and --cases needs src/trg prompts that only cases.json (22 clips) carries. Added two options, both default off so every earlier invocation is unchanged: --methods NAME=DIR (explicit methods, overrides --which) and --fullbench (419 pairs built from edit{T}_FiVE.json: source_prompt/target_prompt/source_object/target_object -> src/trg_prompt/word). Checked against cases.json's 22 clips: prompts IDENTICAL on all 22, so clip_d_prompt (R33's axis of record) is unaffected; the word fields differ on 5 of 22 because cases.json hand-edited them, so full-bench clip_d_word uses FiVE's raw objects. Scores the 4 R35 arms ONLY -- the VP baseline is a separate task. Stride 8, no --max_frames (matches the harness). --time=06:00:00 (R33: 616 method-clips in 2 h; this is 1676). Preflight: 4 x 100/100/100/100/9/10 real pairs; post-run guard: 419 rows per combo with the exact per-type breakdown and 0 MISSING lines (the script prints MISSING and continues). VERIFIED: defaults unchanged; --fullbench yields 419 unique case_ids at the right per-type counts; END-TO-END CPU run of the modified script with --methods on a real render tree (r31_lpips/step14, 0001_bus, 2 frames) in five-bench with HF_HUB_OFFLINE=1 -- CLIP loads offline, one row written; post-run guard passes on a complete 4 x 419 table and fails with exit 1 when one clip is missing."
  - id: score-script
    content: "evaluation/r35_score.py -- the three Pareto figures: R36's VP rho-sweep CURVE (8 points, rho in {2,3,4,6,8,10,20,50}, joined in rho order) with the 4 R35 arms as points, on y = lpips_unedit_part vs x in {clip_similarity_target_image, clip_d_prompt, five_acc yes-no}. R7 is NOT evaluated -- R36 re-renders and scores rho=2 itself. R36's points come from R36's own summary, evaluation/csv/r36_rho_sweep.csv: its FULL-BENCH rows only (n_pairs == 419, exactly one per rho in {2,3,4,6,8,10,20,50}; the 22-clip subset rows are ignored). That file is a plain per-clip mean over all 419 clips (R36 Decisions: Averaging), i.e. the same aggregation as R35's own points, so the two are comparable. R35's 4 points: plain mean over 419 clips from its per-clip CSVs (edit{T}_FiVE_r35_{combo}_frame_stride8.csv, edit{T}_FiVE_r35_fiveacc_{combo}_frame_stride8.csv, r35_clip_directional.csv) -- never {stem}_avg.csv, which is a mean of the six per-edit-type means. Hard-fail if r36_rho_sweep.csv is missing, has other than exactly 8 full-bench rows, or lacks a needed column (R36's exact FiVE-Acc column name is not pinned in its plan -- match it against the header and print the available columns on failure); likewise if any R35 stem is not exactly 100/100/100/100/9/10. Emits evaluation/csv/r35_arms.csv (both sides' points). NOT r31_score.py: no R26 curves, no oracle frontier."
    status: completed
    note: "BUILT 2026-09-24. evaluation/r35_score.py (--all, or --x clip|clipd|fiveacc). R36 side read from r36_rho_sweep.csv, full-bench rows only (n_pairs == 419, exactly one per rho); columns resolved against the header (exact name, then |name suffix, then a regex for the FiVE-Acc yes/no and rho columns, whose names R36's plan does not pin). R35 side: plain mean over 419 clips from the per-clip CSVs, clips joined via file_id = row index into edit{T}_FiVE.json and columns matched on the |metric suffix (evaluate.py prefixes them with the folder name, step14|...). Encoding: R36 = gray reference curve, rho-labelled; R35 = hue by divergence metric (DINO blue, LPIPS orange: palette slots 1-2, documented all-pairs-valid -- the validator itself could not be run, no node on this machine) and marker by regime (unblended filled circle, first2 hollow triangle), each point direct-labelled; r35_arms.csv is the table view. VERIFIED on synthetic fixtures (real R35/R36 data does not exist yet): full run writes 3 PDFs + r35_arms.csv with 8 R36 + 4 R35 points, subset rows ignored, a non-standard FiVE-Acc column name resolved; figures inspected -- rho labels first sat below-right of the curve, exactly where arms that beat it land, and collided with two of them, so they were moved above-left. Hard failures each confirmed with a clear message and non-zero exit: R36 summary missing; a rho missing from its full-bench rows; unrecognisable FiVE-Acc column (lists the available ones); an R35 stem short one clip; the CLIP-D table short one clip; no --all/--x."
  - id: grids-run
    content: "No new grid script needed. Because stage 2 writes the four COMBOS as if they were arms, evaluation/r31_stage3_grids.py --which tau --arms dino_unblended dino_first2 lpips_unblended lpips_first2 yields exactly the 8 rows (4 combos x the m/tau pair). n_rows == 8 also lands on that script's historical geometry branch, which is the tested path. Only --cases must point at the 419-entry manifest."
    status: completed
    note: "VERIFIED 2026-09-24, no code change needed (as specced). r31_stage3_grids.py has every flag the step passes (--which --arms --cases --div_root --rho_root --tau_out_dir), and --cases reads only video_name + edit_type, which r7_anchor_manifest.json carries. END-TO-END run of the step's exact command against a stand-in tree with R35's combo layout (the 4 combo dirs symlinked to the 22-clip r31_div_abs / r31_rho_abs trees): 22 pages written at 8x12, i.e. 4 combos x the m/tau pair, labelled with the correct caps (d/0.65 dino, d/1 lpips); the 397 clips with no data each print [ERROR] and the run exits 1 -- so a missing clip is reported, not skipped. Cosmetic only: the page title is hard-coded 'R31 {clip}' in the script."

steps:
  - id: stage1
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_stage1.sh
    sets_status: running
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r35_stage1
      - ~/Data/dataggen/outputs/five_bench/r35_latents
    status: completed
    completed_at: 2026-09-24
    job_id: "1007599"   # resubmitted L40S-only; 1007492 (L40S,A100) cancelled after ~1 h, A100 OOM
  - id: wait-stage1
    type: manual
    wait_for: stage1
    check_hint: "Both tasks exit 0; 419 latent npz per regime (ls r35_latents/{regime}/edit*/*.npz | wc -l == 419 twice); per-edit-type counts 100/100/100/100/9/10; each task's startup line echoes the right blender_rate vector (unblended all 1s, first2 = [0,0,1,...,1])"
    status: completed
    completed_at: 2026-09-25
    note: "verified 2026-09-25 from the logs of job 1007599: both tasks 419 [ok], 0 [ERROR], 'failures: 0' (their own post-run guard enforces 100/100/100/100/9/10); startup regime lines [1,...,1] (unblended) and [0,0,1,...,1] (first2)"
  - id: stage2
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_stage2.sh
    wait_for: wait-stage1
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r35_div
    status: completed
    completed_at: 2026-09-24
    job_id: "1007668 (DINO, array 0,1) / 1007671 (LPIPS, array 2,3)"   # submitted in two afterok chains off stage 1 (1007599)
  - id: wait-stage2
    type: manual
    wait_for: stage2
    check_hint: "419 npz in each of the 4 combo dirs (1676 total), exact 100/100/100/100/9/10 per combo; 0 [ERROR] rows"
    status: completed
    completed_at: 2026-09-25
    note: "verified 2026-09-25 from the logs of 1007668/1007671: all 4 tasks 419 [ok], 0 [ERROR], 'failures: 0'; the exact per-type breakdown was re-checked by both absnorm jobs' preflight (1007669/1007672, 0 failures)"
  - id: absnorm
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_absnorm.sh
    wait_for: wait-stage2
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r35_div_abs
    status: completed
    completed_at: 2026-09-24
    job_id: "1007669 (DINO) / 1007672 (LPIPS)"   # submitted in two afterok chains off stage 1 (1007599)
  - id: stage3
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_stage3.sh
    wait_for: absnorm
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r35_rho
      - ~/Data/dataggen/outputs/five_bench/r35_arms
    status: completed
    completed_at: 2026-09-24
    job_id: "1008815 (DINO, array 0,1) / 1008816 (LPIPS, array 2,3)"   # 3rd submission 2026-09-25; 1007670/1007673 died 07:04 (suspected quota), 1008669/1008670 failed on missing anchors (moved during cleanup, restored)
  - id: wait-stage3
    type: manual
    wait_for: stage3
    check_hint: "4 tasks exit 0; 419 step14 frame dirs per combo; no non-ok rows in the _manifest_edit*.csv files; spot-check that the 4 combos' renders differ from each other by sha256"
    sets_status: finished
    status: completed
    completed_at: 2026-09-28
    note: "verified 2026-09-28 from disk + logs of 1008815/1008816 (sacct down, exit codes not queryable): 419 step14 dirs per arm (100/100/100/100/9/10), 0 empty; per-file recount 0 non-ok manifest rows in all 4 arms -- the guard's 'non-ok 5/6, failures: N' is a false positive (cat of 6 CSVs, NR==1 skips only the first header, so 5 headers count as failures); the one real failure, dino_first2 edit1 0050_slackline (Errno 122 disk quota), was re-rendered by job 1012688 (57 frames, ok, GATE1 PASS; 1012686 failed on a missing case-json path); sha256 of frame 30 differs across all 4 arms on 0050_slackline, 0001_bus, 0002_girl-dog"
  - id: eval
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_eval.sh
    wait_for: wait-stage3
    status: completed
    completed_at: 2026-09-28
    job_id: "1013063"   # L40S only; 1012706 (L40S,A100) cancelled while pending
  - id: fiveacc
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_fiveacc.sh
    wait_for: wait-stage3
    status: completed
    completed_at: 2026-09-28
    job_id: "1013064"   # L40S only; 1012707 (L40S,A100) cancelled while pending
  - id: clipd
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r35_clipd.sh
    wait_for: wait-stage3
    status: completed
    completed_at: 2026-09-28
    job_id: "1012708"
  - id: wait-eval
    type: manual
    wait_for: eval
    check_hint: "24 per-edit-type CSVs for the 9-metric harness (4 arms x 6 types) + 4 _avg.csv; 0 'Error:'/'out of memory' lines; five_acc and clip_directional CSVs cover 419 rows per arm"
    status: completed
    completed_at: 2026-09-29
    note: "verified 2026-09-29 from logs + CSVs (sacct down): 1013063 / 1013064 / 1012708 all report exit=0, 0 error/OOM lines, 0 Traceback in .err; on disk 24 metric CSVs + 24 five_acc CSVs, 419 rows per arm (100/100/100/100/9/10), 0 NaN, 0 duplicate file_ids; 4+4 r35_*_avg.csv present; r35_clip_directional.csv 1676 rows (419 x 4), 0 NaN; 0 leftover _resize dirs under r35_arms"
  - id: grids
    type: local
    command: python evaluation/r31_stage3_grids.py --which tau --arms dino_unblended dino_first2 lpips_unblended lpips_first2 --cases evaluation/r7_anchor_manifest.json --div_root ~/Data/dataggen/outputs/five_bench/r35_div_abs --rho_root ~/Data/dataggen/outputs/five_bench/r35_rho --tau_out_dir evaluation/figures/r35_tau_grids
    wait_for: wait-stage3
    status: completed
    completed_at: 2026-09-29
  - id: figures
    type: local
    command: python evaluation/r35_score.py --all
    wait_for: wait-eval
    check_hint: "ALSO needs R36's summarize step (outside this plan): evaluation/csv/r36_rho_sweep.csv with 8 full-bench rows (n_pairs == 419) -- r35_score.py hard-fails otherwise"
    sets_status: analyzed
    status: completed
    completed_at: 2026-09-29
---

# R35: Full-Bench Spatial Divergence Routing

## Context

R31 established continuous per-token divergence routing on the 22-clip `cases.json` subset, with four divergence arms and **per-clip min/max** normalisation of the divergence field `m`. R35 takes the method to the **whole 419-pair FiVE-Bench** with three changes:

1. **Absolute normalisation.** `m = clip(d, 0, div_max) / div_max` with a fixed per-arm `div_max`, replacing per-clip min/max. Per-clip normalisation makes `tau` depend on which clip a token sits in, which is a confound at 419 pairs rather than a nuisance.
2. **Two arms only** — DINO and LPIPS. `normals` and `latent` are dropped.
3. **Two stage-1 regimes** — the divergence field is measured against an UNBLENDED target, or against a FIRST2 target (Q/K blend fully on for the first 2 denoising steps, then disabled). Crossing 2 arms x 2 regimes gives the four arms.

**Done when:** the four arms are rendered and scored across all 419 pairs, the 419 tau-grid pages exist, and the three comparison figures place the four arms against R36's full-bench VP rho-sweep curve on the editability/preservation plane.

## Execution steps

| # | Step id | Type | What it does | Sets status |
|---|---------|------|--------------|-------------|
| — | *(prep)* | — | `/build-step` the 7 `todos` — stage-1 `--blend_sched`, 6 Slurm scripts, `r35_score.py` | `implemented` |
| 1 | `stage1` | sbatch | Array 0-1, one regime per task, 419 pairs each, 15 steps. Frames + latents. | `running` |
| 2 | `wait-stage1` | manual | 419 latents per regime; blender_rate vector echoed correctly | |
| 3 | `stage2` | sbatch | Array 0-3, one (arm, regime) combo per task. Divergence npz. | |
| 4 | `wait-stage2` | manual | 419 npz x 4 combos, exact per-type breakdown | |
| 5 | `absnorm` | sbatch | Absolute normalisation + budget-linear rho — **CPU batch job** (`--partition=CPU`), no GPU | |
| 6 | `stage3` | sbatch | Array 0-3, one combo per task in priority order. Gated render. | |
| 7 | `wait-stage3` | manual | 419 step14 dirs per combo; combos differ by sha256 | `finished` |
| 8 | `eval` | sbatch | 9-metric harness, stride 8, array 0-3 | |
| 9 | `fiveacc` | sbatch | `--metrics five_acc`, array 0-3 | |
| 10 | `clipd` | sbatch | CLIP-directional, single job | |
| 11 | `wait-eval` | manual | CSV coverage + 0 error lines | |
| 12 | `grids` | local | 419 tau-grid pages, 8 rows each | |
| 13 | `figures` | local | Three comparison figures + `r35_arms.csv` | `analyzed` |

```
/run-step R35 stage1      /run-step R35 absnorm     /run-step R35 wait-eval
/run-step R35 wait-stage1 /run-step R35 stage3      /run-step R35 grids
/run-step R35 stage2      /run-step R35 wait-stage3 /run-step R35 figures
/run-step R35 wait-stage2 /run-step R35 eval|fiveacc|clipd
```

## Decisions

| Question | Choice |
|---|---|
| Scope | All **419** pairs (100/100/100/100/9/10 by edit type). Stage 1 and 3 pass **no** `--cases_json`; stage 2 and the grids use `evaluation/r7_anchor_manifest.json` (419 entries, carries `video_name` + `edit_type`) |
| Arms (SLURM priority order) | `dino_unblended`, `dino_first2`, `lpips_unblended`, `lpips_first2` — treated as four **combos**, so every tree is keyed by combo name |
| Stage-1 regimes | **Both at `--step 15`**, matching stage 3. UNBLENDED = `--blend_sched zero`; FIRST2 = `--blend_sched first2` (source-anchoring `s(p)=1` for steps 0-1, then 0) |
| Normalisation | **Absolute**, `div_min = 0` both arms; `div_max` = **1.0** LPIPS, **0.65** DINO. No per-clip min/max, **no detrending** |
| Mapping | `budget_linear`, **`tau_min = 2`**, `tau_max = 50`, `--step 15`, `flow_shift 1.0` (R30's calibrated pair; `tau_min = 0` belongs only to the exploratory `linear_threshold` mapping) |
| Render config | `--vp_mode vp` + `--first_frame_edit_dir <anchors>`, `fg_boost 4`, `blend_power 2`, `rollout_chunk_size 21`, `seed 0`, `flow_shift 1.0`, `--step 15` — identical to `r31_stage3.sh`, **except** `--dump_steps 14` (final only), not R31's `all`: measured ~914GB vs ~74GB at full-bench scale, and nothing downstream in this plan reads an intermediate step (confirmed with the user 2026-09-24) |
| `--save_native` | **Off** for stage 2. On at 419 x 2 arms x 2 regimes it writes ~27 GB of pre-reduction LPIPS/DINO maps that only the R31 native-resolution figure consumed |
| Comparison reference | **R36's VP rho-sweep curve** (rho in {2,3,4,6,8,10,20,50}, full bench, same sampler/anchors/seed as R35's stage 3), read from **`evaluation/csv/r36_rho_sweep.csv`**, full-bench rows only (`n_pairs == 419`). R36 averages as a plain per-clip mean over 419 clips, matching R35's own points; neither side uses `{stem}_avg.csv`. R7 is **not** evaluated: R36 re-renders rho=2 itself. R26's uniform/constant-b curves and the per-clip oracle are **out of scope**: they exist only on the 22-clip subset |
| Figures | Three **Pareto figures** — `r35_clip_vs_lpips.pdf`, `r35_clipd_vs_lpips.pdf`, `r35_fiveacc_vs_lpips.pdf` — each plotting the **4 R35 arms as points against R36's VP rho-sweep curve** (8 points, rho in {2,3,4,6,8,10,20,50}, joined in rho order), on y = `lpips_unedit_part` (preservation) vs x = CLIP-target / `clip_d_prompt` / `five_acc` yes-no (one per figure) — the same axes and orientation as R36's own Pareto figures |
| Tau grids | **All 419 pages**, 8 rows each (4 combos x the `m`/`tau` pair) |
| Storage root | `~/Data/dataggen/outputs/five_bench/` — **not** `/projects`, which is unreliable per node (2026-09-22 fix) |
| Node exclusions | `--exclude=node01,node51,node52,node57`, the proven post-2026-09-22 set |
| Out of scope | `normals` / `latent` arms; detrended divergence; R26 + oracle curves; `k != 2` step-schedules |

## Step commands

### stage1

```bash
sbatch slurm_scripts/five_bench/r35_stage1.sh          # array 0-1, ~12-16 h/task
```

### wait-stage1

```bash
D=~/Data/dataggen/outputs/five_bench
for r in unblended first2; do
  echo "$r: $(ls $D/r35_latents/$r/edit*/*.npz 2>/dev/null | wc -l) latents (expect 419)"
  for T in 1 2 3 4 5 6; do echo -n "  edit$T=$(ls $D/r35_latents/$r/edit$T/*.npz 2>/dev/null | wc -l)"; done; echo
done
grep -h "blender_rate" logs/r35_stage1_*.out
```

### stage2

```bash
sbatch slurm_scripts/five_bench/r35_stage2.sh          # array 0-3
```

### wait-stage2

```bash
D=~/Data/dataggen/outputs/five_bench
for c in dino_unblended dino_first2 lpips_unblended lpips_first2; do
  echo "$c: $(ls $D/r35_div/$c/edit*/*.npz | wc -l) (expect 419)"
done
grep -c "\[ERROR\]" logs/r35_stage2_*.out
```

### absnorm

```bash
sbatch slurm_scripts/five_bench/r35_absnorm.sh                  # all 4 combos, CPU node
sbatch slurm_scripts/five_bench/r35_absnorm.sh dino_unblended dino_first2   # one chain's subset
# (also runs as plain `bash ...` locally -- same script, same checks)
# which runs, per combo:
#   python evaluation/r31_div_abs.py --arms <combo> --div_max <combo>=<1.0|0.65> \
#     --div_root $D/r35_div -o $D/r35_div_abs
#   python evaluation/r31_rho_map.py --div_root $D/r35_div_abs/<combo> \
#     -o $D/r35_rho/<combo> --tau_min 2 --tau_max 50 --step 15 --flow_shift 1.0
```

### stage3

```bash
sbatch slurm_scripts/five_bench/r35_stage3.sh          # array 0-3, priority-ordered, ~12-16 h/task
```

### wait-stage3

```bash
D=~/Data/dataggen/outputs/five_bench
for c in dino_unblended dino_first2 lpips_unblended lpips_first2; do
  echo "$c: $(ls -d $D/r35_arms/r35_$c/step14/edit*/*/ | wc -l) step14 dirs (expect 419)"
  cat $D/r35_arms/r35_$c/_manifest_edit*.csv | awk -F, '$3!="status" && $3!="ok"' | wc -l
done
```

### eval

```bash
sbatch slurm_scripts/five_bench/r35_eval.sh            # array 0-3
```

### fiveacc

```bash
sbatch slurm_scripts/five_bench/r35_fiveacc.sh         # array 0-3
```

### clipd

```bash
sbatch slurm_scripts/five_bench/r35_clipd.sh           # single job
```

### wait-eval

```bash
ls evaluation/csv/edit?_FiVE_r35_*_frame_stride8.csv | wc -l     # expect 24
grep -c "Error:\|out of memory" logs/r35_eval_*.metrics.log
wc -l evaluation/csv/r35_clip_directional.csv                    # expect 4*419 + 1
```

### grids

```bash
D=~/Data/dataggen/outputs/five_bench
python evaluation/r31_stage3_grids.py --which tau \
  --arms dino_unblended dino_first2 lpips_unblended lpips_first2 \
  --cases evaluation/r7_anchor_manifest.json \
  --div_root $D/r35_div_abs --rho_root $D/r35_rho \
  --tau_out_dir evaluation/figures/r35_tau_grids
```

### figures

```bash
python evaluation/r35_score.py --all      # 3 PDFs + evaluation/csv/r35_arms.csv
```

## Pipeline

```mermaid
flowchart TD
  BENCH["FiVE-Bench 419 pairs<br/>+ anchors"] --> S1U["r35_stage1.sh task 0<br/>blend_sched=zero, step 15"]
  BENCH --> S1F["r35_stage1.sh task 1<br/>blend_sched=first2, step 15"]

  S1U --> LU["r35_latents/unblended<br/>+ r35_stage1/unblended frames"]
  S1F --> LF["r35_latents/first2<br/>+ r35_stage1/first2 frames"]

  LU --> S2["r35_stage2.sh (array 0-3)<br/>r31_divergence.py, --no-save_native"]
  LF --> S2
  S2 --> DIV["r35_div/{combo}/edit{T}/*.npz<br/>raw d, 4 x 419"]

  DIV --> ABS["r31_div_abs.py<br/>div_max 1.0 lpips / 0.65 dino"]
  ABS --> DABS["r35_div_abs/{combo}"]
  DABS --> RHO["r31_rho_map.py<br/>budget_linear tau 2..50, step 15"]
  RHO --> RHOD["r35_rho/{combo}"]

  RHOD --> S3["r35_stage3.sh (array 0-3)<br/>r31_stage3.py gated render"]
  S3 --> ARMS["r35_arms/r35_{combo}/step14<br/>419 videos x 4"]

  ARMS --> EV["r35_eval.sh · r35_fiveacc.sh · r35_clipd.sh"]
  EV --> CSV["evaluation/csv/edit?_FiVE_r35_*.csv"]

  DABS --> GR["r31_stage3_grids.py --which tau<br/>--arms = the 4 combos"]
  RHOD --> GR
  GR --> GRID["evaluation/figures/r35_tau_grids/<br/>419 pages, 8 rows"]

  CSV --> SC["r35_score.py"]
  R36["R36 r36_rho_sweep.csv<br/>VP rho sweep, 8 rho, full-bench rows"] --> SC
  SC --> FIG["r35_clip_vs_lpips.pdf<br/>r35_clipd_vs_lpips.pdf<br/>r35_fiveacc_vs_lpips.pdf"]
```

## Code to touch

**`evaluation/r31_stage1.py`** — open the pinned blend schedule.
Line 172 is `args.blend_sched = "zero"`, pinned on purpose. Replace with a flag that preserves the guard's intent:

```python
p.add_argument("--blend_sched", choices=("zero", "first1", "first2", "first3"),
               default="zero",
               help="Stage-1 gating regime. 'zero' = Q/K blend OFF for the whole "
                    "rollout (R31's regime, the default, byte-identical to the "
                    "pinned behaviour). 'firstK' = fully source-anchored for the "
                    "first K steps, then disabled.")
# ... after parse: echo the resolved schedule so a wrong regime cannot pass silently
s = [_schedule_blend_rate(args.blend_sched, i, args.step) for i in range(args.step)]
print(f"[r35_stage1] {args.blend_sched}: blender_rate = {[int(1-x) for x in s]}")
```
`blend_sched` is already written into the npz at line 405, so the regime stays traceable. The restricted `choices` keeps the original guarantee that stage 1 cannot be "silently blended" by an arbitrary schedule.

**`slurm_scripts/five_bench/r35_stage1.sh`** *(new)* — array 0-1, `REGIMES=(unblended first2)`, `SCHEDS=(zero first2)`; loops `T` 1..6 with **no** `--cases_json`; `--step 15 --dump_steps 14 --measure_index 14`; `--vp_mode vp --first_frame_edit_dir $ANCHOR_ROOT`; `--time=20:00:00`, `--mem=64G`, `--partition=L40S` (L40S only since 2026-09-24: job 1007492 OOM'd on a 40 GB A100) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Post-run guard: 419 latents for this regime, per-type 100/100/100/100/9/10.

**`slurm_scripts/five_bench/r35_stage2.sh`** *(new)* — array 0-3 over `COMBOS=(dino_unblended dino_first2 lpips_unblended lpips_first2)`; derives `ARM` (`dino_patch`/`lpips`) and `REGIME` from the combo name; loops `T` 1..6 calling `r31_divergence.py --arm $ARM --edit_type $T --cases evaluation/r7_anchor_manifest.json --latent_dir .../r35_latents/$REGIME --target_root .../r35_stage1/$REGIME --no-save_native -o .../r35_div_tmp`, then relocates into `r35_div/$COMBO/`. Simpler alternative: teach `r31_divergence.py` an `--arm_dir_name` override so it writes straight to the combo dir — preferred, one added flag.

**`slurm_scripts/five_bench/r35_absnorm.sh`** *(new)* — **CPU batch job** (`#SBATCH --partition=CPU`, 4 CPUs, 16G, 2 h), also runnable as plain `bash` locally. No model, no GPU: NumPy only, a GPU allocation would only add queue time. It is a batch job rather than a local command so it can sit inside an unattended `--dependency=afterok` chain (stage 2 -> absnorm -> stage 3). Optional positional args select a subset of combos (default all four) -- required for a per-chain run, since the all-four default fails preflight on the other chain's not-yet-computed combos and would block this chain's stage 3. A batch job has none of the interactive shell's state, so it `cd`s to the repo and activates `streamgve` itself. `R35_DIV_ROOT` / `R35_DIV_ABS_ROOT` / `R35_RHO_ROOT` override the roots, for synthetic tests only. Runs the 4 `r31_div_abs.py` + 4 `r31_rho_map.py` calls from **Step commands**; neither tool needed changes.

**`slurm_scripts/five_bench/r35_stage3.sh`** *(new)* — array 0-3, combos in priority order; preflight asserting 419 npz in `r35_div_abs/$COMBO` **before** the model loads; then `r31_stage3.py` per edit type with the R31 render config and **no** `--cases_json`. `--time=20:00:00`.

**`slurm_scripts/five_bench/r35_eval.sh`, `r35_fiveacc.sh`, `r35_clipd.sh`** *(new)* — modelled on `r31_eval.sh` / `r33_fiveacc.sh` / `r34_clip_directional.sh`, with `~/Data` roots and the node-exclusion set. `r35_clipd.sh` stays a single job because the per-clip source embeddings are cached and reused across arms.

**`evaluation/r35_score.py`** *(new)* — three Pareto panels: R36's VP rho-sweep curve (8 points joined in rho order, each labelled with its rho) plus the 4 combos as distinct markers. R36 side: read `evaluation/csv/r36_rho_sweep.csv`, keep rows with `n_pairs == 419`, assert exactly one per rho; resolve the four needed columns (`lpips_unedit_part`, `clip_similarity_target_image`, `clip_d_prompt`, FiVE-Acc yes-no) against the header, `SystemExit` listing the available columns if one is missing. R35 side: one loader per metric family returns `{(combo, video_name, edit_type): value}` from the per-clip CSVs, each point a plain mean over its 419 clips; preflight asserts 100/100/100/100/9/10 per stem. Does **not** read any `*_avg.csv`, R26 curves or oracle helpers. Writes `evaluation/csv/r35_arms.csv` (one row per point: source R35|R36, label, n_clips, the three x metrics, lpips_unedit_part).

**`evaluation/r31_stage3_grids.py`** — no change expected. `--arms` (added 2026-09-24) already accepts the four combo names, and 4 combos x 2 rows = 8 rows lands on the script's tested historical-geometry branch. Confirm the 419-entry manifest is accepted as `--cases` (it needs only `video_name` + `edit_type`).
