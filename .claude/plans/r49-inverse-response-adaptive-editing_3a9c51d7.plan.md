---
name: "R49 plan — Inverse-response adaptive editing (StreamEdit + Wan-Edit)"
overview: "One draft-derived per-token procedure on two training-free video editors: demand q = S * max(clip(D_dino/0.55), clip(D_lpips/1.0)), mapped to each host's native control by inverting the exposure response read from the host's code (StreamEdit rho in [2, 50] via 1 - t^rho; Wan-Edit n = round(50 q) via the velocity gate, floors s_min 9 and 5). 18 clips (evaluation/r49_cases.json), the whole chain submitted at once with afterok dependencies, then the equal-compute matched-59 Wan curve launched last. Frame grids compare each host's official baseline with the variants. Full stage: Wan-Edit arm max_s5 only (chosen 2026-10-20)."
task_id: R49
todos:
  - id: se-stage1-sbatch
    content: "slurm_scripts/five_bench/r49_se_stage1.sh -- StreamEdit draft (N 5, tgt0.9, attention recording) on evaluation/r49_cases.json: r48_se_stage1.sh with CASES r49_cases.json and roots r49_se_stage1 / r49_se_attn; post-run guards = per-type counts (1/9/3/2/3/0) and bit-identity of every z_trg with r38_latents/n5_tgt09"
    needed_by: [chain]
    status: completed
  - id: wan-draft-sbatch
    content: "slurm_scripts/five_bench/r49_wan_draft.sh -- Wan-Edit 17-step draft from step 3 with attention recording on r49_cases.json: r48_wan_draft.sh with CASES r49_cases.json, roots r49_wan_draft/{attn,frames}, same per-type guards"
    needed_by: [chain]
    status: completed
  - id: div-sbatch
    content: "slurm_scripts/five_bench/r49_div.sh -- one GPU job running r31_divergence.py serially for the 4 names se_dino, se_lpips, wan_dino, wan_lpips (as r48_div.sh array tasks 0-3) on r49_cases.json, drafts from r49_se_stage1 / r49_wan_draft, out r49_div; exact per-type npz guards; one queue entry instead of an array"
    needed_by: [chain]
    status: completed
  - id: field-script
    content: "evaluation/r49_field.py -- per clip and host: S via r48_field.compute_E on the r49 attention npz; m_dino = clip(D_dino, 0, 0.55)/0.55, m_lpips = clip(D_lpips, 0, 1.0)/1.0 (r49 DIV_MAX, not r48_field's); div = max, q = S * div; StreamEdit rho = r31_rho_map.budget_linear_tau(q, t_next(15), 2, 50) -> rho npz in the r31_stage3 --rho_dir format; Wan-Edit start s = clip(50 - round(50 q), s_min, 50) for s_min in {9, 5} -> .npy start map + npz; npz store d_dino, d_lpips, m_dino, m_lpips, div, S, q and the mapping constants; --selftest (q = 0 -> rho 2 / s 50; q = 1 -> rho 50 / s s_min; U(rho(q)) == q)"
    needed_by: [chain]
    status: completed
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r49_field.sh -- CPU job: r49_field.py --selftest, r49_field.py on r49_cases.json (out r49_field/{se_max, wan_max_s9, wan_max_s5}), then r48_field.py --arms dino_SD on evaluation/r49_new_clip.json (attention / divergence from the r49 roots, out r48_field) for the reference arm; guards: 18 npz per r49_field dir, 1 reference npz per host"
    needed_by: [chain]
    status: completed
  - id: se-stage3-sbatch
    content: "slurm_scripts/five_bench/r49_se_stage3.sh -- StreamEdit adaptive 15-step render, arm max: r48_se_stage3.sh flags (VP + Qwen anchors, step 15, flow_shift 1, fg_boost 4, blend_power 2, chunk 21, seed 0) with --rho_dir r49_field/se_max, METHOD r49_max_subset, out r49_se_arms, cases r49_cases.json; guards: 18 step14 dirs, manifests all ok"
    needed_by: [chain]
    status: completed
  - id: wan-render-script
    content: "evaluation/r49_wan_render.py -- Wan-Edit renders on r49_cases.json: adaptive arms max_s9 / max_s5 (sample_steps 50, loop skip = s_min, --route_start_map from r49_field/wan_{arm}/edit{T}/{video}.npy) and the global matched-59 configs matched59_s{0.95,0.92,0.85} (r48_wan_sched.matched_n(sg, 59) = N/skip 75/16, 85/26, 111/52, real start sigma 0.948/0.919/0.850 asserted); out r49_wan_arms/{config}_subset/edit{T}/{video}/*.png; resumable, per-clip expected frame count"
    needed_by: [chain, matched59]
    status: completed
  - id: wan-render-sbatch
    content: "slurm_scripts/five_bench/r49_wan_render.sh -- MODE arms: array 0-1 = (max_s9, max_s5), preflight 18 start maps; MODE m59: array 0-2 = (matched59_s0.95, matched59_s0.92, matched59_s0.85); post-run guard: 18 complete clip dirs (r48_units.py --missing with r49_cases.json)"
    needed_by: [chain, matched59]
    status: completed
  - id: newclip-baselines-sbatch
    content: "slurm_scripts/five_bench/r49_newclip_baselines.sh -- one GPU job looping the 10 global Wan-Edit configs (the 9 r48_wan_sched.py --list baselines plus native_s0.978) over evaluation/r49_new_clip.json only, via r48_wan_render.py --mode subset --cases r49_new_clip.json --out_root r48_wan_arms (resumable: the other 17 clips already have their renders there); guard: each config dir holds the new clip's frames; failures counted"
    needed_by: [chain]
    status: completed
  - id: ref-render-sbatch
    content: "slurm_scripts/five_bench/r49_ref_render.sh -- one GPU job rendering the reference arm for the new clip only: StreamEdit dino_SD (r31_stage3.py --rho_dir r48_field/se_dino_SD, --cases_json r49_new_clip.json, method r48_dino_SD_subset, flags as r48_se_stage3.sh) and Wan-Edit adapt_dino_SD (r48_wan_render.py --config adapt_dino_SD --cases r49_new_clip.json); guard: the new clip's dirs exist in both"
    needed_by: [chain]
    status: completed
  - id: units-module
    content: "evaluation/r49_units.py -- units to score by group: main = se_max (r49_se_arms/r49_max_subset/step14), wan_max_s9, wan_max_s5 (r49_wan_arms); base = the 9 baselines native_s0.95 .. native_s0.4 / matched_s0.95 .. matched_s0.85 plus native_s0.978 (r48_wan_arms/{config}_subset), se_r48_dino_SD (r48_se_arms/r48_dino_SD_subset/step14), wan_r48_dino_SD (r48_wan_arms/adapt_dino_SD_subset); m59 = wan_matched59_s0p95 / s0p92 / s0p85 (r49_wan_arms); TSV base/dir/host/config; --group, --count, --index"
    needed_by: [chain, matched59]
    status: completed
  - id: eval-sbatch
    content: "slurm_scripts/five_bench/r49_eval_all.sh -- one L40S job per GROUP (main | base | m59): for each r49_units.py unit of the group, evaluate.py harness metrics (as r48_eval.sh) then five_acc (as r48_fiveacc.sh), stems r49_{base} / r49_fiveacc_{base}, --cases_json evaluation/r49_cases.json; then r33_clip_directional.py over the group's units -> evaluation/csv/r49_clip_directional_{group}.csv; guards: 18 rows per unit and the per-type counts, 0 Error:/OOM, unit failures counted"
    needed_by: [chain, matched59]
    status: completed
  - id: score-script
    content: "evaluation/r49_score.py -- plain per-clip means on the 18 clips; StreamEdit: R39 N=20 curve (equal NFE), official R36 rho2, R38 dino_n5_tgt09, R48 dino_SD reference (base group), R49 se_max; Wan-Edit: native curve (6 points + native_s0.978), matched-55 and matched-59 (when its CSVs exist) curves, official native_s0.92, R48 dino_SD reference, R49 max_s9 / max_s5; deltas vs the curve of the arm's own budget (max_s9 vs matched-55, max_s5 vs matched-59, otherwise native, flagged unmatched_nfe), never extrapolated; paired bootstrap CIs vs the official baseline and vs R48 dino_SD; pareto PDF + PNG per host and x metric (x = alignment, y = preservation); writes r49_points.csv, r49_deltas.csv, figures/r49_pareto"
    needed_by: [chain, matched59, score]
    status: completed
  - id: grids-script
    content: "evaluation/r49_grids.py -- (a) frame grids per host and clip, middle and last frame, cells labelled bg LPIPS / FiVE-Acc: StreamEdit source | official (r36_rho_sweep/r36_rho2_vp) | R48 dino_SD | R49 max; Wan-Edit source | official (native_s0.92) | R48 dino_SD | R49 max s_min 9 | R49 max s_min 5; (b) field pages per host and clip over 6 latent frames: source, d_dino, d_lpips, div, S, q, control (rho / start sigma, value at q = 0.5 from the npz constants)"
    needed_by: [chain, matched59, score]
    status: completed
  - id: post-sbatch
    content: "slurm_scripts/five_bench/r49_post.sh -- CPU job running r49_score.py then r49_grids.py; fails if any expected CSV or figure is missing (36 field pages)"
    needed_by: [chain, matched59]
    status: completed
  - id: chain-driver
    content: "slurm_scripts/five_bench/r49_chain.sh -- submits with --dependency=afterok: newclip_baselines (no dep), se_stage1, wan_draft, div (afterok both drafts), field (afterok div), se_stage3, wan_render array 0-1 and ref_render (afterok field), eval_base (afterok newclip_baselines, ref_render), eval_main (afterok se_stage3, wan_render), post (afterok both evals); 12 queue entries; prints and writes `<stage> <id>` lines to logs/r49_chain_jobs.txt; reports the stage the submit cap refuses, if any; --exclude=node52 on every GPU job"
    needed_by: [chain]
    status: completed
  - id: m59-driver
    content: "slurm_scripts/five_bench/r49_matched59.sh -- reads the main `post` id from logs/r49_chain_jobs.txt; submits r49_wan_render.sh m59 (array 0-2, afterok:<post>) -> r49_eval_all.sh m59 -> r49_post.sh with afterok, appends its ids; runs last"
    needed_by: [matched59]
    status: completed
  - id: se-stage3-full
    content: "slurm_scripts/five_bench/r49_se_stage3.sh -- optional MODE argument subset|full (default subset): full = evaluation/r7_anchor_manifest.json without --cases_json, METHOD r49_max_full, 419-pair guards"
    needed_by: [se-full-chain]
    status: completed
  - id: field-full
    content: "slurm_scripts/five_bench/r49_field.sh -- optional arguments HOSTS ('se', 'wan' or 'se wan', default 'se wan') and MODE (subset|full, default subset) = current behaviour when called with none; passes --hosts to r49_field.py (and, subset only, to r48_field.py for the new-clip reference); full = --cases evaluation/r7_anchor_manifest.json --attn_se r48_se_attn --attn_wan r48_wan_draft/attn --div_root r48_div --out_root r49_field_full, no r48_field reference; guards only the listed hosts' dirs (se_max; wan_max_s9/s5 + start maps) with the mode's count (18 or 419); also covers R50's field-hosts-edit"
    needed_by: [se-full-chain, wan-full-chain]
    status: completed
  - id: se-chain-driver
    content: "slurm_scripts/five_bench/r49_se_full_chain.sh -- argument DRAFT_JOB (the running r48_se_stage1.sh full job); sbatch --parsable with afterok: r48_div.sh full --array=0-1 (afterok:DRAFT_JOB) -> r49_field.sh se full -> r49_se_stage3.sh full -> r49_eval_all.sh se full; refuses if DRAFT_JOB is unknown to squeue and r48_se_attn lacks 419 npz; writes logs/r49_se_full_chain_jobs.txt (draft, div, field, render, eval); on a refused submission prints the stage and exits non-zero"
    needed_by: [se-full-chain]
    status: completed
  - id: wan-render-py-full
    content: "evaluation/r49_wan_render.py -- --mode subset|full (default subset): output dir {config}_{mode}; MATCHED59 gains 0.978 APPENDED (index 3, so subset indices 0-2 are unchanged): matched59_s0.978 = r48_wan_sched.matched_n(0.978, 59) = N 66, skip 7, real start sigma 0.9766 (passes the existing 0.015 assert); nothing else changes"
    needed_by: [wan-full-chain]
    status: completed
  - id: wan-render-sbatch-full
    content: "slurm_scripts/five_bench/r49_wan_render.sh -- optional second argument subset|full (default subset): full = evaluation/r7_anchor_manifest.json, passes --mode full, guard via r48_units.py --mode full --missing; MODE m59 accepts array index 3 (matched59_s0.978); runs on Jean Zay after jz/r50_port.sh"
    needed_by: [wan-full-chain]
    status: completed
  - id: native0978-full-sbatch
    content: "slurm_scripts/five_bench/r48_wan_render.sh -- new MODE explore_full: CASES evaluation/r7_anchor_manifest.json, PYMODE full, configs from r48_wan_sched.py --list_explore (array index 0 = native_s0.978 = N 50, skip 5) -> r48_wan_arms/native_s0.978_full; subset / full / explore unchanged"
    needed_by: [wan-full-chain]
    status: completed
  - id: units-full
    content: "evaluation/r49_units.py -- --mode subset|full (default subset): full dirs end in _full; full-mode groups split by host (one platform each): se = se_max (r49_se_arms/r49_max_full/step14, lab cluster), wan = wan_max_s5 (r49_wan_arms/max_s5_full, Jean Zay), base = the 9 Wan-Edit global configs on 419 pairs (r48_wan_arms/{config}_full, Jean Zay), m59 = matched59_*_full (Jean Zay); no R48 reference units; base names get a _full suffix in full mode"
    needed_by: [se-full-chain, wan-full-chain, se-full-score, wan-full-score]
    status: completed
  - id: eval-full
    content: "slurm_scripts/five_bench/r49_eval_all.sh -- optional second argument subset|full (default subset): full = r7_anchor_manifest.json counts, no --cases_json, CLIP-D --fullbench, output r49_clip_directional_{group}_full.csv; accepts the full-mode groups se, wan, base, m59"
    needed_by: [se-full-chain, wan-full-chain]
    status: completed
  - id: units-full-v2
    content: "evaluation/r49_units.py -- full-mode Wan groups follow the 2026-10-20 config list: base = the 6 native configs (r48_wan_sched.configs()[:6]: native_s0.95 .. native_s0.4, incl. the official native_s0.92) + native_s0.978 (7 units, r48_wan_arms/{config}_full; matched-55 dropped, it only serves max_s9); m59 = matched59_s0.95, s0.92, s0.85, s0.978 (4 units; 0.978 last, like its render index) and base = the 6 natives then native_s0.978 last (index 6); se / wan unchanged; subset groups unchanged"
    needed_by: [wan-full-chain, wan-full-score]
    status: completed
  - id: eval-full-array
    content: "slurm_scripts/five_bench/r49_eval_all.sh -- full mode run as a Slurm array: task i scores only unit i (r49_units.py --index i) and writes CLIP-D to r49_clip_directional_{group}_full_{unit}.csv (unit = the unit base without its _full, e.g. r49_clip_directional_base_full_wan_native_s0p92.csv), so no job exceeds Jean Zay's 20 h (one 419-pair unit took ~5 h harness + <1 h FiVE-Acc on L40S); without an array, unchanged (all units of the group, one CLIP-D file)"
    needed_by: [wan-full-chain, wan-full-score]
    status: completed
  - id: wan-chain-driver
    content: "slurm_scripts/five_bench/r49_wan_full_chain.sh -- run ON JEAN ZAY after jz/r50_port.sh; sbatch --parsable: draft r48_wan_draft.sh full -> r48_div.sh full --array=2-3 -> r49_field.sh wan full -> r49_wan_render.sh arms full --array=1 (max_s5); with no dependency: r48_wan_render.sh full --array=0-5 (6 native) and r48_wan_render.sh explore_full --array=0 (native_s0.978), r49_wan_render.sh m59 full --array=0-3; evals afterok on their renders: r49_eval_all.sh wan full, base full --array=0-6, m59 full --array=0-3; refuses if logs/r49_wan_full_chain_jobs.txt exists; writes it (one '<stage> <id>' per line)"
    needed_by: [wan-full-chain]
    status: completed
  - id: scratch-links-script
    content: "jz/r50_scratch_links.sh -- ON JEAN ZAY, run once before wan-full-chain (and before the R50 jz-repro): for each Wan output folder (r48_wan_draft, r48_div, r48_field, r48_wan_arms, r49_wan_draft, r49_div, r49_field, r49_field_full, r49_wan_arms, r50_preflight; r50_repro dropped, no script writes it) create $SCRATCH/five_bench_out/<dir> and a symlink $JZ_WORK/Data/dataggen/outputs/five_bench/<dir> -> it, so the job scripts (D=~/Data/dataggen/outputs/five_bench) write to $SCRATCH without any edit; idempotent, refuses to replace a non-empty real directory (CONFLICT), prints the idr_quota_user inode lines and '[r50_scratch] failures: N'"
    needed_by: [wan-full-chain]
    status: completed
  - id: fetch-outputs-script
    content: "jz/r50_fetch_outputs.sh -- ON THE LAB FRONTAL (gpu-gw): rsync -a --partial the Wan output folders from zay:$SCRATCH/five_bench_out/ to the lab store (/store/PERSO/skhairi/Outputs/jz_five_bench/), one folder per run (resumable, no --delete), then compares per-folder file counts and total bytes with the Jean Zay side and prints '[r50_fetch] failures: N'; prints, never runs, the commands that delete the scratch copy; no lab-side symlinks into ~/Data (those folders hold L40S renders; H100 and L40S results are never merged), the CSVs travel separately through the wan-full-fetch rsync; --check counts only"
    needed_by: [wan-full-fetch]
    status: completed
  - id: score-full
    content: "evaluation/r49_score.py -- --mode full --host se|wan (one host per run, so each track scores on its own): scope = 419 pairs, also reported on the 401 pairs outside the 18-clip set; se (lab L40S): curve R39 N=20 + official R36 rho2 + R38 (full CSVs exist) and se_max; wan (Jean Zay H100, CSVs fetched by wan-full-fetch): native curve = 7 points from base (native_s0.978 .. native_s0.4, official native_s0.92 marked), equal-NFE curve = matched-59 from m59 (4 points incl. s0.978), wan_max_s5 (deltas vs matched-59 and vs official; paired vs native_s0.978); CLIP-D read from the per-unit files r49_clip_directional_{base,m59}_full_{unit}.csv (unit = base without _full) and r49_clip_directional_{wan,se}_full.csv; paired CIs: se vs official + R38, wan vs official + native_s0.978; writes r49_full_{host}_points.csv, r49_full_{host}_deltas.csv (with a platform column L40S / H100) and figures/r49_full_pareto/{host}_*.{pdf,png} with the platform in the title; R48 references skipped"
    needed_by: [se-full-score, wan-full-score]
    status: completed
steps:
  - id: chain
    type: local
    command: bash slurm_scripts/five_bench/r49_chain.sh
    sets_status: running
    status: completed
    completed_at: 2026-10-16
    job_id: "newclip_baselines 1043292, se_stage1 1043293, wan_draft 1043307 (resubmitted; 1043294 failed on a wrong script path), div 1043295, field 1043296, se_stage3 1043297, wan_render 1043298 (array 0-1), ref_render 1043299, eval_base 1043300, eval_main 1043301, post 1043302"
  - id: wait-chain
    type: manual
    wait_for: chain
    check_hint: "cat logs/r49_chain_jobs.txt; squeue -u $USER -- all 12 entries gone; per job log: r49_se_stage1 / r49_wan_draft 'failures: 0' and 18 attention npz each; r49_div 4 x 18 npz; r49_field 18 npz x 3 dirs; r49_se_arms/r49_max_subset/step14: 18 dirs; r49_wan_arms/{max_s9,max_s5}_subset: 18 complete dirs each; newclip + ref render logs 'failures: 0'; logs/r49_eval_all_*.out '[r49_eval] unit failures: 0' for main and base; r49_post 'failures: 0'; 0 Traceback in .err"
    sets_status: finished
    status: completed
    completed_at: 2026-10-20
  - id: matched59
    type: local
    wait_for: chain
    command: bash slurm_scripts/five_bench/r49_matched59.sh
    status: completed
    completed_at: 2026-10-16
    job_id: "m59_render 1043309 (array 0-2), m59_eval 1043310, m59_post 1043311"
  - id: wait-matched59
    type: manual
    wait_for: matched59
    check_hint: "the 3 m59 ids in logs/r49_chain_jobs.txt gone from squeue; r49_wan_arms/matched59_s{0.95,0.92,0.85}_subset: 18 complete dirs each; logs/r49_eval_all_*.out (m59) '[r49_eval] unit failures: 0'; r49_post 'failures: 0'"
    status: pending
  - id: score
    type: local
    wait_for: [wait-chain, wait-matched59]
    command: |
      python evaluation/r49_score.py
      python evaluation/r49_grids.py
    output_paths:
      - evaluation/csv/r49_points.csv
      - evaluation/csv/r49_deltas.csv
      - evaluation/figures/r49_pareto
      - evaluation/figures/r49_grids
    sets_status: analyzed
    status: pending
  # ---- StreamEdit track: lab cluster (L40S, streamgve); independent of every Wan-Edit step
  - id: se-full-gate
    type: manual
    wait_for: wait-chain
    check_hint: "read the StreamEdit rows of r49_deltas.csv, figures/r49_pareto/se_*.png and r49_grids/frames_se.png (written by the chain's post job); user decides whether StreamEdit max goes to the 419 pairs; record the decision in the R49 Run log"
    status: completed
    completed_at: 2026-10-20
  - id: se-full-draft
    type: sbatch
    wait_for: se-full-gate
    command: sbatch slurm_scripts/five_bench/r48_se_stage1.sh full
    status: completed
    completed_at: 2026-10-20
    job_id: "1052246"
  - id: se-full-chain
    type: local
    wait_for: se-full-draft
    command: bash slurm_scripts/five_bench/r49_se_full_chain.sh 1052246
    status: completed
    completed_at: 2026-10-20
    job_id: "div 1052283 (array 0-1), field 1052284, render 1052285, eval 1052286 (afterok chain behind draft 1052246; FAILED in its preflight, see Run log) -> eval resubmitted alone 1054280"
  - id: wait-se-full-chain
    type: manual
    wait_for: se-full-chain
    check_hint: "cat logs/r49_se_full_chain_jobs.txt; squeue -u $USER -- the draft and the 4 chain entries gone; r48_se_stage1 log 'bit-identical 419/419', 'failures: 0' and 419 npz in r48_se_attn; r48_div tasks 0-1 'failures: 0', 419 npz each; r49_field (se full) 'failures: 0', 419 npz in r49_field_full/se_max; r49_se_stage3 (full) 419 step14 dirs, 'failures: 0'; r49_eval_all (se full, job 1054280) 'unit failures: 0', 419 rows; 0 Traceback in .err"
    status: pending
  - id: se-full-score
    type: local
    wait_for: wait-se-full-chain
    command: python evaluation/r49_score.py --mode full --host se
    output_paths:
      - evaluation/csv/r49_full_se_points.csv
      - evaluation/csv/r49_full_se_deltas.csv
      - evaluation/figures/r49_full_pareto
    status: pending
  # ---- Wan-Edit track: Jean Zay (H100, five-bench via R50 wrapper); every step is run there and recorded here
  - id: wan-full-gate
    type: manual
    wait_for: wait-matched59
    check_hint: "read the Wan-Edit rows of r49_deltas.csv, figures/r49_pareto/wan_*.png (with the matched-59 curve) and r49_grids/frames_wan.png; user decides whether Wan-Edit max_s5 goes to the 419 pairs (floor chosen 2026-10-20); AND R50 jz-compare is completed with PASS; AND jz/r50_port.sh reports 0 leftovers on the full-stage Wan scripts (r48_wan_draft.sh, r48_div.sh, r48_wan_render.sh, r49_wan_render.sh, r49_eval_all.sh) after pulling them; record in the R49 Run log"
    status: pending
  - id: wan-full-chain
    type: manual
    wait_for: wan-full-gate
    check_hint: "ON JEAN ZAY (after git pull + bash jz/r50_port.sh, 0 leftovers): bash slurm_scripts/five_bench/r49_wan_full_chain.sh -> 'submitted', ids in logs/r49_wan_full_chain_jobs.txt; paste the ids here"
    status: pending
  - id: wait-wan-full-chain
    type: manual
    wait_for: wan-full-chain
    check_hint: "ON JEAN ZAY: every id of logs/r49_wan_full_chain_jobs.txt gone from squeue; draft 419 attention npz, failures 0; r48_div tasks 2-3 419 npz each; r49_field (wan full) failures 0, 419 npz + 419 npy per floor; renders 419 complete dirs each for max_s5_full, the 7 native configs (native_s0.978 .. native_s0.4) and the 4 matched59 configs, failures 0 (timed-out render tasks are resubmitted alone: renders are resumable); evals: wan 1 unit, base 7 tasks, m59 4 tasks, 'unit failures: 0', 419 rows each"
    status: pending
  - id: wan-full-fetch
    type: manual
    wait_for: wait-wan-full-chain
    check_hint: "ON THE LAB CLUSTER'S FRONTAL (compute nodes cannot reach Jean Zay): rsync the Jean Zay full-mode Wan CSVs and move the scratch output folders to the lab store (`bash jz/r50_fetch_outputs.sh`: '[r50_fetch] failures: 0', file counts and bytes equal to Jean Zay) — the CSVs (edit*_FiVE_r49_*wan*_full*, edit*_FiVE_r49_fiveacc_*wan*_full*, r49_clip_directional_wan_full.csv, r49_clip_directional_{base,m59}_full_*.csv) into evaluation/csv here; counts match the Jean Zay listing"
    status: pending
  - id: wan-full-score
    type: local
    wait_for: wan-full-fetch
    command: python evaluation/r49_score.py --mode full --host wan
    output_paths:
      - evaluation/csv/r49_full_wan_points.csv
      - evaluation/csv/r49_full_wan_deltas.csv
      - evaluation/figures/r49_full_pareto
    status: pending
  # ---- end: both hosts read together (no compute)
  - id: full-verdict
    type: manual
    wait_for: [se-full-score, wan-full-score]
    check_hint: "read r49_full_{se,wan}_deltas.csv and figures/r49_full_pareto (each host on its own platform); write the full-benchmark verdict in Progress / outcomes"
    sets_status: analyzed
    status: pending
isProject: true
---

# R49: Inverse-Response Adaptive Editing (StreamEdit + Wan)

## Context

Run one draft-derived per-token procedure on two training-free video editors. Stage 1 runs the host once at high editability and records cross-attention. Stage 2 turns the attention support and the draft–source divergence (the larger of the clipped DINO and LPIPS maps) into a per-token demand q, then inverts each host's exposure response, read from the host's own code, to get its native control (StreamEdit ρ, Wan-Edit start step). Stage 3 runs the host again with those per-token controls.

Done when: on the 18 clips of `evaluation/r49_cases.json`, StreamEdit (1 arm) and Wan-Edit (floors s_min 9 and 5) are rendered, scored against each host's official baseline and equal-compute curves, and the frame grids, field pages and pareto/deltas exist. For each host whose full gate says go (`se-full-gate`, `wan-full-gate`): the same on all 419 pairs (also reported on the 401 pairs outside the 18 clips), StreamEdit on the lab cluster and Wan-Edit on Jean Zay, each scored on its own.

## Execution steps

| # | Step id | Type | Waits for | Sets status |
|---|---|---|---|---|
| 0 | *(prep)* todos `se-stage1-sbatch`, `wan-draft-sbatch`, `div-sbatch`, `field-script`, `field-sbatch`, `se-stage3-sbatch`, `wan-render-script`, `wan-render-sbatch`, `newclip-baselines-sbatch`, `ref-render-sbatch`, `units-module`, `eval-sbatch`, `score-script`, `grids-script`, `post-sbatch`, `chain-driver`, `m59-driver` | /build-step | — | implemented |
| 1 | `chain` | local (submits 12 Slurm entries) | — | running |
| 2 | `wait-chain` | manual | chain | finished |
| 3 | `matched59` | local (submits 5 Slurm entries, run last) | chain | — |
| 4 | `wait-matched59` | manual | matched59 | — |
| 5 | `score` | local | wait-chain, wait-matched59 | analyzed |
| — | *(prep)* todos `se-stage3-full`, `field-full`, `se-chain-driver`, `units-full`, `eval-full` (→ `se-full-chain`); `wan-render-py-full`, `wan-render-sbatch-full`, `native0978-full-sbatch`, `units-full-v2`, `eval-full-array`, `wan-chain-driver`, `scratch-links-script` (→ `wan-full-chain`); `fetch-outputs-script` (→ `wan-full-fetch`); `score-full` | /build-step | — | — |
| | **StreamEdit track — lab cluster (L40S); no Wan-Edit dependency** | | | |
| 6 | `se-full-gate` | manual (user decision) | wait-chain | — |
| 7 | `se-full-draft` | sbatch | se-full-gate | — |
| 8 | `se-full-chain` | local (submits div → field → render → eval with afterok on the draft) | se-full-draft | — |
| 9 | `wait-se-full-chain` | manual | se-full-chain | — |
| 10 | `se-full-score` | local | wait-se-full-chain | — |
| | **Wan-Edit track — Jean Zay (H100), run there, recorded here; no StreamEdit dependency** | | | |
| 11 | `wan-full-gate` | manual (user decision + R50 PASS) | wait-matched59 | — |
| 12 | `wan-full-chain` | manual (JZ: one driver submits draft → div 2-3 → field → `max_s5`; 7 native + 4 matched-59 renders; evals as arrays) | wan-full-gate | — |
| 13 | `wait-wan-full-chain` | manual (JZ) | wan-full-chain | — |
| 14 | `wan-full-fetch` | manual (lab frontal: CSVs into `evaluation/csv`; scratch outputs moved to the lab store, counts compared) | wait-wan-full-chain | — |
| 15 | `wan-full-score` | local | wan-full-fetch | — |
| | **End — both hosts read together, no compute** | | | |
| 16 | `full-verdict` | manual | se-full-score, wan-full-score | analyzed |

Steps 1–5 ran both hosts in one chain (already submitted); everything after them is per host.

Invoke: `/run-step R49 wait-chain`, `/run-step R49 wait-matched59`, `/run-step R49 score`, then per host `/run-step R49 se-full-gate` … and `/run-step R49 wan-full-gate` ….

## Decisions

| Item | Choice |
|---|---|
| Clips | `evaluation/r49_cases.json`: 18 clips (types 1:1, 2:9, 3:3, 4:2, 5:3) = the R48 subset minus `0042_gym-ball` (type 6) plus `0004_car-roundabout`, edit 3, grey → yellow Mini Cooper; `evaluation/r49_new_clip.json` holds that one clip. R48's own case file is untouched |
| Reuse | StreamEdit curves: R39 N=20, official R36 ρ=2 and R38 per-clip CSVs (full benchmark, filtered to the 18 clips). Wan-Edit global renders: the 17 clips shared with R48 are reused from `r48_wan_arms/{config}_subset`; only the new clip is rendered there (`r49_newclip_baselines.sh`, resumable) |
| Draft, StreamEdit | N = 5, `--blend_sched tgt0.9` (≡ σ 0.95 on t = 1, .8, .6, .4, .2), other flags as `r38_stage1.sh`; attention at t = 0.8, 0.6; must be bit-identical to `r38_latents/n5_tgt09`; re-run on the 18 clips into `r49_se_stage1` / `r49_se_attn` |
| Draft, Wan-Edit | 17-step grid (shift 5) from step 3 (σ 0.958), 14 active steps, text only, 41 frames 832×480, seed 0; attention at steps 7–13; re-run on the 18 clips into `r49_wan_draft` |
| Attention | first 20 blocks, mean over heads and window steps; trigger tokens via `find_phrase_token_indices` |
| Support S | selectivity-weighted Eq. 7, union of source and target branches, per-frame decay exp(−0.5·dist); frame-0 support left as is |
| Divergence | D_DINO = DINOv3 ViT-B/16, D_LPIPS = LPIPS, on the 30×52 token grid; m_dino = clip(D_dino, 0, 0.55)/0.55, m_lpips = clip(D_lpips, 0, **1.0**)/1.0; div = max(m_dino, m_lpips); no per-video normalisation |
| Demand | q = S · div ∈ [0, 1] |
| StreamEdit control | ρ = U⁻¹(q) over [2, 50], U(ρ) = (B(2) − B(ρ))/(B(2) − B(50)), B(ρ) = mean_i t_i^ρ on the real 15-step `t_next` grid ≡ `budget_linear_tau(q, t_next, 2, 50)`; q = 0 → ρ = 2 (stock) |
| Wan-Edit control | n = round(50·q) (half to even), start s = 50 − n, clipped to s ≥ s_min; s = 50 → never edited |
| Arms | StreamEdit `max`; Wan-Edit `max_s9` (s_min 9, σ 0.958, 14 + 41 = 55 active steps) and `max_s5` (s_min 5, σ 0.978, 14 + 45 = 59) on the 18 clips. **Full stage: Wan-Edit `max_s5` only** (user, 2026-10-20; both floors edit equally on the 18 clips, `max_s5` has the higher CLIP-D, 0.2128 vs 0.2079); its equal-compute reference is matched-59 |
| Wan draft | stays the 17-step grid from step 3 (σ 0.958); drafts from step 2 (σ 0.974) and step 1 (σ 0.988) were tested on the 18 clips and rejected (object/background divergence ratio 2.2× → 1.6× → 1.3×; `evaluation/figures/r49_draft_skip/draft_step3_2_1.pdf`) |
| Official baselines | StreamEdit stock: ρ = 2, N = 15, VP (`r36_rho2_vp`); Wan-Edit stock: N = 50, skip 15 (`native_s0.92`) |
| Reference arm | R48's `dino_SD` (linear map, ρ_min 0.3, LPIPS reference 0.75) on both hosts: the 17 shared clips reused from `r48_se_arms` / `r48_wan_arms`, the new clip rendered by `r49_ref_render.sh` from fields computed with `r48_field.py` on the r49 drafts |
| Curves | StreamEdit: R39 N=20 (equal NFE). Wan-Edit: native N=50 (σ 0.95 … 0.4) plus `native_s0.978` (the global start at s = 5, the same start as s_min 5), matched-55 (N 69 / 79 / 104) for `max_s9`, matched-59 for `max_s5`: σ_g 0.95 / 0.92 / 0.85 at N = 75 / 85 / 111, skip 16 / 26 / 52 (real start σ 0.948 / 0.919 / 0.850) |
| Metrics | FiVE harness (bg LPIPS/PSNR/SSIM, CLIP-T, …), FiVE-Acc, CLIP-D; plain per-clip mean over the 18 clips (never `_avg.csv`); figures x = alignment, y = preservation |
| Grids | frame grids per host (middle and last frame, cells labelled bg LPIPS / FiVE-Acc) and 36 field pages; no grids on the full set |
| Chain | `chain` submits 12 entries with afterok: newclip baselines (GPU, serial, no dependency), StreamEdit draft, Wan draft → divergence (GPU, serial over 4 names) → field (CPU) → StreamEdit render, Wan render array 0-1, reference render → eval base (afterok newclip baselines, reference render), eval main (afterok both renders) → post (CPU). `matched59` submits the m59 render array 0-2 → eval m59 → post, the render `afterok` on the main post job so it runs last |
| Slurm | lab cluster: `--partition=L40S` (GPU) / `CPU`, `--exclude=node52` only; StreamEdit stages in `streamgve`, everything else in `five-bench`; serial loops inside one job instead of arrays where the per-user submit cap matters. Jean Zay: headers and interpreter rewritten by `jz/r50_port.sh` (R50) |
| Host split (full stage) | StreamEdit on the lab cluster (L40S, `streamgve`; its curves R39 / R36 / R38 are L40S already); Wan-Edit, all of it (draft, divergence, field, 9 baselines, `max_s5`, matched-59, eval), on Jean Zay (H100, `five-bench` rebuilt by R50) after R50's 18-clip check passes (part of `wan-full-gate`) (user, 2026-10-20). Each host's arms, baselines and curves come from one platform; full-stage numbers are never mixed with the 18-clip L40S Wan numbers in one table or figure; tables and figures carry the platform |
| Wan full-stage configs | (user, 2026-10-20) adaptive `max_s5`; native curve N = 50, 7 points: `native_s0.978` (uniform start at step 5 = the floor of `max_s5`, 45 active steps) and `native_s0.95 / 0.92 (official) / 0.85 / 0.75 / 0.6 / 0.4`; equal-NFE curve matched-59, 4 points: `matched59_s0.978` (N 66, skip 7, real σ 0.9766), `s0.95` (75/16), `s0.92` (85/26), `s0.85` (111/52); matched-55 dropped (it only serves `max_s9`, not in the full stage); not rerun on the 18 clips |
| Wan full-stage outputs (user, 2026-10-21) | every Wan-Edit output folder of the full stage (drafts, divergence, field, all 12 render arms, preflight/repro) is written to Jean Zay `$SCRATCH` (the full stage is ~220k PNG files, which `$WORK`'s 500k-file quota cannot hold next to the env and the bundle), through symlinks under `$WORK/Data/dataggen/outputs/five_bench/` so no job script changes (`scratch-links-script`); eval CSVs stay in the repo's `evaluation/csv` on `$WORK`; after the evals, `wan-full-fetch` moves the folders to the lab store and the CSVs here; `$SCRATCH` has no per-user line in `idr_quota_user` and, per the IDRIS disk-space page (user, 2026-10-21), only very large security quotas shared by all users (4.6 PB), so the 500k-file `$WORK` limit does not apply; IDRIS purges unused scratch files after a retention period (30 days from my memory, to be confirmed on that page), so fetch right after `wait-wan-full-chain` and delete the scratch copy only after the counts match |
| Jean Zay job sizing | 20 h limit: evals run one unit per array task (~5 h harness + <1 h FiVE-Acc per 419-pair unit on L40S); renders are resumable, a timed-out render task is resubmitted alone (matched59_s0.85 at N = 111 is the longest); the R50 18-clip reproduction gives the H100 per-clip times |
| Full stage | gated per host (`se-full-gate` after `wait-chain`; `wan-full-gate` after `wait-matched59` and R50 PASS), each host scored separately (`r49_score.py --mode full --host se|wan`), `full-verdict` reads both: R48's scripts for the full drafts (`r48_se_stage1.sh` here, `r48_wan_draft.sh` on Jean Zay), the divergence (`r48_div.sh full`, array 0-1 = StreamEdit here, 2-3 = Wan on Jean Zay) and the 9 Wan-Edit global configs on 419 pairs (`r48_wan_render.sh full`, array 0-8, ~155 L40S GPU-h, on Jean Zay); R49 field (`--hosts se` / `--hosts wan`), renders, matched-59, eval (groups `se` / `wan` / `base` / `m59`) and score in `full` mode (outputs end in `_full`); the Wan CSVs come back by rsync for scoring; the claim is read on the 401 pairs outside the 18 clips as well as on all 419 |
| Outputs | `~/Data/dataggen/outputs/five_bench/r49_{se_stage1,se_attn,wan_draft,div,field,se_arms,wan_arms}/`, `evaluation/csv/r49_*.csv`, `evaluation/figures/r49_{pareto,grids}/` |
| Out of scope | mean / DINO-only / LPIPS-only arms; the Wan-Edit global start sweep above σ 0.978; tuning λ, κ or the ρ range after seeing renders |

## Step commands

### chain
```bash
bash slurm_scripts/five_bench/r49_chain.sh
```

### wait-chain
```bash
cat logs/r49_chain_jobs.txt
squeue -u $USER
grep -E 'failures|unit failures' logs/r49_*.out
```

### matched59
```bash
bash slurm_scripts/five_bench/r49_matched59.sh
```

### wait-matched59
```bash
tail -n 3 logs/r49_chain_jobs.txt
squeue -u $USER
grep -E 'dirs total|failures' logs/r49_wan_render_<job>_*.out logs/r49_eval_all_<job>.out
```

### score
```bash
~/anaconda3/envs/five-bench/bin/python evaluation/r49_score.py
~/anaconda3/envs/five-bench/bin/python evaluation/r49_grids.py
```

### se-full-gate
```bash
grep -E '^(host|se)' evaluation/csv/r49_deltas.csv
xdg-open evaluation/figures/r49_pareto/se_five_acc.png    # and se_clip_d_prompt.png, r49_grids/frames_se.png
```

### se-full-draft
```bash
sbatch slurm_scripts/five_bench/r48_se_stage1.sh full
```

### se-full-chain
```bash
bash slurm_scripts/five_bench/r49_se_full_chain.sh 1052246     # 1052246 = the running r48_se_stage1.sh full draft
cat logs/r49_se_full_chain_jobs.txt
```

### wait-se-full-chain
```bash
cat logs/r49_se_full_chain_jobs.txt
squeue -u $USER
grep -E 'bit-identical|failures|unit failures|dirs total' logs/r48_se_stage1_1052246.out logs/r48_div_<div>_*.out logs/r49_field_<field>.out logs/r49_se_stage3_<render>.out logs/r49_eval_all_<eval>.out
```

### se-full-score
```bash
~/anaconda3/envs/five-bench/bin/python evaluation/r49_score.py --mode full --host se
```

### wan-full-gate
```bash
grep -E '^(host|wan)' evaluation/csv/r49_deltas.csv
xdg-open evaluation/figures/r49_pareto/wan_five_acc.png   # and wan_clip_d_prompt.png, r49_grids/frames_wan.png
grep -A3 'jz-compare' daily.md          # R50 Run log: PASS
# on Jean Zay, after git pull of the full-stage scripts:
bash jz/r50_port.sh                     # [r50_port] leftovers: 0
```

### wan-full-chain
```bash
# on Jean Zay (ssh zay), in the clone
cd $WORK/StreamEdit_new && git pull && bash jz/r50_port.sh        # [r50_port] leftovers: 0
bash slurm_scripts/five_bench/r49_wan_full_chain.sh
cat logs/r49_wan_full_chain_jobs.txt
```

### wait-wan-full-chain
```bash
# on Jean Zay
cat logs/r49_wan_full_chain_jobs.txt
squeue -u $USER
grep -hE 'failures|unit failures|dirs total' logs/r48_wan_draft_*.out logs/r48_div_*.out logs/r49_field_*.out logs/r48_wan_render_*.out logs/r49_wan_render_*.out logs/r49_eval_all_*.out | sort | uniq -c
```

### wan-full-fetch
```bash
# lab frontal (gpu-gw), inside tmux; JZ_HOST / JZ_REPO from jz/r50_jz.env
source jz/r50_jz.env
rsync -av --include='*_full*.csv' --include='r49_clip_directional_*_full*.csv' --exclude='*' \
  "$JZ_HOST:$JZ_REPO/evaluation/csv/" evaluation/csv/
bash jz/r50_fetch_outputs.sh 2>&1 | tee logs/r50_fetch.log     # scratch -> lab store; '[r50_fetch] failures: 0'
```

### wan-full-score
```bash
~/anaconda3/envs/five-bench/bin/python evaluation/r49_score.py --mode full --host wan
```

### full-verdict
```bash
cat evaluation/csv/r49_full_se_deltas.csv evaluation/csv/r49_full_wan_deltas.csv
ls evaluation/figures/r49_full_pareto/
```

## Pipeline

```mermaid
flowchart LR
  V[FiVE videos + r49_cases.json] --> S1[r49_se_stage1.sh: StreamEdit draft + attention]
  V --> WD[r49_wan_draft.sh: Wan-Edit draft + attention]
  S1 --> DV[r49_div.sh: DINO / LPIPS divergence]
  WD --> DV
  DV --> F[r49_field.py: S, div = max, q, rho / start step]
  S1 --> F
  WD --> F
  F --> SE[r49_se_stage3.sh: StreamEdit render, arm max]
  F --> WR[r49_wan_render.sh arms: max_s9, max_s5]
  F --> RR[r49_ref_render.sh: R48 dino_SD, new clip]
  NB[r49_newclip_baselines.sh: 10 global Wan configs, new clip] --> EB
  RR --> EB[r49_eval_all.sh base: 12 units]
  SE --> EM[r49_eval_all.sh main: 3 units]
  WR --> EM
  EB --> P[r49_post.sh: r49_score.py + r49_grids.py]
  EM --> P
  P -->|afterok, last| M59[r49_matched59.sh: matched-59 Wan curve, N 75/85/111]
  M59 --> E59[r49_eval_all.sh m59]
  E59 --> P2[r49_post.sh again: adds the matched-59 curve]
  P --> SG{se-full-gate}
  SG -->|go, lab L40S| SEF[StreamEdit full: r48 draft, then r49_se_full_chain.sh: div 0-1 -> r49_field.sh se full -> r49_se_stage3 full -> eval se]
  SEF --> SS[r49_score.py --mode full --host se]
  P2 --> WG{wan-full-gate: go + R50 PASS}
  WG -->|Jean Zay H100| WF[r49_wan_full_chain.sh: draft -> div 2-3 -> r49_field.sh wan full -> max_s5; 7 native incl. native_s0.978 + official; matched-59 x4 incl. s0.978; eval wan / base x7 / m59 x4]
  WF --> FE[wan-full-fetch: CSVs back by rsync]
  FE --> WS[r49_score.py --mode full --host wan]
  SS --> FV{full-verdict}
  WS --> FV
```

## Code to touch

| File | Change |
|---|---|
| `slurm_scripts/five_bench/r49_se_stage1.sh` | new: `r48_se_stage1.sh` with CASES = `evaluation/r49_cases.json`, roots `r49_se_stage1` / `r49_se_attn`; guards per type (1/9/3/2/3/0) + z_trg bit-identity with `r38_latents/n5_tgt09` |
| `slurm_scripts/five_bench/r49_wan_draft.sh` | new: `r48_wan_draft.sh` with CASES = `r49_cases.json`, roots `r49_wan_draft/{attn,frames}`; runs `evaluation/r48_wan_draft.py` unchanged |
| `slurm_scripts/five_bench/r49_div.sh` | new: one job looping `r31_divergence.py` over se_dino, se_lpips, wan_dino, wan_lpips (the four `r48_div.sh` array tasks, same flags, `--n_latent_frames 0` for Wan), drafts from the r49 roots, out `r49_div`; per-type npz guards |
| `evaluation/r49_field.py` | new: imports `r48_field` (`compute_E`, `_load_d`, GRID constants), `r30_b_map.t_next_schedule`, `r31_rho_map.budget_linear_tau`; own `DIV_MAX = {dino: 0.55, lpips: 1.0}`; writes `r49_field/se_max/edit{T}/{vid}.npz` (rho [F, 1560], rho_min 2, rho_max 50, step 15, flow_shift 1 + fields) and `r49_field/wan_max_s{9,5}/edit{T}/{vid}.{npz,npy}` (start_step int, wan_n 50, s_min + fields); CLI `--cases --attn_se --attn_wan --div_root --out_root`; `--selftest` |
| `slurm_scripts/five_bench/r49_field.sh` | new: `#SBATCH --partition=CPU`; selftest, then `r49_field.py` on `r49_cases.json`, then `r48_field.py --arms dino_SD --cases evaluation/r49_new_clip.json` with the r49 attention / divergence roots and `--out_root r48_field`; guards: 18 npz per r49 dir (+ .npy for the two wan dirs), 1 reference npz per host |
| `slurm_scripts/five_bench/r49_se_stage3.sh` | new: copy of `r48_se_stage3.sh` for the single arm max: `--rho_dir r49_field/se_max`, METHOD `r49_max_subset`, out `r49_se_arms`, cases `r49_cases.json`, same render flags |
| `evaluation/r49_wan_render.py` | new: like `r48_wan_render.py`; `--config max_s9 / max_s5` (adaptive: loop skip 9 / 5, start map from `r49_field/wan_{arm}`, `load_route_start_map(map, 0, skip)` keeps its min ≥ skip check) or `matched59_s{0.95,0.92,0.85}` (global: `r48_wan_sched.matched_n(sg, 59)`, start σ asserted against the real scheduler within 0.015) |
| `slurm_scripts/five_bench/r49_wan_render.sh` | new: MODE `arms` (array 0-1 → max_s9, max_s5; preflight 18 start maps) or `m59` (array 0-2 → matched59 configs); post-run guard via `r48_units.py --missing` with `r49_cases.json` |
| `slurm_scripts/five_bench/r49_newclip_baselines.sh` | new: one GPU job, loop over the 9 `r48_wan_sched.py --list` configs plus `native_s0.978`: `r48_wan_render.py --config C --mode subset --cases evaluation/r49_new_clip.json --out_root r48_wan_arms`; counts failures, checks the new clip's frame dir in each config |
| `slurm_scripts/five_bench/r49_ref_render.sh` | new: one GPU job; StreamEdit dino_SD for the new clip (`r31_stage3.py --rho_dir r48_field/se_dino_SD --cases_json evaluation/r49_new_clip.json --method r48_dino_SD_subset`, flags as `r48_se_stage3.sh`), then `r48_wan_render.py --config adapt_dino_SD --mode subset --cases evaluation/r49_new_clip.json`; checks both clip dirs exist |
| `evaluation/r49_units.py` | new: `--group main` = 3 units, `--group base` = 12 units (9 baselines, native_s0.978, se_r48_dino_SD, wan_r48_dino_SD), `--group m59` = 3 units; TSV like `r48_units.py` |
| `slurm_scripts/five_bench/r49_eval_all.sh` | new: argument GROUP (main / base / m59); serial loop over `r49_units.py --group GROUP` with the `r48_eval.sh` and `r48_fiveacc.sh` evaluate.py calls (`--cases_json evaluation/r49_cases.json`), stems `r49_{base}` / `r49_fiveacc_{base}`, then `r33_clip_directional.py --cases evaluation/r49_cases.json` over the group → `r49_clip_directional_{GROUP}.csv`; prints `[r49_eval] unit failures: N` |
| `evaluation/r49_score.py` | new: reuses `r48_score` loaders (`unit_values`, `mean_point`, `interp_at`, `paired_ci`, `plot`) with the 18-clip scope from `r49_cases.json`; Wan figure draws the native, matched-55 and (if its CSVs exist) matched-59 curves; writes `r49_points.csv`, `r49_deltas.csv`, `evaluation/figures/r49_pareto/{se,wan}_{five_acc,clip_d_prompt}.{pdf,png}` |
| `evaluation/r49_grids.py` | new: frame grids (`evaluation/figures/r49_grids/frames_{se,wan}.png`, one row per clip) and field pages (`evaluation/figures/r49_grids/fields/{host}/e{T}_{vid}.png`), styled like the R48 contact sheets and `r48_tau_grids.py` |
| `slurm_scripts/five_bench/r49_post.sh` | new: `#SBATCH --partition=CPU`; runs `r49_score.py` then `r49_grids.py`; checks the CSVs, figure dirs and 36 field pages exist |
| `slurm_scripts/five_bench/r49_chain.sh` | new: `sbatch --parsable` each stage with `--dependency=afterok:<ids>` as in Decisions → Chain; writes `logs/r49_chain_jobs.txt` (one `<stage> <id>` per line: newclip_baselines, se_stage1, wan_draft, div, field, se_stage3, wan_render, ref_render, eval_base, eval_main, post) |
| `slurm_scripts/five_bench/r49_matched59.sh` | new: reads the main `post` id from `logs/r49_chain_jobs.txt`; submits `r49_wan_render.sh m59` (array 0-2, `afterok:<post>`) → `r49_eval_all.sh m59` → `r49_post.sh`, appends its ids; on a refused submission prints the stage and exits non-zero |
| `slurm_scripts/five_bench/r49_field.sh` | full mode + hosts: optional `HOSTS` (default `se wan`) and `MODE` (default subset); full = r7 manifest, r48 attention/divergence roots, out `r49_field_full`, no r48_field reference; guards per listed host with 18 / 419 |
| `slurm_scripts/five_bench/r49_se_full_chain.sh` | new: `DRAFT_JOB` argument; afterok chain `r48_div.sh full` (array 0-1) → `r49_field.sh se full` → `r49_se_stage3.sh full` → `r49_eval_all.sh se full`; ids to `logs/r49_se_full_chain_jobs.txt` |
| `slurm_scripts/five_bench/r49_se_stage3.sh` | full mode: optional MODE argument (default subset); full = r7 manifest without `--cases_json` (as `r48_se_stage3.sh`), field `r49_field_full/se_max`, METHOD `r49_max_full`, 419-pair guards |
| `evaluation/r49_wan_render.py` | full mode: `--mode subset|full` (default subset) → output `{config}_{mode}`; `MATCHED59` += 0.978 appended (index 3) |
| `slurm_scripts/five_bench/r49_wan_render.sh` | full mode: optional second argument subset|full (default subset); full cases, `--mode full`, guard with `r48_units.py --mode full --missing` |
| `evaluation/r49_units.py` | full mode: `--mode subset|full` (default subset); `_full` dirs and base names; full-mode groups by host: `se` = se_max, `wan` = wan_max_s5, `base` = the 9 R48 Wan-Edit configs `{config}_full`, `m59` = matched59 `_full` |
| `slurm_scripts/five_bench/r49_eval_all.sh` | full mode: optional second argument (default subset); r7 counts, no `--cases_json`, CLIP-D `--fullbench`, `r49_clip_directional_{group}_full.csv`; groups `se`, `wan`, `base`, `m59` |
| `slurm_scripts/five_bench/r48_wan_render.sh` | new MODE `explore_full`: r7 manifest, `--mode full`, `--list_explore` index 0 = `native_s0.978` → `r48_wan_arms/native_s0.978_full` |
| `evaluation/r49_units.py` | full Wan groups v2: `base` = 6 native + `native_s0.978` (7), `m59` = 4 incl. `matched59_s0.978` |
| `slurm_scripts/five_bench/r49_eval_all.sh` | full mode as a Slurm array: task i scores unit i only, CLIP-D `r49_clip_directional_{group}_full_{base}.csv` |
| `jz/r50_scratch_links.sh` | new (Jean Zay): symlinks the Wan output folders under `$WORK/Data/dataggen/outputs/five_bench` to `$SCRATCH/five_bench_out/`, so job scripts are unchanged |
| `jz/r50_fetch_outputs.sh` | new (lab frontal): rsync scratch outputs to `/store/PERSO/skhairi/Outputs/jz_five_bench/` (resumable, no --delete), compare file counts and bytes per folder, print (not run) the commands that free the scratch copy; no links into `~/Data` (platforms stay separate) |
| `slurm_scripts/five_bench/r49_wan_full_chain.sh` | new (Jean Zay): draft → div 2-3 → field → `max_s5`; 6 native + `native_s0.978` + 4 matched-59 renders; evals `wan`, `base` array 0-6, `m59` array 0-3, each afterok on its renders; ids to `logs/r49_wan_full_chain_jobs.txt` |
| `evaluation/r49_score.py` | full mode: `--mode full --host se|wan` (one host per run), scope 419 + the 401 outside the 18 clips; se: R39/R36/R38 curves + se_max (L40S); wan: base + m59 curves + wan_max_s5 vs matched-59 (H100, fetched CSVs); `platform` column and figure titles; R48 references skipped; `r49_full_{host}_points.csv`, `r49_full_{host}_deltas.csv`, `figures/r49_full_pareto/{host}_*` |
