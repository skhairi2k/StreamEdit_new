---
name: "R58 — Wan-Edit on FiVE-Bench-Correct (renders on Jean Zay H100, evaluation on lab L40S)"
overview: "Renders, the reproducibility gate and the assembly run on JEAN ZAY (H100, R50 five-bench overlay); the EVALUATION of all 12 configs runs on the LAB CLUSTER L40S with R57's suite, like every other main-table row (user, 2026-10-27). Wan-Edit's three R49 arms on the 415 FiVE-Bench-Correct pairs: native sweep (7 global starts, N=50, native_s0.92 = official Wan-Edit), equal-NFE matched-59 curve (4 starts) and the adaptive max_s5. Only pairs whose Wan-Edit inputs changed are re-generated: 47 pairs (source/target prompt changed) for the 11 baseline configs, 190 pairs (prompt or trigger object changed) for max_s5. Every other clip reuses the R49 full-stage H100 render, after a 2-pair reproducibility gate. Per config, a 415-pair symlink tree is assembled on Jean Zay, then copied to the lab as real files (rsync -L) and scored there with eval_suite.sh (Correct annotations), then read as Pareto curves + paired deltas."
task_id: R58
todos:
  - id: anno-dir-common
    content: "evaluation/r48_wan_common.py -- load_cases(cases_path, anno_dir=None): read edit{T}_FiVE.json from anno_dir when given, else FIVE/edit_prompt (default unchanged, R48/R49/R51 calls keep their behaviour)"
    needed_by: [repro-render, wan-chain]
    status: completed
  - id: anno-dir-draft
    content: "evaluation/r48_wan_draft.py -- new --anno_dir argument (default None = official FiVE) passed to C.load_cases; printed in the start line"
    needed_by: [wan-chain]
    status: completed
  - id: anno-dir-render48
    content: "evaluation/r48_wan_render.py -- new --anno_dir argument (default None) passed to C.load_cases; printed in the start line"
    needed_by: [repro-render, wan-chain]
    status: completed
  - id: anno-dir-render49
    content: "evaluation/r49_wan_render.py -- new --anno_dir argument (default None) passed to C.load_cases; printed in the start line"
    needed_by: [repro-render, wan-chain]
    status: completed
  - id: cases-script
    content: "evaluation/r58_build_cases.py -- compares evaluation/five_bench_correct/edit{T}_FiVE.json with the official ~/Data/FiVE-.../edit_prompt/edit{T}_FiVE.json per pair (415 = r43_anchor_manifest_correct.json); writes evaluation/r58_cases_base.json (pairs with a changed source_prompt/target_prompt, expect 47), evaluation/r58_cases_maxs5.json (changed prompt OR source_object/target_object, expect 190), evaluation/r58_cases_repro.json (2 unchanged pairs: the first unchanged pair of edit1 and of edit4) and evaluation/csv/r58_regen_sets.csv (case_id, changed fields, in_base, in_maxs5); all case files in the r7 manifest format; exit 1 if the counts differ from 47 / 190 / 2 or a case is missing from the official files"
    needed_by: [build-cases]
    status: completed
  - id: render-sbatch
    content: "slurm_scripts/five_bench/r58_wan_render.sh -- MODE correct (array 0-11): 0-5 native_s0.95..s0.4 and 6 native_s0.978 via r48_wan_render.py, 7-10 matched59_s0.95/0.92/0.85/0.978 via r49_wan_render.py, all on r58_cases_base.json; 11 max_s5 via r49_wan_render.py on r58_cases_maxs5.json with --field_root $D/r58_field; MODE repro (array 0-1): native_s0.92 (r48) and matched59_s0.92 (r49) on r58_cases_repro.json -> $D/r58_repro; always --anno_dir evaluation/five_bench_correct --mode full --out_root $D/r58_wan_arms (repro: $D/r58_repro); NO-OVERWRITE guard: refuses to start unless OUT is exactly $D/r58_wan_arms or $D/r58_repro (never an r48_/r49_ folder), and the renderers' resume logic skips any clip that already has its PNGs, so no existing frame is rewritten; max_s5 preflight on the start maps; post-run guard with r48_units.py --missing; lab header with --exclude=node52 (ported by r50_port.sh)"
    needed_by: [repro-render, wan-chain]
    status: completed
  - id: draft-sbatch
    content: "slurm_scripts/five_bench/r58_wan_draft.sh -- r48_wan_draft.py on r58_cases_maxs5.json with --anno_dir evaluation/five_bench_correct -> $D/r58_wan_draft/{attn,frames}; same draft settings as R49 (17-step grid from step 3, seed 0); post-run per-type count guard against the 190-pair case file"
    needed_by: [wan-chain]
    status: completed
  - id: div-sbatch
    content: "slurm_scripts/five_bench/r58_div.sh -- array 0-1 = wan_dino / wan_lpips: r31_divergence.py on r58_cases_maxs5.json, --target_root $D/r58_wan_draft/frames/step00, --n_latent_frames 0, --temporal_reduce mean --no-save_native --seed 0 -> $D/r58_div/{wan_dino,wan_lpips}; per-type count guard"
    needed_by: [wan-chain]
    status: completed
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r58_field.sh -- r49_field.py --selftest, then r49_field.py --hosts wan --cases evaluation/r58_cases_maxs5.json --attn_wan $D/r58_wan_draft/attn --div_root $D/r58_div --out_root $D/r58_field; guard: 190 npz + 190 .npy in wan_max_s5"
    needed_by: [wan-chain]
    status: completed
  - id: chain-driver
    content: "slurm_scripts/five_bench/r58_wan_chain.sh -- Jean Zay driver modelled on r49_wan_full_chain.sh: guards (ported scripts, wrapper python, r58_* case files, $D/r58_* are symlinks to $SCRATCH, the R49 render dirs exist), --dry-run, refuses if logs/r58_wan_chain_jobs.txt exists; submits draft -> div (afterok) -> field (afterok) -> render max_s5 (--array=11, afterok) and render baselines (--array=0-10, no dependency); records '<stage> <id>' lines headed by the git commit"
    needed_by: [wan-chain]
    status: completed
  - id: repro-check-script
    content: "evaluation/r58_repro_check.py -- for each repro pair and config, compares $D/r58_repro/{config}_full/edit{T}/{video}/*.png with the R49 render ($D/r48_wan_arms/native_s0.92_full, $D/r49_wan_arms/matched59_s0.92_full): same frame count, max |diff| per frame and PSNR; PASS iff every frame has max |diff| <= 2 (of 255); writes evaluation/csv/r58_repro_check.csv; exit 1 on FAIL"
    needed_by: [repro-check]
    status: completed
  - id: assemble-script
    content: "evaluation/r58_assemble.py -- for the 12 configs builds $D/r58_wan_correct/{config}/edit{T}/{video} as a symlink, over the 415 pairs of r43_anchor_manifest_correct.json: to $D/r58_wan_arms/{config}_full/... if the pair is in that config's regen set (r58_cases_base.json for the 11 baselines, r58_cases_maxs5.json for max_s5), else to the R49 render ($D/r48_wan_arms/{config}_full for native_*, $D/r49_wan_arms/{config}_full for matched59_* and max_s5); checks the target holds the expected PNG count (r48_wan_common.expected_frames), read through the link; only ever creates symlinks inside $D/r58_wan_correct and never writes into a link target; an existing entry that is not a symlink, or a symlink into neither r58_wan_arms nor the R48/R49 render dirs, is a CONFLICT (left untouched, exit 1); re-running is idempotent (an identical link is kept); writes evaluation/csv/r58_assembly.csv (config, case_id, source new|reused, n_png); exit 1 on any missing/incomplete clip or a count other than 415 per config"
    needed_by: [assemble]
    status: pending
  - id: fetch-renders-script
    content: "jz/r58_fetch_renders.sh -- lab frontal (gpu-gw) only, modelled on jz/r50_fetch_outputs.sh (same r50_jz.env, SSH ControlMaster, $SCRATCH lookup): first rsync of Jean Zay evaluation/csv/r58_assembly.csv into evaluation/csv/; then per config (all 12, or the configs given as arguments) rsync -aL --partial (follows the assembly's symlinks, so the lab gets real files) zay:$SCRATCH/five_bench_out/r58_wan_correct/{config}/ -> $R58_DEST/{config}/ (default R58_DEST=/work/PERSO/skhairi/five_bench/r58_wan_correct); preflight: R58_DEST's filesystem has >= 12 GB free per requested config; post-check per config: 415 clip dirs and every clip's PNG count equals n_png of r58_assembly.csv; --check mode counts only; never deletes anything on either side; ends with '[r58_fetch] failures: N'"
    needed_by: [fetch-renders]
    status: pending
  - id: eval-lab-launcher
    content: "slurm_scripts/five_bench/r58_eval.sh -- lab launcher (bash, not sbatch): for each config given as argument (subset of the 12; no argument = refuse), bash slurm_scripts/five_bench/eval_suite.sh --method wan_{config, dot -> p}_correct --root $R58_DEST/{config} --layout edit_video --tag r58 --allow_nan motion_fidelity_score_edit_part (default --edit_json_dir = Correct, no --cases = 415 pairs); preflight: eval_suite.sh present, 415 clip dirs per root, the user's queue + 3 x n_configs <= 30 tasks; refuses a config whose suite_r58_wan_{config}_correct.csv already exists or whose method already has a line in logs/r58_eval_jobs.txt; appends '<method> <metrics> <cf> <merge>' job ids to logs/r58_eval_jobs.txt"
    needed_by: [eval-1, eval-2, eval-3]
    status: pending
  - id: port-list
    content: "jz/r50_port.sh -- add r58_wan_draft, r58_div, r58_field, r58_wan_render, r58_wan_chain to SCRIPTS"
    needed_by: [jz-sync]
    status: completed
  - id: scratch-folders
    content: "jz/r50_scratch_links.sh -- add r58_wan_draft r58_div r58_field r58_wan_arms r58_repro r58_wan_correct to FOLDERS (and the header comment)"
    needed_by: [jz-sync]
    status: completed
  - id: fetch-folders
    content: "jz/r50_fetch_outputs.sh -- add r58_wan_draft r58_div r58_field r58_wan_arms to ALL (and the header comment); r58_wan_correct is links only and is not fetched by this script (r58_fetch_renders.sh copies it as real files)"
    needed_by: [archive]
    status: pending
  - id: score-script
    content: "evaluation/r58_score.py -- reads evaluation/csv/suite_r58_wan_{config}_correct.csv (12 configs, 415 rows each; exit 1 otherwise); writes evaluation/csv/r58_wan_points.csv (config, role native|m59|arm, plain per-clip mean overall and per edit type, every suite metric), evaluation/csv/r58_wan_deltas.csv (max_s5 minus matched59_s0.92 / native_s0.92 / native_s0.978, paired bootstrap 95% CI, 10k resamples seed 0, overall + per type + regenerated-vs-reused split), and evaluation/figures/r58_pareto/{alignment}_vs_{preservation}.pdf (x = CLIP-T, FiVE-Acc, CLIP-D, EditReward, PSC; y = bg LPIPS, CF; native curve, matched-59 curve, max_s5 point, official native_s0.92 marked)"
    needed_by: [score]
    status: pending
steps:
  - id: build-cases
    type: local
    command: python evaluation/r58_build_cases.py
    output_paths:
      - evaluation/r58_cases_base.json
      - evaluation/r58_cases_maxs5.json
      - evaluation/r58_cases_repro.json
      - evaluation/csv/r58_regen_sets.csv
    status: completed
    completed_at: 2026-10-26
  - id: jz-sync
    type: manual
    wait_for: build-cases
    check_hint: "User: commit + push (incl. the final evaluation/five_bench_correct/*.json and the r58_cases_*.json). On Jean Zay in the clone: `git checkout -- slurm_scripts evaluation/fivebench/config.yaml && git pull && bash jz/r50_verify.sh --code-only && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh`; paste the tails: '[r50_port] leftovers: 0', '[r50_port] failures: 0', '[r50_scratch] failures: 0'; `ls $D/r48_wan_arms/native_s0.92_full/edit1 | wc -l` = 100 (R49 renders still on $SCRATCH)"
    status: completed
    completed_at: 2026-10-26
  - id: repro-render
    type: manual
    wait_for: jz-sync
    check_hint: "User, on Jean Zay: `sbatch -A omt@h100 --qos=qos_gpu_h100-dev --time=01:00:00 --array=0-1 slurm_scripts/five_bench/r58_wan_render.sh repro`; paste the job id and, when done, both log tails ('failures: 0')"
    status: completed
    completed_at: 2026-10-27
  - id: repro-check
    type: manual
    wait_for: repro-render
    check_hint: "User, on Jean Zay: `$WORK/pyenv/five-bench/python evaluation/r58_repro_check.py`; must print PASS (max |diff| <= 2 on every frame of both pairs, both renderers). FAIL = the R49 renders cannot be mixed with new ones: stop, bring evaluation/csv/r58_repro_check.csv back, decide with the user (full 415 re-render)"
    status: completed
    completed_at: 2026-10-27
  - id: wan-chain
    type: manual
    wait_for: repro-check
    check_hint: "User, on Jean Zay: first `git pull && bash jz/r50_port.sh` (the 4 other r58_*.sh scripts were not in the first sync and must now be ported), then `bash slurm_scripts/five_bench/r58_wan_chain.sh --dry-run` then `bash slurm_scripts/five_bench/r58_wan_chain.sh`; paste logs/r58_wan_chain_jobs.txt (5 stages: draft, div, field, render_max_s5, render_base)"
    sets_status: running
    status: completed
    completed_at: 2026-10-27
    job_id: "1617141-1617145"
  - id: wait-chain
    type: manual
    wait_for: wan-chain
    check_hint: "User, on Jean Zay, when `squeue -u $USER` is empty: `grep -h failures logs/r58_*.out | sort | uniq -c` -> every line 'failures: 0'; render logs report 47/47 dirs for configs 0-10 and 190/190 for max_s5; field log 'wan_max_s5: 190 npz'; R49 videos untouched: `find ~/Data/dataggen/outputs/five_bench/{r48_wan_arms,r49_wan_arms}/ -type f -newer logs/r58_wan_chain_jobs.txt | wc -l` = 0"
    sets_status: finished
    status: pending
  - id: assemble
    type: manual
    wait_for: wait-chain
    check_hint: "User, on Jean Zay: `$WORK/pyenv/five-bench/python evaluation/r58_assemble.py`; must end with 12 configs x 415 clips, new = 47 (baselines) / 190 (max_s5), 0 missing; paste the summary"
    status: pending
  - id: probe-mounts
    type: manual
    check_hint: "User: R58_DEST (default /work/PERSO/skhairi/five_bench/r58_wan_correct) must be writable from the frontal AND readable from L40S compute nodes. On gpu-gw: `mkdir -p /work/PERSO/skhairi/five_bench/r58_wan_correct && df -h /work/PERSO/skhairi` (>= 140 GB free for 12 configs x ~9.3 GB); then `srun -p L40S --exclude=node52 --gres=gpu:1 -t 00:05:00 ls -d /work/PERSO/skhairi/five_bench/r58_wan_correct`. If either fails: pick another R58_DEST visible on both (not /store, not on every node), or fetch/evaluate in waves under ~/Data (home quota: ~/Data is at 418 GB)"
    status: pending
  - id: fetch-renders
    type: manual
    wait_for: [assemble, probe-mounts]
    check_hint: "User, on the lab frontal gpu-gw, in tmux: `bash jz/r58_fetch_renders.sh --check` then `bash jz/r58_fetch_renders.sh` (all 12 configs; ~110 GB) -> '[r58_fetch] failures: 0', evaluation/csv/r58_assembly.csv present. The R48/R49 renders must still be on Jean Zay $SCRATCH (IDRIS purge) until this step passes"
    status: pending
  - id: wait-r57
    type: manual
    check_hint: "R57 lab smoke passed: slurm_scripts/five_bench/eval_suite.sh exists, evaluation/csv/suite_r57_smoke_lab_*.csv has no NaN (same gate as R59's wait-r57)"
    status: pending
  - id: eval-1
    type: local
    wait_for: [fetch-renders, wait-r57]
    command: bash slurm_scripts/five_bench/r58_eval.sh native_s0.92 native_s0.978 matched59_s0.92 max_s5
    status: pending
  - id: eval-2
    type: local
    wait_for: eval-1
    command: bash slurm_scripts/five_bench/r58_eval.sh native_s0.95 native_s0.85 native_s0.75 native_s0.6
    status: pending
  - id: eval-3
    type: local
    wait_for: eval-2
    command: bash slurm_scripts/five_bench/r58_eval.sh native_s0.4 matched59_s0.95 matched59_s0.85 matched59_s0.978
    status: pending
  - id: wait-eval
    type: manual
    wait_for: eval-3
    check_hint: "12 files evaluation/csv/suite_r58_wan_*_correct.csv, 415 rows each (merge logs exit 0, or merged by hand when afterok was blocked by the edit-part NaN line); `ls evaluation/csv/suite_r58_wan_*_correct.csv | wc -l` = 12; job ids in logs/r58_eval_jobs.txt"
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
    command: python evaluation/r58_score.py
    output_paths:
      - evaluation/csv/r58_wan_points.csv
      - evaluation/csv/r58_wan_deltas.csv
      - evaluation/figures/r58_pareto/
    status: pending
  - id: archive
    type: manual
    wait_for: wait-eval
    check_hint: "User, on gpu-gw in tmux: `bash jz/r50_fetch_outputs.sh r58_wan_draft r58_div r58_field r58_wan_arms` -> '[r50_fetch] failures: 0' (intermediates + new renders archived to /store/PERSO/skhairi/Outputs/jz_five_bench); only then may the Jean Zay $SCRATCH copies be deleted"
    status: pending
  - id: analyze
    type: manual
    wait_for: score
    check_hint: "Read r58_wan_points.csv / r58_wan_deltas.csv and the Pareto figures: is max_s5 above the matched-59 curve on Correct as on official (R49 full)? per-type pattern; regenerated vs reused split shows no platform/version shift. Record the readout in the Run log."
    sets_status: analyzed
    status: pending
isProject: true
---

# R58: Wan-Edit on FiVE-Bench-Correct

## Context
FiVE-Bench-Correct is the main benchmark of the paper (supervisor, 2026-10-23), so Wan-Edit's R49 arms must be reported on its 415 pairs: the native sweep (official Wan-Edit = native_s0.92), the equal-NFE matched-59 curve and the adaptive max_s5. Wan-Edit is text-only, so a clip whose prompts and trigger words are unchanged by Correct renders exactly as in R49. Generation is therefore limited to the pairs whose inputs changed, while evaluation is redone on all 415 pairs, because Correct changed the FiVE-Acc questions, objects and instructions the metrics read.

Rendering stays on Jean Zay H100 next to the R49 renders it reuses. Every Wan-Edit config is scored on the lab L40S with the same R57 suite as the StreamEdit, AnyV2V and ContextFlow rows, so no main-table number depends on the scoring platform (user, 2026-10-27).

**Done when:** 12 configs × 415 pairs are scored with R57's suite on the lab L40S, and `r58_wan_points.csv` / `r58_wan_deltas.csv` / `r58_pareto/` are read.

## Execution steps

| # | step id | type | runs on | waits for | sets status |
|---|---|---|---|---|---|
| — | *(prep)* assemble-script · fetch-renders-script · eval-lab-launcher · fetch-folders · score-script (others completed) | todos | lab (code) | — | `implemented` |
| 1 | `build-cases` | local | lab | — | — |
| 2 | `jz-sync` | manual | lab → **Jean Zay** | build-cases | — |
| 3 | `repro-render` | manual | **Jean Zay** | jz-sync | — |
| 4 | `repro-check` | manual | **Jean Zay** | repro-render | — |
| 5 | `wan-chain` | manual | **Jean Zay** | repro-check | `running` |
| 6 | `wait-chain` | manual | **Jean Zay** | wan-chain | `finished` |
| 7 | `assemble` | manual | **Jean Zay** | wait-chain | — |
| 8 | `probe-mounts` | manual | lab frontal + L40S node | — | — |
| 9 | `fetch-renders` | manual (user, tmux) | lab frontal | assemble, probe-mounts | — |
| 10 | `wait-r57` | manual | lab | R57 lab smoke | — |
| 11 | `eval-1` | local (submits eval_suite jobs) | **lab L40S** | fetch-renders, wait-r57 | — |
| 12 | `eval-2` | local | **lab L40S** | eval-1 | — |
| 13 | `eval-3` | local | **lab L40S** | eval-2 | — |
| 14 | `wait-eval` | manual | lab | eval-3 | — |
| 15 | `score` | local | lab | wait-eval | — |
| 16 | `archive` | manual (user, tmux) | lab frontal | wait-eval | — |
| 17 | `analyze` | manual | lab | score | `analyzed` |

```
/build-step assemble R58
/build-step fetch-renders R58
/build-step eval-1 R58
/build-step score R58
/build-step archive R58
/run-step R58 wait-chain
/run-step R58 assemble
/run-step R58 probe-mounts
/run-step R58 fetch-renders
/run-step R58 wait-r57
/run-step R58 eval-1
/run-step R58 eval-2
/run-step R58 eval-3
/run-step R58 wait-eval
/run-step R58 score
/run-step R58 archive
/run-step R58 analyze
```

## Decisions

| Decision | Choice |
|---|---|
| Benchmark | FiVE-Bench-Correct, 415 pairs (`evaluation/r43_anchor_manifest_correct.json`, annotations `evaluation/five_bench_correct/`); the 4 excluded pairs drop out. The uncommitted edits to `edit1_FiVE.json` / `edit6_FiVE.json` are committed before `build-cases`; the regeneration sets are computed from the committed files |
| Regeneration sets | Computed from the JSON files (official vs Correct), not from `r43_correct_diff.csv`. Baselines (11 configs): the **47** pairs with a changed `source_prompt` / `target_prompt` (the only fields their render reads). max_s5: the **190** pairs with a changed prompt **or** `source_object` / `target_object` (the draft's attention support reads the trigger tokens). Checked 2026-10-26: 47 / 166 / 190 |
| Configs (12) | native N=50: `native_s0.95`, `s0.92` (official Wan-Edit), `s0.85`, `s0.75`, `s0.6`, `s0.4`, `s0.978`; matched-59: `matched59_s0.95`, `s0.92`, `s0.85`, `s0.978`; adaptive `max_s5` |
| Equal NFE | 59 = 14-step draft + 45 active render steps from the s_min 5 floor: fixed by construction, so it holds on Correct unchanged |
| Settings | Frozen from R49's full stage: draft = 17-step grid from step 3 (σ 0.958), seed 0; div = DINOv3 + LPIPS, temporal mean; field = `r49_field.py` (absolute caps, per clip); render seeds as R49 |
| Render platform | **Jean Zay H100**, R50 `five-bench` overlay, for every render, the gate and the assembly (reused R49 renders are H100 too); no Wan-Edit render is ever made on L40S |
| Eval platform | **Lab L40S**, R57 `eval_suite.sh` (envs `five-bench` + `editreward` of the lab), for **all 12 configs**, the same suite and platform as every other main-table row; Jean Zay's `jz/eval_suite_jz.sh` is not used for R58 |
| No overwrite | No existing video is written, moved or deleted. New renders only in new folders (`r58_wan_arms`, `r58_repro`; the render script refuses any other output root); the renderers skip clips that already have their PNGs; `r58_assemble.py` only creates symlinks inside `r58_wan_correct` and stops on any conflict; `wait-chain` checks that no file in `r48_wan_arms` / `r49_wan_arms` is newer than the chain submission; the fetch scripts never delete |
| Reuse gate | Before mixing new and reused renders: 2 unchanged pairs re-rendered with `native_s0.92` (r48 renderer) and `matched59_s0.92` (r49 renderer); PASS iff max \|diff\| ≤ 2/255 on every frame. FAIL stops the task (decision with the user: full re-render) |
| Output dirs (Jean Zay) | New renders under `$D/r58_wan_arms/{config}_full/`; draft `$D/r58_wan_draft`, divergence `$D/r58_div`, field `$D/r58_field`, repro `$D/r58_repro`, assembly `$D/r58_wan_correct`; all on `$SCRATCH` via `r50_scratch_links.sh`. The R48/R49 renders must stay on `$SCRATCH` until `fetch-renders` passes (IDRIS purge) |
| Assembly (two folders → one root) | Per config, `$D/r58_wan_correct/{config}/edit{T}/{video}` is a symlink on Jean Zay to the new render (`r58_wan_arms`) if the pair is in the regen set, else to the R49 render (`r48_wan_arms` / `r49_wan_arms`) |
| Transfer | `jz/r58_fetch_renders.sh` on gpu-gw: `rsync -aL` of each assembled tree, so the lab holds one ordinary 415-clip tree of real PNGs per config at `R58_DEST/{config}` (default `/work/PERSO/skhairi/five_bench/r58_wan_correct`, ~9.3 GB per config, ~110 GB total; not `/store`, which L40S nodes cannot all read, and not `~/Data`, near the home quota); PNG count per clip checked against `r58_assembly.csv` |
| Evaluation | `eval_suite.sh --root R58_DEST/{config} --layout edit_video --tag r58 --method wan_{config}_correct` (dot → `p`), all 415 pairs, `--allow_nan motion_fidelity_score_edit_part` → `evaluation/csv/suite_r58_wan_{config}_correct.csv`; submitted in 3 batches of 4 configs (12 jobs each) to stay under the 30-task lab queue cap, headline configs first (native_s0.92, native_s0.978, matched59_s0.92, max_s5) |
| Averaging / plots | Plain per-clip mean from the suite CSV, never `_avg.csv`; Pareto plots with x = alignment, y = preservation |
| Readout | max_s5 vs matched-59 (equal NFE, headline), vs native_s0.92 (official) and native_s0.978 (same floor); paired bootstrap 95% CI; per edit type; regenerated vs reused split as a sanity check |
| Out of scope | StreamEdit arms on Correct (separate tasks); the R51 A4 Wan arm; re-rendering unchanged pairs; Jean Zay scoring |

## Step commands

### build-cases
```bash
python evaluation/r58_build_cases.py
```

### jz-sync
```bash
# lab: commit + push (Correct JSONs, r58_cases_*.json, the R58 scripts)
# Jean Zay (ssh zay), in $JZ_REPO:
git checkout -- slurm_scripts evaluation/fivebench/config.yaml && git pull
bash jz/r50_verify.sh --code-only
bash jz/r50_port.sh
bash jz/r50_scratch_links.sh
ls ~/Data/dataggen/outputs/five_bench/r48_wan_arms/native_s0.92_full/edit1 | wc -l   # 100
```

### repro-render
```bash
# Jean Zay
sbatch -A omt@h100 --qos=qos_gpu_h100-dev --time=01:00:00 --array=0-1 slurm_scripts/five_bench/r58_wan_render.sh repro
```

### repro-check
```bash
# Jean Zay
$WORK/pyenv/five-bench/python evaluation/r58_repro_check.py
```

### wan-chain
```bash
# Jean Zay
bash slurm_scripts/five_bench/r58_wan_chain.sh --dry-run
bash slurm_scripts/five_bench/r58_wan_chain.sh
```

### wait-chain
```bash
# Jean Zay, queue empty
grep -h failures logs/r58_*.out | sort | uniq -c
```

### assemble
```bash
# Jean Zay
$WORK/pyenv/five-bench/python evaluation/r58_assemble.py
```

### probe-mounts
```bash
# lab frontal (gpu-gw)
mkdir -p /work/PERSO/skhairi/five_bench/r58_wan_correct && df -h /work/PERSO/skhairi
srun -p L40S --exclude=node52 --gres=gpu:1 -t 00:05:00 ls -d /work/PERSO/skhairi/five_bench/r58_wan_correct
```

### fetch-renders
```bash
# lab frontal (gpu-gw), in tmux
bash jz/r58_fetch_renders.sh --check
bash jz/r58_fetch_renders.sh
```

### wait-r57
```bash
ls slurm_scripts/five_bench/eval_suite.sh
ls evaluation/csv/suite_r57_smoke_lab_*.csv
```

### eval-1
```bash
bash slurm_scripts/five_bench/r58_eval.sh native_s0.92 native_s0.978 matched59_s0.92 max_s5
```

### eval-2
```bash
bash slurm_scripts/five_bench/r58_eval.sh native_s0.95 native_s0.85 native_s0.75 native_s0.6
```

### eval-3
```bash
bash slurm_scripts/five_bench/r58_eval.sh native_s0.4 matched59_s0.95 matched59_s0.85 matched59_s0.978
```

### wait-eval
```bash
cat logs/r58_eval_jobs.txt
ls evaluation/csv/suite_r58_wan_*_correct.csv | wc -l   # 12
```

### score
```bash
python evaluation/r58_score.py
```

### archive
```bash
# lab frontal (gpu-gw), in tmux
bash jz/r50_fetch_outputs.sh r58_wan_draft r58_div r58_field r58_wan_arms
```

### analyze
```bash
# read evaluation/csv/r58_wan_points.csv, evaluation/csv/r58_wan_deltas.csv, evaluation/figures/r58_pareto/
```

## Pipeline

```mermaid
flowchart LR
  A[official edit_prompt JSON] --> B[r58_build_cases.py]
  C[five_bench_correct JSON] --> B
  B --> D[r58_cases_base 47]
  B --> E[r58_cases_maxs5 190]
  B --> F[r58_cases_repro 2]
  F --> G[r58_wan_render.sh repro<br/>Jean Zay] --> H[r58_repro_check.py]
  R49[R49 H100 renders on SCRATCH] --> H
  E --> I[r58_wan_draft.sh] --> J[r58_div.sh] --> K[r58_field.sh] --> L[r58_wan_render.sh max_s5]
  D --> M[r58_wan_render.sh 11 baselines]
  L --> N[r58_assemble.py<br/>Jean Zay]
  M --> N
  R49 --> N
  N --> O[r58_wan_correct 12 x 415 links]
  O --> P[r58_fetch_renders.sh<br/>rsync -L to lab]
  P --> Q[R58_DEST 12 x 415 real PNG trees]
  Q --> U[r58_eval.sh x3 batches<br/>R57 eval_suite.sh, lab L40S]
  C --> U
  U --> V[suite_r58 CSVs]
  V --> S[r58_score.py]
  S --> T[r58_wan_points / deltas / r58_pareto]
  L --> W[r50_fetch_outputs.sh archive to store]
```

## Code to touch

| File | Change | Todo |
|---|---|---|
| `evaluation/r48_wan_common.py` | `load_cases(cases_path, anno_dir=None)`: `anno_dir or FIVE / "edit_prompt"` | `anno-dir-common` ✅ |
| `evaluation/r48_wan_draft.py` | `--anno_dir` (default None) → `C.load_cases(a.cases, a.anno_dir)` | `anno-dir-draft` ✅ |
| `evaluation/r48_wan_render.py` | `--anno_dir` (default None) → `C.load_cases` | `anno-dir-render48` ✅ |
| `evaluation/r49_wan_render.py` | `--anno_dir` (default None) → `C.load_cases` | `anno-dir-render49` ✅ |
| `evaluation/r58_build_cases.py` (new) | Regen sets from JSON comparison → 3 case files + `r58_regen_sets.csv`; count asserts 47 / 190 / 2 | `cases-script` ✅ |
| `slurm_scripts/five_bench/r58_wan_render.sh` (new) | `correct` (array 0-11, config → renderer + case file) and `repro` (array 0-1) modes; all with `--anno_dir evaluation/five_bench_correct --mode full` | `render-sbatch` ✅ |
| `slurm_scripts/five_bench/r58_wan_draft.sh` (new) | Clone of `r48_wan_draft.sh` on the 190 cases → `$D/r58_wan_draft` | `draft-sbatch` ✅ |
| `slurm_scripts/five_bench/r58_div.sh` (new) | Clone of `r48_div.sh` wan tasks (array 0-1) → `$D/r58_div` | `div-sbatch` ✅ |
| `slurm_scripts/five_bench/r58_field.sh` (new) | `r49_field.py --hosts wan` on r58 roots → `$D/r58_field` | `field-sbatch` ✅ |
| `slurm_scripts/five_bench/r58_wan_chain.sh` (new) | Clone of `r49_wan_full_chain.sh`: 5 submissions, afterok chain for max_s5 | `chain-driver` ✅ |
| `evaluation/r58_repro_check.py` (new) | Pixel comparison repro vs R49 render, PASS ≤ 2/255 → `r58_repro_check.csv` | `repro-check-script` ✅ |
| `evaluation/r58_assemble.py` (new) | Symlink tree `$D/r58_wan_correct/{config}` over 415 pairs + `r58_assembly.csv` (runs on Jean Zay) | `assemble-script` |
| `jz/r58_fetch_renders.sh` (new) | gpu-gw: fetch `r58_assembly.csv`, then `rsync -aL` per config → `R58_DEST/{config}`; free-space preflight; per-clip PNG count check; `--check`; `failures: N` | `fetch-renders-script` |
| `slurm_scripts/five_bench/r58_eval.sh` (new) | Lab launcher: `eval_suite.sh` per config given as argument, tag `r58`, queue-cap and duplicate guards, job ids → `logs/r58_eval_jobs.txt` | `eval-lab-launcher` |
| `jz/r50_port.sh` | SCRIPTS += the 5 `r58_*.sh` job scripts | `port-list` ✅ |
| `jz/r50_scratch_links.sh` | FOLDERS += `r58_wan_draft r58_div r58_field r58_wan_arms r58_repro r58_wan_correct` | `scratch-folders` ✅ |
| `jz/r50_fetch_outputs.sh` | ALL += `r58_wan_draft r58_div r58_field r58_wan_arms` (archive only) | `fetch-folders` |
| `evaluation/r58_score.py` (new) | Points (overall + per type), deltas with paired bootstrap, Pareto PDFs | `score-script` |

`r58_wan_render.sh` index → (config, renderer, cases):

```
0-5  native_s0.95 s0.92 s0.85 s0.75 s0.6 s0.4   r48_wan_render.py   r58_cases_base.json
6    native_s0.978                               r48_wan_render.py   r58_cases_base.json
7-10 matched59_s0.95 s0.92 s0.85 s0.978          r49_wan_render.py   r58_cases_base.json
11   max_s5 (--field_root $D/r58_field)          r49_wan_render.py   r58_cases_maxs5.json
repro 0 native_s0.92 / 1 matched59_s0.92         r48 / r49           r58_cases_repro.json -> $D/r58_repro
```
