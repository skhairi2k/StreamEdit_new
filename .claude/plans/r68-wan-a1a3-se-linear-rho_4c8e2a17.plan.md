---
name: "R68 — Wan-Edit A1–A3 + StreamEdit Linear-ρ Ablations"
overview: "Completes the FiVE-Bench-Correct ablation table beyond R62. (a) Wan-Edit A1 nodiv / A2 dino / A3 lpips at s_min 5 on all 415 Correct pairs (never rendered on any benchmark, so no reuse): field from R58's Correct inputs on the 190 changed pairs and from R49's official inputs on the 225 unchanged ones (the same split as A0 = R58 max_s5), A0 bit-exact gate on both parts, render on Jean Zay H100 as an array of 3 (one arm per task), fetched to lab /scratch. (b) StreamEdit B1 = R52's linear map rho = 2 + 48 q on R59's VP + Self-Forcing inputs (A0 = R59 se_max), field + render on the lab L40S. All 4 settings scored with the full R57 suite (core, niqe, motion, cf) on one batch r68.txt; r68_score.py writes the complete per-host ablation table (StreamEdit A0–A4 + B1 from R59/R62/R68, Wan-Edit A0–A4 from R58/R62/R68) with motion fidelity and PSC, plus trade-off figures."
task_id: R68
todos:
  - id: cases-script
    content: "evaluation/r68_build_cases.py -- writes evaluation/r68_cases_reuse225.json = the cases of evaluation/r43_anchor_manifest_correct.json NOT in evaluation/r58_cases_maxs5.json (same dict format, same order as the manifest); asserts 415 - 190 = 225, disjoint, union = manifest; for each of the 225 asserts that source_prompt, target_prompt, source_object, target_object in evaluation/five_bench_correct/edit{T}_FiVE.json equal the official FiVE JSON (the condition under which R49's inputs are valid); prints per-type counts"
    needed_by: [build-cases]
    status: completed
    completed_at: 2026-11-09
  - id: render-adapt
    content: "evaluation/r49_wan_render.py -- add 'nodiv_s5': 5, 'dino_s5': 5, 'lpips_s5': 5 to ADAPT (R68's A1–A3; field dir wan_{arm}_s5, same loop start as max_s5 / nosupp_s5); docstring line updated; no other change"
    needed_by: [wan-chain]
    status: completed
    completed_at: 2026-11-09
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r68_field.sh -- CPU job modelled on r62_field.sh, HOST arg. se: r51_field.py --host se --mapping linear --arms max --cases evaluation/r43_anchor_manifest_correct.json --attn_root $SC/r59_se_attn --div_root $SC/r59_div --gate_ref $SC/r59_field_full --out_root $SC/r68_field, SC = lab /scratch/PERSO/skhairi/five_bench (-> se_max_linrho; 'A0 gate PASS 415/415' + 'linear check PASS 415/415'). wan: two calls into the same --out_root $D/r68_field with --arms nodiv dino lpips: (1) --cases evaluation/r58_cases_maxs5.json --attn_root $D/r58_wan_draft/attn --div_root $D/r58_div --gate_ref $D/r58_field (190), (2) --cases evaluation/r68_cases_reuse225.json --attn_root $D/r48_wan_draft/attn --div_root $D/r48_div --gate_ref $D/r49_field_full (225); per-call input preflight (>= n cases npz in attn and div dirs, gate dir present); selftest first; post-run guard 415 npz per arm dir (+ 415 .npy for wan); 'failures: N'"
    needed_by: [se-chain, wan-chain]
    status: completed
    completed_at: 2026-11-09
  - id: se-render-sbatch
    content: "slurm_scripts/five_bench/r68_se_stage3.sh -- clone of r62_se_stage3.sh with no array: FIELD_DIR $SC/r68_field/se_max_linrho, METHOD r68_max_linrho_full, OUT_ROOT $SC/r68_se_arms, logs r68_se_stage3_%j, --cpus-per-task=4 (8 logical); refuses any OUT_ROOT other than $SC/r68_se_arms; Correct edit JSONs / cases / anchors, render flags, count and manifest guards and the r59_check_prompts.py guard unchanged"
    needed_by: [se-chain]
    status: completed
    completed_at: 2026-11-09
  - id: se-chain-driver
    content: "slurm_scripts/five_bench/r68_se_chain.sh -- lab driver: refuses if logs/r68_se_chain_jobs.txt exists or queue + 2 > ${R68_QUEUE_CAP:-30}; requires 415 real npz in $SC/r59_field_full/se_max, $SC/r59_se_attn, $SC/r59_div/se_{dino,lpips} (staged by stage-r59-inputs; no dependency) and refuses if $SC/r68_field/se_max_linrho or $SC/r68_se_arms/r68_max_linrho_full exists; submits field (r68_field.sh se) -> render (r68_se_stage3.sh, afterok field); writes '<stage> <id>' lines headed by the commit"
    needed_by: [se-chain]
    status: completed
    completed_at: 2026-11-09
  - id: wan-render-sbatch
    content: "slurm_scripts/five_bench/r68_wan_render.sh -- built from r62_wan_render.sh, --array=0-2 -> CONFIG nodiv_s5 / dino_s5 / lpips_s5, --time=20:00:00: preflight 415 .npy under $D/r68_field/wan_${CONFIG} per edit type; r49_wan_render.py --config $CONFIG --mode full --cases evaluation/r43_anchor_manifest_correct.json --anno_dir evaluation/five_bench_correct --field_root $D/r68_field --out_root $D/r68_wan_arms; refuses any OUT other than $D/r68_wan_arms; resumable; r48_units.py --missing post-run guard (415 complete clips); lab header with --exclude=node52 (ported by r50_port.sh); logs r68_wan_render_%A_%a"
    needed_by: [wan-chain]
    status: completed
    completed_at: 2026-11-09
  - id: wan-chain-driver
    content: "slurm_scripts/five_bench/r68_wan_chain.sh -- Jean Zay driver modelled on r62_wan_chain.sh: guards ($WORK, ported scripts, $SCRATCH links for r68_field / r68_wan_arms, evaluation/r68_cases_reuse225.json present, R58 repro-check CSV passing, the six input roots of r68_field.sh wan present with >= 190 / 225 / 415 files and gate dirs r58_field/wan_max_s5 (190 .npy) + r49_field_full/wan_max_s5 (>= 225 .npy) -- if any R48/R49 root was purged from $SCRATCH, FATAL with the /store path to push back from); no upstream dependency; submits field (r68_field.sh wan) -> render (--array=0-2, afterok field); --dry-run; logs/r68_wan_chain_jobs.txt, refuses if it exists"
    needed_by: [wan-chain]
    status: completed
    completed_at: 2026-11-09
  - id: jz-port-list
    content: "jz/r50_port.sh -- add r68_field, r68_wan_render, r68_wan_chain to SCRIPTS"
    needed_by: [jz-sync]
    status: completed
    completed_at: 2026-11-09
  - id: jz-scratch-list
    content: "jz/r50_scratch_links.sh -- add r68_field, r68_wan_arms to FOLDERS"
    needed_by: [jz-sync]
    status: completed
    completed_at: 2026-11-09
  - id: jz-fetch-list
    content: "jz/r50_fetch_outputs.sh -- add r68_field, r68_wan_arms to ALL"
    needed_by: [jz-archive]
    status: pending
  - id: fetch-renders-script
    content: "jz/r68_fetch_renders.sh -- clone of jz/r69_fetch_renders.sh for three configs (nodiv_s5 dino_s5 lpips_s5): rsync -rL --partial --size-only --no-perms --no-owner --no-group (no -t) zay:$SCRATCH/five_bench_out/r68_wan_arms/{cfg}_full/ -> $R68_DEST/{cfg}/ (default R68_DEST=/scratch/PERSO/skhairi/five_bench/r68_wan_correct; refuses any destination outside /scratch); preflight >= 30 GB free; post-check per config 415 clip dirs + r48_units.py --missing on r43_anchor_manifest_correct.json; --check counts only; never deletes on either side; ends '[r68_fetch] failures: N'"
    needed_by: [fetch-renders]
    status: pending
  - id: store-check-entry
    content: "evaluation/check_store_renders.py -- add r68 to TASKS and r68_wan_correct/ to JZ_ROOTS (store path /store/PERSO/skhairi/Outputs/jz_five_bench/r68_wan_correct); the StreamEdit root r68_se_arms maps to /store/PERSO/skhairi/Outputs/five_bench/r68_se_arms like r62_se_arms"
    needed_by: [store-copy]
    status: pending
  - id: eval-batch
    content: "evaluation/eval_suite/batches/r68.txt -- 4 lines `NAME ROOT` (R57 batch format, ROOT relative to --root_base /scratch/PERSO/skhairi/five_bench): r68_max_linrho_full r68_se_arms/r68_max_linrho_full/step14; wan_nodiv_s5_correct r68_wan_correct/nodiv_s5; wan_dino_s5_correct r68_wan_correct/dino_s5; wan_lpips_s5_correct r68_wan_correct/lpips_s5"
    needed_by: [eval-core, eval-niqe, eval-motion, eval-cf]
    status: pending
  - id: score-script
    content: "evaluation/r68_score.py -- --host se|wan; reads only the R57 tables (suite_<tag>_<NAME>.csv / _per_type.csv / _overall.csv), never recomputes a mean. se rows: A0 suite_r59_r59_max_full, A1–A4 suite_r62_r62_{nodiv,dino,lpips,nosupp}_full, B1 suite_r68_r68_max_linrho_full, refs native suite_r59_r59_n15_rho2_vp + equal-NFE suite_r59_r59_n20_rho2_vp, curve suite_r59_r59_n20_rho*_vp. wan rows: A0 suite_r58_wan_max_s5_correct, A1–A3 suite_r68_wan_{nodiv,dino,lpips}_s5_correct, A4 suite_r62_wan_nosupp_s5_correct, refs suite_r58_wan_native_s0p92_correct + suite_r58_wan_matched59_s0p92_correct, curves suite_r58_wan_native_* and suite_r58_wan_matched59_* (+ suite_r69_* if present). Writes evaluation/csv/r68_ablation_{se,wan}.csv (overall + per edit type, all suite columns incl. motion_fidelity_score, psc, cf), r68_deltas_{se,wan}.csv (arm - A0), and motion fidelity at matched FiVE-Acc (equal-NFE curve interpolated at the arm's FiVE-Acc, blank if outside the curve range, never extrapolated); figures evaluation/figures/r68_{host}_{x}_vs_{y}.pdf, x in {five_acc_yes_no, clip_d_prompt, editreward, psc}, y in {lpips_unedit_part, cf, motion_fidelity_score}; exit 1 on any per-clip table with != 415 rows or a missing input"
    needed_by: [score]
    status: pending
steps:
  - id: build-cases
    type: local
    command: python evaluation/r68_build_cases.py
    status: completed
    completed_at: 2026-11-09
  - id: stage-r59-inputs
    type: manual
    check_hint: "User, on gpu-gw in tmux (R59's inputs are on /store since 2026-11-02, which no compute node mounts): copy se_max, the attention and the two SE divergences to lab /scratch without -t (the 30-day purge counts from the copy date); each of the 4 dirs then holds 415 real npz (commands below)"
    status: pending
  - id: se-chain
    type: local
    wait_for: stage-r59-inputs
    command: bash slurm_scripts/five_bench/r68_se_chain.sh
    sets_status: running
    status: pending
  - id: jz-sync
    type: manual
    wait_for: build-cases
    check_hint: "User: commit + push (incl. evaluation/r68_cases_reuse225.json). On Jean Zay in the clone: git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh -> '[r50_port] failures: 0', '[r50_scratch] failures: 0'; ls -ld $D/r68_field $D/r68_wan_arms are symlinks into $SCRATCH"
    status: pending
  - id: wan-chain
    type: manual
    wait_for: jz-sync
    check_hint: "User, on Jean Zay: bash slurm_scripts/five_bench/r68_wan_chain.sh --dry-run (every guard passes; if an R48/R49 input root was purged, push it back from /store first) && bash slurm_scripts/five_bench/r68_wan_chain.sh; paste logs/r68_wan_chain_jobs.txt"
    status: pending
  - id: wait-se-renders
    type: manual
    wait_for: se-chain
    check_hint: "field log: 'A0 gate PASS 415/415', 'linear check PASS 415/415', 'failures: 0', 415 npz in se_max_linrho; render log 'failures: 0', 415 clip dirs, all-ok manifest, '[r59_check_prompts] OK'"
    status: pending
  - id: wait-wan-chain
    type: manual
    wait_for: wan-chain
    check_hint: "User, on Jean Zay: field log 'A0 gate PASS 190/190' and 'A0 gate PASS 225/225', 3 x 415 npz + 415 .npy; 3 render tasks '415/415 clips failures: 0' (a timed-out task is resubmitted alone with --array=<i>: resumable); sacct COMPLETED on all ids"
    status: pending
  - id: fetch-renders
    type: manual
    wait_for: wait-wan-chain
    check_hint: "User, on gpu-gw in tmux: bash jz/r68_fetch_renders.sh --check, then bash jz/r68_fetch_renders.sh -> '[r68_fetch] failures: 0', 3 x 415 complete clips under /scratch/PERSO/skhairi/five_bench/r68_wan_correct"
    status: pending
  - id: jz-archive
    type: manual
    wait_for: wait-wan-chain
    check_hint: "User, on gpu-gw in tmux: bash jz/r50_fetch_outputs.sh --check r68_field r68_wan_arms, then bash jz/r50_fetch_outputs.sh r68_field r68_wan_arms -> '[r50_fetch] failures: 0' for both; do NOT run the printed Jean Zay delete commands before store-copy passes"
    status: pending
  - id: store-copy
    type: manual
    wait_for: [wait-se-renders, fetch-renders, jz-archive]
    check_hint: "User, on gpu-gw in tmux: archive the StreamEdit field + render (written to lab /scratch by the jobs) and the fetched Wan trees on /store (commands below); then check_store_renders.py r68 -> '[check_store] failures: 0' (4 settings x 415 clips, PNG counts equal /scratch)"
    sets_status: finished
    status: pending
  - id: eval-core
    type: local
    wait_for: store-copy
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part core --per_job 2
    status: pending
  - id: eval-niqe
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part niqe --per_job 4
    status: pending
  - id: eval-motion
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part motion --per_job 12
    status: pending
  - id: eval-cf
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part cf --per_job 5
    status: pending
  - id: wait-eval
    type: manual
    wait_for: [eval-core, eval-niqe, eval-motion, eval-cf]
    check_hint: "nothing of tag r68 left in squeue; every logs/eval_suite_{metrics,cf}_<id>.out ends 'failures: 0'; 4 sets suite_r68_<NAME>.csv + _per_type.csv + _overall.csv, 415 rows each, niqe_target_image / motion_fidelity_score / cf filled; NaN cells from the merge logs recorded in the Run log; R62's wait-eval also passed (r68_score.py reads suite_r62_*)"
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
    command: "python evaluation/r68_score.py --host se && python evaluation/r68_score.py --host wan"
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "read r68_ablation_{se,wan}.csv, r68_deltas_{se,wan}.csv (incl. motion fidelity raw + at matched FiVE-Acc, and PSC) and the figures; compare Wan A1–A3 signs with StreamEdit's R62 A1–A3, and B1 - A0 with R52 on official FiVE; write the Run-log verdict"
    sets_status: analyzed
    status: pending
isProject: true
---

# R68: Wan-Edit A1–A3 + StreamEdit Linear-ρ Ablations

## Context

R62 reports R51's demand ablation on FiVE-Bench-Correct, but on Wan-Edit it covers only A4 (no support), and R52's mapping ablation (B1) exists only on official FiVE. R68 closes both gaps so the ablation table is complete on Correct.

- **(a) Wan-Edit A1–A3:** A1 q = S, A2 q = S·m_DINO, A3 q = S·m_LPIPS at s_min 5. Only q changes. Drafts, attention, divergence, the exposure-inverse map and the render are those of R58's `max_s5` (A0). These arms were never rendered on any benchmark, so all 415 pairs are new renders.
- **(b) StreamEdit B1:** the R59 `se_max` demand (VP + Self-Forcing) mapped by ρ = 2 + 48q instead of the exposure inverse. A0 is R59 `se_max`.

Done when all of the following hold:
- the 3 Wan arms × 415 pairs are rendered on Jean Zay H100 and the B1 arm × 415 pairs on the lab L40S;
- all 4 settings are on lab `/scratch` and `/store`, and each is scored with the full R57 suite on the lab;
- `r68_ablation_{se,wan}.csv`, `r68_deltas_{se,wan}.csv` and the figures are read, with the verdict in the Run log.

## Execution steps

| # | Step id | Type | Runs on | Waits for | Sets status |
|---|---|---|---|---|---|
| 0 | *(prep)* `cases-script`, `render-adapt`, `field-sbatch`, `se-render-sbatch`, `se-chain-driver`, `wan-render-sbatch`, `wan-chain-driver`, `jz-port-list`, `jz-scratch-list`, `jz-fetch-list`, `fetch-renders-script`, `store-check-entry`, `eval-batch`, `score-script` | /build-step | lab (code) | — | `implemented` |
| 1 | `build-cases` | local | lab | — | — |
| 1b | `stage-r59-inputs` | manual (user, tmux) | gpu-gw | — | — |
| 2 | `se-chain` | local (field → render afterok) | lab L40S | stage-r59-inputs | `running` |
| 3 | `jz-sync` | manual | **Jean Zay** | build-cases | — |
| 4 | `wan-chain` | manual (field → render array 0-2 afterok) | **Jean Zay** | jz-sync | — |
| 5 | `wait-se-renders` | manual | lab | se-chain | — |
| 6 | `wait-wan-chain` | manual | **Jean Zay** | wan-chain | — |
| 7 | `fetch-renders` | manual (user, tmux) | gpu-gw | wait-wan-chain | — |
| 8 | `jz-archive` | manual (user, tmux) | gpu-gw | wait-wan-chain | — |
| 9 | `store-copy` | manual (user, tmux) | gpu-gw | wait-se-renders, fetch-renders, jz-archive | `finished` |
| 10 | `eval-core` | local (`--part core`) | lab L40S | store-copy | — |
| 11 | `eval-niqe` | local (`--part niqe`) | lab L40S | eval-core | — |
| 12 | `eval-motion` | local (`--part motion`) | lab L40S | eval-core | — |
| 13 | `eval-cf` | local (`--part cf`) | lab H100 | eval-core | — |
| 14 | `wait-eval` | manual | lab | eval-core, eval-niqe, eval-motion, eval-cf | — |
| 15 | `score` | local | lab | wait-eval | — |
| 16 | `verdict` | manual | lab | score | `analyzed` |

```
/build-step build-cases R68
/build-step se-chain R68
/build-step jz-sync R68
/build-step wan-chain R68
/build-step fetch-renders R68
/build-step jz-archive R68
/build-step store-copy R68
/build-step eval-core R68
/build-step score R68
/run-step R68 build-cases
/run-step R68 stage-r59-inputs
/run-step R68 se-chain
/run-step R68 jz-sync
/run-step R68 wan-chain
/run-step R68 wait-se-renders
/run-step R68 wait-wan-chain
/run-step R68 fetch-renders
/run-step R68 jz-archive
/run-step R68 store-copy
/run-step R68 eval-core
/run-step R68 eval-niqe
/run-step R68 eval-motion
/run-step R68 eval-cf
/run-step R68 wait-eval
/run-step R68 score
/run-step R68 verdict
```

## Decisions

| Item | Choice |
|---|---|
| Benchmark | FiVE-Bench-Correct, 415 pairs, `evaluation/r43_anchor_manifest_correct.json`, annotations `evaluation/five_bench_correct/`; no new prompts |
| Wan arms | A1 `nodiv_s5` (q = S), A2 `dino_s5` (q = S·m_dino), A3 `lpips_s5` (q = S·m_lpips), map start = 50 − round(50q), s ≥ 5; A0 = R58 `max_s5` (`r58_wan_correct/max_s5`), A4 = R62 `nosupp_s5`; not re-rendered |
| Wan inputs (user, 2026-11-09) | Split 190 + 225, the same as A0's own assembly. The 190 pairs of `r58_cases_maxs5.json` use R58's Correct roots (`r58_wan_draft/attn`, `r58_div`, gate `r58_field`). The other 225 (`r68_cases_reuse225.json`, Correct inputs = official) use R49's roots (`r48_wan_draft/attn`, `r48_div`, gate `r49_field_full`). Both parts go through the bit-exact A0 gate. No new draft or divergence |
| Wan render (user, 2026-11-09) | All 415 pairs per arm are new renders, with no reuse. `r49_wan_render.py --mode full`, N = 50, loop start 5, per-token start map, seed 0. One array of 3 tasks (one arm each), 20 h, resumable. Estimate ~10–11 h H100 per arm (R62: 190 pairs ≈ 4–5 h) |
| StreamEdit B1 | `r51_field.py --mapping linear --arms max` on R59's inputs (`r59_se_attn`, `r59_div`, gate `r59_field_full`), giving `se_max_linrho`. The render flags are identical to `r59_se_stage3.sh`: VP + `anchors_r43_correct`, step 15, flow_shift 1.0, fg_boost 4, blend_power 2, chunk 21, seed 0, dump step 14. NFE 5 + 15 = 20, the same as A0 |
| Out of scope (user, 2026-11-09) | Wan B1: its inverse is already linear in the step knob, so a Wan B1 would equal A0. B1 on text-only (R60) and on LongLive (R61) inputs. Qualitative grids. Rescoring R62 |
| No overwrite | New outputs only under lab `/scratch/.../r68_{field,se_arms,wan_correct}`, Jean Zay `$D/r68_{field,wan_arms}` and their `/store` archives. R49 / R51 / R58 / R59 / R62 folders are read only |
| Metrics | Full R57 suite (user, 2026-11-09): core, niqe, motion and cf, all on one batch `r68.txt` (4 settings) with tag `r68`. Common arguments `--layout edit_video --src_window render`. `--per_job` is 2 (core), 4 (niqe), 12 (motion) and 5 (cf), as in R62 |
| Table (user, 2026-11-09) | `r68_score.py` writes the complete table per host. StreamEdit: A0 (R59), A1–A4 (R62), B1 (R68). Wan-Edit: A0 (R58), A1–A3 (R68), A4 (R62). Native and equal-NFE references come from R59 / R58, plus R69's `matched59_s0.5` if it has been scored |
| Aggregation | Plain per-clip mean read from the R57 tables, never `_avg.csv`, overall and per edit type. Figures put alignment on x and preservation on y. Motion fidelity is reported raw and at matched FiVE-Acc (interpolated on the equal-NFE curve, never extrapolated), together with PSC. No bootstrap, as in R51 / R62 |
| Platforms | StreamEdit renders on the lab L40S (`streamgve`). Wan-Edit renders on Jean Zay H100 (`five-bench` overlay). All scoring runs on the lab, reading `/scratch` copies only; `/store` is the archive |
| Slurm | Lab: `L40S` / `CPU`, `--exclude=node52`, GPU jobs `--cpus-per-task=4` + `--hint=nomultithread` (8 logical); the lab chain submits 2 jobs, checked against the 30-task cap. Jean Zay headers are rewritten by `jz/r50_port.sh` |
| Wan input availability | Every Wan field input is still on Jean Zay `$SCRATCH`, not purged (user, 2026-11-09). Each is also archived in `/store/PERSO/skhairi/Outputs/jz_five_bench/`: `r48_wan_draft`, `r48_div` and `r49_field_full` since 2026-10-25, `r58_wan_draft`, `r58_div` and `r58_field` since 2026-11-04 (Run log, `failures: 0`). `r68_wan_chain.sh` still checks the roots and, if one is missing, names its `/store` copy |
| Lab storage (2026-11-09) | R58 storage policy: lab jobs read and write `/scratch/PERSO/skhairi/five_bench` only. R59's inputs (`r59_field_full/se_max`, `r59_se_attn`, `r59_div/se_*`) have been on `/store` since 2026-11-02 (`~/Data` holds only links), so `stage-r59-inputs` copies them to `/scratch`. R68's StreamEdit field and render are written straight to `/scratch` (read by the evaluation) and archived on `/store` by `store-copy`. Anchors and FiVE source frames are real `~/Data` directories |
| Outputs | lab `/scratch/PERSO/skhairi/five_bench/r68_field/se_max_linrho/`, `/scratch/PERSO/skhairi/five_bench/r68_se_arms/r68_max_linrho_full/step14/`, Jean Zay `$D/r68_field/wan_{nodiv,dino,lpips}_s5/`, `$D/r68_wan_arms/{arm}_s5_full/`, `/scratch/PERSO/skhairi/five_bench/{r68_se_arms, r68_wan_correct/{arm}_s5}/`, `/store/PERSO/skhairi/Outputs/{five_bench/r68_se_arms, jz_five_bench/r68_wan_correct}/`, `evaluation/r68_cases_reuse225.json`, `evaluation/csv/suite_r68_*.csv`, `evaluation/csv/r68_{ablation,deltas}_{se,wan}.csv`, `evaluation/figures/r68_*.pdf` |
| Compute / storage | StreamEdit: 1 render ≈ 12 h L40S, ~15 GB. Wan-Edit: 3 × ~11 h H100, ~3 × 9 GB. Evaluation: 4 settings × 4 parts |

## Step commands

### build-cases
```bash
python evaluation/r68_build_cases.py
python -c "import json; print(len(json.load(open('evaluation/r68_cases_reuse225.json'))))"   # 225
```

### stage-r59-inputs
```bash
# on gpu-gw (tmux): /store -> lab /scratch, real files, no -t (mtime = copy date for the 30-day purge)
ST=/store/PERSO/skhairi/Outputs/five_bench; SC=/scratch/PERSO/skhairi/five_bench
[ -d $ST ] || { echo '/store not mounted here'; exit 1; }
mkdir -p $SC/r59_field_full $SC/r59_div
rsync -rL --size-only $ST/r59_field_full/se_max $SC/r59_field_full/
rsync -rL --size-only $ST/r59_se_attn/ $SC/r59_se_attn/
rsync -rL --size-only $ST/r59_div/se_dino $ST/r59_div/se_lpips $SC/r59_div/
for X in r59_field_full/se_max r59_se_attn r59_div/se_dino r59_div/se_lpips; do echo "$X $(find $SC/$X -path '*/edit*/*.npz' -type f | wc -l)"; done   # 415 each
```

### se-chain
```bash
bash slurm_scripts/five_bench/r68_se_chain.sh
cat logs/r68_se_chain_jobs.txt
```

### jz-sync
```bash
# lab: commit + push (incl. evaluation/r68_cases_reuse225.json); then on Jean Zay (ssh zay from gpu-gw), in the clone:
git pull
bash jz/r50_port.sh
bash jz/r50_scratch_links.sh
ls -ld ~/Data/dataggen/outputs/five_bench/{r68_field,r68_wan_arms}
```

### wan-chain
```bash
# on Jean Zay, in the clone
bash slurm_scripts/five_bench/r68_wan_chain.sh --dry-run
bash slurm_scripts/five_bench/r68_wan_chain.sh
cat logs/r68_wan_chain_jobs.txt
```

### wait-se-renders
```bash
squeue -u $USER -h -o '%i %j %T' | grep r68
grep -hE 'A0 gate|linear check' logs/r68_field_*.out
grep -h 'r59_check_prompts' logs/r68_se_stage3_*.out
grep -hE 'failures:' logs/r68_field_*.out logs/r68_se_stage3_*.out | sort | uniq -c
ls -d /scratch/PERSO/skhairi/five_bench/r68_se_arms/r68_max_linrho_full/step14/edit*/*/ | wc -l   # 415
```

### wait-wan-chain
```bash
# on Jean Zay
sacct -j "$(grep -v '^#' logs/r68_wan_chain_jobs.txt | awk '{print $2}' | paste -sd,)" --format=JobID,State,Elapsed
grep -h 'A0 gate' logs/r68_field_*.out
grep -hE 'failures:' logs/r68_field_*.out logs/r68_wan_render_*.out | sort | uniq -c
```

### fetch-renders
```bash
# on gpu-gw (tmux)
bash jz/r68_fetch_renders.sh --check
bash jz/r68_fetch_renders.sh
```

### jz-archive
```bash
# on gpu-gw (tmux)
bash jz/r50_fetch_outputs.sh --check r68_field r68_wan_arms
bash jz/r50_fetch_outputs.sh r68_field r68_wan_arms
```

### store-copy
```bash
# on gpu-gw (tmux)
D=~/Data/dataggen/outputs/five_bench; SC=/scratch/PERSO/skhairi/five_bench; ST=/store/PERSO/skhairi/Outputs
[ -d $ST ] || { echo '/store not mounted here'; exit 1; }
mkdir -p $ST/five_bench/r68_se_arms $ST/five_bench/r68_field && rsync -a --size-only $SC/r68_se_arms/ $ST/five_bench/r68_se_arms/   # SE render already on /scratch
rsync -a --size-only $SC/r68_field/ $ST/five_bench/r68_field/                                                                        # SE field (se_max_linrho)
mkdir -p $ST/jz_five_bench/r68_wan_correct && rsync -a --size-only $SC/r68_wan_correct/ $ST/jz_five_bench/r68_wan_correct/
~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py r68
```

### eval-core
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part core --per_job 2
```

### eval-niqe
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part niqe --per_job 4
```

### eval-motion
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part motion --per_job 12
```

### eval-cf
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r68 --layout edit_video --src_window render --part cf --per_job 5
```

### wait-eval
```bash
squeue -u $USER -h -o '%i %k' | grep 'eval_suite|r68|'                    # nothing left
ls evaluation/csv/suite_r68_*_overall.csv | wc -l                         # 4
ls evaluation/csv/suite_r68_*.csv | grep -v -e _per_type -e _overall | xargs wc -l   # 416 lines each
ls evaluation/csv/suite_r62_*_overall.csv | wc -l                         # 5 (R62 inputs of the table)
```

### score
```bash
python evaluation/r68_score.py --host se
python evaluation/r68_score.py --host wan
```

### verdict
```bash
column -s, -t < evaluation/csv/r68_ablation_se.csv | less -S
column -s, -t < evaluation/csv/r68_deltas_se.csv | less -S
column -s, -t < evaluation/csv/r68_ablation_wan.csv | less -S
column -s, -t < evaluation/csv/r68_deltas_wan.csv | less -S
ls evaluation/figures/r68_*.pdf
```

## Pipeline

```mermaid
flowchart LR
  subgraph lab["Lab L40S (StreamEdit B1)"]
    ST59[/store r59 inputs] --> SG[stage-r59-inputs<br/>-> lab /scratch] --> F1[r68_field.sh se<br/>linear map, se_max_linrho]
    F1 --> R1[r68_se_stage3.sh]
  end
  subgraph jz["Jean Zay H100 (Wan A1-A3)"]
    BC[r68_build_cases.py<br/>225 unchanged pairs] --> F2
    R58I[R58 inputs, 190 pairs] --> F2[r68_field.sh wan<br/>2 A0 gates]
    R49I[R48/R49 inputs, 225 pairs] --> F2
    F2 --> R2[r68_wan_render.sh<br/>array 0-2, 415 each]
  end
  R2 --> FX[jz/r68_fetch_renders.sh] --> SC[lab /scratch + /store<br/>check_store_renders r68]
  R1 --> SC
  SC --> E[eval_suite.sh r68.txt<br/>core, niqe, motion, cf]
  E --> S[r68_score.py --host se / wan]
  P[suite_r58 / r59 / r62 / r69 tables] --> S
  S --> O[r68_ablation + r68_deltas CSVs<br/>r68 figures]
```

## Code to touch

| File | Change | Todo |
|---|---|---|
| `evaluation/r68_build_cases.py` (new) | Complement of `r58_cases_maxs5.json` in the Correct manifest → `evaluation/r68_cases_reuse225.json`. Asserts 225 pairs, disjointness and that the 4 prompt/object fields equal the official JSON | `cases-script` |
| `evaluation/r49_wan_render.py` | `ADAPT` += `nodiv_s5`, `dino_s5`, `lpips_s5` (loop start 5) | `render-adapt` |
| `slurm_scripts/five_bench/r68_field.sh` (new) | From `r62_field.sh`. `se`: one linear-map call writing `max`, reading and writing lab `/scratch`. `wan`: two calls (190 with R58 roots, 225 with R48/R49 roots) into the same Jean Zay `$D/r68_field`; post-run guard of 415 per arm dir | `field-sbatch` |
| `slurm_scripts/five_bench/r68_se_stage3.sh` (new) | Clone of `r62_se_stage3.sh` with no array: field `se_max_linrho`, method `r68_max_linrho_full`, out `/scratch/.../r68_se_arms`; 4 CPUs | `se-render-sbatch` |
| `slurm_scripts/five_bench/r68_se_chain.sh` (new) | field → render afterok. Requires R59's staged inputs on `/scratch` (415 real npz each) and refuses if R68's outputs exist; queue-cap check; jobs file with the commit at its head, refuses if the file exists | `se-chain-driver` |
| `slurm_scripts/five_bench/r68_wan_render.sh` (new) | From `r62_wan_render.sh`: `--array=0-2` → config; all 415 Correct cases; out `$D/r68_wan_arms`; `r48_units.py --missing` guard | `wan-render-sbatch` |
| `slurm_scripts/five_bench/r68_wan_chain.sh` (new) | From `r62_wan_chain.sh`: input-root and purge guards, R58 repro-check guard, field → render array afterok, `--dry-run` | `wan-chain-driver` |
| `jz/r50_port.sh` | `SCRIPTS` += `r68_field`, `r68_wan_render`, `r68_wan_chain` | `jz-port-list` |
| `jz/r50_scratch_links.sh` | `FOLDERS` += `r68_field`, `r68_wan_arms` | `jz-scratch-list` |
| `jz/r50_fetch_outputs.sh` | `ALL` += `r68_field`, `r68_wan_arms` | `jz-fetch-list` |
| `jz/r68_fetch_renders.sh` (new) | Clone of `jz/r69_fetch_renders.sh` over 3 configs → `$R68_DEST/{cfg}`; ≥ 30 GB free; `--check` | `fetch-renders-script` |
| `evaluation/check_store_renders.py` | `TASKS` += `r68`; `JZ_ROOTS` += `r68_wan_correct/` | `store-check-entry` |
| `evaluation/eval_suite/batches/r68.txt` (new) | 4 `NAME ROOT` lines (B1 + 3 Wan arms) | `eval-batch` |
| `evaluation/r68_score.py` (new) | Full ablation table per host from the R57 tables of R58 / R59 / R62 / R68 (+ R69). Deltas against A0, motion at matched FiVE-Acc, figures. Reuses the `r49_score` helpers (`point`, `plot`) where they fit; enforces 415 rows | `score-script` |
