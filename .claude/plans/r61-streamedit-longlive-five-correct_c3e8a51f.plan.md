---
name: "R61 — StreamEdit LongLive on FiVE-Correct"
overview: "StreamEdit VP + LongLive on all 415 FiVE-Bench-Correct pairs: the full R45-style rho sweeps at N=15 (native; rho=2 = official LongLive) and N=20 (equal NFE), plus max adaptive (se_max_ll, R55's method: N_d=5 tgt0.9 draft + recorder -> DINO/LPIPS divergence -> R49 field -> 15-step render). Correct prompts via a shared --edit_json_dir on r31_stage1/r31_stage3, Correct anchors (anchors_r43_correct). Scored only through R57's unified suite; trade-off figures place se_max_ll on the N=15 and N=20 Correct curves."
task_id: R61
todos:
  - id: stage1-json-dir
    content: "evaluation/r31_stage1.py -- add --edit_json_dir (default data_root/edit_prompt), used where edit{T}_FiVE.json is read (line ~297), as run_fivebench.py already does; log the directory in the run header"
    needed_by: [smoke, se-chain]
    status: pending
  - id: stage3-json-dir
    content: "evaluation/r31_stage3.py -- add --edit_json_dir (default data_root/edit_prompt), used where edit{T}_FiVE.json is read (line ~203); log the directory in the run header"
    needed_by: [smoke, se-chain]
    status: pending
  - id: smoke-cases-json
    content: "evaluation/r61_smoke_cases.json -- [edit1/0001_bus (anchor changed by the keep-pose suffix), edit6/0042_gym-ball (target prompt changed, removal by negation fixed)] in the r43_anchor_manifest_correct.json entry shape, anchors from anchors_r43_correct"
    needed_by: [smoke]
    status: pending
  - id: smoke-sbatch
    content: "slurm_scripts/five_bench/r61_smoke.sh -- L40S, 3 h, --nodelist=node39 (r45_sweep is a /store link; fail-fast preflight that it is readable); G1 benchmark freeze: per-type counts of evaluation/five_bench_correct == (98 100 99 99 9 10), json pairs == r43_anchor_manifest_correct.json pairs, every anchor file exists; G2 sweep path: run_fivebench.py --backbone ll rho=2 N=15 with --edit_json_dir evaluation/five_bench_correct, anchors_r43_correct, --cases_json r61_smoke_cases.json, both renders differ from r45_sweep/n15/ll_rho2_n15_vp and the log shows the Correct target prompt of 0042_gym-ball; G3 se_max chain on the 2 pairs: r31_stage1 (Correct) -> r31_divergence dino + lpips -> r49_field.py -> r31_stage3 (Correct), stage-1 and stage-3 logs show the Correct target prompt, q finite in [0, 1], render frame count == G2's; G4 default unchanged: r31_stage1 on 0001_bus e1 without --edit_json_dir, official anchors, latents np.array_equal to $D/r55_smoke/on; prints each gate PASS/FAIL and 'failures: N'; out $D/r61_smoke/"
    needed_by: [smoke]
    status: pending
  - id: sweep-sbatch
    content: "slurm_scripts/five_bench/r61_sweep.sh -- clone of r45_sweep.sh: array 0-7 over RHOS (2 3 4 6 8 10 20 50), first argument N in {15, 20} (no r45_select.py lookup), run_fivebench.py --backbone ll --edit_json_dir evaluation/five_bench_correct --first_frame_edit_dir $D/anchors_r43_correct, fg_boost 2, flow_shift 1.0, seed 0; METHOD r61_ll_rho{R}_n{N}, OUT_ROOT $D/r61_sweep/n{N}; EXPECTED (98 100 99 99 9 10), total 415; same skip/resume, count and manifest guards; refuses if git status shows uncommitted changes under evaluation/five_bench_correct or r43_anchor_manifest_correct.json, logs the commit"
    needed_by: [sweep-n15, sweep-n20]
    status: pending
  - id: stage1-sbatch
    content: "slurm_scripts/five_bench/r61_ll_stage1.sh -- clone of r55_ll_stage1.sh: --cases_json evaluation/r43_anchor_manifest_correct.json, --edit_json_dir evaluation/five_bench_correct, ANCHOR_ROOT $D/anchors_r43_correct; out $D/r61_ll_stage1 (frames + latents) and $D/r61_ll_attn; EXPECT (98 100 99 99 9 10); 415 attention + 415 latent npz guards"
    needed_by: [se-chain]
    status: pending
  - id: div-sbatch
    content: "slurm_scripts/five_bench/r61_div.sh -- clone of r55_div.sh: array 0-1 (dino, lpips), --cases evaluation/r43_anchor_manifest_correct.json, target $D/r61_ll_stage1/frames/n5_tgt09/step04, latents $D/r61_ll_stage1/latents, out $D/r61_div/se_{metric}; 415-npz guard"
    needed_by: [se-chain]
    status: pending
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r61_field.sh -- clone of r55_field.sh: r49_field.py --selftest, then --hosts se --cases evaluation/r43_anchor_manifest_correct.json --attn_se $D/r61_ll_attn --div_root $D/r61_div --out_root $D/r61_field; 415-npz guard on se_max"
    needed_by: [se-chain]
    status: pending
  - id: stage3-sbatch
    content: "slurm_scripts/five_bench/r61_ll_stage3.sh -- clone of r55_ll_stage3.sh: --edit_json_dir evaluation/five_bench_correct, ANCHOR_ROOT $D/anchors_r43_correct, FIELD_DIR $D/r61_field/se_max, METHOD r61_max_ll, OUT_ROOT $D/r61_ll_arms; EXPECT (98 100 99 99 9 10); per-type count + manifest guards"
    needed_by: [se-chain]
    status: pending
  - id: chain-driver
    content: "slurm_scripts/five_bench/r61_chain.sh -- lab driver: refuses if logs/r61_chain_jobs.txt exists or the Correct files are uncommitted; prints the queue count; submits stage1 -> div (--array=0-1, afterok) -> field (afterok) -> stage3 (afterok); writes '<stage> <id>' lines headed by the commit to logs/r61_chain_jobs.txt"
    needed_by: [se-chain]
    status: pending
  - id: eval-driver
    content: "slurm_scripts/five_bench/r61_eval_all.sh -- argument arms|sweeps; arms = r61_ll_rho2_n15, r61_ll_rho2_n20, r61_max_ll; sweeps = the other 14 sweep methods; per method calls R57's `bash slurm_scripts/five_bench/eval_suite.sh --method M --root ROOT --layout edit_video --edit_json_dir evaluation/five_bench_correct --tag r61`; refuses if eval_suite.sh is missing or a render root has != 415 frame dirs; appends '<method> <metrics_id> <cf_id> <merge_id>' to logs/r61_eval_jobs.txt"
    needed_by: [eval-arms, eval-sweeps]
    status: pending
  - id: score-script
    content: "evaluation/r61_score.py -- reads evaluation/csv/suite_r61_{method}.csv for the 17 methods (exit 1 on a missing file, != 415 rows or NaN); plain per-clip means over 415; writes r61_points.csv (native, equal-NFE, se_max_ll x every suite metric), r61_sweep_n15.csv, r61_sweep_n20.csv, r61_per_type.csv; figures r61_ll_{five_acc,clip_d,clip_tgt,editreward,psc}.pdf with x = alignment metric, y = bg LPIPS, N=15 and N=20 curves, native / equal-NFE / se_max_ll marked"
    needed_by: [score]
    status: pending
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r61_smoke.sh
    status: pending
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "logs/r61_smoke_{job_id}.out: G1-G4 PASS, 'failures: 0', no Traceback in .err"
    status: pending
  - id: sweep-n15
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=0-7 slurm_scripts/five_bench/r61_sweep.sh 15
    sets_status: running
    status: pending
  - id: sweep-n20
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=0-7 slurm_scripts/five_bench/r61_sweep.sh 20
    status: pending
  - id: se-chain
    type: local
    wait_for: wait-smoke
    command: bash slurm_scripts/five_bench/r61_chain.sh
    status: pending
  - id: wait-renders
    type: manual
    wait_for: [sweep-n15, sweep-n20, se-chain]
    check_hint: "no r61 job in squeue; every r61_sweep / r61_ll_stage* / r61_div / r61_field log 'failures: 0'; 16 sweep methods x 415 frame dirs with all-ok manifests; 415 attention + latent npz, 2 x 415 divergence npz, 415 field npz, 415 r61_max_ll step14 renders"
    sets_status: finished
    status: pending
  - id: r57-check
    type: manual
    check_hint: "R57 is `analyzed` on the workboard (suite adopted after both smokes) and slurm_scripts/five_bench/eval_suite.sh exists"
    status: pending
  - id: eval-arms
    type: local
    wait_for: [wait-renders, r57-check]
    command: bash slurm_scripts/five_bench/r61_eval_all.sh arms
    status: pending
  - id: eval-sweeps
    type: local
    wait_for: eval-arms
    command: bash slurm_scripts/five_bench/r61_eval_all.sh sweeps
    status: pending
  - id: wait-eval
    type: manual
    wait_for: eval-sweeps
    check_hint: "every job in logs/r61_eval_jobs.txt COMPLETED (sacct); 0 'Error:' in the eval_suite logs; 17 evaluation/csv/suite_r61_*.csv (not _per_type) with 415 rows, no NaN"
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
    command: python evaluation/r61_score.py
    output_paths:
      - evaluation/csv/r61_points.csv
      - evaluation/csv/r61_sweep_n15.csv
      - evaluation/csv/r61_sweep_n20.csv
      - evaluation/csv/r61_per_type.csv
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "read r61_points.csv, r61_per_type.csv and the r61_ll_*.pdf figures: where se_max_ll sits relative to the N=20 curve (equal NFE) and to native rho=2 N=15; compare with R55's official-FiVE reading if R55 has run"
    sets_status: analyzed
    status: pending
isProject: true
---

# R61: StreamEdit LongLive on FiVE-Correct

## Context

FiVE-Bench-Correct is now the main-paper benchmark (official FiVE moves to the supplementary), so every StreamEdit arm has to be rendered and scored on Correct. R61 covers the LongLive VP host, with three arms: native baseline (official LongLive, ρ = 2, N = 15), equal-NFE baseline (ρ = 2, N = 20) and max adaptive (se_max_ll, R55's method, NFE 5 + 15 = 20). It also re-runs R45's full 8-ρ sweeps at N = 15 and N = 20 on Correct, so the adaptive point is read against same-benchmark curves. All 415 pairs are re-rendered, because every Correct anchor was regenerated. R55's LongLive attention recorder is reused as is; its smoke (job 1057789) passed G1–G4.

Done when: the 16 sweep methods and se_max_ll are rendered on the 415 pairs, scored through R57's unified suite, and the trade-off figures place se_max_ll on the N = 15 and N = 20 Correct curves.

## Execution steps

| # | Step id | Type | Waits for | Sets status |
|---|---|---|---|---|
| 0 | *(prep)* `stage1-json-dir`, `stage3-json-dir`, `smoke-cases-json`, `smoke-sbatch`, `sweep-sbatch`, `stage1-sbatch`, `div-sbatch`, `field-sbatch`, `stage3-sbatch`, `chain-driver`, `eval-driver`, `score-script` | /build-step | — | implemented |
| 1 | `smoke` | sbatch (node39) | — | — |
| 2 | `wait-smoke` | manual | smoke | — |
| 3 | `sweep-n15` | sbatch (array 0-7) | wait-smoke | running |
| 4 | `sweep-n20` | sbatch (array 0-7) | wait-smoke | — |
| 5 | `se-chain` | local (submits 5 queue entries) | wait-smoke | — |
| 6 | `wait-renders` | manual | sweep-n15, sweep-n20, se-chain | finished |
| 7 | `r57-check` | manual (R57 adopted, under R57) | — | — |
| 8 | `eval-arms` | local (3 × R57 launcher) | wait-renders, r57-check | — |
| 9 | `eval-sweeps` | local (14 × R57 launcher) | eval-arms | — |
| 10 | `wait-eval` | manual | eval-sweeps | — |
| 11 | `score` | local | wait-eval | — |
| 12 | `verdict` | manual | score | analyzed |

Invoke: `/run-step R61 smoke`, `/run-step R61 wait-smoke`, `/run-step R61 sweep-n15`, `/run-step R61 sweep-n20`, `/run-step R61 se-chain`, `/run-step R61 wait-renders`, `/run-step R61 r57-check`, `/run-step R61 eval-arms`, `/run-step R61 eval-sweeps`, `/run-step R61 wait-eval`, `/run-step R61 score`, `/run-step R61 verdict`.

## Decisions

| Item | Choice |
|---|---|
| Benchmark | FiVE-Bench-Correct, 415 pairs (98 / 100 / 99 / 99 / 9 / 10), prompts from `evaluation/five_bench_correct/`, cases `evaluation/r43_anchor_manifest_correct.json`; no R42 new prompts |
| Anchors | `$D/anchors_r43_correct` (R7 generator, corrected instruction + keep-pose suffix); `$D = ~/Data/dataggen/outputs/five_bench` |
| Render scope | all 415 pairs for every method, no reuse of official-FiVE renders (all anchors changed) |
| Backbone settings | `--backbone ll`, VP, fg_boost 2, flow_shift 1.0, chunk 21, seed 0 re-seeded per pair (as R45 / R55) |
| Native baseline | ρ = 2, N = 15 = `r61_ll_rho2_n15` (index 0 of the N = 15 sweep) |
| Equal-NFE baseline | ρ = 2, N = 20 = `r61_ll_rho2_n20` (index 0 of the N = 20 sweep) |
| Max adaptive | `r61_max_ll`: R55's procedure unchanged (N_d = 5, `--blend_sched tgt0.9`, recorder at r31_stage1 defaults, DINOv3 + LPIPS divergence, `r49_field.py` q = S·max, 15-step render, dump step 14); NFE 20 |
| ρ sweeps | ρ ∈ {2, 3, 4, 6, 8, 10, 20, 50} at N = 15 and N = 20 → 16 methods `r61_ll_rho{R}_n{N}` |
| Correct plumbing | shared `--edit_json_dir` on `r31_stage1.py` / `r31_stage3.py` (default unchanged, reused by R59 / R60); `r31_divergence.py` and `r49_field.py` read no prompts, so they are unchanged |
| Benchmark freeze | sweep script and chain driver refuse to run with uncommitted changes to the Correct JSONs or manifest, and log the commit |
| Metrics | R57 unified suite only (harness incl. motion fidelity, FiVE-Acc, CLIP-D, EditReward, PSC, CF), tag `r61`; scoring blocked until R57 is adopted |
| Eval order | 3 main arms first, then the 14 other sweep methods |
| Scoring | plain per-clip means over 415; figures x = alignment (FiVE-Acc, CLIP-D, CLIP-T, EditReward, PSC), y = bg LPIPS; no CIs |
| Platform | lab L40S, `streamgve` for generation; R57 lab launcher for eval; nothing on Jean Zay |
| Slurm | `--partition=L40S` / `CPU`, `--exclude=node52`; smoke 3 h pinned to node39; sweep 24 h per task; draft 12 h; render 20 h |
| Outputs | `$D/r61_smoke/`, `$D/r61_sweep/n{15,20}/r61_ll_rho{R}_n{N}/`, `$D/r61_ll_stage1/`, `$D/r61_ll_attn/`, `$D/r61_div/se_{dino,lpips}/`, `$D/r61_field/se_max/`, `$D/r61_ll_arms/r61_max_ll/step14/`, `evaluation/csv/suite_r61_*.csv`, `evaluation/csv/r61_*.csv`, `evaluation/figures/r61_ll_*.pdf` |

## Step commands

### smoke
```bash
sbatch slurm_scripts/five_bench/r61_smoke.sh
```

### wait-smoke
```bash
squeue -u $USER
grep -E 'G[1-4]|failures' logs/r61_smoke_*.out
grep -c Traceback logs/r61_smoke_*.err
```

### sweep-n15
```bash
sbatch --array=0-7 slurm_scripts/five_bench/r61_sweep.sh 15
```

### sweep-n20
```bash
sbatch --array=0-7 slurm_scripts/five_bench/r61_sweep.sh 20
```

### se-chain
```bash
bash slurm_scripts/five_bench/r61_chain.sh
cat logs/r61_chain_jobs.txt
```

### wait-renders
```bash
squeue -u $USER
grep -hE 'failures' logs/r61_sweep_*.out logs/r61_ll_stage*_*.out logs/r61_div_*.out logs/r61_field_*.out | sort | uniq -c
D=~/Data/dataggen/outputs/five_bench
for N in 15 20; do for R in 2 3 4 6 8 10 20 50; do
  echo "n${N} rho${R}: $(ls -d $D/r61_sweep/n${N}/r61_ll_rho${R}_n${N}/edit*/*/ 2>/dev/null | grep -vc '_resize/$')"
done; done
echo "max_ll: $(ls -d $D/r61_ll_arms/r61_max_ll/step14/edit*/*/ | wc -l)"
```

### r57-check
```bash
grep -n '| \*\*R57\*\*' daily.md | grep -o '`[a-z]*` |$'
ls slurm_scripts/five_bench/eval_suite.sh
```

### eval-arms
```bash
bash slurm_scripts/five_bench/r61_eval_all.sh arms
cat logs/r61_eval_jobs.txt
```

### eval-sweeps
```bash
bash slurm_scripts/five_bench/r61_eval_all.sh sweeps
cat logs/r61_eval_jobs.txt
```

### wait-eval
```bash
squeue -u $USER
for f in evaluation/csv/suite_r61_*.csv; do case $f in *_per_type.csv) continue;; esac; echo "$f $(($(wc -l < $f) - 1))"; done
```

### score
```bash
python evaluation/r61_score.py
```

### verdict
```bash
column -s, -t < evaluation/csv/r61_points.csv
column -s, -t < evaluation/csv/r61_per_type.csv
ls evaluation/figures/r61_ll_*.pdf
```

## Pipeline

```mermaid
flowchart LR
  C[five_bench_correct JSONs + r43_anchor_manifest_correct + anchors_r43_correct] --> SM[r61_smoke.sh G1-G4]
  SM --> SW15[r61_sweep.sh 15: 8 rho]
  SM --> SW20[r61_sweep.sh 20: 8 rho]
  SM --> S1[r61_ll_stage1.sh draft + attention]
  S1 --> DV[r61_div.sh DINO / LPIPS]
  S1 --> FD[r61_field.sh r49_field.py]
  DV --> FD
  FD --> S3[r61_ll_stage3.sh render r61_max_ll]
  R57[R57 eval_suite.sh, adopted under R57] --> EV[r61_eval_all.sh arms, then sweeps]
  SW15 --> EV
  SW20 --> EV
  S3 --> EV
  EV --> SC[r61_score.py]
  SC --> O[r61_points / sweep / per_type CSVs + r61_ll_*.pdf]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r31_stage1.py` | `--edit_json_dir` (default `data_root/edit_prompt`); `edit_json = Path(args.edit_json_dir).expanduser() / f"edit{T}_FiVE.json" if args.edit_json_dir else data_root / "edit_prompt" / ...`, as `run_fivebench.py:267`; print it in the header. |
| `evaluation/r31_stage3.py` | Same flag and header line. |
| `evaluation/r61_smoke_cases.json` (new) | Two entries copied from `r43_anchor_manifest_correct.json`: `edit1/0001_bus`, `edit6/0042_gym-ball`. |
| `slurm_scripts/five_bench/r61_smoke.sh` (new) | L40S, 3 h, `--nodelist=node39`; preflight `ls $D/r45_sweep/n15/ll_rho2_n15_vp/edit1`; G1 Python check of counts / pair sets / anchor files; G2 `run_fivebench.py` ll ρ=2 N=15 into `$D/r61_smoke/sweep`, frames compared with `r45_sweep` (must differ), grep the Correct `target_prompt` of 0042 in the log; G3 stage1 → `r31_divergence.py` (dino, lpips) → `r49_field.py` → stage3 into `$D/r61_smoke/{stage1,attn,div,field,arms}`, q finite in [0, 1], frame counts equal G2's; G4 stage1 default path on 0001_bus e1 with `$D/anchors`, latents `np.array_equal` to `$D/r55_smoke/on`; `failures: N`. |
| `slurm_scripts/five_bench/r61_sweep.sh` (new) | Copy of `r45_sweep.sh`: drop the `r45_select.py` block (`case 15|20`), `--edit_json_dir evaluation/five_bench_correct`, `ANCHOR_ROOT=$FIVE_ROOT/anchors_r43_correct`, `EXPECTED=(98 100 99 99 9 10)`, `TOTAL_EXPECTED=415`, `METHOD=r61_ll_rho${R}_n${STEPS}`, `OUT_ROOT=$FIVE_ROOT/r61_sweep/n${STEPS}`; `git status --porcelain` freeze check + commit echo before the loop. |
| `slurm_scripts/five_bench/r61_ll_stage1.sh` (new) | Copy of `r55_ll_stage1.sh` with Correct cases, `--edit_json_dir`, Correct anchors, `OUT=$D/r61_ll_stage1`, `ATTN=$D/r61_ll_attn`, `EXPECT=(98 100 99 99 9 10)`. |
| `slurm_scripts/five_bench/r61_div.sh` (new) | Copy of `r55_div.sh` with `CASES=evaluation/r43_anchor_manifest_correct.json`, `r61_ll_stage1` targets / latents, `--out_root $D/r61_div`, 415 guard. |
| `slurm_scripts/five_bench/r61_field.sh` (new) | Copy of `r55_field.sh` with Correct cases, `--attn_se $D/r61_ll_attn --div_root $D/r61_div --out_root $D/r61_field`, 415 guard. |
| `slurm_scripts/five_bench/r61_ll_stage3.sh` (new) | Copy of `r55_ll_stage3.sh` with `--edit_json_dir`, Correct anchors, `FIELD_DIR=$D/r61_field/se_max`, `METHOD=r61_max_ll`, `OUT_ROOT=$D/r61_ll_arms`, `EXPECT=(98 100 99 99 9 10)`. |
| `slurm_scripts/five_bench/r61_chain.sh` (new) | As `r55_chain.sh` minus the eval / motion submissions; freeze check; `logs/r61_chain_jobs.txt`. |
| `slurm_scripts/five_bench/r61_eval_all.sh` (new) | `case $1 in arms|sweeps)`; method → root map (`$D/r61_sweep/n{N}/r61_ll_rho{R}_n{N}`, `$D/r61_ll_arms/r61_max_ll/step14`); 415-dir guard per root; loop over R57's `eval_suite.sh` with `--layout edit_video --edit_json_dir evaluation/five_bench_correct --tag r61`; parse the three printed job ids into `logs/r61_eval_jobs.txt`. |
| `evaluation/r61_score.py` (new) | Load the 17 `suite_r61_{method}.csv`; validate rows / NaN; per-clip means; write the four CSVs; plot with `r49_score`'s `plot` helper (x alignment, y `lpips_unedit_part`), N = 15 and N = 20 sweep curves, three marked points. |
