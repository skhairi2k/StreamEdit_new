---
name: "R60 — StreamEdit Text-Only on FiVE-Correct"
overview: "StreamEdit text-only (Self-Forcing, --vp_mode novp) on the 415 FiVE-Bench-Correct pairs, lab L40S: native N=15 curve (8 ρ, official = ρ 2), equal-NFE N=20 curve (8 ρ) and se_max_novp. The 12 uniform-ρ configs R44 rendered (N=15 all ρ, N=20 ρ 2/4/10/50) are re-rendered only on the 190 pairs whose prompt or trigger word changed (StreamEdit reads source_object/target_object in every arm); the other 225 reuse R44 renders after a 2-pair reproducibility gate. The 4 N=20 points R44 lacks (ρ 3/6/8/20; R53, which was to render them, is dropped) are rendered on all 415. se_max_novp runs its whole chain (draft + attention, DINO + LPIPS divergence, field, render) on all 415 with the Correct annotations. Assembled 415-pair trees per config are scored with R57's lab suite, then read as Pareto curves + paired deltas."
task_id: R60
todos:
  - id: stage1-anno-flag
    content: "evaluation/r31_stage1.py -- new --edit_json_dir (default None = data_root/edit_prompt, as run_fivebench.py); edit_json read from it; printed in the start line"
    needed_by: [render-chain]
    status: pending
  - id: stage3-anno-flag
    content: "evaluation/r31_stage3.py -- new --edit_json_dir (default None = data_root/edit_prompt); edit_json read from it; printed in the start line"
    needed_by: [render-chain]
    status: pending
  - id: cases-script
    content: "evaluation/r60_build_cases.py -- compares evaluation/five_bench_correct/edit{T}_FiVE.json with ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/edit_prompt/edit{T}_FiVE.json over the 415 pairs of r43_anchor_manifest_correct.json; writes evaluation/r60_cases_regen.json (source_prompt / target_prompt / source_object / target_object changed, expect 190 = 15/32/54/70/9/10 per type), evaluation/r60_cases_repro.json (first unchanged pair of edit1 and of edit4) and evaluation/csv/r60_regen_sets.csv (case_id, edit_type, video_name, changed fields, regen 0/1); r7 manifest format; exit 1 if the counts differ or a pair is missing from either JSON"
    needed_by: [build-cases]
    status: pending
  - id: render-sbatch
    content: "slurm_scripts/five_bench/r60_render.sh -- copy of r44_infer.sh's run_fivebench.py call (--vp_mode novp, fg_boost 4, flow_shift 1.0, chunk 21, seed 0) plus --edit_json_dir evaluation/five_bench_correct; MODE correct (array 0-15): 0-7 N=15 ρ 2,3,4,6,8,10,20,50 and 8-11 N=20 ρ 2,4,10,50 with --cases_json evaluation/r60_cases_regen.json (190), 12-15 N=20 ρ 3,6,8,20 with --cases_json evaluation/r43_anchor_manifest_correct.json (415), out $D/r60_rho_sweep/r60_n{N}_rho{R}_novp; MODE repro (array 0-1): n15_rho2 / n20_rho2 on r60_cases_repro.json, out $D/r60_repro; per-type expected counts read from the case file; r44_infer.sh's skip / count / manifest guards; L40S, --exclude=node52, 64G, 24 h (indices 12-15: 415 clips at N=20, ~14 h by R39's estimate)"
    needed_by: [repro-render, render-chain]
    status: pending
  - id: stage1-sbatch
    content: "slurm_scripts/five_bench/r60_se_stage1.sh -- copy of r53_se_stage1.sh with CASES=evaluation/r43_anchor_manifest_correct.json and --edit_json_dir evaluation/five_bench_correct; OUT=$D/r60_se_stage1, ATTN=$D/r60_se_attn; guard: 415 attention npz, 415 latents, every latent records vp_mode == novp"
    needed_by: [render-chain]
    status: pending
  - id: div-sbatch
    content: "slurm_scripts/five_bench/r60_div.sh -- copy of r53_div.sh (array 0-1 se_dino / se_lpips) with CASES = Correct manifest, TARGET_ROOT=$D/r60_se_stage1/frames/n5_tgt09/step04, --latent_dir $D/r60_se_stage1/latents, out $D/r60_div; 415-npz guard"
    needed_by: [render-chain]
    status: pending
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r60_field.sh -- copy of r53_field.sh: r49_field.py --selftest, then --hosts se --cases Correct manifest --attn_se $D/r60_se_attn --div_root $D/r60_div --out_root $D/r60_field; 415-npz guard on se_max"
    needed_by: [render-chain]
    status: pending
  - id: stage3-sbatch
    content: "slurm_scripts/five_bench/r60_se_stage3.sh -- copy of r53_se_stage3.sh with CASES = Correct manifest, --edit_json_dir evaluation/five_bench_correct, FIELD_DIR=$D/r60_field/se_max, METHOD=r60_max_novp_correct, OUT_ROOT=$D/r60_se_arms; same preflight, 415-clip and manifest guards"
    needed_by: [render-chain]
    status: pending
  - id: chain-driver
    content: "slurm_scripts/five_bench/r60_chain.sh -- lab driver, --only base|max|all (default all), --dry-run; refuses if logs/r60_chain_jobs.txt exists (all) or already lists the part; prints squeue -u $USER -h -r | wc -l and refuses if it plus the new tasks (base 16, max 5) exceeds 30; base = r60_render.sh correct --array=0-15; max = stage1 -> div (--array=0-1, afterok) -> field (afterok) -> stage3 (afterok); appends '<stage> <id>' lines headed by the git commit"
    needed_by: [render-chain]
    status: pending
  - id: repro-check-script
    content: "evaluation/r60_repro_check.py -- for the 2 repro pairs x {n15_rho2, n20_rho2} compares $D/r60_repro/r60_n{N}_rho2_novp/edit{T}/{video}/*.png with $D/r44_rho_sweep/r44_n{N}_rho2_novp/... (n20 is on /store: run on nodecpu12): same frame count, max |diff| and PSNR per frame; PASS iff every frame has max |diff| <= 2 (of 255); writes evaluation/csv/r60_repro_check.csv; exit 1 on FAIL"
    needed_by: [repro-check]
    status: pending
  - id: assemble-script
    content: "evaluation/r60_assemble.py -- builds $D/r60_correct/{config}/edit{T}/{video} for the 17 configs over the 415 Correct pairs: regen pair -> symlink to $D/r60_rho_sweep/r60_{config}_novp/...; unchanged pair -> source r44_rho_sweep/r44_{config}_novp (n15 all ρ, n20 ρ 2/4/10/50), symlinked if it resolves under /home, copied (shutil.copytree) if it resolves under /store; n20 ρ 3/6/8/20 -> symlink to $D/r60_rho_sweep/r60_{config}_novp/... for all 415; max_novp -> symlink to $D/r60_se_arms/r60_max_novp_correct/step14/... for all 415; checks every clip's PNG count equals its source's; idempotent; writes evaluation/csv/r60_assembly.csv (config, case_id, source new|reused, storage link|copy, n_png); exit 1 on a missing/incomplete clip or a count other than 415 per config"
    needed_by: [assemble]
    status: pending
  - id: assemble-sbatch
    content: "slurm_scripts/five_bench/r60_assemble.sh -- CPU partition, --nodelist=nodecpu12 (sees /store), 4 cpus, 16G, 6 h; fail-fast if /store/PERSO/skhairi/Outputs/five_bench/r44_rho_sweep is not readable; runs r60_assemble.py with the five-bench python; prints '[r60_assemble] failures: N'"
    needed_by: [assemble]
    status: pending
  - id: eval-launcher
    content: "slurm_scripts/five_bench/r60_eval.sh -- bash loop over the configs given as arguments (or all 17): bash slurm_scripts/five_bench/eval_suite.sh --method r60_{config}_correct --root $D/r60_correct/{config} --layout edit_video --tag r60 (default --edit_json_dir = Correct, no --cases = 415 pairs); refuses a config whose evaluation/csv/suite_r60_r60_{config}_correct.csv exists; refuses if queue + 3 x n_configs > 30; appends the job ids per config to logs/r60_eval_jobs.txt"
    needed_by: [eval]
    status: pending
  - id: score-script
    content: "evaluation/r60_score.py -- reads the 17 suite_r60_r60_{config}_correct.csv (415 rows each; exit 1 otherwise); writes evaluation/csv/r60_points.csv (config, N, ρ, role curve15|curve20|arm, plain per-clip mean overall and per edit type, every suite metric), evaluation/csv/r60_deltas.csv (max_novp minus n20_rho2 (equal NFE) and n15_rho2 (official), paired bootstrap 95% CI, 10k resamples seed 0, overall + per type + regen-vs-reused split) and evaluation/figures/r60_pareto/{alignment}_vs_{preservation}.pdf (x = CLIP-T, FiVE-Acc, CLIP-D, EditReward, PSC; y = bg LPIPS, CF; N=15 curve, N=20 curve, max_novp point, official n15_rho2 marked)"
    needed_by: [score]
    status: pending
steps:
  - id: build-cases
    type: local
    command: python evaluation/r60_build_cases.py
    output_paths:
      - evaluation/r60_cases_regen.json
      - evaluation/r60_cases_repro.json
      - evaluation/csv/r60_regen_sets.csv
    status: pending
  - id: repro-render
    type: sbatch
    wait_for: build-cases
    command: sbatch --array=0-1 slurm_scripts/five_bench/r60_render.sh repro
    status: pending
  - id: wait-repro
    type: manual
    wait_for: repro-render
    check_hint: "sacct -j {job_id}: both tasks COMPLETED; both logs 'failures: 0'; 2 clip dirs under each $D/r60_repro/r60_n{15,20}_rho2_novp"
    status: pending
  - id: repro-check
    type: srun
    wait_for: wait-repro
    command: srun --partition=CPU --nodelist=nodecpu12 --cpus-per-task=2 --mem=8G --time=00:30:00 ~/anaconda3/envs/five-bench/bin/python evaluation/r60_repro_check.py
    output_paths:
      - evaluation/csv/r60_repro_check.csv
    status: pending
  - id: render-chain
    type: local
    wait_for: repro-check
    command: bash slurm_scripts/five_bench/r60_chain.sh
    sets_status: running
    status: pending
  - id: wait-chain
    type: manual
    wait_for: render-chain
    check_hint: "no r60 job in squeue; grep -h failures logs/r60_*.out | sort | uniq -c -> all 'failures: 0'; base tasks 0-11 report 190 dirs and 12-15 report 415; 415 attention + latent npz (all novp), 2 x 415 divergence, 415 field npz, 415 r60_max_novp_correct clips"
    sets_status: finished
    status: pending
  - id: assemble
    type: sbatch
    wait_for: wait-chain
    command: sbatch slurm_scripts/five_bench/r60_assemble.sh
    output_paths:
      - evaluation/csv/r60_assembly.csv
    status: pending
  - id: wait-assemble
    type: manual
    wait_for: assemble
    check_hint: "log '[r60_assemble] failures: 0'; summary 17 configs x 415 clips, new = 190 (12 R44 configs) / 415 (n20 ρ 3/6/8/20, max_novp), 0 missing"
    status: pending
  - id: eval
    type: local
    wait_for: wait-assemble
    command: bash slurm_scripts/five_bench/r60_eval.sh max_novp n15_rho2 n20_rho2
    check_hint: "Needs R57 `adopt` (eval_suite.sh validated on the lab). Submit the remaining configs in batches as the queue frees: bash slurm_scripts/five_bench/r60_eval.sh <configs>; 3 jobs per config"
    status: pending
  - id: wait-eval
    type: manual
    wait_for: eval
    check_hint: "17 files evaluation/csv/suite_r60_r60_*_correct.csv, 415 rows each, no NaN column (merge logs exit 0); ls evaluation/csv/suite_r60_r60_*_correct.csv | wc -l = 17"
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
    command: python evaluation/r60_score.py
    output_paths:
      - evaluation/csv/r60_points.csv
      - evaluation/csv/r60_deltas.csv
      - evaluation/figures/r60_pareto/
    status: pending
  - id: analyze
    type: manual
    wait_for: score
    check_hint: "Read r60_points.csv / r60_deltas.csv and the Pareto figures: is max_novp above the N=20 text-only curve on Correct? per-type pattern; regen-vs-reused split shows no shift. Record the readout in the Run log."
    sets_status: analyzed
    status: pending
isProject: true
---

# R60: StreamEdit Text-Only on FiVE-Correct

## Context
FiVE-Bench-Correct is the paper's main benchmark (supervisor, 2026-10-23). This task covers the text-only StreamEdit rows (Self-Forcing, `--vp_mode novp`): the native N=15 ρ curve (the official setting is ρ = 2), the equal-NFE N=20 ρ curve, and the adaptive se_max_novp (5-step draft + 15-step render = 20 NFE). Text-only renders read only `source_prompt`, `target_prompt`, `source_object` and `target_object`, so a pair whose four fields Correct left unchanged renders exactly as on official FiVE. The uniform-ρ configs R44 rendered are therefore re-rendered only on the changed pairs. The 4 N=20 ρ values R44 lacks (3/6/8/20) and se_max_novp have no official render to reuse (R53, which was to produce them, is dropped), so they run on all 415 pairs. Every config is evaluated again on all 415 pairs, because Correct changed the FiVE-Acc questions, objects and instructions that the metrics read.

**Done when:** 17 configs × 415 pairs are scored with R57's lab suite, and `r60_points.csv` / `r60_deltas.csv` / `r60_pareto/` have been read.

## Execution steps

| # | step id | type | waits for | sets status |
|---|---|---|---|---|
| — | *(prep)* stage1-anno-flag · stage3-anno-flag · cases-script · render-sbatch · stage1-sbatch · div-sbatch · field-sbatch · stage3-sbatch · chain-driver · repro-check-script · assemble-script · assemble-sbatch · eval-launcher · score-script | todos | — | `implemented` |
| 1 | `build-cases` | local | — | — |
| 2 | `repro-render` | sbatch (L40S, array 0-1) | build-cases | — |
| 3 | `wait-repro` | manual | repro-render | — |
| 4 | `repro-check` | srun (CPU, nodecpu12) | wait-repro | — |
| 5 | `render-chain` | local (submits 21 tasks) | repro-check | `running` |
| 6 | `wait-chain` | manual | render-chain | `finished` |
| 7 | `assemble` | sbatch (CPU, nodecpu12) | wait-chain | — |
| 8 | `wait-assemble` | manual | assemble | — |
| 9 | `eval` | local (batches; needs R57 `adopt`) | wait-assemble | — |
| 10 | `wait-eval` | manual | eval | — |
| 11 | `score` | local | wait-eval | — |
| 12 | `analyze` | manual | score | `analyzed` |

```
/build-step build-cases R60
/build-step repro-render R60
/build-step repro-check R60
/build-step render-chain R60
/build-step assemble R60
/build-step eval R60
/build-step score R60
/run-step R60 build-cases
/run-step R60 repro-render
/run-step R60 wait-repro
/run-step R60 repro-check
/run-step R60 render-chain
/run-step R60 wait-chain
/run-step R60 assemble
/run-step R60 wait-assemble
/run-step R60 eval
/run-step R60 wait-eval
/run-step R60 score
/run-step R60 analyze
```

## Decisions

| Decision | Choice |
|---|---|
| Benchmark | FiVE-Bench-Correct, 415 pairs (`evaluation/r43_anchor_manifest_correct.json`, annotations `evaluation/five_bench_correct/`). The uncommitted edits to `edit1_FiVE.json` / `edit6_FiVE.json` are committed before `build-cases`, and the regen set is computed from the committed files |
| Regen set (R44 configs) | Computed from the JSON files (official vs Correct): pairs with a changed `source_prompt`, `target_prompt`, `source_object` or `target_object` = **190** (47 prompt + 143 trigger-word only; per type 15/32/54/70/9/10; edit5 and edit6 entirely). Unlike Wan's baselines (47), StreamEdit's cross-attention grounding/boosting reads the trigger words in every arm. Checked 2026-10-26 |
| Configs (17) | `n15_rho{2,3,4,6,8,10,20,50}` (native; `n15_rho2` = official text-only), `n20_rho{2,3,4,6,8,10,20,50}` (equal NFE; `n20_rho2` = equal-NFE official), `max_novp` (se_max text-only) |
| Reuse sources (225 unchanged pairs) | `n15_*` and `n20_rho{2,4,10,50}` ← `$D/r44_rho_sweep/r44_{config}_novp` (all on /store except `n15_rho2`). `n20_rho{3,6,8,20}` have no official render (R53 dropped): rendered on all 415 by R60 |
| se_max_novp | Own run on all 415 (user's choice; R53 dropped): R53's settings, via copies of its never-run `r53_*.sh` scripts, so R60's chain is their first run (N=5 `tgt0.9` draft, seed 0, attention recorder at defaults, DINOv3 + LPIPS divergence with temporal mean, `r49_field.py` unchanged, render N=15 at `dump_steps 14`) plus the Correct annotations |
| Render settings | As R44: `--vp_mode novp`, no anchor, fg_boost 4, flow_shift 1.0, chunk 21, seed 0; only `--edit_json_dir` and `--cases_json` are added |
| Reuse gate | 2 unchanged pairs (first of edit1 and of edit4) rendered again with `n15_rho2` and `n20_rho2` and compared against R44's renders; PASS iff max \|diff\| ≤ 2/255 on every frame. A FAIL stops the task and goes back to the user to decide (full 415 re-render) |
| Platform | Lab L40S, env `streamgve` for every render, `five-bench` for divergence / field / eval; all reused renders are L40S + streamgve too |
| /store | Reused /store clips are **copied** into `$D/r60_correct/` (≈ 11 configs × 225 clips ≈ 90 GB), on nodecpu12, so that eval needs no node pin. /home sources and new renders are symlinked |
| Output dirs | `$D/r60_rho_sweep/r60_{config}_novp/`, `$D/r60_repro/`, `$D/r60_se_stage1/`, `$D/r60_se_attn/`, `$D/r60_div/`, `$D/r60_field/`, `$D/r60_se_arms/r60_max_novp_correct/step14/`, assembled `$D/r60_correct/{config}/` (`D=~/Data/dataggen/outputs/five_bench`) |
| Evaluation | R57 unified suite on the lab (`eval_suite.sh`), all 415 pairs per config, `--tag r60`, method `r60_{config}_correct` → `evaluation/csv/suite_r60_r60_{config}_correct.csv`. Blocked until R57 is adopted |
| Queue | Per-user cap ~30 tasks: the chain driver and eval launcher refuse to exceed it; base (16) and max (5) can be submitted separately, and eval in config batches (3 jobs each) |
| Averaging / plots | Plain per-clip mean from the suite CSV, never `_avg.csv`; Pareto plots with x = alignment, y = preservation |
| Readout | max_novp vs `n20_rho2` (equal NFE, headline) and vs `n15_rho2` (official), placed on both curves; paired bootstrap 95% CI; per edit type; regen-vs-reused split as a sanity check |
| Out of scope | VP StreamEdit on Correct (R59), LongLive, Wan-Edit (R58), any text-only run on official FiVE (R53 dropped) |

## Step commands

### build-cases
```bash
python evaluation/r60_build_cases.py
```

### repro-render
```bash
sbatch --array=0-1 slurm_scripts/five_bench/r60_render.sh repro
```

### wait-repro
```bash
sacct -j {job_id} --format=JobID,State,Elapsed
grep -h failures logs/r60_render_{job_id}_*.out
```

### repro-check
```bash
srun --partition=CPU --nodelist=nodecpu12 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  ~/anaconda3/envs/five-bench/bin/python evaluation/r60_repro_check.py
```

### render-chain
```bash
bash slurm_scripts/five_bench/r60_chain.sh --dry-run
bash slurm_scripts/five_bench/r60_chain.sh            # or --only base / --only max if the queue is busy
cat logs/r60_chain_jobs.txt
```

### wait-chain
```bash
squeue -u $USER
grep -h failures logs/r60_*.out | sort | uniq -c
```

### assemble
```bash
sbatch slurm_scripts/five_bench/r60_assemble.sh
```

### wait-assemble
```bash
tail -n 5 logs/r60_assemble_{job_id}.out
```

### eval
```bash
bash slurm_scripts/five_bench/r60_eval.sh max_novp n15_rho2 n20_rho2
# then, as the queue frees, the other 14 configs in batches, e.g.:
bash slurm_scripts/five_bench/r60_eval.sh n15_rho3 n15_rho4 n15_rho6
```

### wait-eval
```bash
ls evaluation/csv/suite_r60_r60_*_correct.csv | wc -l   # 17
for f in evaluation/csv/suite_r60_r60_*_correct.csv; do echo "$f $(($(wc -l < $f) - 1))"; done   # 415 each
```

### score
```bash
python evaluation/r60_score.py
```

### analyze
```bash
column -s, -t < evaluation/csv/r60_deltas.csv | less -S
ls evaluation/figures/r60_pareto/
```

## Pipeline

```mermaid
flowchart LR
  A[official edit_prompt JSON] --> B[r60_build_cases.py]
  C[five_bench_correct JSON] --> B
  B --> D[r60_cases_regen 190]
  B --> E[r60_cases_repro 2]
  E --> F[r60_render.sh repro] --> G[r60_repro_check.py]
  R44[R44 renders /home + /store] --> G
  D --> H[r60_render.sh correct 0-11 on 190]
  M2[Correct manifest 415] --> H2[r60_render.sh correct 12-15 on 415]
  C --> H
  C --> H2
  H2 --> AS
  M[Correct manifest 415] --> S1[r60_se_stage1.sh draft + attention]
  C --> S1
  S1 --> DV[r60_div.sh DINO / LPIPS] --> FD[r60_field.sh]
  S1 --> FD
  FD --> S3[r60_se_stage3.sh max_novp]
  H --> AS[r60_assemble.sh / r60_assemble.py]
  S3 --> AS
  R44 --> AS
  AS --> T[r60_correct 17 x 415]
  T --> EV[r60_eval.sh -> eval_suite.sh R57]
  EV --> CSV[suite_r60 CSVs]
  CSV --> SC[r60_score.py]
  SC --> O[r60_points / r60_deltas / r60_pareto]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r31_stage1.py` | `--edit_json_dir` (default None → `data_root / "edit_prompt"`), used at the `edit_json` read |
| `evaluation/r31_stage3.py` | `--edit_json_dir` (default None → `data_root / "edit_prompt"`), used at the `edit_json` read |
| `evaluation/r60_build_cases.py` (new) | Regen set from the JSON comparison → `r60_cases_regen.json`, `r60_cases_repro.json`, `r60_regen_sets.csv`; asserts 190 (15/32/54/70/9/10) and 2 |
| `slurm_scripts/five_bench/r60_render.sh` (new) | `correct` (array 0-15: index → (N, ρ, case file)) and `repro` (array 0-1) modes; `--edit_json_dir evaluation/five_bench_correct --cases_json <case file>`; expected per-type counts read from the case file |
| `slurm_scripts/five_bench/r60_se_stage1.sh` (new) | `r53_se_stage1.sh` with the Correct manifest, `--edit_json_dir`, `r60_se_stage1` / `r60_se_attn`, 415 guards |
| `slurm_scripts/five_bench/r60_div.sh` (new) | `r53_div.sh` with the Correct manifest and `r60_*` roots, 415 guard |
| `slurm_scripts/five_bench/r60_field.sh` (new) | `r53_field.sh` with the Correct manifest and `r60_*` roots, 415 guard |
| `slurm_scripts/five_bench/r60_se_stage3.sh` (new) | `r53_se_stage3.sh` with the Correct manifest, `--edit_json_dir`, `r60_field/se_max`, method `r60_max_novp_correct` |
| `slurm_scripts/five_bench/r60_chain.sh` (new) | `--only base\|max\|all`, `--dry-run`, queue-cap check (30); base array 0-15 + afterok chain stage1 → div → field → stage3; `logs/r60_chain_jobs.txt` |
| `evaluation/r60_repro_check.py` (new) | Pixel comparison of the repro renders vs R44 (`n15_rho2` /home, `n20_rho2` /store), PASS ≤ 2/255 → `r60_repro_check.csv` |
| `evaluation/r60_assemble.py` (new) | `$D/r60_correct/{config}` over 415 pairs: link new renders (all 415 for `n20_rho{3,6,8,20}` and `max_novp`), link /home reuse, copy /store reuse; PNG-count check; `r60_assembly.csv` |
| `slurm_scripts/five_bench/r60_assemble.sh` (new) | CPU job pinned to nodecpu12, /store preflight, runs `r60_assemble.py` |
| `slurm_scripts/five_bench/r60_eval.sh` (new) | Loop of `eval_suite.sh` over the given configs, tag `r60`, refuses existing CSVs and queue overflow, `logs/r60_eval_jobs.txt` |
| `evaluation/r60_score.py` (new) | Points (overall + per type), deltas with paired bootstrap, Pareto PDFs |

`r60_render.sh correct` index → config:

```
0-7   N=15  rho 2 3 4 6 8 10 20 50   r60_cases_regen.json (190)              -> r60_n15_rho{R}_novp
8-11  N=20  rho 2 4 10 50            r60_cases_regen.json (190)              -> r60_n20_rho{R}_novp
12-15 N=20  rho 3 6 8 20             r43_anchor_manifest_correct.json (415)  -> r60_n20_rho{R}_novp
repro 0 n15_rho2 / 1 n20_rho2 on r60_cases_repro.json -> $D/r60_repro
```
