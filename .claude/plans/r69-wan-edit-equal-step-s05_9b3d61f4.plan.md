---
name: "R69 — Wan-Edit Equal-Step Sweep at s=0.5"
overview: "One extra point on Wan-Edit's equal-step (matched-59) global curve on FiVE-Bench-Correct: matched59_s0.5 (N = 354, skip 295, 59 active steps = 236 passes per clip, the same compute as R58's matched-59 configs), rendered on all 415 pairs on Jean Zay H100 in one resumable job, fetched to lab /scratch, archived on /store, scored with the full R57 suite (core, NIQE, motion, CF) on the lab. Purpose: R58's most faithful equal-step point (s = 0.85) is less faithful than max_s5 on PSNR, SSIM and LPIPS, so FiVE-Acc at matched background fidelity cannot be interpolated; s = 0.5 should bracket max_s5 on all three. Deliverable: the suite's three tables for the new setting."
task_id: R69
todos:
  - id: matched59-config
    content: "evaluation/r49_wan_render.py -- append 0.5 to MATCHED59 (comment: R69, equal-step point more faithful than max_s5); matched59_configs() then also yields matched59_s0.5 = (354, 295), its assertion (59 active, start sigma within 0.015 of 0.5: 0.4997) passes"
    needed_by: [wan-render]
    status: completed
  - id: render-sbatch
    content: "slurm_scripts/five_bench/r69_wan_render.sh -- single job (lab header, ported to Jean Zay H100 by jz/r50_port.sh; --time=20:00:00): r49_wan_render.py --config matched59_s0.5 --mode full --cases evaluation/r43_anchor_manifest_correct.json --anno_dir evaluation/five_bench_correct --out_root $D/r69_wan_arms; refuses any OUT other than $D/r69_wan_arms; pre-flight: the case file, the anno dir and r49_wan_render.py exist and resolve('matched59_s0.5') gives (354, 295); post-run guard: 415 clip dirs under $D/r69_wan_arms/matched59_s0.5_full, each with r48_wan_common.expected_frames PNGs; ends '[r69_wan_render] failures: N'"
    needed_by: [wan-render]
    status: completed
  - id: port-list
    content: "jz/r50_port.sh -- add r69_wan_render to SCRIPTS"
    needed_by: [jz-sync]
    status: completed
  - id: scratch-folders
    content: "jz/r50_scratch_links.sh -- add r69_wan_arms to FOLDERS (and the header comment)"
    needed_by: [jz-sync]
    status: completed
  - id: fetch-renders-script
    content: "jz/r69_fetch_renders.sh -- lab frontal (gpu-gw) only, modelled on jz/r62_fetch_renders.sh (r50_jz.env, SSH ControlMaster, $SCRATCH lookup): rsync -rL --partial --size-only --no-perms --no-owner --no-group (no -t) zay:$SCRATCH/five_bench_out/r69_wan_arms/matched59_s0.5_full/ -> $R69_DEST/matched59_s0.5/ (default R69_DEST=/scratch/PERSO/skhairi/five_bench/r69_wan_correct; refuses any destination outside /scratch); preflight >= 12 GB free; post-check 415 clip dirs with the expected PNG count each (r48_wan_common.expected_frames); --check mode counts only; never deletes on either side; ends '[r69_fetch] failures: N'"
    needed_by: [fetch-renders]
    status: completed
  - id: store-check-entry
    content: "evaluation/check_store_renders.py -- add r69 to TASKS and r69_wan_correct/ to JZ_ROOTS (store path /store/PERSO/skhairi/Outputs/jz_five_bench/r69_wan_correct)"
    needed_by: [store-copy]
    status: completed
  - id: eval-batch
    content: "evaluation/eval_suite/batches/r69.txt -- 1 line `NAME ROOT` (R57 batch format, ROOT relative to --root_base /scratch/PERSO/skhairi/five_bench): wan_matched59_s0p5_correct r69_wan_correct/matched59_s0.5"
    needed_by: [eval-core, eval-niqe, eval-motion, eval-cf]
    status: completed
steps:
  - id: jz-sync
    type: manual
    check_hint: "User: commit + push. On Jean Zay in the clone: `git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh`; paste the tails: '[r50_port] leftovers: 0', '[r50_port] failures: 0', '[r50_scratch] failures: 0'; `ls -ld $D/r69_wan_arms` is a symlink into $SCRATCH"
    status: pending
  - id: wan-render
    type: manual
    wait_for: jz-sync
    check_hint: "User, on Jean Zay: `sbatch slurm_scripts/five_bench/r69_wan_render.sh`; paste the job id"
    sets_status: running
    status: pending
  - id: wait-render
    type: manual
    wait_for: wan-render
    check_hint: "User, on Jean Zay: sacct COMPLETED; log ends '[r69_wan_render] failures: 0' with 415/415 clips. If the job hit the 20 h limit: resubmit the same command (finished clips are skipped) until 415/415"
    status: pending
  - id: fetch-renders
    type: manual
    wait_for: wait-render
    check_hint: "User, on the lab frontal gpu-gw in tmux: `bash jz/r69_fetch_renders.sh --check` then `bash jz/r69_fetch_renders.sh` -> '[r69_fetch] failures: 0' (415 clip dirs, PNG counts as expected, ~9 GB)"
    status: pending
  - id: store-copy
    type: manual
    wait_for: fetch-renders
    check_hint: "User, on gpu-gw in tmux (/store is mounted on the frontal only): `S=/store/PERSO/skhairi/Outputs/jz_five_bench; [ -d $S ] || { echo '/store not mounted here'; exit 1; }; mkdir -p $S/r69_wan_correct && rsync -a --size-only /scratch/PERSO/skhairi/five_bench/r69_wan_correct/ $S/r69_wan_correct/`, then `~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py r69` -> '[check_store] failures: 0'. The Jean Zay copy ($SCRATCH r69_wan_arms) may be deleted only after this passes and the scores are in"
    sets_status: finished
    status: pending
  - id: eval-core
    type: local
    wait_for: fetch-renders
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part core --per_job 1
    status: pending
  - id: eval-niqe
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part niqe --per_job 1
    status: pending
  - id: eval-motion
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part motion --per_job 1
    status: pending
  - id: eval-cf
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part cf --per_job 1
    status: pending
  - id: wait-eval
    type: manual
    wait_for: [eval-core, eval-niqe, eval-motion, eval-cf]
    check_hint: "every eval_suite job of this task ended (nothing left in `squeue -u $USER -h -o '%i %k' | grep 'eval_suite|r69|'`), its logs end 'failures: 0'; evaluation/csv/suite_r69_wan_matched59_s0p5_correct.csv + _per_type.csv + _overall.csv exist, 415 rows, niqe_target_image, motion_fidelity_score and cf filled; every NaN cell listed by the merge logs recorded in the Run log; scored before the 30-day /scratch purge (else re-stage from /store)"
    status: pending
  - id: verdict
    type: manual
    wait_for: wait-eval
    check_hint: "Read suite_r69_wan_matched59_s0p5_correct_overall.csv against suite_r58_wan_max_s5_correct_overall.csv: bracket check -- s=0.5 has higher PSNR, higher SSIM and lower LPIPS than max_s5 (else that metric stays 'outside the curve'); record the s=0.5 point and the bracket result in the Run log"
    sets_status: analyzed
    status: pending
isProject: true
---

# R69: Wan-Edit Equal-Step Sweep at s=0.5

## Context

R58's equal-step Wan-Edit curve (matched-59: s = 0.85, 0.92, 0.95, 0.978) stops short of the adaptive run's preservation: its most faithful point, s = 0.85 (PSNR 27.82, SSIM 84.07, LPIPS 195.1, FiVE-Acc 28.0), is less faithful than max_s5 (28.44, 85.42, 194.5, 50.7) on every preservation metric, so FiVE-Acc at matched background fidelity cannot be interpolated at equal steps and the protocol forbids extrapolation. R69 adds one gentler equal-step point, s = 0.5, expected to beat max_s5 on PSNR, SSIM and LPIPS (R58's native 50-step sweep reaches SSIM 85.36 at s = 0.6 and 85.84 at s = 0.4; matched-59 runs preserve slightly more than native at the same s).

Done when: matched59_s0.5 is rendered on the 415 FiVE-Bench-Correct pairs (Jean Zay H100), copied to lab /scratch and archived on /store, scored with all four R57 parts on the lab, the three suite tables exist, and the bracket check against max_s5 is recorded.

## Execution steps

| 0 | Step id | Type | Runs on | Waits for | Sets status |
|---|---|---|---|---|---|
| 1 | *(prep)* `matched59-config`, `render-sbatch`, `port-list`, `scratch-folders`, `fetch-renders-script`, `store-check-entry`, `eval-batch` | /build-step | lab (code) | — | `implemented` |
| 2 | `jz-sync` | manual | Jean Zay | — | — |
| 3 | `wan-render` | manual (sbatch, 1 job) | **Jean Zay H100** | jz-sync | `running` |
| 4 | `wait-render` | manual | Jean Zay | wan-render | — |
| 5 | `fetch-renders` | manual (tmux) | gpu-gw | wait-render | — |
| 6 | `store-copy` | manual (tmux) | gpu-gw | fetch-renders | `finished` |
| 7 | `eval-core` | local (eval_suite `--part core`) | **lab L40S** | fetch-renders | — |
| 8 | `eval-niqe` | local (eval_suite `--part niqe`) | lab L40S | eval-core | — |
| 9 | `eval-motion` | local (eval_suite `--part motion`) | lab L40S | eval-core | — |
| 10 | `eval-cf` | local (eval_suite `--part cf`) | **lab H100** | eval-core | — |
| 11 | `wait-eval` | manual | lab | eval-core, eval-niqe, eval-motion, eval-cf | — |
| 12 | `verdict` | manual | lab | wait-eval | `analyzed` |

```
/build-step wan-render R69
/build-step fetch-renders R69
/build-step store-copy R69
/build-step eval-core R69
/run-step R69 jz-sync
/run-step R69 wan-render
/run-step R69 wait-render
/run-step R69 fetch-renders
/run-step R69 store-copy
/run-step R69 eval-core
/run-step R69 eval-niqe
/run-step R69 eval-motion
/run-step R69 eval-cf
/run-step R69 wait-eval
/run-step R69 verdict
```

## Decisions

| Item | Choice |
|---|---|
| Config | `matched59_s0.5`: sample_steps N = 354, skip 295, 59 active steps, start σ = 0.4997 (r48_wan_sched.matched_n, checked 2026-11-09) |
| Compute | 59 active steps × 4 passes = 236 backbone passes per clip, identical to R58's matched-59 configs; skipped steps make no model call (only one latent-sized noise draw each) |
| Strength | s = 0.5 only (user, 2026-11-09: compute for one point); chosen so the point is more faithful than max_s5 on SSIM with margin, and on PSNR / LPIPS |
| Renderer settings | `r49_wan_render.py` as R58's matched-59 configs: text only, seeded noise base_seed 0, n_avg 1, edit.py guide scales, up to 41 frames 832×480 (4k+1) |
| Pairs | All 415 FiVE-Bench-Correct pairs, `evaluation/r43_anchor_manifest_correct.json`, annotations `evaluation/five_bench_correct/`; no reuse (no earlier s = 0.5 render exists) |
| Render job | One sbatch on 1 Jean Zay H100, 20 h, resumable (resubmit on timeout); estimated ~12–13 h |
| Paths | Jean Zay `$D/r69_wan_arms/matched59_s0.5_full/` (→ `$SCRATCH`); lab `/scratch/PERSO/skhairi/five_bench/r69_wan_correct/matched59_s0.5/`; archive `/store/PERSO/skhairi/Outputs/jz_five_bench/r69_wan_correct/` |
| No overwrite | New renders only under `$D/r69_wan_arms`; R48 / R49 / R58 folders never written |
| Evaluation | Common R57 procedure on the lab, all four parts (user, 2026-11-09): `--part core`, `niqe`, `motion`, `cf`; batch `r69.txt`, tag `r69`, `--layout edit_video --src_window render`, `--per_job 1` |
| Deliverable | The suite's three tables `evaluation/csv/suite_r69_wan_matched59_s0p5_correct{,_per_type,_overall}.csv` (user, 2026-11-09: the results CSV; no bootstrap, no extra score script) |
| Out of scope | Other strengths, bootstrap CIs, updating `evaluation/paper_tables.py`'s Wan curve, re-scoring R58 |

## Step commands

### jz-sync
```bash
# lab: commit + push; then on Jean Zay, in the clone:
git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh
ls -ld $D/r69_wan_arms
```

### wan-render
```bash
# on Jean Zay
sbatch slurm_scripts/five_bench/r69_wan_render.sh
```

### wait-render
```bash
# on Jean Zay
sacct -j <job_id> --format=JobID,State,Elapsed
tail -n 5 logs/r69_wan_render_<job_id>.out
# timeout: resubmit the same sbatch until 415/415
```

### fetch-renders
```bash
# lab frontal gpu-gw, in tmux
bash jz/r69_fetch_renders.sh --check
bash jz/r69_fetch_renders.sh
```

### store-copy
```bash
# lab frontal gpu-gw, in tmux
S=/store/PERSO/skhairi/Outputs/jz_five_bench; [ -d $S ] || { echo '/store not mounted here'; exit 1; }
mkdir -p $S/r69_wan_correct && rsync -a --size-only /scratch/PERSO/skhairi/five_bench/r69_wan_correct/ $S/r69_wan_correct/
~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py r69
```

### eval-core
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part core --per_job 1
```

### eval-niqe
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part niqe --per_job 1
```

### eval-motion
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part motion --per_job 1
```

### eval-cf
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r69.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r69 --layout edit_video --src_window render --part cf --per_job 1
```

### wait-eval
```bash
squeue -u $USER -h -o '%i %k' | grep 'eval_suite|r69|'
grep -h 'failures' logs/eval_suite_*_r69*.out | sort | uniq -c
ls -l evaluation/csv/suite_r69_wan_matched59_s0p5_correct*.csv
```

### verdict
```bash
column -s, -t < evaluation/csv/suite_r69_wan_matched59_s0p5_correct_overall.csv
column -s, -t < evaluation/csv/suite_r58_wan_max_s5_correct_overall.csv
```

## Pipeline

```mermaid
flowchart LR
  subgraph jz["Jean Zay H100"]
    A[r43_anchor_manifest_correct.json + five_bench_correct] --> R[r69_wan_render.sh: r49_wan_render.py matched59_s0.5]
    R --> O1[$D/r69_wan_arms/matched59_s0.5_full]
  end
  O1 --> F[jz/r69_fetch_renders.sh] --> O2[lab /scratch r69_wan_correct/matched59_s0.5]
  O2 --> S[store-copy + check_store_renders.py r69] --> O3[/store jz_five_bench/r69_wan_correct]
  O2 --> E[eval_suite.sh r69.txt: core, niqe, motion, cf]
  E --> T[suite_r69_wan_matched59_s0p5_correct tables]
  T --> V[verdict: bracket check vs max_s5]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r49_wan_render.py` | `MATCHED59 = (0.95, 0.92, 0.85, 0.978, 0.5)` with an R69 comment; nothing else changes (existing configs keep their indices) |
| `slurm_scripts/five_bench/r69_wan_render.sh` (new) | Modelled on `r62_wan_render.sh`: `CONFIG=matched59_s0.5`, `CASES=evaluation/r43_anchor_manifest_correct.json`, `ANNO=evaluation/five_bench_correct`, `OUT=$D/r69_wan_arms` (guard: only this root), no `--field_root`; pre-flight files + `resolve()` = (354, 295); `r49_wan_render.py --config $CONFIG --mode full --cases $CASES --anno_dir $ANNO --out_root $OUT`; post-run guard 415 dirs × `expected_frames` PNGs; `--time=20:00:00`, lab header `--exclude=node52`, ported by `r50_port.sh`. Check at build time that `r48_wan_common.load_cases` reads the r43 manifest format (R58 fed it r7-format case files) |
| `jz/r50_port.sh` | Add `r69_wan_render` to `SCRIPTS` |
| `jz/r50_scratch_links.sh` | Add `r69_wan_arms` to `FOLDERS` and the header comment |
| `jz/r69_fetch_renders.sh` (new) | Copy of `jz/r62_fetch_renders.sh` for one root, no assembly CSV: `zay:$SCRATCH/five_bench_out/r69_wan_arms/matched59_s0.5_full/` → `$R69_DEST/matched59_s0.5/`, default `R69_DEST=/scratch/PERSO/skhairi/five_bench/r69_wan_correct`; post-check 415 dirs with `expected_frames` PNGs |
| `evaluation/check_store_renders.py` | `TASKS += ("r69",)`, `JZ_ROOTS += ("r69_wan_correct/",)` |
| `evaluation/eval_suite/batches/r69.txt` (new) | `wan_matched59_s0p5_correct  r69_wan_correct/matched59_s0.5` |
