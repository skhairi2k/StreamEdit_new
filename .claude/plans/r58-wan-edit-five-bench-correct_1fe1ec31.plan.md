---
name: "R58 — Wan-Edit on FiVE-Bench-Correct (Jean Zay H100)"
overview: "Runs on JEAN ZAY (H100, R50 five-bench overlay): every render, the reproducibility gate, the assembly and the evaluation; the lab only builds the case lists and scores the fetched CSVs. Wan-Edit's three R49 arms on the 415 FiVE-Bench-Correct pairs: native sweep (7 global starts, N=50, native_s0.92 = official Wan-Edit), equal-NFE matched-59 curve (4 starts) and the adaptive max_s5. Only pairs whose Wan-Edit inputs changed are re-generated: 47 pairs (source/target prompt changed) for the 11 baseline configs, 190 pairs (prompt or trigger object changed) for max_s5 (draft + attention + divergence + field + render). Every other clip reuses the R49 full-stage H100 render, after a 2-pair reproducibility gate. No existing video is ever written: new renders go to new r58_* folders only. Per config, one 415-pair folder of symlinks (new render or R49 render per clip) is the single root the evaluation reads; it is scored with R57's unified suite (Correct annotations: new FiVE-Acc questions, EditReward, PSC, CF), then read as Pareto curves + paired deltas."
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
  - id: eval-jz-launcher
    content: "jz/r58_eval_jz.sh -- loops jz/eval_suite_jz.sh (R57) over the 12 configs (or the configs given as arguments): --method wan_{config with . -> p}_correct --root $D/r58_wan_correct/{config} --layout edit_video --tag r58 (default --edit_json_dir = Correct, no --cases = 415 pairs), --qos from the command line; prints the submitted job ids per config to logs/r58_eval_jobs.txt; refuses a config whose suite CSV already exists"
    needed_by: [eval]
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
    content: "jz/r50_fetch_outputs.sh -- add r58_wan_draft r58_div r58_field r58_wan_arms to ALL (and the header comment); r58_wan_correct is links only and is not fetched"
    needed_by: [fetch]
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
    status: pending
  - id: repro-check
    type: manual
    wait_for: repro-render
    check_hint: "User, on Jean Zay: `$WORK/pyenv/five-bench/python evaluation/r58_repro_check.py`; must print PASS (max |diff| <= 2 on every frame of both pairs, both renderers). FAIL = the R49 renders cannot be mixed with new ones: stop, bring evaluation/csv/r58_repro_check.csv back, decide with the user (full 415 re-render)"
    status: pending
  - id: wan-chain
    type: manual
    wait_for: repro-check
    check_hint: "User, on Jean Zay: first `git pull && bash jz/r50_port.sh` (the 4 other r58_*.sh scripts were not in the first sync and must now be ported), then `bash slurm_scripts/five_bench/r58_wan_chain.sh --dry-run` then `bash slurm_scripts/five_bench/r58_wan_chain.sh`; paste logs/r58_wan_chain_jobs.txt (5 stages: draft, div, field, render_max_s5, render_base)"
    sets_status: running
    status: pending
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
  - id: eval
    type: manual
    wait_for: assemble
    check_hint: "Needs R57 `adopt` done (jz/eval_suite_jz.sh validated on Jean Zay). User, on Jean Zay: `bash jz/r58_eval_jz.sh --qos qos_gpu_h100-t3`; paste logs/r58_eval_jobs.txt (3 jobs per config: metrics, cf, merge)"
    status: pending
  - id: wait-eval
    type: manual
    wait_for: eval
    check_hint: "User, on Jean Zay: 12 files evaluation/csv/suite_r58_wan_*_correct.csv, 415 rows each, every metric non-NaN (merge logs exit 0); `ls evaluation/csv/suite_r58_wan_*_correct.csv | wc -l` = 12"
    status: pending
  - id: fetch
    type: manual
    wait_for: wait-eval
    check_hint: "User, on the lab frontal in tmux: `rsync -av zay:/lustre/fswork/projects/rech/omt/upx85tg/StreamEdit_new/evaluation/csv/'suite_r58_*' zay:/lustre/fswork/projects/rech/omt/upx85tg/StreamEdit_new/evaluation/csv/'r58_*' evaluation/csv/` and `bash jz/r50_fetch_outputs.sh r58_wan_draft r58_div r58_field r58_wan_arms` -> '[r50_fetch] failures: 0'"
    status: pending
  - id: score
    type: local
    wait_for: fetch
    command: python evaluation/r58_score.py
    output_paths:
      - evaluation/csv/r58_wan_points.csv
      - evaluation/csv/r58_wan_deltas.csv
      - evaluation/figures/r58_pareto/
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

**Platform: Jean Zay.** Every render, the gate, the assembly and the evaluation run on Jean Zay H100 (`ssh zay`, run by the user and pasted back), next to the R49 Wan renders they reuse; the lab only builds the case lists (`build-cases`) and scores the fetched CSVs (`score`).

**Done when:** 12 configs × 415 pairs are scored with R57's suite on Jean Zay, and `r58_wan_points.csv` / `r58_wan_deltas.csv` / `r58_pareto/` are read.

## Execution steps

| # | step id | type | runs on | waits for | sets status |
|---|---|---|---|---|---|
| — | *(prep)* anno-dir-common · anno-dir-draft · anno-dir-render48 · anno-dir-render49 · cases-script · render-sbatch · draft-sbatch · div-sbatch · field-sbatch · chain-driver · repro-check-script · assemble-script · eval-jz-launcher · port-list · scratch-folders · fetch-folders · score-script | todos | lab (code) | — | `implemented` |
| 1 | `build-cases` | local | lab | — | — |
| 2 | `jz-sync` | manual (user: push on lab, then pull + port + scratch links) | lab → **Jean Zay** | build-cases | — |
| 3 | `repro-render` | manual (user, dev qos) | **Jean Zay** | jz-sync | — |
| 4 | `repro-check` | manual (user) | **Jean Zay** | repro-render | — |
| 5 | `wan-chain` | manual (user, chain driver) | **Jean Zay** | repro-check | `running` |
| 6 | `wait-chain` | manual | **Jean Zay** | wan-chain | `finished` |
| 7 | `assemble` | manual (user) | **Jean Zay** | wait-chain | — |
| 8 | `eval` | manual (user; needs R57 `adopt`) | **Jean Zay** | assemble | — |
| 9 | `wait-eval` | manual | **Jean Zay** | eval | — |
| 10 | `fetch` | manual (user) | lab frontal | wait-eval | — |
| 11 | `score` | local | lab | fetch | — |
| 12 | `analyze` | manual | lab | score | `analyzed` |

```
/build-step build-cases R58
/build-step repro-render R58
/build-step wan-chain R58
/build-step assemble R58
/build-step eval R58
/build-step score R58
/run-step R58 build-cases
/run-step R58 jz-sync
/run-step R58 repro-render
/run-step R58 repro-check
/run-step R58 wan-chain
/run-step R58 wait-chain
/run-step R58 assemble
/run-step R58 eval
/run-step R58 wait-eval
/run-step R58 fetch
/run-step R58 score
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
| Platform | **Jean Zay H100**, R50 `five-bench` overlay, for every render, the gate, the assembly and every eval (reused R49 renders are H100 too); never mixed with L40S numbers. Lab: case lists and scoring only |
| No overwrite | No existing video is written, moved or deleted. New renders only in new folders (`r58_wan_arms`, `r58_repro`; the render script refuses any other output root); the renderers skip clips that already have their PNGs; `r58_assemble.py` only creates symlinks inside `r58_wan_correct` and stops on any conflict; `wait-chain` checks that no file in `r48_wan_arms` / `r49_wan_arms` is newer than the chain submission |
| Reuse gate | Before mixing new and reused renders: 2 unchanged pairs re-rendered with `native_s0.92` (r48 renderer) and `matched59_s0.92` (r49 renderer); PASS iff max \|diff\| ≤ 2/255 on every frame. FAIL stops the task (decision with the user: full re-render) |
| Output dirs | New renders under `$D/r58_wan_arms/{config}_full/` (separate root, so the resumable renderers cannot skip a changed pair because its official render exists); draft `$D/r58_wan_draft`, divergence `$D/r58_div`, field `$D/r58_field`, repro `$D/r58_repro`; all on `$SCRATCH` via `r50_scratch_links.sh` |
| Assembly (two folders → one root) | Per config, `$D/r58_wan_correct/{config}/edit{T}/{video}` is a symlink to the clip folder: the new render (`r58_wan_arms`) if the pair is in the regen set, else the R49 render (`r48_wan_arms` / `r49_wan_arms`). The eval gets `--root $D/r58_wan_correct/{config} --layout edit_video` and sees one ordinary 415-clip tree (the FiVE harness lists folders with `os.listdir` / `glob`, which follow symlinks; no `os.walk`). No copy, so the R49 renders must still be on `$SCRATCH` until `wait-eval` (IDRIS purge) |
| Evaluation | R57 unified suite on Jean Zay (`jz/eval_suite_jz.sh`), all 415 pairs per config, `--tag r58`, method name `wan_{config}_correct` (dot → `p`) → `evaluation/csv/suite_r58_wan_{config}_correct.csv`. Blocked until R57 is adopted |
| Averaging / plots | Plain per-clip mean from the suite CSV, never `_avg.csv`; Pareto plots with x = alignment, y = preservation |
| Readout | max_s5 vs matched-59 (equal NFE, headline), vs native_s0.92 (official) and native_s0.978 (same floor); paired bootstrap 95% CI; per edit type; regenerated vs reused split as a sanity check |
| Out of scope | StreamEdit arms on Correct (separate tasks); the R51 A4 Wan arm; re-rendering unchanged pairs |

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

### eval
```bash
# Jean Zay, after R57 adopt
bash jz/r58_eval_jz.sh --qos qos_gpu_h100-t3
```

### wait-eval
```bash
# Jean Zay
ls evaluation/csv/suite_r58_wan_*_correct.csv | wc -l   # 12
```

### fetch
```bash
# lab frontal (gpu-gw), in tmux
rsync -av "zay:/lustre/fswork/projects/rech/omt/upx85tg/StreamEdit_new/evaluation/csv/suite_r58_*" \
          "zay:/lustre/fswork/projects/rech/omt/upx85tg/StreamEdit_new/evaluation/csv/r58_*" evaluation/csv/
bash jz/r50_fetch_outputs.sh r58_wan_draft r58_div r58_field r58_wan_arms
```

### score
```bash
python evaluation/r58_score.py
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
  F --> G[r58_wan_render.sh repro] --> H[r58_repro_check.py]
  R49[R49 H100 renders on SCRATCH] --> H
  E --> I[r58_wan_draft.sh] --> J[r58_div.sh] --> K[r58_field.sh] --> L[r58_wan_render.sh max_s5]
  D --> M[r58_wan_render.sh 11 baselines]
  L --> N[r58_assemble.py]
  M --> N
  R49 --> N
  N --> O[r58_wan_correct 12 x 415]
  O --> P[jz/r58_eval_jz.sh R57 suite]
  C --> P
  P --> Q[suite_r58 CSVs]
  Q --> S[r58_score.py]
  S --> T[r58_wan_points / deltas / r58_pareto]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r48_wan_common.py` | `load_cases(cases_path, anno_dir=None)`: `anno_dir or FIVE / "edit_prompt"` |
| `evaluation/r48_wan_draft.py` | `--anno_dir` (default None) → `C.load_cases(a.cases, a.anno_dir)` |
| `evaluation/r48_wan_render.py` | `--anno_dir` (default None) → `C.load_cases` |
| `evaluation/r49_wan_render.py` | `--anno_dir` (default None) → `C.load_cases` |
| `evaluation/r58_build_cases.py` (new) | Regen sets from JSON comparison → 3 case files + `r58_regen_sets.csv`; count asserts 47 / 190 / 2 |
| `slurm_scripts/five_bench/r58_wan_render.sh` (new) | `correct` (array 0-11, config → renderer + case file) and `repro` (array 0-1) modes; all with `--anno_dir evaluation/five_bench_correct --mode full` |
| `slurm_scripts/five_bench/r58_wan_draft.sh` (new) | Clone of `r48_wan_draft.sh` on the 190 cases → `$D/r58_wan_draft` |
| `slurm_scripts/five_bench/r58_div.sh` (new) | Clone of `r48_div.sh` wan tasks (array 0-1) → `$D/r58_div` |
| `slurm_scripts/five_bench/r58_field.sh` (new) | `r49_field.py --hosts wan` on r58 roots → `$D/r58_field` |
| `slurm_scripts/five_bench/r58_wan_chain.sh` (new) | Clone of `r49_wan_full_chain.sh`: 5 submissions, afterok chain for max_s5, jobs file `logs/r58_wan_chain_jobs.txt`; refuses to run unless `r58_repro_check.csv` has every row `pass=1` and the R49 render dirs exist |
| `evaluation/r58_repro_check.py` (new) | Pixel comparison repro vs R49 render, PASS ≤ 2/255 → `r58_repro_check.csv` |
| `evaluation/r58_assemble.py` (new) | Symlink tree `$D/r58_wan_correct/{config}` over 415 pairs + `r58_assembly.csv` |
| `jz/r58_eval_jz.sh` (new) | Loop of `jz/eval_suite_jz.sh` over the 12 configs, tag `r58` |
| `jz/r50_port.sh` | SCRIPTS += the 5 `r58_*.sh` job scripts |
| `jz/r50_scratch_links.sh` | FOLDERS += `r58_wan_draft r58_div r58_field r58_wan_arms r58_repro r58_wan_correct` |
| `jz/r50_fetch_outputs.sh` | ALL += `r58_wan_draft r58_div r58_field r58_wan_arms` |
| `evaluation/r58_score.py` (new) | Points (overall + per type), deltas with paired bootstrap, Pareto PDFs |

`r58_wan_render.sh` index → (config, renderer, cases):

```
0-5  native_s0.95 s0.92 s0.85 s0.75 s0.6 s0.4   r48_wan_render.py   r58_cases_base.json
6    native_s0.978                               r48_wan_render.py   r58_cases_base.json
7-10 matched59_s0.95 s0.92 s0.85 s0.978          r49_wan_render.py   r58_cases_base.json
11   max_s5 (--field_root $D/r58_field)          r49_wan_render.py   r58_cases_maxs5.json
repro 0 native_s0.92 / 1 matched59_s0.92         r48 / r49           r58_cases_repro.json -> $D/r58_repro
```
