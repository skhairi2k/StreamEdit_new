---
name: "R71 — Wan-Edit Normalized Mapping Rerun"
overview: "Wan-Edit's ControlLift mapping normalized over the operating range, as StreamEdit's: m = round(49·q), s = 50 − m (m_max = 49, s_min = 1), replacing R58's min(round(50·q), 45). New start maps are remapped on CPU from the q stored in the existing fields (no new draft, attention or divergence; a gate proves each source q reproduces its s_min 5 start map). 11 configs rendered on all 415 FiVE-Bench-Correct pairs on Jean Zay H100 (array of 11): adaptive A0 max_s1, ablations A1 nodiv_s1 / A2 dino_s1 / A3 lpips_s1 / A4 nosupp_s1, the equal-NFE matched-63 curve (σ 0.996, 0.978, 0.95, 0.92, 0.85, 0.5); R58's native N = 50 renders are reused as they are (no new native point). Fetched to lab /scratch, archived on /store, scored with the full R57 suite on the lab, read as the Wan-Edit main-table rows, Pareto curves and the A0–A4 ablation table."
task_id: R71
todos:
  - id: remap-script
    content: "evaluation/r71_remap_field.py -- per arm in {max, nosupp, nodiv, dino, lpips} and per pair of evaluation/r43_anchor_manifest_correct.json, reads the s_min 5 field npz from its source root (max: $D/r58_field/wan_max_s5 for the 190 pairs of r58_cases_maxs5.json, $D/r49_field_full/wan_max_s5 for the 225 of r68_cases_reuse225.json; nosupp: $D/r62_field/wan_nosupp_s5 (190) + $D/r51_field_full/wan_nosupp_s5 (225); nodiv/dino/lpips: $D/r68_field/wan_{arm}_s5 (415)); GATE: stored start_step == clip(50 − round(50·q), 5, 50) on every token, else nothing is written for that arm and exit 1; writes $D/r71_field/wan_{arm}_s1/edit{T}/{video}.npz (every source key copied, start_step = 50 − round(49·clip(q, 0, 1)) int64, s_min 1, m_max 49, arm '{arm}_s1', mapping 'inverse_exposure_normalized') + the .npy start map (same dtype/shape as the source .npy); --selftest (q = 0 → s 50, q = 1 → s 1, q = 0.5 → s 25 or 26, monotone); --roots/--out_root overridable; prints per-arm counts and the mean start; never writes outside --out_root"
    needed_by: [wan-chain]
    status: completed
    completed_at: 2026-11-11
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r71_field.sh -- CPU job (modelled on r68_field.sh, --cpus-per-task=4): preflight the 7 source roots hold >= 190 / 225 / 415 npz as listed in remap-script and both case files exist; r71_remap_field.py --selftest, then r71_remap_field.py --out_root $D/r71_field; post-run guard: 415 npz + 415 .npy in each of wan_{max,nosupp,nodiv,dino,lpips}_s1; ends '[r71_field] failures: N'"
    needed_by: [jz-sync, wan-chain]
    status: completed
    completed_at: 2026-11-11
  - id: render-configs
    content: "evaluation/r49_wan_render.py -- ADAPT += max_s1, nosupp_s1, nodiv_s1, dino_s1, lpips_s1 (loop start 1, field dir wan_{config}); A_ACTIVE_63 = 63 (14 draft + 49 final) and MATCHED63 = (0.996, 0.978, 0.95, 0.92, 0.85, 0.5) built like matched59_configs (asserts N − skip = 63 and |real start σ − σ_g| < 0.015; expected N/skip 64/1, 70/7, 80/17, 90/27, 119/56, 378/315); resolve() accepts matched63_s*; docstring updated; existing configs unchanged"
    needed_by: [wan-chain]
    status: completed
    completed_at: 2026-11-11
  - id: render-sbatch
    content: "slurm_scripts/five_bench/r71_wan_render.sh -- built from r68_wan_render.sh, --array=0-10, --time=20:00:00, CONFIGS = (max_s1 nosupp_s1 nodiv_s1 dino_s1 lpips_s1 matched63_s0.996 matched63_s0.978 matched63_s0.95 matched63_s0.92 matched63_s0.85 matched63_s0.5); tasks 0-4 preflight 415 .npy under $D/r71_field/wan_${CONFIG} and run r49_wan_render.py --field_root $D/r71_field; 5-10 r49_wan_render.py (global start); all --mode full --cases evaluation/r43_anchor_manifest_correct.json --anno_dir evaluation/five_bench_correct --out_root $D/r71_wan_arms; refuses any OUT other than $D/r71_wan_arms; resumable; r48_units.py --missing post-run guard (415 complete clips); lab header with --exclude=node52 (ported by r50_port.sh)"
    needed_by: [jz-sync, wan-chain]
    status: completed
    completed_at: 2026-11-11
  - id: chain-driver
    content: "slurm_scripts/five_bench/r71_wan_chain.sh -- Jean Zay driver modelled on r68_wan_chain.sh: guards ($WORK, ported r71_* scripts, $SCRATCH links for r71_field / r71_wan_arms, case files, the 7 source field roots of remap-script present — if any R49/R51 root was purged from $SCRATCH, FATAL with the /store path to push back from), --dry-run, refuses if logs/r71_wan_chain_jobs.txt exists; submits field (r71_field.sh) -> render --array=0-4 (afterok field) and render --array=5-10 (no dependency); writes '<stage> <id>' lines headed by the git commit"
    needed_by: [jz-sync, wan-chain]
    status: completed
    completed_at: 2026-11-11
  - id: jz-port-list
    content: "jz/r50_port.sh -- add r71_field, r71_wan_render, r71_wan_chain to SCRIPTS"
    needed_by: [jz-sync]
    status: completed
    completed_at: 2026-11-11
  - id: jz-scratch-list
    content: "jz/r50_scratch_links.sh -- add r71_field, r71_wan_arms to FOLDERS (and the header comment)"
    needed_by: [jz-sync]
    status: completed
    completed_at: 2026-11-11
  - id: jz-fetch-list
    content: "jz/r50_fetch_outputs.sh -- add r71_field to ALL (the s_min 1 start maps behind every adaptive render; renders travel by r71_fetch_renders.sh)"
    needed_by: [store-copy]
    status: pending
  - id: fetch-renders-script
    content: "jz/r71_fetch_renders.sh -- clone of jz/r69_fetch_renders.sh for the 11 configs (or those given as arguments): rsync -rL --partial --size-only --no-perms --no-owner --no-group (no -t) zay:$SCRATCH/five_bench_out/r71_wan_arms/{cfg}_full/ -> $R71_DEST/{cfg}/ (default R71_DEST=/scratch/PERSO/skhairi/five_bench/r71_wan_correct; refuses any destination outside /scratch); preflight >= 12 GB free per requested config; post-check per config 415 clip dirs + r48_units.py --missing on r43_anchor_manifest_correct.json; --check counts only; never deletes on either side; ends '[r71_fetch] failures: N'"
    needed_by: [fetch-renders]
    status: pending
  - id: store-check-entry
    content: "evaluation/check_store_renders.py -- add r71 to TASKS and r71_wan_correct/ to JZ_ROOTS (store path /store/PERSO/skhairi/Outputs/jz_five_bench/r71_wan_correct)"
    needed_by: [store-copy]
    status: pending
  - id: eval-batch
    content: "evaluation/eval_suite/batches/r71.txt -- 11 lines `NAME ROOT` (R57 batch format, ROOT relative to --root_base /scratch/PERSO/skhairi/five_bench): wan_max_s1_correct r71_wan_correct/max_s1; wan_nosupp_s1_correct, wan_nodiv_s1_correct, wan_dino_s1_correct, wan_lpips_s1_correct; wan_matched63_s0p996_correct, ..._s0p978, _s0p95, _s0p92, _s0p85, _s0p5"
    needed_by: [eval-core, eval-niqe, eval-motion, eval-cf]
    status: pending
  - id: score-script
    content: "evaluation/r71_score.py -- reads only the R57 tables (suite_<tag>_<NAME>.csv / _per_type.csv / _overall.csv), never recomputes a mean; exit 1 if any table is missing or has != 415 rows. Writes evaluation/csv/r71_wan_points.csv (config, role arm|ablation|m63|native, plain per-clip mean overall + per edit type, every suite metric incl. motion fidelity and PSC): arm max_s1; ablations nosupp/nodiv/dino/lpips_s1; m63 curve (6 R71 configs); native curve = R58's 7 (suite_r58_wan_native_*_correct), not re-rendered; reference rows R58 max_s5 and matched59_s0.92 (old mapping, for the verdict only). evaluation/csv/r71_wan_deltas.csv (max_s1 − matched63_s0.92 / native_s0.92 / native_s0.996 / R58 max_s5, plain means, overall + per type); evaluation/csv/r71_wan_ablation.csv (A0–A4 at s_min 1, same metrics); evaluation/figures/r71_pareto/{alignment}_vs_{preservation}.pdf (x = CLIP-T, FiVE-Acc, CLIP-D, EditReward, PSC; y = bg PSNR, bg LPIPS, CF, motion fidelity; native curve, matched-63 curve, max_s1 point, official native_s0.92 marked); no bootstrap"
    needed_by: [score]
    status: pending
steps:
  - id: jz-sync
    type: manual
    check_hint: "User: commit + push. On Jean Zay in the clone: `git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh`; paste the tails '[r50_port] leftovers: 0', '[r50_port] failures: 0', '[r50_scratch] failures: 0'; `ls -ld $D/r71_field $D/r71_wan_arms` are symlinks into $SCRATCH"
    status: pending
  - id: wan-chain
    type: manual
    wait_for: jz-sync
    check_hint: "User, on Jean Zay: `bash slurm_scripts/five_bench/r71_wan_chain.sh --dry-run` (every guard passes; push back any purged R49/R51 field root from /store first), then `bash slurm_scripts/five_bench/r71_wan_chain.sh`; paste logs/r71_wan_chain_jobs.txt (field, render_adapt, render_global)"
    sets_status: running
    status: pending
  - id: wait-chain
    type: manual
    wait_for: wan-chain
    check_hint: "User, on Jean Zay, when `squeue -u $USER` is empty: field log has 'gate PASS' for all 5 arms and 415 npz + 415 .npy per wan_*_s1 dir; `grep -h failures logs/r71_*.out | sort | uniq -c` -> every line 'failures: 0'; each of the 11 render logs reports 415/415 clips. A task that hit the 20 h limit is resubmitted alone (`sbatch --array=<i> slurm_scripts/five_bench/r71_wan_render.sh`; finished clips are skipped)"
    sets_status: finished
    status: pending
  - id: fetch-renders
    type: manual
    wait_for: wait-chain
    check_hint: "User, on the lab frontal gpu-gw in tmux: `myquota` leaves >= 100 GB; `bash jz/r71_fetch_renders.sh --check` then `bash jz/r71_fetch_renders.sh` -> '[r71_fetch] failures: 0' (11 configs × 415 clip dirs, ~100 GB)"
    status: pending
  - id: store-copy
    type: manual
    wait_for: fetch-renders
    check_hint: "User, on gpu-gw in tmux (/store is frontal-only): `S=/store/PERSO/skhairi/Outputs/jz_five_bench; [ -d $S ] || { echo '/store not mounted here'; exit 1; }; mkdir -p $S/r71_wan_correct && rsync -a --size-only /scratch/PERSO/skhairi/five_bench/r71_wan_correct/ $S/r71_wan_correct/`, then `~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py r71` -> '[check_store] failures: 0'; `bash jz/r50_fetch_outputs.sh r71_field` -> 'failures: 0'. The Jean Zay copies may be deleted only after this passes and the scores are in"
    status: pending
  - id: eval-core
    type: local
    wait_for: fetch-renders
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part core --per_job 2
    status: pending
  - id: eval-niqe
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part niqe --per_job 4
    status: pending
  - id: eval-motion
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part motion --per_job 12
    status: pending
  - id: eval-cf
    type: local
    wait_for: eval-core
    command: |
      bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part cf --per_job 5
    status: pending
  - id: wait-eval
    type: manual
    wait_for: [eval-core, eval-niqe, eval-motion, eval-cf]
    check_hint: "every eval_suite job of this task ended (nothing left in `squeue -u $USER -h -o '%i %k' | grep 'eval_suite|r71|'`), logs end 'failures: 0' (a job stopped at 24 h is relaunched with the same command: it resumes); 11 sets evaluation/csv/suite_r71_<NAME>.csv + _per_type.csv + _overall.csv, 415 rows each, niqe_target_image, motion_fidelity_score and cf filled; every NaN cell listed by the merge logs recorded in the Run log; scored before the 30-day /scratch purge (else re-stage from /store)"
    sets_status: finished
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
    command: python evaluation/r71_score.py
    output_paths:
      - evaluation/csv/r71_wan_points.csv
      - evaluation/csv/r71_wan_deltas.csv
      - evaluation/csv/r71_wan_ablation.csv
      - evaluation/figures/r71_pareto/
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "Read r71_wan_deltas.csv and the Pareto figures: max_s1 vs matched63_s0.92 / native_s0.92 (main-table claim, incl. motion fidelity raw + at matched FiVE-Acc, and PSC), max_s1 vs R58 max_s5 (how much the mapping change moved the results), A0–A4 ordering in r71_wan_ablation.csv vs R62/R68's s_min 5 ordering; record in the Run log; the user decides which Wan-Edit rows enter the paper"
    sets_status: analyzed
    status: pending
isProject: true
---

# R71: Wan-Edit Normalized Mapping Rerun

## Context

The paper's ControlLift normalizes each host's exposure over its operating range (Eq. normalized_edit_exposure). StreamEdit's code does this (ρ ∈ [2, 50]), but every Wan-Edit result so far (R49/R58 `max_s5`, R62 A4, R68 A1–A3) used `min(round(50·q), 45)`: normalized over the full 50-step gate, then capped at s = 5. R71 re-runs Wan-Edit with the mapping the paper states, `m = round(m_max·q)`, `s = 50 − m`, at `m_max = 49` (s_min = 1, start σ 0.9957), together with the baselines whose budget depends on it (the equal-NFE curve: 14 draft + 49 = 63 active steps) and the Wan-Edit demand ablations.

Done when: the 11 configs are rendered on all 415 FiVE-Bench-Correct pairs, copied to lab /scratch and /store, scored with the four R57 parts, `r71_score.py` has written the points, deltas, ablation table and Pareto figures, and the verdict is in the Run log.

## Execution steps

| # | Step id | Type | Runs on | Waits for | Sets status |
|---|---|---|---|---|---|
| 1 | *(prep)* `remap-script`, `field-sbatch`, `render-configs`, `render-sbatch`, `chain-driver`, `jz-port-list`, `jz-scratch-list`, `jz-fetch-list`, `fetch-renders-script`, `store-check-entry`, `eval-batch`, `score-script` | /build-step | lab (code) | — | `implemented` |
| 2 | `jz-sync` | manual | Jean Zay | — | — |
| 3 | `wan-chain` | manual (driver: field + 2 render arrays) | **Jean Zay** CPU + H100 | jz-sync | `running` |
| 4 | `wait-chain` | manual | Jean Zay | wan-chain | `finished` |
| 5 | `fetch-renders` | manual (tmux) | gpu-gw | wait-chain | — |
| 6 | `store-copy` | manual (tmux) | gpu-gw | fetch-renders | — |
| 7 | `eval-core` | local (eval_suite `--part core`) | **lab L40S** | fetch-renders | — |
| 8 | `eval-niqe` | local (`--part niqe`) | lab L40S | eval-core | — |
| 9 | `eval-motion` | local (`--part motion`) | lab L40S | eval-core | — |
| 10 | `eval-cf` | local (`--part cf`) | **lab H100** | eval-core | — |
| 11 | `wait-eval` | manual | lab | eval-core, eval-niqe, eval-motion, eval-cf | `finished` |
| 12 | `score` | local | lab | wait-eval | — |
| 13 | `verdict` | manual | lab | score | `analyzed` |

```
/build-step wan-chain R71
/build-step jz-sync R71
/build-step fetch-renders R71
/build-step store-copy R71
/build-step eval-core R71
/build-step score R71
/run-step R71 jz-sync
/run-step R71 wan-chain
/run-step R71 wait-chain
/run-step R71 fetch-renders
/run-step R71 store-copy
/run-step R71 eval-core
/run-step R71 eval-niqe
/run-step R71 eval-motion
/run-step R71 eval-cf
/run-step R71 wait-eval
/run-step R71 score
/run-step R71 verdict
```

## Decisions

| Item | Choice |
|---|---|
| Mapping | `m_i = round(49·q_i)` (numpy, half to even), `s_i = 50 − m_i`; q = 0 → s = 50 (never edited), q = 1 → s = 1; operating range m ∈ {0, …, 49}, normalized exposure m/49 (user, 2026-11-11: same logic as StreamEdit) |
| Fields | Remapped on CPU from the `q` stored in the existing s_min 5 fields (user, 2026-11-11); no new draft, attention or divergence; gate = stored start_step equals `clip(50 − round(50·q), 5, 50)` on every token |
| q sources | A0 max: R58 field (190 changed pairs) + R49 full field (225 reused pairs); A4 nosupp: R62 (190) + R51 full (225); A1–A3 nodiv / dino / lpips: R68 (415) |
| Adaptive configs | `max_s1` (A0), `nosupp_s1` (A4), `nodiv_s1` (A1), `dino_s1` (A2), `lpips_s1` (A3): N = 50, loop start 1, per-token start map |
| Equal-NFE curve | matched-63 = 14 draft + 49 final active steps; σ_g 0.996 / 0.978 / 0.95 / 0.92 / 0.85 / 0.5 → N/skip 64/1, 70/7, 80/17, 90/27, 119/56, 378/315 (r48_wan_sched.matched_n, checked 2026-11-11); replaces matched-59 (R58 + R69) |
| Native curve | R58's 7 renders reused as they are (N = 50 global starts do not depend on the mapping); no new native point (user, 2026-11-11: no time to run it) |
| Pairs | All 415 FiVE-Bench-Correct pairs (`evaluation/r43_anchor_manifest_correct.json`, annotations `evaluation/five_bench_correct/`); every config rendered fresh, no reuse |
| Renderer settings | as R58: text only, seeded noise base_seed 0, n_avg 1, edit.py guide scales, up to 41 frames 832×480 |
| Render jobs | Jean Zay H100, one array of 11 (one config per task), 20 h each, resumable; ~11–12 h per task estimated (R68: ~10–11 h for 415 clips at 59 steps) |
| Paths | Jean Zay `$D/r71_field/wan_{arm}_s1/`, `$D/r71_wan_arms/{config}_full/` (→ `$SCRATCH`); lab `/scratch/PERSO/skhairi/five_bench/r71_wan_correct/{config}/`; archive `/store/PERSO/skhairi/Outputs/jz_five_bench/r71_wan_correct/` |
| No overwrite | New outputs only under `r71_field` / `r71_wan_arms` / `r71_wan_correct`; R49/R51/R58/R62/R68/R69 folders only read |
| Evaluation | Full R57 suite on the lab, all four parts (core, niqe, motion, cf), batch `r71.txt`, tag `r71`, `--layout edit_video --src_window render`; per_job as R58 (2 / 4 / 12 / 5) |
| Analysis | Plain per-clip means from the suite tables; x = alignment, y = preservation; motion fidelity and PSC in every read; no bootstrap |
| R68 / R69 | Their Wan renders are done; their scoring is independent of R71 and not part of this task |
| Out of scope | StreamEdit (its mapping already follows the rule), re-rendering R58's native configs, the paper text (updated after the verdict) |

## Step commands

### jz-sync
```bash
# lab: commit + push; then on Jean Zay, in the clone:
git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh
ls -ld $D/r71_field $D/r71_wan_arms
```

### wan-chain
```bash
# on Jean Zay
bash slurm_scripts/five_bench/r71_wan_chain.sh --dry-run
bash slurm_scripts/five_bench/r71_wan_chain.sh
cat logs/r71_wan_chain_jobs.txt
```

### wait-chain
```bash
# on Jean Zay
squeue -u $USER
grep -h 'gate PASS\|failures' logs/r71_field_*.out
grep -h failures logs/r71_*.out | sort | uniq -c
grep -h 'clips$' logs/r71_wan_render_*.out
# timeout on task i: sbatch --array=<i> slurm_scripts/five_bench/r71_wan_render.sh
```

### fetch-renders
```bash
# lab frontal gpu-gw, in tmux
myquota
bash jz/r71_fetch_renders.sh --check
bash jz/r71_fetch_renders.sh
```

### store-copy
```bash
# lab frontal gpu-gw, in tmux
S=/store/PERSO/skhairi/Outputs/jz_five_bench; [ -d $S ] || { echo '/store not mounted here'; exit 1; }
mkdir -p $S/r71_wan_correct && rsync -a --size-only /scratch/PERSO/skhairi/five_bench/r71_wan_correct/ $S/r71_wan_correct/
~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py r71
bash jz/r50_fetch_outputs.sh r71_field
```

### eval-core
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part core --per_job 2
```

### eval-niqe
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part niqe --per_job 4
```

### eval-motion
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part motion --per_job 12
```

### eval-cf
```bash
bash slurm_scripts/five_bench/eval_suite.sh --batch evaluation/eval_suite/batches/r71.txt --root_base /scratch/PERSO/skhairi/five_bench --tag r71 --layout edit_video --src_window render --part cf --per_job 5
```

### wait-eval
```bash
squeue -u $USER -h -o '%i %k' | grep 'eval_suite|r71|'
grep -h 'failures' logs/eval_suite_*_r71*.out | sort | uniq -c
ls evaluation/csv/suite_r71_*_overall.csv | wc -l   # 11
```

### score
```bash
python evaluation/r71_score.py
```

### verdict
```bash
column -s, -t < evaluation/csv/r71_wan_deltas.csv
column -s, -t < evaluation/csv/r71_wan_ablation.csv
ls evaluation/figures/r71_pareto/
```

## Pipeline

```mermaid
flowchart LR
  subgraph jz["Jean Zay"]
    Q[s_min 5 fields: r58_field + r49_field_full / r62_field + r51_field_full / r68_field] --> M[r71_field.sh: r71_remap_field.py, gate + round 49 q]
    M --> F[$D/r71_field/wan_*_s1]
    F --> RA[r71_wan_render.sh tasks 0-4: max/nosupp/nodiv/dino/lpips_s1]
    A[r43_anchor_manifest_correct.json + five_bench_correct] --> RA
    A --> RG[r71_wan_render.sh tasks 5-10: matched63 x6]
    RA --> O1[$D/r71_wan_arms]
    RG --> O1
  end
  O1 --> FR[jz/r71_fetch_renders.sh] --> O2[lab /scratch r71_wan_correct]
  O2 --> S[store-copy + check_store_renders.py r71] --> O3[/store jz_five_bench/r71_wan_correct]
  O2 --> E[eval_suite.sh r71.txt: core, niqe, motion, cf]
  E --> T[suite_r71_* tables]
  R58[suite_r58 native curve + max_s5] --> SC
  T --> SC[r71_score.py] --> P[r71_wan_points / deltas / ablation + r71_pareto]
  P --> V[verdict]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r71_remap_field.py` | **new** — gate + remap of stored q to `s = 50 − round(49·q)` for 5 arms × 415 pairs; writes npz + .npy under `--out_root`; `--selftest` |
| `slurm_scripts/five_bench/r71_field.sh` | **new** — CPU job: preflight source roots, selftest, remap, 415 + 415 per arm guard |
| `evaluation/r49_wan_render.py` | `ADAPT` += five `*_s1` arms (loop start 1); `A_ACTIVE_63 = 63`, `MATCHED63 = (0.996, 0.978, 0.95, 0.92, 0.85, 0.5)`, `matched63_configs()`; `resolve()` accepts them |
| `slurm_scripts/five_bench/r71_wan_render.sh` | **new** — array 0-10 via `r49_wan_render.py` (tasks 0-4 adaptive, 5-10 matched-63); OUT guard `$D/r71_wan_arms`; 415-clip post guard |
| `slurm_scripts/five_bench/r71_wan_chain.sh` | **new** — Jean Zay driver: field → render 0-4 (afterok), render 5-10 (no dependency); `--dry-run`; jobs file |
| `jz/r50_port.sh` | SCRIPTS += r71_field, r71_wan_render, r71_wan_chain |
| `jz/r50_scratch_links.sh` | FOLDERS += r71_field, r71_wan_arms |
| `jz/r50_fetch_outputs.sh` | ALL += r71_field |
| `jz/r71_fetch_renders.sh` | **new** — clone of r69_fetch_renders.sh for the 11 configs → `/scratch/PERSO/skhairi/five_bench/r71_wan_correct/` |
| `evaluation/check_store_renders.py` | TASKS += r71; JZ_ROOTS += `r71_wan_correct/` |
| `evaluation/eval_suite/batches/r71.txt` | **new** — 11 `NAME ROOT` lines |
| `evaluation/r71_score.py` | **new** — points / deltas / ablation CSVs + Pareto figures from the R57 tables (R71 + R58 native curve and max_s5 reference) |

Remap pseudocode (`r71_remap_field.py`):
```python
start5 = np.clip(50 - np.round(50 * q), 5, 50)       # gate: must equal the stored start_step
assert (start5 == src["start_step"]).all()
start1 = 50 - np.round(49 * np.clip(q, 0, 1)).astype(np.int64)
```
