---
name: R62 — R51's Component Ablation on FiVE-Bench-Correct
overview: "R51's demand ablation re-run on the 415 FiVE-Bench-Correct pairs, mirroring R51: StreamEdit A1 nodiv / A2 dino / A3 lpips / A4 nosupp on the lab L40S, Wan-Edit A4 nosupp_s5 on Jean Zay H100. Only q changes; drafts, attention and divergence are R59's (StreamEdit) and R58's (Wan-Edit) Correct stage 1, A0 = R59 se_max / R58 max_s5, with R51's bit-exact A0 gate against their fields. Wan-Edit regenerates only R58's 190 changed pairs and links the other 225 to R51's A4 renders. Scored with the R57 unified suite; points table + trade-off figures on R59's / R58's existing curves."
task_id: R62
todos:
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r62_field.sh -- CPU job, HOST arg (se | wan): r51_field.py --selftest, then r51_field.py with the Correct roots (se: cases r43_anchor_manifest_correct.json, attn r59_se_attn, div r59_div, gate r59_field_full, arms nodiv dino lpips nosupp; wan: cases r58_cases_maxs5.json, attn r58_wan_draft/attn, div r58_div, gate r58_field, arm nosupp), out $D/r62_field; one-npz-per-case guard (+ .npy for wan)"
    needed_by: [se-chain, wan-chain]
    status: pending
  - id: se-render-sbatch
    content: "slurm_scripts/five_bench/r62_se_stage3.sh -- clone of r59_se_stage3.sh with --array=0-3 = nodiv, dino, lpips, nosupp: FIELD_DIR $D/r62_field/se_${ARM}, METHOD r62_${ARM}_full, OUT_ROOT $D/r62_se_arms; Correct edit JSONs, cases and anchors, flags, count / manifest guards and r59_check_prompts.py guard unchanged"
    needed_by: [se-chain]
    status: pending
  - id: se-chain-driver
    content: "slurm_scripts/five_bench/r62_se_chain.sh -- lab driver: reads 'field <id>' from logs/r59_se_chain_jobs.txt (fatal if absent); field se afterok that id if still in squeue, else only if sacct says COMPLETED (refuse otherwise); render --array=0-3 afterok field; queue-cap check (+5 tasks, cap 30) before submitting; logs/r62_se_chain_jobs.txt headed by the commit, refuses if it exists"
    needed_by: [se-chain]
    status: pending
  - id: wan-render-sbatch
    content: "slurm_scripts/five_bench/r62_wan_render.sh -- single job: preflight 190 .npy under $D/r62_field/wan_nosupp_s5; r49_wan_render.py --config nosupp_s5 --mode full --cases evaluation/r58_cases_maxs5.json --anno_dir evaluation/five_bench_correct --field_root $D/r62_field --out_root $D/r62_wan_arms; refuses any OUT other than $D/r62_wan_arms; r48_units.py --missing post-run guard; lab header with --exclude=node52 (ported by r50_port.sh)"
    needed_by: [wan-chain]
    status: pending
  - id: wan-chain-driver
    content: "slurm_scripts/five_bench/r62_wan_chain.sh -- Jean Zay driver modelled on r51_wan_chain.sh: guards ($WORK, ported scripts, $SCRATCH links, r58_cases_maxs5.json present); reads 'field <id>' from logs/r58_wan_chain_jobs.txt (fatal if absent), afterok it if still queued, else only if R58's field left 190 start maps; field wan -> render (afterok); --dry-run; logs/r62_wan_chain_jobs.txt, refuses if it exists"
    needed_by: [wan-chain]
    status: pending
  - id: jz-port-list
    content: "jz/r50_port.sh -- add r62_field, r62_wan_render, r62_wan_chain to SCRIPTS"
    needed_by: [wan-chain]
    status: pending
  - id: jz-scratch-list
    content: "jz/r50_scratch_links.sh -- add r62_field, r62_wan_arms, r62_wan_correct to FOLDERS"
    needed_by: [wan-chain]
    status: pending
  - id: jz-fetch-list
    content: "jz/r50_fetch_outputs.sh -- add r62_field, r62_wan_arms to ALL (r62_wan_correct is links only, not fetched)"
    needed_by: [wan-fetch]
    status: pending
  - id: assemble-script
    content: "evaluation/r62_assemble.py -- reuses r58_assemble.py's link / frame-count / conflict helpers for one config: $D/r62_wan_correct/nosupp_s5/edit{T}/{video} -> $D/r62_wan_arms/nosupp_s5_full/... for the 190 pairs of r58_cases_maxs5.json, else -> $D/r51_wan_arms/nosupp_s5_full/...; only symlinks inside r62_wan_correct, idempotent, conflict = exit 1; writes evaluation/csv/r62_wan_assembly.csv; exit 1 unless 415 clips (190 new + 225 reused)"
    needed_by: [wan-assemble]
    status: pending
  - id: se-eval-driver
    content: "slurm_scripts/five_bench/r62_eval.sh -- calls R57's eval_suite.sh once per StreamEdit arm (methods r62_{nodiv,dino,lpips,nosupp}_full, roots $D/r62_se_arms/r62_{arm}_full/step14, --layout edit_video, --cases evaluation/r43_anchor_manifest_correct.json, --tag r62); refuses if eval_suite.sh is missing or a render dir has != 415 clips; job ids to logs/r62_eval_jobs.txt"
    needed_by: [se-eval]
    status: pending
  - id: score-script
    content: "evaluation/r62_score.py -- --host se|wan. se: suite_r62_r62_{arm}_full.csv + A0 suite_r59_r59_max_full.csv + native suite_r59_r59_n15_rho2_vp.csv + equal-NFE curve suite_r59_r59_n20_rho*_vp.csv; wan: suite_r62_wan_nosupp_s5_correct.csv + A0 suite_r58_wan_max_s5_correct.csv + native / matched-59 curves from suite_r58_*; plain per-clip means (overall + per edit type; wan also regenerated vs reused) -> evaluation/csv/r62_points_{se,wan}.csv, Ai - A0 deltas -> r62_deltas_{se,wan}.csv; figures evaluation/figures/r62_{host}_{metric}.pdf (x = FiVE-Acc, CLIP-D, EditReward, PSC; y = bg LPIPS, CF); exit 1 on a row count != 415"
    needed_by: [score]
    status: pending
steps:
  - id: se-chain
    type: local
    command: bash slurm_scripts/five_bench/r62_se_chain.sh
    sets_status: running
    status: pending
  - id: wan-chain
    type: manual
    check_hint: "Needs R58's wan-chain submitted (logs/r58_wan_chain_jobs.txt on Jean Zay). User, on Jean Zay in the clone: git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh && bash slurm_scripts/five_bench/r62_wan_chain.sh --dry-run && bash slurm_scripts/five_bench/r62_wan_chain.sh; paste logs/r62_wan_chain_jobs.txt"
    status: pending
  - id: wait-se-renders
    type: manual
    wait_for: se-chain
    check_hint: "field log: 'A0 gate PASS 415/415', 'failures: 0', 4 x 415 npz; 4 render tasks 'failures: 0', 415 clip dirs each, all-ok manifests, '[r59_check_prompts] OK'"
    status: pending
  - id: wait-wan-chain
    type: manual
    wait_for: wan-chain
    check_hint: "User, on Jean Zay: field log 'A0 gate PASS 190/190' (vs r58_field/wan_max_s5), 190 npz + 190 .npy; render log 'nosupp_s5: 190/190 clips failures: 0'; sacct COMPLETED on both ids"
    status: pending
  - id: wan-assemble
    type: manual
    wait_for: [wait-wan-chain, wait-se-renders]
    check_hint: "User, on Jean Zay: $WORK/pyenv/five-bench/python evaluation/r62_assemble.py -> 415 clips, 190 new + 225 reused, 0 missing, 0 conflicts; paste the summary"
    sets_status: finished
    status: pending
  - id: se-eval
    type: local
    wait_for: wait-se-renders
    command: bash slurm_scripts/five_bench/r62_eval.sh
    check_hint: "Needs R57's lab smoke passed (slurm_scripts/five_bench/eval_suite.sh exists)"
    status: pending
  - id: wan-eval
    type: manual
    wait_for: wan-assemble
    check_hint: "Needs R57 adopted on Jean Zay. User, on Jean Zay: bash jz/eval_suite_jz.sh --qos qos_gpu_h100-t3 --method wan_nosupp_s5_correct --root ~/Data/dataggen/outputs/five_bench/r62_wan_correct/nosupp_s5 --layout edit_video --tag r62; paste the 3 job ids"
    status: pending
  - id: wait-se-eval
    type: manual
    wait_for: se-eval
    check_hint: "every job in logs/r62_eval_jobs.txt COMPLETED; 4 evaluation/csv/suite_r62_r62_*_full.csv with 415 rows each (merge exits 1 on NaN); 0 'Error:' in the metrics logs"
    status: pending
  - id: wan-fetch
    type: manual
    wait_for: wan-eval
    check_hint: "User, on Jean Zay: suite_r62_wan_nosupp_s5_correct.csv has 415 rows, merge log exit 0. Then on the lab frontal: rsync of evaluation/csv/suite_r62_wan_* and r62_wan_assembly.csv back to the lab repo; bash jz/r50_fetch_outputs.sh r62_field r62_wan_arms -> 'failures: 0'"
    status: pending
  - id: score
    type: local
    wait_for: [wait-se-eval, wan-fetch]
    command: "python evaluation/r62_score.py --host se && python evaluation/r62_score.py --host wan"
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "read r62_points_{se,wan}.csv, r62_deltas_{se,wan}.csv and the figures; compare each Ai - A0 sign with R51 on official FiVE; write the run-log verdict"
    sets_status: analyzed
    status: pending
isProject: true
---

# R62: R51's Component Ablation on FiVE-Bench-Correct

## Context

FiVE-Bench-Correct is now the main-paper benchmark, so R51's demand ablation (the paper's main ablation table) has to be reported on Correct too. R62 mirrors R51. It changes only q = S · max(m_DINO, m_LPIPS) and keeps the drafts, attention, divergence, mapping and render flags of the Correct full method: R59's se_max on StreamEdit and R58's max_s5 on Wan-Edit. Those runs are A0. R51's field code first recomputes A0 and must match their fields bit for bit.

Done when: StreamEdit A1–A4 are rendered on the 415 Correct pairs (lab L40S), and Wan-Edit A4 is rendered on R58's 190 changed pairs and assembled to 415 (Jean Zay H100). Both are scored with the R57 suite, and `r62_points_{se,wan}.csv`, `r62_deltas_{se,wan}.csv` and the figures are read against R51.

## Execution steps

| # | Step id | Type | Runs on | Waits for | Sets status |
|---|---|---|---|---|---|
| 0 | *(prep)* `field-sbatch`, `se-render-sbatch`, `se-chain-driver`, `wan-render-sbatch`, `wan-chain-driver`, `jz-port-list`, `jz-scratch-list`, `jz-fetch-list`, `assemble-script`, `se-eval-driver`, `score-script` | /build-step | lab (code) | — | `implemented` |
| 1 | `se-chain` | local (field afterok R59's field job → render array 0-3) | lab | — | `running` |
| 2 | `wan-chain` | manual (field afterok R58's field job → render) | **Jean Zay** | R58 `wan-chain` submitted | — |
| 3 | `wait-se-renders` | manual | lab | se-chain | — |
| 4 | `wait-wan-chain` | manual | **Jean Zay** | wan-chain | — |
| 5 | `wan-assemble` | manual | **Jean Zay** | wait-wan-chain, wait-se-renders | `finished` |
| 6 | `se-eval` | local (needs R57 lab smoke) | lab | wait-se-renders | — |
| 7 | `wan-eval` | manual (needs R57 on Jean Zay) | **Jean Zay** | wan-assemble | — |
| 8 | `wait-se-eval` | manual | lab | se-eval | — |
| 9 | `wan-fetch` | manual | Jean Zay → lab | wan-eval | — |
| 10 | `score` | local | lab | wait-se-eval, wan-fetch | — |
| 11 | `verdict` | manual | lab | score | `analyzed` |

```
/build-step se-chain R62
/build-step wan-chain R62
/build-step wan-assemble R62
/build-step se-eval R62
/build-step wan-fetch R62
/build-step score R62
/run-step R62 se-chain
/run-step R62 wan-chain
/run-step R62 wait-se-renders
/run-step R62 wait-wan-chain
/run-step R62 wan-assemble
/run-step R62 se-eval
/run-step R62 wan-eval
/run-step R62 wait-se-eval
/run-step R62 wan-fetch
/run-step R62 score
/run-step R62 verdict
```

## Decisions

| Item | Choice |
|---|---|
| Benchmark | FiVE-Bench-Correct, 415 pairs (98/100/99/99/9/10), `evaluation/r43_anchor_manifest_correct.json`, annotations `evaluation/five_bench_correct/`; no new prompts |
| Arms (StreamEdit) | Mirror R51: A1 `nodiv` q = S · A2 `dino` q = S·m_dino · A3 `lpips` q = S·m_lpips · A4 `nosupp` q = max(m_dino, m_lpips); A0 = R59 `r59_max_full`, not re-rendered |
| Arms (Wan-Edit) | Mirror R51: A4 `nosupp_s5` only (s_min 5); A0 = R58 `max_s5` (assembled `r58_wan_correct/max_s5`) |
| Inputs | Reused from R59 / R58, no new draft: StreamEdit attention `r59_se_attn`, divergence `r59_div/se_{dino,lpips}`; Wan-Edit attention `r58_wan_draft/attn`, divergence `r58_div/wan_{dino,lpips}`; caps and λ as `r49_field` (shared code) |
| A0 gate | `r51_field.py --gate_ref`: recomputed A0 must equal `r59_field_full/se_max` (`rho`, 415/415) or `r58_field/wan_max_s5` (`.npy`, 190/190) bit for bit, else the field job writes nothing and exits 1 |
| Dependency | The StreamEdit field waits afterok on R59's field job (`logs/r59_se_chain_jobs.txt`, 1066779); the Wan-Edit field waits afterok on R58's field job (`logs/r58_wan_chain_jobs.txt` on Jean Zay); both drivers refuse if the upstream field did not complete |
| Wan-Edit regeneration | Only R58's 190 pairs of `r58_cases_maxs5.json` (prompt or trigger object changed). The other 225 have the same Correct and official inputs, so they link to R51's `r51_wan_arms/nosupp_s5_full` render. Mixing old and new renders relies on R58's reproducibility gate (`repro-check` PASS) |
| No overwrite | New renders only under `$D/r62_se_arms`, `$D/r62_wan_arms`; `r62_wan_correct` holds symlinks only; R51 / R58 / R59 folders are never written |
| Render (StreamEdit) | Flags identical to `r59_se_stage3.sh`: VP + `anchors_r43_correct`, `--edit_json_dir evaluation/five_bench_correct`, step 15, flow_shift 1.0, fg_boost 4, blend_power 2, chunk 21, seed 0, dump step 14; NFE 5 + 15 = 20; prompt-sidecar check in every task |
| Render (Wan-Edit) | `r49_wan_render.py --config nosupp_s5 --mode full --anno_dir evaluation/five_bench_correct`, N = 50, loop start 5, per-token start map |
| Metrics | R57 unified suite (harness incl. motion fidelity, FiVE-Acc, CLIP-D, EditReward, PSC, CF), tag `r62`; lab `eval_suite.sh` for StreamEdit, `jz/eval_suite_jz.sh` for Wan-Edit; blocked until R57 is adopted on each platform |
| Reference | Points table A0–A4 next to R59's native (`r59_n15_rho2_vp`) and equal-NFE (`r59_n20_rho2_vp`) baselines, and next to R58's `native_s0.92` / `matched59_s0.92`. The figures also draw R59's N = 20 uniform-ρ curve and R58's native / matched-59 curves: they are already rendered there, so this needs no extra render |
| Aggregation | Plain per-clip mean from `suite_*.csv` (never `_avg.csv`), overall and per edit type; Wan-Edit also split regenerated vs reused; no paired CIs (as R51); figures x = alignment, y = preservation |
| Platforms | StreamEdit lab L40S (`streamgve` render, `five-bench` field / eval); Wan-Edit Jean Zay H100; separate tables, never mixed |
| Slurm | Lab: `L40S` / `CPU`, `--exclude=node52`, render 20 h; 5 queue tasks, checked against the 30-task cap; Jean Zay headers rewritten by `jz/r50_port.sh` |
| Outputs | `$D/r62_field/{se_nodiv,se_dino,se_lpips,se_nosupp,wan_nosupp_s5}/`, `$D/r62_se_arms/r62_{arm}_full/step14/`, `$D/r62_wan_arms/nosupp_s5_full/`, `$D/r62_wan_correct/nosupp_s5/`, `evaluation/csv/suite_r62_*.csv`, `evaluation/csv/r62_{points,deltas}_{se,wan}.csv`, `evaluation/csv/r62_wan_assembly.csv`, `evaluation/figures/r62_*.pdf` |
| Compute / storage | StreamEdit 4 renders × ~12 h L40S, ~60 GB; Wan-Edit 190 pairs ≈ 4–5 h H100; eval 5 × (1-GPU metrics + CF on 4×L40S / 2×H100 + merge) |
| Out of scope | R52's B1 on Correct, qualitative grids (R51's official-FiVE grids stand), Wan-Edit A1–A3, re-scoring R51 on official FiVE |

## Step commands

### se-chain
```bash
cat logs/r59_se_chain_jobs.txt
bash slurm_scripts/five_bench/r62_se_chain.sh
cat logs/r62_se_chain_jobs.txt
```

### wan-chain
```bash
# on Jean Zay (ssh zay from gpu-gw), in the clone
cat logs/r58_wan_chain_jobs.txt
git pull
bash jz/r50_port.sh
bash jz/r50_scratch_links.sh
bash slurm_scripts/five_bench/r62_wan_chain.sh --dry-run
bash slurm_scripts/five_bench/r62_wan_chain.sh
cat logs/r62_wan_chain_jobs.txt
```

### wait-se-renders
```bash
squeue -u $USER -h -o '%i %j %T' | grep r62
grep -H 'A0 gate' logs/r62_field_*.out
grep -h 'r59_check_prompts' logs/r62_se_stage3_*.out
grep -hE 'failures:' logs/r62_field_*.out logs/r62_se_stage3_*.out | sort | uniq -c
D=~/Data/dataggen/outputs/five_bench
for A in nodiv dino lpips nosupp; do echo "$A $(ls -d $D/r62_se_arms/r62_${A}_full/step14/edit*/*/ | wc -l)"; done
```

### wait-wan-chain
```bash
# on Jean Zay
cat logs/r62_wan_chain_jobs.txt
sacct -j "$(grep -v '^#' logs/r62_wan_chain_jobs.txt | awk '{print $2}' | paste -sd,)" --format=JobID,State,Elapsed
grep -H 'A0 gate' logs/r62_field_*.out
grep -hE 'failures:' logs/r62_field_*.out logs/r62_wan_render_*.out | sort | uniq -c
```

### wan-assemble
```bash
# on Jean Zay
$WORK/pyenv/five-bench/python evaluation/r62_assemble.py
```

### se-eval
```bash
ls slurm_scripts/five_bench/eval_suite.sh
bash slurm_scripts/five_bench/r62_eval.sh
cat logs/r62_eval_jobs.txt
```

### wan-eval
```bash
# on Jean Zay
bash jz/eval_suite_jz.sh --qos qos_gpu_h100-t3 --method wan_nosupp_s5_correct \
  --root ~/Data/dataggen/outputs/five_bench/r62_wan_correct/nosupp_s5 --layout edit_video --tag r62
```

### wait-se-eval
```bash
for J in $(awk '{print $2}' logs/r62_eval_jobs.txt); do sacct -j $J -X -n -o JobID,State; done
for F in evaluation/csv/suite_r62_r62_*_full.csv; do echo "$F $(($(wc -l < $F) - 1))"; done
```

### wan-fetch
```bash
# on the lab frontal (tmux)
source jz/r50_jz.env
rsync -av "$JZ_HOST:$JZ_WORK/StreamEdit_new/evaluation/csv/" evaluation/csv/ --include='suite_r62_wan_*' --include='r62_wan_assembly.csv' --exclude='*'
bash jz/r50_fetch_outputs.sh r62_field r62_wan_arms
```

### score
```bash
python evaluation/r62_score.py --host se
python evaluation/r62_score.py --host wan
```

### verdict
```bash
column -s, -t < evaluation/csv/r62_points_se.csv | less -S
column -s, -t < evaluation/csv/r62_deltas_se.csv | less -S
column -s, -t < evaluation/csv/r62_points_wan.csv | less -S
column -s, -t < evaluation/csv/r62_deltas_wan.csv | less -S
ls evaluation/figures/r62_*.pdf
```

## Pipeline

```mermaid
flowchart LR
  subgraph lab["Lab L40S (StreamEdit)"]
    R59F[R59 field job<br/>r59_se_attn + r59_div] --> F1[r62_field.sh se<br/>A0 gate vs r59_field_full/se_max]
    F1 --> R1[r62_se_stage3.sh A1..A4]
    R1 --> E1[r62_eval.sh → eval_suite.sh]
    E1 --> S1[r62_score.py --host se]
    R59C[suite_r59_* A0 + baselines + N=20 curve] --> S1
  end
  subgraph jz["Jean Zay H100 (Wan-Edit)"]
    R58F[R58 field job<br/>r58_wan_draft/attn + r58_div] --> F2[r62_field.sh wan<br/>A0 gate vs r58_field/wan_max_s5, 190 pairs]
    F2 --> R2[r62_wan_render.sh nosupp_s5, 190 pairs]
    R2 --> AS[r62_assemble.py<br/>190 new + 225 r51_wan_arms links]
    R51W[r51_wan_arms/nosupp_s5_full] --> AS
    AS --> E2[eval_suite_jz.sh]
  end
  E2 --> FX[fetch CSVs] --> S2[r62_score.py --host wan]
  R58C[suite_r58_* A0 + native + matched-59] --> S2
```

## Code to touch

| File | Change |
|---|---|
| `slurm_scripts/five_bench/r62_field.sh` (new) | Built from `r51_field.sh` (CPU, 4 cpus, 32G, 2 h) but calls `r51_field.py` directly with explicit roots per HOST. `se`: `--cases evaluation/r43_anchor_manifest_correct.json --attn_root $D/r59_se_attn --div_root $D/r59_div --gate_ref $D/r59_field_full --arms nodiv dino lpips nosupp`. `wan`: `--cases evaluation/r58_cases_maxs5.json --attn_root $D/r58_wan_draft/attn --div_root $D/r58_div --gate_ref $D/r58_field --arms nosupp`. Both `--out_root $D/r62_field`. Selftest first; guard N npz per arm dir = len(cases) (+ N `.npy` for wan); `failures: N`. |
| `slurm_scripts/five_bench/r62_se_stage3.sh` (new) | Clone of `r59_se_stage3.sh`: `ARMS=(nodiv dino lpips nosupp)`, task id → arm, `FIELD_DIR=$D/r62_field/se_${ARM}`, `METHOD=r62_${ARM}_full`, `OUT_ROOT=$D/r62_se_arms`, logs `r62_se_stage3_%A_%a`; every other flag and guard is unchanged (Correct cases / edit JSONs / anchors, `r59_check_prompts.py` on `_prompts_edit*.json`). |
| `slurm_scripts/five_bench/r62_se_chain.sh` (new) | Refuses if `logs/r62_se_chain_jobs.txt` exists or the queue + 5 tasks > `${R62_QUEUE_CAP:-30}`; reads `field <id>` from `logs/r59_se_chain_jobs.txt`; dependency `afterok:<id>` if `squeue -j <id>` knows it, else `sacct` must say COMPLETED (submit with no dependency), else exit 1; submits `field` (`r62_field.sh se`), `render` (`--array=0-3`, afterok field); writes `<stage> <id>` lines headed by the commit. |
| `slurm_scripts/five_bench/r62_wan_render.sh` (new) | As in todo `wan-render-sbatch`; built from `r51_wan_render.sh` with the Correct case file and `--anno_dir`; output-root guard as `r58_wan_render.sh`. |
| `slurm_scripts/five_bench/r62_wan_chain.sh` (new) | Jean Zay only; `$WORK` / ported-script / `$SCRATCH`-link guards copied from `r51_wan_chain.sh`; upstream id from `logs/r58_wan_chain_jobs.txt` (`field <id>`), afterok if queued, else require 190 `.npy` in `$D/r58_field/wan_max_s5`; submits `field` (`r62_field.sh wan`) → `render` (afterok); `--dry-run`; `logs/r62_wan_chain_jobs.txt`, refuses if it exists. |
| `jz/r50_port.sh` | `SCRIPTS` += `r62_field`, `r62_wan_render`, `r62_wan_chain`. |
| `jz/r50_scratch_links.sh` | `FOLDERS` += `r62_field`, `r62_wan_arms`, `r62_wan_correct`. |
| `jz/r50_fetch_outputs.sh` | `ALL` += `r62_field`, `r62_wan_arms`. |
| `evaluation/r62_assemble.py` (new) | Imports `r58_assemble`'s helpers (symlink creation, expected PNG count via `r48_wan_common.expected_frames`, conflict rule) with one config `nosupp_s5`: new root `$D/r62_wan_arms/nosupp_s5_full`, reuse root `$D/r51_wan_arms/nosupp_s5_full`, regen set `r58_cases_maxs5.json`, out `$D/r62_wan_correct/nosupp_s5`, cases the Correct manifest. If `r58_assemble.py` does not expose them as functions, factor them out there with no behaviour change. Writes `evaluation/csv/r62_wan_assembly.csv` (case_id, source new\|reused, n_png). |
| `slurm_scripts/five_bench/r62_eval.sh` (new) | Built from `r59_eval.sh`: 4 methods `r62_{arm}_full` over `$D/r62_se_arms/r62_{arm}_full/step14`, `--layout edit_video --cases evaluation/r43_anchor_manifest_correct.json --tag r62`; preflight 415 clip dirs per root; ids to `logs/r62_eval_jobs.txt`, refuses if it exists. |
| `evaluation/r62_score.py` (new) | As in todo `score-script`; reuses `r59_score.py` / `r58_score.py` loaders and plotting where they exist, otherwise `r49_score` helpers (`point`, `plot`); every input CSV must have 415 rows. |
