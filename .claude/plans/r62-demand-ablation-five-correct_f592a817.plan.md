---
name: R62 — R51's Component Ablation on FiVE-Bench-Correct
overview: "R51's demand ablation re-run on the 415 FiVE-Bench-Correct pairs, mirroring R51: StreamEdit A1 nodiv / A2 dino / A3 lpips / A4 nosupp on the lab L40S, Wan-Edit A4 nosupp_s5 on Jean Zay H100. Only q changes; drafts, attention and divergence are R59's (StreamEdit) and R58's (Wan-Edit) Correct stage 1, A0 = R59 se_max / R58 max_s5, with R51's bit-exact A0 gate against their fields. Wan-Edit regenerates only R58's 190 changed pairs and links the other 225 to R51's A4 renders. Wan-Edit renders are copied to the lab (rsync -L) and every arm is scored with the R57 unified suite on the lab L40S; points table + trade-off figures on R59's / R58's existing curves."
task_id: R62
todos:
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r62_field.sh -- CPU job, HOST arg (se | wan): r51_field.py --selftest, then r51_field.py with the Correct roots (se: cases r43_anchor_manifest_correct.json, attn r59_se_attn, div r59_div, gate r59_field_full, arms nodiv dino lpips nosupp; wan: cases r58_cases_maxs5.json, attn r58_wan_draft/attn, div r58_div, gate r58_field, arm nosupp), out $D/r62_field; one-npz-per-case guard (+ .npy for wan)"
    needed_by: [se-chain, wan-chain]
    status: completed
  - id: se-render-sbatch
    content: "slurm_scripts/five_bench/r62_se_stage3.sh -- clone of r59_se_stage3.sh with --array=0-3 = nodiv, dino, lpips, nosupp: FIELD_DIR $D/r62_field/se_${ARM}, METHOD r62_${ARM}_full, OUT_ROOT $D/r62_se_arms; Correct edit JSONs, cases and anchors, flags, count / manifest guards and r59_check_prompts.py guard unchanged"
    needed_by: [se-chain]
    status: completed
  - id: se-chain-driver
    content: "slurm_scripts/five_bench/r62_se_chain.sh -- lab driver: reads 'field <id>' from logs/r59_se_chain_jobs.txt (fatal if absent); field se afterok that id if still in squeue, else only if sacct says COMPLETED (refuse otherwise); render --array=0-3 afterok field; queue-cap check (+5 tasks, cap 30) before submitting; logs/r62_se_chain_jobs.txt headed by the commit, refuses if it exists"
    needed_by: [se-chain]
    status: completed
  - id: wan-render-sbatch
    content: "slurm_scripts/five_bench/r62_wan_render.sh -- single job: preflight 190 .npy under $D/r62_field/wan_nosupp_s5; r49_wan_render.py --config nosupp_s5 --mode full --cases evaluation/r58_cases_maxs5.json --anno_dir evaluation/five_bench_correct --field_root $D/r62_field --out_root $D/r62_wan_arms; refuses any OUT other than $D/r62_wan_arms; r48_units.py --missing post-run guard; lab header with --exclude=node52 (ported by r50_port.sh)"
    needed_by: [wan-chain]
    status: completed
  - id: wan-chain-driver
    content: "slurm_scripts/five_bench/r62_wan_chain.sh -- Jean Zay driver modelled on r51_wan_chain.sh: guards ($WORK, ported scripts, $SCRATCH links, r58_cases_maxs5.json present); reads 'field <id>' from logs/r58_wan_chain_jobs.txt (fatal if absent), afterok it if still queued, else only if R58's field left 190 start maps; field wan -> render (afterok); --dry-run; logs/r62_wan_chain_jobs.txt, refuses if it exists"
    needed_by: [wan-chain]
    status: completed
  - id: jz-port-list
    content: "jz/r50_port.sh -- add r62_field, r62_wan_render, r62_wan_chain to SCRIPTS"
    needed_by: [wan-chain]
    status: completed
  - id: jz-scratch-list
    content: "jz/r50_scratch_links.sh -- add r62_field, r62_wan_arms, r62_wan_correct to FOLDERS"
    needed_by: [wan-chain]
    status: completed
  - id: jz-fetch-list
    content: "jz/r50_fetch_outputs.sh -- add r62_field, r62_wan_arms to ALL (archive copy on /store; r62_wan_correct is links only, not fetched)"
    needed_by: [fetch-renders]
    status: completed
  - id: fetch-renders-script
    content: "jz/r62_fetch_renders.sh -- lab frontal (gpu-gw) only, modelled on jz/r58_fetch_renders.sh (r50_jz.env, SSH ControlMaster, $SCRATCH lookup) for the one config: rsync of Jean Zay evaluation/csv/r62_wan_assembly.csv into evaluation/csv/, then rsync -aL --partial zay:$SCRATCH/five_bench_out/r62_wan_correct/nosupp_s5/ -> $R62_DEST/nosupp_s5/ (default R62_DEST=/work/PERSO/skhairi/five_bench/r62_wan_correct, same filesystem as R58_DEST); preflight >= 12 GB free; post-check 415 clip dirs and every clip's PNG count = n_png of r62_wan_assembly.csv; --check mode counts only; never deletes on either side; ends with '[r62_fetch] failures: N'"
    needed_by: [fetch-renders]
    status: completed
  - id: assemble-script
    content: "evaluation/r62_assemble.py -- reuses r58_assemble.py's link / frame-count / conflict helpers for one config: $D/r62_wan_correct/nosupp_s5/edit{T}/{video} -> $D/r62_wan_arms/nosupp_s5_full/... for the 190 pairs of r58_cases_maxs5.json, else -> $D/r51_wan_arms/nosupp_s5_full/...; only symlinks inside r62_wan_correct, idempotent, conflict = exit 1; writes evaluation/csv/r62_wan_assembly.csv; exit 1 unless 415 clips (190 new + 225 reused)"
    needed_by: [wan-assemble]
    status: completed
  - id: eval-driver
    content: "slurm_scripts/five_bench/r62_eval.sh -- lab launcher (bash), HOST arg se | wan: calls R57's eval_suite.sh once per method (se: r62_{nodiv,dino,lpips,nosupp}_full over $D/r62_se_arms/r62_{arm}_full/step14; wan: wan_nosupp_s5_correct over $R62_DEST/nosupp_s5), --layout edit_video --cases evaluation/r43_anchor_manifest_correct.json --tag r62 --allow_nan motion_fidelity_score_edit_part; preflight: eval_suite.sh present, 415 clip dirs per root, queue + 3 x n_methods <= 30 tasks; refuses a method whose suite CSV exists or that already has a line in logs/r62_eval_jobs.txt; appends '<method> <metrics> <cf> <merge>' ids there"
    needed_by: [se-eval, wan-eval]
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
    status: completed
    completed_at: 2026-10-29
    job_id: "field 1074368, render 1074369 (lab; R59 field 1070891 had already finished)"
  - id: wan-chain
    type: manual
    check_hint: "Needs R58's wan-chain submitted (logs/r58_wan_chain_jobs.txt on Jean Zay). User, on Jean Zay in the clone: git pull && bash jz/r50_port.sh && bash jz/r50_scratch_links.sh && bash slurm_scripts/five_bench/r62_wan_chain.sh --dry-run && bash slurm_scripts/five_bench/r62_wan_chain.sh; paste logs/r62_wan_chain_jobs.txt"
    status: completed
    completed_at: 2026-10-28
    job_id: "field 1670342, render 1670343 (Jean Zay; field afterok R58 field 1667321)"
  - id: wait-se-renders
    type: manual
    wait_for: se-chain
    check_hint: "field log: 'A0 gate PASS 415/415', 'failures: 0', 4 x 415 npz; 4 render tasks 'failures: 0', 415 clip dirs each, all-ok manifests, '[r59_check_prompts] OK'"
    status: completed
    completed_at: 2026-11-01
  - id: wait-wan-chain
    type: manual
    wait_for: wan-chain
    check_hint: "User, on Jean Zay: field log 'A0 gate PASS 190/190' (vs r58_field/wan_max_s5), 190 npz + 190 .npy; render log 'nosupp_s5: 190/190 clips failures: 0'; sacct COMPLETED on both ids"
    status: completed
    completed_at: 2026-11-01
  - id: wan-assemble
    type: manual
    wait_for: wait-wan-chain
    check_hint: "User, on Jean Zay: $WORK/pyenv/five-bench/python evaluation/r62_assemble.py -> 415 clips, 190 new + 225 reused, 0 missing, 0 conflicts; paste the summary"
    status: pending
  - id: fetch-renders
    type: manual
    wait_for: [wan-assemble, wait-se-renders]
    check_hint: "Needs R58's probe-mounts passed (/work/PERSO/skhairi visible from gpu-gw and L40S nodes). User, on gpu-gw in tmux: bash jz/r62_fetch_renders.sh --check && bash jz/r62_fetch_renders.sh -> '[r62_fetch] failures: 0' (~9.3 GB); then bash jz/r50_fetch_outputs.sh r62_field r62_wan_arms -> '[r50_fetch] failures: 0'"
    sets_status: finished
    status: pending
  - id: se-eval
    type: local
    wait_for: wait-se-renders
    command: bash slurm_scripts/five_bench/r62_eval.sh se
    check_hint: "Needs R57's lab smoke passed (slurm_scripts/five_bench/eval_suite.sh exists)"
    status: pending
  - id: wan-eval
    type: local
    wait_for: fetch-renders
    command: bash slurm_scripts/five_bench/r62_eval.sh wan
    check_hint: "Needs R57's lab smoke passed"
    status: pending
  - id: wait-eval
    type: manual
    wait_for: [se-eval, wan-eval]
    check_hint: "every job in logs/r62_eval_jobs.txt COMPLETED; 5 evaluation/csv/suite_r62_*.csv (4 StreamEdit + wan_nosupp_s5_correct) with 415 rows each, merge exit 0; 0 'Error:' in the metrics logs"
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
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

Done when: StreamEdit A1–A4 are rendered on the 415 Correct pairs (lab L40S), and Wan-Edit A4 is rendered on R58's 190 changed pairs and assembled to 415 (Jean Zay H100). The Wan-Edit renders are copied to the lab, every arm is scored with the R57 suite on the lab L40S, and `r62_points_{se,wan}.csv`, `r62_deltas_{se,wan}.csv` and the figures are read against R51.

## Execution steps

| # | Step id | Type | Runs on | Waits for | Sets status |
|---|---|---|---|---|---|
| 0 | *(prep)* `field-sbatch`, `se-render-sbatch`, `se-chain-driver`, `wan-render-sbatch`, `wan-chain-driver`, `jz-port-list`, `jz-scratch-list`, `jz-fetch-list`, `fetch-renders-script`, `assemble-script`, `eval-driver`, `score-script` | /build-step | lab (code) | — | `implemented` |
| 1 | `se-chain` | local (field afterok R59's field job → render array 0-3) | lab | — | `running` |
| 2 | `wan-chain` | manual (field afterok R58's field job → render) | **Jean Zay** | R58 `wan-chain` submitted | — |
| 3 | `wait-se-renders` | manual | lab | se-chain | — |
| 4 | `wait-wan-chain` | manual | **Jean Zay** | wan-chain | — |
| 5 | `wan-assemble` | manual | **Jean Zay** | wait-wan-chain | — |
| 6 | `fetch-renders` | manual (rsync -L to the lab) | gpu-gw | wan-assemble, wait-se-renders | `finished` |
| 7 | `se-eval` | local (needs R57 lab smoke) | lab | wait-se-renders | — |
| 8 | `wan-eval` | local (needs R57 lab smoke) | lab | fetch-renders | — |
| 9 | `wait-eval` | manual | lab | se-eval, wan-eval | — |
| 10 | `score` | local | lab | wait-eval | — |
| 11 | `verdict` | manual | lab | score | `analyzed` |

```
/build-step se-chain R62
/build-step wan-chain R62
/build-step wan-assemble R62
/build-step fetch-renders R62
/build-step se-eval R62
/build-step score R62
/run-step R62 se-chain
/run-step R62 wan-chain
/run-step R62 wait-se-renders
/run-step R62 wait-wan-chain
/run-step R62 wan-assemble
/run-step R62 fetch-renders
/run-step R62 se-eval
/run-step R62 wan-eval
/run-step R62 wait-eval
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
| No overwrite | New renders only under `$D/r62_se_arms`, `$D/r62_wan_arms`; `r62_wan_correct` on Jean Zay holds symlinks only; the lab copy `$R62_DEST` is a new folder; R51 / R58 / R59 folders are never written |
| Render (StreamEdit) | Flags identical to `r59_se_stage3.sh`: VP + `anchors_r43_correct`, `--edit_json_dir evaluation/five_bench_correct`, step 15, flow_shift 1.0, fg_boost 4, blend_power 2, chunk 21, seed 0, dump step 14; NFE 5 + 15 = 20; prompt-sidecar check in every task |
| Render (Wan-Edit) | `r49_wan_render.py --config nosupp_s5 --mode full --anno_dir evaluation/five_bench_correct`, N = 50, loop start 5, per-token start map |
| Metrics | R57 unified suite (harness incl. motion fidelity, FiVE-Acc, CLIP-D, EditReward, PSC, CF), tag `r62`, `--allow_nan motion_fidelity_score_edit_part` (empty-mask clips), lab `eval_suite.sh` for both hosts; blocked until R57's lab smoke passes |
| Reference | Points table A0–A4 next to R59's native (`r59_n15_rho2_vp`) and equal-NFE (`r59_n20_rho2_vp`) baselines, and next to R58's `native_s0.92` / `matched59_s0.92`. The figures also draw R59's N = 20 uniform-ρ curve and R58's native / matched-59 curves: they are already rendered there, so this needs no extra render |
| Aggregation | Plain per-clip mean from `suite_*.csv` (never `_avg.csv`), overall and per edit type; Wan-Edit also split regenerated vs reused; no paired CIs (as R51); figures x = alignment, y = preservation |
| Platforms | Renders: StreamEdit lab L40S (`streamgve`), Wan-Edit Jean Zay H100 (`five-bench` overlay). **Scoring: both hosts on the lab L40S** with the R57 suite (user, 2026-10-27); Wan-Edit renders copied to the lab with `rsync -L` into `$R62_DEST=/work/PERSO/skhairi/five_bench/r62_wan_correct` (same filesystem as R58's, gated by R58's `probe-mounts`); separate tables per host |
| Slurm | Lab: `L40S` / `CPU`, `--exclude=node52`, render 20 h; 5 queue tasks, checked against the 30-task cap; Jean Zay headers rewritten by `jz/r50_port.sh` |
| Outputs | `$D/r62_field/{se_nodiv,se_dino,se_lpips,se_nosupp,wan_nosupp_s5}/`, `$D/r62_se_arms/r62_{arm}_full/step14/`, `$D/r62_wan_arms/nosupp_s5_full/`, `$D/r62_wan_correct/nosupp_s5/` (Jean Zay, links), `$R62_DEST/nosupp_s5/` (lab copy), `evaluation/csv/suite_r62_*.csv`, `evaluation/csv/r62_{points,deltas}_{se,wan}.csv`, `evaluation/csv/r62_wan_assembly.csv`, `evaluation/figures/r62_*.pdf` |
| Compute / storage | StreamEdit 4 renders × ~12 h L40S, ~60 GB; Wan-Edit 190 pairs ≈ 4–5 h H100; eval on the lab: 5 × (1-GPU metrics + CF on 4×L40S + merge) = 15 queue tasks |
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

### fetch-renders
```bash
# on gpu-gw (tmux)
bash jz/r62_fetch_renders.sh --check
bash jz/r62_fetch_renders.sh
bash jz/r50_fetch_outputs.sh r62_field r62_wan_arms
```

### se-eval
```bash
ls slurm_scripts/five_bench/eval_suite.sh
bash slurm_scripts/five_bench/r62_eval.sh se
cat logs/r62_eval_jobs.txt
```

### wan-eval
```bash
bash slurm_scripts/five_bench/r62_eval.sh wan
cat logs/r62_eval_jobs.txt
```

### wait-eval
```bash
for J in $(awk '{print $2, $3, $4}' logs/r62_eval_jobs.txt); do sacct -j $J -X -n -o JobID,State; done
for F in evaluation/csv/suite_r62_*.csv; do echo "$F $(($(wc -l < $F) - 1))"; done
grep -c 'Error:' logs/eval_suite_metrics_*.out | grep -v ':0$'
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
    R1 --> E1[r62_eval.sh se → eval_suite.sh]
    E1 --> S1[r62_score.py --host se]
    R59C[suite_r59_* A0 + baselines + N=20 curve] --> S1
  end
  subgraph jz["Jean Zay H100 (Wan-Edit)"]
    R58F[R58 field job<br/>r58_wan_draft/attn + r58_div] --> F2[r62_field.sh wan<br/>A0 gate vs r58_field/wan_max_s5, 190 pairs]
    F2 --> R2[r62_wan_render.sh nosupp_s5, 190 pairs]
    R2 --> AS[r62_assemble.py<br/>190 new + 225 r51_wan_arms links]
    R51W[r51_wan_arms/nosupp_s5_full] --> AS
  end
  AS --> FX[r62_fetch_renders.sh rsync -L → lab] --> E2[r62_eval.sh wan → eval_suite.sh, lab L40S]
  E2 --> S2[r62_score.py --host wan]
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
| `jz/r62_fetch_renders.sh` (new) | As in todo `fetch-renders-script`; one config, `R62_DEST` override, `--check`. |
| `evaluation/r62_assemble.py` (new) | Imports `r58_assemble`'s helpers (symlink creation, expected PNG count via `r48_wan_common.expected_frames`, conflict rule) with one config `nosupp_s5`: new root `$D/r62_wan_arms/nosupp_s5_full`, reuse root `$D/r51_wan_arms/nosupp_s5_full`, regen set `r58_cases_maxs5.json`, out `$D/r62_wan_correct/nosupp_s5`, cases the Correct manifest. If `r58_assemble.py` does not expose them as functions, factor them out there with no behaviour change. Writes `evaluation/csv/r62_wan_assembly.csv` (case_id, source new\|reused, n_png). |
| `slurm_scripts/five_bench/r62_eval.sh` (new) | Built from `r58_eval.sh` / `r59_eval.sh`: `HOST` arg; `se` = 4 methods `r62_{arm}_full` over `$D/r62_se_arms/r62_{arm}_full/step14`, `wan` = `wan_nosupp_s5_correct` over `$R62_DEST/nosupp_s5`; `--layout edit_video --cases evaluation/r43_anchor_manifest_correct.json --tag r62 --allow_nan motion_fidelity_score_edit_part`; preflights and `logs/r62_eval_jobs.txt` as in todo `eval-driver`. |
| `evaluation/r62_score.py` (new) | As in todo `score-script`; reuses `r59_score.py` / `r58_score.py` loaders and plotting where they exist, otherwise `r49_score` helpers (`point`, `plot`); every input CSV must have 415 rows. |
