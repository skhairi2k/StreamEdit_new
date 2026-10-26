---
name: R56 — Draft-Budget Ablation (StreamEdit)
overview: "se_max on StreamEdit with a 15-step draft instead of 5 (same release rule sigma_d = 0.96: --blend_sched tgt0.96 anchors only t = 1.0, release from t = 0.933), VP, 419 pairs, lab L40S; R49's divergence / field / render unchanged; read at matched NFE: se_max(15-step draft, 30 NFE) on R39's uniform-rho N=30 curve next to se_max(5-step draft, 20 NFE) on R39 N=20; harness + FiVE-Acc + CLIP-D + motion fidelity; trade-off figure, per-type means."
task_id: R56
todos:
  - id: stage1-sbatch
    content: "slurm_scripts/five_bench/r56_se_stage1.sh -- r48_se_stage1.sh full with --step 15 --blend_sched tgt0.96 --dump_steps 14 --measure_index 14 (method n15_tgt096), VP + anchors, recorder; out r56_se_stage1 (frames + latents) and r56_se_attn; no bit-identity check; guards: 419 attention + 419 latent npz, every latent records (vp, tgt0.96, 15)"
    needed_by: [se-chain]
    status: completed
  - id: div-sbatch
    content: "slurm_scripts/five_bench/r56_div.sh -- array 0-1 (dino, lpips): r53_div.sh's r31_divergence.py call on target r56_se_stage1/frames/n15_tgt096/step14, latents r56_se_stage1/latents, out r56_div/se_{metric}; 419-npz guard"
    needed_by: [se-chain]
    status: completed
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r56_field.sh -- CPU: r49_field.py --selftest, then --hosts se --cases evaluation/r7_anchor_manifest.json --attn_se r56_se_attn --div_root r56_div --out_root r56_field_full; 419-npz guard on se_max"
    needed_by: [se-chain]
    status: completed
  - id: render-sbatch
    content: "slurm_scripts/five_bench/r56_se_stage3.sh -- r49_se_stage3.sh full (VP + anchors, step 15, fg_boost 4, blend_power 2, chunk 21, seed 0), field r56_field_full/se_max, method r56_max_n15_full, out r56_se_arms; same guards"
    needed_by: [se-chain]
    status: completed
  - id: units-r56-groups
    content: "evaluation/r51_units.py -- groups r56_se and r56_motion = (se, max_n15, r56_se_arms/r56_max_n15_full/step14), stem prefix r56"
    needed_by: [se-chain, score]
    status: completed
  - id: eval-sbatch-groups
    content: "slurm_scripts/five_bench/r51_eval.sh -- accept the group r56_se in the usage `case`"
    needed_by: [se-chain]
    status: completed
  - id: motion-sbatch-groups
    content: "slurm_scripts/five_bench/r51_motion.sh -- accept the group r56_motion in the usage `case`"
    needed_by: [se-chain]
    status: completed
  - id: chain-driver
    content: "slurm_scripts/five_bench/r56_se_chain.sh -- lab driver: stage1 -> div (array 0-1) -> field -> render -> eval r56_se + motion r56_motion (afterok render); refuses if the queue + 7 tasks exceeds the per-user cap or if logs/r56_se_chain_jobs.txt exists; checks the R51 scripts carry the R56 groups"
    needed_by: [se-chain]
    status: completed
  - id: score-script
    content: "evaluation/r56_score.py -- se_max N5 (R49 se_max full) on R39 N=20 and se_max N15 on R39 N=30, plus the official rho=2: figures r56_se_{five_acc,clip_d,clip_tgt}.pdf, r56_points_se.csv (419, 401 and per edit type), r56_motion.csv (N15 from R56, N5 and official from R51's motion units)"
    needed_by: [score]
    status: completed
steps:
  - id: se-chain
    type: local
    command: bash slurm_scripts/five_bench/r56_se_chain.sh
    sets_status: running
    status: completed
    completed_at: 2026-10-23
    job_id: "stage1 1060455, div 1060456, field 1060457, render 1060458, eval 1060459, motion 1060460 (resubmitted 2026-10-23; the first chain 1059976-1059981 failed: tgt0.96 was not an allowed --blend_sched choice)"
  - id: wait-se-chain
    type: manual
    wait_for: se-chain
    check_hint: "no r56 job left in squeue; every r56 log 'failures: 0' / 'unit failures: 0'; the stage-1 log echoes the per-step blender_rate [0, 1 x14] (only t = 1 anchored); 419 attention + latent npz (vp, tgt0.96, 15), 2 x 419 divergence, 419 field npz, 419 renders with ok manifests; 1 eval unit and 1 motion unit with full row counts"
    sets_status: finished
    status: completed
    completed_at: 2026-10-26
    job_id: "checked 2026-10-26: jobs 1060455-1060460 + motion rerun 1064395"
  - id: score
    type: local
    wait_for: wait-se-chain
    command: python evaluation/r56_score.py
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "read r56_points_se.csv (overall and per type, types 3-5 first), r56_motion.csv and the figures: gap of each arm to its matched curve (N5 vs R39 N=20, N15 vs R39 N=30)"
    sets_status: analyzed
    status: pending
isProject: true
---

# R56: Draft-Budget Ablation (StreamEdit)

## Context

The method's draft is defined by a noise level, not a step count: the target is released at the first grid point at or below σ_d = 0.96. On StreamEdit's 5-step draft grid that point is σ = 0.8 (R49's `tgt0.9`, which anchors the same calls as `tgt0.96`). R56 gives the draft its full budget: 15 steps, same rule. The finer grid then releases at σ = 0.933 (only t = 1.0 anchored). Divergence, field and render are R49's, unchanged. The ablation is read at matched compute. se_max with the 15-step draft costs 15 + 15 = 30 NFE and is placed on R39's uniform-ρ N = 30 curve, which is already rendered and scored. It sits next to se_max with the 5-step draft (20 NFE) on R39's N = 20 curve. Two outcomes are possible. If the gain over the matched curve grows, the procedure scales with draft compute. If it holds, the short draft suffices. Colour / material / add-object edits (types 3–5) are the main readout, since `method.tex` notes the 5-step draft barely changes colour edits. StreamEdit only: a full-length Wan-Edit draft (~86 active steps with the render) would need a new matched curve on Jean Zay.

Done when: se_max with the 15-step draft is rendered and scored on the 419 pairs (harness, FiVE-Acc, CLIP-D, motion fidelity) on the lab L40S, and the trade-off figure and per-type means place both draft budgets on their matched-NFE curves.

## Execution steps

| # | Step id | Type | Waits for | Sets status |
|---|---|---|---|---|
| 0 | *(prep)* todos `stage1-sbatch`, `div-sbatch`, `field-sbatch`, `render-sbatch`, `units-r56-groups`, `eval-sbatch-groups`, `motion-sbatch-groups`, `chain-driver`, `score-script` | /build-step | — | implemented |
| 1 | `se-chain` | local (submits 6 queue entries = 7 tasks on the lab cluster) | — | running |
| 2 | `wait-se-chain` | manual | se-chain | finished |
| 3 | `score` | local | wait-se-chain | — |
| 4 | `verdict` | manual | score | analyzed |

Invoke: `/run-step R56 se-chain`, `/run-step R56 wait-se-chain`, `/run-step R56 score`, `/run-step R56 verdict`.

## Decisions

| Item | Choice |
|---|---|
| Draft | StreamEdit, VP + Qwen anchors, N = 15, `--blend_sched tgt0.96` (anchors t > 0.96: only t = 1.0; release from t = 0.933), fg_boost 4, blend_power 2, flow_shift 1, chunk 21, seed 0, `--dump_steps 14 --measure_index 14` (the final x0), attention recorder at r31_stage1.py defaults (first 20 blocks, σ window [0.55, 0.9]) |
| Release rule | σ_d = 0.96 on every host and grid: StreamEdit 5-step draft releases at 0.8, 15-step at 0.933, Wan-Edit 17-step draft starts at 0.958; any σ_d in [0.96, 0.97) gives identical drafts. R49's N = 5 runs (`tgt0.9`) anchor the same calls as `tgt0.96`, so they are not rerun |
| Draft check | guards only (no stored draft uses this schedule: R35 rendered `first2` / `unblended`, R44 `unblended` / `tgt0.9`); counts, (vp, tgt0.96, 15) in every latent, and the echoed per-step blender_rate [0, 1 × 14] |
| Divergence / field / render | unchanged from R49: DINOv3 + LPIPS (`r31_divergence.py`, temporal mean), `r49_field.py` (caps 0.55 / 1.0, λ 0.5, q = S·max, budget-linear inverse on the 15-step render grid), `r49_se_stage3.sh` flags |
| Matched compute | se_max N15 (30 NFE) vs R39 uniform ρ N = 30 (8 ρ, scored); se_max N5 = R49's `se_max` full (20 NFE) vs R39 N = 20; official = stock ρ = 2, N = 15 (`r36_rho2_vp`) |
| Metrics | 9 harness metrics, FiVE-Acc, CLIP-D, `motion_fidelity_score` + `_edit_part` (N5 and official motion from R51's units) |
| Scoring | plain per-clip means over 419, over the 401 outside `r49_cases.json`, and per edit type (1–6); figures x = alignment (FiVE-Acc, CLIP-D, CLIP-T), y = bg LPIPS; no curve-gap interpolation per type, no CIs (user choice) |
| Out of scope | Wan-Edit full-length draft (needs a matched ~86-step curve on Jean Zay); other draft lengths |
| Queue | 7 tasks; the driver refuses when the queue plus 7 exceeds the per-user cap (30) |
| Slurm | `--partition=L40S` / `CPU`, `--exclude=node52`; draft 20 h (R38 measured ~109 s / clip at N = 15), render 20 h, eval 12 h, motion 12 h |
| Outputs | `$D/r56_se_stage1/`, `$D/r56_se_attn/`, `$D/r56_div/se_{dino,lpips}/`, `$D/r56_field_full/se_max/`, `$D/r56_se_arms/r56_max_n15_full/step14/`, `evaluation/csv/*r56_*`, `evaluation/figures/r56_se_*.pdf` |

## Step commands

### se-chain
```bash
bash slurm_scripts/five_bench/r56_se_chain.sh
cat logs/r56_se_chain_jobs.txt
```

### wait-se-chain
```bash
cat logs/r56_se_chain_jobs.txt
squeue -u $USER
grep -hE 'failures|unit failures' logs/r56_*.out | sort | uniq -c
grep -m2 -i 'blender_rate\|blend' logs/r56_se_stage1_*.out
```

### score
```bash
python evaluation/r56_score.py
```

### verdict
```bash
column -s, -t < evaluation/csv/r56_points_se.csv
column -s, -t < evaluation/csv/r56_motion.csv
ls evaluation/figures/r56_se_*.pdf
```

## Pipeline

```mermaid
flowchart LR
  SRC[FiVE source videos + VP anchors] --> S1[r56_se_stage1.sh 15-step draft tgt0.96 + attention]
  S1 --> DV[r56_div.sh DINO / LPIPS]
  S1 --> FD[r56_field.sh r49_field.py]
  DV --> FD
  FD --> R[r56_se_stage3.sh VP render]
  R --> E[r51_eval.sh r56_se]
  R --> M[r51_motion.sh r56_motion]
  E --> SC[r56_score.py]
  M --> SC
  C[R39 N=20 / N=30 curves, R49 se_max N5, R51 motion N5 + official] --> SC
  SC --> O[r56_points_se.csv, r56_motion.csv, r56_se_*.pdf]
```

## Code to touch

| File | Change |
|---|---|
| `slurm_scripts/five_bench/r56_se_stage1.sh` (new) | Copy of `r48_se_stage1.sh` full mode: `METHOD=n15_tgt096`, `--step 15 --blend_sched tgt0.96 --dump_steps 14 --measure_index 14`, VP + anchors and the other R48 flags unchanged, `--attn_out_dir $D/r56_se_attn`, `OUT=$D/r56_se_stage1`; drop the R38 bit-identity block; guards: 419 attention + 419 latents, every latent records `vp_mode == vp`, `blend_sched == tgt0.96`, `n_steps == 15`, attention present per latent. |
| `slurm_scripts/five_bench/r56_div.sh` (new) | Copy of `r53_div.sh` with `TARGET_ROOT=$D/r56_se_stage1/frames/n15_tgt096/step14`, `--latent_dir $D/r56_se_stage1/latents`, `OUT=$D/r56_div`. |
| `slurm_scripts/five_bench/r56_field.sh` (new) | Copy of `r53_field.sh` with `--attn_se $D/r56_se_attn --div_root $D/r56_div --out_root $D/r56_field_full`. |
| `slurm_scripts/five_bench/r56_se_stage3.sh` (new) | Copy of `r49_se_stage3.sh` full mode: `FIELD_DIR=$D/r56_field_full/se_max`, `METHOD=r56_max_n15_full`, `OUT_ROOT=$D/r56_se_arms`; render flags unchanged (VP + anchors, step 15, fg_boost 4, blend_power 2, chunk 21, seed 0, dump 14). |
| `evaluation/r51_units.py` | Groups `r56_se` and `r56_motion` = `(se, max_n15, $D/r56_se_arms/r56_max_n15_full/step14)`; `STEM_PREFIX` gains both with `r56`; earlier groups byte-identical. |
| `slurm_scripts/five_bench/r51_eval.sh` | the usage `case` accepts `r56_se`. |
| `slurm_scripts/five_bench/r51_motion.sh` | the usage `case` accepts `r56_motion`. |
| `slurm_scripts/five_bench/r56_se_chain.sh` (new) | Refuses if `logs/r56_se_chain_jobs.txt` exists, if the R51 scripts lack the R56 groups, or if `squeue -r` count + 7 > cap (`R56_QUEUE_CAP`, default 30); submits `stage1`, `div` (`--array=0-1`, afterok stage1), `field` (afterok div), `render` (afterok field), `eval` (`r51_eval.sh r56_se --array=0`) and `motion` (`r51_motion.sh r56_motion --array=0`), both afterok render; writes `<stage> <id>` lines headed by the commit. |
| `evaluation/r56_score.py` (new) | Reuses `r51_score` helpers (`plot`, `write_csv`, `scopes_and_names`, `motion_table`) and `r48_score.unit_values`. Arms: N5 = `r49_se_max_full` (A0 stems), N15 = `r56_se_max_n15_full`; curves R39 `r39_n{20,30}_rho{R}_vp` (harness, `r39_fiveacc_*`, `r39_clip_directional.csv`); official `r36_rho2_vp`. Points CSV rows for scopes 419, 401 and `type1`..`type6` (per-type means of each arm and curve point). Motion items: N15 `r56_motion_se_max_n15_full`, N5 `r51_motion_se_max_full`, official `r51_motion_se_r36_rho2_vp_full`; fails with one clear message when an input is missing. |
