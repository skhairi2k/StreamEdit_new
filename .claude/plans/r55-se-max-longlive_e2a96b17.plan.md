---
name: R55 — se_max on LongLive (optional)
overview: "Optional: run R49's StreamEdit method (se_max) on the LongLive backbone with R45's settings (VP, fg_boost 2, step 15) and the Self-Forcing draft logic (N=5, only t=1 anchored, target released from t=0.8). Port the cross-attention recorder to LongLive behind a 2-clip smoke gate, then draft → divergence → R49 field → render → eval + motion fidelity on the 419 pairs; read against R45's LongLive ρ curves once R45 has been scored."
task_id: R55
todos:
  - id: ll-model-recorder
    content: "LongLive_StreamEdit/wan/modules/model.py -- port Self-Forcing's R48 per-token cross-attention accumulation (accum_src_tok / accum_trg_tok / accum_*_prompt, first n_blocks_used blocks, sigma window, divided by n_blocks * n_window); no-op when attn_record is None"
    needed_by: [smoke, chain]
    status: pending
  - id: ll-pipeline-recorder
    content: "LongLive_StreamEdit/pipeline/edit_causal_inference.py -- accept attn_record in inference() and pass it (window + n_blocks_used, per-window copy as Self-Forcing's rollout) down to the model calls, mirroring Self-Forcing's edit_causal_inference.py R48 lines"
    needed_by: [smoke, chain]
    status: pending
  - id: backbone-ll-attn
    content: "evaluation/backbone.py -- drop attn_record from _SF_ONLY and forward it in the ll branch of edit_call"
    needed_by: [smoke, chain]
    status: pending
  - id: stage1-help-text
    content: "evaluation/r31_stage1.py -- --attn_out_dir help text: both backbones (was 'sf backbone only')"
    needed_by: [smoke]
    status: pending
  - id: smoke-sbatch
    content: "slurm_scripts/five_bench/r55_smoke.sh -- 2 clips (0001_bus e1, 0011_lucia e5), LongLive draft with recorder on and off: G1 latents bit-identical on vs off, G2 attention npz shapes consistent with the latent frame count and the 30x52 grid, finite, G3 Self-Forcing regression: recorder output on 0001_bus e1 equal to r48_se_attn, G4 r48-style DINO / LPIPS divergence + r49_field.py build on the 2 clips with finite q in [0, 1]; prints each gate PASS/FAIL and 'failures: N'"
    needed_by: [smoke]
    status: pending
  - id: stage1-sbatch
    content: "slurm_scripts/five_bench/r55_ll_stage1.sh -- r48_se_stage1.sh full on --backbone ll: N=5, --blend_sched tgt0.9, VP + Qwen anchors, fg_boost 2, flow_shift 1, chunk 21, seed 0, attention recorder; out r55_ll_stage1 (frames + latents) and r55_ll_attn; no bit-identity check; 419-npz guards"
    needed_by: [chain]
    status: pending
  - id: div-sbatch
    content: "slurm_scripts/five_bench/r55_div.sh -- array 0-1 (dino, lpips): r31_divergence.py as r48_div.sh's se tasks, target r55_ll_stage1/frames/n5_tgt09/step04, latents r55_ll_stage1/latents, out r55_div/se_{metric}; 419-npz guard"
    needed_by: [chain]
    status: pending
  - id: field-sbatch
    content: "slurm_scripts/five_bench/r55_field.sh -- CPU: r49_field.py --selftest, then --hosts se --cases evaluation/r7_anchor_manifest.json --attn_se r55_ll_attn --div_root r55_div --out_root r55_field_full; 419-npz guard on se_max"
    needed_by: [chain]
    status: pending
  - id: render-sbatch
    content: "slurm_scripts/five_bench/r55_ll_stage3.sh -- r31_stage3.py --backbone ll with the R45 stage-3 flags (VP + anchors, step 15, flow_shift 1.0, fg_boost 2, blend_power 2, chunk 21, seed 0, dump step 14), rho field r55_field_full/se_max, method r55_max_ll_full, out r55_ll_arms; per-type count + manifest guards"
    needed_by: [chain]
    status: pending
  - id: units-r55-groups
    content: "evaluation/r51_units.py -- groups r55_se (se, max_ll, r55_ll_arms/r55_max_ll_full/step14) and r55_motion (that unit, then official LongLive rho=2 N=15 at r45_sweep/n15/ll_rho2_n15_vp, last), stem prefix r55"
    needed_by: [chain, score]
    status: pending
  - id: chain-driver
    content: "slurm_scripts/five_bench/r55_chain.sh -- lab driver: stage1 -> div (array 0-1) -> field -> render -> eval r55_se + motion r55_motion index 0 (afterok render); motion index 1 pinned to node39 (no dependency); prints the queue count first; logs/r55_chain_jobs.txt, refuses if it exists"
    needed_by: [chain]
    status: pending
  - id: score-script
    content: "evaluation/r55_score.py -- se_max_ll and official LongLive rho=2 on R45's N=20 (equal NFE) and N=15 curves: figures r55_ll_{five_acc,clip_d,clip_tgt}.pdf, r55_points_ll.csv (419 and 401), r55_motion.csv"
    needed_by: [score]
    status: pending
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r55_smoke.sh
    status: pending
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "logs/r55_smoke_{job_id}.out: G1-G4 PASS, 'failures: 0', no Traceback in .err"
    status: pending
  - id: chain
    type: local
    wait_for: wait-smoke
    command: bash slurm_scripts/five_bench/r55_chain.sh
    sets_status: running
    status: pending
  - id: wait-chain
    type: manual
    wait_for: chain
    check_hint: "no r55 job left in squeue; every r55 log 'failures: 0'; 419 attention + latent npz, 2 x 419 divergence, 419 field npz, 419 renders with ok manifests; 1 eval unit (harness, FiVE-Acc, CLIP-D) and 2 motion units with full row counts"
    sets_status: finished
    status: pending
  - id: r45-check
    type: manual
    wait_for: wait-chain
    check_hint: "R45's eval / fiveacc / clipd done: r45_ll_rho{R}_n15_vp and r45_ll_rho{R}_n20_vp harness + r45_fiveacc_* CSVs for the 8 rho, and R45's CLIP-D CSVs, all 419 rows"
    status: pending
  - id: score
    type: local
    wait_for: r45-check
    command: python evaluation/r55_score.py
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "read r55_points_ll.csv, r55_motion.csv and the figures; compare se_max_ll vs R45 N=20 with se_max vs R39 N=20 on Self-Forcing"
    sets_status: analyzed
    status: pending
isProject: true
---

# R55: se_max on LongLive (optional)

## Context

Optional: run only once R51–R54 are done and compute is left. R55 checks that R49's StreamEdit procedure carries over to StreamEdit's second backbone, LongLive. LongLive is also distilled from Wan2.1-T2V-1.3B, so this is a robustness check, not a new host. It uses R45's LongLive settings, so R45's ρ sweeps are like-for-like references. The draft follows Self-Forcing's logic exactly: N = 5, only the t = 1 call anchored, target released from t = 0.8. R49's support needs the cross-attention recorder, which exists only on Self-Forcing, so it is ported to LongLive and gated by a 2-clip smoke test. R45's own baselines are rendered but not scored; R45's pending eval steps are run under R45, and R55's scoring waits on them.

Done when: se_max_ll is rendered and scored on the 419 pairs (harness, FiVE-Acc, CLIP-D; motion fidelity for se_max_ll and the official LongLive ρ = 2), and the trade-off figures place it on R45's N = 20 and N = 15 LongLive curves.

## Execution steps

| # | Step id | Type | Waits for | Sets status |
|---|---|---|---|---|
| 0 | *(prep)* todos `ll-model-recorder`, `ll-pipeline-recorder`, `backbone-ll-attn`, `stage1-help-text`, `smoke-sbatch`, then `stage1-sbatch`, `div-sbatch`, `field-sbatch`, `render-sbatch`, `units-r55-groups`, `chain-driver`, `score-script` (after R51 and R52's script changes) | /build-step | — | implemented |
| 1 | `smoke` | sbatch | — | — |
| 2 | `wait-smoke` | manual | smoke | — |
| 3 | `chain` | local (submits 7 queue entries on the lab cluster) | wait-smoke | running |
| 4 | `wait-chain` | manual | chain | finished |
| 5 | `r45-check` | manual (R45's eval done under R45) | wait-chain | — |
| 6 | `score` | local | r45-check | — |
| 7 | `verdict` | manual | score | analyzed |

Invoke: `/run-step R55 smoke`, `/run-step R55 wait-smoke`, `/run-step R55 chain`, `/run-step R55 wait-chain`, `/run-step R55 r45-check`, `/run-step R55 score`, `/run-step R55 verdict`.

## Decisions

| Item | Choice |
|---|---|
| Backbone | LongLive (`--backbone ll`), VP with Qwen anchors (LongLive triples the VP frame, as in R45) |
| Draft | N = 5, `--blend_sched tgt0.9` (t = 1, .8, .6, .4, .2; only the t = 1 call anchored, target released from t = 0.8, as on Self-Forcing), fg_boost 2, flow_shift 1, chunk 21, seed 0; attention recorder at r31_stage1.py defaults (first 20 blocks, σ window [0.55, 0.9]) |
| Recorder port | Self-Forcing's R48 recorder moved to LongLive's model and edit pipeline; `attn_record` no longer refused on ll; Self-Forcing behaviour unchanged (smoke G3) |
| Smoke gate | 2 clips (0001_bus e1, 0011_lucia e5): G1 recorder read-only (latents bit-identical on vs off), G2 attention shapes / finiteness, G3 Self-Forcing regression vs `r48_se_attn`, G4 divergence + field build; the full chain is submitted only after all PASS |
| Divergence / field | DINOv3 + LPIPS as `r48_div.sh`; `r49_field.py` unchanged (caps 0.55 / 1.0, λ 0.5, q = S·max, budget-linear inverse on [2, 50] on the 15-step t_next grid, which R45's stage 3 already uses for LongLive) |
| Render | `r31_stage3.py --backbone ll`, step 15, flow_shift 1.0, fg_boost 2, blend_power 2, chunk 21, seed 0, dump step 14; NFE 5 + 15 = 20 |
| References | R45 LongLive VP sweeps: N = 20 (equal NFE, 8 ρ) and N = 15 (native, 8 ρ, ρ = 2 = official LongLive), scored under R45's own plan (`wait-renders` → `eval` → `fiveacc` → `clipd`); R55 does not score them |
| Metrics | 9 harness metrics, FiVE-Acc, CLIP-D on se_max_ll; motion fidelity on se_max_ll and official LongLive ρ = 2 |
| Official renders on /store | `r45_sweep` is a /store link → its motion unit is pinned to `node39` (fail-fast preflight) |
| Scoring | plain per-clip means over 419 and over the 401 outside `r49_cases.json`; figures x = alignment (FiVE-Acc, CLIP-D, CLIP-T), y = bg LPIPS; no CIs, no grids, no go/no-go step (user choice) |
| Reuse | eval / motion through R51's scripts with R52's stem-prefix column; R55 builds after R51 and R52's changes |
| Slurm | `--partition=L40S` / `CPU`, `--exclude=node52`; smoke 2 h, draft 12 h, render 20 h, eval 12 h, motion 12 h |
| Outputs | `$D/r55_smoke/`, `$D/r55_ll_stage1/`, `$D/r55_ll_attn/`, `$D/r55_div/se_{dino,lpips}/`, `$D/r55_field_full/se_max/`, `$D/r55_ll_arms/r55_max_ll_full/step14/`, `evaluation/csv/*r55_*`, `evaluation/figures/r55_ll_*.pdf` |

## Step commands

### smoke
```bash
sbatch slurm_scripts/five_bench/r55_smoke.sh
```

### wait-smoke
```bash
squeue -u $USER
grep -E 'G[1-4]|failures' logs/r55_smoke_*.out
grep -c Traceback logs/r55_smoke_*.err
```

### chain
```bash
bash slurm_scripts/five_bench/r55_chain.sh
cat logs/r55_chain_jobs.txt
```

### wait-chain
```bash
cat logs/r55_chain_jobs.txt
squeue -u $USER
grep -hE 'failures|unit failures' logs/r55_*.out | sort | uniq -c
grep -hE 'failures|unit failures' $(ls -t logs/r51_eval_*.out logs/r51_motion_*.out | head -3) | sort | uniq -c
```

### r45-check
```bash
for N in 15 20; do for R in 2 3 4 6 8 10 20 50; do
  for T in 1 2 3 4 5 6; do f=evaluation/csv/edit${T}_FiVE_r45_ll_rho${R}_n${N}_vp_frame_stride8.csv; [ -s "$f" ] || echo "MISSING $f"; done
done; done
ls evaluation/csv/ | grep -E 'r45_(fiveacc|clip_directional)' | head
```

### score
```bash
python evaluation/r55_score.py
```

### verdict
```bash
column -s, -t < evaluation/csv/r55_points_ll.csv
column -s, -t < evaluation/csv/r55_motion.csv
ls evaluation/figures/r55_ll_*.pdf
```

## Pipeline

```mermaid
flowchart LR
  P[recorder port: LongLive model + pipeline + backbone.py] --> SM[r55_smoke.sh G1-G4]
  SM --> S1[r55_ll_stage1.sh LongLive draft + attention]
  S1 --> DV[r55_div.sh DINO / LPIPS]
  S1 --> FD[r55_field.sh r49_field.py]
  DV --> FD
  FD --> R[r55_ll_stage3.sh LongLive render]
  R --> E[r51_eval.sh r55_se]
  R --> M[r51_motion.sh r55_motion]
  OFF[r45_sweep n15 rho2 on /store, node39] --> M
  R45[R45 eval / fiveacc / clipd, run under R45] --> SC[r55_score.py]
  E --> SC
  M --> SC
  SC --> O[r55_points_ll.csv, r55_motion.csv, r55_ll_*.pdf]
```

## Code to touch

| File | Change |
|---|---|
| `LongLive_StreamEdit/wan/modules/model.py` | Port the R48 recorder from `Self-Forcing_StreamEdit/wan/modules/model.py`: when an `attn_record` dict is passed and the call's σ is inside `window`, accumulate trigger-token cross-attention for the source and target branches over the first `n_blocks_used` blocks into `accum_src_tok`, `accum_trg_tok`, `accum_src_prompt`, `accum_trg_prompt`, normalised as on Self-Forcing; `None` → no extra op. |
| `LongLive_StreamEdit/pipeline/edit_causal_inference.py` | `inference(..., attn_record=None)`; pass it to the denoising calls exactly as Self-Forcing's `edit_causal_inference.py` does (lines tagged R48), including the per-window copy of `window` / `n_blocks_used`; account for the tripled VP frame when indexing frames. |
| `evaluation/backbone.py` | Remove `"attn_record"` from `_SF_ONLY`; forward `attn_record` in the ll branch of `edit_call`. |
| `evaluation/r31_stage1.py` | `--attn_out_dir` help: "both backbones". |
| `slurm_scripts/five_bench/r55_smoke.sh` (new) | L40S, 2 h; two `r31_stage1.py --backbone ll` runs (recorder on / off) on the 2 clips into `$D/r55_smoke/{on,off}`; G1 `np.array_equal` on latents; G2 attention keys and shapes vs latent frame count, 1560 tokens, finite; G3 `r31_stage1.py --backbone sf` with recorder on 0001_bus e1 vs `r48_se_attn/edit1/0001_bus.npz` (`np.array_equal`); G4 `r31_divergence.py` (dino, lpips) + `r49_field.py` on the smoke roots, q finite in [0, 1]; `failures: N`. |
| `slurm_scripts/five_bench/r55_ll_stage1.sh` (new) | Copy of `r48_se_stage1.sh` full mode with `--backbone ll --fg_boost_factor 2`, drop the R38 bit-identity block; `OUT=$D/r55_ll_stage1`, `ATTN=$D/r55_ll_attn`; guards: 419 attention + 419 latent npz. |
| `slurm_scripts/five_bench/r55_div.sh` (new) | Array 0-1 = `se_dino` / `se_lpips` as `r48_div.sh`; `TARGET_ROOT=$D/r55_ll_stage1/frames/n5_tgt09/step04`, `--latent_dir $D/r55_ll_stage1/latents`, `--out_root $D/r55_div`; 419-npz guard. |
| `slurm_scripts/five_bench/r55_field.sh` (new) | CPU, 4 cpus, 32G, 2 h; selftest, then `r49_field.py --hosts se --cases evaluation/r7_anchor_manifest.json --attn_se $D/r55_ll_attn --div_root $D/r55_div --out_root $D/r55_field_full`; 419-npz guard. |
| `slurm_scripts/five_bench/r55_ll_stage3.sh` (new) | Copy of `r49_se_stage3.sh` full mode with `--backbone ll --fg_boost_factor 2`; `FIELD_DIR=$D/r55_field_full/se_max`, `METHOD=r55_max_ll_full`, `OUT_ROOT=$D/r55_ll_arms`; same guards. |
| `evaluation/r51_units.py` | Groups `r55_se` = `(se, max_ll, $D/r55_ll_arms/r55_max_ll_full/step14)`; `r55_motion` = that unit, then `(se, ll_rho2_n15_vp, $D/r45_sweep/n15/ll_rho2_n15_vp)`; prefix `r55`. |
| `slurm_scripts/five_bench/r55_chain.sh` (new) | Refuses if `logs/r55_chain_jobs.txt` exists; prints the queue count; submits `stage1`, `div` (`--array=0-1`, afterok stage1), `field` (afterok div), `render` (afterok field), `eval` (`r51_eval.sh r55_se --array=0`) and `motion` (`r51_motion.sh r55_motion --array=0`), both afterok render, `motion_official` (`--array=1 --nodelist=node39`); writes `<stage> <id>` lines headed by the commit. |
| `evaluation/r55_score.py` (new) | Reuses `r49_score` helpers (`point`, `plot`); LongLive curves from R45's CSVs (`r45_ll_rho{R}_n{15,20}_vp`, `r45_fiveacc_*`, R45 CLIP-D), official = ρ 2 at N = 15; se_max_ll from `r55_*`; per-clip means on 419 and 401; motion table from the two `r55_motion_*` stems; fails clearly if an R45 CSV is missing. |
