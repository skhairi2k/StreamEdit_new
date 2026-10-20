---
name: "R41: Binary vs Continuous Rho Mapping"
overview: "Ablation of the divergence routing on the FINAL draft length N (R35's N=15 or R38's N=5, chosen after R38 is analyzed): keep that draft's absolute-normalised DINO/LPIPS fields, replace the budget-linear m -> rho mapping with a two-level map (rho=2 if m<theta, rho=50 if m>=theta), sweep theta (default {0.3, 0.5, 0.7}) on every surviving combo, render, score as R35, and read the binary theta-curves against the continuous arms at the SAME N and R36's VP rho curve."
task_id: R41
todos:
  - id: config
    content: "slurm_scripts/five_bench/r41_config.sh (new, sourced by every R41 script; Python scripts take --draft_n). Holds DRAFT_N (EMPTY on purpose until gate-draft), COMBOS, THETAS (default 0.3 0.5 0.7). COMBOS use ONE vocabulary at both N: {dino, lpips} x {unblended, tgt09} = dino_unblended dino_tgt09 lpips_unblended lpips_tgt09 (default: all four). tgt09 at N=15 IS R35's first2 (R38 smoke G1, byte-identical), so the regime is the same at both N; only the upstream folder names differ, and they appear ONLY in a lookup here: SRC_NAME[combo] = R35 name at N=15 (dino_unblended, dino_first2, lpips_unblended, lpips_first2) or R38 name at N=5 (dino_n5_unblended, dino_n5_tgt09, lpips_n5_unblended, lpips_n5_tgt09). Resolves from DRAFT_N: N=15 -> fields r35_div_abs/SRC_NAME, continuous rho r35_rho/SRC_NAME, continuous renders r35_arms/r35_SRC_NAME, continuous CSV stems r35_SRC_NAME / r35_fiveacc_SRC_NAME / r35_clip_directional.csv; N=5 -> the r38_ equivalents and r38_clip_directional.csv. ARMS = COMBOS x THETAS, named {combo}_t03/_t05/_t07 at both N. Refuses (exit 1) if DRAFT_N is unset or not in {5, 15}, or a COMBO is not one of the four. Prints a banner '[r41] DRAFT_N=<N> fields=<root> combos=<...> thetas=<...>' that every script echoes at startup. `bash r41_config.sh --max_index` prints len(ARMS)-1 for --array."
    status: completed
  - id: mapping-binary
    content: "evaluation/r31_rho_map.py -- add `--mapping binary`: rho = tau_min if m < thresh else tau_max. `--thresh` default becomes None; linear_threshold falls back to its old 0.4 (old callers unchanged), binary REFUSES to run without an explicit --thresh. Record mapping / thresh / tau_at_m_half in the npz as the other modes do. budget_linear must stay byte-identical (smoke gate G1)."
    status: completed
  - id: fgstats
    content: "evaluation/r41_fg_stats.py (new, numpy only, --draft_n required). Two modes. --precheck (used at gate-draft, needs no rho): from the chosen N's div_abs, per combo and per theta in a list, FG fraction (m >= theta) as plain per-clip mean, per edit type, and fraction of clips with < 0.5% FG; writes evaluation/csv/r41_precheck_n{N}.csv. Default mode (after rhomap): per arm FG fraction (rho == 50) and mean source budget mean_i A_disc(rho) (r30_b_map.a_disc, 15-step grid) next to its continuous counterpart's budget at the SAME N; writes evaluation/csv/r41_n{N}_fg_stats.csv."
    status: completed
  - id: rhomap-script
    content: "slurm_scripts/five_bench/r41_rhomap.sh -- CPU batch job (r35_absnorm.sh header and conventions), sources r41_config.sh. Preflight: chosen N's div_abs complete for every COMBO (419, 100/100/100/100/9/10). Then one r31_rho_map.py --mapping binary --thresh <theta> --tau_min 2 --tau_max 50 --step 15 --flow_shift 1.0 per arm into r41_rho/n{N}/{arm}. Post-run guard: 419 per arm, exact per-type breakdown. No r31_div_abs.py call -- the chosen draft's normalised fields are reused as-is."
    status: completed
  - id: grids
    content: "evaluation/r41_grids.py (new, --draft_n required) -- one page per clip (all 419); per combo rows [m, rho continuous, rho at each theta] (20 rows at 4 combos x 3 thetas), 8 evenly spaced latent frames; title states DRAFT_N. rho rows share the [2,50] inferno scale. Imports evenly_spaced (and colormap conventions) from r31_stage3_grids.py without modifying it. Hard-fails on any missing npz. Writes evaluation/figures/r41_n{N}_grids/."
    status: completed
  - id: smoke
    content: "slurm_scripts/five_bench/r41_smoke.sh -- 1x L40S, sources r41_config.sh, 4 gates. CPU: G1 r31_rho_map.py --mapping budget_linear rebuilt into the smoke dir for every COMBO == the chosen N's stored continuous rho (np.array_equal, 419 per combo); G2 every r41_rho/n{N} npz has values in {2,50}, (rho==50) == (m>=thresh) against the chosen div_abs, and recorded thresh matches its arm suffix. GPU, on the 2 clips of r38_smoke.sh (0001_bus, 0034_cows; case entries copied from its CASES block), using the LAST combo in COMBOS: G3 r31_stage3.py with that combo's continuous rho == its stored continuous render at the same N (sha256 of every step14 PNG) -- today's tree, including the uncommitted R38 pipeline edits, reproduces the continuous arm the binary arms are compared to; G4 the same render with that combo's t05 binary rho completes with the same frame count and is NOT identical to G3. Prints GATE G1..G4 PASS/FAIL. Output under r41_smoke/n{N}."
    status: completed
  - id: stage3-script
    content: "slurm_scripts/five_bench/r41_stage3.sh -- r35_stage3.sh verbatim except: sources r41_config.sh; one ARM per array task, theta-major with t05 first (all combos at t05, then t03, then t07), combo order as COMBOS; RHO_ROOT=r41_rho/n{N}, OUT_ROOT=r41_arms/n{N}, METHOD=r41_n{N}_{arm}; rejects an array id >= len(ARMS); preflight checks r41_rho/n{N}/{arm} only. Stage 3 stays 15 steps whatever N is (N is the DRAFT length)."
    status: completed
  - id: eval-scripts
    content: "slurm_scripts/five_bench/r41_eval.sh, r41_fiveacc.sh (one ARM per array task, R35 resources and metric lists verbatim, stems r41_n{N}_{arm} / r41_fiveacc_n{N}_{arm}), r41_clipd.sh (one THETA per array task, all COMBOS per task -- R35's single-job load at 4 combos -- writing evaluation/csv/r41_n{N}_clip_directional_t{03,05,07}.csv). All source r41_config.sh."
    status: completed
  - id: score-script
    content: "evaluation/r41_score.py (new, --draft_n required) -- R35's 5 figures. Loaders imported from r35_score (as r38_score.py does), plain per-clip mean over 419. One PDF per x metric, one panel per combo, shared axes: R36 VP curve (gray), the SAME-N continuous point (R35 arms for N=15, R38 arms for N=5), binary theta-curve joined in theta order and labelled. Figure titles state DRAFT_N. Hard-fails if any continuous or binary input is missing. Writes evaluation/csv/r41_n{N}_arms.csv."
    status: completed
steps:
  - id: gate-draft
    type: manual
    check_hint: "USER DECISION -- REMIND THE USER: R41 runs on the FINAL draft length; ask explicitly 'N=5 or N=15?' before anything else. Precondition: R38 is `analyzed` in daily.md (r38_arms.csv + figures exist). Read evaluation/csv/r38_arms.csv and r38_div_stats.csv; then run `python evaluation/r41_fg_stats.py --precheck --draft_n 5` and `--draft_n 15` (FG fraction at each theta, clips with < 0.5% FG). User sets in slurm_scripts/five_bench/r41_config.sh: DRAFT_N (5 or 15), COMBOS (all four by default; at N=5, remove dino_tgt09 / lpips_tgt09 only if R38's gate-lowN dropped them), THETAS (keep 0.3 0.5 0.7 unless the precheck shows a cut collapsing most clips to rho=2). Confirm with `bash slurm_scripts/five_bench/r41_config.sh` (banner) and record the choice in daily.md's Run log."
    status: completed
    completed_at: 2026-10-12
  - id: rhomap
    type: sbatch
    wait_for: gate-draft
    command: sbatch slurm_scripts/five_bench/r41_rhomap.sh
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r41_rho
    status: pending
  - id: wait-rhomap
    type: manual
    wait_for: rhomap
    check_hint: "grep -E 'DRAFT_N|npz total|failures' logs/r41_rhomap_{job_id}.out -- banner shows the N chosen at gate-draft; len(ARMS) x 419, mismatched types=0, failures: 0"
    status: pending
  - id: fgstats
    type: local
    wait_for: wait-rhomap
    command: python evaluation/r41_fg_stats.py --draft_n "$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)"
    output_paths:
      - evaluation/csv
    status: pending
  - id: grids
    type: local
    wait_for: wait-rhomap
    command: python evaluation/r41_grids.py --draft_n "$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)" --cases evaluation/r7_anchor_manifest.json
    output_paths:
      - evaluation/figures
    status: pending
  - id: smoke
    type: sbatch
    wait_for: wait-rhomap
    command: sbatch slurm_scripts/five_bench/r41_smoke.sh
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r41_smoke
    status: pending
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "grep -E 'DRAFT_N|GATE' logs/r41_smoke_{job_id}.out -- banner N matches gate-draft; G1..G4 all PASS; 0 Traceback"
    status: pending
  - id: stage3
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) slurm_scripts/five_bench/r41_stage3.sh
    sets_status: running
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r41_arms
    status: pending
  - id: wait-stage3
    type: manual
    wait_for: stage3
    check_hint: "every task: grep -h 'DRAFT_N\\|step14 dirs total\\|non-ok manifest\\|failures:' logs/r41_stage3_{job_id}_*.out -- one N throughout, 419 each, 0 non-ok rows, failures: 0"
    sets_status: finished
    status: pending
  - id: eval
    type: sbatch
    wait_for: wait-stage3
    command: sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) slurm_scripts/five_bench/r41_eval.sh
    status: pending
  - id: fiveacc
    type: sbatch
    wait_for: wait-stage3
    command: sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) slurm_scripts/five_bench/r41_fiveacc.sh
    status: pending
  - id: clipd
    type: sbatch
    wait_for: wait-stage3
    command: sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_theta_index) slurm_scripts/five_bench/r41_clipd.sh
    status: pending
  - id: wait-eval
    type: manual
    wait_for: [eval, fiveacc, clipd]
    check_hint: "6 per-type metric CSVs + 6 five_acc CSVs per arm, 419 rows per arm, 0 NaN; one r41_n{N}_clip_directional_t*.csv per theta with len(COMBOS) x 419 rows; 0 Error:/OOM lines"
    status: pending
  - id: figures
    type: local
    wait_for: wait-eval
    command: python evaluation/r41_score.py --all --draft_n "$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)"
    output_paths:
      - evaluation/csv
      - evaluation/figures
    sets_status: analyzed
    status: pending
isProject: true
---

# R41: Binary vs Continuous Rho Mapping

## Context

The divergence routing maps each token's divergence `m` to an exponent with a continuous, budget-linear m -> rho map. R41 asks whether the graded timing matters, or whether a two-level map built from the same field does as well: rho=2 below a cut theta, rho=50 above. Everything upstream of the mapping (draft, divergence network, normalisation) is reused from disk. A binary map needs a cut, and any single cut can be called badly chosen, so theta is swept and the binary arm is read as a curve.

The ablation must be run on the method as shipped, so R41 is built on the **final** stage-1 draft length N: R35's fields if N=15, R38's if N=5. N is chosen at `gate-draft`, after R38 is analyzed; both the binary arms and the continuous arms they are compared to use that same N.

**Done when:** the binary arms (surviving combos x theta) at the chosen N are rendered and scored on all 419 pairs, and the Pareto figures place each combo's binary theta-curve next to its continuous point at the same N and R36's VP rho curve.

## Execution steps

| # | Step id | Type | What it does | Sets status |
|---|---------|------|--------------|-------------|
| — | *(prep)* | — | `/build-step` the 9 `todos` (can be done before R38 finishes) | `implemented` |
| 0 | `gate-draft` | manual | **User** chooses N (5 or 15), combos and thetas in `r41_config.sh`, after R38 is `analyzed` | |
| 1 | `rhomap` | sbatch | CPU: binary rho trees from the chosen N's fields | |
| 2 | `wait-rhomap` | manual | len(ARMS) x 419 npz; banner N correct | |
| 3 | `fgstats` | local | FG fraction + source budget per arm vs continuous | |
| 4 | `grids` | local | 419 pages | |
| 5 | `smoke` | sbatch | 4 gates: mapping regression, binary correctness, continuous render parity, binary render | |
| 6 | `wait-smoke` | manual | G1-G4 PASS | |
| 7 | `stage3` | sbatch | One arm per array task, 419 pairs | `running` |
| 8 | `wait-stage3` | manual | 419 step14 dirs per arm | `finished` |
| 9 | `eval` | sbatch | 9-metric harness, one arm per task | |
| 10 | `fiveacc` | sbatch | FiVE-Acc, one arm per task | |
| 11 | `clipd` | sbatch | CLIP-D, one theta per task | |
| 12 | `wait-eval` | manual | All CSVs complete | |
| 13 | `figures` | local | 5 Pareto figures + arms CSV | `analyzed` |

```
/run-step R41 gate-draft   /run-step R41 rhomap   /run-step R41 smoke
/run-step R41 stage3       /run-step R41 eval     /run-step R41 fiveacc
/run-step R41 clipd        /run-step R41 figures
```

## Decisions

| Topic | Decision |
|-------|----------|
| Question tested | Continuous (budget-linear) vs binary m -> rho, with draft, divergence network, regime and normalisation held fixed |
| Draft length N | **The final method's N, not fixed in this plan.** Chosen at `gate-draft` once R38 is `analyzed`; binary and continuous arms always share it |
| When to run | After R38. Code can be built before; no step runs until `gate-draft` sets `DRAFT_N` |
| Single source of N | `slurm_scripts/five_bench/r41_config.sh` (`DRAFT_N` empty by default; every script refuses to run without it and prints it in a banner); Python scripts take `--draft_n` |
| Combos | Two regimes x two divergence networks, same names at both N: `dino_unblended`, `dino_tgt09`, `lpips_unblended`, `lpips_tgt09`. `tgt09` at N=15 is R35's `first2` (byte-identical, R38 smoke G1). At N=5 the `tgt09` pair is removed only if R38's gate-lowN dropped it |
| Inputs at N=15 | Fields `r35_div_abs`, continuous rho `r35_rho`, continuous renders and CSVs `r35_*` (upstream names `*_first2` for `tgt09`) |
| Inputs at N=5 | Fields `r38_div_abs`, continuous rho `r38_rho`, continuous renders and CSVs `r38_*` (upstream names `*_n5_unblended`, `*_n5_tgt09`) |
| Upstream names | Appear only in `r41_config.sh`'s lookup; every R41 path, stem, figure label and CSV row uses the four combo names above |
| Normalisation | The chosen draft's own absolute `div_max` (0.65 DINO / 1.0 LPIPS at both N) |
| Mapping | `rho = 2 if m < theta else 50`; `tau_min` / `tau_max` identical to the continuous arm |
| Cuts | Default theta in {0.3, 0.5, 0.7}, read as a curve; no single cut is defended. Set from N=15 data (at m >= 0.8 the `tgt09` combos have < 0.5% FG in 36-48% of clips); re-checked at `gate-draft` with `--precheck` on the chosen N |
| Arms | COMBOS x THETAS (12 at 4 combos), named `{combo}_t03 / _t05 / _t07` (e.g. `lpips_tgt09_t05`); render method `r41_n{N}_{arm}` |
| Array order | theta-major, t05 first, combos in COMBOS order |
| Render config | Identical to `r35_stage3.sh`: vp, anchors, stage 3 at `--step 15` whatever the draft N, `flow_shift 1.0`, `fg_boost 4`, `blend_power 2`, chunk 21, seed 0, `--dump_steps 14`, L40S |
| Continuous reference | Stored continuous arms at the chosen N (R35 or R38), not re-rendered. Smoke G3 proves today's tree reproduces them byte-for-byte |
| Pipeline code | No change. Only `r31_rho_map.py` gains a mode |
| Evaluation | R35's three jobs, same metric lists: `eval` (9 metrics), `fiveacc`, `clipd` |
| Aggregation | Plain per-clip mean over 419, never `{stem}_avg.csv`; x = alignment, y = `lpips_unedit_part` |
| Figures | R35's 5 x-metrics; one panel per combo: R36 curve, same-N continuous point, binary theta-curve; titles state N |
| Significance test | None (user decision); figures and per-arm means only |
| FG statistics | `r41_precheck_n{N}.csv` at the gate; `r41_n{N}_fg_stats.csv` after rhomap (FG fraction, source budget next to the continuous arm's) |
| Grids | All 419 pages, per combo rows m / continuous rho / binary rho per theta, 8 frames, `evaluation/figures/r41_n{N}_grids/` |
| Naming | Every output carries `n{N}`: `r41_rho/n{N}`, `r41_arms/n{N}`, `r41_smoke/n{N}`, stems `r41_n{N}_*`, so N=5 and N=15 outputs can never mix |
| Disk | ~15 GB per arm (measured on `r35_lpips_first2`) -> ~180 GB at 12 arms; 38 TB free |
| Out of scope | Running both N; per-clip, Otsu or budget-matched cuts; segmentation-mask routing (R26's row of Table 3) |

## Step commands

### gate-draft
```bash
# precondition: R38 row is `analyzed` in daily.md
cat evaluation/csv/r38_arms.csv evaluation/csv/r38_div_stats.csv
python evaluation/r41_fg_stats.py --precheck --draft_n 15
python evaluation/r41_fg_stats.py --precheck --draft_n 5
# user edits DRAFT_N / COMBOS / THETAS in slurm_scripts/five_bench/r41_config.sh, then:
bash slurm_scripts/five_bench/r41_config.sh          # prints the banner
```

### rhomap
```bash
sbatch slurm_scripts/five_bench/r41_rhomap.sh
```

### wait-rhomap
```bash
grep -E 'DRAFT_N|npz total|failures' logs/r41_rhomap_<job_id>.out
```

### fgstats
```bash
python evaluation/r41_fg_stats.py --draft_n "$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)"
```

### grids
```bash
python evaluation/r41_grids.py --draft_n "$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)" \
  --cases evaluation/r7_anchor_manifest.json
```

### smoke
```bash
sbatch slurm_scripts/five_bench/r41_smoke.sh
```

### wait-smoke
```bash
grep -E 'DRAFT_N|GATE' logs/r41_smoke_<job_id>.out
grep -c Traceback logs/r41_smoke_<job_id>.err
```

### stage3
```bash
sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) slurm_scripts/five_bench/r41_stage3.sh
```

### wait-stage3
```bash
grep -h 'DRAFT_N\|step14 dirs total\|non-ok manifest\|failures:' logs/r41_stage3_<job_id>_*.out
```

### eval
```bash
sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) slurm_scripts/five_bench/r41_eval.sh
```

### fiveacc
```bash
sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) slurm_scripts/five_bench/r41_fiveacc.sh
```

### clipd
```bash
sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_theta_index) slurm_scripts/five_bench/r41_clipd.sh
```

### wait-eval
```bash
grep -h 'OK\|FAILED' logs/r41_eval_<job_id>_*.out logs/r41_fiveacc_<job_id>_*.out logs/r41_clipd_<job_id>_*.out
N=$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)
ls evaluation/csv/edit*_FiVE_r41_n${N}_*_frame_stride8.csv | wc -l     # expect 12 x len(ARMS)
```

### figures
```bash
python evaluation/r41_score.py --all --draft_n "$(bash slurm_scripts/five_bench/r41_config.sh --draft_n)"
```

## Pipeline

```mermaid
flowchart LR
  R38["R38 analyzed"] --> GATE["gate-draft<br/>DRAFT_N, COMBOS, THETAS<br/>in r41_config.sh"]
  GATE --> DA["fields at N<br/>r35_div_abs (N=15) / r38_div_abs (N=5)"]
  DA --> RM["r31_rho_map.py --mapping binary"]
  RM --> RHO["r41_rho/n{N}/{arm}"]
  CRHO["continuous rho at N<br/>r35_rho / r38_rho"] --> FG["r41_fg_stats.py"]
  RHO --> FG --> FGC["r41_n{N}_fg_stats.csv"]
  DA --> GR["r41_grids.py"]
  RHO --> GR
  CRHO --> GR --> GRO["r41_n{N}_grids/"]
  RHO --> SM["r41_smoke.sh G1-G4"]
  RHO --> S3["r31_stage3.py<br/>one arm per task"] --> ARMS["r41_arms/n{N}"]
  ARMS --> EV["evaluate.py (9 metrics)"]
  ARMS --> FA["evaluate.py five_acc"]
  ARMS --> CD["r33_clip_directional.py"]
  EV --> SC["r41_score.py"]
  FA --> SC
  CD --> SC
  CCSV["continuous CSVs at N<br/>r35_* / r38_*"] --> SC
  R36["r36_rho_sweep.csv"] --> SC
  SC --> OUT["r41_n{N}_*_vs_lpips.pdf (5)<br/>r41_n{N}_arms.csv"]
```

## Code to touch

| File | Change |
|------|--------|
| `slurm_scripts/five_bench/r41_config.sh` | **new** — `DRAFT_N=` (empty), `COMBOS`, `THETAS`; resolves roots / stems and the combo -> upstream-name lookup from N; validates; banner; `--draft_n`, `--max_index`, `--max_theta_index` print modes |
| `evaluation/r41_config.py` | **new** (added at build) — Python side of `r41_config.sh`: runs `--json N`, refuses an N that differs from the configured `DRAFT_N` (except `--precheck`); honours the test-only `R41_CONFIG` / `R41_TEST_D` overrides |
| `evaluation/r31_rho_map.py` | Add `--mapping binary` (`binary_tau`: `np.where(m >= thresh, tau_max, tau_min)`, NaN guard, `0 < thresh < 1`); `--thresh` default None, linear_threshold keeps 0.4, binary requires it; record `mapping`, `thresh`, `tau_at_m_half` |
| `evaluation/r41_fg_stats.py` | **new** — numpy only, `--draft_n` required, `--precheck` mode on div_abs; imports `a_disc`, `t_next_schedule` from `r30_b_map` |
| `slurm_scripts/five_bench/r41_rhomap.sh` | **new** — CPU batch (`--partition=CPU`, 4 CPUs, 16G, 2 h), r35_absnorm.sh conventions; preflight + post-run 419 guard |
| `evaluation/r41_grids.py` | **new** — imports helpers from `r31_stage3_grids.py` without modifying it |
| `slurm_scripts/five_bench/r41_smoke.sh` | **new** — 1x L40S, 64G, 2 h; CPU gates G1-G2 then 2-clip renders for G3-G4 |
| `slurm_scripts/five_bench/r41_stage3.sh` | **new** — copy of `r35_stage3.sh`; no `#SBATCH --array` (passed on the command line from `--max_index`) |
| `slurm_scripts/five_bench/r41_eval.sh` | **new** — copy of `r35_eval.sh`, one arm per task |
| `slurm_scripts/five_bench/r41_fiveacc.sh` | **new** — copy of `r35_fiveacc.sh`, one arm per task |
| `slurm_scripts/five_bench/r41_clipd.sh` | **new** — copy of `r35_clipd.sh`, one theta per task, all COMBOS |
| `evaluation/r41_score.py` | **new** — loaders from `r35_score`; one panel per combo; continuous inputs resolved from `--draft_n` |

`r41_config.sh` sketch:

```bash
DRAFT_N=            # set at gate-draft: 5 or 15
COMBOS=(dino_unblended dino_tgt09 lpips_unblended lpips_tgt09)   # same names at both N
THETAS=(0.3 0.5 0.7)
case "$DRAFT_N" in
  15) DIV_ABS_ROOT=.../r35_div_abs; CONT_RHO_ROOT=.../r35_rho; CONT_PREFIX=r35
      declare -A SRC_NAME=([dino_unblended]=dino_unblended [dino_tgt09]=dino_first2
                           [lpips_unblended]=lpips_unblended [lpips_tgt09]=lpips_first2) ;;
  5)  DIV_ABS_ROOT=.../r38_div_abs; CONT_RHO_ROOT=.../r38_rho; CONT_PREFIX=r38
      declare -A SRC_NAME=([dino_unblended]=dino_n5_unblended [dino_tgt09]=dino_n5_tgt09
                           [lpips_unblended]=lpips_n5_unblended [lpips_tgt09]=lpips_n5_tgt09) ;;
  *)  echo "[r41] DRAFT_N not set -- run gate-draft first (N=5 or N=15?)"; exit 1 ;;
esac
# fields of combo c: $DIV_ABS_ROOT/${SRC_NAME[$c]};  continuous arm: ${CONT_PREFIX}_${SRC_NAME[$c]}
# ARMS = for theta in THETAS (t05 first): for c in COMBOS: "${c}_t${theta/./}"
```

Gate pseudocode for `r41_smoke.sh`:

```python
# G1: continuous path unchanged
for c in COMBOS: rebuild budget_linear into SMOKE/rho_bl/c; assert all(np.array_equal(new, CONT_RHO/SRC_NAME[c]/clip))
# G2: binary maps are what they claim
for arm in ARMS: z = load(r41_rho/n{N}/arm/clip); m = load(DIV_ABS/SRC_NAME[combo]/clip)["m"]
    assert set(np.unique(z["rho"])) <= {2., 50.} and np.array_equal(z["rho"] == 50, m >= z["thresh"])
# G3: continuous render parity at N (sha256 of every step14 PNG, 2 clips, last combo)
# G4: binary t05 render: same frame count as G3, not byte-identical to G3
```
