---
name: R36 — Sweep Rho in Visual Prompting
overview: "Extend R7 (StreamEdit + paper §4.5 visual prompting, full FiVE-Bench, 419 pairs) from the single paper value blend_power rho=2 to R26's uniform-baseline grid rho in {2,3,4,6,8,10,20,50}. All 8 rho values, including rho=2, are rendered fresh; R7's stored renders and scores are not reused (user decision, 2026-09-24). 8-task L40S render array (one task per rho, each looping all 6 edit types), then FiVE eval without motion fidelity (R26's 9-metric pass + a separate FiVE-Acc pass, both L40S/A100 — no H100 available) plus directional CLIP (CLIP-D) in a single job cloned from r35_clipd.sh, summarized as a plain per-clip mean into one per-rho CSV and three Pareto figures (x = CLIP-T / CLIP-D / FiVE-Acc, y = background LPIPS). Two free correctness gates: the 22 cases.json clips must be sha256-identical to R26's diagonal renders (same sampler, same per-pair seed), and index-0 pairs at rho=2 must be identical to R7's stored renders."
task_id: R36
todos:
  - id: smoke-cases
    content: "evaluation/r36_smoke_cases.json — 2-entry subset of evaluation/cases.json: 0001_bus (edit1) and 0007_guitar-violin (edit5). Both are index 0 of their edit{T}_FiVE.json, so R7's pre-reseed render of them used the same noise as today's runner."
    status: completed
  - id: smoke-script
    content: "slurm_scripts/five_bench/r36_smoke.sh — render the 2 smoke clips at rho=2 and rho=3 (vp, anchors, --cases_json r36_smoke_cases.json) into r36_smoke/, then call evaluation/r36_check_parity.py --smoke for 3 gates: G1 rho=3 sha256 == R26 taubg3_taufg3_vp on both clips; G2 rho=2 sha256 == R7 r7_visual_prompting on both clips; G3 rho=3 != rho=2 (flag reaches the bridge). Exit non-zero on any FAIL."
    status: completed
  - id: infer-script
    content: "slurm_scripts/five_bench/r36_infer.sh — array 0-7, one task per rho (RHOS=(2 3 4 6 8 10 20 50), rho = RHOS[TID]), each task looping all 6 edit types (419 pairs); run_fivebench.py --method r36_rho{R}_vp --vp_mode vp --first_frame_edit_dir anchors --blend_power R, no --cases_json (full bench); a failed edit type does not stop the loop, the task exits non-zero at the end; L40S, --mem=64G, --time=20:00:00 (~10.5 h at R35's measured ~1.5 min/clip; partition max 24 h); conda via ~/anaconda3/etc/profile.d/conda.sh (not .bashrc — the R25 job 960685 failure); post-run count of real frame dirs vs expected 100/100/100/100/9/10, non-zero exit on shortfall."
    status: completed
  - id: parity-script
    content: "evaluation/r36_check_parity.py — (a) --smoke mode: the 3 smoke gates above; (b) full mode: for each rho, sha256 every PNG of the 22 cases.json clips in r36_rho_sweep/r36_rho{R}_vp against R26 r26_spatial_tau/taubg{R}_taufg{R}_vp, plus the index-0 pair of each edit type at rho=2 against R7; also asserts 419 real dirs + all-'ok' _manifest.csv per arm. Writes evaluation/csv/r36_parity.csv (arm, edit_type, clip, ref, identical)."
    status: completed
  - id: eval-script
    content: "slurm_scripts/five_bench/r36_eval.sh — array 0-7 over rho, r21_eval.sh structure but R26's explicit 9-metric --metrics list (structure_distance, psnr/lpips/mse/ssim_unedit_part, clip_similarity_source_image, clip_similarity_target_image, clip_similarity_target_image_edit_part, niqe_target_image); the list MUST be passed on the CLI (config.yaml's metrics: key is vestigial — job 961342). five-bench env, --partition=L40S,A100, --mem=64G, --time=12:00:00, --exclude=node01,node51,node52,node57 (r33_fiveacc.sh's set), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, all 6 annotation files, no --cases_json, stem r36_rho{R}_vp; expect 419 real dirs; grep metrics log for 'Error:' / 'out of memory' and fail loudly (evaluate.py swallows per-metric exceptions)."
    status: completed
  - id: fiveacc-script
    content: "slurm_scripts/five_bench/r36_fiveacc.sh — array 0-7 over rho, clone of r33_fiveacc.sh with --metrics five_acc only (Qwen2.5-VL alone, no CoTracker co-resident — the combination that OOMs off H100), full bench (no --cases_json, 419-dir check), same partition/mem/exclude as r36_eval.sh, --time=04:00:00 (r35_fiveacc.sh's limit; R33 measured 22 clips in <2 min), stem r36_fiveacc_rho{R}_vp."
    status: completed
  - id: clipd-script
    content: "slurm_scripts/five_bench/r36_clipd.sh — clone of r35_clipd.sh (single job, NOT an array: r33_clip_directional.py caches source embeddings per clip and reuses them across methods). Uses R35's existing --fullbench --methods NAME=DIR mode unchanged, with METHODS = r36_rho{R}_vp=$OUT_ROOT/r36_rho{R}_vp for the 8 rho; preflight 100/100/100/100/9/10 real dirs per arm; -o evaluation/csv/r36_clip_directional.csv; post-run guard asserts 419 rows per method with the per-edit-type breakdown and zero MISSING lines. Same SBATCH as r35_clipd.sh (L40S,A100, exclude node01/51/52/57, 32G, --time=06:00:00 — R33 job 1002662 scored 616 method-clips in ~4.5 min, so 8 x 419 = 3352 is ~25 min)."
    status: completed
  - id: summarize-script
    content: "evaluation/r36_summarize.py — per rho, a plain mean over all 419 clips (each clip weight 1) of the per-clip rows in edit{T}_FiVE_r36_rho{R}_vp_frame_stride8.csv, edit{T}_FiVE_r36_fiveacc_rho{R}_vp_frame_stride8.csv and the method == r36_rho{R}_vp rows of r36_clip_directional.csv — NOT the {stem}_avg.csv files, which are a mean of the six per-edit-type means; → evaluation/csv/r36_rho_sweep.csv (one row per rho: 9 harness metrics + five_acc (yn/mc) + clip_d_prompt + clip_d_word + n_pairs, assert n_pairs == 419), plus a 22-clip-subset row per rho (same plain mean, restricted to the cases.json clips)."
    status: completed
  - id: pareto-script
    content: "evaluation/r36_pareto.py — three PDFs, x = semantic alignment (clip_similarity_target_image / clip_d_prompt / five_acc yn), y = preservation (lpips_unedit_part). Series: full-bench VP rho curve (8 points, labelled by rho); same 8 rho restricted to the 22 cases.json clips (dashed); R26's 22-clip diagonal (markers only — must sit on the dashed curve given the parity gate). Reuse load helpers from r26_tradeoff_figure.py / r33_report_figures.py rather than re-deriving."
    status: completed
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r36_smoke.sh
    status: completed
    completed_at: 2026-09-24
    job_id: "1007548"
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "logs/r36_smoke_{job_id}.out shows G1 IDENTICAL x2, G2 IDENTICAL x2, G3 DIFFERS x2, zero GATE*-FAIL, zero Traceback"
    status: completed
    completed_at: 2026-09-24
  - id: infer
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r36_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-09-24
    job_id: "1007603"
    rerun_job_id: "1008877"
    rerun_note: "2026-09-25: 1007603 finished rho 2/3/4 (419/419 each); rho 6-50 were killed by the home disk-quota hit at 07:04. Resubmitted --array=3-7 as 1008877 with the new skip (an edit type with EXP clip dirs + an EXP-row all-ok manifest is not re-rendered): skips rho6 e1-3, rho8 e1-2, rho10 e1-2 (700 clips, all 47,544 PNGs decode, frame counts match manifests). wait-infer must check 1007603 tasks 0-2 AND 1008877 tasks 3-7."
    note: "2026-09-24: the original 48-task (rho x edit type) layout hit QOSMaxSubmitJobPerUserLimit (per-user cap 27-32 jobs); its partial submission, job 1007589 (tasks 0-23), was cancelled while all tasks were still PENDING -- nothing rendered. Replaced by the 8-task one-per-rho layout."
  - id: wait-infer
    type: manual
    wait_for: infer
    check_hint: "sacct -j {job_id} (or logs if slurmdbd is down): 8/8 tasks COMPLETED; each logs/r36_infer_{job_id}_{i}.out ends with 'total frame dirs: 419' and 'failures: 0'; every r36_rho{R}_vp holds 419 real frame dirs"
    sets_status: finished
    status: completed
    completed_at: 2026-09-28
  - id: check-parity
    type: local
    command: python evaluation/r36_check_parity.py --out evaluation/csv/r36_parity.csv
    status: completed
    completed_at: 2026-09-28
  - id: evaluate
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r36_eval.sh
    status: completed
    completed_at: 2026-09-28
    job_id: "1012691"
  - id: five-acc
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r36_fiveacc.sh
    status: completed
    completed_at: 2026-09-28
    job_id: "1012692"
  - id: clip-d
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r36_clipd.sh
    status: completed
    completed_at: 2026-09-28
    job_id: "1012693"
  - id: wait-eval
    type: manual
    wait_for: [evaluate, five-acc, clip-d]
    check_hint: "per rho, 6/6 per-edit-type CSVs for r36_rho{R}_vp and r36_fiveacc_rho{R}_vp with 100/100/100/100/9/10 rows; evaluation/csv/r36_clip_directional.csv has 8 x 419 rows; zero 'Error:' / OOM lines in logs/r36_eval_*.metrics.log and logs/r36_fiveacc_*.metrics.log; r36_clipd log ends with OK"
    status: completed
    completed_at: 2026-09-29
    note: "verified 2026-09-29 from logs + CSVs (sacct down): 1012691 / 1012692 all 16 tasks exit=0, 0 Error:/OOM/Traceback in .metrics.log and .err; 1012693 ends OK, missing_lines=0; per rho 6/6 harness + 6/6 five_acc CSVs at 100/100/100/100/9/10, 0 NaN, 0 duplicate file_id; r36_clip_directional.csv 3352 rows = 8 x 419, 0 NaN, 0 duplicate (method, case_id); 0 leftover _resize dirs under r36_rho_sweep"
  - id: summarize
    type: local
    command: python evaluation/r36_summarize.py --out evaluation/csv/r36_rho_sweep.csv
    status: completed
    completed_at: 2026-09-29
  - id: figures
    type: local
    command: python evaluation/r36_pareto.py --sweep evaluation/csv/r36_rho_sweep.csv --outdir evaluation/figures
    status: pending
  - id: verdict
    type: manual
    wait_for: figures
    check_hint: "Read the three Pareto PDFs; record in daily.md where the full-bench VP rho curve sits vs rho=2, and whether the 22-clip subset curve matches R26's diagonal"
    sets_status: analyzed
    status: pending
isProject: true
---

# R36: Sweep Rho in Visual Prompting

## Context

R7 rendered StreamEdit + visual prompting (paper §4.5, Qwen-edited first-frame anchor) on the full FiVE-Bench at only the paper value `rho = --blend_power = 2`. R26's uniform baseline swept `rho ∈ {2,3,4,6,8,10,20,50}` under the same VP setup, but only on the 22 `cases.json` clips. R36 runs that same rho grid on all 419 pairs, so that the preservation/edit trade-off curve for VP is measured on the full benchmark and not just a 22-clip subset.

**Done when:** `evaluation/csv/r36_rho_sweep.csv` has one row per rho with the 9 FiVE harness metrics, FiVE-Acc and CLIP-D, and three Pareto figures (x = CLIP-T, CLIP-D or FiVE-Acc; y = background LPIPS) show the full-bench VP rho curve against R26's 22-clip diagonal.

## Execution steps

| # | Step id | Type | What | Sets status |
|---|---------|------|------|-------------|
| — | *(prep)* | — | build every item in `todos` (see **Code to touch**) | `implemented` |
| 1 | `smoke` | sbatch | 2 clips × rho {2,3}; parity against R26 and R7 | — |
| 2 | `wait-smoke` | manual | all 3 gates pass | — |
| 3 | `infer` | sbatch | 8-task render array, one per rho, all 6 edit types each | `running` |
| 4 | `wait-infer` | manual | 8/8 done, 419 dirs per arm | `finished` |
| 5 | `check-parity` | local | 22 clips × 8 rho sha256 vs R26 diagonal; index-0 vs R7 | — |
| 6 | `evaluate` | sbatch | 9-metric FiVE eval, array over 8 rho (L40S/A100) | — |
| 7 | `five-acc` | sbatch | FiVE-Acc only, array over 8 rho (L40S/A100) | — |
| 8 | `clip-d` | sbatch | directional CLIP, single job over 8 rho | — |
| 9 | `wait-eval` | manual | all eval CSVs complete, no metric crashes | — |
| 10 | `summarize` | local | `r36_rho_sweep.csv` | — |
| 11 | `figures` | local | 3 Pareto PDFs | — |
| 12 | `verdict` | manual | read figures, log outcome | `analyzed` |

```
/run-step R36 smoke
/run-step R36 wait-smoke
/run-step R36 infer
/run-step R36 wait-infer
/run-step R36 check-parity
/run-step R36 evaluate
/run-step R36 five-acc
/run-step R36 clip-d
/run-step R36 wait-eval
/run-step R36 summarize
/run-step R36 figures
/run-step R36 verdict
```

## Decisions

| Topic | Choice |
|-------|--------|
| Clip set | Full FiVE-Bench, 419 pairs (100/100/100/100/9/10). The 424 entries seen in R7's output count the `_manifest.csv` in each edit dir; there are 419 real clips. |
| Rho grid | `{2,3,4,6,8,10,20,50}`, identical to R26's uniform diagonal, so the curves line up point for point |
| rho = 2 | **Re-rendered, not reused** (user decision 2026-09-24). R7's renders and `r21_ref_vp` scores are not used for any R36 number or figure. |
| Anchoring | `--vp_mode vp`, anchors `~/Data/dataggen/outputs/five_bench/anchors` (R7's Qwen-Image-Edit-2511 set) |
| Sampler | step 15, fg_boost 4, flow_shift 1.0, seed 0, chunk 21, overlap 1, sink 0. Byte-identical to R7 and R26, which the parity gates require. |
| Output root | `~/Data/dataggen/outputs/five_bench/r36_rho_sweep/r36_rho{R}_vp/edit{T}/` (`/projects/dataggen` was moved to `~/Data/dataggen` on 2026-09-21) |
| Render array | 8 tasks, one per rho (`rho = RHOS[TID]`), each looping edit types 1–6 (419 pairs). L40S, 64G, 20h (partition max 24h). ~10.5h per task at R35 stage 1's measured ~1.5 min/clip incl. model loads; total ≈ 85 L40S GPU-h. The 48-task (rho × edit type) layout was dropped: it exceeds the per-user submit cap (27–32 jobs). |
| Metrics | **Motion fidelity dropped** (`motion_fidelity_score` and `motion_fidelity_score_edit_part`; CoTracker co-resident with Qwen2.5-VL OOMs off H100, job 961342, and no H100 access). Remaining metrics run in two L40S/A100 passes: (1) R26's explicit 9-metric list; (2) `five_acc` alone, as in R33 job 1004302, which ran 20/20 arms without H100. Plus CLIP-D (`clip_d_prompt`, `clip_d_word`). `--metrics` is always passed on the CLI. |
| Pareto axes | x = semantic alignment (`clip_similarity_target_image`, `clip_d_prompt`, `five_acc` yn — one per figure); y = preservation (`lpips_unedit_part`). Same orientation as R35 |
| Averaging | Plain mean over all 419 clips from the per-clip CSVs; never `{stem}_avg.csv` (mean of per-edit-type means) |
| Gate: R26 parity | For each rho, the 22 `cases.json` clips must be sha256-identical to `r26_spatial_tau/taubg{R}_taufg{R}_vp`, which used the same scalar `--blend_power` path, the same per-pair seed and the same sampler. Any mismatch is a pipeline regression since 2026-09-01 and blocks eval. |
| Gate: R7 parity | At rho=2, the index-0 pair of each edit type must be identical to R7's stored render (unaffected by the reseed fix). |
| CLIP-D | Single job cloned from `r35_clipd.sh`, using R35's `--fullbench --methods` mode of `r33_clip_directional.py` unchanged; no edit to that script |
| CSV stems | `r36_rho{R}_vp` (9-metric harness), `r36_fiveacc_rho{R}_vp`, `r36_clip_directional` (all 8 rho, keyed by `method`), `r36_rho_sweep`, `r36_parity` |
| Out of scope | Motion fidelity (can be added later by one H100 pass over the stored renders; nothing needs re-rendering); no-VP and persistent-VP rho sweeps; spatial/routed arms |

## Step commands

### smoke
```bash
sbatch slurm_scripts/five_bench/r36_smoke.sh
```

### wait-smoke
```bash
grep -E 'GATE|IDENTICAL|DIFFERS|FAIL' logs/r36_smoke_<job_id>.out
grep -c Traceback logs/r36_smoke_<job_id>.err   # expect 0
```

### infer
```bash
sbatch slurm_scripts/five_bench/r36_infer.sh
```

### wait-infer
```bash
sacct -j <job_id> --format=JobID,State,Elapsed | grep -v COMPLETED   # expect only header
for R in 2 3 4 6 8 10 20 50; do
  echo "rho=$R: $(ls -d ~/Data/dataggen/outputs/five_bench/r36_rho_sweep/r36_rho${R}_vp/edit*/*/ | grep -vc '_resize/$')"   # expect 419
done
```

### check-parity
```bash
python evaluation/r36_check_parity.py --out evaluation/csv/r36_parity.csv
```

### evaluate
```bash
sbatch slurm_scripts/five_bench/r36_eval.sh
```

### five-acc
```bash
sbatch slurm_scripts/five_bench/r36_fiveacc.sh
```

### clip-d
```bash
sbatch slurm_scripts/five_bench/r36_clipd.sh
```

### wait-eval
```bash
ls evaluation/csv/edit?_FiVE_r36_rho*_vp_frame_stride8.csv | wc -l          # expect 48
ls evaluation/csv/edit?_FiVE_r36_fiveacc_rho*_vp_frame_stride8.csv | wc -l  # expect 48
wc -l evaluation/csv/r36_clip_directional.csv                               # expect 8*419 + 1 = 3353
grep -lE 'Error:|out of memory' logs/r36_eval_*.metrics.log logs/r36_fiveacc_*.metrics.log   # expect nothing
tail -1 logs/r36_clipd_<job_id>.out                                         # expect "[r36_clipd] OK -> ..."
```

### summarize
```bash
python evaluation/r36_summarize.py --out evaluation/csv/r36_rho_sweep.csv
```

### figures
```bash
python evaluation/r36_pareto.py --sweep evaluation/csv/r36_rho_sweep.csv --outdir evaluation/figures
# -> r36_cliptgt_vs_lpips.pdf, r36_clipd_vs_lpips.pdf, r36_fiveacc_vs_lpips.pdf
```

### verdict
Open the three PDFs and `r36_rho_sweep.csv`, then record the outcome in `daily.md`.

## Pipeline

```mermaid
flowchart LR
  A[FiVE-Bench 419 pairs] --> I
  B[Qwen anchors] --> I
  I[r36_infer.sh<br/>8 tasks, one per rho] --> R[r36_rho_sweep/r36_rho R _vp]
  R --> P[r36_check_parity.py]
  R26[R26 diagonal renders] --> P
  R7[R7 renders] --> P
  P --> PC[r36_parity.csv]
  R --> E[r36_eval.sh<br/>9 metrics]
  R --> FA[r36_fiveacc.sh<br/>FiVE-Acc]
  R --> D[r36_clipd.sh<br/>CLIP-D]
  E --> S[r36_summarize.py]
  FA --> S
  D --> S
  S --> CSV[r36_rho_sweep.csv]
  CSV --> F[r36_pareto.py]
  R26CSV[R26 / R33 CSVs] --> F
  F --> PDF[3 Pareto PDFs]
```

## Code to touch

| File | Change |
|------|--------|
| `evaluation/r36_smoke_cases.json` | **new**: 2 entries copied verbatim from `cases.json` (`0001_bus` edit1, `0007_guitar-violin` edit5) |
| `slurm_scripts/five_bench/r36_smoke.sh` | **new**: single task, L40S, 64G, 2h; 2 renders (rho 2, 3) into `r36_smoke/`, then `r36_check_parity.py --smoke` |
| `slurm_scripts/five_bench/r36_infer.sh` | **new**: 8-task array (one per rho), looping edit types 1–6 per task; pattern of `r26_infer.sh`'s degenerate branch minus `--cases_json` |
| `evaluation/r36_check_parity.py` | **new**: sha256 frame-by-frame comparison, `--smoke` and full modes |
| `slurm_scripts/five_bench/r36_eval.sh` | **new**: `r21_eval.sh` structure with `r26_eval.sh`'s explicit 9-metric `--metrics` list, L40S/A100, array 0-7 over rho, count check expects 419 |
| `slurm_scripts/five_bench/r36_fiveacc.sh` | **new**: clone of `r33_fiveacc.sh`, `--metrics five_acc`, full bench, array 0-7 over rho |
| `slurm_scripts/five_bench/r36_clipd.sh` | **new**: clone of `r35_clipd.sh`, `COMBOS` → the 8 rho arms, `OUT_ROOT` → `r36_rho_sweep`, arm dirs without R35's `/step14`, `--time=12:00:00` |
| `evaluation/r36_summarize.py` | **new**: per rho, plain mean over the 419 per-clip rows of the 9-metric, FiVE-Acc and CLIP-D CSVs; plus 22-clip subset rows |
| `evaluation/r36_pareto.py` | **new**: 3 Pareto PDFs |

Render loop in `r36_infer.sh`:

```bash
RHOS=(2 3 4 6 8 10 20 50); EXPECTED=(100 100 100 100 9 10)
R=${RHOS[$TID]}; METHOD="r36_rho${R}_vp"
for T in 1 2 3 4 5 6; do
  python evaluation/run_fivebench.py --edit_type "$T" --method "$METHOD" \
    --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --blend_power "$R" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --flow_shift 1.0 --seed 0 || FAILED=$((FAILED+1))
  # count real dirs in $OUT_ROOT/$METHOD/edit$T (excl. *_resize) == EXPECTED[T-1],
  # and every _manifest.csv row status == ok; else FAILED++ (loop continues)
done
exit $(( FAILED > 0 ))
```

Parity check core:

```python
def frames_sha(d: Path) -> list[str]:
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.png"))]
# full mode: for rho in RHOS, for case in cases.json:
#   identical = frames_sha(r36/f"r36_rho{rho}_vp"/f"edit{T}"/vid) == frames_sha(r26/f"taubg{rho}_taufg{rho}_vp"/f"edit{T}"/vid)
# plus rho=2 index-0 pair per edit type vs r7_visual_prompting; exit 1 if any False
```
