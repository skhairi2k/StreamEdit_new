---
name: R22 — Metric Convergence Over Steps/Frames
overview: For each of the 22 clips in evaluation/cases.json, build an (nb_frames x 15) matrix M(i,j) = metric(V_j[:i]) under three arms (paper_novp, paper_vp, cos_third_pvp), by dumping the target-branch x0 prediction at every denoising step of every block during one unperturbed rollout, decoding the 15 resulting videos, scoring them per-frame at stride 1, and taking cumulative prefix means; the bottom-right cell is validated three ways, the substantive one being bit-parity of the step-14 render against the arm's stored full-bench reference. Extended 2026-08-18 with two further views over the same artefacts — absolute-value (not deviation) heatmaps for the CLIP and LPIPS metrics, and a per-(video, arm) image mosaic whose (i,j) cell is the i-th pixel frame decoded at denoising step j, sampled at i%10==0 and j in {0,2,5,8,11,14}.
task_id: R22
todos:
  - id: patch-step-dump
    content: "Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py — add `step_dump: Optional[list] = None` to `inference()` and `rollout_inference()`. In `inference`: allocate `step_out = output.unsqueeze(0).repeat(S,1,1,1,1,1).clone()` right after `denoising_step_list` is read (line ~442, AFTER Step 2 has written the initial-latent prefix into `output`), write `step_out[index][:, cs:cs+n] = denoised_pred` immediately after the S.O.G. update at line 559, and `assert torch.equal(step_out[-1], output)` after the block loop before appending to the caller's list. In `rollout_inference`: forward the arg on the `rollout_chunk_size < 0` early return, collect per window, slice windows >= 2 as `[:, :, rollout_overlap:]` (frame axis is dim 2 now), cat along dim 2, append. Default None => zero added work, bit-for-bit unchanged. BUILT 2026-08-12: the `output[:, 1:]` / `output[:, 3:]` trims at Step 4 (independent/triple first frame) also had to be mirrored on the step buffer as `[:, :, 1:]` / `[:, :, 3:]` BEFORE the assert -- they were not in the original spec, and without them the assert would fire on every `vp` clip. Assert added at BOTH levels (per window in `inference`, again after the rollout stitch) so a bad concat axis cannot pass."
    status: completed
  - id: write-r22-driver
    content: "evaluation/r22_dump_steps.py — new driver modelled on run_fivebench.py (same per-pair `set_seed`, same transform, same VAE encode ordering). Calls `rollout_inference(..., return_latents=True, wo_video_decode=True, step_dump=steps)`, asserts `torch.equal(steps[0][-1], latents)`, then decodes and saves each of the 15 step latents ONE AT A TIME (`decode_to_pixel(use_cache=False)` + `vae.model.clear_cache()` between decodes, `*0.5+0.5` clamp) to {out_root}/{arm}/step{jj}/edit{T}/{video}/. Never hold 15 decoded videos in memory (15 x [81,3,480,832] fp32 = 7.3 GB). Writes _manifest.csv per (arm, step). BUILT 2026-08-12: manifest is per edit type (`_manifest_edit{T}.csv`) at the arm root, not per (arm, step) -- one row per pair covers all 15 steps since they succeed or fail together. Driver exits 1 if any pair errored, so the SLURM `||` guard actually fires."
    status: completed
  - id: write-r22-smoke
    content: "slurm_scripts/five_bench/r22_smoke.sh — single clip (0001_bus, edit1), arm paper_novp. Gate 1: the in-code latent assert fires clean. Gate 2: step14 PNGs BIT-IDENTICAL (cmp -s per frame) to /projects/dataggen/outputs/five_bench/baseline/edit1/0001_bus/. Gate 2 is the real check — it proves the 15 sequential VAE decodes do not carry temporal-conv state into each other and that the dump path did not perturb sampling. Exit non-zero on any mismatch. BUILT 2026-08-12: writes to a dedicated `five_bench/r22_smoke` root rather than `r22_step_dump`, so a partial single-clip tree cannot skew wait-dump's 330-dir-per-arm count. Reference verified present: 69 frames, mtime 2026-07-22 21:59 = post-seeding-fix generation, so the byte comparison is valid."
    status: completed
  - id: write-r22-dump
    content: "slurm_scripts/five_bench/r22_dump.sh — array 0-2, one arm per task, each looping EDIT_TYPES=(1 2 5 6). 0=paper_novp (no --first_frame_edit_dir, no --blend_sched), 1=paper_vp (--vp_mode vp --first_frame_edit_dir $ANCHOR_ROOT), 2=cos_third_pvp (--vp_mode pvp --first_frame_edit_dir $ANCHOR_ROOT --blend_sched cos_third). L40S, --mem=64G, --time=06:00:00, same conda dance as r20_infer.sh (deactivate then activate streamgve, no `set -u`). ~5 min/clip (1 inference + 15 decodes) => ~2h/arm. BUILT 2026-08-12: smoke measured ~4 min/clip => ~1.5h/arm, so the 6h budget is headroom for the multi-window clips (one inference per window). The two `paper` arms pass NO --blend_sched rather than --blend_sched paper — identical code path, but flag-off is exactly how the stored R1/R7 references were rendered. Verified before marking done: all 22 anchors present under dataggen/anchors/edit{T}/, per-edit-type split 1/16/3/2 as the plan states, arm dispatch dry-run correct for all 3 tasks, 11/11 #SBATCH lines confirmed. Script's tail check counts leaf frame dirs (expect 330/arm) and GATE1 PASS lines (expect 22)."
    status: completed
  - id: write-r22-eval
    content: "slurm_scripts/five_bench/r22_eval.sh — array 0-2, one arm per task, inner loop j=0..14. evaluate.py with --per_frame --frame_stride 1, the 8 prefix-decomposable metrics ONLY, --tgt_layout edit_video, --cases_json, --tgt_key r22_{arm}_s{jj}, --result_path evaluation/csv/r22_raw/{arm}_s{jj}.csv. `five-bench` env. rm -rf the {video}_resize dirs after each j (stride 1 doubles the PNG count, ~34 GB of derived data). --time=20:00:00: stride 1 is ~7.5x the image-metric work of R20's stride 8, so ~7h/arm is the estimate and 6h would be too tight. BUILT 2026-08-12: result stem is `r22_{arm}_s{jj}` (task-id prefix per the naming convention), so the per-frame file r22_build_matrix.py globs is `evaluation/csv/r22_raw/r22_{arm}_s{jj}_frame_stride1_per_frame.csv` — still matched by the plan's `*_frame_stride1_per_frame.csv` glob. Runs on **L40S, not H100**: R21 needed the bigger card only because Qwen2.5-VL + CoTracker were co-resident, and dropping five_acc/motion_fidelity removes that pressure entirely (`expandable_segments` kept anyway as free insurance). Gate is three-way — evaluate.py exit code, per-frame CSV non-empty, and 0 `Error:` lines — because R21's evaluate.py:487 patch stops a raised metric from shifting columns but does NOT stop it from omitting that metric's rows, which would punch holes in the matrix. Verified before marking done: all 8 metric names recognised by evaluate.py, dump dirs are valid `--tgt_layout edit_video` roots ({root}/edit{T}/{video}/*.png), 11/11 #SBATCH lines, dispatch dry-run correct for all 3 tasks."
    status: completed
  - id: write-r22-matrix
    content: "evaluation/r22_build_matrix.py — glob evaluation/csv/r22_raw/*_s??_frame_stride1_per_frame.csv, pivot to (arm, video_name, edit_type, metric, j, frame_idx) -> value, cumulative-mean over frame_idx to get M(i,j) for i=1..nb_frames, emit long-format evaluation/csv/r22_matrix.csv. Also emit evaluation/csv/r22_bottom_right_check.csv asserting M(nb_frames,14) == plain mean of the step-14 per-frame column per (arm, clip, metric) to float tolerance. BUILT 2026-08-18: pairs are keyed on `(arm, editing_type_id, file_id, video_name)`, NOT video_name — `0011_lucia` appears under edit types 2 AND 5 in cases.json, so a name-only key would have merged two different pairs into one interleaved prefix sequence (silent, and it would still have passed the L3 check). arm/j are parsed from the `method` column (`r22_{arm}_s{jj}`) rather than the filename. Added a hard `validate()` gate — 15 steps per cell, frame_idx contiguous 0..n-1 per (cell,step), frame count constant across steps — because a single missing row shifts every later prefix mean silently; `--tol` (default 1e-9) exposes the L3 tolerance and the script exits 1 on any failing cell. Prefix mean is nan-skipping (running sum / running count of present values) so a future nan degrades one cell instead of poisoning the whole row. Verified on the paper_novp arm before marking done: 15 files / 177840 rows / 22 pairs x 8 metrics, validate clean, **L3 PASS on all 176 cells with max |diff| = 0**, and 200 randomly sampled cells (13476 prefix values) match a brute-force `np.cumsum/arange` recomputation to 7.1e-15."
    status: completed
  - id: write-r22-figures
    content: "evaluation/r22_figures.py — per (arm, metric): a 22-panel grid of per-clip heatmaps at native row count, plus a clip-averaged mean surface built on NORMALIZED prefix fraction i/nb_frames resampled to 50 bins (clips run 21-117 frames, so a raw row-wise mean is not defined). Diverging colormap centred on the column-14 value so the figure reads as convergence-to-final. -> evaluation/figures/r22_{metric}_{arm}.pdf + r22_mean_surfaces.pdf. BUILT 2026-08-18: **resolved the one ambiguity in the spec** — 'centred on the column-14 value' can mean the scalar M(nb_frames,14) or the column M(i,14); the **column** is used, i.e. panels plot D(i,j) = M(i,j) − M(i,14). Two reasons: the scalar leaves the last column non-white wherever M(i,14) varies with i (it does everywhere), so the figure would not actually read as convergence-to-final; and the 22 clips sit at very different absolute levels (per-clip PSNR ~18-32 dB), so absolute values force either a useless shared colourbar or 22 incomparable per-panel ones, while deviations share one scale. Absolute levels remain in r22_matrix.csv. Colour limit is a symmetric 98th-percentile of |D| (outliers must not flatten the panel); per-clip grids share one scale across the 22 panels; the mean-surface page shares one scale per metric row across the 3 arms so arms are comparable. Clips are grouped on the same (editing_type_id, file_id, video_name) key as r22_build_matrix.py, so 0011_lucia correctly appears as two panels (e2, e5). Added `--only_metric` (restricts the per-clip grids only) and a stdout summary giving the earliest step from which the full-clip prefix stays within 1% of its final value. Verified before marking done: py_compile, a single-metric rehearsal to /tmp, and both figure families rendered to PNG and visually inspected — 22 panels at native row counts 21-93f, titles carry edit type, and clip_similarity_source_image comes out uniformly white at 1e-12 (the expected null control, since it is computed on the source video)."
    status: completed
  - id: write-r22-absolute
    content: "evaluation/r22_figures.py — ADD an absolute-value mode alongside the existing deviation mode (requested 2026-08-18): same heatmap geometry, but plotting M(i,j) itself rather than D(i,j)=M(i,j)-M(i,14), on a SEQUENTIAL colormap (viridis) instead of the diverging one. Restricted to the CLIP and LPIPS families: `clip_similarity_target_image`, `clip_similarity_target_image_edit_part`, `lpips_unedit_part` — exposed as `--abs_metrics` so the set is changeable. `clip_similarity_source_image` is deliberately excluded: it is computed on the source video and is constant along j, so an absolute panel of it carries no information the deviation figure did not already show. Emits per-clip grids `evaluation/figures/r22_abs_{metric}_{arm}.pdf` (9 = 3 metrics x 3 arms) + a combined `evaluation/figures/r22_abs_mean_surfaces.pdf` (3 metrics x 3 arms on normalised prefix fraction). **Colour scale: shared across the 22 panels of a figure AND across the 3 arms of a metric row** — the whole point of an absolute view is cross-clip and cross-arm comparability, which a per-panel autoscale would destroy. Trade-off to accept explicitly: a shared absolute scale compresses within-clip structure (that is what the deviation figures are for); the two modes are complementary, not redundant. Reuses `clip_surface` / `resample_fraction`; do not duplicate them. BUILT 2026-08-18: implemented as `absolute`/`vlim` keyword args on the existing `per_clip_grid` and `mean_surfaces` (defaults preserve the old behaviour) plus a new `abs_vlim()` that computes robust 1st/99th-percentile limits per metric over ALL arms at once, so a metric's three arms share one scale by construction rather than by coincidence. `--absolute` returns before `print_summary`, which is a deviation-mode diagnostic. **Regression-verified**: re-rendering the deviation figure after the edit is byte-identical to the pre-edit PNG, so the already-published deviation family is provably untouched."
    status: completed
  - id: write-r22-mosaic
    content: "evaluation/r22_mosaic.py — NEW (requested 2026-08-18). Per (video, arm) image mosaic read straight off the step dump: rows = pixel frames with `i % --frame_mod == 0` (default 10, 0-based to match the filenames `{iiiii}.png`, so 0,10,20,...), cols = denoising steps `--steps` (default 0 2 5 8 11 14). Cell (i,j) is the i-th frame of that clip decoded at step j. Source `/projects/dataggen/outputs/five_bench/r22_step_dump/{arm}/step{jj}/edit{T}/{video}/{iiiii}.png` (verified on disk 2026-08-18: 36 GB intact, frames 832x480 RGB, 0-indexed). Clip list from `evaluation/cases.json`. Output `evaluation/figures/r22_mosaic/{arm}/{video}_e{T}.pdf` — **the `_e{T}` suffix is load-bearing: `0011_lucia` appears under edit types 2 AND 5, so a `{video}.pdf` name would silently overwrite one with the other** (same keying trap as r22_build_matrix.py). 66 files = 22 clips x 3 arms; grids run 3x6 (0072_dog-agility, 21f) to 10x6 (0034_cows, 93f). **Tiles are downsampled to `--tile_width` (default 208 px = 832/4) before embedding** — 60 full-res tiles per page would make each PDF ~30 MB and the set ~2 GB. Row labels = frame index, column labels = step index, suptitle = video / edit type / arm. Skip-and-report any missing frame rather than crashing the whole run. BUILT 2026-08-18: row count is derived from the frames actually on disk (`count_frames` globs the first requested step dir), not from cases.json, so a clip's grid always matches its real length; a clip with no frames is skipped with a message instead of producing an empty page. Added `--arms` (defaults to every arm dir under `--dump_root`) and `--only_video` for cheap spot-checks. Verified before marking done: py_compile; a 3-page rehearsal (`0072_dog-agility` 3x6 of 21f, `0011_lucia` 7x6 of 69f under BOTH e2 and e5) producing two distinct lucia files of different byte size — the `_e{T}` guard working on the real trap case; and a PNG render inspected visually, which already reads as a result (on `0072_dog-agility`/`cos_third_pvp` the source dog is still present at j=0 and the white-horse edit has resolved by j=2-5). ~950 KB per 7x6 page => ~60 MB for the full 66-page set at the default tile width."
    status: completed
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r22_smoke.sh
    status: completed
    completed_at: 2026-08-12
    job_id: "937952"
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; grep -E '^\\[r22_smoke\\] (GATE1|GATE2)' logs/r22_smoke_*.out  # both must read PASS; a FAIL on GATE2 means the step-14 render is not the arm's real output — stop, do not launch dump"
    status: completed
    completed_at: 2026-08-12
  - id: dump
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=0-2 slurm_scripts/five_bench/r22_dump.sh
    sets_status: running
    status: completed
    completed_at: 2026-08-12
    job_id: "938004"
  - id: wait-dump
    type: manual
    wait_for: dump
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; OUT=/projects/dataggen/outputs/five_bench/r22_step_dump; for a in paper_novp paper_vp cos_third_pvp; do echo $a $(ls -d $OUT/$a/step*/*/*/ 2>/dev/null | wc -l); done  # each must be 330 = 22 clips x 15 steps; grep -c 'FAILED' logs/r22_dump_*.out  # expect 0"
    sets_status: finished
    status: completed
    completed_at: 2026-08-12
  - id: evaluate
    type: sbatch
    wait_for: wait-dump
    command: sbatch --array=0-2 slurm_scripts/five_bench/r22_eval.sh
    status: completed
    completed_at: 2026-08-12
    job_id: "941675"
  - id: wait-eval
    type: manual
    wait_for: evaluate
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; ls evaluation/csv/r22_raw/*_frame_stride1_per_frame.csv | wc -l  # expect 45 = 3 arms x 15 steps; grep -cE '^Error:' logs/r22_eval_*.out  # MUST be 0 — upstream's `except: continue` drops a metric silently. NOTE: anchor the grep at ^ — an unanchored 'Error:' matches the script's own gate line (\"'Error:' lines: 0\") and reports 1 per task on a clean run."
    status: completed
    completed_at: 2026-08-18
  - id: build-matrix
    type: local
    wait_for: wait-eval
    command: |
      python evaluation/r22_build_matrix.py \
        --per_frame_glob 'evaluation/csv/r22_raw/*_frame_stride1_per_frame.csv' \
        -o evaluation/csv/r22_matrix.csv \
        --check_out evaluation/csv/r22_bottom_right_check.csv
    output_paths:
      - evaluation/csv/r22_matrix.csv
      - evaluation/csv/r22_bottom_right_check.csv
    status: completed
    completed_at: 2026-08-18
  - id: figures
    type: local
    wait_for: build-matrix
    command: |
      python evaluation/r22_figures.py \
        --matrix evaluation/csv/r22_matrix.csv \
        --outdir evaluation/figures
    output_paths:
      - evaluation/figures/r22_mean_surfaces.pdf
    sets_status: analyzed
    status: completed
    completed_at: 2026-08-18
  - id: figures-absolute
    type: local
    wait_for: build-matrix
    command: |
      python evaluation/r22_figures.py \
        --matrix evaluation/csv/r22_matrix.csv \
        --outdir evaluation/figures \
        --absolute
    output_paths:
      - evaluation/figures/r22_abs_mean_surfaces.pdf
    status: completed
    completed_at: 2026-08-18
  - id: mosaic
    type: local
    wait_for: wait-dump
    command: |
      python evaluation/r22_mosaic.py \
        --dump_root /projects/dataggen/outputs/five_bench/r22_step_dump \
        --cases_json evaluation/cases.json \
        --frame_mod 10 \
        --steps 0 2 5 8 11 14 \
        --outdir evaluation/figures/r22_mosaic
    output_paths:
      - evaluation/figures/r22_mosaic
    status: completed
    completed_at: 2026-08-18
isProject: true
---

# R22: Metric Convergence Over Steps/Frames

## Context

For each of the 22 clips in `evaluation/cases.json`, produce an `nb_frames x 15` matrix `M(i,j)` = a FiVE metric evaluated on the first `i` pixel frames of the video as it stood at denoising step `j`. The two axes ask different questions — how much of the clip you have seen (streaming latency) versus how far the sampler has run (compute) — and the surface says where quality actually arrives.

The pipeline's 15-step loop is **nested inside a per-block loop** ([edit_causal_inference.py:487](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L487)), and `output[:, cs:cs+n] = denoised_pred` fires once per block after that block's 15 steps finish ([line 574](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L574)). There is therefore no instant at which the whole video sits at step `j`. `V_j` is defined as **the assembly of each block's own step-`j` x0 prediction, collected during one unperturbed rollout** — blocks still commit their step-14 value to the KV cache, so generation is untouched and `V_14` is the real output by construction. `V_j` for `j<14` is a trajectory preview, not what a `j`-step sampler would emit: block `b`'s step-`j` latent was produced while attending to blocks `<b` that were already fully denoised and re-cached at clean context (Step 3.3).

**Done when:** `evaluation/csv/r22_matrix.csv` holds `M(i,j)` for 8 metrics x 15 steps x every pixel-frame prefix, for all 22 clips under all 3 arms; `evaluation/csv/r22_bottom_right_check.csv` shows the identity holding everywhere; the smoke's GATE2 recorded PASS (step-14 render bit-identical to the arm's stored reference); heatmap PDFs written.

**Extended 2026-08-18** (after the task first reached `analyzed`), also done when: the absolute-value CLIP/LPIPS figures exist (`r22_abs_{metric}_{arm}.pdf` + `r22_abs_mean_surfaces.pdf`), and the 66 per-(video, arm) frame mosaics exist under `evaluation/figures/r22_mosaic/{arm}/{video}_e{T}.pdf`. Both are read-only views over artefacts that already exist — the matrix CSV and the 36 GB step dump — so nothing regenerates and the numbers cannot move.

## Execution steps

| # | Step id | Type | What |
|---|---------|------|------|
| — | *(prep)* | — | `patch-step-dump`, `write-r22-driver`, `write-r22-smoke`, `write-r22-dump`, `write-r22-eval`, `write-r22-matrix`, `write-r22-figures` — see **Code to touch** |
| 1 | `smoke` | sbatch | 1 clip x paper_novp; latent assert + step-14 PNG bit-parity vs `five_bench/baseline` |
| 2 | `wait-smoke` | manual | Both gates PASS. **GATE2 FAIL is a stop condition** — do not launch `dump` |
| 3 | `dump` | sbatch | Array 0-2 = 3 arms, 22 clips each, 15 step-videos per clip → `running` |
| 4 | `wait-dump` | manual | 330 frame dirs per arm; 0 `FAILED` → `finished` |
| 5 | `evaluate` | sbatch | Array 0-2; 15 x `evaluate.py --per_frame --frame_stride 1`, 8 metrics |
| 6 | `wait-eval` | manual | 45 per-frame CSVs; **0 `Error:` lines** |
| 7 | `build-matrix` | local | Cumulative prefix means → `r22_matrix.csv` + identity check |
| 8 | `figures` | local | Per-clip heatmap grids + mean surfaces (deviation) → `analyzed` |
| 9 | `figures-absolute` | local | Same geometry, **actual values** of CLIP + LPIPS, sequential colormap |
| 10 | `mosaic` | local | Per (video, arm) **image** mosaic; rows = frames `i%10==0`, cols = steps {0,2,5,8,11,14} |

Steps 9–10 were added 2026-08-18, after the task first reached `analyzed`. Both are
additive read-only views over artefacts that already exist (`r22_matrix.csv` for 9, the
step dump for 10) — neither re-runs inference or evaluation, and neither carries
`sets_status`, so the task stays `analyzed` throughout.

```
/run-step R22 smoke
/run-step R22 wait-smoke
/run-step R22 dump
/run-step R22 wait-dump
/run-step R22 evaluate
/run-step R22 wait-eval
/run-step R22 build-matrix
/run-step R22 figures
/run-step R22 figures-absolute
/run-step R22 mosaic
```

## Decisions

| Question | Choice |
|---|---|
| Definition of `V_j` | Per-block step-`j` x0 prediction (`denoised_pred`, [line 559](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L559)), assembled with the same `[:, cs:cs+n]` write `output` uses, over one unperturbed rollout. Not a `j`-step sampler — a trajectory preview. |
| Why x0 and not the noisy latent | `denoised_pred` is the clean-image estimate; decoding `noisy_pred_input` through the VAE gives noise. At `index=14` it *is* what line 574 writes, which is what makes the bottom-right identity exact rather than approximate. |
| Column count | 15 (`j = 0..14`), `--step 15`. `j=0` is the first model update, not the source: `denoised_pred` is initialised to `src_input` at [line 452](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L452) but the dump writes *after* the S.O.G. update. |
| Row axis `i` | **Every pixel frame**, `i = 1..nb_frames` (21–117 rows, clip-dependent). Free under cumulative means. Long-format CSV, so ragged row counts are fine. |
| *(corrected 2026-08-18)* Row-axis **direction** in figures | **Matrix convention: `i` increases DOWNWARD**, so `i=1` is the top row and every panel's bottom-right cell is `M(nb_frames,14)` — the same cell this plan, the L3 check and the reports all call "the bottom-right entry". The first version drew `i` upward (matplotlib's `origin="lower"` default habit): tick labels were numerically correct, but the panel's bottom-right was `M(1,14)`, which reads as the opposite claim about how quality moves through a clip and directly contradicted the plan's own wording. Caught by the user on inspection. The mosaic already used frame 0 at top, so all three families now agree. Do not flip back without rewording every "bottom-right" reference. |
| Metrics (8) | `structure_distance`, `psnr_unedit_part`, `lpips_unedit_part`, `mse_unedit_part`, `ssim_unedit_part`, `clip_similarity_source_image`, `clip_similarity_target_image`, `clip_similarity_target_image_edit_part`. All are per-frame means, so `M(i,j)` = cumulative mean of the per-frame values ⇒ **15 evaluations per clip, not `nb_frames`x15**. |
| Metrics excluded | `motion_fidelity_score{,_edit_part}` and `five_acc` are whole-clip (CoTracker tracklet sets / VLM verdict) — not decomposable, and `--per_frame` already emits them with `frame_idx=-1`. `niqe_target_image` is decomposable in principle but **`calculate_NIQE` returns the literal string `"nan"` per frame** ([metrics_calculator.py:807](../evaluation/fivebench/metrics_calculator.py#L807)); real values go only to a txt that is deleted per video, so the long CSV cannot carry it. |
| Arms (3) | `paper_novp` (Eq.4, no anchor — R1 config), `paper_vp` (Eq.4 + §4.5 anchor — R7 config), `cos_third_pvp` (R21's strongest editor under the persistent bank). |
| Why these 3 | Each has a **stored full-bench reference render of the same post-seeding-fix generation**, so step-14 can be bit-compared: `five_bench/baseline` (R1), `five_bench/r7_visual_prompting` (R7), `r21_blend_full/cos_third_pvp` (R21). |
| Bottom-right identity — 3 levels | **L1 latent:** `assert torch.equal(step_out[-1], output)` inside `inference`, per window — exact and free. **L2 pixel (the substantive one):** step-14 PNGs bit-identical to the stored reference arm, checked in the smoke. **L3 metric:** `M(nb_frames,14)` equals the plain mean of the step-14 per-frame column — true by construction of the cumulative mean, kept as a regression assert. |
| Identity trap | `M(nb_frames,14)` will **not** equal `r21_*_avg.csv` / `r20_*_avg.csv`. Those ran at `--frame_stride 8` and their overall row is a mean-of-edit-type-means, not a per-clip mean. Do not write that comparison into the check. |
| Clip set | The 22 clips of `evaluation/cases.json`, spanning edit types 1/2/5/6 (1/16/3/2) — same subset R20 used, and comparable to full runs since the 2026-07-22 per-pair reseed. |
| Output root | `/projects/dataggen/outputs/five_bench/r22_step_dump/{arm}/step{jj}/edit{T}/{video}/` — the `step{jj}` level sits above `edit{T}` precisely so each step dir is a valid `--tgt_layout edit_video` root. |
| Sampler config | `--step 15 --fg_boost_factor 4 --blend_power 2 --seed 0 --flow_shift 1.0`, `rollout_chunk_size=21`, per-pair reseed on — the run_fivebench.py defaults R1/R7/R20/R21 used. Anchors from `/projects/dataggen/outputs/five_bench/anchors/edit{T}/`. |
| Eval GPU | **L40S**, `--mem=64G`. R21 needed H100 only because Qwen2.5-VL + CoTracker were co-resident; dropping `five_acc` and `motion_fidelity` removes that pressure entirely. |
| Disk | ~67k dump PNGs (~34 GB) + an equal volume of `{video}_resize` copies that `evaluate.py` writes at stride 1. The eval script `rm -rf`s the `_resize` dirs after each `j`. |
| Mean surface | Per-clip heatmaps at native row count; the clip-averaged surface uses **normalised prefix fraction `i/nb_frames` resampled to 50 bins**, since raw row-wise averaging is undefined across 21–117-frame clips. |
| Colour semantics — deviation figures | `D(i,j) = M(i,j) − M(i,14)`, diverging map centred at 0, so the last column is white **by construction**. The plan's original "centred on the column-14 value" admitted a scalar reading (`M(nb_frames,14)`); the **column** is used because the scalar leaves the last column non-white wherever `M(i,14)` varies with `i` (everywhere), and because the 22 clips sit at very different absolute levels (per-clip PSNR ~18–32 dB). |
| *(added 2026-08-18)* Absolute-value figures | The deviation view deliberately discards level, so a second family plots **`M(i,j)` itself** for the CLIP and LPIPS metrics on a **sequential** colormap. Metrics: `clip_similarity_target_image`, `clip_similarity_target_image_edit_part`, `lpips_unedit_part` (`--abs_metrics` to change). `clip_similarity_source_image` excluded — computed on the source video, constant along `j`. |
| Absolute colour scale | **Shared across the 22 panels of a figure and across the 3 arms of a metric row.** Cross-clip and cross-arm comparability is the only reason to look at absolute values; per-panel autoscale would destroy it. Accepted cost: within-clip structure is compressed — that is what the deviation family is for. The two modes are complementary, not redundant. |
| *(added 2026-08-18)* Frame mosaic | Per (video, arm): rows = pixel frames with `i % 10 == 0` (0-based, matching the dump filenames), cols = steps `{0,2,5,8,11,14}`, cell = that frame decoded at that step. Read straight off the step dump — **no re-inference, no re-decode**. 66 pages (22 clips × 3 arms), grids 3×6 (`0072_dog-agility`, 21f) to 10×6 (`0034_cows`, 93f). |
| Mosaic file naming | `r22_mosaic/{arm}/{video}_e{T}.pdf` — the `_e{T}` is **required**, not cosmetic: `0011_lucia` appears under edit types 2 and 5, so `{video}.pdf` would silently overwrite one with the other. Same keying trap that `r22_build_matrix.py` hit. |
| Mosaic tile size | Frames are 832×480; tiles are downsampled to `--tile_width` 208 px (÷4) before embedding. At full resolution a 60-tile page is ~30 MB and the 66-page set ~2 GB, which is not a figure set anyone can open. |
| Out of scope | Non-decomposable metrics, full FiVE-Bench (22 clips only), `zero`/`cos_full`/`cos_half` arms, head-gated arms, a published report artifact. |

## Step commands

### smoke

```bash
sbatch slurm_scripts/five_bench/r22_smoke.sh
```

### wait-smoke

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
grep -E '^\[r22_smoke\] (GATE1|GATE2)' logs/r22_smoke_*.out   # both must read PASS
```

### dump

```bash
sbatch --array=0-2 slurm_scripts/five_bench/r22_dump.sh
```

### wait-dump

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
OUT=/projects/dataggen/outputs/five_bench/r22_step_dump
for a in paper_novp paper_vp cos_third_pvp; do
  echo "$a $(ls -d $OUT/$a/step*/*/*/ 2>/dev/null | wc -l)"   # each 330 = 22 x 15
done
grep -c 'FAILED' logs/r22_dump_*.out                          # expect 0
```

### evaluate

```bash
sbatch --array=0-2 slurm_scripts/five_bench/r22_eval.sh
```

### wait-eval

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
ls evaluation/csv/r22_raw/*_frame_stride1_per_frame.csv | wc -l   # expect 45
grep -c 'Error:' logs/r22_eval_*.out logs/r22_eval_*.err          # MUST be 0
```

### build-matrix

```bash
python evaluation/r22_build_matrix.py \
  --per_frame_glob 'evaluation/csv/r22_raw/*_frame_stride1_per_frame.csv' \
  -o evaluation/csv/r22_matrix.csv \
  --check_out evaluation/csv/r22_bottom_right_check.csv
```

### figures

```bash
python evaluation/r22_figures.py \
  --matrix evaluation/csv/r22_matrix.csv \
  --outdir evaluation/figures
```

### figures-absolute

```bash
python evaluation/r22_figures.py \
  --matrix evaluation/csv/r22_matrix.csv \
  --outdir evaluation/figures \
  --absolute
```

### mosaic

```bash
python evaluation/r22_mosaic.py \
  --dump_root /projects/dataggen/outputs/five_bench/r22_step_dump \
  --cases_json evaluation/cases.json \
  --frame_mod 10 \
  --steps 0 2 5 8 11 14 \
  --outdir evaluation/figures/r22_mosaic
```

## Pipeline

```mermaid
flowchart TD
    A[FiVE-Bench videos<br/>cases.json 22 clips<br/>edit 1,2,5,6] --> D
    B[dataggen/anchors/editT/*.png] --> D
    D[r22_dump_steps.py<br/>step_dump hook in inference<br/>3 arms x 15 step latents] --> E[r22_step_dump/<br/>arm/stepJJ/editT/video/]
    D -.L1 latent assert.-> Z[step_out-1 == output]
    R1[five_bench/baseline] --> S[r22_smoke.sh<br/>L2 PNG bit-parity]
    R7[five_bench/r7_visual_prompting] --> S
    R21[r21_blend_full/cos_third_pvp] --> S
    E --> S
    E --> F[fivebench/evaluate.py<br/>--per_frame --frame_stride 1<br/>8 decomposable metrics]
    F --> G[csv/r22_raw/<br/>arm_sJJ_frame_stride1_per_frame.csv]
    G --> H[r22_build_matrix.py<br/>cumulative prefix means]
    H --> I[csv/r22_matrix.csv<br/>csv/r22_bottom_right_check.csv]
    I --> J[r22_figures.py]
    J --> K[figures/r22_*.pdf<br/>deviation from final column]
    I --> L[r22_figures.py --absolute<br/>CLIP + LPIPS actual values]
    L --> M2[figures/r22_abs_*.pdf]
    E --> N[r22_mosaic.py<br/>rows i%10==0 x cols j in 0,2,5,8,11,14]
    N --> O[figures/r22_mosaic/arm/video_eT.pdf<br/>66 pages]
```

## Code to touch

| File | Change |
|---|---|
| `Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py` | Add `step_dump: Optional[list] = None` to `inference()` and `rollout_inference()`; three write sites in `inference` (allocate / write / assert+append) and forwarding + window-slicing in `rollout_inference`. Default `None` ⇒ no added work, bit-for-bit unchanged. |
| `evaluation/r22_dump_steps.py` | **New** — driver: load pipe once, per-pair `set_seed`, `rollout_inference(return_latents=True, wo_video_decode=True, step_dump=steps)`, assert, then decode+save each step latent one at a time. |
| `slurm_scripts/five_bench/r22_smoke.sh` | **New** — 1 clip x paper_novp, GATE1 latent assert + GATE2 PNG bit-parity vs `five_bench/baseline/edit1/0001_bus`. L40S, 1h. |
| `slurm_scripts/five_bench/r22_dump.sh` | **New** — `#SBATCH --array=0-2`, L40S, `--mem=64G`, 6h. Task→arm; inner loop `EDIT_TYPES=(1 2 5 6)`. Same conda dance as `r20_infer.sh`. |
| `slurm_scripts/five_bench/r22_eval.sh` | **New** — `#SBATCH --array=0-2`, L40S, `--mem=64G`, `--time=20:00:00`. `five-bench` env. Inner loop `j=0..14` → `evaluate.py --per_frame --frame_stride 1 --metrics <8> --tgt_layout edit_video --cases_json --tgt_key r22_{arm}_s{jj}`. `rm -rf` `_resize` dirs after each `j`; exit non-zero on any `Error:` line. |
| `evaluation/r22_build_matrix.py` | **New** — glob → pivot → cumulative mean → `r22_matrix.csv`; L3 identity assert → `r22_bottom_right_check.csv`. |
| `evaluation/r22_figures.py` | **New** — per-clip heatmap grids + normalised-fraction mean surfaces. **Extended 2026-08-18**: `--absolute` mode plotting `M(i,j)` on a sequential colormap for the CLIP/LPIPS metrics (`--abs_metrics`), emitting `r22_abs_{metric}_{arm}.pdf` + `r22_abs_mean_surfaces.pdf`. Reuses `clip_surface` / `resample_fraction`; the deviation path must stay bit-identical. |
| `evaluation/r22_mosaic.py` | **New 2026-08-18** — per (video, arm) image mosaic straight off the step dump; rows = frames `i % --frame_mod == 0`, cols = `--steps`. Downsamples tiles to `--tile_width`. Skips and reports missing frames rather than aborting the set. |

Implementation notes:

- **Allocation point matters.** `step_out` must be cloned from `output` *after* Step 2's `output[:, left:right] = current_trg_ref_latents` ([line 425](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L425)) so the initial-latent prefix (VP first frame under `vp`, overlap seeding on windows ≥ 2) is present in every step slice. `denoising_step_list` is read at line 442, which is the natural place.

  ```python
  # after: denoising_step_list = self.denoising_step_list
  if step_dump is not None:
      step_out = output.unsqueeze(0).repeat(len(denoising_step_list), *([1] * output.ndim)).clone()
  ...
  # inside the step loop, immediately after `denoised_pred = noisy_pred_input - t_i * v_t`
  if step_dump is not None:
      step_out[index][:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
  ...
  # after the block loop, before returning
  if step_dump is not None:
      assert torch.equal(step_out[-1], output), "R22: step-14 dump diverged from output"
      step_dump.append(step_out)
  ```

- **Rollout slicing must mirror `rollout_latent`.** `rollout_inference` keeps the whole first window and `rollout_latent[:, rollout_overlap:]` thereafter ([line 193](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L193)). The step tensor carries a leading step axis, so the same cut is `[:, :, rollout_overlap:]` and the concat is `dim=2`. Getting this wrong silently misaligns frames only on the 1–8 multi-window clips, which is exactly the class of bug the smoke (a single-window clip) cannot catch — check `0034_cows` explicitly in `wait-dump`.

- **Decode one at a time.** 15 decoded videos at `[81,3,480,832]` fp32 is 7.3 GB. Decode → save PNGs → free, with `pipeline.vae.model.clear_cache()` between calls, mirroring the `use_cache=False` pattern `rollout_inference` uses at [line 216](../Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py#L216). If the 15 decodes are *not* mutually independent, GATE2 fails — that is precisely what it is for.

- **`--frame_stride 1` is load-bearing.** `--per_frame` records `frame_idx = f_i * frame_stride` ([evaluate.py:475](../evaluation/fivebench/evaluate.py#L475)), so at the default stride 8 the matrix would have ~9 rows, not `nb_frames`. Stride 1 is the only reason the row axis is per-pixel-frame.

- **Cumulative mean is exact, not an approximation.** `evaluate.py` reduces each image metric with `calculate_mean` over the per-frame list ([line 484](../evaluation/fivebench/evaluate.py#L484)), so the prefix-truncated value is the prefix mean of the same values — identical to re-running the metric on `V[:i]`. This is what turns an `nb_frames x 15` cost into a `15` cost. It holds only for the 8 metrics listed; do not extend the set without re-deriving it.

- **Do NOT check the bottom-right cell against `r21_*_avg.csv` or `r20_*_avg.csv`** — they will not match, and the mismatch is not a bug. Two independent reasons: (a) those runs used `--frame_stride 8`, so their per-clip value is a mean over every 8th frame while `M(nb_frames,14)` is a mean over *all* frames; (b) the `_avg.csv` overall row is `evaluate.py`'s **mean of per-edit-type means**, so on the 22-clip subset edit1 (1 clip) weighs as much as edit2 (16), whereas `M` is per clip. The only legitimate targets for the L3 assert are internal: the plain mean of the same step-14 per-frame column, per (arm, clip, metric). If a cross-check against a stored arm is wanted, it belongs at **L2** (bit-identical PNGs in the smoke), where it is exact and unambiguous.

- **QOS cap.** R21 hit `QOSMaxSubmitJobPerUserLimit` at ~8 *array tasks* (not jobs). Every array here is 3 tasks, so `dump` and `evaluate` each fit comfortably even with interactive sessions queued — but do not merge them into one 45-task array.
