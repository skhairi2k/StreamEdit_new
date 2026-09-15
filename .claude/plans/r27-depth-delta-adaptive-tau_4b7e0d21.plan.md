---
name: R27 — Depth-Delta Adaptive Tau
overview: "Replace R25's IoU routing signal with a depth-based measure of how much shape change an edit actually demands. IoU is dominated by the covered surface, not by geometry: adding a small hat scores about the same IoU as a pure colour change, so it routes to a small tau and fails. R27 measures instead the mean ABSOLUTE difference between the monocular depth map of the source first frame and of the Qwen anchor, averaged over the pixels where that difference exceeds the estimator's own noise floor, which makes the quantity independent of how much area the edit covers. Depth comes from Depth-Anything-V2-Large; the anchor's depth is least-squares affine-aligned to the source's over the BACKGROUND (outside R25's dumped frame-0 masks) before differencing, to cancel the per-image scale/shift ambiguity of relative depth; the threshold defining L is 3 * a robust MAD scale of that aligned background residual. The magnitude is mapped to tau by interpolating the SOURCE-ANCHORING BUDGET A(tau) = sum_i t_i**tau linearly and inverting, not by interpolating tau itself: tau is an exponent whose effect saturates, so a linear map over [2, 50] would put a mid-magnitude edit at tau = 25.5 with 1/22 of Eq. 4's source budget. tau_min = 2.0 is NOT a tuned floor but a boundary condition -- m = 0 means the measured depth change is indistinguishable from estimator noise, and the schedule must then reduce exactly to StreamGVE Eq. 4. tau reaches the model through R25's existing `--tau_map` flag on run_fivebench.py, so no pipeline change is needed. Two arms render over the same 22 cases as R25: the depth-adaptive map, and a constant-tau control at the realized mean, so the arm can be attributed without the phase-1/phase-2 split R25 needed."
task_id: R27
todos:
  - id: write-depth
    content: "evaluation/r27_depth_delta.py — per case, the depth-delta magnitude, no video generation. Load source `{data_root}/images/{video}/00001.jpg` and anchor `{anchor_root}/edit{T}/{video}.png` exactly as r25_iou.py:293 does. Run Depth-Anything-V2-Large on each; resample the anchor depth onto the SOURCE grid with BILINEAR (depth is continuous — nearest would alias silhouettes, which is where the whole signal lives), matching r25_iou.py's convention that every quantity is computed on the source jpg's (H, W). Read the background region from R25's dumped frame-0 masks `evaluation/figures/r25_iou_masks/{case_id}.npz` (np.unpackbits over `shape`): B = NOT(m_src | m_edit). Fit (a, b) by least squares minimising ||a*D_edit + b - D_src||^2 over B, apply it, then R = |D_src - (a*D_edit + b)| — ABSOLUTE, per the locked decision. Noise floor sigma = 1.4826 * median(|R_B - median(R_B)|) over B; L = {p : R(p) > 3*sigma}; D_raw = mean_{p in L} R(p), and 0 if L is empty. Also emit D_norm = D_raw / IQR(D_src): DAv2 min-max normalises per image, so after aligning to the source the units are the SOURCE's depth range — without this a shallow scene reads a small D no matter what the edit does. D_norm is the routed signal. Columns: case_id, video_name, edit_type, D_raw, D_norm, sigma, a, b, |L|/HW, frac_L_in_mask, iqr_src, aspect_src, aspect_anchor. Optional --viz_dir writes a per-case 2x3 panel (source / anchor / D_src / D_edit_aligned / R / L) — the only artefact that shows WHERE the signal came from. HF_HUB_OFFLINE=1 after the model is fetched."
    status: completed
  - id: write-taumap
    content: "evaluation/r27_tau_map.py — read r27_depth.csv, normalise, map to tau, emit the same (video_name, edit_type, iou, tau) schema r25_tau_map.py emits so run_fivebench.py --tau_map needs no change (the `iou` column carries D_norm here; keeping the header identical is deliberate — it is what lets R27 reuse R25's plumbing untouched). m = clip((D_norm - D_lo) / (D_hi - D_lo), 0, 1) with D_lo = the case's own noise floor expressed in the same normalised units (so m = 0 IS the statistical-indistinguishability point, not a dataset minimum) and D_hi = the MAXIMUM measured D_norm over the case set (`--d_hi_case auto`), resolved at calibration time and then WRITTEN INTO THE MAP HEADER AS AN ABSOLUTE CONSTANT so the map is reproducible and applies to a single new video. Revised 2026-08-31: anchoring on 0011_lucia_e5 was measured to clip 11 of 21 cases to tau_max, because e5 reads D_norm 0.395 against a maximum of 1.202. `auto` clips nobody by construction; r27_tau_map.py must still print the resolved anchor case, its D_norm and the runner-up's, since a lone outlier at the top compresses everyone else. Then BUDGET-LINEAR: A(tau) = sum_i t_i**tau over the 15 t_next values of the --step 15 schedule; A_m = (1-m)*A(tau_min) + m*A(tau_max); tau = A^{-1}(A_m) by Brent on the monotone A. tau_min = 2.0, tau_max = 50.0. Store the step count in the CSV header comment — A is schedule-dependent, and a map calibrated at --step 15 silently changes meaning at any other step count. --assign {depth,constant}: `constant` writes the realized mean tau for every pair. Refuse to write if fewer than 3 distinct tau values result (a degenerate map carries no routing signal and the arm is not worth rendering)."
    status: completed
  - id: write-infer
    content: "slurm_scripts/five_bench/r27_infer.sh — modelled on r25_infer.sh, `#SBATCH --array=0-5` one edit type per task, `ARM=${R27_ARM:-depth}` selecting METHOD (`r27_${ARM}_vp`) and TAU_MAP (`evaluation/csv/r27_tau_map_${ARM}.csv`). `--vp_mode vp --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors --cases_json evaluation/cases.json --step 15 --fg_boost_factor 4 --seed 0`, OUT_ROOT /projects/dataggen/outputs/five_bench/r27_depth_tau. L40S, --mem=64G, --time=04:00:00, `--exclude=node52` (node52 advertises gpu:8 while exposing no device to batch jobs; it killed two R25 and one R26 submission). Everything except the tau map must be identical across the two arms — the attribution argument rests on that. Preflight: refuse to load the model unless the map covers all 22 pairs."
    status: completed
  - id: write-eval
    content: "slurm_scripts/five_bench/r27_eval.sh — modelled on r25_eval.sh, `ARM=${R27_ARM:-depth}`, single task, partition L40S,A100, --exclude=node52, --mem=64G, --time=06:00:00, `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Pass the SAME explicit 9-metric `--metrics` list R25 and R26 used (structure_distance, psnr/lpips/mse/ssim_unedit_part, clip_similarity_source/target/target_edit_part, niqe_target_image) — comparability with the stored R25 numbers is the point of R27, and a different metric set would break it. config.yaml's `metrics:` key is VESTIGIAL: evaluate.py:200 reads args.metrics, so editing the config changes nothing."
    status: completed
  - id: write-summarize
    content: "evaluation/r27_summarize.py — join the six edit{T}_FiVE_r27_{arm}_vp_frame_stride8.csv per arm against the stored edit{T}_FiVE_r21_ref_vp_frame_stride8.csv on `file_id`, re-averaging the reference over ONLY these 22 cases (never its 419-pair mean). Emits evaluation/csv/r27_depth_tau.csv: per case D_norm, m, tau, both arms' 9 metrics, per-metric deltas against Eq. 4, an overall-mean row and a per-edit-type block. Report the two contrasts separately: depth-adaptive - Eq.4 (does it help at all) and depth-adaptive - constant (does VARYING tau help, i.e. is the depth routing doing the work). Per-case sign tests on both, not just means: R25's +1.064 mean CLIP gain hid a 15/7 split at p = 0.134."
    status: pending
  - id: write-figures
    content: "evaluation/r27_figures.py — three pages. (a) r27_tau_depth.pdf: realized tau against D_norm, one marker per edit type, Eq.4 rho=2 line drawn across, with the R25 IoU-derived tau overlaid per case — this is the figure that shows whether depth separates the cases IoU could not (the hat/colour collapse that motivated R27). (b) r27_delta.pdf: per-case delta for clip_similarity_target_image and lpips_unedit_part against |tau - 2|. This is the diagnostic that decides the task, and it is the one R25 failed: if the gain is a flat cloud (R25 measured r = +0.101, p = 0.656) the routing contributed nothing even if the mean moved. (c) r27_grids/{case_id}.pdf, F x 4 qualitative pages — source / Eq.4 render / depth-residual R overlay / adaptive render, rows aligned by NORMALIZED TIME because the VAE changes frame count (80 source vs 69 rendered on 0001_bus)."
    status: pending
steps:
  - id: fetch-model
    type: local
    command: |
      python -c "
      from transformers import AutoImageProcessor, AutoModelForDepthEstimation
      m='depth-anything/Depth-Anything-V2-Large-hf'
      AutoImageProcessor.from_pretrained(m); AutoModelForDepthEstimation.from_pretrained(m)
      print('cached:', m)"
    output_paths:
      - ~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Large-hf
    status: completed
    completed_at: 2026-08-31

  - id: depth
    type: srun
    wait_for: fetch-model
    command: |
      source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
      srun --partition=L40S --gres=gpu:1 --mem=32G --time=01:00:00 --pty \
        env HF_HUB_OFFLINE=1 python evaluation/r27_depth_delta.py \
          --cases evaluation/cases.json \
          --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
          --anchor_root /projects/dataggen/outputs/five_bench/anchors \
          --mask_dir evaluation/figures/r25_iou_masks \
          --mad_k 6.0 --min_effect 0.10 --erode_radius 3 \
          --viz_dir evaluation/figures/r27_depth_panels \
          -o evaluation/csv/r27_depth.csv
    output_paths:
      - evaluation/csv/r27_depth.csv
      - evaluation/figures/r27_depth_panels/
    status: completed
    completed_at: 2026-08-31
    job_id: "965047"

  - id: check-depth
    type: manual
    wait_for: depth
    check_hint: "Five gates, and gate 4 can kill the task. (1) ALIGNMENT SANITY: `a` should sit near 1 and `b` near 0 for every case; a wildly different `a` means the background fit latched onto edited pixels, i.e. the R25 masks under-cover this edit. (2) GEOMETRY: aspect_src vs aspect_anchor must match to within ~2%. If they differ the anchor resize WARPS the scene and ΔDepth picks up spurious structure along every edge, not just the edited object — the whole measurement is then invalid and the anchors must be re-rendered at source aspect. (3) LOCALISATION: read **frac_in_top**, computed on the same eroded slice D is averaged over. It must be >= 0.5. **Re-measured after erosion was adopted into the magnitude: 22/22 pass, median 1.000** (it was 20/22 before, failing on 0016_horsejump-high 0.31 and 0017_kid-football 0.42 — both were halo, and erosion removed it). frac_L_in_mask stays in the CSV as a secondary diagnostic and is NOT the gate. (4) THE BOUNDARY CONDITION, edit3 only: the COLOUR cases must read D_norm at the bottom of the range. **After erosion this PASSES as a measured result rather than an assumption**: 0058_boat 0.006 and 0017_kid-football 0.012 against a swap minimum of 0.689, i.e. **56.8x separation** (it was 0.95x and INVERTED before erosion). Both route to tau 2.01 = Eq. 4 exactly. Type 4 remains an observation, not pass/fail: 0089_A_dog (-> plush) 1.550 sits among the swaps and 0028_kite-walk_e4 (-> wooden) 1.072 lower, ordered as the physics would order them. ⚠ this result depends on min_effect = 0.10; see the Known weakness row. (5) CEILING: D_hi is `auto`, the argmax of D_norm, so exactly one case sits at m = 1 and NOTHING clips — that is the point of the revision, after anchoring on 0011_lucia_e5 was measured to clip 11 of 21. Record which case won the argmax and by what margin over the runner-up: a single outlier setting the ceiling compresses everyone else, and if the top case leads the second by more than ~2x, consider whether it is a genuine maximum or a measurement artefact (read its panel). ALSO record where 0011_lucia_e5 — the case that motivated R27 — now lands on the m axis and what tau it receives. It is no longer guaranteed tau_max, so R27 must now EARN its founding case rather than define it; if e5 routes low and still fails in the render, the premise that this edit needs detachment is what is wrong, not the mapping."
    status: completed
    completed_at: 2026-08-31

  - id: taumap
    type: local
    wait_for: check-depth
    command: |
      python evaluation/r27_tau_map.py --assign depth \
        --depth_csv evaluation/csv/r27_depth.csv \
        --d_hi_case auto --tau_min 2.0 --tau_max 50.0 --step 15 \
        -o evaluation/csv/r27_tau_map.csv
      python evaluation/r27_tau_map.py --assign constant \
        --depth_csv evaluation/csv/r27_depth.csv \
        --d_hi_case auto --tau_min 2.0 --tau_max 50.0 --step 15 \
        -o evaluation/csv/r27_tau_map_constant.csv
    output_paths:
      - evaluation/csv/r27_tau_map.csv
      - evaluation/csv/r27_tau_map_constant.csv
    status: completed
    completed_at: 2026-09-01

  - id: check-taumap
    type: manual
    wait_for: taumap
    check_hint: "Both maps must hold the SAME 22 (video_name, edit_type) pairs — run_fivebench.py hard-errors on missing coverage, so a short map fails at startup rather than rendering a half-adaptive arm that looks like a result. depth: every tau in [2.0, 50.0]; EXACTLY ONE case may read 50.00 — the argmax that defines D_hi (definitional, NOT evidence the routing works). Any second case at 50.00 means something clipped, which `auto` makes impossible, so it would indicate the map was built from a different CSV than check-depth read. Confirm the header records the resolved D_hi value and the case it came from; the colour/material cases must read close to 2.00 per check-depth gate 4. Spot-check ONE row by hand against the budget formula — A(2) = 4.5062 and A(50) = 0.0320 at --step 15, so m = 0.6 must give A = 1.8217 and tau = 5.53. constant: every row equal to the realized mean tau, to 2 dp. Confirm the CSV header records --step 15: A is schedule-dependent and the map is only valid for the schedule it was built for."
    status: completed
    completed_at: 2026-09-01

  - id: infer
    type: sbatch
    wait_for: check-taumap
    command: |
      R27_ARM=depth    sbatch --export=R27_ARM=depth    slurm_scripts/five_bench/r27_infer.sh
      R27_ARM=constant sbatch --export=R27_ARM=constant slurm_scripts/five_bench/r27_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-09-01
    job_id: "972542,972543"

  - id: wait-infer
    type: manual
    wait_for: infer
    check_hint: "Per arm: 1/13/2/2/3/1 frame dirs = 22, `failures: 0` in all six task logs, and the per-pair tau echoed in each [ok] line matching that arm's map. For `constant` every [ok] line must read the same tau — if any row differs the wrong map was picked up. Check the GPU diagnostics header (node, nvidia-smi -L, torch cuda_ok) even though node52 is excluded. Note: --export=R27_ARM=<arm> and NOT --export=ALL,R27_ARM — ALL re-inherits this shell's LD_LIBRARY_PATH, and the R25 runs that succeeded had a clean environment; the scripts source conda by absolute path so they need nothing inherited."
    sets_status: finished
    status: completed
    completed_at: 2026-09-01

  - id: eval
    type: sbatch
    wait_for: wait-infer
    command: |
      R27_ARM=depth    sbatch --export=R27_ARM=depth    slurm_scripts/five_bench/r27_eval.sh
      R27_ARM=constant sbatch --export=R27_ARM=constant slurm_scripts/five_bench/r27_eval.sh
    status: completed
    completed_at: 2026-09-01
    job_id: "973845,973846"

  - id: wait-eval
    type: manual
    wait_for: eval
    check_hint: "Per arm: 6 per-edit-type _avg.csv, 0 `Error:` lines, 0 OOM lines, row counts 1/13/2/2/3/1, no column misalignment (upstream's `except: continue` DROPS a metric column and shifts every later one, so a partial run is corrupt rather than merely incomplete — quarantine, do not keep). Both arms must be scored with the identical 9-metric --metrics list, which is also what the stored R21/R25 references used."
    status: pending

  - id: summarize
    type: local
    wait_for: wait-eval
    command: |
      python evaluation/r27_summarize.py \
        --arms depth constant \
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
        --depth_csv evaluation/csv/r27_depth.csv \
        --tau_map evaluation/csv/r27_tau_map.csv \
        -o evaluation/csv/r27_depth_tau.csv
    output_paths:
      - evaluation/csv/r27_depth_tau.csv
    status: pending

  - id: figures
    type: local
    wait_for: summarize
    command: |
      python evaluation/r27_figures.py \
        --summary_csv evaluation/csv/r27_depth_tau.csv \
        --r25_tau_map evaluation/csv/r25_tau_map.csv \
        --clip_col clip_similarity_target_image \
        --lpips_col lpips_unedit_part \
        --tau_baseline 2.0 \
        --out_dir evaluation/figures
    output_paths:
      - evaluation/figures/r27_tau_depth.pdf
      - evaluation/figures/r27_delta.pdf
      - evaluation/figures/r27_grids/
    status: pending

  - id: verdict
    type: manual
    wait_for: figures
    check_hint: "Record in daily.md, in this order. (a) Did depth SEPARATE what IoU could not? r27_tau_depth.pdf overlays the R25 tau per case: the founding claim is that IoU collapses small-addition edits onto colour edits. If the two signals rank the 22 cases the same way, R27 measured a more expensive version of R25 and the story ends there. (b) depth-adaptive - Eq.4 on the 22-case means, per metric, with the editability/fidelity trade stated explicitly — R25 bought +1.064 CLIP for -3.04 dB PSNR, so a bare CLIP gain is not a result. (c) THE DECIDING CONTRAST, depth-adaptive - constant: same tau distribution in both arms, so a win here is attributable to the ROUTING rather than to a better constant exponent. This is what R25 could not answer in phase 1. (d) Does per-case gain track |tau - 2| (r27_delta.pdf) or is it the flat cloud R25 measured at r = +0.101, p = 0.656? A flat cloud disqualifies the routing regardless of (b). (e) THE PREMISE TEST. 0011_lucia_e5 routes to tau 2.89 (rank 19/22) because the depth delta reads an added dog as a small displacement. If it SUCCEEDS at 2.89, the belief that it needed tau ~50 was wrong and R27 is vindicated on its own founding case. If it FAILS, that is a clean negative result: a depth delta is the wrong proxy for additions, because it measures how far depth MOVED while an addition demands content that does not exist in the source at all. State which, and name what the next signal would have to capture. Check the other two additions (0007_guitar-violin 3.35, 0069_car-turn 3.02) the same way — n=3, so this is weak evidence either way and must be labelled as such. (f) Per-case sign tests on (b) and (c), not just means, at n = 22. CAVEATS to carry into the write-up: only 9 of 16 metrics are scored, so there is NO temporal-consistency metric (motion_fidelity_score was dropped when H100 proved unavailable) — the axis most at risk from faster source release is unmeasured, and a flicker blowout would currently read as a win; and tau_max = 50 is a stated assumption from one qualitative observation on 0011_lucia_e5, never swept."
    sets_status: analyzed
    status: pending
isProject: true
---

# R27: Depth-Delta Adaptive Tau

## Context

StreamGVE's blender rate (Eq. 4) is implemented at [causal_model.py:334](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L334) as `blender_rate = 1 - t_{i+1} ** blend_power`, i.e. a source-anchoring weight

$$W^{src}(t_i) = t_i^{\tau}, \qquad r = 1 - W^{src}$$

with a single global exponent `tau = blend_power = 2`. Large tau releases the source anchor fast; small tau holds it.

R25 made that exponent per-video, driven by the IoU between the object region in the source first frame and in the Qwen anchor. The defect R27 addresses is that **IoU responds to covered surface, not to demanded shape change**: adding a small hat barely moves the IoU, so it routes to nearly the same tau as a pure colour edit and then fails for want of detachment. R27 replaces the signal with a depth-based magnitude,

$$D = \operatorname*{mean}_{p \in L} \bigl| \Delta\mathrm{Depth}(p) \bigr|, \qquad L = \{\, p : |\Delta\mathrm{Depth}(p)| > 3\sigma \,\}$$

measured once on clean pixels before any denoising, between the depth map of the source first frame and that of the anchor. Averaging over `L` alone is what makes it surface-independent.

**Done when:** the two arms are scored over the same 22 pairs as R25, and the verdict states (i) whether the depth signal ranks the cases differently from IoU, (ii) whether the adaptive arm beats the constant-tau control, and (iii) whether the per-case gain tracks `|tau - 2|` rather than forming the flat cloud R25 measured.

## Execution steps

| # | Step id | Type | What it does | sets_status |
|---|---------|------|--------------|-------------|
| — | *(prep)* | todos | `r27_depth_delta.py`, `r27_tau_map.py`, `r27_infer.sh`, `r27_eval.sh`, `r27_summarize.py`, `r27_figures.py` | — |
| 1 | `fetch-model` | local | Cache Depth-Anything-V2-Large (needs network) | — |
| 2 | `depth` | srun | Depth delta per case → `r27_depth.csv` + panels | — |
| 3 | `check-depth` | manual | Alignment, aspect, localisation, **boundary condition**, motivating case | — |
| 4 | `taumap` | local | Budget-linear map + constant control map | — |
| 5 | `check-taumap` | manual | Coverage, endpoints, hand-check one budget row | — |
| 6 | `infer` | sbatch | Two arms × 6 edit types, 22 clips each | `running` |
| 7 | `wait-infer` | manual | 22 dirs/arm, `failures: 0`, tau echoed per pair | `finished` |
| 8 | `eval` | sbatch | 9-metric evaluation, both arms | — |
| 9 | `wait-eval` | manual | 6 `_avg.csv`/arm, no `Error:`/OOM, no column shift | — |
| 10 | `summarize` | local | Three-way table vs stored Eq. 4 reference | — |
| 11 | `figures` | local | Mapping, per-case delta, qualitative grids | — |
| 12 | `verdict` | manual | Record the two contrasts + caveats in `daily.md` | `analyzed` |

Invoke with `/run-step R27 <step-id>`, e.g. `/run-step R27 depth`. Bare `/run-step R27` takes the first `pending` step whose `wait_for` is satisfied.

## Decisions

| Question | Locked choice | Why |
|---|---|---|
| Signal | Mean over a **zero-padded fixed-size** top slice (largest 0.5% of frame) of `\|ΔDepth\|/IQR(D_src)`, taken over `erode(L, r=3)` | *Revised twice.* First from `mean over L` (threshold was the conditioning event, inflating D mechanically). Then, 2026-08-31, erosion added: the anchor is a re-synthesis so object OUTLINES shift sub-pixel and blaze at depth discontinuities; those halos are 1–3 px wide and an erosion removes them. Zero-padding the slice (denominator always n) is what makes a halo-only case score ≈ 0 rather than averaging its few extreme survivors. **Colour/swap separation 0.95× → 56.8×**, `frac_in_top` failures **2 → 0**, and both colour edits now land at `tau` 2.01 |
| Signed or absolute | **Absolute** | A hat moves depth toward the camera, a removal away; sign would cancel |
| Alignment | Least-squares affine `a·D_edit + b → D_src` over background | Monocular relative depth is scale/shift ambiguous per image |
| Background region | `NOT(m_src ∪ m_edit)` from `evaluation/figures/r25_iou_masks/` | Already on disk from R25, already on the source grid |
| Comparison grid | **Anchor grid (480×832)**; images resized bilinear before depth, masks nearest | *Corrected at build time.* The plan said source grid; `run_fivebench.py:225` does `transforms.Resize((480,832))`, an anisotropic squeeze, so the anchor already carries the squeezed geometry and stretching it back to 864 adds a 3.7% warp that was never in the data. `--grid source` reproduces the old wording |
| `L` and its threshold | `thr = max(6σ, 0.10·IQR(D_src))`, σ = `1.4826 · MAD` of the **signed** background residual | MAD not std (depth error clusters at edges); signed not folded (the residual is zero-mean over the background by construction). The `min_effect` floor exists because σ is estimated on the BACKGROUND only, so a featureless background collapses it — `0091_A_hawk` gave `MAD = 0` exactly. ⚠ **`L` is NO LONGER diagnostic-only**: since 2026-08-31 it is the candidate set the routed slice is drawn from, so `D` depends on `thr`. Measured: `D` is EXACTLY invariant for most cases (median rel. change 0.00% across `mad_k` {3,6} × `min_effect` {0.02,0.10}) because a real region-level change leaves far more than 1997 px after erosion — but **highly sensitive for low-signal cases** (`0017_kid-football` 0.012 → 0.372 when `min_effect` drops to 0.02). `mad_k` is nearly irrelevant; **`min_effect = 0.10` is load-bearing** |
| 🛑 **BLOCKED 2026-08-31** | The depth signal cannot produce the required ordering; `taumap` onward is on hold pending a synthesis-sensitive signal | Target is colour/material < 5, swap ~10, add/remove 30–50, i.e. **add/remove ABOVE swap**. Measured class means put **addition BELOW swap** under every statistic tested (A: add 0.906 vs swap 1.346; B: 0.554 vs 0.841). Budget-linear is monotone, so no `tau_min`/`tau_max`/curvature/`D_lo`/`D_hi` choice can invert that order — this is not a calibration gap. Separately, the targets need add/remove at `m ∈ [0.976, 1.0]` (the top 2.4% of the range) while the measured distribution tops out at `m = 0.74`. Root cause: a depth delta measures how far depth MOVED, bounded for an addition by the added object's own extent, whereas the requirement is about SYNTHESIS — content with no counterpart in the source |
| ⚠ Premise risk | The depth signal systematically **under-rates ADDITIONS**, the class R27 exists to rescue | All three type-5 cases sit below the swap median (`guitar-violin` 1.062, `car-turn` 0.871, `lucia_e5` 0.784). Geometrically correct — an addition displaces depth only over the added object's extent, a swap over the union of two silhouettes, a removal over the whole vacated region — but R27's premise was that additions need detachment because content must be SYNTHESISED, which a displacement measure cannot see. **No `D_hi` choice repairs it**: `0011_lucia_e5` would need `D_hi = 0.977` to reach `tau = 10`, at which point 15 of 22 cases clip. Decision: proceed and let the render adjudicate |
| ⚠ Known weakness | `min_effect = 0.10` was chosen by inspecting `frac_L_in_mask`, and now determines a headline result | The colour edits' `tau` 2.01 — the evidence for the `tau_min` boundary condition — depends on this constant. It has a physical reading (a change under 10% of the scene's depth span is not a shape change) but was not selected independently of the outcome. **The verdict must report the colour cases at `min_effect` ∈ {0.05, 0.10} so the claim's dependence is visible** |
| Validity gate | `frac_in_top` ≥ 0.5, computed on the same eroded slice `D` uses | **22/22 pass, median 1.000** (was 20/22). Fraction of the routed slice inside the edit region; low ⇒ the anchor differs from the source somewhere other than where it was asked to |
| `D_lo` (m = 0) | **0** | Chosen 2026-08-31. The planned "case's own noise floor" has no referent under `topq`, which always returns a positive top slice. A matched null (same statistic over the background) was measured and rejected: it makes the boundary condition fire cleanly on both colour edits, but `0016_horsejump-high`'s under-covering mask contaminates its own null and drops a genuine swap to `tau` 2.00. `D_lo = 0` has no dependence on mask quality |
| Routed quantity | `D_norm = D_raw / IQR(D_src)` | DAv2 min-max normalises per image; without this a shallow scene reads small `D` regardless of the edit |
| Depth model | `depth-anything/Depth-Anything-V2-Large-hf` | Deterministic single pass; Marigold is stochastic and would inject variance into `tau` |
| `D_lo` (m = 0) | The case's own noise floor | Makes `m = 0` the statistical-indistinguishability point, not a dataset minimum |
| `D_hi` (m = 1) | **`--d_hi_case auto`**: the case with the maximum measured `D_norm` | *Revised 2026-08-31 after the calibration preview.* `0011_lucia_e5` read `D_norm = 0.395`, rank 13 of 21, so anchoring there clipped **11 of 21** cases to `tau_max`. `auto` resolves to the argmax (currently `0011_lucia_e2`, person → lion, `D_norm = 1.202`), which by construction clips nobody. The resolved VALUE is written into the map header as an absolute constant, so the map stays reproducible and defined for a single new video even though the anchor was picked from this set |
| Mapping `f` | **Budget-linear**: interpolate `A(τ)=Σ_i t_i^τ`, invert | `tau` is an exponent whose effect saturates; linear over `[2,50]` puts `m=0.5` at `tau=25.5` |
| `tau_min` | **2.0**, fixed | Boundary condition, not a tuned floor: `m=0` ⇒ Eq. 4 exactly. Zero free parameters at the low end; the method strictly contains the baseline |
| `tau_max` | **50.0** | Stated assumption from the `0011_lucia_e5` observation — **not swept** |
| Schedule | `--step 15`, recorded in the map header | `A` is schedule-dependent; at `--step 7`, `tau=50` is degenerate (`W^src ≈ 4e-4`) |
| Arms | `depth` + `constant` (realized mean tau) | Separates "routing helps" from "a better constant helps" — what R25 phase 1 could not do |
| Reference | Stored `r21_ref_vp`, re-averaged over these 22 cases | Never its 419-pair mean |
| Metrics | Same explicit 9-metric `--metrics` list as R25/R26 | Comparability is the point; `config.yaml`'s `metrics:` key is vestigial (`evaluate.py:200`) |
| Conda env | `streamgve` for `depth` and `infer`; `five-bench` for `eval` | Matches R25/R26. Activate EXPLICITLY — `PATH` can point at an env's python while `CONDA_DEFAULT_ENV` still says `base`, which is how `fetch-model` ran |
| Cases | The same 22 pairs of `evaluation/cases.json` | Directly comparable with the R25 arm and its controls |
| Out of scope | Sweeping `tau_max`; fitting the curvature from per-clip optima; temporal metrics | Time-boxed; recorded as caveats in `verdict` |

## Step commands

### fetch-model

```bash
python -c "
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
m='depth-anything/Depth-Anything-V2-Large-hf'
AutoImageProcessor.from_pretrained(m); AutoModelForDepthEstimation.from_pretrained(m)
print('cached:', m)"
```

### depth

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
srun --partition=L40S --gres=gpu:1 --mem=32G --time=01:00:00 --pty \
  env HF_HUB_OFFLINE=1 python evaluation/r27_depth_delta.py \
    --cases evaluation/cases.json \
    --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
    --anchor_root /projects/dataggen/outputs/five_bench/anchors \
    --mask_dir evaluation/figures/r25_iou_masks \
    --mad_k 6.0 --min_effect 0.10 --erode_radius 3 \
    --viz_dir evaluation/figures/r27_depth_panels \
    -o evaluation/csv/r27_depth.csv
```

### taumap

```bash
python evaluation/r27_tau_map.py --assign depth \
  --depth_csv evaluation/csv/r27_depth.csv \
  --d_hi_case auto --tau_min 2.0 --tau_max 50.0 --step 15 \
  -o evaluation/csv/r27_tau_map.csv

python evaluation/r27_tau_map.py --assign constant \
  --depth_csv evaluation/csv/r27_depth.csv \
  --d_hi_case auto --tau_min 2.0 --tau_max 50.0 --step 15 \
  -o evaluation/csv/r27_tau_map_constant.csv
```

### infer

```bash
R27_ARM=depth    sbatch --export=R27_ARM=depth    slurm_scripts/five_bench/r27_infer.sh
R27_ARM=constant sbatch --export=R27_ARM=constant slurm_scripts/five_bench/r27_infer.sh
```

### eval

```bash
R27_ARM=depth    sbatch --export=R27_ARM=depth    slurm_scripts/five_bench/r27_eval.sh
R27_ARM=constant sbatch --export=R27_ARM=constant slurm_scripts/five_bench/r27_eval.sh
```

### summarize

```bash
python evaluation/r27_summarize.py \
  --arms depth constant \
  --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
  --depth_csv evaluation/csv/r27_depth.csv \
  --tau_map evaluation/csv/r27_tau_map.csv \
  -o evaluation/csv/r27_depth_tau.csv
```

### figures

```bash
python evaluation/r27_figures.py \
  --summary_csv evaluation/csv/r27_depth_tau.csv \
  --r25_tau_map evaluation/csv/r25_tau_map.csv \
  --clip_col clip_similarity_target_image \
  --lpips_col lpips_unedit_part \
  --tau_baseline 2.0 \
  --out_dir evaluation/figures
```

## Pipeline

```mermaid
flowchart TD
  SRC[FiVE images/&lt;video&gt;/00001.jpg] --> DD[r27_depth_delta.py]
  ANC[anchors/edit&lt;T&gt;/&lt;video&gt;.png] --> DD
  MSK[r25_iou_masks/*.npz<br/>background region] --> DD
  DAV2[Depth-Anything-V2-Large] --> DD
  DD --> DCSV[csv/r27_depth.csv]
  DD --> DPAN[figures/r27_depth_panels/]
  DCSV --> TM[r27_tau_map.py]
  TM --> TMAP[csv/r27_tau_map.csv]
  TM --> TCON[csv/r27_tau_map_constant.csv]
  TMAP --> INF[r27_infer.sh<br/>run_fivebench.py --tau_map]
  TCON --> INF
  INF --> FRAMES[r27_depth_tau/r27_&lt;arm&gt;_vp/]
  FRAMES --> EV[r27_eval.sh<br/>evaluate.py, 9 metrics]
  EV --> AVG[csv/edit?_FiVE_r27_&lt;arm&gt;_vp_*.csv]
  REF[csv/edit?_FiVE_r21_ref_vp_*.csv] --> SUM[r27_summarize.py]
  AVG --> SUM
  DCSV --> SUM
  SUM --> OUT[csv/r27_depth_tau.csv]
  OUT --> FIG[r27_figures.py]
  R25[csv/r25_tau_map.csv] --> FIG
  FIG --> F1[figures/r27_tau_depth.pdf]
  FIG --> F2[figures/r27_delta.pdf]
  FIG --> F3[figures/r27_grids/]
```

## Code to touch

| File | Status | Change |
|---|---|---|
| `evaluation/r27_depth_delta.py` | **written, verified 22/22** | images → anchor grid → DAv2 → background affine alignment → `\|ΔD\|/IQR` → erode(L, 3) → zero-padded top-0.5% mean. Flags: `--erode_radius`, `--magnitude`, `--top_q`, `--min_effect`, `--mad_k`, `--grid`, `--fold_mad` |
| `evaluation/r27_tau_map.py` | new | `D_norm` → `m` → budget-linear `tau`; `--assign {depth,constant}`; emits R25's CSV schema |
| `slurm_scripts/five_bench/r27_infer.sh` | new | Array 0-5, `ARM=${R27_ARM:-depth}`, L40S, `--exclude=node52`, 22-pair coverage preflight |
| `slurm_scripts/five_bench/r27_eval.sh` | new | Single task, `L40S,A100`, explicit 9-metric `--metrics` list |
| `evaluation/r27_summarize.py` | new | Two arms vs re-averaged Eq. 4 reference; both contrasts + per-case sign tests |
| `evaluation/r27_figures.py` | new | `r27_tau_depth.pdf`, `r27_delta.pdf`, `r27_grids/` |
| `evaluation/run_fivebench.py` | **unchanged** | `--tau_map` (R25) already carries a per-pair `blend_power`; R27 needs no pipeline change |
| `Self-Forcing_StreamEdit/**` | **unchanged** | `blend_power` at `causal_model.py:334` is the only exponent knob and is already reachable |

Budget-linear inversion, for reference (`--step 15`, `t_next = [0.933, 0.866, …, 0.066, 0]`):

```python
A     = lambda tau: sum(t**tau for t in T)          # source-anchoring budget
A_m   = (1 - m) * A(tau_min) + m * A(tau_max)       # linear in m
tau   = brentq(lambda x: A(x) - A_m, 0.0, 400.0)    # invert the monotone A
# check: A(2)=4.5062, A(50)=0.0320; m=0.6 -> A_m=1.8217 -> tau=5.53
```
