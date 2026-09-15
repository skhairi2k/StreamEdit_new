---
name: R20 — Blend-Rate Schedule Sweep
overview: Replace the StreamGVE Eq. 4 blender rate with three cosine source-anchor schedules and score them under no-VP, §4.5 VP, and persistent-VP injection on the 22 clips of evaluation/cases.json, extending the shared run_fivebench.py runner so every arm shares one code path with R1/R7.
task_id: R20
todos:
  - id: plumb-sched
    content: "Add `blend_sched` end-to-end: `_schedule_blend_rate(sched, step_idx, num_steps)` helper in pipeline/utils.py; pipeline writes `shared_dict_dual['blender_rate']` per denoising step; causal_model.py:329 reads it when present. Default None ⇒ Eq. 4 evaluated unchanged, bit-for-bit."
    status: completed
  - id: plumb-vp-rollout
    content: "Forward `blend_off` / `vp_latent` / `vp_head_gate` / `blend_sched` through the rollout_inference window loop, relax the `rollout_chunk_size >= 0` guard (edit_causal_inference.py:96), and narrow the vp_latent-vs-trg_initial_latent guard (:249) so it fires on the §4.5 anchor path but not on rollout overlap seeding."
    status: completed
  - id: extend-run-fivebench
    content: "Extend the SHARED evaluation/run_fivebench.py (not a new r20 driver) with `--vp_mode {novp,vp,pvp}`, `--blend_sched`, and a `--cases_json` subset filter. `--vp_mode vp` + `--first_frame_edit_dir` must reproduce R7 exactly; defaults must reproduce R1 exactly. This is what puts all 12 R20 arms and both existing references on one code path."
    status: completed
  - id: write-slurm
    content: "slurm_scripts/five_bench/r20_smoke.sh (R1/R7 parity + 3-mode render), r20_infer.sh (array 0-3 over arm groups), r20_eval.sh (16-metric evaluate.py over 9 arms + pvp_paper + the two external references, all --cases_json restricted)."
    status: completed
  - id: write-summarize
    content: "evaluation/r20_summarize.py — join the arm CSVs against the 3 paper-schedule reference CSVs into evaluation/csv/r20_blend_sched.csv (metrics + per-metric deltas vs the matching VP mode's reference; parser validated against r10_all_avg.csv). Also prints a per-VP-mode verdict of whether any cosine schedule beats Eq.4."
    status: completed
  - id: smoke
    content: "Run r20_smoke.sh: gate 1 = index-0 pairs reproduce baseline/r7 bit-for-bit; gate 2 = edit6 subset vs full render identically (subset independence); stage 3 = cos_full renders on all 3 VP modes."
    status: completed
  - id: launch-vp
    content: "Phase 1a: 3 schedules × {vp, pvp} × 22 clips (132 runs), plus the pvp_paper reference arm (22 runs) in the same array submission."
    status: completed
  - id: launch-novp
    content: "Phase 1b: 3 schedules × novp × 22 clips = 66 runs."
    status: completed
  - id: launch-phase2
    content: "Phase 2: degenerate anchors `const` (pure source every step) and `zero` (no blending) x 3 VP modes x 22 clips = 132 runs. Bracket the cosine schedules -- no schedule should score outside [const, zero] on preservation."
    status: completed
  - id: evaluate
    content: "16-metric FiVE evaluation of the 9 arms + pvp_paper + baseline + r7_visual_prompting, every one restricted to the same 22 case_ids."
    status: completed
  - id: summarize
    content: "Build the comparison CSV and record the verdict. VERDICT (corrected 2026-07-23 after user flagged the visuals vs metrics mismatch): the blend schedule is a SINGLE preservation↔editability knob; arms order MONOTONICALLY by anchor-hold time: const → cos_full → Eq.4 → cos_half → cos_third → zero (preserve→edit). e.g. pvp struct↓ 9.0/12.4/16.6/17.1/21.6/46.5. On the RELIABLE edit2 bulk (16 clips), zero is the strongest editor in all 3 VP modes (five_acc_union 0.84/0.94/0.94), cos_third/cos_half beat Eq.4 on edit success at modest preservation cost, cos_full/const under-edit. Preservation (psnr/ssim/lpips/mse) ranks inversely and is unaffected. No Pareto win over Eq.4. pvp caveat: all cosines hurt motion_fidelity vs Eq.4. DATA BUG (root-caused): five_acc/motion/niqe in r20_blend_sched.csv are unreliable due to RUNTIME metric crashes, NOT a code defect vs the paper. evaluate.py + metrics_calculator.py are byte-identical to upstream FiVE-Bench for all metric logic (calculate_metric, get_score, MFS, structure_distance, calculate_mean, scaling, the except handler); local deltas are only StreamEdit I/O flags + 3 compat fixes (transformers-5 CLIP shim, niqe utf-8/sys.executable). The upstream `except Exception: print; continue` appends NO placeholder, so when a metric RAISES the row loses a column and everything after it shifts, dropping the trailing five_acc composite. In our run motion_fidelity_score hit CUDA OOM 144x (CoTracker on 44GB) and niqe missing-save-file 140x -> shifted rows on affected clips. RELIABLE metrics (computed before the crash point, always present): preservation (struct/psnr/ssim/lpips/mse) + clipT + clip_tgt_edit. The frontier verdict rests on those and holds. FIX for paper reproduction = NO code change; fix the environment: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (+bigger GPU) for motion, valid IQA_PyTorch_model_path for niqe, then re-run and confirm zero 'Error:' lines. Do NOT patch the except handler (would deviate from the paper's evaluator). Separately, the overall is a mean-of-edit-type-means (edit1's 1 clip = edit2's 16) -> distorting on the imbalanced 22-clip subset; report per-edit-type/clip-weighted for R20."
    status: completed
  - id: grid-figures
    content: "evaluation/r20_grid_figure.py — per-(video x VP mode) qualitative grid, 7 rows (Source + Eq.4 + cos_full/half/third + const/zero) x 5 time-sampled frames. 66 PNGs under figures/r20_grids/{novp,vp,pvp}/. Source row aligned by normalized time (VAE changes frame count). Eq.4 row maps to baseline/r7/paper_pvp per VP mode."
    status: completed
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r20_smoke.sh
    status: completed
    completed_at: 2026-07-22
    job_id: "907447"

  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "tail -60 logs/r20_smoke_{job_id}.out; grep -c '^IDENTICAL' logs/r20_smoke_{job_id}.out  # expect 4 (2 index-0 pairs x 2 arms); grep -c '^SUBSET-INDEPENDENT' # expect 2; then '[ok]' lines for cos_full on novp/vp/pvp"
    status: completed
    completed_at: 2026-07-22

  - id: launch-vp
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=1,2,3 slurm_scripts/five_bench/r20_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-07-22
    job_id: "907467"

  - id: wait-vp
    type: manual
    wait_for: launch-vp
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; ls -d /projects/dataggen/outputs/five_bench/r20_blend_sched/*_vp/*/*/ | wc -l  # 66; ...*_pvp/*/*/ | wc -l  # 88 incl. paper_pvp; ...paper_pvp/*/*/ | wc -l  # 22"
    status: completed
    completed_at: 2026-07-23

  - id: launch-novp
    type: sbatch
    wait_for: wait-vp
    command: sbatch --array=0 slurm_scripts/five_bench/r20_infer.sh
    status: completed
    completed_at: 2026-07-22
    job_id: "907470"

  - id: wait-novp
    type: manual
    wait_for: launch-novp
    check_hint: "(launched in parallel with launch-vp, not after it) sacct -j {job_id} --format=JobID,State,Elapsed; ls -d /projects/dataggen/outputs/five_bench/r20_blend_sched/*_novp/*/*/ | wc -l  # expect 66; grep -c '^\\[ERROR\\]' logs/r20_infer_*.out"
    sets_status: finished
    status: completed
    completed_at: 2026-07-23

  - id: launch-phase2
    type: sbatch
    wait_for: wait-novp
    command: sbatch --array=4,5,6 slurm_scripts/five_bench/r20_infer.sh
    status: completed
    completed_at: 2026-07-23
    job_id: "907504"

  - id: wait-phase2
    type: manual
    wait_for: launch-phase2
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; ls -d /projects/dataggen/outputs/five_bench/r20_blend_sched/{const,zero}_*/*/*/ | wc -l  # expect 132; grep -c '^\\[ERROR\\]' logs/r20_infer_*.out"
    status: completed
    completed_at: 2026-07-23

  - id: evaluate
    type: sbatch
    wait_for: wait-phase2
    command: sbatch slurm_scripts/five_bench/r20_eval.sh
    status: completed
    completed_at: 2026-07-23
    job_id: "907650"

  - id: summarize
    type: local
    wait_for: evaluate
    command: |
      python evaluation/r20_summarize.py \
        --arm_csv_glob 'evaluation/csv/r20_*_avg.csv' \
        --ref_novp evaluation/csv/r20_ref_novp_avg.csv \
        --ref_vp evaluation/csv/r20_ref_vp_avg.csv \
        --ref_pvp evaluation/csv/r20_paper_pvp_avg.csv \
        -o evaluation/csv/r20_blend_sched.csv
    output_paths:
      - evaluation/csv/r20_blend_sched.csv
    sets_status: analyzed
    status: completed
    completed_at: 2026-07-23
isProject: true
---

# R20: Blend-Rate Schedule Sweep

## Context

StreamGVE's blender rate (Eq. 4) is implemented at [causal_model.py:329](../../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L329) as `blender_rate = 1 - t_{i+1}**rho`, where `blender_rate=1` is pure target and `blender_rate=0` is pure source. It rises 0.13 → 1.00 across the 15 steps, i.e. the source anchor decays on a fixed convex profile tied to the noise level. R20 asks whether that profile is the right one: three cosine schedules indexed by denoising step replace it, spanning a longer hold (`cos_full`) and two early releases (`cos_half`, `cos_third`).

Each schedule runs under all three visual-prompting regimes — none, the paper's §4.5 cached first frame, and R10's persistent re-roped anchor bank — because the blend schedule and the anchor mechanism both control the same quantity (how hard the target branch is pulled toward a reference), so a schedule that helps without an anchor may hurt with one.

**Done when:** `evaluation/csv/r20_blend_sched.csv` exists, holding the 16 FiVE metrics for all 9 arms plus the 3 paper-schedule references, over the same 22 clips, with per-metric deltas against the reference of the matching VP mode.

## Execution steps

| # | Step id | Type | What |
|---|---------|------|------|
| — | *(prep)* | — | `plumb-sched`, `plumb-vp-rollout`, `extend-run-fivebench`, `write-slurm`, `write-summarize` — see **Code to touch** |
| 1 | `smoke` | sbatch | Gate 1 parity (index-0 pairs), gate 2 subset independence, then `cos_full` on all 3 VP modes |
| 2 | `wait-smoke` | manual | 4 `IDENTICAL` + 2 `SUBSET-INDEPENDENT`; all 3 VP modes render |
| 3 | `launch-vp` | sbatch | Array 1,2,3 → `vp` (66), `pvp` (66), `paper_pvp` reference (22) |
| 4 | `wait-vp` | manual | 154 frame dirs present |
| 5 | `launch-novp` | sbatch | Array 0 → `novp`, 3 scheds × 22 clips (66 runs) |
| 6 | `wait-novp` | manual | 66 frame dirs present → `finished` |
| 7 | `launch-phase2` | sbatch | Array 4,5,6 → `const`/`zero` × 3 VP modes (132 runs) — degenerate anchors |
| 8 | `wait-phase2` | manual | 132 frame dirs present |
| 9 | `evaluate` | sbatch | 16-metric `evaluate.py` × 10 R20 dirs + 2 external references, `--cases_json` restricted (22 clips) |
| 10 | `summarize` | local | Join into the comparison CSV → `analyzed` |

```
/run-step R20 smoke
/run-step R20 wait-smoke
/run-step R20 launch-vp
/run-step R20 wait-vp
/run-step R20 launch-novp
/run-step R20 wait-novp
/run-step R20 launch-phase2
/run-step R20 wait-phase2
/run-step R20 evaluate
/run-step R20 summarize
```

## Decisions

| Question | Choice |
|---|---|
| Scheduler sign convention | `_schedule_blend_rate()` returns the **source-anchoring weight** `s(p)`; the pipeline sets `blender_rate = 1 - s(p)`. Keeps the paper's direction (anchor at high noise, release at low). |
| Step coordinate | `p = index / (total_timestep - 1)` from the existing `current_timestep_index` / `total_timestep`; `p` resets per block, exactly as the paper's `t_{i+1}` dependence does. |
| Schedules run | `cos_full`, `cos_half`, `cos_third` = `0.5*(1 + cos(pi * min(1, k*p)))`, k=1/2/3. `paper` selectable, run only for the `pvp` reference. |
| Dropped | `lin_half`, `lin_third`, `lin_quarter`, `cos_quarter` — not run at all. |
| Phase 2 | `const` (s≡1 ⇒ pure source every step, the "no edit" floor) and `zero` (s≡0 ⇒ no blending, ≈ R10 `blend_off`) × 3 VP modes. They **bracket** the cosine arms: a schedule scoring outside [const, zero] on preservation indicates a plumbing fault, not a good schedule. `zero` also cross-checks R10, which used a different driver. |
| VP modes | `novp` (no anchor) · `vp` (§4.5, `independent_first_frame` + `trg_initial_latent`) · `pvp` (persistent `vp_latent` bank, all 360 heads, no gating) |
| Driver | **Extend the shared `evaluation/run_fivebench.py`** rather than fork an `r20_*` driver, so all 12 arms and both external references run one code path. New flags: `--vp_mode`, `--blend_sched`, `--cases_json`. |
| **Per-pair seeding** | `run_fivebench.py` now re-seeds before every pair (default, no flag). Previously seeded once per process, so a pair's noise depended on its index in the edit{T} json and a `--cases_json` subset produced different videos than a full run (`0042_gym-ball`: 93% of pixels). This is what makes R20's 22-clip subset comparable to a full-bench reference at all. |
| **R1/R7 re-render (blocking `evaluate`)** | The stored `baseline` / `r7_visual_prompting` predated per-pair seeding, so they were a different generation. **Re-run complete (verified 2026-07-23):** both dirs re-rendered with all frame mtimes 2026-07-22 21:59 → 2026-07-23 02:38 (postdate the seeding change), and all 22 `cases.json` clips present in each. `evaluate` is now unblocked. |
| Paper-schedule references | `novp` → `five_bench/baseline` (R1) · `vp` → `five_bench/r7_visual_prompting` (R7) — **both verified present for all 22 clips**. `pvp` → rendered here as the `paper_pvp` arm (no existing run is config-matched). |
| Sampler config | `rollout_chunk_size=21`, `rollout_overlap_block_num=1`, no truncation, `step=15`, `fg_boost_factor=4`, `blend_power=2`, `flow_shift=1.0`, `seed=0`, `sink_size=0` — the `run_fivebench.py` defaults R1/R7 used. |
| Clip set | All **22** `case_id`s in the shared `evaluation/cases.json` — no R20-specific subset file (`0011_lucia` appears under edit2 and edit5 ⇒ output keyed `{arm}/edit{T}/{video}`). Spans edit types 1/2/5/6 at 1/16/3/2. |
| Windowing on this clip set | At `rollout_chunk_size=21`: `novp`/`pvp` are single-window on 21/22 clips, `vp` splits on 8/22. **`0034_cows` (24 latent frames) is the sole clip that splits under `pvp`** — the only real test of the anchor bank surviving a window boundary. It is also the clip that crashed R9 on the single-window path, so `-1` is not a valid simplification. |
| Default behaviour | `blend_sched=None`, `vp_mode=novp` ⇒ Eq. 4 unchanged, bit-for-bit (verified by the `smoke` parity check against both R1 and R7) |
| Persistent VP across windows | Bank rebuilt and re-stamped in every rollout window so the anchor never fades. Exercised by `0034_cows` only — watch that clip specifically when reading the `pvp` arms. |
| Evaluation | Standard 16-metric `evaluation/fivebench/evaluate.py` (no `--per_frame`), `--tgt_layout edit_video`, `--cases_json evaluation/cases.json` on every arm *and* both references (all 22 clips) |
| Out of scope | Full FiVE-Bench promotion, head-gated `pvp`, per-frame fading curves, linear schedules |

## Step commands

### smoke

```bash
sbatch slurm_scripts/five_bench/r20_smoke.sh
```

### wait-smoke

```bash
tail -40 logs/r20_smoke_{job_id}.out
grep -c '^IDENTICAL' logs/r20_smoke_{job_id}.out          # expect 4 (index-0 pairs)
grep -c '^SUBSET-INDEPENDENT' logs/r20_smoke_{job_id}.out # expect 2
grep '^\[ok\]' logs/r20_smoke_{job_id}.out                # cos_full on novp/vp/pvp
```

### launch-vp

```bash
sbatch --array=1,2,3 slurm_scripts/five_bench/r20_infer.sh
```

### wait-vp

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
OUT=/projects/dataggen/outputs/five_bench/r20_blend_sched
ls -d $OUT/*_vp/*/*/ | wc -l        # 66
ls -d $OUT/*_pvp/*/*/ | wc -l       # 88  (paper_pvp also ends in _pvp)
ls -d $OUT/paper_pvp/*/*/ | wc -l   # 22
grep -c '^\[ERROR\]' logs/r20_infer_*.out
```

### launch-novp

```bash
sbatch --array=0 slurm_scripts/five_bench/r20_infer.sh
```

### wait-novp

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
ls -d /projects/dataggen/outputs/five_bench/r20_blend_sched/*_novp/*/*/ | wc -l  # 66
```

### evaluate

```bash
sbatch slurm_scripts/five_bench/r20_eval.sh
```

### summarize

```bash
python evaluation/r20_summarize.py \
  --arm_csv_glob 'evaluation/csv/r20_*_avg.csv' \
  --ref_novp evaluation/csv/r20_ref_novp_avg.csv \
  --ref_vp evaluation/csv/r20_ref_vp_avg.csv \
  --ref_pvp evaluation/csv/r20_paper_pvp_avg.csv \
  -o evaluation/csv/r20_blend_sched.csv
```

## Pipeline

```mermaid
flowchart TD
    A[FiVE-Bench videos] --> D
    B[evaluation/cases.json<br/>22 clips] --> D
    C[dataggen/anchors/editT/*.png] --> D
    D[evaluation/run_fivebench.py<br/>--vp_mode x --blend_sched<br/>--cases_json] --> E[r20_blend_sched/<br/>{sched}_{vpmode}/editT/video/]
    D --> P[r20_blend_sched/<br/>paper_pvp/editT/video/]
    E --> F[fivebench/evaluate.py<br/>--cases_json --tgt_layout edit_video]
    P --> F
    R1[five_bench/baseline<br/>R1, paper+novp] --> F
    R7[five_bench/r7_visual_prompting<br/>R7, paper+vp] --> F
    F --> G[csv/r20_*_avg.csv<br/>csv/r20_ref_*_avg.csv]
    G --> H[evaluation/r20_summarize.py]
    H --> I[csv/r20_blend_sched.csv]
```

## Code to touch

| File | Change |
|---|---|
| `Self-Forcing_StreamEdit/pipeline/utils.py` | **New** module-level `_schedule_blend_rate(sched, step_idx, num_steps) -> float` returning the source-anchoring weight. |
| `Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py` | Add `blend_sched: Optional[str] = None` to `rollout_inference` and `inference`; in the denoising loop (~:480) set `shared_dict_dual['blender_rate']`. |
| `Self-Forcing_StreamEdit/pipeline/edit_causal_inference.py` | Forward `blend_off` / `vp_latent` / `vp_head_gate` / `blend_sched` into the rollout window loop (:151); relax the `rollout_chunk_size >= 0` guard (:96); narrow the `vp_latent` vs `trg_initial_latent` guard (:249). |
| `Self-Forcing_StreamEdit/wan/modules/causal_model.py` | At :329, read `shared_dict['blender_rate']` when present, else evaluate Eq. 4 unchanged. |
| `evaluation/run_fivebench.py` | `set_seed(args.seed)` before every pair (behaviour change, default on) + docstring warning that pre-2026-07-22 outputs are a different generation. |
| `evaluation/run_fivebench.py` | Add `--vp_mode {novp,vp,pvp}` (default `vp`, so `--first_frame_edit_dir` alone still means R7), `--blend_sched` (default `None`), `--cases_json` subset filter, and encode the anchor to `vp_latent` in `pvp` mode. Shared file — every addition defaults to today's behaviour. |
| `slurm_scripts/five_bench/r20_smoke.sh` | **New** — gate 1: bit-parity vs R1/R7 on edit types 1 and 6, asserted only on index-0 pairs while the references predate per-pair seeding. Gate 2: edit6 subset (2 pairs) vs full (10 pairs) must be bit-identical — the subset-independence invariant. Stage 3: `cos_full` on all 3 VP modes. |
| `slurm_scripts/five_bench/r20_infer.sh` | **New** — `#SBATCH --array=0-6`, L40S, `--mem=64G`, 6h. Phase 1: 0=`novp`×3 scheds, 1=`vp`×3, 2=`pvp`×3, 3=`pvp`×`paper` (reference). Phase 2: 4=`novp`×{const,zero}, 5=`vp`×{const,zero}, 6=`pvp`×{const,zero}. Loops the 4 edit types the 22 clips span (1, 2, 5, 6) with `--cases_json`. |
| `slurm_scripts/five_bench/r20_eval.sh` | **New** — `evaluate.py` over the 10 R20 arm dirs plus `baseline` and `r7_visual_prompting`, all `--cases_json evaluation/cases.json`. |
| `evaluation/r20_summarize.py` | **New** — join arm + reference CSVs, emit per-metric deltas within each VP mode. |

Schedule helper:

```python
def _schedule_blend_rate(sched: str, step_idx: int, num_steps: int) -> float:
    """Source-anchoring weight s(p) in [0, 1]; blender_rate = 1 - s(p)."""
    p = 0.0 if num_steps <= 1 else step_idx / (num_steps - 1)
    if sched == "cos_full":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if sched in ("cos_half", "cos_third"):
        k = {"cos_half": 2, "cos_third": 3}[sched]
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, k * p)))
    if sched == "const":
        return 1.0
    if sched == "zero":
        return 0.0
    raise ValueError(f"unknown blend schedule: {sched!r}")
```

Pipeline hook, alongside the existing `shared_dict_dual` writes:

```python
shared_dict_dual['blender_rate'] = (
    None if blend_sched in (None, "paper")
    else 1.0 - _schedule_blend_rate(blend_sched, index, len(denoising_step_list))
)
```

Attention hook, replacing the `else` branch at `causal_model.py:329`:

```python
elif shared_dict.get('blender_rate') is not None:
    blender_rate = shared_dict['blender_rate']
else:
    blender_rate = 1 - shared_dict['current_timestep_next'] ** shared_dict['blend_power']
```

`run_fivebench.py` anchor branch, replacing the `first_frame_edit_dir is not None` block:

```python
# vp  -> paper §4.5: anchor enters kv_cache_trg as an initial latent (fades)
# pvp -> R10: anchor enters a private re-roped K/V bank (never fades)
if args.vp_mode == "vp":
    independent_first_frame = True
    src_first_frame_latent, trg_first_frame_latent = _encode_pair(...)
    vp_latent = None
elif args.vp_mode == "pvp":
    independent_first_frame = False
    src_first_frame_latent = trg_first_frame_latent = None
    vp_latent = _encode_anchor(...)
else:                                   # novp -- R1 baseline path, unchanged
    independent_first_frame = False
    src_first_frame_latent = trg_first_frame_latent = vp_latent = None
```

Implementation notes:

- **`run_fivebench.py` is shared with R1/R7.** Every new flag must default to today's behaviour, and the `smoke` step is the gate: 3 clips under defaults must be bit-identical to `five_bench/baseline`, and 3 under `--vp_mode vp` bit-identical to `five_bench/r7_visual_prompting`. If either differs, stop — the references are invalidated, not the arms.
- The prev-key blend at `causal_model.py:366` writes **in place into a view of `kv_cache["k"]`**, so source keys accumulate in the target cache across denoising steps within a block. A schedule reaching `s=0` early (`cos_half`, `cos_third`) does not undo what earlier steps injected — it stops adding. Expect less separation than the `s(p)` curves suggest; report it, don't fix it.
- The `:249` guard exists to stop the anchor being cached as an initial latent (the fading §4.5 path). On the rollout path `trg_initial_latent` is *also* used for overlap seeding (`:195`), which is unrelated — gate the check on `independent_first_frame` / first window only, or `pvp` will raise on window 2 once this runs on longer clips.
- `--cases_json` must reach `evaluate.py` for the reference dirs too: R1 covers 419 pairs and R7 443, and averaging those against 22-clip arms would be an unfair comparison, not a baseline.
- Arm directory names are `{sched}_{vp_mode}`, so the reference arm is `paper_pvp`. `ls *_pvp` matches it as well as the three cosine `pvp` arms — count with `paper_pvp` excluded explicitly when verifying 60 vs 20.
