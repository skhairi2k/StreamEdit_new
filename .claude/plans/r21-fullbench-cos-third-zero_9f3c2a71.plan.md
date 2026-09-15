---
name: R21 — Full-Bench cos_third & zero (VP + Persistent VP)
overview: Promote R20's two most promising blend-rate arms, cos_third and zero, to the whole FiVE-Bench (all 6 edit types, ~419 pairs) under the §4.5 VP and R10 persistent-VP regimes, reusing the shared run_fivebench.py plumbing built in R20; render a full-bench paper_pvp Eq.4 reference so pvp deltas are apples-to-apples, and score with the metric-crash environment fixes so all 16 metrics are reliable.
task_id: R21
todos:
  - id: write-r21-infer
    content: "slurm_scripts/five_bench/r21_infer.sh — array 0-4, ONE arm per task (each loops all 6 edit types internally; the 30-task arm×edit-type layout was rejected by QOSMaxSubmitJobPerUserLimit). ARMS = (cos_third vp)(cos_third pvp)(zero vp)(zero pvp)(paper pvp); EDIT_TYPES=(1 2 3 4 5 6); arm=TASK_ID. Model loads once per task. Reuses run_fivebench.py with NO --cases_json (full bench). L40S, --mem=64G, --time=20h (~10h/arm). All arms pass --vp_mode + --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors; paper arm passes --blend_sched paper. New OUT_ROOT r21_blend_full so it never mixes with R20's 22-clip dirs of the same name."
    status: completed
  - id: write-r21-eval
    content: "slurm_scripts/five_bench/r21_eval.sh — 16-metric evaluate.py over the 5 arm dirs (cos_third_vp, cos_third_pvp, zero_vp, zero_pvp, paper_pvp) + r7_visual_prompting (vp reference). NO --cases_json (full bench). Annotations = all 6 edit{T}_FiVE.json. METRIC-CRASH FIX: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, request H100 (both the motion_fidelity OOMs and the niqe failures are one root cause — GPU exhaustion on the 44GB L40S). No code change to evaluate.py/metrics_calculator.py. csvs → r21_{arm}_avg.csv, r21_ref_vp_avg.csv. BUILT AS --array=0-5 (one method per task): serial would need ~31h at full-bench scale vs a 24h partition ceiling."
    status: completed
  - id: write-r21-novp-infer
    content: "slurm_scripts/five_bench/r21_novp_infer.sh — ONE arm: cos_third x novp, full bench, all 6 edit types. Added 2026-08-20: R21 originally scoped novp OUT, so cos_third_novp is the one cell missing from the arm grid — the only full-bench cos_third_novp render does not exist (R20 has one, but on its 22-clip subset, not comparable to 419 pairs). Same invocation as r21_infer.sh with --blend_sched cos_third --vp_mode novp --method cos_third_novp, NO --first_frame_edit_dir (novp ignores the anchor), out_root r21_blend_full. L40S, --mem=64G, --time=20h (~10h for 419 pairs)."
    status: completed
  - id: write-r21-summarize
    status_note: "BUILT 2026-08-20 -- emits BOTH reference configurations (jpg = official/comparable-to-published-baselines, mp4 = reproduces StreamGVE Table 1); long format with per-arm deltas against the matching VP-mode reference."
    content: "evaluation/r21_summarize.py — adaptation of r20_summarize.py: join the 4 arm CSVs + paper_pvp against the vp reference (r21_ref_vp_avg.csv, R7) and pvp reference (r21_paper_pvp_avg.csv), emit per-metric deltas within each VP mode into evaluation/csv/r21_blend_full.csv. Single overall mean (evaluate.py's mean-of-edit-type-means over the 6 types); no per-type breakdown. NOTE (2026-08-11): the eval now also produces r21_ref_novp_avg.csv (the un-anchored R1 Eq.4 floor, added as eval task 6) — carry it as a reference ROW so the table has a no-anchoring anchor, and make sure --arm_csv_glob 'r21_*_avg.csv' does not mistake either r21_ref_* CSV for an experiment arm."
    status: completed
steps:
  - id: infer
    type: sbatch
    command: sbatch --array=0-4 slurm_scripts/five_bench/r21_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-07-23
    job_id: "908240"
  - id: wait-infer
    type: manual
    wait_for: infer
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; for a in cos_third_vp cos_third_pvp zero_vp zero_pvp paper_pvp; do echo $a $(ls -d /projects/dataggen/outputs/five_bench/r21_blend_full/$a/*/*/ 2>/dev/null | wc -l); done  # each ~419; grep -c 'FAILED' logs/r21_infer_*.out  # expect 0"
    sets_status: finished
    status: completed
    completed_at: 2026-08-11
  - id: evaluate
    type: sbatch
    wait_for: wait-infer
    command: sbatch slurm_scripts/five_bench/r21_eval.sh
    status: completed
    completed_at: 2026-08-11
    job_id: "937950"
  - id: wait-eval
    type: manual
    wait_for: evaluate
    check_hint: "ls evaluation/csv/r21_*_avg.csv | wc -l  # expect 7. NOTE the '0 Error: lines' gate was WRONG as written -- what must be 0 are the ENVIRONMENTAL classes: grep -c 'out of memory' and grep -c 'Error: niqe' over logs/r21_eval_*.metrics.log. 'Error: motion_fidelity_score_edit_part' is the known empty-mask degeneracy (R2/R14), is written as nan by the evaluate.py:487 patch, and is EXPECTED (4/arm, same clip 0010_giant-slalom in edit1-4, identical across arms). Real check is CSV integrity: every r21_*_avg.csv 17 cols; per-type intermediates 100/100/100/100/9/10 rows; 0 misaligned widths; no nan in any overall mean."
    status: completed
    completed_at: 2026-08-14
  - id: infer-novp
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r21_novp_infer.sh
    status: completed
    completed_at: 2026-08-20
    job_id: "953404"
  - id: wait-infer-novp
    type: manual
    wait_for: infer-novp
    check_hint: "ls -d /projects/dataggen/outputs/five_bench/r21_blend_full/cos_third_novp/*/*/ | grep -vc '_resize/$'  # expect 419 (100/100/100/100/9/10); grep 'failures:' logs/r21_novp_infer_*.out  # expect 0"
    status: pending
  - id: eval-novp
    type: sbatch
    wait_for: wait-infer-novp
    command: sbatch slurm_scripts/five_bench/r21_novp_eval.sh
    status: pending
  - id: wait-eval-novp
    type: manual
    wait_for: eval-novp
    check_hint: "ls evaluation/csv/r21bg_mp4_cos_third_novp_avg.csv evaluation/csv/r21_cos_third_novp_avg.csv; grep -c 'out of memory' logs/r21_novp_eval_*.metrics.log  # MUST be 0. Score against BOTH references like job 951202/937950 so the row is comparable to the other arms."
    status: pending
  - id: summarize
    type: local
    wait_for: [wait-eval, wait-eval-novp]
    command: |
      python evaluation/r21_summarize.py \
        --arm_csv_glob 'evaluation/csv/r21_*_avg.csv' \
        --ref_vp evaluation/csv/r21_ref_vp_avg.csv \
        --ref_pvp evaluation/csv/r21_paper_pvp_avg.csv \
        -o evaluation/csv/r21_blend_full.csv
    output_paths:
      - evaluation/csv/r21_blend_full.csv
    sets_status: analyzed
    status: completed
    completed_at: 2026-08-20
    note: "Run over the 7 arms that exist. cos_third_novp (steps infer-novp..wait-eval-novp) is still unrendered; re-run this step to fold it in once it lands."
isProject: true
---

# R21: Full-Bench cos_third & zero (VP + Persistent VP)

## Context

R20 swept blend-rate schedules on a 22-clip subset and found the arms order monotonically on a single preservation↔editability knob, with `cos_third` (strong editor, modest preservation cost over Eq.4) and `zero` (strongest editor) the two most promising. R21 promotes exactly those two to the **whole FiVE-Bench** (all 6 edit types, ~419 pairs) under the two anchoring regimes the user cares about: **`vp`** (paper §4.5 cached first frame, R7's `--first_frame_edit_dir`) and **`pvp`** (R10's never-evicted re-roped persistent anchor bank). No `novp`.

All plumbing already exists in the shared `evaluation/run_fivebench.py` (R20's `--vp_mode`, `--blend_sched`, per-pair seeding); R21 is a scale-up, not new pipeline code — it drops `--cases_json` and adds two Slurm scripts + a summarizer. Two things it must get right at scale: a full-bench `paper_pvp` reference (none exists — R20 only rendered it on 22 clips) so pvp deltas are apples-to-apples, and the R20 metric-crash environment fix so `motion_fidelity`/`niqe`/`five_acc` are reliable rather than silently column-shifted.

**Done when:** `evaluation/csv/r21_blend_full.csv` exists, holding the 16 FiVE metrics for the 4 arms (`cos_third`/`zero` × `vp`/`pvp`) plus the `paper_pvp` reference, over the full FiVE-Bench, with per-metric deltas against the matching VP-mode Eq.4 reference (R7 for `vp`, `paper_pvp` for `pvp`), and the eval log verified crash-free (0 `Error:` lines).

## Execution steps

| # | Step id | Type | What |
|---|---------|------|------|
| — | *(prep)* | — | `write-r21-infer`, `write-r21-eval`, `write-r21-summarize` — see **Code to touch** |
| 1 | `infer` | sbatch | Array 0-29 = 5 arms × 6 edit types, full bench, no `--cases_json` → `running` |
| 2 | `wait-infer` | manual | Each of the 5 arm dirs ~419 frame dirs; 0 `FAILED` → `finished` |
| 3 | `evaluate` | sbatch | 16-metric `evaluate.py` × 5 arms + R7 ref, A100/H100 + env fix, full bench |
| 4 | `wait-eval` | manual | 6 `_avg.csv`; **0 `Error:` lines** (metric-crash fix held) |
| 5 | `infer-novp` | sbatch | **(added 2026-08-20)** `cos_third` × `novp`, full bench — the missing cell in the arm grid |
| 6 | `wait-infer-novp` | manual | 419 real frame dirs; 0 failures |
| 7 | `eval-novp` | sbatch | Score it on **both** references (mp4 + jpg) so the row is comparable to the others |
| 8 | `wait-eval-novp` | manual | Both `_avg.csv` present; 0 OOM |
| 9 | `summarize` | local | Join into `r21_blend_full.csv` with vp/pvp deltas → `analyzed` |

```
/run-step R21 infer
/run-step R21 wait-infer
/run-step R21 evaluate
/run-step R21 wait-eval
/run-step R21 infer-novp        # added 2026-08-20
/run-step R21 wait-infer-novp
/run-step R21 eval-novp
/run-step R21 wait-eval-novp
/run-step R21 summarize
```

## Decisions

| Question | Choice |
|---|---|
| Arms | `cos_third` and `zero` only (R20's two most promising), each under `vp` and `pvp` → 4 experiment arms. **Amended 2026-08-20: `cos_third` × `novp` added** (steps `infer-novp` … `wait-eval-novp`). The original "no `novp`" scoping left a hole in the arm grid — `paper` has all three VP modes covered (novp = R1 `baseline`, vp = R7, pvp = `paper_pvp`) but `cos_third` had only `vp`/`pvp`, so the schedule effect could not be read at `novp` where the two references are strongest. R20 has a `cos_third_novp` but only on its 22-clip subset, which is not comparable to these 419-pair rows. Still no `zero`×`novp`, no `cos_full`/`cos_half`/`const`. |
| pvp Eq.4 reference | **Render a full-bench `paper_pvp` arm** (`--blend_sched paper`, `--vp_mode pvp`) — none exists (R20's is 22-clip). This is the 5th render arm; gives pvp deltas an apples-to-apples baseline. |
| vp Eq.4 reference | Existing `five_bench/r7_visual_prompting` (R7) — verified full-bench + re-rendered under per-pair seeding (mtimes 2026-07-23). Scored as `r21_ref_vp`. |
| Clip set | **Whole FiVE-Bench**, all 6 edit types — no `--cases_json` on inference or eval. **419 pairs: edit1-6 = 100/100/100/100/9/10.** Confirmed three ways (2026-08-11): the six `edit{T}_FiVE.json` annotation files, all five rendered arm dirs, and the R7 reference. *(An earlier draft of this row said 101/116/100/100/12/12 = 441 — wrong. 441 is what a bare `ls` reports on `r7_visual_prompting`, which holds 419 real pairs plus 22 `{video}_resize` dirs that R20's eval wrote for its 22-clip subset.)* |
| Compute layout | SLURM array over **arm × edit-type** = 30 tasks (`--array=0-29`); model loads once per (arm, edit-type) task so all 6 types parallelize per arm. L40S, `--mem=64G` (partition default host-OOMs on UMT5-XXL load), 6h. |
| Array mapping | `ARMS=("cos_third vp" "cos_third pvp" "zero vp" "zero pvp" "paper pvp")`, `EDIT_TYPES=(1 2 3 4 5 6)`; `arm=TASK_ID/6`, `type=TASK_ID%6`. |
| Output root | **New** `/projects/dataggen/outputs/five_bench/r21_blend_full/{sched}_{vp_mode}/edit{T}/{video}/` — separate from R20's `r20_blend_sched/` so full-bench arms never mix with R20's identically-named 22-clip arms. |
| Anchors | `/projects/dataggen/outputs/five_bench/anchors/edit{T}/*.png` (R7's, all 6 types, 419 present) — same files R7/R10/R20 read, so arms differ only by injection mechanism. |
| Column-shift fix (**new 2026-08-12**) | The "no code change" rule held only while the problem was environmental. Job 936242 showed it is not: `motion_fidelity_score_edit_part` **raises** on the empty-edit-mask degeneracy (R2 outcome; R14 owns the real fix) — ~1 clip per 53, so ~8 over the full bench — and upstream's `except: continue` appends nothing, dropping that cell and shifting every later column LEFT so `five_acc` inherits motion_fidelity's value. Patched `evaluate.py:487` to append the **arity-aware placeholder upstream already uses** for its own no-images case at line 378 (`["nan"]*5` for five_acc, else `"nan"`). Behaviour is unchanged on any row where no metric raised, so no already-correct number moves. **Not retroactive:** R20's CSVs were computed with the shifting bug and remain corrupt on that column — do not mix R20 and R21 `five_acc`. |
| Metric reliability | **Fix the environment, not the code.** `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + eval on **H100 (80GB)**. `evaluate.py`/`metrics_calculator.py` stay byte-identical to upstream. `wait-eval` asserts 0 `Error:` lines. **Root cause revised 2026-08-11:** R20's two crash classes (motion_fidelity 144×, niqe 140×) are *one* cause — GPU exhaustion in the parent (44.35/44.39 GiB). Errors are absent from edit1 (scored first, memory free) and concentrated in edit2/5/6 once Qwen2.5-VL + CoTracker are resident. niqe is that exhaustion one level down: `calculate_NIQE` shells out to `inference_iqa.py`, the child cannot init CUDA, writes no txt, and `average_niqe_from_txt` raises `FileNotFoundError`. **`IQA_PyTorch_model_path` was never wrong** — the exact failing command reproduces clean standalone on a free GPU (niqe 3.044), so no config change. **A100 rejected as fallback**: this cluster does not expose GPU memory via `scontrol`, and a 40GB A100 would be worse than the L40S that already failed; H100 is the confirmed-larger card (R7's anchor job moved there for the same reason). |
| Verdict granularity | **Single overall mean** (`evaluate.py`'s mean-of-edit-type-means over the 6 types). No per-edit-type / clip-weighted breakdown. |
| Summarizer | New `evaluation/r21_summarize.py` — r20_summarize.py restricted to `{vp, pvp}` (r20's requires a novp reference file; R21 has no novp arm). |
| Sampler config | `--step 15 --fg_boost_factor 4 --blend_power 2 --seed 0`, `rollout_chunk_size=21` (run_fivebench.py defaults R1/R7/R20 used), per-pair reseed on by default. |
| Out of scope | `novp` arm, `cos_full`/`cos_half`/`const` at full scale, head-gated `pvp`, per-frame fading curves, qualitative grids. |

## Step commands

### infer

```bash
sbatch --array=0-29 slurm_scripts/five_bench/r21_infer.sh
```

### wait-infer

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
OUT=/projects/dataggen/outputs/five_bench/r21_blend_full
for a in cos_third_vp cos_third_pvp zero_vp zero_pvp paper_pvp; do
  echo "$a $(ls -d $OUT/$a/*/*/ 2>/dev/null | wc -l)"   # each ~419
done
grep -c 'FAILED' logs/r21_infer_*.out                    # expect 0
```

### evaluate

```bash
sbatch slurm_scripts/five_bench/r21_eval.sh
```

### wait-eval

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
ls evaluation/csv/r21_*_avg.csv | wc -l                  # expect 6 (5 arms + ref_vp)
grep -c 'Error:' logs/r21_eval_*.out logs/r21_eval_*.err # MUST be 0 (metric-crash fix)
```

### summarize

```bash
python evaluation/r21_summarize.py \
  --arm_csv_glob 'evaluation/csv/r21_*_avg.csv' \
  --ref_vp evaluation/csv/r21_ref_vp_avg.csv \
  --ref_pvp evaluation/csv/r21_paper_pvp_avg.csv \
  -o evaluation/csv/r21_blend_full.csv
```

## Pipeline

```mermaid
flowchart TD
    A[FiVE-Bench videos<br/>all 6 edit types ~419] --> D
    C[dataggen/anchors/editT/*.png] --> D
    D[run_fivebench.py<br/>cos_third,zero,paper x vp,pvp<br/>no --cases_json] --> E[r21_blend_full/<br/>{sched}_{vpmode}/editT/video/]
    D --> P[r21_blend_full/<br/>paper_pvp/editT/video/]
    E --> F[fivebench/evaluate.py<br/>A100/H100 + alloc-conf + IQA path<br/>full bench, all 6 anno]
    P --> F
    R7[five_bench/r7_visual_prompting<br/>R7, vp Eq.4 reference] --> F
    F --> G[csv/r21_*_avg.csv<br/>csv/r21_ref_vp_avg.csv]
    G --> H[evaluation/r21_summarize.py]
    H --> I[csv/r21_blend_full.csv]
```

## Code to touch

| File | Change |
|---|---|
| `slurm_scripts/five_bench/r21_infer.sh` | **New** — `#SBATCH --array=0-29`, L40S, `--mem=64G`, 6h. `ARMS=("cos_third vp" "cos_third pvp" "zero vp" "zero pvp" "paper pvp")`, `EDIT_TYPES=(1 2 3 4 5 6)`; task→(arm,type) via `/6` and `%6`. `run_fivebench.py --edit_type $T --method {sched}_{vpmode} --blend_sched $SCHED --data_root … --out_root /projects/dataggen/outputs/five_bench/r21_blend_full --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0`, **no `--cases_json`**. vp/pvp add `--vp_mode $VP --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors`. Same conda dance as r20 (`conda deactivate` before `activate streamgve`; no `set -u`). |
| `slurm_scripts/five_bench/r21_eval.sh` | **Built 2026-08-11** — H100, `--mem=64G`, **`--array=0-5`, `--time=20:00:00`**. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before `evaluate.py`. `five-bench` env (deactivate-then-activate). Task→method: `cos_third_{vp,pvp}`, `zero_{vp,pvp}`, `paper_pvp` → `r21_{arm}`; task 5 = `$REF_ROOT/r7_visual_prompting` → `r21_ref_vp`. Annotations = **all six** `edit{T}_FiVE.json`. **No `--cases_json`** (full bench). Each task tees its own output and exits non-zero on any `Error:` line or a missing `_avg.csv` — upstream's `except: continue` means exit code 0 alone does not prove the CSV is clean. **Array, not serial:** R20 measured ~45 s/clip, so 6 methods × 419 clips ≈ 31 h in series vs a 24 h partition ceiling; per-method scoring is independent (own result CSV, own niqe txt, own `_resize` dirs), so the array is numerically identical to the serial form at ~5 h/task. |
| `slurm_scripts/five_bench/r21_novp_infer.sh` | **New (2026-08-20)** — single arm `cos_third` × `novp`, full bench, all 6 edit types. `run_fivebench.py --blend_sched cos_third --vp_mode novp --method cos_third_novp`, **no** `--first_frame_edit_dir` (novp ignores the anchor; `run_fivebench.py` also forces `vp_mode=novp` when no anchor dir is given), `--out_root .../r21_blend_full`, no `--cases_json`. L40S, `--mem=64G`, `--time=20h`. |
| `slurm_scripts/five_bench/r21_novp_eval.sh` | **New (2026-08-20)** — score `cos_third_novp` on **both** source references so its row is comparable: mp4 (`mp4_src_root`, paper-comparable preservation) and jpg (benchmark default, target-only metrics). Reuse `r21_eval.sh`'s H100 + `expandable_segments` setup and the corrected gate (fail on OOM/niqe/unrecognised only — `motion_fidelity_score_edit_part`'s empty-mask raise is expected). → `r21bg_mp4_cos_third_novp_avg.csv`, `r21_cos_third_novp_avg.csv`. |
| `evaluation/r21_summarize.py` | **New** — copy of `r20_summarize.py` with `VP_MODES=("vp","pvp")`, `--ref_vp`/`--ref_pvp` only (drop `--ref_novp` and its existence check / `paper_novp` row). Emits `r21_blend_full.csv` with per-metric value + `Δref` within each VP mode. |

Implementation notes:

- **No pipeline code changes.** R20 already added `--vp_mode`, `--blend_sched`, per-pair seeding, and the rollout-loop plumbing to `run_fivebench.py` / `edit_causal_inference.py` / `causal_model.py`. R21 exercises the same paths at full scale with `--cases_json` omitted — the R1/R7 default. Confirm the runner tolerates a missing `--cases_json` (R1/R7 ran that way) as the first thing the infer job prints.
- **Arm-dir collision guard.** `cos_third_pvp` / `zero_pvp` / `paper_pvp` also exist under R20's `r20_blend_sched/` (22 clips). The distinct `--out_root` (`r21_blend_full`) is what keeps the generations separate — do not point R21 at `r20_blend_sched`.
- **`paper_pvp` is the only render whose purpose is a reference**, not an experiment arm. It is the pvp Eq.4 baseline; without it, `pvp` `cos_third`/`zero` have nothing config-matched to be read against (R10b's all-heads arm went through a different driver, and R20's paper_pvp is 22-clip).
- **The metric-crash fix is the point of the eval step.** In R20, `motion_fidelity` CUDA-OOM'd 144× and `niqe` failed its save-file 140×; the upstream `except: continue` appends no placeholder, so a raised metric drops a column and shifts every column after it, silently corrupting `five_acc`. At full-bench scale this must not recur — hence bigger GPU + `expandable_segments` + a real IQA path, and `wait-eval` treats any `Error:` line as a stop condition, not a warning.
- **Windowing at `rollout_chunk_size=21`.** Most clips are single-window; clips >21 latent frames (e.g. `0034_cows`) split, and are the only real test of the persistent bank being rebuilt across a window boundary under `pvp`. Watch those specifically when reading the `pvp` arms, as in R20.
