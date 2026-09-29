#!/bin/bash
# R35 stage 3 -- the SPATIALLY-GATED render, on the whole 419-pair FiVE-Bench, for the
# four (arm, regime) COMBOS.
#
# ARRAY LAYOUT: one COMBO per task, in the user's stated SLURM priority order (same order
# as r35_stage2.sh -- do not reorder one without the other):
#   0 dino_unblended   1 dino_first2   2 lpips_unblended   3 lpips_first2
#
# UNLIKE r31_stage3.sh, this task does NOT build its own rho field -- r35_absnorm.sh
# (this plan's `absnorm` step, a `wait_for` dependency of this one) already built BOTH
# r35_div_abs/{combo} and r35_rho/{combo} for all four combos, CPU-only, before any GPU
# job here starts. Rebuilding rho per-task would be redundant (cheap, but pointless) and
# would duplicate the one place the tau_min/tau_max/step/flow_shift constants live.
# Preflight here therefore checks BOTH trees are already complete for this combo.
#
# --step 15 IS PINNED, matching r35_absnorm.sh's rho build and r31_stage3.sh's own
# reasoning: r31_rho_map.py inverted the source budget on the 15-step t_next grid, and a
# different --step here would silently invalidate every exponent in the field.
#
# --dump_steps 14 (FINAL ONLY, user-decided 2026-09-24) -- deliberately NOT R31's `all`.
# Measured: R31's --dump_steps all costs ~12 GB per arm at 22 clips; extrapolated to
# R35's 419 clips x 4 combos that is ~914 GB, against ~74 GB for final-only. Nothing in
# R35's plan (grids, eval, fiveacc, clipd, the Pareto figures) ever reads an intermediate
# step -- only step14. Same reasoning as the plan's existing --no-save_native decision:
# skip persisting what nothing downstream consumes.
#
# --time=20:00:00 per the plan's Decisions (full-bench render, largest single cost in
# this task alongside stage1).
# --mem=64G matches every other five_bench render script (the partition default host-OOMs
# loading UMT5-XXL + the checkpoint, the R9 job-900404 failure).
# --exclude: the proven post-2026-09-22 node set (R33/R34), kept defensively.
#SBATCH --job-name=r35_stage3
# L40S ONLY (2026-09-24): same diffusion model and clip lengths as stage 1, whose first
# submission (job 1007492) hit CUDA OOM on a 40 GB A100 at 39.2/39.5 GiB.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r35_stage3_%A_%a.out
#SBATCH --error=logs/r35_stage3_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DIV_ABS_ROOT=~/Data/dataggen/outputs/five_bench/r35_div_abs
RHO_ROOT=~/Data/dataggen/outputs/five_bench/r35_rho
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r35_arms
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).

COMBOS=(dino_unblended dino_first2 lpips_unblended lpips_first2)
STEP=15
FLOW_SHIFT=1.0

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r35_stage3] bad array id ${TID} (expected 0-3)"; exit 1
fi
COMBO=${COMBOS[$TID]}
METHOD="r35_${COMBO}"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

# Conda: streamgve -- this stage RENDERS (load_pipe / rollout_inference), unlike
# r35_absnorm.sh, which is CPU-only numpy and needs no GPU env at all.
cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r35_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} method=${METHOD}"
echo "[r35_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: BOTH r35_div_abs and r35_rho must be complete for this combo BEFORE
# the model loads. Failing here costs seconds; failing after a ~20-minute model load
# costs the rest of the allocation.
FAILED=0
for ROOT_NAME in "div_abs:$DIV_ABS_ROOT" "rho:$RHO_ROOT"; do
  LABEL=${ROOT_NAME%%:*}; ROOT=${ROOT_NAME#*:}
  N_TOTAL=0
  BAD=0
  for T in "${EDIT_TYPES[@]}"; do
    N=$(ls "$ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + N))
    [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r35_stage3] ${COMBO} ${LABEL}: ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r35_stage3] FATAL: ${COMBO} ${LABEL} incomplete under ${ROOT}/${COMBO}"
    echo "[r35_stage3]   -- run /run-step R35 absnorm first"
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi

# ---- render all 6 edit types, model loaded once per combo. r31_stage3.py's own
# preflight re-validates every rho npz (shape, frame count, schedule) before the model
# loads, so a bad field is still caught before the ~20-minute load, not just here.
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r35_stage3] ${METHOD} edit${T} ---"
  python evaluation/r31_stage3.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --rho_dir "$RHO_ROOT/$COMBO" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 14 \
    --step "$STEP" \
    --flow_shift "$FLOW_SHIFT" \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r35_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown of step14 dirs, plus manifest status.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r35_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- ${STATUS}"
done
echo "[r35_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

N_BAD=$(cat "$OUT_ROOT/$METHOD"/_manifest_edit*.csv 2>/dev/null \
       | awk -F, 'FNR==1{next} $3!="ok"' | wc -l)
echo "[r35_stage3] ${METHOD}: non-ok manifest rows: ${N_BAD}"
if [ "$N_BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r35_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
