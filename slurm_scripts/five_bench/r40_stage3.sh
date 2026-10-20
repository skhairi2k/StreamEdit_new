#!/bin/bash
# R40 stage 3 -- the SPATIALLY-GATED render for the two draft-free combos, on the whole
# 419-pair FiVE-Bench. Copy of r35_stage3.sh: same render config, same preflight, same
# guards -- only the combos and the roots change, so R40 vs R35 differs in the rho field
# alone (anchor map + flow warp vs per-frame draft map).
#
# ARRAY LAYOUT: one COMBO per task:
#   0 dino_warp   1 lpips_warp
#
# Does NOT build its own rho field: the `absnorm` step (local, r31_div_abs.py +
# r31_rho_map.py) already built r40_div_abs/{combo} and r40_rho/{combo}. Preflight checks
# BOTH trees are complete for this combo before the model loads.
#
# ⚠️ QUEUE ONLY AFTER R38/R39 RENDERS have left the queue (plan Decisions).
#
# --step 15 IS PINNED: r31_rho_map.py inverted the source budget on the 15-step t_next
# grid; a different --step would silently invalidate every exponent in the field.
# --dump_steps 14 (final only), as R35: nothing downstream reads an intermediate step.
# --time=20:00:00, --mem=64G, --exclude: as r35_stage3.sh.
#SBATCH --job-name=r40_stage3
# L40S ONLY (2026-09-24): same diffusion model and clip lengths as stage 1, whose first
# submission (job 1007492) hit CUDA OOM on a 40 GB A100 at 39.2/39.5 GiB.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --exclude=node52
#SBATCH --output=logs/r40_stage3_%A_%a.out
#SBATCH --error=logs/r40_stage3_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DIV_ABS_ROOT=~/Data/dataggen/outputs/five_bench/r40_div_abs
RHO_ROOT=~/Data/dataggen/outputs/five_bench/r40_rho
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r40_arms
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).

COMBOS=(dino_warp lpips_warp)
STEP=15
FLOW_SHIFT=1.0

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 1 ]; then
  echo "[r40_stage3] bad array id ${TID} (expected 0-1)"; exit 1
fi
COMBO=${COMBOS[$TID]}
METHOD="r40_${COMBO}"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

# Conda: streamgve -- this stage RENDERS (load_pipe / rollout_inference), unlike
# the absnorm step, which is CPU-only numpy and needs no GPU env at all.
cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r40_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} method=${METHOD}"
echo "[r40_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: BOTH r40_div_abs and r40_rho must be complete for this combo BEFORE
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
  echo "[r40_stage3] ${COMBO} ${LABEL}: ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r40_stage3] FATAL: ${COMBO} ${LABEL} incomplete under ${ROOT}/${COMBO}"
    echo "[r40_stage3]   -- run /run-step R40 absnorm first"
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi

# ---- render all 6 edit types, model loaded once per combo. r31_stage3.py's own
# preflight re-validates every rho npz (shape, frame count, schedule) before the model
# loads, so a bad field is still caught before the ~20-minute load, not just here.
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r40_stage3] ${METHOD} edit${T} ---"
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
    || { echo "[r40_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown of step14 dirs, plus manifest status.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r40_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- ${STATUS}"
done
echo "[r40_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

N_BAD=$(cat "$OUT_ROOT/$METHOD"/_manifest_edit*.csv 2>/dev/null \
       | awk -F, 'FNR==1{next} $3!="ok"' | wc -l)
echo "[r40_stage3] ${METHOD}: non-ok manifest rows: ${N_BAD}"
if [ "$N_BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r40_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
