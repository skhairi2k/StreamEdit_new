#!/bin/bash
# R41 stage 3 -- the BINARY-rho render, whole 419-pair FiVE-Bench, one ARM (combo x theta)
# per array task, at the draft length N set in r41_config.sh.
#
# Submit with the array size from the config (no #SBATCH --array here, so the ARMS list and
# the array can never disagree):
#   sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) \
#          slurm_scripts/five_bench/r41_stage3.sh
# Task i renders ARMS[i]: theta-major, theta 0.5 first, combos in COMBOS order.
#
# r35_stage3.sh VERBATIM otherwise: same r31_stage3.py flags (also identical to
# r38_stage3.sh), so a binary arm differs from its continuous counterpart ONLY by the rho
# field. Stage 3 stays at --step 15 whatever N is -- N is the DRAFT length; the binary rho
# npz records step=15 and r31_stage3.py refuses a field built for another schedule.
#
# --dump_steps 14 (final only): nothing downstream reads an intermediate step.
#SBATCH --job-name=r41_stage3
# L40S ONLY: same model and clip lengths as R35 stage 1, whose job 1007492 OOMed on a 40 GB A100.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r41_stage3_%A_%a.out
#SBATCH --error=logs/r41_stage3_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors
EDIT_TYPES=(1 2 3 4 5 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO" || exit 1
# exits here if DRAFT_N is unset; R41_CONFIG is for synthetic tests only (see r41_config.sh)
source "${R41_CONFIG:-slurm_scripts/five_bench/r41_config.sh}"

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "${#ARMS[@]}" ]; then
  echo "[r41_stage3] bad array id '${TID}' (expected 0-$(( ${#ARMS[@]} - 1 )))"; exit 1
fi
ARM=${ARMS[$TID]}
METHOD="r41_${TAG}_${ARM}"

mkdir -p logs "$ARMS_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r41_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} (combo=${ARM_COMBO[$ARM]}, theta=${ARM_THETA[$ARM]}) method=${METHOD}"
echo "[r41_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: this arm's binary rho tree complete BEFORE the model loads.
N_TOTAL=0
BAD=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls "$RHO_ROOT/$ARM/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
done
echo "[r41_stage3] ${ARM} rho: ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
  echo "[r41_stage3] FATAL: ${ARM} rho incomplete under ${RHO_ROOT}/${ARM} -- run /run-step R41 rhomap first"
  exit 1
fi

# ---- render all 6 edit types. r31_stage3.py re-validates every rho npz (shape, frame
# count, schedule) before its model load.
FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r41_stage3] ${METHOD} edit${T} ---"
  python evaluation/r31_stage3.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --rho_dir "$RHO_ROOT/$ARM" \
    --data_root "$DATA_ROOT" \
    --out_root "$ARMS_ROOT" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 14 \
    --step 15 \
    --flow_shift 1.0 \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r41_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown of step14 dirs, plus manifest status.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls -d "$ARMS_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r41_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- ${STATUS}"
done
echo "[r41_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

# Files passed to awk directly (not via cat): FNR resets per file, so each of the 6 headers
# is skipped. Through `cat` FNR == NR and 5 headers would count as non-ok rows.
N_MANIFESTS=$(ls "$ARMS_ROOT/$METHOD"/_manifest_edit*.csv 2>/dev/null | wc -l)
N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "$ARMS_ROOT/$METHOD"/_manifest_edit*.csv 2>/dev/null | wc -l)
echo "[r41_stage3] ${METHOD}: ${N_MANIFESTS} manifests (expect 6)"
if [ "$N_MANIFESTS" -ne 6 ]; then FAILED=$((FAILED+1)); fi
echo "[r41_stage3] ${METHOD}: non-ok manifest rows: ${N_BAD}"
if [ "$N_BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r41_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
