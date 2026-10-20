#!/bin/bash
# R45 stage 1 -- LongLive DRAFTS (the divergence-measurement targets) on the WHOLE 419-pair
# FiVE-Bench, for the stage-1 RUNS the method-gate selection needs.
#
#   sbatch --array=$(python evaluation/r45_select.py stage1) slurm_scripts/five_bench/r45_stage1.sh
#
# ARRAY LAYOUT: one RUN (= draft N x regime) per task, the same order as r45_select.py RUNS;
# the model loads once per task, then loops all 6 edit types (as r38_stage1.sh).
#   task 0  ->  n15_unblended  (--blend_sched zero,   N=15)
#   task 1  ->  n15_tgt09      (--blend_sched tgt0.9, N=15, anchored iff input t > 0.9)
#   task 2  ->  n5_unblended   (--blend_sched zero,   N=5)
#   task 3  ->  n5_tgt09       (--blend_sched tgt0.9, N=5)
# A task whose run no selected combo uses refuses to start (r45_select.py check).
#
# MEASURED LATENT: --dump_steps N-1 --measure_index N-1, the final x0 (R35/R38 convention).
# Backbone LongLive-native: --backbone ll, tripled VP frame, fg_boost 2; seed 0,
# flow_shift 1.0, blend_power 2 (inert under a named schedule). Otherwise r38_stage1.sh.
#
# --time=20:00:00: R35/R38's budget kept; an N=15 LL draft costs about one LL render --
# re-check against smoke G4 and this run's wall clock.
#SBATCH --job-name=r45_stage1
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r45_stage1_%A_%a.out
#SBATCH --error=logs/r45_stage1_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors

RUNS=(n15_unblended n15_tgt09 n5_unblended n5_tgt09)
SCHEDS=(zero tgt0.9 zero tgt0.9)
NSTEPS=(15 15 5 5)

TID=${SLURM_ARRAY_TASK_ID}
MAX_TID=$(( ${#RUNS[@]} - 1 ))
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt "$MAX_TID" ]; then
  echo "[r45_stage1] bad array id ${TID} (expected 0-${MAX_TID})"; exit 1
fi
RUN=${RUNS[$TID]}
SCHED=${SCHEDS[$TID]}
STEPS=${NSTEPS[$TID]}
METHOD="$RUN"
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r45_stage1           # -> {run}/step{NN}/edit{T}/{video}
LATENT_DIR=~/Data/dataggen/outputs/five_bench/r45_latents/$RUN  # -> edit{T}/{video}.npz
DUMP_STEPS=$((STEPS - 1))
MEASURE_INDEX=$((STEPS - 1))

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve
cd "$REPO"
python evaluation/r45_select.py check "$TID" stage1 || exit 1
mkdir -p logs "$OUT_ROOT" "$LATENT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r45_stage1] job=${SLURM_ARRAY_JOB_ID}_${TID} run=${RUN} blend_sched=${SCHED} steps=${STEPS} backbone=ll"
echo "[r45_stage1] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r45_stage1] frames -> ${OUT_ROOT}/${METHOD}   latents -> ${LATENT_DIR}/edit{T}"
nvidia-smi -L 2>&1 | head -2

# r31_stage1.py echoes the per-step blender_rate on the pipeline's real warped grid:
# unblended [1,...], tgt0.9 [0,0,1,...] at N=15 / [0,1,1,1,1] at N=5. Checked at wait-stage1.
FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r45_stage1] ${RUN} / edit${T} ---"
  python evaluation/r31_stage1.py --backbone ll \
    --edit_type "$T" \
    --method "$METHOD" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --latent_dir "$LATENT_DIR" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps "$DUMP_STEPS" \
    --measure_index "$MEASURE_INDEX" \
    --step "$STEPS" \
    --flow_shift 1.0 \
    --fg_boost_factor 2 \
    --blend_power 2 \
    --blend_sched "$SCHED" \
    --seed 0 \
    || { echo "[r45_stage1] FAILED ${RUN} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: 419 latents for THIS run, with the exact per-type breakdown.
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
N_TOTAL=0
BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r45_stage1] ${RUN} edit${T}: ${N} latents -- ${STATUS}"
done
echo "[r45_stage1] ${RUN}: ${N_TOTAL} latents total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi
echo "[r45_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
