#!/bin/bash
# R41 eval -- the 9-metric FiVE-Bench table for the binary arms, one ARM per array task,
# scored on each arm's FINAL stage-3 render (step14) over the WHOLE 419-pair bench.
#
# Submit with the array size from the config:
#   sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) \
#          slurm_scripts/five_bench/r41_eval.sh
#
# r35_eval.sh verbatim otherwise -- same metric list, resources and guards -- so the binary
# arms land on exactly the columns the continuous arms (R35 at N=15, R38 at N=5) were scored
# on. The continuous arms are NOT re-scored.
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE: config.yaml's `metrics:` key is vestigial,
# and evaluate.py's default includes five_acc and motion_fidelity, which exhaust an L40S.
# five_acc and CLIP-D are scored by r41_fiveacc.sh / r41_clipd.sh.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, and a raised metric DROPS
# its column. The log is teed and grepped (any `Error:` line fails the task), and the
# per-edit-type CSV row counts are checked.
#
# NOTE for the score step: {stem}_avg.csv is a mean of the six per-type means, not over the
# 419 clips. r41_score.py reads the per-edit-type CSVs (plain per-clip mean).
#SBATCH --job-name=r41_eval
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r41_eval_%A_%a.out
#SBATCH --error=logs/r41_eval_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
# exits here if DRAFT_N is unset; R41_CONFIG is for synthetic tests only (see r41_config.sh)
source "${R41_CONFIG:-slurm_scripts/five_bench/r41_config.sh}"

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "${#ARMS[@]}" ]; then
  echo "[r41_eval] bad array id '${TID}' (expected 0-$(( ${#ARMS[@]} - 1 )))"; exit 1
fi
ARM=${ARMS[$TID]}
STEM="r41_${TAG}_${ARM}"
DIR="$ARMS_ROOT/${STEM}/step14"

mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r41_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r41_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} stem=${STEM}"
echo "[r41_eval] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the render must be complete (real pair dirs, not *_resize siblings).
if [ ! -d "$DIR" ]; then
  echo "[r41_eval] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r41_eval] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r41_eval] FAILED ${STEM} -- render incomplete, run /run-step R41 wait-stage3 first"
  exit 1
fi
echo "[r41_eval] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r41_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part \
            mse_unedit_part ssim_unedit_part clip_similarity_source_image \
            clip_similarity_target_image clip_similarity_target_image_edit_part \
            niqe_target_image \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" \
  --tgt_layout edit_video \
  --tmp_resize \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r41_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r41_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  FAILED=1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r41_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi
for T in 1 2 3 4 5 6; do
  CSV="evaluation/csv/edit${T}_FiVE_${STEM}_frame_stride8.csv"
  N=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  STATUS="ok"
  [ "$N" -ne "${EXPECT[$T]}" ] && { STATUS="MISMATCH (expect ${EXPECT[$T]})"; FAILED=1; }
  echo "[r41_eval] ${STEM} edit${T}: ${N} scored rows -- ${STATUS}"
done

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r41_eval] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
