#!/bin/bash
# R27 evaluate -- 9 FiVE metrics over one arm's 22 clips (same explicit --metrics
# list as R25/R26; five_acc and motion_fidelity excluded).
#
# SINGLE TASK per arm. --cases_json restricts to the 22 pairs; file_ids stay aligned
# with the stored r21_ref_vp CSVs for r27_summarize.py to join on.
#
#SBATCH --job-name=r27_eval
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r27_eval_%j.out
#SBATCH --error=logs/r27_eval_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

# sbatch --export=R27_ARM=depth    slurm_scripts/five_bench/r27_eval.sh
# sbatch --export=R27_ARM=constant slurm_scripts/five_bench/r27_eval.sh

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r27_depth_tau
CASES="$REPO/evaluation/cases.json"

ARM=${R27_ARM:-depth}
case "$ARM" in
  depth|constant) ;;
  *) echo "[r27_eval] bad R27_ARM='$ARM' (expected depth|constant)"; exit 1 ;;
esac
METHOD="r27_${ARM}_vp"
METHOD_DIR="$OUT_ROOT/$METHOD"
STEM="$METHOD"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

echo "[r27_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -3
python -c "import torch; print('[r27_eval] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -1

echo "[r27_eval] job=${SLURM_JOB_ID} arm=$ARM stem=$STEM dir=$METHOD_DIR"
echo "[r27_eval] cases_json=$CASES annotations=${#ANNOTATIONS[@]}"
echo "[r27_eval] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if [ ! -d "$METHOD_DIR" ]; then
  echo "[r27_eval] FAILED -- method dir does not exist: $METHOD_DIR"
  echo "[r27_eval] run the infer step first"
  exit 1
fi

NDIRS=$(ls -d "$METHOD_DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r27_eval] real frame dirs: $NDIRS  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r27_eval] FAILED -- expected 22 real pairs, found $NDIRS"
  exit 1
fi

METRICS_LOG="logs/r27_eval_${SLURM_JOB_ID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part \
            mse_unedit_part ssim_unedit_part clip_similarity_source_image \
            clip_similarity_target_image clip_similarity_target_image_edit_part \
            niqe_target_image \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$METHOD_DIR" \
  --tgt_layout edit_video \
  --cases_json "$CASES" \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r27_eval] $STEM exit=$RC error_lines=$NERR oom_lines=$NOOM"

NAVG=$(ls evaluation/csv/edit?_FiVE_${STEM}_frame_stride8_avg.csv 2>/dev/null | wc -l)
echo "[r27_eval] per-edit-type _avg.csv written: $NAVG  # expect 6"

if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ] || [ "$NAVG" -ne 6 ]; then
  echo "[r27_eval] FAILED -- see $METRICS_LOG"
  exit 1
fi

echo "[r27_eval] OK"
exit 0
