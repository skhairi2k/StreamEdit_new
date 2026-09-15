#!/bin/bash
# R22 evaluation -- per-frame scoring of the 15 step-videos per arm, at stride 1.
#
# ARRAY LAYOUT (one arm per task; 15 evaluate.py invocations inside each):
#   0  paper_novp
#   1  paper_vp
#   2  cos_third_pvp
#
# WHY ONLY 8 METRICS. The matrix cell M(i,j) is the metric on the first i frames of
# the step-j video. Every metric below is reduced by evaluate.py as a MEAN over
# per-frame values (calculate_mean, evaluate.py:484), so the prefix-truncated value
# is exactly the prefix mean of those same values -- i.e. M(i,j) is a cumulative
# mean, and the whole nb_frames x 15 matrix costs 15 evaluations per clip instead of
# nb_frames x 15. That identity is what makes R22 affordable, and it holds ONLY for
# these 8. Do not add a metric here without re-deriving it:
#   * motion_fidelity_score{,_edit_part} compare CoTracker TRACKLET SETS over the
#     whole clip -- no frame axis (--per_frame already emits them at frame_idx=-1).
#   * five_acc is a single VLM verdict per clip -- likewise.
#   * niqe_target_image looks decomposable but is not recoverable: calculate_NIQE
#     returns the literal string "nan" per frame (metrics_calculator.py:807) and the
#     real values go only to a txt that is deleted per video.
# Dropping the first three also removes R21's OOM pressure (Qwen2.5-VL + CoTracker
# were what exhausted the 44GB L40S), which is why this runs on L40S, not H100.
#
# WHY --frame_stride 1. --per_frame records frame_idx = f_i * frame_stride
# (evaluate.py:475). At the default stride 8 the matrix would have ~9 rows instead of
# nb_frames, so the row axis would not be per-pixel-frame at all. Stride 1 is the
# whole point, and it is also why this is ~7.5x the image-metric work of R20's eval
# -- hence --time=20:00:00 rather than 6h.
#
# ⚠️ DO NOT compare the resulting M(nb_frames,14) against r20_*_avg.csv or
# r21_*_avg.csv. Those ran at --frame_stride 8 (so their per-clip value averages every
# 8th frame) AND their overall row is evaluate.py's mean-of-edit-type-means, which on
# this 22-clip subset weighs edit1's single clip as much as edit2's sixteen. The
# legitimate bottom-right check is internal (r22_build_matrix.py) plus the pixel-level
# parity already established at wait-dump: step14 is byte-identical to the stored
# reference render for all 22 clips x 3 arms (4446/4446 frames).
#
# The 22 clips span edit types 1, 2, 5, 6, so only those four annotation files are
# passed; --cases_json restricts to the 22 within them.
#
# Uses the five-bench env, NOT streamgve: .bashrc auto-activates streamgve and
# `conda activate five-bench` alone does not pop it (the R2 job-880402 failure,
# ModuleNotFoundError: torchmetrics). Hence the explicit deactivate.
#SBATCH --job-name=r22_eval
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/r22_eval_%A_%a.out
#SBATCH --error=logs/r22_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DUMP_ROOT=/projects/dataggen/outputs/five_bench/r22_step_dump
CASES=$REPO/evaluation/cases.json
CSV_DIR=$REPO/evaluation/csv/r22_raw

STEPS=15

case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=paper_novp ;;
  1) ARM=paper_vp ;;
  2) ARM=cos_third_pvp ;;
  *) echo "[r22_eval] bad array id ${SLURM_ARRAY_TASK_ID} (expected 0-2)"; exit 1 ;;
esac

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench

cd "$REPO"
mkdir -p logs "$CSV_DIR"

# Free GPU memory is not the bottleneck here (no Qwen2.5-VL / CoTracker), but the
# allocator setting is free insurance and matches R21's environment.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

METRICS=(
  structure_distance
  psnr_unedit_part
  lpips_unedit_part
  mse_unedit_part
  ssim_unedit_part
  clip_similarity_source_image
  clip_similarity_target_image
  clip_similarity_target_image_edit_part
)

METRICS_LOG="logs/r22_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.metrics.log"
: > "$METRICS_LOG"

echo "[r22_eval] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} arm=${ARM} steps=${STEPS}"
echo "[r22_eval] metrics=${#METRICS[@]} stride=1 -> ${CSV_DIR}"

FAILED=0
for ((J=0; J<STEPS; J++)); do
  JJ=$(printf "%02d" "$J")
  STEP_ROOT="$DUMP_ROOT/$ARM/step$JJ"
  STEM="r22_${ARM}_s${JJ}"
  # evaluate.py appends _frame_stride{N} to the stem, then derives the per-frame file
  # from that -- so this is the file r22_build_matrix.py will glob.
  PF_CSV="$CSV_DIR/${STEM}_frame_stride1_per_frame.csv"

  echo "[r22_eval] === ${ARM} step ${JJ} ===" | tee -a "$METRICS_LOG"

  if [ ! -d "$STEP_ROOT" ]; then
    echo "[r22_eval] FAILED ${STEM} -- missing dump dir $STEP_ROOT" | tee -a "$METRICS_LOG"
    FAILED=$((FAILED+1)); continue
  fi

  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --src_image_folder "$DATA_ROOT" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --metrics "${METRICS[@]}" \
    --tgt_methods "$STEP_ROOT" \
    --tgt_layout edit_video \
    --tgt_key "$STEM" \
    --cases_json "$CASES" \
    --frame_stride 1 \
    --per_frame \
    --result_path "$CSV_DIR/${STEM}.csv" 2>&1 | tee -a "$METRICS_LOG"
  RC=${PIPESTATUS[0]}

  if [ "$RC" -ne 0 ]; then
    echo "[r22_eval] FAILED ${STEM} -- evaluate.py exited ${RC}" | tee -a "$METRICS_LOG"
    FAILED=$((FAILED+1))
  elif [ ! -s "$PF_CSV" ]; then
    echo "[r22_eval] FAILED ${STEM} -- per-frame CSV missing/empty: $PF_CSV" | tee -a "$METRICS_LOG"
    FAILED=$((FAILED+1))
  else
    echo "[r22_eval] OK ${STEM} -- $(wc -l < "$PF_CSV") rows" | tee -a "$METRICS_LOG"
  fi

  # Drop the {video}_resize copies evaluate.py writes beside the scored frames. At
  # stride 1 it copies EVERY frame, so leaving them would roughly double the ~36 GB
  # dump. They are pure derived data, regenerated on any re-score.
  rm -rf "$STEP_ROOT"/edit*/*_resize
done

# Upstream's `except` no longer shifts columns (evaluate.py:487 was patched during R21
# to append an arity-aware placeholder), but a raised metric still leaves that metric's
# rows OUT of the per-frame CSV -- which would punch holes in the matrix rather than
# corrupt it. So any `Error:` line is still a stop condition, not a warning.
N_ERR=$(grep -c 'Error:' "$METRICS_LOG" 2>/dev/null)
N_CSV=$(ls "$CSV_DIR"/r22_${ARM}_s??_frame_stride1_per_frame.csv 2>/dev/null | wc -l)

echo "[r22_eval] arm=${ARM} per-frame CSVs: ${N_CSV} (expect ${STEPS})"
echo "[r22_eval] arm=${ARM} 'Error:' lines: ${N_ERR} (MUST be 0)"
echo "[r22_eval] failures: ${FAILED}"

if [ "$FAILED" -ne 0 ] || [ "$N_ERR" -ne 0 ] || [ "$N_CSV" -ne "$STEPS" ]; then
  echo "[r22_eval] TASK FAILED"
  exit 1
fi
echo "[r22_eval] TASK OK"
exit 0
