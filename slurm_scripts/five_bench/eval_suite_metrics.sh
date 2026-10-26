#!/bin/bash
# R57 unified metric suite, job 1 of 3 (1x L40S): the FiVE harness (incl. motion fidelity), FiVE-Acc, CLIP-D, EditReward
# and PSC of one method on FiVE-Bench-Correct, one stage after the other. Submitted by eval_suite.sh together with
# eval_suite_cf.sh (4 GPUs, runs in parallel) and eval_suite_merge.sh (afterok on both); to run it alone:
#   sbatch slurm_scripts/five_bench/eval_suite_metrics.sh --method M --root R --tag T [--layout edit_video] [--cases C]
# Arguments and output names: eval_suite_common.sh. Stages: each has its exit-code / 'Error:' gate (evaluate.py swallows
# metric exceptions into nan cells + an 'Error:' line) and a per-type row-count guard; the job fails at the end if any
# stage failed, but later stages still run.
#   harness  evaluate.py: structure distance, PSNR/LPIPS/MSE/SSIM on the unedited part, CLIP source/target/edited part, NIQE,
#            motion fidelity (CoTracker) + its edit-part variant
#   fiveacc  evaluate.py --metrics five_acc alone (Qwen2.5-VL-7B co-resident with the rest OOMed on 40-48 GB cards, R20)
#   clipd    r33_clip_directional.py
#   editreward  editreward_score.py   in the `editreward` env (transformers 4.57.0), not five-bench
#   psc      psc_score.py (VideoCLIP-XL-v2)
#SBATCH --job-name=suite_metrics
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/eval_suite_metrics_%j.out
#SBATCH --error=logs/eval_suite_metrics_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
# shellcheck source=/dev/null
source "$REPO/slurm_scripts/five_bench/eval_suite_common.sh"
suite_parse_args "$@" || exit 1
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
suite_expect || exit 1
echo "[eval_suite] metrics job=${JOB} method=${METHOD} tag=${TAG} layout=${LAYOUT} root=${ROOT}"
echo "[eval_suite] cases=${CASES:-<every pair of $EDIT_JSON_DIR>} expect per type: ${EXPECT[*]}"
echo "[eval_suite] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

F=0
HARNESS_METRICS=(structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part
                 clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part
                 niqe_target_image motion_fidelity_score motion_fidelity_score_edit_part)

run_evaluate() {  # $1 = kind (harness|fiveacc), $2 = stem, then the metrics
  local KIND=$1 STEM=$2; shift 2
  local LOG="logs/eval_suite_${JOB}_${KIND}.metrics.log"
  echo "=== [eval_suite] ${KIND} $(date '+%F %T')"
  $PY_FB evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml --metrics "$@" \
    --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --tgt_methods "$ROOT" --tgt_layout "$LAYOUT" --tgt_key "$METHOD" --frame_stride 8 --tmp_resize "${CASES_ARG[@]}" \
    --result_path "evaluation/csv/${STEM}.csv" > "$LOG" 2>&1
  local RC=$?
  stage_gate "$STEM" "$RC" "$LOG" || F=1
  check_harness_rows "$STEM" "$KIND" || F=1
}

# ---- harness (+ motion fidelity) and FiVE-Acc
run_evaluate harness "suite_${TAG}_${METHOD}" "${HARNESS_METRICS[@]}"
run_evaluate fiveacc "suite_fiveacc_${TAG}_${METHOD}" five_acc

# ---- CLIP-D. --cases of r33_clip_directional.py needs prompts, which the suite's cases file does not carry:
# a full-format cases file is built from the annotations first.
echo "=== [eval_suite] clipd $(date '+%F %T')"
CLIPD_OUT="evaluation/csv/suite_clipd_${TAG}_${METHOD}.csv"
CLIPD_LOG="logs/eval_suite_${JOB}_clipd.run.log"
if [ -n "$CASES" ]; then
  CLIPD_CASES=$(mktemp -p "${TMPDIR:-/tmp}" suite_clipd_cases.XXXXXX)
  $PY_FB -I "$SUITE_DIR/suite_common.py" clipd_cases --edit_json_dir "$EDIT_JSON_DIR" --cases "$CASES" \
    --data_root "$DATA_ROOT" --out "$CLIPD_CASES" > "$CLIPD_LOG" 2>&1 || { echo "[eval_suite] clipd cases FAILED"; F=1; }
  CLIPD_SCOPE=(--cases "$CLIPD_CASES")
else
  CLIPD_SCOPE=(--fullbench --edit_json_dir "$EDIT_JSON_DIR")
fi
$PY_FB evaluation/r33_clip_directional.py "${CLIPD_SCOPE[@]}" --methods "${METHOD}=${ROOT}" --data_root "$DATA_ROOT" \
  --frame_stride 8 -o "$CLIPD_OUT" >> "$CLIPD_LOG" 2>&1
RC=$?
stage_gate clipd "$RC" "$CLIPD_LOG" || F=1
check_keyed_rows "$CLIPD_OUT" clipd "$METHOD" || F=1

# ---- EditReward (own env) and PSC
for STAGE in editreward psc; do
  echo "=== [eval_suite] ${STAGE} $(date '+%F %T')"
  OUT="evaluation/csv/suite_${STAGE}_${TAG}_${METHOD}.csv"
  LOG="logs/eval_suite_${JOB}_${STAGE}.run.log"
  if [ "$STAGE" = editreward ]; then PY=$PY_ER; else PY=$PY_FB; fi
  $PY -I "$SUITE_DIR/${STAGE}_score.py" --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" \
    --data_root "$DATA_ROOT" "${CASES_PY[@]}" --out "$OUT" > "$LOG" 2>&1
  RC=$?
  stage_gate "$STAGE" "$RC" "$LOG" || F=1
  check_keyed_rows "$OUT" "$STAGE" || F=1
done

echo "[eval_suite] metrics stage failures: ${F}"
exit "$F"
