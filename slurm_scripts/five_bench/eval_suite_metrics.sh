#!/bin/bash
# R57 unified metric suite, the 1-GPU job (1x L40S): the FiVE harness (incl. motion fidelity), FiVE-Acc, CLIP-D, EditReward
# and PSC on FiVE-Bench-Correct, one stage after the other, for each of the job's methods in turn (--unit NAME=DIR, repeatable,
# or --method/--root for one). Submitted by eval_suite.sh (with eval_suite_cf.sh and eval_suite_merge.sh); to run it alone:
#   sbatch slurm_scripts/five_bench/eval_suite_metrics.sh --method M --root R --tag T [--layout edit_video] [--cases C]
#   sbatch slurm_scripts/five_bench/eval_suite_metrics.sh --unit M1=R1 --unit M2=R2 --tag T
# Arguments and output names: eval_suite_common.sh. Stages: each has its exit-code / 'Error:' gate (evaluate.py swallows
# metric exceptions into nan cells + an 'Error:' line) and a per-type row-count guard; the job fails at the end if any
# stage of any method failed, but later stages and methods still run. Every stage start is timestamped (=== lines), so the
# .out gives the time per stage and per method.
# Resumable (a job killed at 24 h is relaunched with the same command): the harness and FiVE-Acc score only the pairs with
# no usable row yet (evaluate.py on those pairs, rows appended); EditReward and PSC skip scored pairs themselves; CLIP-D is
# skipped when complete, else redone (~15 min). Merge: at the end of each method whose stages all passed, its tables are
# written -- with CF when CF is complete, else with empty CF columns (the CF job re-merges when it finishes later).
#   harness  evaluate.py: structure distance, PSNR/LPIPS/MSE/SSIM on the unedited part, CLIP source/target/edited part, NIQE,
#            motion fidelity (CoTracker, whole video)
#   fiveacc  evaluate.py --metrics five_acc alone (Qwen2.5-VL-7B co-resident with the rest OOMed on 40-48 GB cards, R20)
#   clipd    r33_clip_directional.py
#   editreward  editreward_score.py   in the `editreward` env (transformers 4.57.0), not five-bench
#   psc      psc_score.py (VideoCLIP-XL-v2)
# Time: ~7-9 h per method on 415 pairs (estimate, 2026-11-03), so 24 h (the L40S partition maximum) holds two methods.
#SBATCH --job-name=suite_metrics
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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
echo "[eval_suite] metrics job=${JOB} tag=${TAG} layout=${LAYOUT} src_window=${SRC_WINDOW} methods=${#UNITS[@]}: ${UNITS[*]%%=*}"
echo "[eval_suite] cases=${CASES:-<every pair of $EDIT_JSON_DIR>} expect per type: ${EXPECT[*]}"
echo "[eval_suite] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

F=0
HARNESS_METRICS=(structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part
                 clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part
                 niqe_target_image motion_fidelity_score)
# = FiVE-Bench's Table 2 columns (+ clip_similarity_source_image); its code's motion_fidelity_score_edit_part is not
# reported in the paper and is NaN on empty-mask clips, so it is not computed (decided 2026-11-03)

# CLIP-D's --cases needs prompts, which the suite's cases file does not carry: build a full-format cases file once
CLIPD_CASES=""
if [ -n "$CASES" ]; then
  CLIPD_CASES=$(mktemp -p "${TMPDIR:-/tmp}" suite_clipd_cases.XXXXXX)
  trap 'rm -f "$CLIPD_CASES"' EXIT
  $PY_FB -I "$SUITE_DIR/suite_common.py" clipd_cases --edit_json_dir "$EDIT_JSON_DIR" --cases "$CASES" \
    --data_root "$DATA_ROOT" --out "$CLIPD_CASES" > "logs/eval_suite_${JOB}_clipd_cases.log" 2>&1 \
    || { echo "[eval_suite] clipd cases FAILED (logs/eval_suite_${JOB}_clipd_cases.log)"; F=1; }
fi

run_evaluate() {  # $1 = kind (harness|fiveacc), $2 = stem, then the metrics. Scores only the pairs without a usable row.
  local KIND=$1 STEM=$2; shift 2
  local LOG="logs/eval_suite_${JOB}_${METHOD}_${KIND}.metrics.log" PART="${STEM}__part${JOB}" MISS N RC
  echo "=== [eval_suite] ${METHOD} ${KIND} $(date '+%F %T')"
  rm -f evaluation/csv/*"${STEM}"__part*      # leftovers of a job killed before its append
  MISS=$(mktemp -p "${TMPDIR:-/tmp}" suite_miss.XXXXXX)
  N=$(suite_py harness_missing --stem "$STEM" --out "$MISS" ${ALLOW_NAN[@]:+--allow_nan "${ALLOW_NAN[@]}"} | tail -1)
  if [ "$N" = 0 ]; then
    echo "[eval_suite] ${METHOD} ${KIND}: every pair already scored -- skipped"
  else
    echo "[eval_suite] ${METHOD} ${KIND}: ${N} pair(s) to score"
    $PY_FB evaluation/fivebench/evaluate.py \
      --config_path evaluation/fivebench/config.yaml --metrics "$@" \
      --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
      --tgt_methods "$ROOT" --tgt_layout "$LAYOUT" --tgt_key "$METHOD" --frame_stride 8 --tmp_resize --cases_json "$MISS" \
      --result_path "evaluation/csv/${PART}.csv" > "$LOG" 2>&1
    RC=$?
    stage_gate "${METHOD} ${KIND}" "$RC" "$LOG" || UF=1
    suite_py harness_append --stem "$STEM" --part_stem "$PART" || UF=1
  fi
  rm -f "$MISS"
  check_harness_rows "$STEM" "$KIND" || UF=1
}

for U in "${UNITS[@]}"; do
  unit_set "$U"
  UF=0
  echo "##### [eval_suite] method ${METHOD} root=${ROOT} $(date '+%F %T')"

  # ---- harness (+ motion fidelity) and FiVE-Acc
  run_evaluate harness "suite_${TAG}_${METHOD}" "${HARNESS_METRICS[@]}"
  run_evaluate fiveacc "suite_fiveacc_${TAG}_${METHOD}" five_acc

  # ---- CLIP-D
  echo "=== [eval_suite] ${METHOD} clipd $(date '+%F %T')"
  CLIPD_OUT="evaluation/csv/suite_clipd_${TAG}_${METHOD}.csv"
  CLIPD_LOG="logs/eval_suite_${JOB}_${METHOD}_clipd.run.log"
  if [ -n "$CASES" ]; then CLIPD_SCOPE=(--cases "$CLIPD_CASES"); else CLIPD_SCOPE=(--fullbench --edit_json_dir "$EDIT_JSON_DIR"); fi
  if suite_py keyed_complete --csv "$CLIPD_OUT" --cols clip_d_prompt,clip_d_word --method "$METHOD"; then
    echo "[eval_suite] ${METHOD} clipd: complete -- skipped"
  else
    $PY_FB evaluation/r33_clip_directional.py "${CLIPD_SCOPE[@]}" --methods "${METHOD}=${ROOT}" --data_root "$DATA_ROOT" \
      --frame_stride 8 -o "$CLIPD_OUT" > "$CLIPD_LOG" 2>&1
    RC=$?
    stage_gate "${METHOD} clipd" "$RC" "$CLIPD_LOG" || UF=1
  fi
  check_keyed_rows "$CLIPD_OUT" "${METHOD} clipd" "$METHOD" || UF=1

  # ---- EditReward (own env) and PSC
  for STAGE in editreward psc; do
    echo "=== [eval_suite] ${METHOD} ${STAGE} $(date '+%F %T')"
    OUT="evaluation/csv/suite_${STAGE}_${TAG}_${METHOD}.csv"
    LOG="logs/eval_suite_${JOB}_${METHOD}_${STAGE}.run.log"
    # PSC takes --src_window (its s_src); EditReward pairs source and render frames positionally already
    if [ "$STAGE" = editreward ]; then PY=$PY_ER; WIN=(); else PY=$PY_FB; WIN=("${SRC_WINDOW_PY[@]}"); fi
    $PY -I "$SUITE_DIR/${STAGE}_score.py" --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" \
      --data_root "$DATA_ROOT" "${CASES_PY[@]}" "${WIN[@]}" --out "$OUT" > "$LOG" 2>&1
    RC=$?
    stage_gate "${METHOD} ${STAGE}" "$RC" "$LOG" || UF=1
    check_keyed_rows "$OUT" "${METHOD} ${STAGE}" || UF=1
  done

  # ---- merge this method's tables (full when its CF is complete, else with empty CF columns)
  if [ "$UF" -eq 0 ]; then
    merge_unit || UF=1
  else
    echo "[eval_suite] ${METHOD}: a stage failed -- not merged (relaunch the job: it resumes)"
  fi
  [ "$UF" -eq 0 ] || F=1
  echo "##### [eval_suite] method ${METHOD} done (failures: ${UF}) $(date '+%F %T')"
done

echo "[eval_suite] metrics stage failures: ${F}"
exit "$F"
