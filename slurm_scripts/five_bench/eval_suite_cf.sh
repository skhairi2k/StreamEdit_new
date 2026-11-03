#!/bin/bash
# R57 unified metric suite, the 4-GPU job (4x L40S on one node): IVEBench's Content Fidelity, for each of the job's methods
# in turn (--unit NAME=DIR, repeatable, or --method/--root). The Qwen2.5-VL-72B judge is 147 GB in bf16, so it is spread
# over the node's four GPUs (device_map=auto); it runs in its own job so the other stages do not hold four GPUs. Submitted
# by eval_suite.sh; to run it alone:
#   sbatch slurm_scripts/five_bench/eval_suite_cf.sh --method M --root R --tag T [--layout edit_video] [--cases C]
# Arguments and output names: eval_suite_common.sh. Output evaluation/csv/suite_cf_{tag}_{method}.csv (cf, cf_raw).
# The judge is reloaded for each method (cf_score.py runs once per method, ~3 min to load). Resumable: cf_score.py skips
# scored pairs. Merge: after each method, the full table is written if its metrics are complete (else the metrics job does it).
# Resumable: a rerun skips the pairs already scored.
# ffmpeg: IVEBench's video preparation hard-codes libx264, which ~/anaconda3/bin/ffmpeg (4.3) lacks. five-bench ships
# imageio-ffmpeg's static 7.0.2 build; it is linked as `ffmpeg` into a per-job directory placed first on PATH.
#SBATCH --job-name=suite_cf
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_suite_cf_%j.out
#SBATCH --error=logs/eval_suite_cf_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
# shellcheck source=/dev/null
source "$REPO/slurm_scripts/five_bench/eval_suite_common.sh"
suite_parse_args "$@" || exit 1
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
suite_expect || exit 1
echo "[eval_suite] cf job=${JOB} tag=${TAG} layout=${LAYOUT} methods=${#UNITS[@]}: ${UNITS[*]%%=*}"
echo "[eval_suite] cases=${CASES:-<every pair of $EDIT_JSON_DIR>} expect per type: ${EXPECT[*]}"
echo "[eval_suite] node=$(hostname)"; nvidia-smi -L 2>&1

FFBIN=$(mktemp -d -p "${TMPDIR:-/tmp}" suite_ffbin.XXXXXX)
trap 'rm -rf "$FFBIN"' EXIT
FFEXE=$($PY_FB -I -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') || { echo "[eval_suite] FAILED: imageio_ffmpeg not importable in five-bench"; exit 1; }
ln -s "$FFEXE" "$FFBIN/ffmpeg"
export PATH="$FFBIN:$PATH"
"$FFBIN/ffmpeg" -hide_banner -encoders 2>/dev/null | grep -q ' libx264 ' \
  || { echo "[eval_suite] FAILED: ${FFEXE} has no libx264 encoder"; exit 1; }
echo "[eval_suite] ffmpeg=${FFEXE} (libx264 ok)"

F=0
for U in "${UNITS[@]}"; do
  unit_set "$U"
  OUT="evaluation/csv/suite_cf_${TAG}_${METHOD}.csv"
  LOG="logs/eval_suite_${JOB}_${METHOD}_cf.run.log"
  echo "=== [eval_suite] ${METHOD} cf $(date '+%F %T') root=${ROOT}"
  $PY_FB -I "$SUITE_DIR/cf_score.py" --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" \
    --data_root "$DATA_ROOT" "${CASES_PY[@]}" --ivebench_dir ~/Code/IVEBench \
    --model_path "$QWEN72B_DIR" --out "$OUT" > "$LOG" 2>&1
  RC=$?
  UF=0
  stage_gate "${METHOD} cf" "$RC" "$LOG" || UF=1
  check_keyed_rows "$OUT" "${METHOD} cf" || UF=1
  # full merge if this method's metrics are already complete; otherwise the metrics job merges when it finishes
  if [ "$UF" -eq 0 ]; then
    if metrics_done; then merge_unit || UF=1
    else echo "[eval_suite] ${METHOD}: metrics not complete yet -- the metrics job will merge (with CF) when it finishes"; fi
  fi
  [ "$UF" -eq 0 ] || F=1
done
echo "=== [eval_suite] cf done $(date '+%F %T')"
echo "[eval_suite] cf stage failures: ${F}"
exit "$F"
