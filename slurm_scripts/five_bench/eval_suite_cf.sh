#!/bin/bash
# R57 unified metric suite, job 2 of 3 (4x L40S on one node): IVEBench's Content Fidelity of one method. The Qwen2.5-VL-72B
# judge is 147 GB in bf16, so it is spread over the node's four GPUs (device_map=auto); it runs in its own job, in parallel
# with eval_suite_metrics.sh, so the other stages do not hold four GPUs. Submitted by eval_suite.sh; to run it alone:
#   sbatch slurm_scripts/five_bench/eval_suite_cf.sh --method M --root R --tag T [--layout edit_video] [--cases C]
# Arguments and output names: eval_suite_common.sh. Output evaluation/csv/suite_cf_{tag}_{method}.csv (cf, cf_raw).
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
#SBATCH --time=20:00:00
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
echo "[eval_suite] cf job=${JOB} method=${METHOD} tag=${TAG} layout=${LAYOUT} root=${ROOT}"
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
OUT="evaluation/csv/suite_cf_${TAG}_${METHOD}.csv"
LOG="logs/eval_suite_${JOB}_cf.run.log"
echo "=== [eval_suite] cf $(date '+%F %T')"
$PY_FB -I "$SUITE_DIR/cf_score.py" --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" \
  --data_root "$DATA_ROOT" "${CASES_PY[@]}" --ivebench_dir ~/Code/IVEBench \
  --model_path "$WEIGHTS_DIR/Qwen2.5-VL-72B-Instruct" --out "$OUT" > "$LOG" 2>&1
RC=$?
stage_gate cf "$RC" "$LOG" || F=1
check_keyed_rows "$OUT" cf || F=1
echo "[eval_suite] cf stage failures: ${F}"
exit "$F"
