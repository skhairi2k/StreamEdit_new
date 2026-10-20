#!/bin/bash
# R37 bg metrics -- distance-weighted PSNR/SSIM/LPIPS (+ the official FiVE column) for the
# 12 R35/R36 arms over the whole 419-pair bench, ONE ARM AFTER ANOTHER on the current
# interactive node (decided 2026-10-03 instead of an sbatch array: ~1.5 s/clip on an L40S
# -> ~10 min/arm, ~2 h total). Not an sbatch script: run it inside your interactive
# allocation, detached so a closed terminal does not kill it:
#
#   nohup bash slurm_scripts/five_bench/r37_bg_metrics_local.sh > logs/r37_bg_local.out 2>&1 &
#
# Per arm: preflight (100/100/100/100/9/10 real pairs) -> bg_metrics.py CLI ->
# evaluation/csv/r37_bg_${STEM}.csv (+ r37_bg_edgecases_${STEM}.csv), log
# logs/r37_bg_${STEM}.out. An arm whose CSV already has the full 419 x 23 rows is
# skipped, so re-running after an interruption resumes. A failing arm does not stop the
# others; the script exits 1 at the end if any arm failed.
#
# STEM matches the existing official CSV stems (edit{T}_FiVE_${STEM}_frame_stride8.csv),
# which r37_summarize.py uses for the parity check.

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort at `conda activate` (R20 job 907410).
set -o pipefail

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT=~/Data/dataggen/outputs/five_bench

# "STEM|render dir holding edit{T}/{video}/"
ARMS=(
  "r35_dino_unblended|$OUT/r35_arms/r35_dino_unblended/step14"
  "r35_dino_first2|$OUT/r35_arms/r35_dino_first2/step14"
  "r35_lpips_unblended|$OUT/r35_arms/r35_lpips_unblended/step14"
  "r35_lpips_first2|$OUT/r35_arms/r35_lpips_first2/step14"
)
for R in 2 3 4 6 8 10 20 50; do
  ARMS+=("r36_rho${R}_vp|$OUT/r36_rho_sweep/r36_rho${R}_vp")
done
# Second pass (2026-10-07): R38 N=5 routing arms + R39 matched-compute rho sweeps.
# The first 12 arms are skipped automatically (complete CSVs).
for C in dino_n5_unblended dino_n5_tgt09 lpips_n5_unblended lpips_n5_tgt09; do
  ARMS+=("r38_${C}|$OUT/r38_arms/r38_${C}/step14")
done
for N in 20 30; do
  for R in 2 3 4 6 8 10 20 50; do
    ARMS+=("r39_n${N}_rho${R}_vp|$OUT/r39_rho_sweep/r39_n${N}_rho${R}_vp")
  done
done

N_ALPHA_ROWS=23   # 21 alphas (0..1 step 0.05) + avg + official
EXPECT_ROWS=$((419 * N_ALPHA_ROWS))
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

cd "$REPO" || exit 1
# The eval env is five-bench; .bashrc auto-activates streamgve, so pop it first.
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r37_bg] node=$(hostname) job=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r37_bg] $(date -Is) start: ${#ARMS[@]} arms, expect ${EXPECT_ROWS} rows each"

FAILED=()
for ENTRY in "${ARMS[@]}"; do
  STEM=${ENTRY%%|*}
  DIR=${ENTRY#*|}
  CSV="evaluation/csv/r37_bg_${STEM}.csv"
  EDGE="evaluation/csv/r37_bg_edgecases_${STEM}.csv"
  LOG="logs/r37_bg_${STEM}.out"

  if [ -f "$CSV" ] && [ $(( $(wc -l < "$CSV") - 1 )) -eq "$EXPECT_ROWS" ]; then
    echo "[r37_bg] ${STEM}: complete CSV exists -- skipped"
    continue
  fi

  # ---- preflight: count REAL pairs (old evaluate.py runs left {video}_resize siblings).
  BAD=0
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r37_bg] ${STEM} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=1; }
  done
  if [ "$BAD" -ne 0 ]; then
    echo "[r37_bg] FAILED ${STEM} -- render incomplete: $DIR"
    FAILED+=("$STEM"); continue
  fi

  echo "[r37_bg] $(date -Is) ${STEM}: scoring $DIR"
  python evaluation/fivebench/bg_metrics.py \
    --method_root "$DIR" \
    --method_name "$STEM" \
    --data_root "$DATA_ROOT" \
    --edit_types 1 2 3 4 5 6 \
    --stride 8 \
    --alpha_step 0.05 --alpha_to 1.0 --alpha_max 0.5 \
    --out_csv "$CSV" \
    --edgecase_csv "$EDGE" \
    --seed 0 \
    > "$LOG" 2>&1
  RC=$?

  NERR=$(grep -c '^ERROR' "$LOG")
  NROWS=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NROWS" -ne "$EXPECT_ROWS" ]; then
    echo "[r37_bg] FAILED ${STEM} -- exit=${RC} ERROR_lines=${NERR} rows=${NROWS}/${EXPECT_ROWS} (see $LOG)"
    FAILED+=("$STEM"); continue
  fi
  echo "[r37_bg] $(date -Is) OK ${STEM}: ${NROWS} rows -> $CSV"
done

if [ "${#FAILED[@]}" -ne 0 ]; then
  echo "[r37_bg] $(date -Is) DONE with failures: ${FAILED[*]}"
  exit 1
fi
echo "[r37_bg] $(date -Is) DONE: all ${#ARMS[@]} arms OK"
exit 0
