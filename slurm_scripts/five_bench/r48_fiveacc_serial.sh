#!/bin/bash
# R48 fiveacc, SERIAL variant: the same work as r48_fiveacc.sh, all units one after another in ONE job.
#
# Why it exists: r48_fiveacc.sh is a 17-task array, and the per-user job cap (QOSMaxSubmitJobPerUserLimit) can refuse that many
# tasks at once. This wrapper takes one submit slot instead. It runs the real r48_fiveacc.sh once per unit, exactly as an array
# task would (it sets SLURM_ARRAY_TASK_ID by hand), so every guard, CSV and log of the array version is unchanged -- only the
# parallelism differs (one unit at a time, so ~17x slower, and the time limit below is shared by all units).
#
# Usage: sbatch [--dependency=afterok:<render jobs>] slurm_scripts/five_bench/r48_fiveacc_serial.sh subset|full [FIRST LAST]
# (FIRST LAST = optional inclusive unit range, e.g. `subset 12 16`, to cover only the units a job-cap refusal left out)
# Scores FiVE-Acc (r48_fiveacc.sh). The units come from evaluation/r48_units.py (subset: 17).
# Logs: logs/r48_fiveacc_serial_<job>.out; per-unit metrics logs logs/r48_fiveacc_<job>_<unit>.metrics.log.
# Exit code is non-zero if any unit failed ("[serial] unit failures: N"); a failed unit does not stop the loop.
#SBATCH --job-name=r48_fiveacc_serial
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_fiveacc_serial_%j.out
#SBATCH --error=logs/r48_fiveacc_serial_%j.err

MODE=${1:-}
case "$MODE" in subset|full) ;; *) echo "[r48_fiveacc_serial] usage: r48_fiveacc_serial.sh subset|full"; exit 1 ;; esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
PY_FB=~/anaconda3/envs/five-bench/bin/python
cd "$REPO" || exit 1
mkdir -p logs

N=$($PY_FB evaluation/r48_units.py --mode "$MODE" --count 2>/dev/null)
[ -n "$N" ] && [ "$N" -gt 0 ] || { echo "[r48_fiveacc_serial] FATAL: no units for mode ${MODE}"; exit 1; }
FIRST=${2:-0}; LAST=${3:-$((N - 1))}      # optional unit range: r48_fiveacc_serial.sh subset FIRST LAST (inclusive)
[ "$FIRST" -ge 0 ] && [ "$LAST" -lt "$N" ] && [ "$FIRST" -le "$LAST" ] || { echo "[r48_fiveacc_serial] bad unit range ${FIRST}-${LAST} (units 0-$((N - 1)))"; exit 1; }
echo "[r48_fiveacc_serial] job=${SLURM_JOB_ID} mode=${MODE} units=${FIRST}-${LAST} of ${N}"

FAIL=0
for I in $(seq $FIRST $LAST); do
  echo "=== [r48_fiveacc_serial] unit ${I} of ${N} ($(date '+%F %T')) ==="
  SLURM_ARRAY_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=$I bash slurm_scripts/five_bench/r48_fiveacc.sh "$MODE" || FAIL=$((FAIL + 1))
done
echo "[serial] unit failures: ${FAIL}"
exit $(( FAIL > 0 ))
