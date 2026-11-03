#!/bin/bash
# R64 reproduction check: FlowDirector's 4 README demos with full prompts (app.py) rendered by upstream edit.py (untouched,
# seed 42) and by our driver (same inputs, same seed), then compared with each other and with the README GIFs.
# Expected: 8 renders x ~11 min + 5 model loads ~= 1 h 40 on L40S.
#
# Usage: sbatch slurm_scripts/five_bench/r64_repro_demo.sh
# Output: ~/Data/dataggen/outputs/five_bench/r64_flowdirector/repro_demo/{upstream,driver}/{demo}.mp4,
#         evaluation/csv/r64_repro_demo.csv, evaluation/figures/r64_repro_demo_{demo}.jpg
# RESUMABLE: existing mp4s are skipped, so a timed-out job is simply resubmitted.
#SBATCH --job-name=r64_repro_demo
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# 8 cores x 2 hyperthreads (nomultithread) = 16 logical CPUs = one GPU share on L40S nodes (64 CPUs / 4 GPUs)
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r64_repro_demo_%j.out
#SBATCH --error=logs/r64_repro_demo_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT=~/Data/dataggen/outputs/five_bench/r64_flowdirector/repro_demo
CSV=evaluation/csv/r64_repro_demo.csv
PY_FD=~/anaconda3/envs/flowdirector/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

echo "[r64_repro_demo] job=${SLURM_JOB_ID:-local} node=$(hostname) -> ${OUT}"
nvidia-smi -L 2>&1 | head -2

FAILED=0
for MODE in upstream driver compare; do
  T0=$(date +%s)
  $PY_FD evaluation/r64_repro_demo.py --$MODE --out_root "$OUT" \
    || { echo "[r64_repro_demo] --$MODE reported failures"; FAILED=$((FAILED + 1)); }
  echo "[r64_repro_demo] --$MODE wall time: $(( $(date +%s) - T0 )) s"
done

# ---- post-run guard: 8 videos (4 upstream + 4 driver) and a 4-row comparison csv
N_MP4=$(ls "$OUT"/upstream/*.mp4 "$OUT"/driver/*.mp4 2>/dev/null | wc -l)
[ "$N_MP4" -ne 8 ] && { echo "[r64_repro_demo] ${N_MP4}/8 mp4 present"; FAILED=$((FAILED + 1)); }
N_ROWS=0
[ -f "$CSV" ] && N_ROWS=$(( $(wc -l < "$CSV") - 1 ))
[ "$N_ROWS" -ne 4 ] && { echo "[r64_repro_demo] ${N_ROWS}/4 rows in ${CSV}"; FAILED=$((FAILED + 1)); }
echo "[r64_repro_demo] ${N_MP4}/8 mp4, ${N_ROWS}/4 csv rows"
echo "[r64_repro_demo] failures: ${FAILED}"
exit $(( FAILED > 0 ))
