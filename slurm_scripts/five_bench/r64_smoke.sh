#!/bin/bash
# R64 smoke: FlowDirector (official V1 code @cecd56e, released script defaults except seed 0) on the 3 R57 smoke clips
# (edit1/0001_bus replacement, edit5/0007_guitar-violin addition, edit6/0048_hike removal), before the 415-pair array.
# Checks the full chain on L40S and gives the per-clip time that sizes r64_render.sh (N_SHARDS, --time).
#
# Usage: sbatch slurm_scripts/five_bench/r64_smoke.sh          (after `bash evaluation/r64_make_env.sh` and check-triggers)
# Output: ~/Data/dataggen/outputs/five_bench/r64_flowdirector/smoke/edit{T}/{video}/*.png   (edit_video layout)
# RESUMABLE: finished clips are skipped.
#SBATCH --job-name=r64_smoke
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
#SBATCH --output=logs/r64_smoke_%j.out
#SBATCH --error=logs/r64_smoke_%j.err

CASES=evaluation/r57_smoke_cases.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT=~/Data/dataggen/outputs/five_bench/r64_flowdirector/smoke
PY_FD=~/anaconda3/envs/flowdirector/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

echo "[r64_smoke] job=${SLURM_JOB_ID:-local} node=$(hostname) cases=${CASES} -> ${OUT}"
nvidia-smi -L 2>&1 | head -2

FAILED=0
T0=$(date +%s)
$PY_FD evaluation/r64_flowdirector_render.py --cases "$CASES" --out_root "$OUT" \
  || { echo "[r64_smoke] r64_flowdirector_render.py reported failures"; FAILED=$((FAILED + 1)); }
echo "[r64_smoke] render wall time: $(( $(date +%s) - T0 )) s"

# ---- post-run guard: every smoke clip has its expected PNG count
MISSING=$($PY_FD evaluation/r64_flowdirector_render.py --cases "$CASES" --out_root "$OUT" --missing)
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r64_smoke] ${N_MISS} clip(s) without a complete render: $(echo "$MISSING" | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r64_smoke] ${N_MISS} incomplete clip(s)"
echo "[r64_smoke] failures: ${FAILED}"
exit $(( FAILED > 0 ))
