#!/bin/bash
# R64: FlowDirector (official V1 code @cecd56e, released script defaults incl. its seed 42) on the 415 FiVE-Bench-Correct pairs.
# One array task = one shard (cases[shard::12], ~35 clips x ~11 min ~= 6.5 h) of evaluation/r43_anchor_manifest_correct.json; the model loads once per task.
#
# Usage: sbatch slurm_scripts/five_bench/r64_render.sh          (after `bash evaluation/r64_make_env.sh` and the smoke run)
# Output: ~/Data/dataggen/outputs/five_bench/r64_flowdirector/correct_full/edit{T}/{video}/*.png   (edit_video layout)
# RESUMABLE: finished clips are skipped, so a timed-out task is simply resubmitted (sbatch --array=<id> ...).
#SBATCH --job-name=r64_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
# 8 cores x 2 hyperthreads (nomultithread) = 16 logical CPUs = one GPU share on L40S nodes (64 CPUs / 4 GPUs)
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-11
#SBATCH --exclude=node52
#SBATCH --output=logs/r64_render_%A_%a.out
#SBATCH --error=logs/r64_render_%A_%a.err

N_SHARDS=12
SHARD=${SLURM_ARRAY_TASK_ID:-0}
CASES=evaluation/r43_anchor_manifest_correct.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT=~/Data/dataggen/outputs/five_bench/r64_flowdirector/correct_full
PY_FD=~/anaconda3/envs/flowdirector/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

echo "[r64_render] job=${SLURM_ARRAY_JOB_ID:-local}_${SHARD} node=$(hostname) shard=${SHARD}/${N_SHARDS} cases=${CASES} -> ${OUT}"
nvidia-smi -L 2>&1 | head -2

FAILED=0
T0=$(date +%s)
$PY_FD evaluation/r64_flowdirector_render.py --cases "$CASES" --out_root "$OUT" --shard "$SHARD" --n_shards "$N_SHARDS" \
  || { echo "[r64_render] r64_flowdirector_render.py reported failures"; FAILED=$((FAILED + 1)); }
echo "[r64_render] render wall time: $(( $(date +%s) - T0 )) s"

# ---- post-run guard: every clip of this shard has its expected PNG count
MISSING=$($PY_FD evaluation/r64_flowdirector_render.py --cases "$CASES" --out_root "$OUT" --shard "$SHARD" --n_shards "$N_SHARDS" --missing)
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r64_render] ${N_MISS} clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r64_render] shard ${SHARD}: ${N_MISS} incomplete clip(s)"
echo "[r64_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
