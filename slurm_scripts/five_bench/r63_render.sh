#!/bin/bash
# R63: UniEdit-Flow native baseline on FiVE-Bench-Correct (lab L40S, env unieditflow).
#
# Usage: sbatch --time=02:00:00 slurm_scripts/five_bench/r63_render.sh smoke            # 3 clips: edit1 / edit5 / edit6
#        sbatch --array=0-5 slurm_scripts/five_bench/r63_render.sh full                 # array index = edit type - 1, 415 pairs
# Output: $D/r63_uniedit_flow/{smoke,full}/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a task that hits
# the time limit is simply resubmitted alone with its own --array index.
#SBATCH --job-name=r63_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r63_render_%A_%a.out
#SBATCH --error=logs/r63_render_%A_%a.err

MODE=${1:-}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
PY=/home/ids/skhairi/anaconda3/envs/unieditflow/bin/python

cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

case "$MODE" in
  smoke)
    CASES=evaluation/r57_smoke_cases.json
    OUT=$D/r63_uniedit_flow/smoke
    TYPE_ARGS=()
    ;;
  full)
    TID=${SLURM_ARRAY_TASK_ID:-}
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 5 ]; then echo "[r63_render] bad array id '${TID}' (need 0-5)"; exit 1; fi
    CASES=evaluation/r43_anchor_manifest_correct.json
    OUT=$D/r63_uniedit_flow/full
    TYPE_ARGS=(--edit_type $((TID + 1)))
    ;;
  *) echo "[r63_render] usage: r63_render.sh smoke|full"; exit 1 ;;
esac
[ -x "$PY" ] || { echo "[r63_render] MISSING: $PY"; exit 1; }

echo "[r63_render] job=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0} mode=${MODE} cases=${CASES} out=${OUT}"
echo "[r63_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
$PY evaluation/r63_uniedit_render.py --cases "$CASES" "${TYPE_ARGS[@]}" --out_root "$OUT" \
  || { echo "[r63_render] r63_uniedit_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every selected clip has a complete render
$PY evaluation/r63_uniedit_render.py --cases "$CASES" "${TYPE_ARGS[@]}" --out_root "$OUT" --verify \
  || { echo "[r63_render] clips missing after the render"; FAILED=$((FAILED + 1)); }

echo "[r63_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
