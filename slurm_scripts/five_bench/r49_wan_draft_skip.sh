#!/bin/bash
# R49 side experiment: Wan-Edit draft (17-step grid) started at step 2 (sigma 0.974), frames only, on the 18 clips of
# evaluation/r49_cases.json -- the closest 17-step start to skip 5 (sigma 0.978) on the 50-step grid.
# Usage: sbatch slurm_scripts/five_bench/r49_wan_draft_skip.sh [SKIP]      (default 2)
# Output: $D/r49_wan_draft/frames/skip{SKIP}/edit{T}/{video}/*.png   (resumable)
#SBATCH --job-name=r49_draft_skip
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_draft_skip_%j.out
#SBATCH --error=logs/r49_draft_skip_%j.err

SKIP=${1:-2}
CASES=evaluation/r49_cases.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "[r49_draft_skip] job=${SLURM_JOB_ID:-local} node=$(hostname) skip=${SKIP}"; nvidia-smi -L 2>&1 | head -2

$PY_FB evaluation/r49_wan_draft_skip.py --skip "$SKIP" --cases "$CASES" --out_frames "$D/r49_wan_draft/frames"
RC=$?
MISSING=$($PY_FB evaluation/r48_units.py --mode subset --missing "$D/r49_wan_draft/frames/skip${SKIP}" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
echo "[r49_draft_skip] clips without a complete draft: ${N_MISS}"
echo "[r49_draft_skip] failures: $(( (RC != 0) + (N_MISS != 0) ))"
exit $(( (RC != 0) || (N_MISS != 0) ))
