#!/bin/bash
# R5 -- FiVE-Bench real-length background-normalization inference (--bg_realwords on, selectivity off).
# SLURM array over the six edit types: task i runs edit type (i+1) via the shared runner.
# omega/rho identical to R1 baseline for a fair A/B; only the bg denominator (Lk_bg=L_real) differs.
#SBATCH --job-name=r5_infer
#SBATCH --partition=L40S
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r5_infer_%A_%a.out
#SBATCH --error=logs/r5_infer_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench


cd
source .bashrc
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

# edit type T = SLURM_ARRAY_TASK_ID + 1  (1..6)
T=$((SLURM_ARRAY_TASK_ID + 1))
echo "[r5_infer] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} -> edit_type=${T}"

python evaluation/run_fivebench.py \
  --edit_type "$T" \
  --method r5_bg_realwords \
  --bg_realwords \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --step 15 \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --flow_shift 1.0 \
  --seed 0
