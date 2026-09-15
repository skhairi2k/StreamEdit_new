#!/bin/bash
# R7 -- FiVE-Bench inference with Visual Prompting (paper §4.5, independent first frame).
# Baseline pipeline config (selectivity/bg_realwords off) + Qwen-edited oracle anchors,
# for a clean A/B against R2's baseline_avg.csv.
# SLURM array over the six edit types: task i runs edit type (i+1) via the shared runner.
#SBATCH --job-name=r7_infer
#SBATCH --partition=L40S
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r7_infer_%A_%a.out
#SBATCH --error=logs/r7_infer_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench


cd
source .bashrc
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

# edit type T = SLURM_ARRAY_TASK_ID + 1  (1..6)
T=$((SLURM_ARRAY_TASK_ID + 1))
echo "[r7_infer] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} -> edit_type=${T}"

python evaluation/run_fivebench.py \
  --edit_type "$T" \
  --method r7_visual_prompting \
  --first_frame_edit_dir "${OUT_ROOT}/anchors" \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --step 15 \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --flow_shift 1.0 \
  --seed 0
