#!/bin/bash
# R7 -- Qwen-Image-Edit-2511 oracle anchor generation for visual prompting.
# SLURM array over the six edit types: task i generates edit type (i+1) anchors.
# NO `conda activate`: the login .bashrc auto-activates streamgve (diffusers 0.31, no
# QwenImageEditPlusPipeline), so we invoke the qwen-edit env's python DIRECTLY with
# -E -s to ignore PYTHON* env vars and user site-packages.
#SBATCH --job-name=r7_gen_anchors
#SBATCH --partition=H100
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/r7_gen_anchors_%A_%a.out
#SBATCH --error=logs/r7_gen_anchors_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
QWEN_PYTHON=/home/ids/skhairi/anaconda3/envs/qwen-edit/bin/python

cd "$REPO"
mkdir -p logs evaluation/anchors

# edit type T = SLURM_ARRAY_TASK_ID + 1  (1..6)
T=$((SLURM_ARRAY_TASK_ID + 1))
echo "[r7_gen_anchors] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID} -> edit_type=${T}"

"$QWEN_PYTHON" -E -s evaluation/gen_anchors.py \
  --cases evaluation/r7_anchor_manifest.json \
  --edit_type "$T" \
  --data_root "$DATA_ROOT" \
  --out evaluation/anchors
