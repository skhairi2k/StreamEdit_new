#!/bin/bash
# Quick one-off grid: source / grounding mask / R26 baseline b=2 / R30 lpips+normals+dino_patch
# for a single clip. CPU-only (no VAE decode, no model inference) -- submitted only because
# /projects/dataggen is not mounted on the login node. See evaluation/r30_quick_single_grid.py.
#
#SBATCH --job-name=r30_quick_grid
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/r30_quick_grid_%j.out
#SBATCH --error=logs/r30_quick_grid_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark

cd "$REPO" || exit 1
mkdir -p logs evaluation/figures

source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

python evaluation/r30_quick_single_grid.py \
  --case_id 0011_lucia_e2 \
  --baseline_b 2 \
  --arms lpips,normals,dino_patch \
  --data_root "$DATA_ROOT" \
  --out evaluation/figures/r30_quick_0011_lucia_e2.png

echo "[r30_quick_grid] done -> evaluation/figures/r30_quick_0011_lucia_e2.png"
