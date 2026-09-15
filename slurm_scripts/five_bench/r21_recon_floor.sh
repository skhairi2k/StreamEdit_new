#!/bin/bash
# Reconstruction-floor ladder on 3 clips -- see evaluation/r21_recon_floor.py.
# Decides whether our LPIPS gap to StreamGVE Table 1 is a pipeline defect or an
# unreachable floor, BEFORE committing ~10h to a full-bench --src_frames re-render.
#SBATCH --job-name=r21_recon
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r21_recon_%j.out
#SBATCH --error=logs/r21_recon_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
cd
source .bashrc
conda deactivate
conda activate streamgve
cd "$REPO"
mkdir -p logs
python evaluation/r21_recon_floor.py \
  --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
  --out_root /projects/dataggen/outputs/five_bench/r21_recon_floor \
  --edit_type 1 --videos 0001_bus 0002_girl-dog 0004_car-roundabout \
  --src_frames "${SRCF:-video}" --tag "${TAG:-}" \
  --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0
echo "[recon] exit=$?"
