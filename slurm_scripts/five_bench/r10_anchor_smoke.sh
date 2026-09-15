#!/bin/bash
# R10 -- anchor-sim smoke test on a single clip, ALL six arms + the R7 baseline.
# rand_spatial is included deliberately: the first run showed `spatial` adherence
# sitting close to what pure head-COUNT scaling predicts, and rand_spatial is the
# arm that separates count from type. One clip is not a result, but it is a cheap
# early read on the arm that decides the experiment.
#
# Outputs go into the repo, NOT /tmp: /tmp is node-local on this cluster, so job
# 907281's CSVs died with the job and only stdout survived.
#SBATCH --job-name=anchor_smoke
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/anchor_smoke_%j.out
#SBATCH --error=logs/anchor_smoke_%j.err

cd; source .bashrc; conda deactivate; conda activate five-bench
cd /home/ids/skhairi/Code/StreamEdit_bigchantier
mkdir -p evaluation/csv/smoke

python evaluation/r10_anchor_sim.py \
  --in_root /projects/dataggen/outputs/five_bench/r10_vp_arms \
  --arms none all spatial temporal rand_spatial rand_temporal \
  --r7_root /projects/dataggen/outputs/five_bench/r7_visual_prompting \
  --cases 0057_dog \
  --out_csv evaluation/csv/smoke/r10_anchor_sim.csv
