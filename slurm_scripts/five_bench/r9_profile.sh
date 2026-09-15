#!/bin/bash
# R9 -- E0 head-taxonomy profiling: degenerate edit (trg=src) + online SVG criterion.
# Single GPU, no array: the 5 cases run sequentially in one job (~minutes each).
# Case picks (motion coverage, from evaluation/cases.json):
#   0001_bus        rigid translation, static camera
#   0028_kite-walk  articulated human walking
#   0034_cows       animal locomotion
#   0069_car-turn   camera motion (following car)
#   0045_butterfly  low/fine motion
#SBATCH --job-name=r9_profile
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/r9_profile_%j.out
#SBATCH --error=logs/r9_profile_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r9_head_profile

cd
source .bashrc
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

# default full set; override with script args, e.g. `sbatch r9_profile.sh 0034_cows`
CASES=(0001_bus 0028_kite-walk 0034_cows 0069_car-turn 0045_butterfly)
if [ "$#" -gt 0 ]; then CASES=("$@"); fi

for CASE in "${CASES[@]}"; do
  echo "[r9_profile] job=${SLURM_JOB_ID} case=${CASE}"
  python evaluation/r9_head_profiler.py \
    --case "$CASE" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --map_mode marginal \
    --step 15 \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --seed 0
done

echo "[r9_profile] done: $(ls "$OUT_ROOT"/*/r9_scalars.npz 2>/dev/null | wc -l)/5 npz present"
echo "[r9_profile] npz size: $(du -sh "$OUT_ROOT" 2>/dev/null | cut -f1) (marginal mode ~5 MB/video; --dump_maps would add ~81 MB each)"
