#!/bin/bash
# R19 -- equal-budget head classification (SVG probe, block-causal).
# One GPU, no array: the 21 videos run sequentially in one job so the model
# loads once (~4 min/video amortised => ~1.5 h).
#
# Key sets (see evaluation/r9_head_profiler.py :: build_key_sets):
#   spatial  = the query's own frame, minus its own token          (1559 keys)
#   temporal = span = round(1560/(N-1)) positions near the query,
#              in each of the N-1 OTHER frames                     (~1560 keys)
# Budget-matched so the probe measures head type rather than key count, and
# disjoint so a good temporal score is attributable. Both band shapes (flat /
# disk) are computed in the same pass.
#
# evaluation/r19_gates.py MUST pass before this is launched -- it is the check
# that the two properties above actually hold.
#SBATCH --job-name=r19_profile
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/r19_profile_%j.out
#SBATCH --error=logs/r19_profile_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r19_head_profile

cd
source .bashrc
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

# Gate first: a failure here means the key sets no longer measure head type, and
# every margin downstream would be uninterpretable.
echo "[r19_profile] running blocking gates"
python evaluation/r19_gates.py || { echo "[r19_profile] GATES FAILED -- aborting"; exit 1; }

# The 21 unique video_names in evaluation/cases.json. 0011_lucia appears there
# under both edit 2 and edit 5; the probe runs a DEGENERATE edit (trg = src), so
# the two entries would give byte-identical captures -- only _e2 is listed.
CASES=(0001_bus 0028_kite-walk 0040_tennis 0057_dog 0011_lucia_e2 \
       0014_burnout 0016_horsejump-high 0068_planes-water 0072_dog-agility \
       0074_rhino 0075_A_bicycle 0076_A_rabbit 0079_A_bus 0090_A_deer \
       0091_A_hawk 0007_guitar-violin 0069_car-turn 0042_gym-ball \
       0002_girl-dog 0034_cows 0045_butterfly)
if [ "$#" -gt 0 ]; then CASES=("$@"); fi

echo "[r19_profile] job=${SLURM_JOB_ID} cases=${#CASES[@]} -> ${OUT_ROOT}"

for CASE in "${CASES[@]}"; do
  echo "[r19_profile] case=${CASE}"
  python evaluation/r9_head_profiler.py \
    --case "$CASE" \
    --cases_json evaluation/cases.json \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --map_mode marginal \
    --step 15 \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --seed 0
done

echo "[r19_profile] done: $(ls "$OUT_ROOT"/*/r9_scalars.npz 2>/dev/null | wc -l)/${#CASES[@]} npz present"
echo "[r19_profile] size: $(du -sh "$OUT_ROOT" 2>/dev/null | cut -f1)"
