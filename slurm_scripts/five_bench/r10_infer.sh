#!/bin/bash
# R10 -- claim-4 pre-test: head-gated persistent visual-prompt injection.
# Single GPU, no array: 20 clips x 6 arms run sequentially in one job (the model
# loads once, so one job beats an array here). ~85s/run => ~2.9h for 120 runs,
# inside the 6h limit.
# Arms (identical base: blend_off, bg source-KV injection KEPT):
#   none          no VP             -> motion ceiling / edit floor + plumbing check
#   all           VP -> 360 heads   -> expected static repro (P3)
#   spatial       VP -> 117 heads   -> expected motion back, edit persists (P1)
#   temporal      VP -> 49 heads    -> expected static + drifting (P2, claim 4)
#   rand_spatial  VP -> 117 RANDOM  -> control for `spatial`
#   rand_temporal VP -> 49 RANDOM   -> control for `temporal`
# The two rand_* arms are count-matched: spatial and temporal differ in size, so
# without them a spatial-vs-temporal gap confounds head type with head count.
# Gates come from R19 (equal-budget disjoint key sets), NOT the superseded
# r10_head_gates.pt built on the 1560-vs-18 comparison.
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r10_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=logs/r10_infer_%j.out
#SBATCH --error=logs/r10_infer_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r10_vp_arms
# Same anchor files the SS4.5/R7 arm reads, so the comparison differs only by
# injection mechanism (persistent bank vs cached initial latent).
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors

cd
source .bashrc
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

# default full set; override with script args, e.g. `sbatch r10_infer.sh 0001_bus`
# edit1(1) + edit2(14) + edit5(3) + edit6(2) = 20. 0011_lucia runs under BOTH
# edit2 and edit5, so its case_ids carry an _e{T} suffix and outputs are keyed
# {arm}/edit{T}/{video} -- a flat layout would have the two runs collide.
CASES=(0001_bus \
       0028_kite-walk 0040_tennis 0057_dog 0011_lucia_e2 0014_burnout \
       0016_horsejump-high 0068_planes-water 0072_dog-agility 0074_rhino \
       0075_A_bicycle 0076_A_rabbit 0079_A_bus 0090_A_deer 0091_A_hawk \
       0007_guitar-violin 0069_car-turn 0011_lucia_e5 \
       0042_gym-ball 0002_girl-dog)
if [ "$#" -gt 0 ]; then CASES=("$@"); fi

echo "[r10_infer] job=${SLURM_JOB_ID} cases=${#CASES[@]} -> ${OUT_ROOT}"

python evaluation/r10_vp_arms.py \
  --cases "${CASES[@]}" \
  --arms none all spatial temporal rand_spatial rand_temporal \
  --gates evaluation/r19_head_gates.pt \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --anchor_root "$ANCHOR_ROOT" \
  --step 15 \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --seed 0

echo "[r10_infer] frame dirs present: $(ls -d "$OUT_ROOT"/*/*/*/ 2>/dev/null | wc -l) (expect 6 arms x ${#CASES[@]} cases = 120)"
