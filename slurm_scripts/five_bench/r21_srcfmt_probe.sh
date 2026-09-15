#!/bin/bash
# Does feeding the model the jpg rendition (the one evaluate.py scores against)
# close the 4x LPIPS gap to StreamGVE Table 1?
#
# Arm A (--src_frames video)  = control. MUST come out bit-identical to the stored
#   R1 baseline on these 3 clips; that proves the --src_frames plumbing changed
#   nothing else and that subset renders reproduce full-bench ones (per-pair reseed).
# Arm B (--src_frames images) = the experiment.
#
# Decides between two worlds:
#   B ~ LPIPS 50-60  -> the mp4/jpg rendition mismatch was the whole story
#   B ~ LPIPS 200    -> real background-preservation gap in our implementation
#SBATCH --job-name=r21_srcfmt
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r21_srcfmt_%j.out
#SBATCH --error=logs/r21_srcfmt_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT=/projects/dataggen/outputs/five_bench/r21_srcfmt
CASES=$REPO/evaluation/cases_srcfmt.json

cd
source .bashrc
conda deactivate
conda activate streamgve
cd "$REPO"
mkdir -p logs

for MODE in video images; do
  echo "[srcfmt] === --src_frames $MODE ==="
  python evaluation/run_fivebench.py \
    --edit_type 1 --method "src_$MODE" \
    --src_frames "$MODE" \
    --cases_json "$CASES" \
    --data_root "$DATA_ROOT" --out_root "$OUT" \
    --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0 \
    || { echo "[srcfmt] FAILED $MODE"; exit 1; }
done

echo "[srcfmt] --- control: arm A must be bit-identical to the stored baseline ---"
BASE=/projects/dataggen/outputs/five_bench/baseline/edit1
SAME=0; DIFF=0
for v in 0001_bus 0002_girl-dog 0004_car-roundabout; do
  if diff -rq "$OUT/src_video/edit1/$v" "$BASE/$v" >/dev/null 2>&1; then
    echo "  $v: IDENTICAL to stored baseline"; SAME=$((SAME+1))
  else
    echo "  $v: DIFFERS from stored baseline"; DIFF=$((DIFF+1))
  fi
done
echo "[srcfmt] control: $SAME identical / $DIFF differ  (expect 3/0)"
echo "[srcfmt] done"
