#!/bin/bash
# R49 stage 2b (CPU): inverse-response per-token controls for both hosts on the 18 clips of evaluation/r49_cases.json.
#
# Usage: sbatch slurm_scripts/five_bench/r49_field.sh        (normally submitted by r49_chain.sh, afterok on r49_div.sh)
# Runs `r49_field.py --selftest`, then the field on r49_cases.json (attention from r49_se_attn / r49_wan_draft/attn,
# divergence from r49_div). Then R48's field for the reference arm dino_SD on the NEW clip only (evaluation/r49_new_clip.json)
# from the same r49 inputs, into $D/r48_field -- the 17 other clips already have it from R48.
# Output: $D/r49_field/{se_max, wan_max_s9, wan_max_s5}/edit{T}/{video}.npz (+ .npy start maps for wan)
#         $D/r48_field/{se,wan}_dino_SD/edit3/0004_car-roundabout.npz (+ .npy for wan)
#SBATCH --job-name=r49_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r49_field_%j.out
#SBATCH --error=logs/r49_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r49_field
CASES=evaluation/r49_cases.json
NEW_CLIP=evaluation/r49_new_clip.json
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r49_field] job=${SLURM_JOB_ID:-local} node=$(hostname) cases=${CASES}"

$PY_FB evaluation/r49_field.py --selftest || { echo "[r49_field] FAILED selftest"; exit 1; }
$PY_FB evaluation/r49_field.py --cases "$CASES" --out_root "$OUT"
RC=$?
$PY_FB evaluation/r48_field.py --mode subset --cases "$NEW_CLIP" --arms dino_SD --out_root "$D/r48_field" \
  --attn_se "$D/r49_se_attn" --attn_wan "$D/r49_wan_draft/attn" --div_root "$D/r49_div"
RC_REF=$?

# ---- post-run guard: one npz per case clip in each output dir (and one .npy per clip for the wan floors)
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=$(( (RC != 0) + (RC_REF != 0) ))
for DIR in se_max wan_max_s9 wan_max_s5; do
  N=$(ls "$OUT/$DIR"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r49_field] ${DIR}: ${N} npz (expect ${N_CASES})"
  [ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
  if [[ "$DIR" == wan_* ]]; then
    NN=$(ls "$OUT/$DIR"/edit*/*.npy 2>/dev/null | wc -l)
    [ "$NN" -ne "$N_CASES" ] && { echo "[r49_field] ${DIR}: ${NN} start maps (expect ${N_CASES})"; FAILED=$((FAILED + 1)); }
  fi
done
for H in se wan; do   # the reference field of the new clip (type 3) exists for both hosts
  [ -s "$D/r48_field/${H}_dino_SD/edit3/0004_car-roundabout.npz" ] || { echo "[r49_field] MISSING reference field ${H}_dino_SD for 0004_car-roundabout"; FAILED=$((FAILED + 1)); }
done
[ -s "$D/r48_field/wan_dino_SD/edit3/0004_car-roundabout.npy" ] || { echo "[r49_field] MISSING wan reference start map"; FAILED=$((FAILED + 1)); }
echo "[r49_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
