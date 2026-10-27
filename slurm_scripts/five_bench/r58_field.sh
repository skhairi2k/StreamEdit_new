#!/bin/bash
# R58 stage 2b (ON JEAN ZAY; GPU-hours account only, so it also runs on an H100): inverse-response per-token controls of Wan-Edit's
# max_s5 / max_s9 arms for the 190 pairs of evaluation/r58_cases_maxs5.json (as r49_field.sh wan full, on the R58 roots).
#
# Usage: sbatch slurm_scripts/five_bench/r58_field.sh        (r58_wan_chain.sh submits it, afterok r58_div)
# Input : attention $D/r58_wan_draft/attn, divergence $D/r58_div (wan_dino, wan_lpips)
# Output: $D/r58_field/{wan_max_s9,wan_max_s5}/edit{T}/{video}.npz + .npy start maps (r58_wan_render.sh reads wan_max_s5)
# Never writes into $D/r49_field_full: the output root is $D/r58_field only.
#SBATCH --job-name=r58_field
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r58_field_%j.out
#SBATCH --error=logs/r58_field_%j.err

CASES=evaluation/r58_cases_maxs5.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r58_field
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
[ -s "$CASES" ] || { echo "[r58_field] FATAL: missing $CASES (git pull?)"; exit 1; }
mkdir -p logs "$OUT"
echo "[r58_field] job=${SLURM_JOB_ID:-local} node=$(hostname) cases=${CASES} out=${OUT}"

$PY_FB evaluation/r49_field.py --selftest || { echo "[r58_field] FAILED selftest"; exit 1; }
$PY_FB evaluation/r49_field.py --hosts wan --cases "$CASES" --attn_wan "$D/r58_wan_draft/attn" --div_root "$D/r58_div" --out_root "$OUT"
RC=$?

# ---- post-run guard: one npz and one .npy start map per case clip in each arm dir
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=$(( RC != 0 ))
for DIR in wan_max_s9 wan_max_s5; do
  N=$(ls "$OUT/$DIR"/edit*/*.npz 2>/dev/null | wc -l)
  NN=$(ls "$OUT/$DIR"/edit*/*.npy 2>/dev/null | wc -l)
  echo "[r58_field] ${DIR}: ${N} npz, ${NN} start maps (expect ${N_CASES})"
  [ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
  [ "$NN" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
done
echo "[r58_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
