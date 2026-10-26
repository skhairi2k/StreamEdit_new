#!/bin/bash
# R59 stage 2b (CPU): R49's inverse-response per-token rho field for se_max, built from the 5-step Correct draft's attention and divergence.
#
# Usage: sbatch slurm_scripts/five_bench/r59_field.sh        (r59_se_chain.sh, afterok on r59_div.sh)
# Runs `r49_field.py --selftest`, then `r49_field.py --hosts se` UNCHANGED (R49's caps, lambda, q = S * max, budget-linear inverse on [2, 50],
# 15-step t_next) on the 415 pairs of evaluation/r43_anchor_manifest_correct.json.
# Inputs : attention $D/r59_se_attn, divergence $D/r59_div/se_{dino,lpips}
# Output : $D/r59_field_full/se_max/edit{T}/{video}.npz   (key `rho`, r31_stage3 --rho_dir format)
#SBATCH --job-name=r59_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r59_field_%j.out
#SBATCH --error=logs/r59_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r43_anchor_manifest_correct.json
OUT=$D/r59_field_full
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r59_field] job=${SLURM_JOB_ID:-local} node=$(hostname) cases=${CASES} out=${OUT}"

$PY_FB evaluation/r49_field.py --selftest || { echo "[r59_field] FAILED selftest"; exit 1; }
$PY_FB evaluation/r49_field.py --hosts se --cases "$CASES" --attn_se "$D/r59_se_attn" --div_root "$D/r59_div" --out_root "$OUT"
RC=$?

# ---- post-run guard: one npz per pair
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=$(( RC != 0 ))
N=$(ls "$OUT"/se_max/edit*/*.npz 2>/dev/null | wc -l)
echo "[r59_field] se_max: ${N} npz (expect ${N_CASES})"
[ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
echo "[r59_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
