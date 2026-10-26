#!/bin/bash
# R54 stage 2 (CPU): the draft-free rho field -- R40's flow-warped anchor maps, max of the two capped maps, R49's exposure inverse.
#
# Usage: sbatch slurm_scripts/five_bench/r54_field.sh        (r54_se_chain.sh)
# Reads R40's raw warped maps $D/r40_div/{dino_warp,lpips_warp}/edit{T}/{video}.npz (key `d` [F, 1560]) -- r40_div is a link to /store, which only
# some nodes mount, so this job is PINNED to nodecpu12 (a CPU node with the mount). The field is written to /home ($D/r54_field_full), so the
# render that follows runs on any L40S. r54_field.py checks every pair's latent-frame count against R49's field ($D/r49_field_full/se_max).
# Output: $D/r54_field_full/se_warp_max/edit{T}/{video}.npz   (key `rho`, r31_stage3 --rho_dir format)
#SBATCH --job-name=r54_field
#SBATCH --partition=CPU
#SBATCH --nodelist=nodecpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r54_field_%j.out
#SBATCH --error=logs/r54_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r7_anchor_manifest.json
WARP=$D/r40_div
REF=$D/r49_field_full
OUT=$D/r54_field_full
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs
echo "[r54_field] job=${SLURM_JOB_ID:-local} node=$(hostname) cases=${CASES} warp=${WARP} out=${OUT}"

# ---- preflight: R40's maps must be readable HERE (a /store link) and complete
if [ ! -d /store/PERSO/skhairi/Outputs/five_bench/r40_div ]; then
  echo "[r54_field] FATAL: /store not mounted on $(hostname) -- R40's maps live there (nodecpu12 / node39 / gpu-gw have it)"; exit 1
fi
FAILED=0
for M in dino_warp lpips_warp; do
  N=$(ls "$WARP/$M"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r54_field] ${M}: ${N} npz (need 419)"
  [ "$N" -ne 419 ] && FAILED=$((FAILED + 1))
done
N=$(ls "$REF"/se_max/edit*/*.npz 2>/dev/null | wc -l)
echo "[r54_field] r49_field_full/se_max: ${N} npz (need 419)"
[ "$N" -ne 419 ] && FAILED=$((FAILED + 1))
[ "$FAILED" -ne 0 ] && { echo "[r54_field] FATAL: inputs incomplete"; echo "[r54_field] failures: ${FAILED}"; exit 1; }

mkdir -p "$OUT"          # only once the inputs are known to be readable (a failed preflight leaves nothing behind)
$PY_FB evaluation/r54_field.py --selftest || { echo "[r54_field] FAILED selftest"; exit 1; }
$PY_FB evaluation/r54_field.py --cases "$CASES" --warp_root "$WARP" --ref_root "$REF" --out_root "$OUT"
RC=$?

# ---- post-run guard: one npz per pair
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=$(( RC != 0 ))
N=$(ls "$OUT"/se_warp_max/edit*/*.npz 2>/dev/null | wc -l)
echo "[r54_field] se_warp_max: ${N} npz (expect ${N_CASES})"
[ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
echo "[r54_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
