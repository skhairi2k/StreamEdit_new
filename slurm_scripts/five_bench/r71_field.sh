#!/bin/bash
# R71 stage 1 (CPU, ON JEAN ZAY after jz/r50_port.sh): Wan-Edit start maps under the range-normalized ControlLift mapping
#   m = round(49 q), s = 50 - m   (m_max = 49, s_min = 1)          instead of R49/R58's  s = clip(50 - round(50 q), 5, 50).
# Remapped from the demand q stored in the existing s_min 5 fields (no new draft, attention or divergence): evaluation/r71_remap_field.py
# gates every pair (stored start map == clip(50 - round(50 q), 5, 50), float32 rounding ties tolerated and counted), then writes
#   $D/r71_field/wan_{max,nosupp,nodiv,dino,lpips}_s1/edit{T}/{video}.npz + .npy      415 per arm
# Sources (the 190 / 225 split is A0 = R58 max_s5's own assembly):
#   max      r58_field/wan_max_s5 (190)          + r49_field_full/wan_max_s5 (225)
#   nosupp   r62_field/wan_nosupp_s5 (190)       + r51_field_full/wan_nosupp_s5 (225)
#   nodiv / dino / lpips   r68_field/wan_{arm}_s5 (415)
# Usage: sbatch slurm_scripts/five_bench/r71_field.sh            (r71_wan_chain.sh)
#SBATCH --job-name=r71_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r71_field_%j.out
#SBATCH --error=logs/r71_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench       # Jean Zay: links into $SCRATCH (r50_scratch_links.sh)
OUT=$D/r71_field
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
ARMS=(max nosupp nodiv dino lpips)

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r71_field] job=${SLURM_JOB_ID:-local} node=$(hostname) arms=${ARMS[*]} out=${OUT}"

# ---- preflight: every source root complete
FAILED=0
check() {  # $1 = dir under $D, $2 = minimum npz count
  local n; n=$(ls "$D/$1"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r71_field] $1: ${n} npz (need >= $2)"
  [ "$n" -ge "$2" ] || FAILED=$((FAILED + 1))
}
check r58_field/wan_max_s5 190;       check r49_field_full/wan_max_s5 225
check r62_field/wan_nosupp_s5 190;    check r51_field_full/wan_nosupp_s5 225
check r68_field/wan_nodiv_s5 415;     check r68_field/wan_dino_s5 415;     check r68_field/wan_lpips_s5 415
for F in evaluation/r43_anchor_manifest_correct.json evaluation/r58_cases_maxs5.json evaluation/r68_cases_reuse225.json evaluation/r71_remap_field.py; do
  [ -s "$F" ] || { echo "[r71_field] FATAL: missing $F (git pull?)"; FAILED=$((FAILED + 1)); }
done
for A in "${ARMS[@]}"; do
  [ -n "$(ls -A "$OUT/wan_${A}_s1" 2>/dev/null)" ] && { echo "[r71_field] REFUSE: $OUT/wan_${A}_s1 already holds files -- R71 fields are never overwritten"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -ne 0 ] && { echo "[r71_field] FATAL: inputs incomplete or outputs present -- nothing run"; echo "[r71_field] failures: ${FAILED}"; exit 1; }

$PY_FB evaluation/r71_remap_field.py --selftest || { echo "[r71_field] FAILED selftest"; echo "[r71_field] failures: 1"; exit 1; }
RC=0
$PY_FB evaluation/r71_remap_field.py --d_root "$D" --out_root "$OUT" --arms "${ARMS[@]}" || RC=1

# ---- post-run guard: one npz and one .npy start map per manifest pair in each arm dir
N_MANIFEST=$($PY_FB -c "import json; print(len(json.load(open('evaluation/r43_anchor_manifest_correct.json'))))")
FAILED=$RC
for A in "${ARMS[@]}"; do
  N=$(ls "$OUT/wan_${A}_s1"/edit*/*.npz 2>/dev/null | wc -l); NN=$(ls "$OUT/wan_${A}_s1"/edit*/*.npy 2>/dev/null | wc -l)
  echo "[r71_field] wan_${A}_s1: ${N} npz, ${NN} start maps (expect ${N_MANIFEST})"
  { [ "$N" -ne "$N_MANIFEST" ] || [ "$NN" -ne "$N_MANIFEST" ]; } && FAILED=$((FAILED + 1))
done
echo "[r71_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
