#!/bin/bash
# R62 stage 2 (CPU): R51's demand-component ablation fields on FiVE-Bench-Correct, from R59's (StreamEdit) / R58's (Wan-Edit) Correct inputs.
#
# Usage: sbatch slurm_scripts/five_bench/r62_field.sh se     # lab: A1 nodiv, A2 dino, A3 lpips, A4 nosupp, 415 pairs   (r62_se_chain.sh)
#        sbatch slurm_scripts/five_bench/r62_field.sh wan    # Jean Zay: A4 nosupp at s_min 5, R58's 190 changed pairs   (r62_wan_chain.sh)
# Runs `r51_field.py --selftest`, then the field. r51_field.py first recomputes A0 (R49's own q = S * max) for every pair and requires
# bit-equality with R59's se_max `rho` (se) / R58's wan_max_s5 start maps (wan); on any mismatch it writes nothing and exits 1.
# Inputs (no new draft):
#   se : attention $D/r59_se_attn, divergence $D/r59_div/se_{dino,lpips}, gate $D/r59_field_full/se_max, cases r43_anchor_manifest_correct.json (415)
#   wan: attention $D/r58_wan_draft/attn, divergence $D/r58_div/wan_{dino,lpips}, gate $D/r58_field/wan_max_s5, cases r58_cases_maxs5.json (190)
# Output: $D/r62_field/{se_nodiv, se_dino, se_lpips, se_nosupp | wan_nosupp_s5}/edit{T}/{video}.npz (+ .npy start maps for wan)
#SBATCH --job-name=r62_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r62_field_%j.out
#SBATCH --error=logs/r62_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
HOST=${1:-}
OUT=$D/r62_field
case "$HOST" in
  se)  CASES=evaluation/r43_anchor_manifest_correct.json; ARMS="nodiv dino lpips nosupp"
       ATTN=$D/r59_se_attn; DIV=$D/r59_div; GATE=$D/r59_field_full; PRE=se_;  POST="" ;;
  wan) CASES=evaluation/r58_cases_maxs5.json;             ARMS="nosupp"
       ATTN=$D/r58_wan_draft/attn; DIV=$D/r58_div; GATE=$D/r58_field; PRE=wan_; POST=_s5 ;;
  *) echo "[r62_field] usage: r62_field.sh se|wan"; exit 1 ;;
esac
DIRS=(); for A in $ARMS; do DIRS+=("${PRE}${A}${POST}"); done
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r62_field] job=${SLURM_JOB_ID:-local} node=$(hostname) host=${HOST} arms=${ARMS} cases=${CASES} attn=${ATTN} div=${DIV} gate=${GATE} out=${OUT}"

# ---- preflight: the upstream inputs must be complete for every case before the A0 gate runs
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=0
for X in "$ATTN" "$DIV/${PRE}dino" "$DIV/${PRE}lpips"; do
  N=$(ls "$X"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r62_field] ${X}: ${N} npz (need >= ${N_CASES})"
  [ "$N" -lt "$N_CASES" ] && FAILED=$((FAILED + 1))
done
[ "$FAILED" -ne 0 ] && { echo "[r62_field] FATAL: upstream inputs incomplete -- is the ${HOST} stage-1 / divergence chain finished?"; echo "[r62_field] failures: ${FAILED}"; exit 1; }

$PY_FB evaluation/r51_field.py --selftest || { echo "[r62_field] FAILED selftest"; echo "[r62_field] failures: 1"; exit 1; }
$PY_FB evaluation/r51_field.py --host "$HOST" --arms $ARMS --cases "$CASES" --attn_root "$ATTN" --div_root "$DIV" --gate_ref "$GATE" --out_root "$OUT"
RC=$?

# ---- post-run guard: one npz per case in each arm dir (and one .npy start map per case for wan)
FAILED=$(( RC != 0 ))
for DIR in "${DIRS[@]}"; do
  N=$(ls "$OUT/$DIR"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r62_field] ${DIR}: ${N} npz (expect ${N_CASES})"
  [ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
  if [[ "$DIR" == wan_* ]]; then
    NN=$(ls "$OUT/$DIR"/edit*/*.npy 2>/dev/null | wc -l)
    echo "[r62_field] ${DIR}: ${NN} start maps (expect ${N_CASES})"
    [ "$NN" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
  fi
done
echo "[r62_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
