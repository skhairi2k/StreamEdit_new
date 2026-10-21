#!/bin/bash
# R49 stage 2b (CPU): inverse-response per-token controls.
#
# Usage: sbatch slurm_scripts/five_bench/r49_field.sh [HOSTS] [MODE]
#   HOSTS: "se wan" (default), se or wan  -- only those hosts' fields are built and checked (one host per platform:
#          StreamEdit on the lab cluster, Wan-Edit on Jean Zay)
#   MODE : subset (default) = the 18 clips of evaluation/r49_cases.json (r49_chain.sh, afterok on r49_div.sh)
#          full             = the 419 pairs of evaluation/r7_anchor_manifest.json (R49 full stage: r49_se_full_chain.sh with
#                             `se full`, Jean Zay `wan full`)
#   No argument = the original behaviour (both hosts, subset).
# Runs `r49_field.py --selftest`, then the field.
#   subset: attention from r49_se_attn / r49_wan_draft/attn, divergence from r49_div, out r49_field; then R48's field for
#           the reference arm dino_SD on the NEW clip only (evaluation/r49_new_clip.json) from the same r49 inputs, into
#           $D/r48_field -- the 17 other clips already have it from R48.
#   full  : attention from r48_se_attn / r48_wan_draft/attn, divergence from r48_div (the R48 full-mode drafts and
#           divergence), out r49_field_full; no reference arm.
# Output: $D/r49_field{,_full}/{se_max | wan_max_s9, wan_max_s5}/edit{T}/{video}.npz (+ .npy start maps for wan)
#         subset only: $D/r48_field/{host}_dino_SD/edit3/0004_car-roundabout.npz (+ .npy for wan)
#SBATCH --job-name=r49_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r49_field_%j.out
#SBATCH --error=logs/r49_field_%j.err

HOSTS=${1:-se wan}
MODE=${2:-subset}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
NEW_CLIP=evaluation/r49_new_clip.json
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
read -r -a HOST_LIST <<< "$HOSTS"
for H in "${HOST_LIST[@]}"; do
  case "$H" in se|wan) ;; *) echo "[r49_field] usage: r49_field.sh [\"se wan\"|se|wan] [subset|full] -- bad host '$H'"; exit 1 ;; esac
done
case "$MODE" in
  subset) CASES=evaluation/r49_cases.json; OUT=$D/r49_field
          ROOTS=(--attn_se "$D/r49_se_attn" --attn_wan "$D/r49_wan_draft/attn" --div_root "$D/r49_div") ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; OUT=$D/r49_field_full
          ROOTS=(--attn_se "$D/r48_se_attn" --attn_wan "$D/r48_wan_draft/attn" --div_root "$D/r48_div") ;;
  *) echo "[r49_field] usage: r49_field.sh [\"se wan\"|se|wan] [subset|full] -- bad mode '$MODE'"; exit 1 ;;
esac

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r49_field] job=${SLURM_JOB_ID:-local} node=$(hostname) hosts=${HOST_LIST[*]} mode=${MODE} cases=${CASES} out=${OUT}"

$PY_FB evaluation/r49_field.py --selftest || { echo "[r49_field] FAILED selftest"; exit 1; }
$PY_FB evaluation/r49_field.py --hosts "${HOST_LIST[@]}" --cases "$CASES" "${ROOTS[@]}" --out_root "$OUT"
RC=$?
RC_REF=0
if [ "$MODE" = subset ]; then
  $PY_FB evaluation/r48_field.py --mode subset --hosts "${HOST_LIST[@]}" --cases "$NEW_CLIP" --arms dino_SD --out_root "$D/r48_field" \
    "${ROOTS[@]}"
  RC_REF=$?
fi

# ---- post-run guard: one npz per case clip in each output dir of the listed hosts (and one .npy per clip for the wan floors)
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=$(( (RC != 0) + (RC_REF != 0) ))
DIRS=()
for H in "${HOST_LIST[@]}"; do
  case "$H" in se) DIRS+=(se_max) ;; wan) DIRS+=(wan_max_s9 wan_max_s5) ;; esac
done
for DIR in "${DIRS[@]}"; do
  N=$(ls "$OUT/$DIR"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r49_field] ${DIR}: ${N} npz (expect ${N_CASES})"
  [ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
  if [[ "$DIR" == wan_* ]]; then
    NN=$(ls "$OUT/$DIR"/edit*/*.npy 2>/dev/null | wc -l)
    [ "$NN" -ne "$N_CASES" ] && { echo "[r49_field] ${DIR}: ${NN} start maps (expect ${N_CASES})"; FAILED=$((FAILED + 1)); }
  fi
done
if [ "$MODE" = subset ]; then
  for H in "${HOST_LIST[@]}"; do   # the reference field of the new clip (type 3) exists for each listed host
    [ -s "$D/r48_field/${H}_dino_SD/edit3/0004_car-roundabout.npz" ] || { echo "[r49_field] MISSING reference field ${H}_dino_SD for 0004_car-roundabout"; FAILED=$((FAILED + 1)); }
  done
  if [[ " ${HOST_LIST[*]} " == *" wan "* ]]; then
    [ -s "$D/r48_field/wan_dino_SD/edit3/0004_car-roundabout.npy" ] || { echo "[r49_field] MISSING wan reference start map"; FAILED=$((FAILED + 1)); }
  fi
fi
echo "[r49_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
