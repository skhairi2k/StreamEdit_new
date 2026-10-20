#!/bin/bash
# R44 absnorm -- ABSOLUTE normalisation + budget-linear rho for the TEXT-ONLY combos.
# r38_absnorm.sh with R44's roots and the r44_combos.sh selection; CPU ONLY (NumPy), no
# model, no GPU.
#
# Runs either way:
#   sbatch slurm_scripts/five_bench/r44_absnorm.sh [COMBO ...]   # CPU node / dependency chain
#   bash   slurm_scripts/five_bench/r44_absnorm.sh [COMBO ...]   # locally
# With no arguments it processes the selected combos (evaluation/r44_selected_combos.txt);
# explicit COMBO arguments override the selection (each must be one of the 8 R44 combos).
#
# SAME CONSTANTS AS R35/R38, NOT RECALIBRATED FOR TEXT ONLY (R44 Decisions): div_max
# 0.65 dino_patch / 1.0 lpips, div_min 0, chosen by the combo's metric; budget-linear rho
# with tau_min=2 tau_max=50 --step 15 --flow_shift 1.0 -- the STAGE-3 schedule, which stays
# at 15 steps whatever the draft N. How far the text-only raw d moves against these caps
# is what step `divstats` reports; this script does not adapt them.
#
# Caps passed explicitly so they cannot drift with r31_div_abs.py's DEFAULT_DIV_MAX.
# `--arms $COMBO` is a directory-name selector only (r31_div_abs.py does not validate it):
# stage 2 wrote r44_div/{combo}/edit{T}/*.npz keyed by combo.
#
# This is `stage3`'s `wait_for` dependency: stage 3 reads these trees, it does not rebuild
# rho itself.
#
# R44_DIV_ROOT / R44_DIV_ABS_ROOT / R44_RHO_ROOT override the three roots -- for testing
# only.
#
# No `set -e` (conda activation hooks); failures tracked via FAILED.
#SBATCH --job-name=r44_absnorm
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r44_absnorm_%j.out
#SBATCH --error=logs/r44_absnorm_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DIV_ROOT=${R44_DIV_ROOT:-~/Data/dataggen/outputs/five_bench/r44_div}
DIV_ABS_ROOT=${R44_DIV_ABS_ROOT:-~/Data/dataggen/outputs/five_bench/r44_div_abs}
RHO_ROOT=${R44_RHO_ROOT:-~/Data/dataggen/outputs/five_bench/r44_rho}

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
if [ "$#" -gt 0 ]; then
  COMBOS=("$@")
else
  r44_load || exit 1
  COMBOS=("${R44_COMBOS[@]}")
fi
declare -A DIV_MAX=()
for C in "${COMBOS[@]}"; do
  r44_parse "$C" || exit 1           # -> METRIC
  case "$METRIC" in
    dino)  DIV_MAX[$C]=0.65 ;;
    lpips) DIV_MAX[$C]=1.0  ;;
  esac
done

# A batch job starts with none of the interactive shell's state. .bashrc auto-activates
# streamgve (the R2 env bug), hence deactivate before activate.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve
mkdir -p logs "$DIV_ABS_ROOT" "$RHO_ROOT"

echo "[r44_absnorm] job=${SLURM_JOB_ID:-local} node=$(hostname) combos=${COMBOS[*]}"
echo "[r44_absnorm] div=${DIV_ROOT} -> div_abs=${DIV_ABS_ROOT}, rho=${RHO_ROOT}"

FAILED=0
for COMBO in "${COMBOS[@]}"; do
  CAP=${DIV_MAX[$COMBO]}
  echo "=== [r44_absnorm] ${COMBO}  (div_max=${CAP}) ==="

  # ---- preflight: stage 2's output for this combo must be complete before normalising.
  N_TOTAL=0
  BAD=0
  for T in 1 2 3 4 5 6; do
    NF=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + NF))
    [ "$NF" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r44_absnorm] ${COMBO}: ${N_TOTAL} stage-2 npz found (expect 419)"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r44_absnorm] FATAL: ${COMBO} stage-2 output incomplete under ${DIV_ROOT}/${COMBO}"
    echo "[r44_absnorm]   -- run /run-step R44 stage2 first"
    FAILED=$((FAILED+1)); continue
  fi

  python evaluation/r31_div_abs.py \
    --arms "$COMBO" \
    --div_max "${COMBO}=${CAP}" \
    --div_min "${COMBO}=0" \
    --div_root "$DIV_ROOT" \
    -o "$DIV_ABS_ROOT" \
    || { echo "[r44_absnorm] FAILED div_abs ${COMBO}"; FAILED=$((FAILED+1)); continue; }

  python evaluation/r31_rho_map.py \
    --div_root "$DIV_ABS_ROOT/$COMBO" \
    -o "$RHO_ROOT/$COMBO" \
    --tau_min 2 --tau_max 50 \
    --step 15 --flow_shift 1.0 \
    || { echo "[r44_absnorm] FAILED rho_map ${COMBO}"; FAILED=$((FAILED+1)); continue; }
done

echo
echo "=== [r44_absnorm] post-run guard: exact per-type breakdown, both trees ==="
for COMBO in "${COMBOS[@]}"; do
  for ROOT_NAME in "div_abs:$DIV_ABS_ROOT" "rho:$RHO_ROOT"; do
    LABEL=${ROOT_NAME%%:*}; ROOT=${ROOT_NAME#*:}
    N_TOTAL=0
    BAD=0
    for T in 1 2 3 4 5 6; do
      NF=$(ls "$ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
      N_TOTAL=$((N_TOTAL + NF))
      [ "$NF" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
    done
    echo "[r44_absnorm] ${COMBO} ${LABEL}: ${N_TOTAL} npz total (expect 419), mismatched types=${BAD}"
    if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi
  done
done

echo "[r44_absnorm] failures: ${FAILED}"
exit $(( FAILED > 0 ))
