#!/bin/bash
# R45 absnorm -- ABSOLUTE normalisation + budget-linear rho for the SELECTED LongLive combos.
# r38_absnorm.sh with R45's roots and combo names; CPU ONLY (NumPy), no model, no GPU.
#
#   sbatch slurm_scripts/five_bench/r45_absnorm.sh            # the method-gate selection
#   sbatch slurm_scripts/five_bench/r45_absnorm.sh COMBO ...  # an explicit subset
#   bash   slurm_scripts/five_bench/r45_absnorm.sh [COMBO ...] # locally
# With no arguments it processes exactly the combos in evaluation/r45_method.json.
#
# SAME CONSTANTS AS SF (R45 Decisions, same d -> m function on both backbones): div_max 0.65
# dino_patch / 1.0 lpips, div_min 0, by combo metric; budget-linear rho with tau_min=2
# tau_max=50 --step 15 --flow_shift 1.0 (the stage-3 render grid). r31_rho_map.py is used
# unchanged: its t_next grid is the one LL's bridge applies (paper Eq. 4, verified).
#
# R45_DIV_ROOT / R45_DIV_ABS_ROOT / R45_RHO_ROOT override the three roots -- synthetic tests only.
# No `set -e` (conda activation hooks); failures tracked via FAILED.
#SBATCH --job-name=r45_absnorm
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r45_absnorm_%j.out
#SBATCH --error=logs/r45_absnorm_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DIV_ROOT=${R45_DIV_ROOT:-~/Data/dataggen/outputs/five_bench/r45_div}
DIV_ABS_ROOT=${R45_DIV_ABS_ROOT:-~/Data/dataggen/outputs/five_bench/r45_div_abs}
RHO_ROOT=${R45_RHO_ROOT:-~/Data/dataggen/outputs/five_bench/r45_rho}

ALL_COMBOS=(ll_dino_n15_unblended ll_dino_n15_tgt09 ll_lpips_n15_unblended ll_lpips_n15_tgt09
            ll_dino_n5_unblended  ll_dino_n5_tgt09  ll_lpips_n5_unblended  ll_lpips_n5_tgt09)
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# A batch job starts with none of the interactive shell's state. .bashrc auto-activates
# streamgve (the R2 env bug), hence deactivate before activate.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve

if [ "$#" -gt 0 ]; then
  COMBOS=("$@")
else
  IDX=$(python evaluation/r45_select.py stage2) || exit 1
  COMBOS=()
  for i in ${IDX//,/ }; do COMBOS+=("${ALL_COMBOS[$i]}"); done
fi

declare -A DIV_MAX=()
for C in "${COMBOS[@]}"; do
  case " ${ALL_COMBOS[*]} " in
    *" $C "*) ;;
    *) echo "[r45_absnorm] unknown combo '$C' (expected one of ${ALL_COMBOS[*]})"; exit 1 ;;
  esac
  case "$C" in
    ll_dino_*)  DIV_MAX[$C]=0.65 ;;
    ll_lpips_*) DIV_MAX[$C]=1.0  ;;
  esac
done

mkdir -p logs "$DIV_ABS_ROOT" "$RHO_ROOT"
echo "[r45_absnorm] job=${SLURM_JOB_ID:-local} node=$(hostname) combos=${COMBOS[*]}"
echo "[r45_absnorm] div=${DIV_ROOT} -> div_abs=${DIV_ABS_ROOT}, rho=${RHO_ROOT}"

FAILED=0
for COMBO in "${COMBOS[@]}"; do
  CAP=${DIV_MAX[$COMBO]}
  echo "=== [r45_absnorm] ${COMBO}  (div_max=${CAP}) ==="
  # ---- preflight: stage 2's output for this combo must be complete before normalising.
  N_TOTAL=0
  BAD=0
  for T in 1 2 3 4 5 6; do
    N=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + N))
    [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r45_absnorm] ${COMBO}: ${N_TOTAL} stage-2 npz found (expect 419)"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r45_absnorm] FATAL: ${COMBO} stage-2 output incomplete under ${DIV_ROOT}/${COMBO} -- run /run-step R45 stage2 first"
    FAILED=$((FAILED+1)); continue
  fi

  python evaluation/r31_div_abs.py \
    --arms "$COMBO" \
    --div_max "${COMBO}=${CAP}" \
    --div_min "${COMBO}=0" \
    --div_root "$DIV_ROOT" \
    -o "$DIV_ABS_ROOT" \
    || { echo "[r45_absnorm] FAILED div_abs ${COMBO}"; FAILED=$((FAILED+1)); continue; }

  python evaluation/r31_rho_map.py \
    --div_root "$DIV_ABS_ROOT/$COMBO" \
    -o "$RHO_ROOT/$COMBO" \
    --tau_min 2 --tau_max 50 \
    --step 15 --flow_shift 1.0 \
    || { echo "[r45_absnorm] FAILED rho_map ${COMBO}"; FAILED=$((FAILED+1)); continue; }
done

echo
echo "=== [r45_absnorm] post-run guard: exact per-type breakdown, both trees ==="
for COMBO in "${COMBOS[@]}"; do
  for ROOT_NAME in "div_abs:$DIV_ABS_ROOT" "rho:$RHO_ROOT"; do
    LABEL=${ROOT_NAME%%:*}; ROOT=${ROOT_NAME#*:}
    N_TOTAL=0
    BAD=0
    for T in 1 2 3 4 5 6; do
      N=$(ls "$ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
      N_TOTAL=$((N_TOTAL + N))
      [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
    done
    echo "[r45_absnorm] ${COMBO} ${LABEL}: ${N_TOTAL} npz total (expect 419), mismatched types=${BAD}"
    if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi
  done
done
echo "[r45_absnorm] failures: ${FAILED}"
exit $(( FAILED > 0 ))
