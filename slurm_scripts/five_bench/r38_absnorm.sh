#!/bin/bash
# R38 absnorm -- ABSOLUTE normalisation + budget-linear rho for the N=5 COMBOS.
# r35_absnorm.sh with R38's roots and combo names; CPU ONLY (NumPy), no model, no GPU.
#
# Runs either way:
#   sbatch slurm_scripts/five_bench/r38_absnorm.sh [COMBO ...]   # CPU node / dependency chain
#   bash   slurm_scripts/five_bench/r38_absnorm.sh [COMBO ...]   # locally
# With no arguments it processes all four N=5 combos. Pass a subset when gate-lowN dropped
# the tgt09 combos (e.g. `dino_n5_unblended lpips_n5_unblended`) -- the all-four default
# would fail preflight on the dropped combos and exit non-zero.
#
# SAME CONSTANTS AS R35 FOR EVERY N (R38 Decisions): div_max 0.65 dino_patch / 1.0 lpips,
# div_min 0, chosen by the combo PREFIX (dino_* / lpips_*); budget-linear rho with
# tau_min=2 tau_max=50 --step 15 --flow_shift 1.0 -- the STAGE-3 schedule, which stays at
# 15 steps; only the stage-1 draft is N=5. Whether the N=5 draft's raw d still fits these
# caps is what step `divstats` reports; this script does not adapt them.
#
# Caps passed explicitly so they cannot drift with r31_div_abs.py's DEFAULT_DIV_MAX.
# `--arms $COMBO` is a directory-name selector only (r31_div_abs.py does not validate it):
# stage 2 wrote r38_div/{combo}/edit{T}/*.npz keyed by combo.
#
# This is `stage3`'s `wait_for` dependency: stage 3 reads these trees, it does not rebuild
# rho itself.
#
# R38_DIV_ROOT / R38_DIV_ABS_ROOT / R38_RHO_ROOT override the three roots -- for testing
# on synthetic data only.
#
# No `set -e` (conda activation hooks); failures tracked via FAILED.
#SBATCH --job-name=r38_absnorm
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r38_absnorm_%j.out
#SBATCH --error=logs/r38_absnorm_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DIV_ROOT=${R38_DIV_ROOT:-~/Data/dataggen/outputs/five_bench/r38_div}
DIV_ABS_ROOT=${R38_DIV_ABS_ROOT:-~/Data/dataggen/outputs/five_bench/r38_div_abs}
RHO_ROOT=${R38_RHO_ROOT:-~/Data/dataggen/outputs/five_bench/r38_rho}

ALL_COMBOS=(dino_n5_unblended dino_n5_tgt09 lpips_n5_unblended lpips_n5_tgt09)
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

if [ "$#" -gt 0 ]; then COMBOS=("$@"); else COMBOS=("${ALL_COMBOS[@]}"); fi
declare -A DIV_MAX=()
for C in "${COMBOS[@]}"; do
  case " ${ALL_COMBOS[*]} " in
    *" $C "*) ;;
    *) echo "[r38_absnorm] unknown combo '$C' (expected one of ${ALL_COMBOS[*]})"; exit 1 ;;
  esac
  case "$C" in
    dino_*)  DIV_MAX[$C]=0.65 ;;
    lpips_*) DIV_MAX[$C]=1.0  ;;
  esac
done

# A batch job starts with none of the interactive shell's state. .bashrc auto-activates
# streamgve (the R2 env bug), hence deactivate before activate.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve
mkdir -p logs "$DIV_ABS_ROOT" "$RHO_ROOT"

echo "[r38_absnorm] job=${SLURM_JOB_ID:-local} node=$(hostname) combos=${COMBOS[*]}"
echo "[r38_absnorm] div=${DIV_ROOT} -> div_abs=${DIV_ABS_ROOT}, rho=${RHO_ROOT}"

FAILED=0
for COMBO in "${COMBOS[@]}"; do
  CAP=${DIV_MAX[$COMBO]}
  echo "=== [r38_absnorm] ${COMBO}  (div_max=${CAP}) ==="

  # ---- preflight: stage 2's output for this combo must be complete before normalising.
  N_TOTAL=0
  BAD=0
  for T in 1 2 3 4 5 6; do
    N=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + N))
    [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r38_absnorm] ${COMBO}: ${N_TOTAL} stage-2 npz found (expect 419)"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r38_absnorm] FATAL: ${COMBO} stage-2 output incomplete under ${DIV_ROOT}/${COMBO}"
    echo "[r38_absnorm]   -- run /run-step R38 stage2 first"
    FAILED=$((FAILED+1)); continue
  fi

  python evaluation/r31_div_abs.py \
    --arms "$COMBO" \
    --div_max "${COMBO}=${CAP}" \
    --div_min "${COMBO}=0" \
    --div_root "$DIV_ROOT" \
    -o "$DIV_ABS_ROOT" \
    || { echo "[r38_absnorm] FAILED div_abs ${COMBO}"; FAILED=$((FAILED+1)); continue; }

  python evaluation/r31_rho_map.py \
    --div_root "$DIV_ABS_ROOT/$COMBO" \
    -o "$RHO_ROOT/$COMBO" \
    --tau_min 2 --tau_max 50 \
    --step 15 --flow_shift 1.0 \
    || { echo "[r38_absnorm] FAILED rho_map ${COMBO}"; FAILED=$((FAILED+1)); continue; }
done

echo
echo "=== [r38_absnorm] post-run guard: exact per-type breakdown, both trees ==="
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
    echo "[r38_absnorm] ${COMBO} ${LABEL}: ${N_TOTAL} npz total (expect 419), mismatched types=${BAD}"
    if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi
  done
done

echo "[r38_absnorm] failures: ${FAILED}"
exit $(( FAILED > 0 ))
