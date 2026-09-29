#!/bin/bash
# R35 absnorm -- ABSOLUTE normalisation + budget-linear rho, per (arm, regime) COMBO.
# CPU ONLY (vectorized NumPy, a few seconds per combo-edit-type); no model, no GPU.
#
# Runs either way:
#   sbatch slurm_scripts/five_bench/r35_absnorm.sh [COMBO ...]   # CPU node, in a dependency chain
#   bash   slurm_scripts/five_bench/r35_absnorm.sh [COMBO ...]   # locally
# With no arguments it processes all four combos. Pass a subset to run one chain's arms only
# (2026-09-24: R35 runs two overnight chains, DINO and LPIPS, each
# stage 2 -> absnorm -> stage 3 via --dependency=afterok). This MUST take a subset: the
# all-four default would fail preflight on the other chain's not-yet-computed combos, exit
# non-zero, and block this chain's stage 3.
#
# Two passes per combo, both already-existing tools, no code changes needed to either:
#   1. r31_div_abs.py   -- d -> m via a FIXED div_max (1.0 lpips, 0.65 dino_patch),
#                          not R31's per-clip min/max. `--arms $COMBO` is used purely as
#                          a directory-name selector here (r31_div_abs.py does not
#                          validate it against the real ARMS list): stage 2 already wrote
#                          r35_div/{combo}/edit{T}/*.npz keyed by combo, not by bare arm.
#   2. r31_rho_map.py   -- m -> rho, budget-linear, tau_min=2 tau_max=50, --step 15 (the
#                          STAGE-3 schedule, not stage 1's) -- R30's calibrated pair,
#                          identical to r31_stage3.sh's own rho build.
#
# CAPS ARE PASSED EXPLICITLY (1.0 lpips, 0.65 dino_patch) even though they now match
# r31_div_abs.py's DEFAULT_DIV_MAX (corrected 0.6 -> 0.65 on 2026-09-24), so this run's
# caps are pinned here and cannot drift if that default ever changes again.
#
# This is `stage3`'s `wait_for` dependency (see the plan's `steps` block): stage3 reads
# the trees this script builds and does NOT rebuild rho itself.
#
# R35_DIV_ROOT / R35_DIV_ABS_ROOT / R35_RHO_ROOT override the three roots -- used only to
# test this script on synthetic data without touching the real trees.
#
# No `set -e` -- matches every other five_bench script's convention (strict mode
# interacts badly with conda's activation hooks elsewhere in this repo); failures are
# tracked explicitly via the FAILED counter instead.
#SBATCH --job-name=r35_absnorm
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r35_absnorm_%j.out
#SBATCH --error=logs/r35_absnorm_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DIV_ROOT=${R35_DIV_ROOT:-~/Data/dataggen/outputs/five_bench/r35_div}
DIV_ABS_ROOT=${R35_DIV_ABS_ROOT:-~/Data/dataggen/outputs/five_bench/r35_div_abs}
RHO_ROOT=${R35_RHO_ROOT:-~/Data/dataggen/outputs/five_bench/r35_rho}

ALL_COMBOS=(dino_unblended dino_first2 lpips_unblended lpips_first2)
declare -A DIV_MAX=([dino_unblended]=0.65 [dino_first2]=0.65 [lpips_unblended]=1.0 [lpips_first2]=1.0)
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

if [ "$#" -gt 0 ]; then COMBOS=("$@"); else COMBOS=("${ALL_COMBOS[@]}"); fi
for C in "${COMBOS[@]}"; do
  [ -n "${DIV_MAX[$C]}" ] || { echo "[r35_absnorm] unknown combo '$C' (expected one of ${ALL_COMBOS[*]})"; exit 1; }
done

# A batch job starts with none of the interactive shell's state: activate the env the
# NumPy tools were verified in, and run from the repo (the python paths are relative).
# .bashrc auto-activates streamgve (the R2 env bug), hence deactivate before activate.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve
mkdir -p logs "$DIV_ABS_ROOT" "$RHO_ROOT"

echo "[r35_absnorm] job=${SLURM_JOB_ID:-local} node=$(hostname) combos=${COMBOS[*]}"
echo "[r35_absnorm] div=${DIV_ROOT} -> div_abs=${DIV_ABS_ROOT}, rho=${RHO_ROOT}"

FAILED=0
for COMBO in "${COMBOS[@]}"; do
  CAP=${DIV_MAX[$COMBO]}
  echo "=== [r35_absnorm] ${COMBO}  (div_max=${CAP}) ==="

  # ---- preflight: stage 2's output for this combo must be complete before normalising.
  N_TOTAL=0
  BAD=0
  for T in 1 2 3 4 5 6; do
    N=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + N))
    [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r35_absnorm] ${COMBO}: ${N_TOTAL} stage-2 npz found (expect 419)"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r35_absnorm] FATAL: ${COMBO} stage-2 output incomplete under ${DIV_ROOT}/${COMBO}"
    echo "[r35_absnorm]   -- run /run-step R35 stage2 first"
    FAILED=$((FAILED+1)); continue
  fi

  python evaluation/r31_div_abs.py \
    --arms "$COMBO" \
    --div_max "${COMBO}=${CAP}" \
    --div_min "${COMBO}=0" \
    --div_root "$DIV_ROOT" \
    -o "$DIV_ABS_ROOT" \
    || { echo "[r35_absnorm] FAILED div_abs ${COMBO}"; FAILED=$((FAILED+1)); continue; }

  python evaluation/r31_rho_map.py \
    --div_root "$DIV_ABS_ROOT/$COMBO" \
    -o "$RHO_ROOT/$COMBO" \
    --tau_min 2 --tau_max 50 \
    --step 15 --flow_shift 1.0 \
    || { echo "[r35_absnorm] FAILED rho_map ${COMBO}"; FAILED=$((FAILED+1)); continue; }
done

echo
echo "=== [r35_absnorm] post-run guard: exact per-type breakdown, both trees ==="
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
    echo "[r35_absnorm] ${COMBO} ${LABEL}: ${N_TOTAL} npz total (expect 419), mismatched types=${BAD}"
    if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi
  done
done

echo "[r35_absnorm] failures: ${FAILED}"
exit $(( FAILED > 0 ))
