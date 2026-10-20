#!/bin/bash
# R41 rhomap -- the BINARY rho trees, one per arm (combo x theta), from the chosen draft's
# absolute-normalised fields. CPU ONLY (NumPy); no model, no GPU.
#
# Runs either way:
#   sbatch slurm_scripts/five_bench/r41_rhomap.sh     # CPU node
#   bash   slurm_scripts/five_bench/r41_rhomap.sh     # locally
#
# Everything N-dependent comes from r41_config.sh (sourced below): it refuses to run until
# gate-draft has set DRAFT_N, and its banner is the first line of the log. The fields are
# REUSED as-is (R35's r35_div_abs at N=15, R38's r38_div_abs at N=5) -- no r31_div_abs.py
# call, so binary and continuous arms share draft, divergence network AND div_max.
#
# One r31_rho_map.py call per arm:
#   --mapping binary --thresh <theta> --tau_min 2 --tau_max 50 --step 15 --flow_shift 1.0
# tau_min / tau_max / step / flow_shift are the continuous arm's own values (r35_absnorm.sh,
# r38_absnorm.sh); --step is STAGE 3's step count (15 at both N), recorded in every npz so
# r31_stage3.py can refuse a field built for another schedule.
#
# No `set -e` -- matches every other five_bench script's convention (strict mode interacts
# badly with conda's activation hooks); failures are counted explicitly instead.
#SBATCH --job-name=r41_rhomap
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r41_rhomap_%j.out
#SBATCH --error=logs/r41_rhomap_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier

# A batch job starts with none of the interactive shell's state: run from the repo (the
# python and config paths are relative) and activate the env the NumPy tools were verified
# in. .bashrc auto-activates streamgve (the R2 env bug), hence deactivate before activate.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve
mkdir -p logs

# exits here if DRAFT_N is unset; R41_CONFIG is for synthetic tests only (see r41_config.sh)
source "${R41_CONFIG:-slurm_scripts/five_bench/r41_config.sh}"
echo "[r41_rhomap] job=${SLURM_JOB_ID:-local} node=$(hostname) -> ${RHO_ROOT}"

# count_npz <dir>: prints "<total> <n_mismatched_types>" against the 100/100/100/100/9/10 split
count_npz () {
  local total=0 bad=0 n T
  for T in 1 2 3 4 5 6; do
    n=$(ls "$1/edit${T}"/*.npz 2>/dev/null | wc -l)
    total=$((total + n))
    [ "$n" -ne "${EXPECT[$T]}" ] && bad=$((bad+1))
  done
  echo "$total $bad"
}

# ---- preflight: the chosen draft's fields must be complete for every combo.
FAILED=0
for C in "${COMBOS[@]}"; do
  read -r N_TOTAL BAD <<< "$(count_npz "$DIV_ABS_ROOT/${SRC_NAME[$C]}")"
  echo "[r41_rhomap] fields ${C} (${SRC_NAME[$C]}): ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi
done
if [ "$FAILED" -ne 0 ]; then
  echo "[r41_rhomap] FATAL: fields incomplete under ${DIV_ABS_ROOT} -- nothing built"
  exit 1
fi

mkdir -p "$RHO_ROOT"
for ARM in "${ARMS[@]}"; do
  C=${ARM_COMBO[$ARM]}; TH=${ARM_THETA[$ARM]}
  echo "=== [r41_rhomap] ${ARM}  (combo=${C} <- ${SRC_NAME[$C]}, theta=${TH}) ==="
  python evaluation/r31_rho_map.py \
    --div_root "$DIV_ABS_ROOT/${SRC_NAME[$C]}" \
    -o "$RHO_ROOT/$ARM" \
    --mapping binary --thresh "$TH" \
    --tau_min 2 --tau_max 50 \
    --step 15 --flow_shift 1.0 \
    || { echo "[r41_rhomap] FAILED ${ARM}"; FAILED=$((FAILED+1)); }
done

echo
echo "=== [r41_rhomap] post-run guard: exact per-type breakdown per arm ==="
for ARM in "${ARMS[@]}"; do
  read -r N_TOTAL BAD <<< "$(count_npz "$RHO_ROOT/$ARM")"
  echo "[r41_rhomap] ${ARM}: ${N_TOTAL} npz total (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi
done

echo "[r41_rhomap] failures: ${FAILED}"
exit $(( FAILED > 0 ))
