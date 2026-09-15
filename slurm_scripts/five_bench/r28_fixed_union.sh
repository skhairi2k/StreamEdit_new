#!/bin/bash
# R28 step 2 -- rebuild the FIXED cross-arm union from every arm's dumped target mask.
#
# Pure reduction (evaluation/r28_fixed_union.py): no grounding, no models, no GPU. Reads
# every arm dir under $MASK_ROOT (auto-discovered, so it needs no arm list of its own --
# see discover_arms()) and ORs their M_tgt with M_src into one shared background region
# per (edit_type, video), written to $MASK_ROOT/_fixed_union/.
#
# Wrapped as its own sbatch job (rather than run inline / locally) so it can sit in a
# dependency chain between r28_dump.sh and r28_eval.sh: r28_eval.sh's --fixed_union_dir
# is not valid until EVERY arm has been dumped AND this reduction has re-run over all of
# them, so both must be enforced as `afterok` job dependencies, not by launching commands
# in sequence and hoping the timing works out. Submit as:
#
#   sbatch --dependency=afterok:<r28_dump jobid> slurm_scripts/five_bench/r28_fixed_union.sh
#
# and then chain r28_eval.sh off THIS job's id the same way.
#SBATCH --job-name=r28_fixed_union
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r28_fixed_union_%j.out
#SBATCH --error=logs/r28_fixed_union_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Same interpreter as r28_dump.sh/r28_eval.sh -- absolute path, no conda activation
# stacking possible. Pure numpy/PIL, so no GPU-specific packages are needed here, but
# using the same env keeps this a one-environment pipeline end to end.
PY=~/anaconda3/envs/streamgve/bin/python

DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
MASK_ROOT=/projects/dataggen/outputs/five_bench/r28_tgt_masks
CASES=evaluation/cases.json

mkdir -p logs

echo "[r28_fixed_union] node=$(hostname) job=${SLURM_JOB_ID}"
N_ARMS=$(find "$MASK_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name '_fixed_union' | wc -l)
echo "[r28_fixed_union] ${N_ARMS} arm dirs under ${MASK_ROOT} (expected 54)"
if [ "$N_ARMS" -ne 54 ]; then
  echo "[r28_fixed_union] ABORT -- expected 54 arm dirs, found ${N_ARMS}."
  echo "[r28_fixed_union]   A fixed union built from the wrong arm set is exactly the"
  echo "[r28_fixed_union]   failure r28_fixed_union.py's arm-set check exists to catch"
  echo "[r28_fixed_union]   downstream, but catching it HERE avoids a wasted r28_eval.sh run."
  exit 1
fi

$PY evaluation/r28_fixed_union.py \
  --mask_root "$MASK_ROOT" \
  --data_root "$DATA_ROOT" \
  --cases "$CASES" \
  -o "$MASK_ROOT/_fixed_union"
RC=$?

echo "[r28_fixed_union] exit=${RC}"
if [ "$RC" -ne 0 ]; then
  echo "[r28_fixed_union] FAILED (rc=${RC})"
  exit 1
fi
echo "[r28_fixed_union] OK -> $MASK_ROOT/_fixed_union"
exit 0
