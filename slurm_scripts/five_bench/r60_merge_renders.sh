#!/bin/bash
# R60 merge-renders (CPU, pinned to nodecpu12): one 415-pair tree per config in $D/r60_correct/{config}/edit{T}/{video}.
#
# Usage: sbatch slurm_scripts/five_bench/r60_merge_renders.sh            (after render-max and render-base finished)
# Runs evaluation/r60_merge_renders.py: new renders (190 regen pairs of the 12 R44 configs, all 415 of n20_rho{3,6,8,20} and
# max_novp) are symlinked; R44's renders of the 225 unchanged pairs are symlinked from /home or COPIED from /store.
# nodecpu12 is pinned because it mounts /store (the /store mount is per node: missing on nodemm06, node50). 4 cpus: the copies
# are I/O bound. Idempotent -- rerun after a failure; clips already merged are left alone.
#SBATCH --job-name=r60_merge
#SBATCH --partition=CPU
#SBATCH --nodelist=nodecpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r60_merge_renders_%j.out
#SBATCH --error=logs/r60_merge_renders_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
R44_STORE=/store/PERSO/skhairi/Outputs/five_bench/r44_rho_sweep

cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
echo "[r60_merge_renders] job=${SLURM_JOB_ID:-local} node=$(hostname)"

# fail fast on a node without the /store mount: the reused clips would all be reported missing one by one
if ! ls "$R44_STORE" > /dev/null 2>&1; then
  echo "[r60_merge_renders] FATAL: ${R44_STORE} is not readable on $(hostname) (/store is mounted on gpu-gw, nodecpu12, node39)"
  echo "[r60_merge_renders] failures: 1"
  exit 1
fi

$PY_FB -I evaluation/r60_merge_renders.py
RC=$?
echo "[r60_merge_renders] exit ${RC}"
exit $RC
