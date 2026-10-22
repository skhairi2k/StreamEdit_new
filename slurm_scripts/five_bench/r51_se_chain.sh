#!/bin/bash
# R51 StreamEdit track (lab cluster): submit the demand-component ablation chain with afterok dependencies, in one go.
#
#   field            r51_field.sh se                         A0 gate + A1-A4 fields              -> r51_field_full/se_*      (no dependency)
#     -> render      r51_se_stage3.sh --array=0-3            nodiv, dino, lpips, nosupp           -> r51_se_arms              afterok field
#        -> eval     r51_eval.sh se --array=0-3              harness, FiVE-Acc, CLIP-D per arm                                afterok render
#        -> motion   r51_motion.sh --array=0-4               motion fidelity, A1-A4 + A0 (R49 se_max)                         afterok render
#   motion_official  r51_motion.sh --array=5 --nodelist=node39   official rho = 2 (r36_rho2_vp, on /store)                    (no dependency)
# 5 queue entries. Scoring (r51_score.py --host se) and grids stay local steps.
# Writes "<stage> <id>" lines to logs/r51_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists
# (a second chain would render everything twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r51_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
SB=slurm_scripts/five_bench
JOBS=logs/r51_se_chain_jobs.txt
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r51_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r51_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r51_field.sh $SB/r51_se_stage3.sh $SB/r51_eval.sh $SB/r51_motion.sh evaluation/r51_field.py evaluation/r51_units.py; do
  [ -f "$F" ] || { echo "[r51_se_chain] FATAL: $F missing"; exit 1; }
done

# ---- inputs: R49's StreamEdit full stage must be complete (nothing here waits on it)
FAILED=0
count() { ls "$1"/edit*/*.npz 2>/dev/null | wc -l; }
for X in "$D/r48_se_attn" "$D/r48_div/se_dino" "$D/r48_div/se_lpips" "$D/r49_field_full/se_max"; do
  N=$(count "$X"); echo "[r51_se_chain] ${X}: ${N} npz (need 419)"
  [ "$N" -ne 419 ] && FAILED=$((FAILED + 1))
done
N=$(ls -d "$D"/r49_se_arms/r49_max_full/step14/edit*/*/ 2>/dev/null | wc -l)
echo "[r51_se_chain] A0 renders r49_max_full: ${N} clip dirs (need 419)"
[ "$N" -ne 419 ] && FAILED=$((FAILED + 1))
[ "$FAILED" -ne 0 ] && { echo "[r51_se_chain] REFUSE: R49's StreamEdit full-stage inputs are incomplete"; exit 1; }
echo "[r51_se_chain] queue now: $(squeue -u "$USER" -h 2>/dev/null | wc -l) entries of yours (5 to submit)"

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r51_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r51_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r51_se_chain] ${stage}: job ${ID}"
}

submit field           $SB/r51_field.sh se;                                                          FIELD=$ID
submit render          --dependency=afterok:"$FIELD"  --array=0-3 $SB/r51_se_stage3.sh;              RENDER=$ID
submit eval            --dependency=afterok:"$RENDER" --array=0-3 $SB/r51_eval.sh se;                EVAL=$ID
submit motion          --dependency=afterok:"$RENDER" --array=0-4 $SB/r51_motion.sh;                 MOTION=$ID
submit motion_official --array=5 --nodelist=node39 $SB/r51_motion.sh;                                MOFF=$ID

echo "[r51_se_chain] submitted: field ${FIELD} -> render ${RENDER} -> eval ${EVAL} + motion ${MOTION} | motion_official ${MOFF} (ids in ${JOBS})"
