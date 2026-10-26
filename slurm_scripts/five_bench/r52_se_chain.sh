#!/bin/bash
# R52 StreamEdit track (lab cluster): the B1 arm (rho = 2 + 48 q on R49's q) on the 419 pairs, chained with afterok dependencies.
# Reuses R51's scripts through their optional arguments (mapping / arms / roots / prefix); R51's behaviour is the default of each.
#
#   field    r51_field.sh se linear max $D/r52_field_full    A0 gate + B1 field, rho = 2 + 48 q   -> r52_field_full/se_max_linrho     (no dependency)
#     -> render  r51_se_stage3.sh max_linrho ... r52 (--array=0)   B1 render, step 15            -> r52_se_arms/r52_max_linrho_full   afterok field
#        -> eval   r51_eval.sh r52_se (--array=0)        harness, FiVE-Acc, CLIP-D                                                       afterok render
#        -> motion r51_motion.sh r52_motion (--array=0)  motion fidelity (CoTracker alone)                                               afterok render
# 4 queue entries, one task each. Scoring (r52_score.py) stays a local step and needs R51's motion units for A0 and the official render.
# Writes "<stage> <id>" lines to logs/r52_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists
# (a second chain would render everything twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r52_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
SB=slurm_scripts/five_bench
JOBS=logs/r52_se_chain_jobs.txt
FIELD_ROOT=$D/r52_field_full
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r52_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r52_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r51_field.sh $SB/r51_se_stage3.sh $SB/r51_eval.sh $SB/r51_motion.sh evaluation/r51_field.py evaluation/r51_units.py; do
  [ -f "$F" ] || { echo "[r52_se_chain] FATAL: $F missing"; exit 1; }
done
# the R51 scripts must carry the R52 options (a stale checkout would silently run R51's defaults)
grep -q -- '--mapping' evaluation/r51_field.py && grep -q 'MAPPING=' $SB/r51_field.sh && grep -q 'FIELD_ROOT=' $SB/r51_se_stage3.sh \
  && grep -q 'r52_se' $SB/r51_eval.sh && grep -q 'r52_motion' $SB/r51_motion.sh && grep -q 'r52_se' evaluation/r51_units.py \
  || { echo "[r52_se_chain] FATAL: the R51 scripts in this checkout lack the R52 options (git pull / build-step se-chain R52?)"; exit 1; }

# ---- inputs: R49's StreamEdit full-stage inputs must be complete (nothing here waits on them)
FAILED=0
count() { ls "$1"/edit*/*.npz 2>/dev/null | wc -l; }
for X in "$D/r48_se_attn" "$D/r48_div/se_dino" "$D/r48_div/se_lpips" "$D/r49_field_full/se_max"; do
  N=$(count "$X"); echo "[r52_se_chain] ${X}: ${N} npz (need 419)"
  [ "$N" -ne 419 ] && FAILED=$((FAILED + 1))
done
[ "$FAILED" -ne 0 ] && { echo "[r52_se_chain] REFUSE: R49's StreamEdit full-stage inputs are incomplete"; exit 1; }
echo "[r52_se_chain] queue now: $(squeue -u "$USER" -h 2>/dev/null | wc -l) entries of yours (4 to submit)"

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r52_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r52_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r52_se_chain] ${stage}: job ${ID}"
}

submit field  $SB/r51_field.sh se linear max "$FIELD_ROOT";                                          FIELD=$ID
submit render --dependency=afterok:"$FIELD"  --array=0 $SB/r51_se_stage3.sh max_linrho "$FIELD_ROOT" r52;   RENDER=$ID
submit eval   --dependency=afterok:"$RENDER" --array=0 $SB/r51_eval.sh r52_se;                       EVAL=$ID
submit motion --dependency=afterok:"$RENDER" --array=0 $SB/r51_motion.sh r52_motion;                 MOTION=$ID

echo "[r52_se_chain] submitted: field ${FIELD} -> render ${RENDER} -> eval ${EVAL} + motion ${MOTION} (ids in ${JOBS})"
