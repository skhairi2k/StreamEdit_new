#!/bin/bash
# R54 StreamEdit track (lab cluster): the draft-free arm (R40's flow-warped anchor maps, max of the two capped maps, R49's exposure inverse)
# on the 419 pairs, VP, chained with afterok dependencies. Reuses R51's render / eval / motion scripts through their optional arguments.
#
#   field    r54_field.sh                                  R40 maps -> q = max -> rho (nodecpu12: R40's maps are on /store)  -> r54_field_full/se_warp_max
#     -> render  r51_se_stage3.sh warp_max ... r54 (--array=0)   15-step render, VP + anchors   -> r54_se_arms/r54_warp_max_full         afterok field
#        -> eval   r51_eval.sh r54_se (--array=0)         harness, FiVE-Acc, CLIP-D                                                     afterok render
#        -> motion r51_motion.sh r54_motion (--array=0)   motion fidelity (CoTracker alone)                                             afterok render
# 4 queue entries = 4 array tasks. NFE 15 (no draft): read it against A4 (R51, NFE 20) and, for equal NFE, the R36 N=15 curve.
# The per-user submit cap (30 tasks, seen as QOSMaxSubmitJobPerUserLimit in R44/R45) is checked BEFORE anything is submitted: the driver REFUSES
# when the queue plus 4 would exceed it (R54_QUEUE_CAP=<n> overrides the cap value). Scoring (r54_score.py) stays a local step and needs
# R51's A4 CSVs and motion units. Writes "<stage> <id>" lines to logs/r54_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if
# that file exists (a second chain would render everything twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r54_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
SB=slurm_scripts/five_bench
JOBS=logs/r54_se_chain_jobs.txt
N_TASKS=4
CAP=${R54_QUEUE_CAP:-30}
FIELD_ROOT=$D/r54_field_full
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r54_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r54_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r54_field.sh $SB/r51_se_stage3.sh $SB/r51_eval.sh $SB/r51_motion.sh evaluation/r54_field.py evaluation/r51_units.py; do
  [ -f "$F" ] || { echo "[r54_se_chain] FATAL: $F missing"; exit 1; }
done
# the shared R51 scripts must know the R54 groups (a stale checkout would reject them only once the jobs run)
grep -q 'r54_se' $SB/r51_eval.sh && grep -q 'r54_motion' $SB/r51_motion.sh && grep -q 'r54_se' evaluation/r51_units.py \
  || { echo "[r54_se_chain] FATAL: r51_eval.sh / r51_motion.sh / r51_units.py in this checkout lack the R54 groups (git pull / build-step se-chain R54?)"; exit 1; }

# ---- inputs R54 reads on a node that may differ from this one: only R49's field (on /home) can be counted here; the field job itself
# checks R40's maps on nodecpu12 and fails fast, before writing anything, when /store is missing or the maps are incomplete
N=$(ls "$D"/r49_field_full/se_max/edit*/*.npz 2>/dev/null | wc -l)
echo "[r54_se_chain] ${D}/r49_field_full/se_max: ${N} npz (need 419)"
[ "$N" -ne 419 ] && { echo "[r54_se_chain] REFUSE: R49's StreamEdit field is incomplete"; exit 1; }

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r54_se_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r54_se_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r54_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r54_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r54_se_chain] ${stage}: job ${ID}"
}

submit field  $SB/r54_field.sh;                                                                          FIELD=$ID
submit render --dependency=afterok:"$FIELD"  --array=0 $SB/r51_se_stage3.sh warp_max "$FIELD_ROOT" r54;  RENDER=$ID
submit eval   --dependency=afterok:"$RENDER" --array=0 $SB/r51_eval.sh r54_se;                           EVAL=$ID
submit motion --dependency=afterok:"$RENDER" --array=0 $SB/r51_motion.sh r54_motion;                     MOTION=$ID

echo "[r54_se_chain] submitted: field ${FIELD} -> render ${RENDER} -> eval ${EVAL} + motion ${MOTION} (ids in ${JOBS})"
