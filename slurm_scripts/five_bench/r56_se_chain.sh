#!/bin/bash
# R56 StreamEdit track (lab cluster): the draft-budget ablation -- se_max with a 15-step draft (sigma_d = 0.96: tgt0.96, release from t = 0.933),
# VP, 419 pairs, chained with afterok dependencies. R51's eval / motion scripts are reused through their group argument.
#
#   stage1   r56_se_stage1.sh                      15-step VP draft + attention recorder      -> r56_se_stage1, r56_se_attn            (no dependency)
#     -> div     r56_div.sh --array=0-1            DINO / LPIPS divergence                    -> r56_div/se_{dino,lpips}               afterok stage1
#        -> field  r56_field.sh                     R49's field on the 15-step draft's inputs  -> r56_field_full/se_max                 afterok div
#           -> render  r56_se_stage3.sh             R49's 15-step VP render                    -> r56_se_arms/r56_max_n15_full          afterok field
#              -> eval   r51_eval.sh r56_se (--array=0)         harness, FiVE-Acc, CLIP-D                                                  afterok render
#              -> motion r51_motion.sh r56_motion (--array=0)   motion fidelity                                                            afterok render
# 6 queue entries = 7 array tasks; total NFE of the arm = 15 + 15 = 30 (read on R39's uniform-rho N = 30 curve). The per-user submit cap (30 tasks,
# seen as QOSMaxSubmitJobPerUserLimit in R44/R45) is checked BEFORE anything is submitted: the driver REFUSES when the queue plus 7 would exceed it
# (R56_QUEUE_CAP=<n> overrides the cap value). Scoring (r56_score.py) stays a local step and needs R51's motion units for the 5-step arm and the
# official render. Writes "<stage> <id>" lines to logs/r56_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists
# (a second chain would render everything twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r56_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
SB=slurm_scripts/five_bench
JOBS=logs/r56_se_chain_jobs.txt
N_TASKS=7
CAP=${R56_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r56_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r56_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r56_se_stage1.sh $SB/r56_div.sh $SB/r56_field.sh $SB/r56_se_stage3.sh $SB/r51_eval.sh $SB/r51_motion.sh \
         evaluation/r51_units.py evaluation/r49_field.py; do
  [ -f "$F" ] || { echo "[r56_se_chain] FATAL: $F missing"; exit 1; }
done
# the shared R51 scripts must know the R56 groups (a stale checkout would reject them only once the jobs run)
grep -q 'r56_se' $SB/r51_eval.sh && grep -q 'r56_motion' $SB/r51_motion.sh && grep -q 'r56_se' evaluation/r51_units.py \
  || { echo "[r56_se_chain] FATAL: r51_eval.sh / r51_motion.sh / r51_units.py in this checkout lack the R56 groups (git pull / build-step se-chain R56?)"; exit 1; }

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r56_se_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r56_se_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r56_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r56_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r56_se_chain] ${stage}: job ${ID}"
}

submit stage1 $SB/r56_se_stage1.sh;                                                         S1=$ID
submit div    --dependency=afterok:"$S1"     --array=0-1 $SB/r56_div.sh;                    DIV=$ID
submit field  --dependency=afterok:"$DIV"    $SB/r56_field.sh;                              FIELD=$ID
submit render --dependency=afterok:"$FIELD"  $SB/r56_se_stage3.sh;                          RENDER=$ID
submit eval   --dependency=afterok:"$RENDER" --array=0 $SB/r51_eval.sh r56_se;              EVAL=$ID
submit motion --dependency=afterok:"$RENDER" --array=0 $SB/r51_motion.sh r56_motion;        MOTION=$ID

echo "[r56_se_chain] submitted: stage1 ${S1} -> div ${DIV} -> field ${FIELD} -> render ${RENDER} -> eval ${EVAL} + motion ${MOTION} (ids in ${JOBS})"
