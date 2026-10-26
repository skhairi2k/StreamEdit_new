#!/bin/bash
# R53 StreamEdit track (lab cluster): se_max WITHOUT visual prompting on the 419 pairs, plus the 4 extra text-only N=20 uniform-rho points,
# chained with afterok dependencies. R51's eval / motion scripts are reused through their group argument.
#
#   stage1   r53_se_stage1.sh                      text-only N=5 draft + attention recorder   -> r53_se_stage1, r53_se_attn            (no dependency)
#     -> div     r53_div.sh --array=0-1            DINO / LPIPS divergence                    -> r53_div/se_{dino,lpips}               afterok stage1
#        -> field  r53_field.sh                     R49's field, text-only inputs              -> r53_field_full/se_max                 afterok div
#           -> render  r53_se_stage3.sh             text-only 15-step render                   -> r53_se_arms/r53_max_novp_full         afterok field
#              -> eval   r51_eval.sh r53_se (--array=0)         harness, FiVE-Acc, CLIP-D                                                  afterok render
#              -> motion r51_motion.sh r53_motion (--array=0)   motion fidelity of se_max_novp                                             afterok render
#   motion_official  r51_motion.sh r53_motion --array=1 --nodelist=node39   text-only official rho=2, N=15 (r44_n15_rho2_novp, on /store)  (no dependency)
#   sweep    r53_sweep_n20.sh --array=0-3          uniform text-only N=20, rho 3, 6, 8, 20    -> r53_rho_sweep                         (no dependency)
#     -> eval_sweep  r51_eval.sh r53_sweep --array=0-3   harness, FiVE-Acc, CLIP-D per point                                             afterok sweep
# 9 queue entries = 16 array tasks. The per-user submit cap (30 tasks, seen as QOSMaxSubmitJobPerUserLimit in R44/R45) is checked BEFORE
# anything is submitted: the driver REFUSES when the queue plus 16 would exceed it (R53_QUEUE_CAP=<n> overrides the cap value).
# Scoring (r53_score.py) stays a local step. Writes "<stage> <id>" lines to logs/r53_se_chain_jobs.txt, headed by the git commit, and
# REFUSES to run if that file exists (a second chain would render everything twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r53_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
SB=slurm_scripts/five_bench
JOBS=logs/r53_se_chain_jobs.txt
N_TASKS=16
CAP=${R53_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r53_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r53_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r53_se_stage1.sh $SB/r53_div.sh $SB/r53_field.sh $SB/r53_se_stage3.sh $SB/r53_sweep_n20.sh $SB/r51_eval.sh $SB/r51_motion.sh \
         evaluation/r51_units.py evaluation/r49_field.py; do
  [ -f "$F" ] || { echo "[r53_se_chain] FATAL: $F missing"; exit 1; }
done
# the shared R51 scripts must know the R53 groups (a stale checkout would reject them only once the jobs run)
grep -q 'r53_sweep' $SB/r51_eval.sh && grep -q 'r53_motion' $SB/r51_motion.sh && grep -q 'r53_sweep' evaluation/r51_units.py \
  || { echo "[r53_se_chain] FATAL: r51_eval.sh / r51_motion.sh / r51_units.py in this checkout lack the R53 groups (git pull / build-step se-chain R53?)"; exit 1; }

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r53_se_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r53_se_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain and leave it half submitted. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r53_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r53_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r53_se_chain] ${stage}: job ${ID}"
}

submit stage1          $SB/r53_se_stage1.sh;                                                         S1=$ID
submit div             --dependency=afterok:"$S1"     $SB/r53_div.sh;                                DIV=$ID
submit field           --dependency=afterok:"$DIV"    $SB/r53_field.sh;                              FIELD=$ID
submit render          --dependency=afterok:"$FIELD"  $SB/r53_se_stage3.sh;                          RENDER=$ID
submit eval            --dependency=afterok:"$RENDER" --array=0 $SB/r51_eval.sh r53_se;              EVAL=$ID
submit motion          --dependency=afterok:"$RENDER" --array=0 $SB/r51_motion.sh r53_motion;        MOTION=$ID
submit motion_official --array=1 --nodelist=node39 $SB/r51_motion.sh r53_motion;                     MOFF=$ID
submit sweep           $SB/r53_sweep_n20.sh;                                                         SWEEP=$ID
submit eval_sweep      --dependency=afterok:"$SWEEP"  --array=0-3 $SB/r51_eval.sh r53_sweep;         ESWEEP=$ID

echo "[r53_se_chain] submitted: stage1 ${S1} -> div ${DIV} -> field ${FIELD} -> render ${RENDER} -> eval ${EVAL} + motion ${MOTION} | motion_official ${MOFF} | sweep ${SWEEP} -> eval_sweep ${ESWEEP} (ids in ${JOBS})"
