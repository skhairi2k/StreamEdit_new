#!/bin/bash
# R55 StreamEdit-on-LongLive track (lab cluster): se_max on the LongLive backbone (VP), 419 pairs, chained with afterok dependencies.
# R51's eval / motion scripts are reused through their group argument. Run it only after the smoke test PASSED (r55_smoke.sh).
#
#   stage1   r55_ll_stage1.sh                      LongLive N=5 draft + attention recorder   -> r55_ll_stage1, r55_ll_attn            (no dependency)
#     -> div     r55_div.sh --array=0-1            DINO / LPIPS divergence                   -> r55_div/se_{dino,lpips}               afterok stage1
#        -> field  r55_field.sh                     R49's field on the LongLive inputs        -> r55_field_full/se_max                 afterok div
#           -> render  r55_ll_stage3.sh             LongLive 15-step render, VP               -> r55_ll_arms/r55_max_ll_full           afterok field
#              -> eval   r51_eval.sh r55_se (--array=0)         harness, FiVE-Acc, CLIP-D                                                   afterok render
#              -> motion r51_motion.sh r55_motion (--array=0)   motion fidelity of se_max_ll                                                afterok render
#   motion_official  r51_motion.sh r55_motion --array=1 --nodelist=node39   official LongLive rho=2, N=15 (r45_sweep/n15, on /store)      (no dependency)
# 7 queue entries = 8 array tasks. Gates, checked BEFORE anything is submitted: (1) the latest r55_smoke log must end with `failures: 0`
# (R55_SKIP_SMOKE_CHECK=1 overrides); (2) the per-user submit cap (30 tasks, seen as QOSMaxSubmitJobPerUserLimit in R44/R45): the driver REFUSES when
# the queue plus 8 would exceed it (R55_QUEUE_CAP=<n> overrides the cap value). The scoring step (r55_score.py) needs R45's LongLive sweeps scored
# first. Writes "<stage> <id>" lines to logs/r55_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second chain
# would render everything twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r55_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
SB=slurm_scripts/five_bench
JOBS=logs/r55_chain_jobs.txt
N_TASKS=8
CAP=${R55_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r55_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r55_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r55_ll_stage1.sh $SB/r55_div.sh $SB/r55_field.sh $SB/r55_ll_stage3.sh $SB/r51_eval.sh $SB/r51_motion.sh \
         evaluation/r51_units.py evaluation/r49_field.py; do
  [ -f "$F" ] || { echo "[r55_chain] FATAL: $F missing"; exit 1; }
done
# the shared R51 scripts must know the R55 groups (a stale checkout would reject them only once the jobs run)
grep -q 'r55_se' $SB/r51_eval.sh && grep -q 'r55_motion' $SB/r51_motion.sh && grep -q 'r55_se' evaluation/r51_units.py \
  || { echo "[r55_chain] FATAL: r51_eval.sh / r51_motion.sh / r51_units.py in this checkout lack the R55 groups (git pull / build-step chain R55?)"; exit 1; }
# the recorder port must be in this checkout (the smoke test proved it)
grep -q 'attn_record' LongLive_StreamEdit/pipeline/edit_causal_inference.py \
  || { echo "[r55_chain] FATAL: LongLive's pipeline has no attention recorder in this checkout"; exit 1; }

# ---- gate 1: the smoke test passed
if [ "${R55_SKIP_SMOKE_CHECK:-0}" != 1 ]; then
  SMOKE_LOG=$(ls -t logs/r55_smoke_*.out 2>/dev/null | head -1)
  if [ -z "$SMOKE_LOG" ] || ! grep -q '^\[r55_smoke\] failures: 0' "$SMOKE_LOG"; then
    echo "[r55_chain] REFUSE: no passing smoke test (latest log: ${SMOKE_LOG:-none}; need '[r55_smoke] failures: 0'). Run r55_smoke.sh first (R55_SKIP_SMOKE_CHECK=1 overrides)."
    exit 1
  fi
  echo "[r55_chain] smoke test passed: ${SMOKE_LOG}"
fi

# ---- gate 2: the per-user submit cap -- refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r55_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r55_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r55_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r55_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r55_chain] ${stage}: job ${ID}"
}

submit stage1          $SB/r55_ll_stage1.sh;                                                         S1=$ID
submit div             --dependency=afterok:"$S1"     --array=0-1 $SB/r55_div.sh;                    DIV=$ID
submit field           --dependency=afterok:"$DIV"    $SB/r55_field.sh;                              FIELD=$ID
submit render          --dependency=afterok:"$FIELD"  $SB/r55_ll_stage3.sh;                          RENDER=$ID
submit eval            --dependency=afterok:"$RENDER" --array=0 $SB/r51_eval.sh r55_se;              EVAL=$ID
submit motion          --dependency=afterok:"$RENDER" --array=0 $SB/r51_motion.sh r55_motion;        MOTION=$ID
submit motion_official --array=1 --nodelist=node39 $SB/r51_motion.sh r55_motion;                     MOFF=$ID

echo "[r55_chain] submitted: stage1 ${S1} -> div ${DIV} -> field ${FIELD} -> render ${RENDER} -> eval ${EVAL} + motion ${MOTION} | motion_official ${MOFF} (ids in ${JOBS})"
