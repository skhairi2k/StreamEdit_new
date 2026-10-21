#!/bin/bash
# R49 full stage, StreamEdit track (lab cluster, L40S): chain the four jobs after the full draft with afterok.
#
# Usage: bash slurm_scripts/five_bench/r49_se_full_chain.sh DRAFT_JOB     # DRAFT_JOB = the r48_se_stage1.sh full job id
#
#   DRAFT_JOB (r48_se_stage1.sh full: 419 drafts + attention -> r48_se_stage1, r48_se_attn)
#     -> div    r48_div.sh full, array 0-1 (se_dino, se_lpips)            -> r48_div/se_{dino,lpips}
#     -> field  r49_field.sh se full                                       -> r49_field_full/se_max
#     -> render r49_se_stage3.sh full                                      -> r49_se_arms/r49_max_full/step14
#     -> eval   r49_eval_all.sh se full (harness, FiVE-Acc, CLIP-D)        -> evaluation/csv/*r49*_se_max_full*
# Every stage waits afterok on the previous one, so a failure stops the rest (they stay PENDING with
# DependencyNeverSatisfied; scancel them). Scoring (r49_score.py --mode full --host se) stays a separate local step.
#
# If DRAFT_JOB is still known to squeue, the divergence waits afterok on it. If it has already left the queue, the chain
# is submitted only when the draft finished cleanly: 419 attention npz in r48_se_attn and 'failures: 0' in its log.
# Writes logs/r49_se_full_chain_jobs.txt (one "<stage> <id>" per line) and refuses to run if that file exists
# (a second submission would render everything twice); delete it deliberately to resubmit.
set -u

DRAFT_JOB=${1:-}
[[ "$DRAFT_JOB" =~ ^[0-9]+$ ]] || { echo "[r49_se_full_chain] usage: bash slurm_scripts/five_bench/r49_se_full_chain.sh DRAFT_JOB"; exit 1; }
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
JOBS=logs/r49_se_full_chain_jobs.txt
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r49_se_full_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r49_se_full_chain] delete it deliberately to resubmit"; exit 1
fi
for F in slurm_scripts/five_bench/r48_div.sh slurm_scripts/five_bench/r49_field.sh slurm_scripts/five_bench/r49_se_stage3.sh \
         slurm_scripts/five_bench/r49_eval_all.sh; do
  [ -f "$F" ] || { echo "[r49_se_full_chain] FATAL: $F missing"; exit 1; }
done

STATE=$(squeue -h -j "$DRAFT_JOB" -o %T 2>/dev/null | head -1)
if [ -n "$STATE" ]; then
  DEP=(--dependency=afterok:"$DRAFT_JOB")
  echo "[r49_se_full_chain] draft ${DRAFT_JOB} is ${STATE}: divergence waits afterok on it"
else
  N_ATTN=$(ls "$D"/r48_se_attn/edit*/*.npz 2>/dev/null | wc -l)
  LOG=logs/r48_se_stage1_${DRAFT_JOB}.out
  if [ "$N_ATTN" -eq 419 ] && grep -q '\[r48_se_stage1\] failures: 0' "$LOG" 2>/dev/null; then
    DEP=()
    echo "[r49_se_full_chain] draft ${DRAFT_JOB} left the queue; it finished cleanly (419 attention npz, ${LOG}: failures 0) -- no dependency"
  else
    echo "[r49_se_full_chain] REFUSE: draft ${DRAFT_JOB} is not in squeue and did not finish cleanly (${N_ATTN} attention npz, log: $(grep -h 'failures:' "$LOG" 2>/dev/null | tail -1))"
    exit 1
  fi
fi

echo "draft ${DRAFT_JOB}" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r49_se_full_chain] FAILED to submit ${stage}; already submitted:"; cat "$JOBS"
    echo "[r49_se_full_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r49_se_full_chain] ${stage}: job ${ID}"
}

submit div    "${DEP[@]}" --array=0-1 slurm_scripts/five_bench/r48_div.sh full;          DIV=$ID
submit field  --dependency=afterok:"$DIV"    slurm_scripts/five_bench/r49_field.sh se full;      FIELD=$ID
submit render --dependency=afterok:"$FIELD"  slurm_scripts/five_bench/r49_se_stage3.sh full;     RENDER=$ID
submit eval   --dependency=afterok:"$RENDER" slurm_scripts/five_bench/r49_eval_all.sh se full;   EVAL=$ID

echo "[r49_se_full_chain] submitted: div ${DIV} -> field ${FIELD} -> render ${RENDER} -> eval ${EVAL} (ids in ${JOBS})"
