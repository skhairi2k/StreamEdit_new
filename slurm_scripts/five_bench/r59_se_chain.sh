#!/bin/bash
# R59 StreamEdit track (lab cluster): se_max on FiVE-Bench-Correct -- R49's method unchanged (5-step tgt0.9 VP draft + attention recorder,
# DINO / LPIPS divergence, inverse-response rho field, 15-step render = 20 NFE), 415 pairs, chained with afterok dependencies.
#
#   stage1   r59_se_stage1.sh                      5-step VP draft + attention recorder       -> r59_se_stage1, r59_se_attn      (no dependency)
#     -> div     r59_div.sh --array=0-1            DINO / LPIPS divergence                    -> r59_div/se_{dino,lpips}         afterok stage1
#        -> field  r59_field.sh                     R49's field                                -> r59_field_full/se_max           afterok div
#           -> render  r59_se_stage3.sh             R49's 15-step VP render                    -> r59_se_arms/r59_max_full        afterok field
# Every stage runs r59_check_prompts.py (Correct prompts, trigger words, anchors) and exits non-zero on a mismatch, so afterok stops the rest.
# No eval stage: scoring waits for R57's suite (r59_eval.sh). 4 queue entries = 5 array tasks. The per-user submit cap (30 tasks, seen as
# QOSMaxSubmitJobPerUserLimit in R44/R45) is checked BEFORE anything is submitted: the driver REFUSES when the queue plus 5 would exceed it
# (R59_QUEUE_CAP=<n> overrides the cap value; the 9-task r59_infer.sh array counts once it is queued). Writes "<stage> <id>" lines to
# logs/r59_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second chain would render everything
# twice); delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r59_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
SB=slurm_scripts/five_bench
JOBS=logs/r59_se_chain_jobs.txt
N_TASKS=5
CAP=${R59_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r59_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r59_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r59_se_stage1.sh $SB/r59_div.sh $SB/r59_field.sh $SB/r59_se_stage3.sh \
         evaluation/r59_check_prompts.py evaluation/r49_field.py evaluation/r43_anchor_manifest_correct.json; do
  [ -f "$F" ] || { echo "[r59_se_chain] FATAL: $F missing"; exit 1; }
done
# a stale checkout would otherwise render on the official prompts: the stage scripts must carry the R59 flag and sidecar
grep -q -- '--edit_json_dir' evaluation/r31_stage1.py && grep -q 'write_sidecar' evaluation/r31_stage1.py \
  && grep -q -- '--edit_json_dir' evaluation/r31_stage3.py && grep -q 'write_sidecar' evaluation/r31_stage3.py \
  || { echo "[r59_se_chain] FATAL: r31_stage1.py / r31_stage3.py in this checkout lack --edit_json_dir / the prompt sidecar (git pull / build-step se-chain R59?)"; exit 1; }
[ -d ~/Data/dataggen/outputs/five_bench/anchors_r43_correct ] \
  || { echo "[r59_se_chain] FATAL: ~/Data/dataggen/outputs/five_bench/anchors_r43_correct missing"; exit 1; }

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r59_se_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r59_se_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r59_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r59_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r59_se_chain] ${stage}: job ${ID}"
}

submit stage1 $SB/r59_se_stage1.sh;                                                         S1=$ID
submit div    --dependency=afterok:"$S1"     --array=0-1 $SB/r59_div.sh;                    DIV=$ID
submit field  --dependency=afterok:"$DIV"    $SB/r59_field.sh;                              FIELD=$ID
submit render --dependency=afterok:"$FIELD"  $SB/r59_se_stage3.sh;                          RENDER=$ID

echo "[r59_se_chain] submitted: stage1 ${S1} -> div ${DIV} -> field ${FIELD} -> render ${RENDER} (ids in ${JOBS})"
