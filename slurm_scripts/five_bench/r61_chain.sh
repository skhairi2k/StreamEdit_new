#!/bin/bash
# R61 StreamEdit on LongLive track (lab cluster): se_max_ll on FiVE-Bench-Correct -- R55's method unchanged (5-step tgt0.9 VP draft +
# attention recorder, DINO / LPIPS divergence, inverse-response rho field, 15-step render = 20 NFE), 415 pairs, chained with afterok
# dependencies.
#
#   stage1   r61_ll_stage1.sh                      5-step VP draft + attention recorder       -> r61_ll_stage1, r61_ll_attn      (no dependency)
#     -> div     r61_div.sh --array=0-1            DINO / LPIPS divergence                    -> r61_div/se_{dino,lpips}         afterok stage1
#        -> field  r61_field.sh                     R49's field                                -> r61_field/se_max                afterok div
#           -> render  r61_ll_stage3.sh             R55's 15-step VP render                    -> r61_ll_arms/r61_max_ll          afterok field
# Every stage runs r59_check_prompts.py (Correct prompts, trigger words, anchors), as R59/R60, and exits non-zero on a mismatch, so
# afterok stops the rest. No eval stage: scoring waits for R57's suite (r61_eval_all.sh). 4 queue entries = 5 array tasks. The
# per-user submit cap (30 tasks, seen as QOSMaxSubmitJobPerUserLimit in R44/R45) is checked BEFORE anything is submitted: the driver
# REFUSES when the queue plus 5 would exceed it (R61_QUEUE_CAP=<n> overrides the cap value). Writes "<stage> <id>" lines to
# logs/r61_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second chain would render everything
# twice) or if the Correct benchmark files have uncommitted changes; delete the file / commit the changes deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r61_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
SB=slurm_scripts/five_bench
JOBS=logs/r61_chain_jobs.txt
N_TASKS=5
CAP=${R61_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r61_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r61_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r61_ll_stage1.sh $SB/r61_div.sh $SB/r61_field.sh $SB/r61_ll_stage3.sh \
         evaluation/r59_check_prompts.py evaluation/r49_field.py evaluation/r43_anchor_manifest_correct.json; do
  [ -f "$F" ] || { echo "[r61_chain] FATAL: $F missing"; exit 1; }
done
# a stale checkout would otherwise render on the official prompts: the stage scripts must carry the --edit_json_dir flag and sidecar
grep -q -- '--edit_json_dir' evaluation/r31_stage1.py && grep -q 'write_sidecar' evaluation/r31_stage1.py \
  && grep -q -- '--edit_json_dir' evaluation/r31_stage3.py && grep -q 'write_sidecar' evaluation/r31_stage3.py \
  || { echo "[r61_chain] FATAL: r31_stage1.py / r31_stage3.py in this checkout lack --edit_json_dir / the prompt sidecar (git pull / build-step smoke R61?)"; exit 1; }
[ -d ~/Data/dataggen/outputs/five_bench/anchors_r43_correct ] \
  || { echo "[r61_chain] FATAL: ~/Data/dataggen/outputs/five_bench/anchors_r43_correct missing"; exit 1; }
# benchmark freeze: the Correct files must not have changed under us
if ! git diff --quiet HEAD -- evaluation/five_bench_correct evaluation/r43_anchor_manifest_correct.json 2>/dev/null; then
  echo "[r61_chain] REFUSE: evaluation/five_bench_correct or r43_anchor_manifest_correct.json has uncommitted changes -- commit them first"
  exit 1
fi

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r61_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r61_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r61_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r61_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r61_chain] ${stage}: job ${ID}"
}

submit stage1 $SB/r61_ll_stage1.sh;                                                         S1=$ID
submit div    --dependency=afterok:"$S1"     --array=0-1 $SB/r61_div.sh;                    DIV=$ID
submit field  --dependency=afterok:"$DIV"    $SB/r61_field.sh;                              FIELD=$ID
submit render --dependency=afterok:"$FIELD"  $SB/r61_ll_stage3.sh;                          RENDER=$ID

echo "[r61_chain] submitted: stage1 ${S1} -> div ${DIV} -> field ${FIELD} -> render ${RENDER} (ids in ${JOBS})"
