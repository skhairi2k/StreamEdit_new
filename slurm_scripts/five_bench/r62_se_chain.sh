#!/bin/bash
# R62 StreamEdit track (lab cluster): R51's demand-component ablation on FiVE-Bench-Correct (415 pairs), chained on R59's field job.
#
#   field   r62_field.sh se                 A0 gate vs r59_field_full/se_max + A1-A4 fields   -> r62_field/se_*        afterok R59's field job
#     -> render  r62_se_stage3.sh --array=0-3   nodiv, dino, lpips, nosupp                    -> r62_se_arms          afterok field
# Scoring is not chained: it waits for R57's suite (r62_eval.sh se).  5 queue tasks (1 + 4).
# R59's field job id is read from logs/r59_se_chain_jobs.txt ("field <id>"; fatal if absent). If that job is still known to squeue, our field
# waits afterok on it; if it already left the queue, the chain is submitted (without dependency) only when sacct says it COMPLETED and its 415
# se_max fields are on disk; otherwise it REFUSES. NOTE: if R59's field job fails later, ours stays in DependencyNeverSatisfied: scancel it.
# The per-user submit cap (30 tasks) is checked BEFORE anything is submitted (R62_QUEUE_CAP=<n> overrides). Writes "<stage> <id>" lines to
# logs/r62_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second chain would render everything twice);
# delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r62_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
SB=slurm_scripts/five_bench
JOBS=logs/r62_se_chain_jobs.txt
UPSTREAM=logs/r59_se_chain_jobs.txt
N_TASKS=5
CAP=${R62_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r62_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r62_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r62_field.sh $SB/r62_se_stage3.sh evaluation/r51_field.py evaluation/r59_check_prompts.py evaluation/r43_anchor_manifest_correct.json; do
  [ -f "$F" ] || { echo "[r62_se_chain] FATAL: $F missing"; exit 1; }
done
# a stale checkout would otherwise render on the official prompts: the stage script must carry the R59 flag and sidecar
grep -q -- '--edit_json_dir' evaluation/r31_stage3.py && grep -q 'write_sidecar' evaluation/r31_stage3.py \
  || { echo "[r62_se_chain] FATAL: r31_stage3.py in this checkout lacks --edit_json_dir / the prompt sidecar (git pull / R59 build?)"; exit 1; }
[ -d "$D/anchors_r43_correct" ] || { echo "[r62_se_chain] FATAL: $D/anchors_r43_correct missing"; exit 1; }

# ---- upstream: R59's field job
[ -f "$UPSTREAM" ] || { echo "[r62_se_chain] FATAL: $UPSTREAM missing -- R59's se-chain was not submitted from this checkout"; exit 1; }
R59_FIELD=$(grep -v '^#' "$UPSTREAM" | grep -o 'field [0-9][0-9]*' | tail -1 | awk '{print $2}')   # the last "field <id>" on any line
[[ "$R59_FIELD" =~ ^[0-9]+$ ]] || { echo "[r62_se_chain] FATAL: no 'field <id>' line in $UPSTREAM"; exit 1; }
if [ -n "$(squeue -j "$R59_FIELD" -h 2>/dev/null)" ]; then
  DEP="--dependency=afterok:${R59_FIELD}"
  echo "[r62_se_chain] R59 field job ${R59_FIELD} is still queued/running: our field waits afterok on it"
else
  STATE=$(sacct -j "$R59_FIELD" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  N_FIELD=$(ls "$D"/r59_field_full/se_max/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r62_se_chain] R59 field job ${R59_FIELD}: state '${STATE}', ${N_FIELD} se_max npz on disk"
  if [ "$STATE" = "COMPLETED" ] && [ "$N_FIELD" -eq 415 ]; then
    DEP=""
  else
    echo "[r62_se_chain] REFUSE: R59's field job is neither queued nor cleanly COMPLETED with 415 fields -- fix R59 first"; exit 1
  fi
fi

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r62_se_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r62_se_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname) | after R59 field ${R59_FIELD}" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r62_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r62_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r62_se_chain] ${stage}: job ${ID}"
}

submit field  $DEP                                  $SB/r62_field.sh se;                    FIELD=$ID
submit render --dependency=afterok:"$FIELD" --array=0-3 $SB/r62_se_stage3.sh;               RENDER=$ID

echo "[r62_se_chain] submitted: field ${FIELD} -> render ${RENDER} (ids in ${JOBS})"
