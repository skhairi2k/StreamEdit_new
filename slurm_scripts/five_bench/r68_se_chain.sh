#!/bin/bash
# R68 StreamEdit track (lab cluster): R52's B1 (se_max's demand, linear map rho = 2 + 48 q) on FiVE-Bench-Correct (415 pairs), from R59's inputs.
#
#   field   r68_field.sh se            A0 gate vs r59_field_full/se_max + B1 field (linear check)  -> r68_field/se_max_linrho
#     -> render  r68_se_stage3.sh      r68_max_linrho_full, 415 pairs                              -> r68_se_arms            afterok field
# Scoring is not chained: it runs afterwards with the common R57 procedure (eval_suite.sh --batch evaluation/eval_suite/batches/r68.txt).
# 2 queue tasks. No upstream dependency: R59's field finished long ago, so the chain REFUSES unless R59's outputs are complete on lab /scratch
# ($SC/r59_field_full/se_max, $SC/r59_se_attn, $SC/r59_div/se_{dino,lpips}: 415 npz each), staged there from /store by step stage-r59-inputs
# (~/Data/.../r59_* are links to /store, which no compute node mounts). R68's field and render are written to $SC as well.
# The per-user submit cap (30 tasks) is checked BEFORE anything is submitted (R68_QUEUE_CAP=<n> overrides). Writes "<stage> <id>" lines to
# logs/r68_se_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second chain would render twice);
# delete it deliberately to resubmit.
#
# Usage: bash slurm_scripts/five_bench/r68_se_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench       # anchors only
SC=/scratch/PERSO/skhairi/five_bench       # R59 inputs (staged) + R68 outputs
SB=slurm_scripts/five_bench
JOBS=logs/r68_se_chain_jobs.txt
N_TASKS=2
N_PAIRS=415
CAP=${R68_QUEUE_CAP:-30}
cd "$REPO" || exit 1
mkdir -p logs

if [ -e "$JOBS" ]; then
  echo "[r68_se_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r68_se_chain] delete it deliberately to resubmit"; exit 1
fi
for F in $SB/r68_field.sh $SB/r68_se_stage3.sh evaluation/r51_field.py evaluation/r59_check_prompts.py evaluation/r43_anchor_manifest_correct.json; do
  [ -f "$F" ] || { echo "[r68_se_chain] FATAL: $F missing"; exit 1; }
done
# a stale checkout would otherwise render on the official prompts: the stage script must carry the R59 flag and sidecar
grep -q -- '--edit_json_dir' evaluation/r31_stage3.py && grep -q 'write_sidecar' evaluation/r31_stage3.py \
  || { echo "[r68_se_chain] FATAL: r31_stage3.py in this checkout lacks --edit_json_dir / the prompt sidecar (git pull / R59 build?)"; exit 1; }
# r51_field.py must know R52's linear map (B1)
grep -q -- '--mapping' evaluation/r51_field.py || { echo "[r68_se_chain] FATAL: r51_field.py lacks --mapping (R52 build?)"; exit 1; }
[ -d "$D/anchors_r43_correct" ] || { echo "[r68_se_chain] FATAL: $D/anchors_r43_correct missing"; exit 1; }

# ---- upstream: R59's Correct stage 1 + field must be complete on lab /scratch (real files, not links into /store)
BAD=0
for X in "$SC/r59_field_full/se_max" "$SC/r59_se_attn" "$SC/r59_div/se_dino" "$SC/r59_div/se_lpips"; do
  N=$(find "$X" -path '*/edit*/*.npz' -type f 2>/dev/null | wc -l)
  echo "[r68_se_chain] ${X}: ${N} npz (need ${N_PAIRS})"
  [ "$N" -lt "$N_PAIRS" ] && BAD=$((BAD + 1))
done
[ "$BAD" -ne 0 ] && { echo "[r68_se_chain] REFUSE: R59's inputs are not staged on ${SC} -- run step stage-r59-inputs (gpu-gw) first"; exit 1; }
for X in "$SC/r68_field/se_max_linrho" "$SC/r68_se_arms/r68_max_linrho_full"; do
  [ -e "$X" ] && { echo "[r68_se_chain] REFUSE: ${X} already exists -- R68 outputs are never overwritten"; exit 1; }
done

# ---- the per-user submit cap: refuse BEFORE submitting anything
QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r68_se_chain] queue now: ${QUEUE} task(s) of yours; this chain adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r68_se_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Wait for the queue to drain."
  exit 1
fi

echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname) | R59 inputs on disk" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r68_se_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r68_se_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r68_se_chain] ${stage}: job ${ID}"
}

submit field                                     $SB/r68_field.sh se;   FIELD=$ID
submit render --dependency=afterok:"$FIELD"      $SB/r68_se_stage3.sh;  RENDER=$ID

echo "[r68_se_chain] submitted: field ${FIELD} -> render ${RENDER} (ids in ${JOBS})"
