#!/bin/bash
# R60 render chain (lab cluster): StreamEdit TEXT ONLY on FiVE-Bench-Correct, chained with afterok dependencies.
#
#   base   r60_render.sh correct --array=0-15     uniform-rho renders: N=15 rho x8 and N=20 rho 2/4/10/50 on the 190 changed pairs,
#                                                 N=20 rho 3/6/8/20 on all 415                                     (no dependency)
#   stage1 r60_se_stage1.sh                       text-only N=5 draft + attention recorder, 415 pairs              (no dependency)
#     -> div    r60_div.sh --array=0-1            DINO / LPIPS divergence                                          afterok stage1
#        -> field  r60_field.sh                    R49's field, text-only inputs                                    afterok div
#           -> render r60_se_stage3.sh             text-only 15-step render -> r60_se_arms/r60_max_novp_correct     afterok field
# base = 16 tasks, max = 5 tasks (stage1 1 + div 2 + field 1 + render 1). The per-user submit cap (30 tasks) is checked BEFORE anything
# is submitted: the driver REFUSES when the queue plus the new tasks would exceed it (R60_QUEUE_CAP=<n> overrides the cap value).
# Scoring and evaluation stay separate steps (assemble, eval, score).
#
# Usage: bash slurm_scripts/five_bench/r60_chain.sh [--only base|max|all] [--dry-run]
# Writes "<stage> <id>" lines to logs/r60_chain_jobs.txt, headed by the git commit. REFUSES if the part is already listed there (a second
# submission would render everything twice); delete the lines deliberately to resubmit.
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
SB=slurm_scripts/five_bench
JOBS=logs/r60_chain_jobs.txt
CAP=${R60_QUEUE_CAP:-30}
ONLY=all; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY=${2:-}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "[r60_chain] usage: bash $SB/r60_chain.sh [--only base|max|all] [--dry-run]"; exit 1 ;;
  esac
done
case "$ONLY" in base|max|all) ;; *) echo "[r60_chain] bad --only '${ONLY}'"; exit 1 ;; esac
cd "$REPO" || exit 1
mkdir -p logs

for F in $SB/r60_render.sh $SB/r60_se_stage1.sh $SB/r60_div.sh $SB/r60_field.sh $SB/r60_se_stage3.sh \
         evaluation/r60_cases_regen.json evaluation/r43_anchor_manifest_correct.json evaluation/r49_field.py; do
  [ -f "$F" ] || { echo "[r60_chain] FATAL: $F missing"; exit 1; }
done
# stale checkout guard: the stage scripts must read the Correct annotations
grep -q -- '--edit_json_dir' evaluation/r31_stage1.py && grep -q -- '--edit_json_dir' evaluation/r31_stage3.py \
  || { echo "[r60_chain] FATAL: r31_stage1.py / r31_stage3.py lack --edit_json_dir (git pull?)"; exit 1; }

N_BASE=16; N_MAX=5
case "$ONLY" in base) N_TASKS=$N_BASE ;; max) N_TASKS=$N_MAX ;; all) N_TASKS=$((N_BASE + N_MAX)) ;; esac
if [ -e "$JOBS" ]; then
  for P in base stage1; do
    { [ "$ONLY" = all ] || { [ "$ONLY" = base ] && [ $P = base ]; } || { [ "$ONLY" = max ] && [ $P = stage1 ]; }; } || continue
    grep -q "^${P} " "$JOBS" && { echo "[r60_chain] REFUSE: ${P} already submitted:"; cat "$JOBS"; echo "[r60_chain] delete its line deliberately to resubmit"; exit 1; }
  done
fi

QUEUE=$(squeue -r -u "$USER" -h 2>/dev/null | wc -l)
echo "[r60_chain] part=${ONLY} queue now: ${QUEUE} task(s) of yours; this submission adds ${N_TASKS}; cap ${CAP}"
if [ $((QUEUE + N_TASKS)) -gt "$CAP" ]; then
  echo "[r60_chain] REFUSE: ${QUEUE} + ${N_TASKS} > ${CAP} -- the per-user submit cap would reject part of the chain. Submit a smaller part (--only base|max) or wait for the queue to drain."
  exit 1
fi
if [ "$DRY" -eq 1 ]; then
  echo "[r60_chain] dry run OK: nothing submitted"; exit 0
fi

[ -e "$JOBS" ] || echo "# commit $(git rev-parse HEAD 2>/dev/null) | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"
submit() {  # $1 = stage name, rest = sbatch arguments; sets ID
  local stage=$1; shift
  ID=$(sbatch --parsable "$@") || {
    echo "[r60_chain] FAILED to submit ${stage}; already submitted:"; grep -v '^#' "$JOBS"
    echo "[r60_chain] scancel those ids before retrying"; exit 1; }
  ID=${ID%%;*}
  echo "${stage} ${ID}" >> "$JOBS"
  echo "[r60_chain] ${stage}: job ${ID}"
}

if [ "$ONLY" = base ] || [ "$ONLY" = all ]; then
  submit base $SB/r60_render.sh correct;                                                        BASE=$ID
fi
if [ "$ONLY" = max ] || [ "$ONLY" = all ]; then
  submit stage1 $SB/r60_se_stage1.sh;                                                           S1=$ID
  submit div    --dependency=afterok:"$S1"     $SB/r60_div.sh;                                  DIV=$ID
  submit field  --dependency=afterok:"$DIV"    $SB/r60_field.sh;                                FIELD=$ID
  submit render --dependency=afterok:"$FIELD"  $SB/r60_se_stage3.sh;                            RENDER=$ID
fi
echo "[r60_chain] submitted part=${ONLY} (ids in ${JOBS})"
