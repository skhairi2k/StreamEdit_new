#!/bin/bash
# R62 Wan-Edit track (ON JEAN ZAY, H100, in the clone, after r50_port.sh and r50_scratch_links.sh): R51's no-support arm A4 on FiVE-Bench-Correct,
# chained on R58's Wan field job. Only R58's 190 changed pairs are rendered; the other 225 reuse R51's A4 render (r62_assemble.py links them).
#
#   field   r62_field.sh wan                 A0 gate vs r58_field/wan_max_s5 (190/190) + A4 start maps -> r62_field   afterok R58 field
#     -> render  r62_wan_render.sh           nosupp_s5 (N = 50, loop start 5), 190 pairs               -> r62_wan_arms afterok field
# No eval stage: every arm is scored on the lab (R57 suite) after fetch-renders.
# R58's field job id is read from logs/r58_wan_chain_jobs.txt ("field <id>"). If that job is still known to squeue, our field waits afterok on
# it; if it already left the queue, the chain is submitted (no dependency) only when R58's 190 wan_max_s5 start maps exist. The reuse of R51's
# renders is only valid if R58's reproducibility gate passed: the same check as r58_wan_chain.sh is repeated here on evaluation/csv/r58_repro_check.csv.
# Writes "<stage> <id>" lines to logs/r62_wan_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists;
# delete it deliberately to resubmit. The render is RESUMABLE: a timed-out render is resubmitted alone (sbatch slurm_scripts/five_bench/r62_wan_render.sh).
# NOTE: if R58's field job fails later, ours stays in DependencyNeverSatisfied: scancel it.
#
# Usage: bash slurm_scripts/five_bench/r62_wan_chain.sh [--dry-run]    (--dry-run: run every guard, print the sbatch lines, submit nothing)
#        R50_ENV=<file> overrides jz/r50_jz.env (tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r62_wan_chain] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r62_wan_chain] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: ported scripts, inputs, outputs on $SCRATCH, reproducibility gate
SB=slurm_scripts/five_bench
SCRIPTS=($SB/r62_field.sh $SB/r62_wan_render.sh)
for F in "${SCRIPTS[@]}" evaluation/r51_field.py evaluation/r58_cases_maxs5.json evaluation/five_bench_correct/edit1_FiVE.json; do
  [ -s "$F" ] || { echo "[r62_wan_chain] FATAL: missing $F (git pull?)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r62_wan_chain] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
grep -q '"nosupp_s5"' evaluation/r49_wan_render.py && grep -q -- '--anno_dir' evaluation/r49_wan_render.py \
  || { echo "[r62_wan_chain] FATAL: r49_wan_render.py lacks the nosupp_s5 config or --anno_dir (git pull?)"; exit 1; }
D=$HOME/Data/dataggen/outputs/five_bench
if [ "${R62_ALLOW_WORK_OUTPUT:-0}" != 1 ]; then
  for X in r62_field r62_wan_arms; do
    [ -L "$D/$X" ] || { echo "[r62_wan_chain] FATAL: $D/$X is not a symlink to \$SCRATCH -- run: bash jz/r50_scratch_links.sh (R62_ALLOW_WORK_OUTPUT=1 overrides, at the risk of the \$WORK file quota)"; exit 1; }
  done
fi
# the R51 renders the unchanged pairs will reuse must still exist (R58's own outputs are checked below, only if R58's field is no longer queued)
[ -d "$D/r51_wan_arms/nosupp_s5_full" ] || { echo "[r62_wan_chain] FATAL: $D/r51_wan_arms/nosupp_s5_full missing (the \$SCRATCH purge removed it?)"; exit 1; }
REPRO=evaluation/csv/r58_repro_check.csv
[ -s "$REPRO" ] || { echo "[r62_wan_chain] FATAL: $REPRO missing -- R58's repro-check has not run"; exit 1; }
if awk -F, 'NR > 1 && $NF != 1 { bad++ } END { exit (bad > 0 || NR < 5) }' "$REPRO"; then :; else
  echo "[r62_wan_chain] FATAL: $REPRO has a failing row or fewer than 4 rows -- new renders do NOT reproduce the old ones; do not mix R51's renders with new ones"; exit 1
fi

# ---- R58's Wan field job
R58_JOBS=logs/r58_wan_chain_jobs.txt
# the LAST "field <id>" anywhere in the file: a resubmitted chain is appended as one line ("resubmit-<date> draft <id> div <id> field <id> ...")
R58_FIELD=$(grep -v '^#' "$R58_JOBS" 2>/dev/null | grep -o 'field [0-9][0-9]*' | tail -1 | awk '{print $2}')
[[ "$R58_FIELD" =~ ^[0-9]+$ ]] || { echo "[r62_wan_chain] FATAL: no 'field <id>' line in $R58_JOBS -- submit R58's Wan chain first"; exit 1; }
STATE=$(squeue -h -j "$R58_FIELD" -o %T 2>/dev/null | head -1)
if [ -n "$STATE" ]; then
  DEP=(--dependency=afterok:"$R58_FIELD")
  echo "[r62_wan_chain] R58 field ${R58_FIELD} is ${STATE}: the R62 field waits afterok on it"
else
  N=$(ls "$D"/r58_field/wan_max_s5/edit*/*.npy 2>/dev/null | wc -l)
  for X in r58_wan_draft/attn r58_div/wan_dino r58_div/wan_lpips; do
    [ -d "$D/$X" ] || { echo "[r62_wan_chain] REFUSE: R58 field ${R58_FIELD} is not in squeue and $D/$X is missing"; exit 1; }
  done
  if [ "$N" -eq 190 ]; then
    DEP=()
    echo "[r62_wan_chain] R58 field ${R58_FIELD} left the queue; 190 wan_max_s5 start maps exist -- no dependency"
  else
    echo "[r62_wan_chain] REFUSE: R58 field ${R58_FIELD} is not in squeue and r58_field/wan_max_s5 holds ${N} start maps (need 190)"
    exit 1
  fi
fi

JOBS=logs/r62_wan_chain_jobs.txt
if [ -e "$JOBS" ] && [ "$DRY" -eq 0 ]; then
  echo "[r62_wan_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r62_wan_chain] delete it deliberately to resubmit"; exit 1
fi
[ "$DRY" -eq 0 ] && echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname) | after R58 field ${R58_FIELD}" > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch arguments; prints the id, records it, aborts if refused
  local stage=$1; shift
  if [ "$DRY" -eq 1 ]; then
    echo "[r62_wan_chain] dry-run ${stage}: sbatch $*" >&2; echo "9$(printf '%s' "$stage" | cksum | cut -c1-5)"; return
  fi
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r62_wan_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r62_wan_chain] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r62_wan_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r62_wan_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

FD=$(submit field "${DEP[@]}" $SB/r62_field.sh wan) || exit 1
RD=$(submit render --dependency=afterok:${FD} $SB/r62_wan_render.sh) || exit 1

if [ "$DRY" -eq 1 ]; then
  echo "[r62_wan_chain] DRY RUN, nothing submitted (the ids above are placeholders): field ${FD} -> render ${RD}"
else
  echo "[r62_wan_chain] submitted: field ${FD} -> render ${RD}"
  echo "[r62_wan_chain] ids in ${JOBS}; follow with: squeue -u \$USER"
fi
