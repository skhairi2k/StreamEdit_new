#!/bin/bash
# R51 Wan-Edit track (ON JEAN ZAY, H100, in the clone, after r50_port.sh and r50_scratch_links.sh): the no-support arm A4 on the
# 419 pairs, chained on R49's Wan full-stage field job.
#
#   field   r51_field.sh wan                  A0 gate vs r49_field_full/wan_max_s5 + A4 start maps -> r51_field_full   afterok R49 field
#     -> render  r51_wan_render.sh            nosupp_s5 (N = 50, loop start 5)                  -> r51_wan_arms        afterok field
#        -> eval r51_eval.sh wan --array=0    harness, FiVE-Acc, CLIP-D                                                afterok render
# R49's field job id is read from logs/r49_wan_full_chain_jobs.txt ("field <id>"). If that job is still known to squeue, the field
# waits afterok on it; if it already left the queue, the chain is submitted only when R49's field finished cleanly (419 start maps
# under r49_field_full/wan_max_s5), with no dependency.
# Writes "<stage> <id>" lines to logs/r51_wan_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists;
# delete it deliberately to resubmit. The render is RESUMABLE: a timed-out render is resubmitted alone
# (sbatch slurm_scripts/five_bench/r51_wan_render.sh), then the eval by hand (sbatch --array=0 .../r51_eval.sh wan).
#
# Usage: bash slurm_scripts/five_bench/r51_wan_chain.sh [--dry-run]    (--dry-run: run every guard, print the sbatch lines, submit nothing)
#        R50_ENV=<file> overrides jz/r50_jz.env (tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r51_wan_chain] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r51_wan_chain] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: ported scripts, inputs, outputs on $SCRATCH
SB=slurm_scripts/five_bench
SCRIPTS=($SB/r51_field.sh $SB/r51_wan_render.sh $SB/r51_eval.sh)
for F in "${SCRIPTS[@]}" evaluation/r51_field.py evaluation/r51_units.py evaluation/r7_anchor_manifest.json; do
  [ -s "$F" ] || { echo "[r51_wan_chain] FATAL: missing $F (git pull?)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r51_wan_chain] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
grep -q '"nosupp_s5"' evaluation/r49_wan_render.py || { echo "[r51_wan_chain] FATAL: r49_wan_render.py has no nosupp_s5 config (git pull?)"; exit 1; }
D=$HOME/Data/dataggen/outputs/five_bench
if [ "${R49_ALLOW_WORK_OUTPUT:-0}" != 1 ]; then
  for X in r48_wan_draft r48_div r49_field_full r51_field_full r51_wan_arms; do
    [ -L "$D/$X" ] || { echo "[r51_wan_chain] FATAL: $D/$X is not a symlink to \$SCRATCH -- run: bash jz/r50_scratch_links.sh (R49_ALLOW_WORK_OUTPUT=1 overrides, at the risk of the \$WORK file quota)"; exit 1; }
  done
fi

# ---- R49's Wan field job
R49_JOBS=logs/r49_wan_full_chain_jobs.txt
R49_FIELD=$(awk '$1 == "field" {print $2}' "$R49_JOBS" 2>/dev/null | tail -1)
[[ "$R49_FIELD" =~ ^[0-9]+$ ]] || { echo "[r51_wan_chain] FATAL: no 'field <id>' line in $R49_JOBS -- submit R49's Wan full chain first"; exit 1; }
STATE=$(squeue -h -j "$R49_FIELD" -o %T 2>/dev/null | head -1)
if [ -n "$STATE" ]; then
  DEP=(--dependency=afterok:"$R49_FIELD")
  echo "[r51_wan_chain] R49 field ${R49_FIELD} is ${STATE}: the R51 field waits afterok on it"
else
  N=$(ls "$D"/r49_field_full/wan_max_s5/edit*/*.npy 2>/dev/null | wc -l)
  if [ "$N" -eq 419 ]; then
    DEP=()
    echo "[r51_wan_chain] R49 field ${R49_FIELD} left the queue; 419 wan_max_s5 start maps exist -- no dependency"
  else
    echo "[r51_wan_chain] REFUSE: R49 field ${R49_FIELD} is not in squeue and r49_field_full/wan_max_s5 holds ${N} start maps (need 419)"
    exit 1
  fi
fi

JOBS=logs/r51_wan_chain_jobs.txt
if [ -e "$JOBS" ] && [ "$DRY" -eq 0 ]; then
  echo "[r51_wan_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r51_wan_chain] delete it deliberately to resubmit"; exit 1
fi
[ "$DRY" -eq 0 ] && echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch arguments; prints the id, records it, aborts if refused
  local stage=$1; shift
  if [ "$DRY" -eq 1 ]; then
    echo "[r51_wan_chain] dry-run ${stage}: sbatch $*" >&2; echo "9$(printf '%s' "$stage" | cksum | cut -c1-5)"; return
  fi
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r51_wan_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r51_wan_chain] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r51_wan_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r51_wan_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

FD=$(submit field "${DEP[@]}" $SB/r51_field.sh wan) || exit 1
RD=$(submit render --dependency=afterok:${FD} $SB/r51_wan_render.sh) || exit 1
EV=$(submit eval --dependency=afterok:${RD} --array=0 $SB/r51_eval.sh wan) || exit 1

if [ "$DRY" -eq 1 ]; then
  echo "[r51_wan_chain] DRY RUN, nothing submitted (the ids above are placeholders): field ${FD} -> render ${RD} -> eval ${EV}"
else
  echo "[r51_wan_chain] submitted: field ${FD} -> render ${RD} -> eval ${EV}"
  echo "[r51_wan_chain] ids in ${JOBS}; follow with: squeue -u \$USER"
fi
