#!/bin/bash
# R49: submit the 59-active-step matched Wan-Edit curve LAST (run locally, not via sbatch):
#   r49_wan_render.sh m59 (L40S, array 0-2: matched59_s0.95 / s0.92 / s0.85; afterok on the main chain's `post` job)
#     -> r49_eval_all.sh m59 (L40S)
#     -> r49_post.sh (CPU: re-scores everything, now with the matched-59 curve for max_s5)
# The main `post` id is read from logs/r49_chain_jobs.txt (written by r49_chain.sh). While that job is still in the queue the
# renders depend on it; once it has left the queue its log must say "failures: 0". Appends "<stage> <id>" lines to the same file.
# Usage: bash slurm_scripts/five_bench/r49_matched59.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
cd "$REPO" || exit 1
JOBS=logs/r49_chain_jobs.txt
[ -s "$JOBS" ] || { echo "[r49_matched59] ${JOBS} missing -- run r49_chain.sh first"; exit 1; }
grep -q '^m59_render ' "$JOBS" && { echo "[r49_matched59] already submitted ($(grep '^m59_' "$JOBS" | paste -sd' ')) -- not resubmitting"; exit 1; }
POST=$(awk '$1=="post"{print $2; exit}' "$JOBS")
[[ "$POST" =~ ^[0-9]+$ ]] || { echo "[r49_matched59] no main 'post' id in ${JOBS}"; exit 1; }

if [ -n "$(squeue -h -j "$POST" 2>/dev/null)" ]; then
  DEP="--dependency=afterok:${POST}"; STATE="in queue"
else
  grep -q '^\[r49_post\] failures: 0$' "logs/r49_post_${POST}.out" 2>/dev/null \
    || { echo "[r49_matched59] main post job ${POST} left the queue without '[r49_post] failures: 0' in logs/r49_post_${POST}.out -- fix the main chain first"; exit 1; }
  DEP=""; STATE="finished, failures: 0"
fi
echo "[r49_matched59] main post job ${POST}: ${STATE} -> renders wait on it: ${DEP:-no}"

submit() {  # $1 = stage name, rest = sbatch args; prints the id, records it, aborts if refused
  local stage=$1; shift
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r49_matched59] REFUSED at stage '${stage}' (per-user submit cap?): ${out} $(cat "$err")" >&2
    echo "[r49_matched59] submitted so far: $(grep '^m59_' "$JOBS" | paste -sd' ')" >&2
    rm -f "$err"
    exit 1
  fi
  [ -s "$err" ] && echo "[r49_matched59] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r49_matched59] ${stage}: job ${id}" >&2
  echo "$id"
}

RENDER=$(submit m59_render ${DEP} --array=0-2 slurm_scripts/five_bench/r49_wan_render.sh m59) || exit 1
EVAL=$(submit m59_eval --dependency=afterok:${RENDER} slurm_scripts/five_bench/r49_eval_all.sh m59) || exit 1
POST2=$(submit m59_post --dependency=afterok:${EVAL} slurm_scripts/five_bench/r49_post.sh) || exit 1
echo "[r49_matched59] submitted: m59_render ${RENDER} (array 0-2) -> m59_eval ${EVAL} -> m59_post ${POST2}; ids appended to ${JOBS}"
