#!/bin/bash
# R50 step jz-repro (ON JEAN ZAY, in the clone, after r50_pip_overlay.sh, r50_port.sh and a passing r50_preflight.sh):
# submit the Wan-Edit side of R49 on the 18 clips of evaluation/r49_cases.json, with afterok dependencies, in one go.
#
#   wan_draft  r49_wan_draft.sh                      the 17-step draft from step 3, attention recorded           (no dependency)
#     -> div   r49_div.sh wan                        DINOv3 + LPIPS divergence of the Wan drafts                 afterok wan_draft
#        -> field    r49_field.sh wan                demand q and per-token start steps (max_s9, max_s5)        afterok div
#           -> wan_render  r49_wan_render.sh arms    --array=1 = max_s5 only (the arm of the full stage)        afterok field
#   official   r50_wan_official.sh                   native_s0.92 = the official Wan-Edit baseline               (no dependency)
#   eval       r49_eval_all.sh wan_pair              FiVE harness + FiVE-Acc + CLIP-D for max_s5 and native_s0.92 afterok wan_render, official
#
# 5 queue entries. Writes "<stage> <job id>" lines to logs/r50_repro_jobs.txt, headed by the git commit and the number of modified files
# (the port edits make the tree "dirty" by design), so every result carries the code version that produced it.
# When the queue is empty:  grep -h failures logs/r49_*.out logs/r50_*.out    then    python evaluation/r50_compare.py
# Rough cost on an H100 (the lab's L40S took ~90 s per Wan draft clip): a few hours of GPU plus queue time.
#
# Usage: bash jz/r50_repro_chain.sh        (R50_ENV=<file> overrides jz/r50_jz.env, for tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r50_repro] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r50_repro] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: the scripts must be the ported ones, and the interpreter and inputs must be there
SCRIPTS=(slurm_scripts/five_bench/{r49_wan_draft,r49_div,r49_field,r49_wan_render,r50_wan_official,r49_eval_all}.sh)
for F in "${SCRIPTS[@]}" evaluation/r49_cases.json evaluation/r49_new_clip.json; do
  [ -s "$F" ] || { echo "[r50_repro] FATAL: missing $F (git pull?)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r50_repro] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
WRAP=$WORK/pyenv/five-bench/python
[ -x "$WRAP" ] || { echo "[r50_repro] FATAL: $WRAP missing -- run: bash jz/r50_pip_overlay.sh"; exit 1; }
[ -d "$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/videos" ] || { echo "[r50_repro] FATAL: ~/Data/FiVE-.../videos missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }
[ -e Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B ] || { echo "[r50_repro] FATAL: the Wan2.1 weights link is missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }

JOBS=logs/r50_repro_jobs.txt
if [ -s "$JOBS" ] && grep -q '^eval ' "$JOBS"; then
  echo "[r50_repro] ${JOBS} already lists a reproduction chain:"; cat "$JOBS"
  echo "[r50_repro] move it aside to submit a new one (a second chain would render everything twice)"; exit 1
fi
{
  echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)"
} > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch args; prints the id, records it, aborts if refused
  local stage=$1; shift
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r50_repro] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r50_repro] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r50_repro] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r50_repro] ${stage}: job ${id}" >&2
  echo "$id"
}

WD=$(submit wan_draft slurm_scripts/five_bench/r49_wan_draft.sh) || exit 1
DV=$(submit div --dependency=afterok:${WD} slurm_scripts/five_bench/r49_div.sh wan) || exit 1
FD=$(submit field --dependency=afterok:${DV} slurm_scripts/five_bench/r49_field.sh wan) || exit 1
WR=$(submit wan_render --dependency=afterok:${FD} --array=1 slurm_scripts/five_bench/r49_wan_render.sh arms) || exit 1
OF=$(submit official slurm_scripts/five_bench/r50_wan_official.sh) || exit 1
EV=$(submit eval --dependency=afterok:${WR}:${OF} slurm_scripts/five_bench/r49_eval_all.sh wan_pair) || exit 1

echo "[r50_repro] submitted: wan_draft ${WD} -> div ${DV} -> field ${FD} -> wan_render ${WR} (max_s5) | official ${OF} | eval ${EV} (afterok ${WR}, ${OF})"
echo "[r50_repro] ids in ${JOBS}; follow with: squeue -u \$USER"
