#!/bin/bash
# R49 full stage, Wan-Edit track (ON JEAN ZAY, H100, in the clone, after r50_pip_overlay.sh, r50_port.sh, r50_scratch_links.sh and a
# passing R50 jz-compare): submit every Wan-Edit job of the 419 pairs with afterok dependencies, in one go.
#
#   ADAPTIVE   draft   r48_wan_draft.sh full                   17-step draft from step 3 + attention, 419 clips             (no dependency)
#                -> div    r48_div.sh full --array=2-3         DINOv3 / LPIPS divergence of the Wan drafts (wan_dino, wan_lpips)   afterok draft
#                   -> field  r49_field.sh wan full            demand q and per-token start steps -> r49_field_full         afterok div
#                      -> render_max_s5  r49_wan_render.sh arms full --array=1      adaptive max_s5 (floor step 5)              afterok field
#   BASELINE SWEEP (global start, N = 50)  -- r48_wan_render.sh, the same renderer as R48                                    (no dependency)
#              render_native      r48_wan_render.sh full --array=0-5       native_s0.95 .. native_s0.4 (native_s0.92 = official Wan-Edit)
#              render_native0978  r48_wan_render.sh explore_full --array=0 native_s0.978 (the global start = max_s5's floor)
#   EQUAL NFE (59 active steps = max_s5's budget)                                                                              (no dependency)
#              render_m59         r49_wan_render.sh m59 full --array=0-3   matched59_s0.95 / s0.92 / s0.85 / s0.978
#   EVAL       eval_wan   r49_eval_all.sh wan full                      1 unit (max_s5)                                       afterok render_max_s5
#              eval_base  r49_eval_all.sh base full --array=0-6         7 units: the 6 natives + native_s0.978                 afterok render_native, render_native0978
#              eval_m59   r49_eval_all.sh m59 full --array=0-3          4 units                                               afterok render_m59
#
# 10 queue entries (the arrays count as one each); 12 renders (max_s5 + 7 native + 4 matched-59) of 419 pairs. Writes "<stage> <job id>"
# lines to logs/r49_wan_full_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second chain would render
# everything twice); delete it deliberately to resubmit. Every render is RESUMABLE (finished clips are skipped): a task that hits the time
# limit is resubmitted ALONE, e.g.  sbatch --array=3 slurm_scripts/five_bench/r48_wan_render.sh full  -- the evals keep waiting only if the
# array job still holds a pending task; if an afterok dependency was lost, resubmit the eval by hand once the render is complete.
# Outputs go to $SCRATCH through the symlinks of jz/r50_scratch_links.sh (the guard below refuses to start without them: ~220k PNG files
# would not fit the 500k-file quota of $WORK); bring them to the lab with jz/r50_fetch_outputs.sh.
# When the queue is empty:  grep -h "failures" logs/r4[89]_*.out | sort | uniq -c   (every line must read 'failures: 0' / 'unit failures: 0')
#
# Usage: bash slurm_scripts/five_bench/r49_wan_full_chain.sh [--dry-run]    (--dry-run: run every guard, print the sbatch lines, submit nothing)
#        R50_ENV=<file> overrides jz/r50_jz.env (tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r49_wan_full_chain] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r49_wan_full_chain] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: ported scripts, interpreter, inputs, outputs on $SCRATCH
SB=slurm_scripts/five_bench
SCRIPTS=($SB/{r48_wan_draft,r48_div,r49_field,r48_wan_render,r49_wan_render,r49_eval_all}.sh)
for F in "${SCRIPTS[@]}" evaluation/r7_anchor_manifest.json; do
  [ -s "$F" ] || { echo "[r49_wan_full_chain] FATAL: missing $F (git pull?)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r49_wan_full_chain] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
grep -q 'explore_full' $SB/r48_wan_render.sh && grep -q 'PYMODE' $SB/r49_wan_render.sh || { echo "[r49_wan_full_chain] FATAL: the full-mode render scripts are not in this clone (git pull?)"; exit 1; }
WRAP=$WORK/pyenv/five-bench/python
[ -x "$WRAP" ] || { echo "[r49_wan_full_chain] FATAL: $WRAP missing -- run: bash jz/r50_pip_overlay.sh"; exit 1; }
[ -d "$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/videos" ] || { echo "[r49_wan_full_chain] FATAL: ~/Data/FiVE-.../videos missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }
[ -e Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B ] || { echo "[r49_wan_full_chain] FATAL: the Wan2.1 weights link is missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }
D=$HOME/Data/dataggen/outputs/five_bench
if [ "${R49_ALLOW_WORK_OUTPUT:-0}" != 1 ]; then
  for X in r48_wan_draft r48_div r49_field_full r48_wan_arms r49_wan_arms; do
    [ -L "$D/$X" ] || { echo "[r49_wan_full_chain] FATAL: $D/$X is not a symlink to \$SCRATCH -- run: bash jz/r50_scratch_links.sh (R49_ALLOW_WORK_OUTPUT=1 overrides, at the risk of the \$WORK file quota)"; exit 1; }
  done
fi

JOBS=logs/r49_wan_full_chain_jobs.txt
if [ -e "$JOBS" ] && [ "$DRY" -eq 0 ]; then
  echo "[r49_wan_full_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r49_wan_full_chain] delete it deliberately to resubmit"; exit 1
fi
[ "$DRY" -eq 0 ] && echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch arguments; prints the id, records it, aborts if refused
  local stage=$1; shift
  if [ "$DRY" -eq 1 ]; then
    echo "[r49_wan_full_chain] dry-run ${stage}: sbatch $*" >&2; echo "9$(printf '%s' "$stage" | cksum | cut -c1-5)"; return
  fi
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r49_wan_full_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r49_wan_full_chain] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r49_wan_full_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r49_wan_full_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

DR=$(submit draft $SB/r48_wan_draft.sh full) || exit 1
DV=$(submit div --dependency=afterok:${DR} --array=2-3 $SB/r48_div.sh full) || exit 1
FD=$(submit field --dependency=afterok:${DV} $SB/r49_field.sh wan full) || exit 1
MX=$(submit render_max_s5 --dependency=afterok:${FD} --array=1 $SB/r49_wan_render.sh arms full) || exit 1
NA=$(submit render_native --array=0-5 $SB/r48_wan_render.sh full) || exit 1
N0=$(submit render_native0978 --array=0 $SB/r48_wan_render.sh explore_full) || exit 1
M5=$(submit render_m59 --array=0-3 $SB/r49_wan_render.sh m59 full) || exit 1
EW=$(submit eval_wan --dependency=afterok:${MX} $SB/r49_eval_all.sh wan full) || exit 1
EB=$(submit eval_base --dependency=afterok:${NA}:${N0} --array=0-6 $SB/r49_eval_all.sh base full) || exit 1
EM=$(submit eval_m59 --dependency=afterok:${M5} --array=0-3 $SB/r49_eval_all.sh m59 full) || exit 1

echo "[r49_wan_full_chain] submitted: draft ${DR} -> div ${DV} -> field ${FD} -> max_s5 ${MX} | native ${NA} native0978 ${N0} m59 ${M5} | eval wan ${EW} base ${EB} m59 ${EM}"
echo "[r49_wan_full_chain] ids in ${JOBS}; follow with: squeue -u \$USER"
