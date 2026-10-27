#!/bin/bash
# R58 (ON JEAN ZAY, H100, in the clone, after r50_port.sh, r50_scratch_links.sh and a PASSING r58_repro_check.py): submit every Wan-Edit
# job of the FiVE-Bench-Correct regeneration in one go, with afterok dependencies.
#
#   ADAPTIVE   draft   r58_wan_draft.sh                        17-step draft + attention, 190 pairs (prompt or trigger object changed)  (no dependency)
#                -> div    r58_div.sh --array=0-1              DINOv3 / LPIPS divergence of the drafts (wan_dino, wan_lpips)            afterok draft
#                   -> field  r58_field.sh                     demand q and per-token start steps -> r58_field                         afterok div
#                      -> render_max_s5  r58_wan_render.sh correct --array=11   max_s5 on the 190 pairs                                  afterok field
#   BASELINES  render_base r58_wan_render.sh correct --array=0-10   7 native + 4 matched-59 configs on the 47 pairs with a changed prompt  (no dependency)
#
# 5 queue entries. Everything goes to NEW folders ($D/r58_wan_draft, r58_div, r58_field, r58_wan_arms); no R48/R49 video is written. Every
# render is RESUMABLE (finished clips are skipped): a task that hits the time limit is resubmitted ALONE, e.g.
#   sbatch --array=11 slurm_scripts/five_bench/r58_wan_render.sh correct
# Writes "<stage> <job id>" lines to logs/r58_wan_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists (a second
# chain would render everything twice); delete it deliberately to resubmit.
# When the queue is empty:  grep -h "failures" logs/r58_*.out | sort | uniq -c   (every line must read 'failures: 0')
#
# Usage: bash slurm_scripts/five_bench/r58_wan_chain.sh [--dry-run]    (--dry-run: run every guard, print the sbatch lines, submit nothing)
#        R50_ENV=<file> overrides jz/r50_jz.env (tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r58_wan_chain] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r58_wan_chain] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: ported scripts, interpreter, case lists, inputs, outputs on $SCRATCH, reproducibility gate
SB=slurm_scripts/five_bench
SCRIPTS=($SB/{r58_wan_draft,r58_div,r58_field,r58_wan_render}.sh)
CASEFILES=(evaluation/r58_cases_base.json evaluation/r58_cases_maxs5.json evaluation/five_bench_correct/edit1_FiVE.json)
for F in "${SCRIPTS[@]}" "${CASEFILES[@]}" evaluation/r58_repro_check.py; do
  [ -s "$F" ] || { echo "[r58_wan_chain] FATAL: missing $F (git pull?)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r58_wan_chain] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
WRAP=$WORK/pyenv/five-bench/python
[ -x "$WRAP" ] || { echo "[r58_wan_chain] FATAL: $WRAP missing -- run: bash jz/r50_pip_overlay.sh"; exit 1; }
[ -d "$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/videos" ] || { echo "[r58_wan_chain] FATAL: ~/Data/FiVE-.../videos missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }
[ -e Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B ] || { echo "[r58_wan_chain] FATAL: the Wan2.1 weights link is missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }
D=$HOME/Data/dataggen/outputs/five_bench
if [ "${R58_ALLOW_WORK_OUTPUT:-0}" != 1 ]; then
  for X in r58_wan_draft r58_div r58_field r58_wan_arms; do
    [ -L "$D/$X" ] || { echo "[r58_wan_chain] FATAL: $D/$X is not a symlink to \$SCRATCH -- run: bash jz/r50_scratch_links.sh (R58_ALLOW_WORK_OUTPUT=1 overrides, at the risk of the \$WORK file quota)"; exit 1; }
  done
fi
# the R49 renders the unchanged pairs will reuse must still exist
for X in r48_wan_arms/native_s0.92_full r49_wan_arms/matched59_s0.92_full; do
  [ -d "$D/$X" ] || { echo "[r58_wan_chain] FATAL: $D/$X missing -- the R49 renders to reuse are gone"; exit 1; }
done
# reproducibility gate: r58_repro_check.py must have run on Jean Zay and passed (every row pass=1)
REPRO=evaluation/csv/r58_repro_check.csv
[ -s "$REPRO" ] || { echo "[r58_wan_chain] FATAL: $REPRO missing -- run the repro-render / repro-check steps first"; exit 1; }
if awk -F, 'NR > 1 && $NF != 1 { bad++ } END { exit (bad > 0 || NR < 5) }' "$REPRO"; then :; else
  echo "[r58_wan_chain] FATAL: $REPRO has a failing row or fewer than 4 (config, pair) rows -- the new renders do NOT reproduce R49; do not run the chain"; exit 1
fi

JOBS=logs/r58_wan_chain_jobs.txt
if [ -e "$JOBS" ] && [ "$DRY" -eq 0 ]; then
  echo "[r58_wan_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r58_wan_chain] delete it deliberately to resubmit"; exit 1
fi
[ "$DRY" -eq 0 ] && echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname)" > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch arguments; prints the id, records it, aborts if refused
  local stage=$1; shift
  if [ "$DRY" -eq 1 ]; then
    echo "[r58_wan_chain] dry-run ${stage}: sbatch $*" >&2; echo "9$(printf '%s' "$stage" | cksum | cut -c1-5)"; return
  fi
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r58_wan_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r58_wan_chain] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r58_wan_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r58_wan_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

DR=$(submit draft $SB/r58_wan_draft.sh) || exit 1
DV=$(submit div --dependency=afterok:${DR} --array=0-1 $SB/r58_div.sh) || exit 1
FD=$(submit field --dependency=afterok:${DV} $SB/r58_field.sh) || exit 1
MX=$(submit render_max_s5 --dependency=afterok:${FD} --array=11 $SB/r58_wan_render.sh correct) || exit 1
BA=$(submit render_base --array=0-10 $SB/r58_wan_render.sh correct) || exit 1

if [ "$DRY" -eq 1 ]; then
  echo "[r58_wan_chain] DRY RUN, nothing submitted (the ids above are placeholders): draft ${DR} -> div ${DV} -> field ${FD} -> max_s5 ${MX} | baselines ${BA}"
else
  echo "[r58_wan_chain] submitted: draft ${DR} -> div ${DV} -> field ${FD} -> max_s5 ${MX} | baselines ${BA}"
  echo "[r58_wan_chain] ids in ${JOBS}; follow with: squeue -u \$USER"
fi
