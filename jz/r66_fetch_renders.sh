#!/bin/bash
# R66 step fetch-renders (ON THE LAB CLUSTER'S FRONTAL, gpu-gw, inside tmux): bring the 415 ContextFlow renders (FiVE-Bench-Correct, 41 frames)
# from Jean Zay to the lab /scratch, as ordinary files, so that they are scored on the lab with the R57 suite (every main-paper number is scored on the lab).
#
#   1. $JZ_HOST:$SCRATCH/five_bench_out/r66_contextflow/correct/manifest_n41_shard{0..3}.csv  ->  evaluation/csv/r66_render_manifest_shard{i}.csv
#   2. $JZ_HOST:$SCRATCH/five_bench_out/r66_contextflow/correct/n41/  ->  $R66_DEST/n41/
#      (default R66_DEST=/scratch/PERSO/skhairi/five_bench/r66_contextflow/correct, the root of evaluation/eval_suite/batches/r66.txt)
#
# Only the frames travel: the RF-Solver latents (latents/n41, ~22 MB each), the smoke runs and the nvidia-smi logs stay on Jean Zay (the latents are
# an intermediate, never scored). The Jean Zay tree holds real PNGs; -L is kept as in the R58 / R62 fetch scripts. About 10 GB.
# rsync -rL --partial --size-only --no-perms --no-owner --no-group, NO --delete, NO -t: mtimes are not preserved, so every file carries the COPY date
# and the 30-day /scratch purge counts from it; resumable (rerun), never removes anything on either side.
# R66_DEST must be under the lab /scratch (read by every compute node; never $HOME, ~/Data, /work or /store, which is mounted per node). It needs >= 15 GB free.
# The renders stay on Jean Zay $SCRATCH (IDRIS purges unused files after a retention period): this script never deletes them.
#
# Pre-checks (on Jean Zay): 415 clip dirs under n41, and the 4 shard exit-code files all say 0 (a shard that stopped at an OOM / error is not fetched).
# Post-checks: same PNG count and the same total bytes on both sides, 415 clip dirs here, and every clip holds exactly the n41 PNGs of
# evaluation/r66_cases_correct.json.
#
# Usage: bash jz/r66_fetch_renders.sh            fetch + verify
#        bash jz/r66_fetch_renders.sh --check    send nothing: count the Jean Zay tree and the lab copy
# R50_ENV=<file> overrides jz/r50_jz.env; R66_DEST=<dir> the destination; JZ_SCRATCH=<dir> gives $SCRATCH on Jean Zay if the login-shell lookup fails.
# Ends with "[r66_fetch] failures: N".
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r66_fetch] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
: "${JZ_HOST:?JZ_HOST not set in $CFG}"
R66_DEST=${R66_DEST:-/scratch/PERSO/skhairi/five_bench/r66_contextflow/correct}
R66_DEST=$(readlink -m "$R66_DEST")
case "$R66_DEST" in
  /scratch/*) ;;
  *) echo "[r66_fetch] FATAL: destination $R66_DEST is not under /scratch (never \$HOME, ~/Data, /work or /store)"; echo "[r66_fetch] failures: 1"; exit 1 ;;
esac
CASES=evaluation/r66_cases_correct.json
N_EXPECT=415
N_SHARDS=4

PARENT=$(dirname "$R66_DEST")
[ -d "$(dirname "$PARENT")" ] || { echo "[r66_fetch] FATAL: $(dirname "$PARENT") not visible on $(hostname) -- run this on the lab frontal (gpu-gw) and check that /scratch/PERSO/skhairi/five_bench exists"; echo "[r66_fetch] failures: 1"; exit 1; }
SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
if ! $SSH_CMD "$JZ_HOST" true; then
  echo "[r66_fetch] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname) (run on the lab frontal, not on a compute node)"
  echo "[r66_fetch] failures: 1"; exit 255
fi
# $SCRATCH is set by the login profile, which a non-interactive `ssh host cmd` does not read: ask a login shell, or take JZ_SCRATCH
SRC_ROOT=${JZ_SCRATCH:-}
[ -n "$SRC_ROOT" ] || SRC_ROOT=$($SSH_CMD "$JZ_HOST" 'bash -lc "echo \${SCRATCH:-}"' 2>/dev/null | tail -n 1) || SRC_ROOT=""
[ -n "$SRC_ROOT" ] || { echo "[r66_fetch] FATAL: \$SCRATCH is empty on $JZ_HOST even in a login shell; set JZ_SCRATCH=<path of \$SCRATCH on Jean Zay> and rerun"; echo "[r66_fetch] failures: 1"; exit 1; }
JZ_OUT=$SRC_ROOT/five_bench_out/r66_contextflow/correct
SRC=$JZ_OUT/n41
DEST=$R66_DEST/n41
cd "$REPO" || exit 1
[ -f "$CASES" ] || { echo "[r66_fetch] FATAL: $CASES missing in $REPO"; echo "[r66_fetch] failures: 1"; exit 1; }
echo "[r66_fetch] on $(hostname): $JZ_HOST:$SRC -> $DEST $([ $CHECK -eq 1 ] && echo '(check only)') $(date '+%F %T')"

# clip dirs of a tree (symlinks followed) and "<png files> <bytes>": remote / local
remote_clips() { $SSH_CMD "$JZ_HOST" "find -L '$1' -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l"; }
local_clips()  { find -L "$1" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l; }
remote_png()   { $SSH_CMD "$JZ_HOST" "find -L '$1' -type f -name '*.png' -printf '%s\n' 2>/dev/null | awk '{n++; b+=\$1} END {printf \"%d %d\n\", n, b}'"; }
local_png()    { find -L "$1" -type f -name '*.png' -printf '%s\n' 2>/dev/null | awk '{n++; b+=$1} END {printf "%d %d\n", n, b}'; }
FAILED=0

R_CLIPS=$(remote_clips "$SRC") || R_CLIPS=0
echo "[r66_fetch] Jean Zay: ${R_CLIPS} clip dirs under ${SRC} (expect ${N_EXPECT})"
[ "$R_CLIPS" -ne "$N_EXPECT" ] && { echo "[r66_fetch] FAILED: the render on Jean Zay is not complete (see jz-wait-render)"; FAILED=$((FAILED + 1)); }
for I in $(seq 0 $((N_SHARDS - 1))); do
  CODE=$($SSH_CMD "$JZ_HOST" "cat '$JZ_OUT/exitcode_n41_shard${I}.txt' 2>/dev/null" | tr -d '[:space:]')
  if [ "$CODE" != "0" ]; then
    echo "[r66_fetch] FAILED: shard ${I} exit-code file says '${CODE:-missing}' (0 expected): resubmit that shard on Jean Zay first"; FAILED=$((FAILED + 1))
  fi
done

if [ $CHECK -eq 0 ]; then
  [ "$FAILED" -ne 0 ] && { echo "[r66_fetch] failures: ${FAILED}"; exit 1; }
  FREE_KB=$(df -P "$PARENT" | awk 'NR==2 {print $4}')
  if [ "${FREE_KB:-0}" -lt $((15 * 1024 * 1024)) ]; then
    echo "[r66_fetch] FATAL: only $((${FREE_KB:-0} / 1024 / 1024)) GB free on the filesystem of ${PARENT} (need >= 15 GB)"; echo "[r66_fetch] failures: 1"; exit 1
  fi
  mkdir -p "$DEST" || { echo "[r66_fetch] FATAL: cannot create $DEST"; echo "[r66_fetch] failures: 1"; exit 1; }
  mkdir -p evaluation/csv
  for I in $(seq 0 $((N_SHARDS - 1))); do
    rsync -a -e "$SSH_CMD" "$JZ_HOST:$JZ_OUT/manifest_n41_shard${I}.csv" "evaluation/csv/r66_render_manifest_shard${I}.csv" \
      || { echo "[r66_fetch] FAILED: cannot fetch manifest_n41_shard${I}.csv"; FAILED=$((FAILED + 1)); }
  done
  rsync -rL --partial --size-only --no-perms --no-owner --no-group -e "$SSH_CMD" "$JZ_HOST:$SRC/" "$DEST/"
  RC=$?
  [ $RC -ne 0 ] && { echo "[r66_fetch] FAILED: rsync exit $RC -- rerun this script (resumes)"; FAILED=$((FAILED + 1)); }
fi

# ---- post-check: clip count, PNG count + bytes on both sides, per-clip PNG count = n41 of the case file
L_CLIPS=$(local_clips "$DEST")
echo "[r66_fetch] lab: ${L_CLIPS} clip dirs under ${DEST} (expect ${N_EXPECT})"
R_PNG=$(remote_png "$SRC") || R_PNG="0 0"
L_PNG=$(local_png "$DEST")
echo "[r66_fetch] PNG files / bytes: Jean Zay '${R_PNG}', lab '${L_PNG}'"
if [ $CHECK -eq 0 ]; then
  [ "$L_CLIPS" -ne "$N_EXPECT" ] && FAILED=$((FAILED + 1))
  [ "$R_PNG" != "$L_PNG" ] && { echo "[r66_fetch] FAILED: the two sides differ"; FAILED=$((FAILED + 1)); }
fi
if [ "$L_CLIPS" -gt 0 ]; then
  PYBIN=python3; [ -x "$HOME/anaconda3/envs/five-bench/bin/python" ] && PYBIN=$HOME/anaconda3/envs/five-bench/bin/python
  BAD=$("$PYBIN" -I - "$CASES" "$DEST" <<'EOF'
import json, sys
from pathlib import Path
cases = json.loads(Path(sys.argv[1]).read_text())
dest = Path(sys.argv[2])
bad = []
for c in cases:
    d = dest / f"edit{c['edit_type']}" / c["video_name"]
    got = len(list(d.glob("*.png"))) if d.is_dir() else 0
    if got != int(c["n41"]):
        bad.append(f"{c['case_id']}: {got} PNGs here, {c['n41']} expected")
print(f"[r66_fetch] {len(cases)} pairs checked against {sys.argv[1]}, {len(bad)} with a different PNG count")
for b in bad[:5]:
    print("  MISMATCH " + b)
print(len(bad))
EOF
)
  echo "$BAD" | sed '$d'
  NBAD=$(echo "$BAD" | tail -1)
  [[ "$NBAD" =~ ^[0-9]+$ ]] || { echo "[r66_fetch] FAILED: per-clip check did not run"; NBAD=1; }
  if [ "$CHECK" -eq 0 ] || [ "$L_CLIPS" -eq "$N_EXPECT" ]; then
    [ "$NBAD" -ne 0 ] && FAILED=$((FAILED + 1))
  fi
else
  [ $CHECK -eq 1 ] && echo "[r66_fetch] (nothing on the lab yet: per-clip PNG counts not compared)" || FAILED=$((FAILED + 1))
fi
[ $CHECK -eq 0 ] && [ "$FAILED" -eq 0 ] && echo "[r66_fetch] ${N_EXPECT} clip dirs, every clip has its n41 PNGs"
echo "[r66_fetch] failures: ${FAILED}"
exit $(( FAILED > 0 ))
