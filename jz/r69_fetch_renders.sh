#!/bin/bash
# R69 step fetch-renders (ON THE LAB CLUSTER'S FRONTAL, gpu-gw, inside tmux): bring the Wan-Edit equal-step point matched59_s0.5 (415 Correct pairs)
# from Jean Zay to the lab /scratch, as ordinary files, so that it is scored on the lab with the R57 suite (every main-paper number is scored on the lab).
#
#   $JZ_HOST:$SCRATCH/five_bench_out/r69_wan_arms/matched59_s0.5_full/  ->  $R69_DEST/matched59_s0.5/   (default R69_DEST=/scratch/PERSO/skhairi/five_bench/r69_wan_correct)
#
# No assembly step: every clip is a new render (no reuse), so the render folder is fetched as is (~9 GB).
# rsync -rL --partial --size-only --no-perms --no-owner --no-group, NO --delete, NO -t (as jz/r62_fetch_renders.sh): mtimes are not preserved, so every
# file carries the COPY date and the 30-day /scratch purge counts from it; resumable, never removes anything on either side.
# Post-check: 415 clip dirs under $R69_DEST/matched59_s0.5/edit{T}/{video}, and every case of r43_anchor_manifest_correct.json has a complete render
# (r48_units.py --missing: r48_wan_common.expected_frames PNGs, the same guard as the render job).
# R69_DEST must be under the lab /scratch (read by every compute node; never $HOME, ~/Data, /work or /store, which is mounted per node).
# It needs >= 12 GB free.
# The renders stay on Jean Zay $SCRATCH (IDRIS purges unused files after a retention period): this script never deletes them.
#
# Usage: bash jz/r69_fetch_renders.sh            fetch + verify
#        bash jz/r69_fetch_renders.sh --check    send nothing: count the Jean Zay tree and the lab copy
# R50_ENV=<file> overrides jz/r50_jz.env; R69_DEST=<dir> the destination; JZ_SCRATCH=<dir> gives $SCRATCH on Jean Zay if the login-shell lookup fails.
# Ends with "[r69_fetch] failures: N".
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r69_fetch] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
: "${JZ_HOST:?JZ_HOST not set in $CFG}"
R69_DEST=${R69_DEST:-/scratch/PERSO/skhairi/five_bench/r69_wan_correct}
R69_DEST=$(readlink -m "$R69_DEST")
case "$R69_DEST" in
  /scratch/*) ;;
  *) echo "[r69_fetch] FATAL: destination $R69_DEST is not under /scratch (never \$HOME, ~/Data, /work or /store)"; echo "[r69_fetch] failures: 1"; exit 1 ;;
esac
CONFIG=matched59_s0.5
CASES=evaluation/r43_anchor_manifest_correct.json
PY_FB=$HOME/anaconda3/envs/five-bench/bin/python
N_EXPECT=415

PARENT=$(dirname "$R69_DEST")
[ -d "$PARENT" ] || { echo "[r69_fetch] FATAL: $PARENT not visible on $(hostname) -- run this on the lab frontal (gpu-gw) and check that /scratch/PERSO/skhairi/five_bench exists"; echo "[r69_fetch] failures: 1"; exit 1; }
[ -x "$PY_FB" ] || { echo "[r69_fetch] FATAL: $PY_FB missing (needed for the post-check)"; echo "[r69_fetch] failures: 1"; exit 1; }
SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
if ! $SSH_CMD "$JZ_HOST" true; then
  echo "[r69_fetch] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname) (run on the lab frontal, not on a compute node)"
  echo "[r69_fetch] failures: 1"; exit 255
fi
# $SCRATCH is set by the login profile, which a non-interactive `ssh host cmd` does not read: ask a login shell, or take JZ_SCRATCH
SRC_ROOT=${JZ_SCRATCH:-}
[ -n "$SRC_ROOT" ] || SRC_ROOT=$($SSH_CMD "$JZ_HOST" 'bash -lc "echo \${SCRATCH:-}"' 2>/dev/null | tail -n 1) || SRC_ROOT=""
[ -n "$SRC_ROOT" ] || { echo "[r69_fetch] FATAL: \$SCRATCH is empty on $JZ_HOST even in a login shell; set JZ_SCRATCH=<path of \$SCRATCH on Jean Zay> and rerun"; echo "[r69_fetch] failures: 1"; exit 1; }
SRC=$SRC_ROOT/five_bench_out/r69_wan_arms/${CONFIG}_full
DEST=$R69_DEST/$CONFIG
cd "$REPO" || exit 1
echo "[r69_fetch] on $(hostname): $JZ_HOST:$SRC -> $DEST $([ $CHECK -eq 1 ] && echo '(check only)') $(date '+%F %T')"

# clip dirs of a tree (symlinks followed): remote / local
remote_clips() { $SSH_CMD "$JZ_HOST" "find -L '$1' -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l"; }
local_clips()  { find -L "$1" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l; }
FAILED=0

R_CLIPS=$(remote_clips "$SRC") || R_CLIPS=0
echo "[r69_fetch] Jean Zay: ${R_CLIPS} clip dirs under ${SRC} (expect ${N_EXPECT})"
[ "$R_CLIPS" -ne "$N_EXPECT" ] && { echo "[r69_fetch] FAILED: the render on Jean Zay is not complete -- check the r69_wan_render log ('failures: 0', 415/415) or resubmit it"; FAILED=$((FAILED + 1)); }

if [ $CHECK -eq 0 ]; then
  [ "$FAILED" -ne 0 ] && { echo "[r69_fetch] failures: ${FAILED}"; exit 1; }
  FREE_KB=$(df -P "$PARENT" | awk 'NR==2 {print $4}')
  if [ "${FREE_KB:-0}" -lt $((12 * 1024 * 1024)) ]; then
    echo "[r69_fetch] FATAL: only $((${FREE_KB:-0} / 1024 / 1024)) GB free on the filesystem of ${PARENT} (need >= 12 GB)"; echo "[r69_fetch] failures: 1"; exit 1
  fi
  mkdir -p "$DEST" || { echo "[r69_fetch] FATAL: cannot create $DEST"; echo "[r69_fetch] failures: 1"; exit 1; }
  rsync -rL --partial --size-only --no-perms --no-owner --no-group -e "$SSH_CMD" "$JZ_HOST:$SRC/" "$DEST/"
  RC=$?
  [ $RC -ne 0 ] && { echo "[r69_fetch] FAILED: rsync exit $RC -- rerun this script (resumes)"; FAILED=$((FAILED + 1)); }
fi

# ---- post-check: clip count, and every case clip has a complete render (the render job's own guard)
L_CLIPS=$(local_clips "$DEST")
echo "[r69_fetch] lab: ${L_CLIPS} clip dirs under ${DEST} (expect ${N_EXPECT})"
if [ -d "$DEST" ]; then
  MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$DEST" --cases "$CASES") || { echo "[r69_fetch] FAILED: r48_units.py --missing did not run"; FAILED=$((FAILED + 1)); MISSING=""; }
  N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
  echo "[r69_fetch] $((N_EXPECT - N_MISS))/${N_EXPECT} case clips complete here"
  [ "$N_MISS" -ne 0 ] && echo "  incomplete, e.g. $(echo "$MISSING" | head -5 | tr '\n' ' ')"
  [ $CHECK -eq 0 ] && [ "$N_MISS" -ne 0 ] && FAILED=$((FAILED + 1))
fi
[ $CHECK -eq 0 ] && [ "$L_CLIPS" -ne "$N_EXPECT" ] && FAILED=$((FAILED + 1))
echo "[r69_fetch] failures: ${FAILED}"
exit $(( FAILED > 0 ))
