#!/bin/bash
# R62 step fetch-renders (ON THE LAB CLUSTER'S FRONTAL, gpu-gw, inside tmux): bring R51's Wan-Edit A4 (nosupp_s5) renders for the 415 Correct pairs
# from Jean Zay to the lab, as ordinary files, so that they are scored on the lab with the R57 suite (every main-paper number is scored on the lab).
#
#   1. $JZ_REPO/evaluation/csv/r62_wan_assembly.csv   ->  evaluation/csv/r62_wan_assembly.csv      (written by r62_assemble.py on Jean Zay)
#   2. $JZ_HOST:$SCRATCH/five_bench_out/r62_wan_correct/nosupp_s5/  ->  $R62_DEST/nosupp_s5/          (default R62_DEST=/work/PERSO/skhairi/five_bench/r62_wan_correct)
#
# The Jean Zay tree is made of symlinks (190 to the new renders, 225 to R51's): rsync -aL follows them on the sending side, so the lab receives
# real PNGs, about 9.3 GB. rsync --partial, NO --delete: resumable, never removes anything on either side.
# Post-check: 415 clip dirs under $R62_DEST/nosupp_s5/edit{T}/{video} and every clip's PNG count equals n_png in r62_wan_assembly.csv.
# R62_DEST must be writable from the frontal AND readable from the L40S compute nodes (not /store, which is mounted per node): see R58's
# probe-mounts step. It needs >= 12 GB free.
# The renders stay on Jean Zay $SCRATCH (IDRIS purges unused files after a retention period): this script never deletes them.
#
# Usage: bash jz/r62_fetch_renders.sh            fetch + verify
#        bash jz/r62_fetch_renders.sh --check    send nothing: count the Jean Zay tree and the lab copy, compare with the CSV if it is already here
# R50_ENV=<file> overrides jz/r50_jz.env; R62_DEST=<dir> the destination; JZ_SCRATCH=<dir> gives $SCRATCH on Jean Zay if the login-shell lookup fails.
# Ends with "[r62_fetch] failures: N".
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r62_fetch] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
: "${JZ_HOST:?JZ_HOST not set in $CFG}"
: "${JZ_REPO:?JZ_REPO not set in $CFG}"
R62_DEST=${R62_DEST:-/work/PERSO/skhairi/five_bench/r62_wan_correct}
CONFIG=nosupp_s5
CSV=evaluation/csv/r62_wan_assembly.csv
N_EXPECT=415

PARENT=$(dirname "$R62_DEST")
[ -d "$PARENT" ] || { echo "[r62_fetch] FATAL: $PARENT not visible on $(hostname) -- run this on the lab frontal (gpu-gw) and check R58's probe-mounts / R62_DEST"; echo "[r62_fetch] failures: 1"; exit 1; }
SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
if ! $SSH_CMD "$JZ_HOST" true; then
  echo "[r62_fetch] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname) (run on the lab frontal, not on a compute node)"
  echo "[r62_fetch] failures: 1"; exit 255
fi
# $SCRATCH is set by the login profile, which a non-interactive `ssh host cmd` does not read: ask a login shell, or take JZ_SCRATCH
SRC_ROOT=${JZ_SCRATCH:-}
[ -n "$SRC_ROOT" ] || SRC_ROOT=$($SSH_CMD "$JZ_HOST" 'bash -lc "echo \${SCRATCH:-}"' 2>/dev/null | tail -n 1) || SRC_ROOT=""
[ -n "$SRC_ROOT" ] || { echo "[r62_fetch] FATAL: \$SCRATCH is empty on $JZ_HOST even in a login shell; set JZ_SCRATCH=<path of \$SCRATCH on Jean Zay> and rerun"; echo "[r62_fetch] failures: 1"; exit 1; }
SRC=$SRC_ROOT/five_bench_out/r62_wan_correct/$CONFIG
DEST=$R62_DEST/$CONFIG
cd "$REPO" || exit 1
echo "[r62_fetch] on $(hostname): $JZ_HOST:$SRC -> $DEST $([ $CHECK -eq 1 ] && echo '(check only)') $(date '+%F %T')"

# clip dirs of a tree (symlinks followed): remote / local
remote_clips() { $SSH_CMD "$JZ_HOST" "find -L '$1' -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l"; }
local_clips()  { find -L "$1" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l; }
FAILED=0

R_CLIPS=$(remote_clips "$SRC") || R_CLIPS=0
echo "[r62_fetch] Jean Zay: ${R_CLIPS} clip dirs under ${SRC} (expect ${N_EXPECT})"
[ "$R_CLIPS" -ne "$N_EXPECT" ] && { echo "[r62_fetch] FAILED: the assembled tree on Jean Zay is not complete -- run r62_assemble.py there first"; FAILED=$((FAILED + 1)); }

if [ $CHECK -eq 0 ]; then
  [ "$FAILED" -ne 0 ] && { echo "[r62_fetch] failures: ${FAILED}"; exit 1; }
  FREE_KB=$(df -P "$PARENT" | awk 'NR==2 {print $4}')
  if [ "${FREE_KB:-0}" -lt $((12 * 1024 * 1024)) ]; then
    echo "[r62_fetch] FATAL: only $((${FREE_KB:-0} / 1024 / 1024)) GB free on the filesystem of ${PARENT} (need >= 12 GB)"; echo "[r62_fetch] failures: 1"; exit 1
  fi
  mkdir -p "$DEST" || { echo "[r62_fetch] FATAL: cannot create $DEST"; echo "[r62_fetch] failures: 1"; exit 1; }
  rsync -a -e "$SSH_CMD" "$JZ_HOST:$JZ_REPO/$CSV" "$CSV" || { echo "[r62_fetch] FAILED: cannot fetch $CSV from $JZ_REPO"; echo "[r62_fetch] failures: 1"; exit 1; }
  echo "[r62_fetch] fetched $CSV ($(($(wc -l < "$CSV") - 1)) rows)"
  rsync -aL --partial -e "$SSH_CMD" "$JZ_HOST:$SRC/" "$DEST/"
  RC=$?
  [ $RC -ne 0 ] && { echo "[r62_fetch] FAILED: rsync exit $RC -- rerun this script (resumes)"; FAILED=$((FAILED + 1)); }
fi

# ---- post-check against the assembly CSV (case_id,source,n_png,need): clip count and per-clip PNG count
L_CLIPS=$(local_clips "$DEST")
echo "[r62_fetch] lab: ${L_CLIPS} clip dirs under ${DEST} (expect ${N_EXPECT})"
if [ -s "$CSV" ]; then
  BAD=0; N_ROWS=0
  while IFS=, read -r CASE SOURCE N_PNG NEED; do
    N_ROWS=$((N_ROWS + 1))
    N=$(ls "$DEST/$CASE"/*.png 2>/dev/null | wc -l)
    [ "$N" -ne "$N_PNG" ] && { BAD=$((BAD + 1)); [ "$BAD" -le 5 ] && echo "  MISMATCH ${CASE} (${SOURCE}): ${N} PNGs here, ${N_PNG} in the assembly CSV"; }
  done < <(tail -n +2 "$CSV")
  echo "[r62_fetch] ${N_ROWS} CSV rows checked, ${BAD} clip(s) with a different PNG count"
  [ "$N_ROWS" -ne "$N_EXPECT" ] && FAILED=$((FAILED + 1))
  [ "$BAD" -ne 0 ] && FAILED=$((FAILED + 1))
elif [ $CHECK -eq 1 ]; then
  echo "[r62_fetch] ($CSV not here yet: per-clip PNG counts not compared)"
else
  FAILED=$((FAILED + 1)); echo "[r62_fetch] FAILED: $CSV missing"
fi
[ $CHECK -eq 0 ] && [ "$L_CLIPS" -ne "$N_EXPECT" ] && FAILED=$((FAILED + 1))
echo "[r62_fetch] failures: ${FAILED}"
exit $(( FAILED > 0 ))
