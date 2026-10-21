#!/bin/bash
# R50 step push (this cluster): send the bundle stage and the filled config to Jean Zay in ONE rsync connection.
#
#   $LOCAL_STAGE/  ->  $JZ_HOST:$JZ_WORK/r50_stage/     (plus r50_jz.env at its root, for jz-clone-setup)
#
# --copy-unsafe-links: the stage's links point outside it, so their targets are sent as real files; the Hugging
#                      Face cache's internal relative links (snapshots -> blobs) stay links, so nothing doubles.
# --partial:           an interrupted transfer resumes where it stopped -- just rerun this script.
# No --delete: nothing on Jean Zay is ever removed by this script.
#
# Usage: bash jz/r50_push.sh               (35 GiB the first time; run it inside tmux/screen, it can take a while)
#        bash jz/r50_push.sh --dry-run     (extra arguments go to rsync: lists what would be sent, sends nothing)
# Rerun after a new `make-bundle`: only the changed files (e.g. CODE.sha256) are sent.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG="$REPO/jz/r50_jz.env"
[ -f "$CFG" ] || { echo "[r50_push] FATAL: $CFG missing -- cp jz/r50_jz.env.example jz/r50_jz.env and fill it"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep -q '<'; then
  echo "[r50_push] FATAL: unfilled <...> value in $CFG:"; set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep '<'; exit 1
fi
STAGE=${LOCAL_STAGE:?LOCAL_STAGE not set in $CFG}
: "${JZ_HOST:?JZ_HOST not set in $CFG}" "${JZ_WORK:?JZ_WORK not set in $CFG}"

for F in MANIFEST.sha256 CODE.sha256 freeze/five-bench.txt; do
  [ -s "$STAGE/$F" ] || { echo "[r50_push] FATAL: $STAGE/$F missing -- run bash jz/r50_make_bundle.sh first"; exit 1; }
done

# the filled config travels inside the stage (a real file, not a link), so one connection sends everything
cp "$CFG" "$STAGE/r50_jz.env"

DEST="$JZ_HOST:$JZ_WORK/r50_stage/"
# one SSH connection for the test and the rsync (ControlMaster): at most one password / key prompt
SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
echo "[r50_push] on $(hostname): $STAGE/ -> $DEST  $(date '+%F %T')"
# Jean Zay only accepts SSH from registered machines (the lab frontal, not compute nodes such as nodemm06):
# test the connection first so a wrong machine fails in 20 s instead of hanging silently.
if ! $SSH_CMD "$JZ_HOST" true; then
  echo "[r50_push] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname)."
  echo "[r50_push]   run this on the lab cluster's frontal (e.g. via 'ssh -A <lab-frontal>' from your laptop), not on a compute node"
  echo "[r50_push] failures: 1"; exit 255
fi
T0=$(date +%s)
set +e
rsync -a --copy-unsafe-links --partial --info=progress2 --stats --human-readable -e "$SSH_CMD" "$@" "$STAGE/" "$DEST"
RC=$?
set -e
echo "[r50_push] rsync exit $RC, elapsed $(( $(date +%s) - T0 )) s"
case "$RC" in
  0)  echo "[r50_push] OK -- next, on the Jean Zay frontal: git clone (see jz/r50_README.md step 4)" ;;
  23|24) echo "[r50_push] PARTIAL transfer (some files not sent or vanished) -- rerun bash jz/r50_push.sh" ;;
  *)  echo "[r50_push] FAILED -- check ssh access to $JZ_HOST, then rerun (resumes with --partial)" ;;
esac
echo "[r50_push] failures: $(( RC != 0 ))"
exit "$RC"
