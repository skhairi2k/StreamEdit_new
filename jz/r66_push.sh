#!/bin/bash
# R66 step jz-push (ON THE LAB CLUSTER'S FRONTAL, gpu-gw, inside tmux): send what ContextFlow needs on Jean Zay and is not there yet,
# straight from where it lives on the lab (no staging copy), in one SSH connection:
#
#   weights  /work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P                 ->  $JZ_WORK/weights/Wan2.1-I2V-14B-480P                 (77 GB, 33 files)
#   anchors  ~/Data/dataggen/outputs/five_bench/anchors_r43_correct          ->  $JZ_WORK/Data/dataggen/outputs/five_bench/anchors_r43_correct
#                                                                                (222 MB, 416 PNG; ~/Data -> $WORK/Data on Jean Zay)
# The FiVE source frames are already on Jean Zay (R50: $WORK/Data/FiVE-Fine-Grained-Video-Editing-Benchmark); code goes through git;
# the ContextFlow clone and the env are made on Jean Zay (jz-env). The weights' .cache/ (Hugging Face download metadata) is not sent.
#
# rsync -a --partial, no --delete: an interrupted transfer resumes on rerun; files already identical on Jean Zay (size + mtime) are skipped;
# nothing is ever removed on either side. -L on the anchors so a symlinked file arrives as a real PNG.
# Verify, per part: same number of files and same total bytes (sum of file sizes, not du) on both sides, plus sha256 of the weights'
# diffusion_pytorch_model.safetensors.index.json and config.json. The lab weights are deleted later, by hand, at the lab-cleanup step.
#
# Usage: bash jz/r66_push.sh 2>&1 | tee logs/r66_push.log     send + verify
#        bash jz/r66_push.sh --check                          send nothing: compare the two sides only
# R50_ENV=<file> overrides jz/r50_jz.env (JZ_HOST, JZ_WORK). Ends with "[r66_push] failures: N".
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r66_push] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
: "${JZ_HOST:?JZ_HOST not set in $CFG}" "${JZ_WORK:?JZ_WORK not set in $CFG}"

NAMES=(weights anchors)
SRC_weights=/work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P
DST_weights=$JZ_WORK/weights/Wan2.1-I2V-14B-480P
SRC_anchors=$HOME/Data/dataggen/outputs/five_bench/anchors_r43_correct
DST_anchors=$JZ_WORK/Data/dataggen/outputs/five_bench/anchors_r43_correct
SHA_FILES=(diffusion_pytorch_model.safetensors.index.json config.json)

SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
echo "[r66_push] on $(hostname) $([ $CHECK -eq 1 ] && echo '(check only)') $(date '+%F %T')"
for N in "${NAMES[@]}"; do
  S=SRC_$N
  [ -d "${!S}" ] || { echo "[r66_push] FATAL: ${!S} not found on $(hostname) -- run this on the lab frontal"; echo "[r66_push] failures: 1"; exit 1; }
done
# Jean Zay only accepts SSH from registered machines (the lab frontal): non-interactive test, 10 s per address of
# jean-zay.idris.fr (~1 min in all from a compute node such as node57, measured) instead of a silent hang
if ! $SSH_CMD -o BatchMode=yes -o ConnectTimeout=10 "$JZ_HOST" true; then
  echo "[r66_push] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname) (run on the lab frontal, not on a compute node)"
  echo "[r66_push] failures: 1"; exit 255
fi
echo "[r66_push] Jean Zay quota (idr_quota_user, for information):"
$SSH_CMD "$JZ_HOST" 'bash -lc "idr_quota_user 2>/dev/null || echo \"  (idr_quota_user not available)\""' | sed 's/^/  /'

# "<files> <bytes>" of a tree, .cache excluded: local / remote
count_local()  { find -L "$1" -type f -not -path '*/.cache/*' -printf '%s\n' 2>/dev/null | awk '{n++; b+=$1} END {printf "%d %d\n", n, b}'; }
count_remote() { $SSH_CMD "$JZ_HOST" "find -L '$1' -type f -not -path '*/.cache/*' -printf '%s\n' 2>/dev/null | awk '{n++; b+=\$1} END {printf \"%d %d\n\", n, b}'"; }

FAILED=0
for N in "${NAMES[@]}"; do
  S=SRC_$N; D=DST_$N; SRC=${!S}; DST=${!D}
  echo "==== $N: $SRC -> $JZ_HOST:$DST ===="
  if [ $CHECK -eq 0 ]; then
    $SSH_CMD "$JZ_HOST" "mkdir -p '$DST'" || { echo "[r66_push] FAILED: mkdir $DST on $JZ_HOST"; FAILED=$((FAILED + 1)); continue; }
    LINKS=(); [ "$N" = anchors ] && LINKS=(-L)
    T0=$(date +%s)
    rsync -a "${LINKS[@]+"${LINKS[@]}"}" --partial --exclude='.cache/' --info=progress2 --stats --human-readable \
      -e "$SSH_CMD" "$SRC/" "$JZ_HOST:$DST/"
    RC=$?
    echo "[r66_push] $N: rsync exit $RC, elapsed $(( $(date +%s) - T0 )) s"
    [ "$RC" -ne 0 ] && { echo "[r66_push] FAILED: $N rsync exit $RC (rerun: --partial resumes)"; FAILED=$((FAILED + 1)); continue; }
  fi
  L=$(count_local "$SRC"); R=$(count_remote "$DST") || R="0 0"
  if [ "$L" = "$R" ]; then
    echo "[r66_push] $N: ${L% *} files, ${L#* } bytes: identical on Jean Zay"
  else
    echo "[r66_push] FAILED: $N differs: lab '${L}' vs Jean Zay '${R}' (files bytes)"; FAILED=$((FAILED + 1))
  fi
done

# the weights' small descriptive files must match byte for byte
for F in "${SHA_FILES[@]}"; do
  LS=$(sha256sum "$SRC_weights/$F" 2>/dev/null | cut -d' ' -f1)
  RS=$($SSH_CMD "$JZ_HOST" "sha256sum '$DST_weights/$F' 2>/dev/null | cut -d' ' -f1")
  if [ -n "$LS" ] && [ "$LS" = "$RS" ]; then
    echo "[r66_push] sha256 $F: identical"
  else
    echo "[r66_push] FAILED: sha256 $F: lab '${LS:-missing}' vs Jean Zay '${RS:-missing}'"; FAILED=$((FAILED + 1))
  fi
done
echo "[r66_push] failures: ${FAILED}"
exit $(( FAILED > 0 ))
