#!/bin/bash
# R49 step wan-full-fetch (ON THE LAB CLUSTER'S FRONTAL, gpu-gw): bring the Wan-Edit output folders from Jean Zay $SCRATCH to the lab store.
#
#   $JZ_HOST:$SCRATCH/five_bench_out/<folder>/  ->  $LAB_DEST/<folder>/          (default LAB_DEST=/store/PERSO/skhairi/Outputs/jz_five_bench)
#
# Why the frontal: Jean Zay only accepts SSH from it, and /store is mounted there (it is NOT on every compute node, e.g. nodemm06).
# Why a separate destination: these are H100 renders. The lab's own ~/Data/.../five_bench/r48_wan_arms etc. hold L40S renders; the two platforms
# are never merged (a table never mixes them), so nothing here is linked into ~/Data and no lab folder is touched.
# rsync -a --partial, NO --delete: resumable, never removes anything on either side. After each folder the file count and the total bytes of
# the two sides are compared; the script prints (never runs) the commands that delete the Jean Zay copy -- do that only after "failures: 0".
# Run it inside tmux: ~220k files (a few hundred GB). Eval CSVs travel separately (the rsync of wan-full-fetch, evaluation/csv).
#
# Folders: r48_wan_draft r48_div r48_field r48_wan_arms r49_wan_draft r49_div r49_field r49_field_full r49_wan_arms r50_preflight
#          r51_field_full r51_wan_arms
# Usage: bash jz/r50_fetch_outputs.sh                 all folders
#        bash jz/r50_fetch_outputs.sh r49_wan_arms r48_wan_arms     only these
#        bash jz/r50_fetch_outputs.sh --check         count what is on Jean Zay and in the store, send nothing
# R50_ENV=<file> overrides jz/r50_jz.env; LAB_DEST=<dir> overrides the destination; JZ_SCRATCH=<dir> gives $SCRATCH on Jean Zay if the login-shell lookup fails. Ends with "[r50_fetch] failures: N".
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; ONLY=()
for A in "$@"; do if [ "$A" = "--check" ]; then CHECK=1; else ONLY+=("$A"); fi; done
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r50_fetch] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
: "${JZ_HOST:?JZ_HOST not set in $CFG}"
LAB_DEST=${LAB_DEST:-/store/PERSO/skhairi/Outputs/jz_five_bench}
ALL=(r48_wan_draft r48_div r48_field r48_wan_arms r49_wan_draft r49_div r49_field r49_field_full r49_wan_arms r50_preflight
     r51_field_full r51_wan_arms)
FOLDERS=("${ALL[@]}"); [ "${#ONLY[@]}" -gt 0 ] && FOLDERS=("${ONLY[@]}")
for X in "${FOLDERS[@]}"; do
  case " ${ALL[*]} " in *" $X "*) ;; *) echo "[r50_fetch] FATAL: unknown folder '$X' (choose from: ${ALL[*]})"; exit 1 ;; esac
done

[ -d "$(dirname "$LAB_DEST")" ] || { echo "[r50_fetch] FATAL: $(dirname "$LAB_DEST") not visible on $(hostname) -- /store is mounted per node; run this on the lab frontal (gpu-gw)"; exit 1; }
SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
if ! $SSH_CMD "$JZ_HOST" true; then
  echo "[r50_fetch] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname) (run on the lab frontal, not on a compute node)"
  echo "[r50_fetch] failures: 1"; exit 255
fi
# $SCRATCH is set by the login profile, which a non-interactive `ssh host cmd` does not read: ask a login shell, or take JZ_SCRATCH
# (environment or jz/r50_jz.env), e.g. JZ_SCRATCH=/lustre/fsn1/projects/rech/omt/<user>
SRC_ROOT=${JZ_SCRATCH:-}
[ -n "$SRC_ROOT" ] || SRC_ROOT=$($SSH_CMD "$JZ_HOST" 'bash -lc "echo \${SCRATCH:-}"' 2>/dev/null | tail -n 1) || SRC_ROOT=""
[ -n "$SRC_ROOT" ] || { echo "[r50_fetch] FATAL: \$SCRATCH is empty on $JZ_HOST even in a login shell; set JZ_SCRATCH=<path of \$SCRATCH on Jean Zay> and rerun"; echo "[r50_fetch] failures: 1"; exit 1; }
SRC_ROOT=$SRC_ROOT/five_bench_out
mkdir -p "$LAB_DEST" || { echo "[r50_fetch] FATAL: cannot create $LAB_DEST"; exit 1; }
echo "[r50_fetch] on $(hostname): $JZ_HOST:$SRC_ROOT -> $LAB_DEST $([ $CHECK -eq 1 ] && echo '(check only)') $(date '+%F %T')"

# "<files> <bytes>" of a folder, remote or local
remote_cb() { $SSH_CMD "$JZ_HOST" "find '$1' -type f -printf '%s\\n' 2>/dev/null | awk '{n++; b+=\$1} END {printf \"%d %d\", n+0, b+0}'"; }
local_cb()  { find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{n++; b+=$1} END {printf "%d %d", n+0, b+0}'; }

FAILED=0; OKS=()
for X in "${FOLDERS[@]}"; do
  R=$(remote_cb "$SRC_ROOT/$X") || R="? ?"
  if [ "${R%% *}" = "0" ] || [ "${R%% *}" = "?" ]; then echo "  skip $X: nothing on Jean Zay (${R% *} files)"; continue; fi
  if [ $CHECK -eq 1 ]; then echo "  $X: Jean Zay ${R% *} files / ${R#* } bytes; store $(local_cb "$LAB_DEST/$X")"; continue; fi
  echo "  $X: sending ${R% *} files / ${R#* } bytes"
  rsync -a --partial -e "$SSH_CMD" "$JZ_HOST:$SRC_ROOT/$X/" "$LAB_DEST/$X/"
  RC=$?
  L=$(local_cb "$LAB_DEST/$X")
  if [ $RC -eq 0 ] && [ "$R" = "$L" ]; then echo "  ok: $X  ${L% *} files, ${L#* } bytes identical on both sides"; OKS+=("$X")
  else echo "  FAILED: $X rsync exit $RC, Jean Zay '$R' vs store '$L' (files bytes) -- rerun this script (resumes)"; FAILED=$((FAILED + 1)); fi
done

if [ ${#OKS[@]} -gt 0 ]; then
  echo "[r50_fetch] verified folders: ${OKS[*]}"
  echo "[r50_fetch] free the Jean Zay copy ONLY after the CSVs are scored and these lines read 'failures: 0'. On the Jean Zay frontal (not run here):"
  for X in "${OKS[@]}"; do echo "    rm -rf \"\$SCRATCH/five_bench_out/$X\"/* "; done
  echo "[r50_fetch] (the \$WORK symlinks to \$SCRATCH stay valid; to start a fresh run rerun jz/r50_scratch_links.sh)"
fi
echo "[r50_fetch] failures: ${FAILED}"
exit $(( FAILED > 0 ))
