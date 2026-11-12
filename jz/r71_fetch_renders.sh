#!/bin/bash
# R71 step fetch-renders (ON THE LAB CLUSTER'S FRONTAL, gpu-gw, inside tmux): bring the 11 Wan-Edit configs of R71 (normalized mapping,
# 415 Correct pairs each) from Jean Zay to the lab /scratch, as ordinary files, so that they are scored on the lab with the R57 suite.
#
#   $JZ_HOST:$SCRATCH/five_bench_out/r71_wan_arms/{cfg}_full/  ->  $R71_DEST/{cfg}/   (default R71_DEST=/scratch/PERSO/skhairi/five_bench/r71_wan_correct)
#
# Configs: max_s1 nosupp_s1 nodiv_s1 dino_s1 lpips_s1 (adaptive A0-A4) and matched63_s{0.996,0.978,0.95,0.92,0.85,0.5} (equal-step curve);
# all of them by default, or only those given as arguments. Every clip is a new render (no reuse): folders are fetched as they are (~9 GB each).
# rsync -rL --partial --size-only --no-perms --no-owner --no-group, NO --delete, NO -t (as jz/r69_fetch_renders.sh): mtimes are not preserved, so
# every file carries the COPY date and the 30-day /scratch purge counts from it; resumable, never removes anything on either side.
# Post-check per config: 415 clip dirs under $R71_DEST/{cfg}/edit{T}/{video}, and every case of r43_anchor_manifest_correct.json has a complete
# render (r48_units.py --missing: r48_wan_common.expected_frames PNGs, the same guard as the render job).
# R71_DEST must be under the lab /scratch (read by every compute node; never $HOME, ~/Data, /work or /store, which is mounted per node).
# It needs >= 12 GB free per requested config.
# The renders stay on Jean Zay $SCRATCH (IDRIS purges unused files after a retention period): this script never deletes them.
#
# Usage: bash jz/r71_fetch_renders.sh [cfg ...]            fetch + verify (all 11 configs, or the ones given)
#        bash jz/r71_fetch_renders.sh --check [cfg ...]    send nothing: count the Jean Zay trees and the lab copies
# R50_ENV=<file> overrides jz/r50_jz.env; R71_DEST=<dir> the destination; JZ_SCRATCH=<dir> gives $SCRATCH on Jean Zay if the login-shell lookup fails.
# Ends with "[r71_fetch] failures: N".
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; [ "${1:-}" = "--check" ] && { CHECK=1; shift; }
ALL_CONFIGS=(max_s1 nosupp_s1 nodiv_s1 dino_s1 lpips_s1
             matched63_s0.996 matched63_s0.978 matched63_s0.95 matched63_s0.92 matched63_s0.85 matched63_s0.5)
if [ $# -gt 0 ]; then CONFIGS=("$@"); else CONFIGS=("${ALL_CONFIGS[@]}"); fi
for c in "${CONFIGS[@]}"; do
  ok=0; for a in "${ALL_CONFIGS[@]}"; do [ "$c" = "$a" ] && ok=1; done
  [ $ok -eq 1 ] || { echo "[r71_fetch] FATAL: unknown config '$c' (known: ${ALL_CONFIGS[*]})"; echo "[r71_fetch] failures: 1"; exit 1; }
done
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r71_fetch] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
: "${JZ_HOST:?JZ_HOST not set in $CFG}"
R71_DEST=${R71_DEST:-/scratch/PERSO/skhairi/five_bench/r71_wan_correct}
R71_DEST=$(readlink -m "$R71_DEST")
case "$R71_DEST" in
  /scratch/*) ;;
  *) echo "[r71_fetch] FATAL: destination $R71_DEST is not under /scratch (never \$HOME, ~/Data, /work or /store)"; echo "[r71_fetch] failures: 1"; exit 1 ;;
esac
CASES=evaluation/r43_anchor_manifest_correct.json
PY_FB=$HOME/anaconda3/envs/five-bench/bin/python
N_EXPECT=415
GB_PER_CONFIG=12

PARENT=$(dirname "$R71_DEST")
[ -d "$PARENT" ] || { echo "[r71_fetch] FATAL: $PARENT not visible on $(hostname) -- run this on the lab frontal (gpu-gw) and check that /scratch/PERSO/skhairi/five_bench exists"; echo "[r71_fetch] failures: 1"; exit 1; }
[ -x "$PY_FB" ] || { echo "[r71_fetch] FATAL: $PY_FB missing (needed for the post-check)"; echo "[r71_fetch] failures: 1"; exit 1; }
SSH_CMD="ssh -o ConnectTimeout=20 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-r50-%r@%h:%p -o ControlPersist=10m"
if ! $SSH_CMD "$JZ_HOST" true; then
  echo "[r71_fetch] FAILED: cannot open an SSH connection to $JZ_HOST from $(hostname) (run on the lab frontal, not on a compute node)"
  echo "[r71_fetch] failures: 1"; exit 255
fi
# $SCRATCH is set by the login profile, which a non-interactive `ssh host cmd` does not read: ask a login shell, or take JZ_SCRATCH
SRC_ROOT=${JZ_SCRATCH:-}
[ -n "$SRC_ROOT" ] || SRC_ROOT=$($SSH_CMD "$JZ_HOST" 'bash -lc "echo \${SCRATCH:-}"' 2>/dev/null | tail -n 1) || SRC_ROOT=""
[ -n "$SRC_ROOT" ] || { echo "[r71_fetch] FATAL: \$SCRATCH is empty on $JZ_HOST even in a login shell; set JZ_SCRATCH=<path of \$SCRATCH on Jean Zay> and rerun"; echo "[r71_fetch] failures: 1"; exit 1; }
cd "$REPO" || exit 1
echo "[r71_fetch] on $(hostname): $JZ_HOST:$SRC_ROOT/five_bench_out/r71_wan_arms/{cfg}_full -> $R71_DEST/{cfg}, ${#CONFIGS[@]} config(s) $([ $CHECK -eq 1 ] && echo '(check only)') $(date '+%F %T')"

# clip dirs of a tree (symlinks followed): remote / local
remote_clips() { $SSH_CMD "$JZ_HOST" "find -L '$1' -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l"; }
local_clips()  { find -L "$1" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l; }
FAILED=0

# ---- remote completeness of every requested config, before anything is sent
READY=()
for c in "${CONFIGS[@]}"; do
  SRC=$SRC_ROOT/five_bench_out/r71_wan_arms/${c}_full
  R_CLIPS=$(remote_clips "$SRC") || R_CLIPS=0
  echo "[r71_fetch] Jean Zay ${c}: ${R_CLIPS} clip dirs (expect ${N_EXPECT})"
  if [ "$R_CLIPS" -ne "$N_EXPECT" ]; then
    echo "[r71_fetch] FAILED: ${c} is not complete on Jean Zay -- check its r71_wan_render log ('failures: 0', 415/415) or resubmit it"
    FAILED=$((FAILED + 1))
  else
    READY+=("$c")
  fi
done

if [ $CHECK -eq 0 ]; then
  [ "$FAILED" -ne 0 ] && { echo "[r71_fetch] nothing sent: fix the incomplete config(s) above or pass only the complete ones as arguments"; echo "[r71_fetch] failures: ${FAILED}"; exit 1; }
  FREE_KB=$(df -P "$PARENT" | awk 'NR==2 {print $4}')
  NEED_GB=$(( GB_PER_CONFIG * ${#READY[@]} ))
  if [ "${FREE_KB:-0}" -lt $(( NEED_GB * 1024 * 1024 )) ]; then
    echo "[r71_fetch] FATAL: only $((${FREE_KB:-0} / 1024 / 1024)) GB free on the filesystem of ${PARENT} (need >= ${NEED_GB} GB for ${#READY[@]} config(s))"; echo "[r71_fetch] failures: 1"; exit 1
  fi
  for c in "${READY[@]}"; do
    SRC=$SRC_ROOT/five_bench_out/r71_wan_arms/${c}_full
    DEST=$R71_DEST/$c
    mkdir -p "$DEST" || { echo "[r71_fetch] FATAL: cannot create $DEST"; FAILED=$((FAILED + 1)); continue; }
    echo "[r71_fetch] ${c}: rsync $(date '+%T')"
    rsync -rL --partial --size-only --no-perms --no-owner --no-group -e "$SSH_CMD" "$JZ_HOST:$SRC/" "$DEST/"
    RC=$?
    [ $RC -ne 0 ] && { echo "[r71_fetch] FAILED: ${c}: rsync exit $RC -- rerun this script (resumes)"; FAILED=$((FAILED + 1)); }
  done
fi

# ---- post-check per config: clip count, and every case clip has a complete render (the render job's own guard)
for c in "${CONFIGS[@]}"; do
  DEST=$R71_DEST/$c
  L_CLIPS=$(local_clips "$DEST")
  echo "[r71_fetch] lab ${c}: ${L_CLIPS} clip dirs under ${DEST} (expect ${N_EXPECT})"
  if [ -d "$DEST" ]; then
    MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$DEST" --cases "$CASES") || { echo "[r71_fetch] FAILED: ${c}: r48_units.py --missing did not run"; FAILED=$((FAILED + 1)); MISSING=""; }
    N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
    echo "[r71_fetch] ${c}: $((N_EXPECT - N_MISS))/${N_EXPECT} case clips complete here"
    [ "$N_MISS" -ne 0 ] && echo "  incomplete, e.g. $(echo "$MISSING" | head -5 | tr '\n' ' ')"
    [ $CHECK -eq 0 ] && [ "$N_MISS" -ne 0 ] && FAILED=$((FAILED + 1))
  fi
  [ $CHECK -eq 0 ] && [ "$L_CLIPS" -ne "$N_EXPECT" ] && FAILED=$((FAILED + 1))
done
echo "[r71_fetch] failures: ${FAILED}"
exit $(( FAILED > 0 ))
