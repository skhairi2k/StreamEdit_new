#!/bin/bash
# R50/R49 (ON JEAN ZAY, in the clone): send the Wan-Edit output folders to $SCRATCH without touching a job script.
#
# The job scripts write to D=~/Data/dataggen/outputs/five_bench/<folder>; ~/Data is a link to $WORK/Data. The full Wan stage is ~220k PNG
# files (12 render arms x 419 pairs x 41 frames, plus drafts), which the 500k-file quota of $WORK cannot hold next to the overlay and the
# bundle. For every folder below this script creates $SCRATCH/five_bench_out/<folder> and replaces $WORK/Data/.../five_bench/<folder> by a
# symlink to it, so every job lands in $SCRATCH. Eval CSVs stay in the repo's evaluation/csv ($WORK, small). Bring the results to the lab
# store with jz/r50_fetch_outputs.sh (run on the lab frontal). IDRIS purges unused $SCRATCH files after a retention period: fetch promptly.
#
# Folders: r48_wan_draft r48_div r48_field r48_wan_arms r49_wan_draft r49_div r49_field r49_field_full r49_wan_arms r50_preflight
#          r51_field_full r51_wan_arms r58_wan_draft r58_div r58_field r58_wan_arms r58_repro r58_wan_correct
#
# Usage: bash jz/r50_scratch_links.sh            create folders and links (idempotent)
#        bash jz/r50_scratch_links.sh --check    change nothing; say what would happen
#        bash jz/r50_scratch_links.sh --adopt    a folder that already holds files in $WORK (e.g. after jz-repro) is first copied into $SCRATCH
#                                                 (rsync), the copy is verified (same file count and bytes), then the $WORK folder is removed and linked
# A non-empty real folder without --adopt, or a link pointing elsewhere, is a CONFLICT: nothing is touched for it. R50_ENV=<file> overrides jz/r50_jz.env.
# Ends with "[r50_scratch] failures: N" (0 = go ahead).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHECK=0; ADOPT=0
for A in "$@"; do
  case "$A" in --check) CHECK=1 ;; --adopt) ADOPT=1 ;; *) echo "[r50_scratch] usage: bash jz/r50_scratch_links.sh [--check] [--adopt]"; exit 1 ;; esac
done
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r50_scratch] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r50_scratch] FATAL: $WORK does not exist on $(hostname) -- this script is for Jean Zay only"; exit 1; }
SCR=${SCRATCH:-}
[ -n "$SCR" ] && [ -d "$SCR" ] || { echo "[r50_scratch] FATAL: \$SCRATCH is not set or not a directory on $(hostname) (run it in a login shell on a Jean Zay frontal)"; exit 1; }
D=$WORK/Data/dataggen/outputs/five_bench
DEST=$SCR/five_bench_out
FOLDERS=(r48_wan_draft r48_div r48_field r48_wan_arms r49_wan_draft r49_div r49_field r49_field_full r49_wan_arms r50_preflight
         r51_field_full r51_wan_arms r58_wan_draft r58_div r58_field r58_wan_arms r58_repro r58_wan_correct)
[ -d "$D" ] || { echo "[r50_scratch] FATAL: $D missing -- run: bash jz/r50_setup_dirs.sh"; exit 1; }
echo "[r50_scratch] $([ $CHECK -eq 1 ] && echo CHECK || echo LINK) $D -> $DEST $(date '+%F %T') on $(hostname)"

FAILED=0
count_bytes() { find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{n++; b+=$1} END {printf "%d %d", n+0, b+0}'; }
for X in "${FOLDERS[@]}"; do
  L=$D/$X; T=$DEST/$X
  if [ -L "$L" ]; then
    if [ "$(readlink "$L")" = "$T" ]; then echo "  ok (already linked): $X"; [ $CHECK -eq 0 ] && mkdir -p "$T"
    else echo "  CONFLICT: $L is a link to $(readlink "$L"), not $T"; FAILED=$((FAILED + 1)); fi
    continue
  fi
  if [ -e "$L" ] && [ ! -d "$L" ]; then echo "  CONFLICT: $L exists and is not a directory"; FAILED=$((FAILED + 1)); continue; fi
  if [ -d "$L" ] && [ -n "$(ls -A "$L")" ]; then          # real folder with files
    if [ $ADOPT -eq 0 ]; then
      echo "  CONFLICT: $L holds $(find "$L" -type f | wc -l) file(s) -- rerun with --adopt to move them to \$SCRATCH"; FAILED=$((FAILED + 1)); continue
    fi
    if [ $CHECK -eq 1 ]; then echo "  would adopt: $X ($(find "$L" -type f | wc -l) files) -> $T"; continue; fi
    mkdir -p "$T" || { echo "  FAILED to create $T"; FAILED=$((FAILED + 1)); continue; }
    rsync -a "$L/" "$T/" || { echo "  FAILED rsync $X (the $WORK folder is untouched)"; FAILED=$((FAILED + 1)); continue; }
    A=$(count_bytes "$L"); B=$(count_bytes "$T")
    if [ "$A" != "$B" ]; then echo "  FAILED to verify $X: work '$A' vs scratch '$B' (files bytes) -- the $WORK folder is untouched"; FAILED=$((FAILED + 1)); continue; fi
    rm -rf "$L" && ln -s "$T" "$L" && echo "  adopted: $X (${A% *} files, verified) -> $T" || { echo "  FAILED to replace $L"; FAILED=$((FAILED + 1)); }
    continue
  fi
  # absent or empty
  if [ $CHECK -eq 1 ]; then echo "  would link: $X -> $T"; continue; fi
  mkdir -p "$T" || { echo "  FAILED to create $T"; FAILED=$((FAILED + 1)); continue; }
  [ -d "$L" ] && rmdir "$L"
  ln -s "$T" "$L" && echo "  linked: $X -> $T" || { echo "  FAILED to link $L"; FAILED=$((FAILED + 1)); }
done

if command -v idr_quota_user >/dev/null 2>&1; then
  echo "[r50_scratch] quota (idr_quota_user):"; idr_quota_user 2>&1 | cut -c1-200 | sed 's/^/  /'
fi
echo "[r50_scratch] failures: ${FAILED}"
exit $(( FAILED > 0 ))
