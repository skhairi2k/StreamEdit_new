#!/bin/bash
# R50 step jz-clone-setup (ON JEAN ZAY, frontal): put the pushed stage where the code expects it. Safe to rerun.
#
# Input : $JZ_WORK/r50_stage/   (jz/r50_push.sh)   weights/ cache/ data/ ref/ code/ freeze/ MANIFEST.sha256 CODE.sha256 r50_jz.env
# Moves (same filesystem: instant, no extra files; nothing is deleted):
#   stage/weights/wan_models                         -> $WORK/weights/wan_models
#   stage/cache/{huggingface,torch}                  -> $WORK/cache/{huggingface,torch}
#   stage/data/FiVE-Fine-Grained-Video-Editing-Benchmark -> $WORK/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
#   stage/ref/evaluation_csv                         -> $WORK/r50_ref/evaluation_csv   (made read-only: the compare reference)
#   stage/code/iqa_cli                               -> $WORK/code/iqa_cli
#   stage/{MANIFEST.sha256, CODE.sha256, freeze/, r50_jz.env} stay in the stage (read by r50_verify.sh / r50_pip_overlay.sh)
# Links, so that the code runs unchanged ($HOME is only 3 GB; the code hard-codes ~/Data and ~/.cache/{huggingface,torch}):
#   ~/Data -> $WORK/Data    ~/.cache/huggingface -> $WORK/cache/huggingface    ~/.cache/torch -> $WORK/cache/torch
#   <repo>/Self-Forcing_StreamEdit/wan_models -> $WORK/weights/wan_models
# Creates $WORK/Data/dataggen/outputs/five_bench (all outputs), <repo>/logs, and seeds <repo>/evaluation/csv with a writable
# copy of the reference CSVs (only if it is empty: jobs overwrite this copy, never the read-only reference).
# A target that already exists with other content is reported as CONFLICT and left alone.
#
# Usage: cd $WORK/StreamEdit_new && bash jz/r50_setup_dirs.sh      (R50_ENV=<file> overrides jz/r50_jz.env, for tests)
# Ends with "[r50_setup] failures: N".
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r50_setup] FATAL: $CFG missing -- cp \$WORK/r50_stage/r50_jz.env $REPO/jz/"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep -q '<'; then
  echo "[r50_setup] FATAL: unfilled <...> value in $CFG"; exit 1
fi
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
STAGE=$WORK/r50_stage
[ -d "$WORK" ]  || { echo "[r50_setup] FATAL: $WORK does not exist on $(hostname) -- run this on Jean Zay"; exit 1; }
[ -d "$STAGE" ] || { echo "[r50_setup] FATAL: $STAGE missing -- run jz/r50_push.sh from the lab frontal first"; exit 1; }
FAILED=0
echo "[r50_setup] on $(hostname): stage=$STAGE repo=$REPO $(date '+%F %T')"

mvdir() {  # $1 = source (in the stage), $2 = destination
  if [ -e "$2" ]; then
    if [ -e "$1" ]; then echo "  CONFLICT: both $1 and $2 exist -- merge by hand, nothing moved"; FAILED=$((FAILED + 1))
    else echo "  already in place: $2"; fi
  elif [ -e "$1" ]; then
    mkdir -p "$(dirname "$2")" && mv "$1" "$2" && echo "  moved: $1 -> $2" || { echo "  FAILED to move $1"; FAILED=$((FAILED + 1)); }
  else
    echo "  MISSING: $1 (not in the stage, not at $2)"; FAILED=$((FAILED + 1))
  fi
}
mklink() {  # $1 = link path, $2 = target
  if [ -L "$1" ]; then
    if [ "$(readlink "$1")" = "$2" ]; then echo "  link ok: $1"
    else echo "  CONFLICT: $1 -> $(readlink "$1"), wanted $2"; FAILED=$((FAILED + 1)); fi
  elif [ -e "$1" ]; then
    if [ -d "$1" ] && [ -z "$(ls -A "$1")" ]; then rmdir "$1" && ln -s "$2" "$1" && echo "  linked (replaced an empty dir): $1"
    else echo "  CONFLICT: $1 exists and is not a link (non-empty) -- move it away, then rerun"; FAILED=$((FAILED + 1)); fi
  else
    mkdir -p "$(dirname "$1")" && ln -s "$2" "$1" && echo "  linked: $1 -> $2" || { echo "  FAILED to link $1"; FAILED=$((FAILED + 1)); }
  fi
}

echo "[r50_setup] 1. move the stage into place"
mvdir "$STAGE/weights/wan_models"                                           "$WORK/weights/wan_models"
mvdir "$STAGE/cache/huggingface"                                            "$WORK/cache/huggingface"
mvdir "$STAGE/cache/torch"                                                  "$WORK/cache/torch"
mvdir "$STAGE/data/FiVE-Fine-Grained-Video-Editing-Benchmark"               "$WORK/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
mvdir "$STAGE/ref/evaluation_csv"                                           "$WORK/r50_ref/evaluation_csv"
mvdir "$STAGE/code/iqa_cli"                                                 "$WORK/code/iqa_cli"
[ -d "$WORK/r50_ref/evaluation_csv" ] && chmod -R a-w "$WORK/r50_ref/evaluation_csv" && echo "  reference CSVs made read-only"
mkdir -p "$WORK/Data/dataggen/outputs/five_bench" "$REPO/logs"

echo "[r50_setup] 2. links"
mklink "$HOME/Data"                                   "$WORK/Data"
mklink "$HOME/.cache/huggingface"                     "$WORK/cache/huggingface"
mklink "$HOME/.cache/torch"                           "$WORK/cache/torch"
mklink "$REPO/Self-Forcing_StreamEdit/wan_models"     "$WORK/weights/wan_models"

echo "[r50_setup] 3. writable evaluation/csv (a copy of the reference)"
CSV=$REPO/evaluation/csv
mkdir -p "$CSV"
if [ -z "$(ls -A "$CSV")" ]; then
  if cp -rT --no-preserve=mode,ownership "$WORK/r50_ref/evaluation_csv" "$CSV" && chmod -R u+w "$CSV"; then
    echo "  seeded $CSV with $(find "$CSV" -type f | wc -l) reference files"
  else echo "  FAILED to seed $CSV"; FAILED=$((FAILED + 1)); fi
else
  echo "  $CSV is not empty: left as is ($(find "$CSV" -type f | wc -l) files)"
fi

echo "[r50_setup] still in the stage (small files, read by verify / overlay; empty folders omitted):"
find "$STAGE" -mindepth 1 \( -type f -o -type l \) -printf '  %P\n' | head -12
echo "[r50_setup] failures: ${FAILED}"
[ "$FAILED" -eq 0 ] && echo "[r50_setup] next: bash jz/r50_verify.sh"
exit $(( FAILED > 0 ))
