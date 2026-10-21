#!/bin/bash
# R50 step jz-clone-setup (ON JEAN ZAY): prove that what arrived is what the lab cluster had.
#
#   1. MANIFEST.sha256 : SHA-256 of every shipped file (weights, caches, FiVE data, reference CSVs, iqa_cli), recomputed at its
#                        final location after r50_setup_dirs.sh (the manifest lists stage-relative paths; they are mapped).
#   2. CODE.sha256     : SHA-256 of the repo files the R49 chain runs, recomputed in this clone. A failure means the clone
#                        differs from the lab working copy of the last `make-bundle` (unpushed / newer edits, or the files
#                        that r50_port.sh rewrote: run verify BEFORE the port, or undo the port first -- see below).
#   3. layout          : the links and folders the code relies on exist.
#
# Usage: bash jz/r50_verify.sh               everything (36 GB to hash: a few minutes, 8 parallel readers)
#        bash jz/r50_verify.sh --code-only   only the 426 repo files (seconds): use after a `git pull`
# After r50_port.sh: undo it first, verify, then port again:
#        git checkout -- slurm_scripts evaluation/fivebench/config.yaml && git pull && bash jz/r50_verify.sh --code-only && bash jz/r50_port.sh
# R50_ENV=<file> overrides jz/r50_jz.env (tests). Ends with "[r50_verify] failures: N".
set -u

CODE_ONLY=0; [ "${1:-}" = "--code-only" ] && CODE_ONLY=1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r50_verify] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
STAGE=$WORK/r50_stage
JOBS=${R50_HASH_JOBS:-8}
FAILED=0
TMP=$(mktemp -d "${TMPDIR:-/tmp}/r50_verify.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT
echo "[r50_verify] on $(hostname): repo=$REPO stage=$STAGE $(date '+%F %T')"

# ---- 1. shipped files
if [ "$CODE_ONLY" -eq 0 ]; then
  if [ ! -s "$STAGE/MANIFEST.sha256" ]; then echo "[r50_verify] FATAL: $STAGE/MANIFEST.sha256 missing"; exit 1; fi
  sed -e "s#  \./weights/#  $WORK/weights/#" -e "s#  \./cache/#  $WORK/cache/#" -e "s#  \./data/#  $WORK/Data/#" \
      -e "s#  \./ref/evaluation_csv/#  $WORK/r50_ref/evaluation_csv/#" -e "s#  \./code/#  $WORK/code/#" \
      -e "s#  \./#  $STAGE/#" "$STAGE/MANIFEST.sha256" > "$TMP/manifest.mapped"   # last rule: freeze/, r50_jz.env... stay in the stage
  N=$(wc -l < "$TMP/manifest.mapped")
  echo "[r50_verify] 1. hashing ${N} shipped files with ${JOBS} readers ..."
  T0=$(date +%s)
  split -n "l/${JOBS}" "$TMP/manifest.mapped" "$TMP/part_"
  for P in "$TMP"/part_*; do ( sha256sum -c --quiet "$P" > "$P.out" 2>&1 ) & done
  wait
  # one line per bad file ends in ': FAILED' (checksum differs) or ': FAILED open or read' (missing); the extra
  # 'sha256sum: <path>: No such file' line of a missing file is not counted again
  BAD=$(cat "$TMP"/part_*.out | grep -c ': FAILED')
  echo "[r50_verify]    $(( N - BAD )) of ${N} ok, ${BAD} bad, $(( $(date +%s) - T0 )) s"
  [ "$BAD" -gt 0 ] && { cat "$TMP"/part_*.out | grep ': FAILED' | head -10 | sed 's/^/[r50_verify]    /'
                        echo "[r50_verify]    -> rerun jz/r50_push.sh from the lab frontal, then this script"; }
  FAILED=$((FAILED + BAD))
fi

# ---- 2. repo code
if [ ! -s "$STAGE/CODE.sha256" ]; then echo "[r50_verify] FATAL: $STAGE/CODE.sha256 missing"; exit 1; fi
NC=$(wc -l < "$STAGE/CODE.sha256")
( cd "$REPO" && sha256sum -c --quiet "$STAGE/CODE.sha256" > "$TMP/code.out" 2>&1 )
BADC=$(grep -c ': FAILED' "$TMP/code.out")
echo "[r50_verify] 2. repo code: $(( NC - BADC )) of ${NC} files identical to the lab working copy, ${BADC} differ"
if [ "$BADC" -gt 0 ]; then
  grep ': FAILED' "$TMP/code.out" | sed 's/: FAILED open or read.*/  MISSING in the clone/; s/: FAILED$/  differs/' | head -15 | sed 's/^/[r50_verify]    /'
  echo "[r50_verify]    -> lab side: commit + push, then \`bash jz/r50_make_bundle.sh\` and \`bash jz/r50_push.sh\`; here: git pull"
  echo "[r50_verify]    -> if r50_port.sh already ran, these are its edits: git checkout -- slurm_scripts evaluation/fivebench/config.yaml, verify, port again"
fi
FAILED=$((FAILED + BADC))

# ---- 3. layout
if [ "$CODE_ONLY" -eq 0 ]; then
  echo "[r50_verify] 3. layout"
  for P in "$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/videos" \
           "$HOME/.cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m" \
           "$HOME/.cache/torch/hub/checkpoints" \
           "$REPO/Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
           "$WORK/Data/dataggen/outputs/five_bench" "$WORK/r50_ref/evaluation_csv" "$WORK/code/iqa_cli" "$REPO/evaluation/csv"; do
    [ -e "$P" ] || { echo "[r50_verify]    MISSING $P"; FAILED=$((FAILED + 1)); }
  done
  echo "[r50_verify]    videos: $(ls "$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/videos" 2>/dev/null | wc -l) | evaluation/csv files: $(find "$REPO/evaluation/csv" -type f 2>/dev/null | wc -l)"
fi
echo "[r50_verify] failures: ${FAILED}"
exit $(( FAILED > 0 ))
