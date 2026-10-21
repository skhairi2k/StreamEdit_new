#!/bin/bash
# R50 step make-bundle (this cluster): stage everything Jean Zay needs that is NOT in git, plus checksums.
# Jean Zay runs the Wan-Edit side of R49 only (StreamEdit stays on the lab cluster), all in our five-bench env.
#
# The stage ($LOCAL_STAGE) holds SYMLINKS to our own files -- no copy, works across filesystems --
# and jz/r50_push.sh sends it with `rsync --copy-unsafe-links`: links that leave the stage are sent as real
# files, the Hugging Face cache's internal relative links (snapshots -> blobs) stay links, so nothing doubles.
#
#   weights/wan_models/Wan2.1-T2V-1.3B          Wan-Edit backbone (r48_wan_common.CKPT)
#   cache/huggingface/hub/models--...           DINOv3 ViT-B/16 (divergence), CLIP ViT-L/14, Qwen2.5-VL-7B-Instruct (eval)
#   cache/torch/hub                             DINO ViT-B/8 repo + weights, AlexNet/VGG (LPIPS), RAFT, pyiqa NIQE
#   data/FiVE-Fine-Grained-Video-Editing-Benchmark/{videos,images,bmasks,edit_prompt}
#   ref/evaluation_csv                          our CSVs: the reference r50_compare.py reads (kept apart on Jean Zay)
#   code/iqa_cli                                pyiqa source (editable install, NIQE via inference_iqa.py)
#   freeze/five-bench.txt, .editable.json, .versions.txt   what r50_pip_overlay.sh must reproduce
#   MANIFEST.sha256   sha256 of every file reachable in the stage (paths relative to the stage)
#   CODE.sha256       sha256 of every repo file the clone must reproduce (paths relative to the repo)
#
# Usage: bash jz/r50_make_bundle.sh          (rerunnable: links are replaced, checksums recomputed)
# Env:   R50_HASH_JOBS (default 8) parallel sha256sum processes.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG="$REPO/jz/r50_jz.env"
[ -f "$CFG" ] || { echo "[r50_bundle] FATAL: $CFG missing -- cp jz/r50_jz.env.example jz/r50_jz.env and fill it"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep -q '<'; then
  echo "[r50_bundle] FATAL: unfilled <...> value in $CFG:"; set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep '<'; exit 1
fi

STAGE=${LOCAL_STAGE:?LOCAL_STAGE not set in $CFG}
CONDA_ENVS=${CONDA_ENVS:-$HOME/anaconda3/envs}
HF_HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub
TORCH_HUB=${TORCH_HOME:-$HOME/.cache/torch}/hub
FIVE=$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
HASH_JOBS=${R50_HASH_JOBS:-8}
T0=$(date +%s)
FAILED=0

# stage path -> source path (resolved to an absolute real path below)
declare -A SRC=(
  [weights/wan_models/Wan2.1-T2V-1.3B]="$REPO/Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B"
  [cache/huggingface/hub/models--facebook--dinov3-vitb16-pretrain-lvd1689m]="$HF_HUB/models--facebook--dinov3-vitb16-pretrain-lvd1689m"
  [cache/huggingface/hub/models--openai--clip-vit-large-patch14]="$HF_HUB/models--openai--clip-vit-large-patch14"
  [cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct]="$HF_HUB/models--Qwen--Qwen2.5-VL-7B-Instruct"
  [cache/torch/hub]="$TORCH_HUB"
  [data/FiVE-Fine-Grained-Video-Editing-Benchmark/videos]="$FIVE/videos"
  [data/FiVE-Fine-Grained-Video-Editing-Benchmark/images]="$FIVE/images"
  [data/FiVE-Fine-Grained-Video-Editing-Benchmark/bmasks]="$FIVE/bmasks"
  [data/FiVE-Fine-Grained-Video-Editing-Benchmark/edit_prompt]="$FIVE/edit_prompt"
  [ref/evaluation_csv]="$REPO/evaluation/csv"
  [code/iqa_cli]="$HOME/Code/iqa_cli"
)

echo "[r50_bundle] stage=$STAGE repo=$REPO $(date '+%F %T')"
mkdir -p "$STAGE"

# ---- 1. links -----------------------------------------------------------------------------------------
for DST in $(printf '%s\n' "${!SRC[@]}" | sort); do
  S=${SRC[$DST]}
  if [ ! -e "$S" ]; then echo "[r50_bundle] MISSING source $S (for $DST)"; FAILED=$((FAILED + 1)); continue; fi
  R=$(readlink -f "$S")
  if [ -e "$STAGE/$DST" ] && [ ! -L "$STAGE/$DST" ]; then
    echo "[r50_bundle] REFUSE: $STAGE/$DST exists and is not a link (not overwriting real data)"; FAILED=$((FAILED + 1)); continue
  fi
  mkdir -p "$(dirname "$STAGE/$DST")"
  ln -sfn "$R" "$STAGE/$DST"
  printf '[r50_bundle] link %-72s -> %s\n' "$DST" "$R"
done
[ "$FAILED" -gt 0 ] && { echo "[r50_bundle] failures: $FAILED (fix the sources above, then rerun)"; exit 1; }

# ---- 2. package list of the five-bench env (read by r50_pip_overlay.sh on Jean Zay) --------------------
mkdir -p "$STAGE/freeze"
for ENV in five-bench; do
  PY="$CONDA_ENVS/$ENV/bin/python"
  [ -x "$PY" ] || { echo "[r50_bundle] MISSING interpreter $PY"; FAILED=$((FAILED + 1)); continue; }
  # `pip list --format=freeze`, not `pip freeze`: conda-built packages (certifi, gmpy2, PyYAML) come out as `name @ file:///...` with no
  # version in the latter. pip / setuptools / wheel are listed too; r50_pip_overlay.sh never touches them.
  "$PY" -m pip list --format=freeze --exclude-editable > "$STAGE/freeze/$ENV.txt"
  "$PY" -m pip list --editable --format=json > "$STAGE/freeze/$ENV.editable.json"
  "$PY" -c 'import sys, torch; print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda)' \
    > "$STAGE/freeze/$ENV.versions.txt" 2>/dev/null
  echo "[r50_bundle] freeze $ENV: $(wc -l < "$STAGE/freeze/$ENV.txt") packages, editable: $("$PY" -c 'import json,sys; print(", ".join(p["name"] for p in json.load(open(sys.argv[1]))) or "none")' "$STAGE/freeze/$ENV.editable.json"); $(tr '\n' ' ' < "$STAGE/freeze/$ENV.versions.txt")"
done

# ---- 3. CODE.sha256: repo files the clone on Jean Zay must reproduce (tracked + new, not ignored) -------
CODE_PATHS=(slurm_scripts/five_bench evaluation Self-Forcing_StreamEdit wan-edit jz)
(
  cd "$REPO"
  git ls-files --cached --others --exclude-standard -z -- "${CODE_PATHS[@]}" | sort -zu \
    | while IFS= read -r -d '' f; do [ -f "$f" ] && printf '%s\0' "$f"; done \
    | xargs -0 -r sha256sum
) > "$STAGE/CODE.sha256"
echo "[r50_bundle] CODE.sha256: $(wc -l < "$STAGE/CODE.sha256") repo files (${CODE_PATHS[*]})"
DIRTY=$(cd "$REPO" && git status --porcelain -- "${CODE_PATHS[@]}")
if [ -n "$DIRTY" ]; then
  echo "[r50_bundle] WARNING: $(echo "$DIRTY" | wc -l) uncommitted/untracked code files -- commit + push them before cloning on Jean Zay, or r50_verify.sh reports them:"
  echo "$DIRTY" | head -n 25 | sed 's/^/[r50_bundle]   /'
fi

# ---- 4. MANIFEST.sha256 over everything reachable in the stage ----------------------------------------
cd "$STAGE"
BROKEN=$(find -L . -type l | grep -v '^\./\(MANIFEST\|CODE\)\.sha256' || true)
if [ -n "$BROKEN" ]; then
  echo "[r50_bundle] BROKEN links inside the sources (would arrive broken on Jean Zay):"; echo "$BROKEN" | head -n 20
  FAILED=$((FAILED + $(echo "$BROKEN" | wc -l)))
fi
echo "[r50_bundle] hashing with $HASH_JOBS processes ..."
find -L . -type f ! -path ./MANIFEST.sha256 ! -path ./CODE.sha256 ! -path ./MANIFEST.sha256.tmp -print0 \
  | xargs -0 -r -P "$HASH_JOBS" -n 1 sha256sum | sort -k 2 > MANIFEST.sha256.tmp
mv MANIFEST.sha256.tmp MANIFEST.sha256

# ---- 5. summary: what lands on Jean Zay ($WORK bytes and inodes) --------------------------------------
N_FILES=$(wc -l < MANIFEST.sha256)
N_LINKS=$(find -L . -path '*/snapshots/*' -type f | wc -l)          # stay links after rsync (HF cache)
BYTES=$(find -L . -type f ! -path '*/snapshots/*' -printf '%s\n' | awk '{s += $1} END {printf "%.1f", s / 2^30}')
echo "[r50_bundle] MANIFEST.sha256: $N_FILES files (incl. $N_LINKS HF snapshot paths that stay links)"
echo "[r50_bundle] to send: ${BYTES} GiB, ~$N_FILES inodes on Jean Zay \$WORK"
for TOP in weights cache data ref code; do
  printf '[r50_bundle]   %-8s %8s files %8s GiB\n' "$TOP" "$(find -L "$TOP" -type f | wc -l)" \
    "$(find -L "$TOP" -type f ! -path '*/snapshots/*' -printf '%s\n' | awk '{s += $1} END {printf "%.2f", s / 2^30}')"
done
echo "[r50_bundle] elapsed $(( $(date +%s) - T0 )) s"
echo "[r50_bundle] failures: $FAILED"
exit $(( FAILED > 0 ))
