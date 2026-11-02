#!/usr/bin/env bash
# SUPERSEDED 2026-11-01: the env `contextflow` is now built by r66_make_contextflow_official_env.sh (the authors'
# 2-GPU setup: requirements.txt verbatim + xfuser, stated minimum versions). This script built the earlier single-GPU
# env of the same name (removed for disk space; record in r66_contextflow_env_freeze.txt); its clone and weight-download
# steps are still the ones that produced ~/Code/ContextFlow and the weights.
#
# R66: build the isolated `contextflow` conda env, clone ContextFlow at a pinned
# commit and fetch Wan2.1-I2V-14B-480P. Run on the lab frontal (needs internet);
# compute nodes use HF_HUB_OFFLINE=1 afterwards.
#
# ContextFlow runs on the Wan2.1 code base, whose requirements.txt lists torch,
# torchvision, torchaudio and flash_attn unpinned (flash_attn would be built from source).
# They are installed first from pinned wheels and filtered out of the repo
# requirements; gradio and dashscope (demo / prompt-extension only) are dropped
# as well. The other requirements are open-ended (>=), and their newest releases
# need a newer torch (diffusers 0.41 calls torch.accelerator, torch >= 2.6), so a
# constraints file pins them to the repo's own minimums, which import with torch
# 2.5.1 (checked 2026-10-28): diffusers 0.31.0, transformers 4.49.0, accelerate
# 1.1.1, huggingface_hub < 1, numpy < 2.
# The released code hard-codes 81 frames (49 for deletion); r66_contextflow_frames.patch
# generalises it to frame_num and is applied to the clone after checkout.
#
# Never touches an existing env: refuses if `contextflow` already exists (remove it
# yourself to rebuild), and every pip call goes through the new env's own python.
#
# Usage: bash evaluation/r66_make_contextflow_env.sh
set -euo pipefail

ENV_NAME="contextflow"
REPO_URL="https://github.com/yyChen233/ContextFlow.git"
REPO_DIR="${CF_DIR:-$HOME/Code/ContextFlow}"
COMMIT="a9b903edf4"
WEIGHTS_DIR="${CF_WEIGHTS:-/work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P}"
MIN_FREE_GB=100
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREEZE_OUT="$EVAL_DIR/r66_contextflow_env_freeze.txt"
FRAMES_PATCH="$EVAL_DIR/r66_contextflow_frames.patch"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY="$ENV_PREFIX/bin/python"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ---- 0. guards: no existing env, patch present, free space ------------------
# Only the prefix directory counts: ~/.conda/environments.txt keeps a stale line after
# the directory is deleted, and conda create re-registers the path anyway.
if [[ -e "$ENV_PREFIX" ]]; then
  echo "ERROR: conda env '$ENV_NAME' already exists ($ENV_PREFIX); refusing to modify it." >&2
  exit 1
fi
if [[ ! -f "$FRAMES_PATCH" ]]; then
  echo "ERROR: $FRAMES_PATCH not found." >&2
  exit 1
fi
mkdir -p "$WEIGHTS_DIR"
free_kb="$(df -Pk "$WEIGHTS_DIR" | awk 'NR==2 {print $4}')"
if (( free_kb < MIN_FREE_GB * 1024 * 1024 )); then
  echo "ERROR: $WEIGHTS_DIR has $((free_kb / 1024 / 1024)) GB free, need >= $MIN_FREE_GB GB." >&2
  exit 1
fi

# ---- 1. ContextFlow clone at the pinned commit ------------------------------
if [[ -d "$REPO_DIR/.git" ]]; then
  head_sha="$(git -C "$REPO_DIR" rev-parse HEAD)"
  want_sha="$(git -C "$REPO_DIR" rev-parse --verify "${COMMIT}^{commit}" 2>/dev/null || true)"
  if [[ -z "$want_sha" || "$head_sha" != "$want_sha" ]]; then
    echo "ERROR: $REPO_DIR is at $head_sha, not $COMMIT; refusing to touch it." >&2
    exit 1
  fi
  echo "[clone] $REPO_DIR already at $COMMIT"
elif [[ -e "$REPO_DIR" ]]; then
  echo "ERROR: $REPO_DIR exists and is not a git clone." >&2
  exit 1
else
  git clone "$REPO_URL" "$REPO_DIR"
  git -C "$REPO_DIR" checkout --quiet "$COMMIT"
  echo "[clone] $REPO_DIR checked out at $COMMIT"
fi

# ---- 1b. frame-count patch (81 / 49 hard-coded -> frame_num) -----------------
if git -C "$REPO_DIR" apply --reverse --check "$FRAMES_PATCH" 2>/dev/null; then
  echo "[patch] $FRAMES_PATCH already applied"
elif git -C "$REPO_DIR" apply --check "$FRAMES_PATCH"; then
  git -C "$REPO_DIR" apply "$FRAMES_PATCH"
  echo "[patch] applied $FRAMES_PATCH"
else
  echo "ERROR: $FRAMES_PATCH does not apply cleanly to $REPO_DIR." >&2
  exit 1
fi

# ---- 2. conda env (new prefix only) ------------------------------------------
"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" python=3.10
if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY missing after conda create." >&2
  exit 1
fi

"$PY" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install "$FLASH_ATTN_WHEEL"

CONSTRAINTS="$(mktemp)"
FILTERED_REQ="$(mktemp)"
trap 'rm -f "$CONSTRAINTS" "$FILTERED_REQ"' EXIT
printf '%s\n' "numpy<2" "diffusers==0.31.0" "transformers==4.49.0" "accelerate==1.1.1" "huggingface_hub<1" > "$CONSTRAINTS"
grep -v -i -E '^[[:space:]]*(gradio|dashscope|flash[-_]attn|torch|torchvision|torchaudio)([[:space:]]*[<>=!~;#]|[[:space:]]*$)' \
  "$REPO_DIR/requirements.txt" > "$FILTERED_REQ"
"$PY" -m pip install -r "$FILTERED_REQ" -c "$CONSTRAINTS" "numpy<2" huggingface_hub

"$PY" -m pip freeze > "$FREEZE_OUT"
echo "[env] pip freeze -> $FREEZE_OUT"

# ---- 3. Wan2.1-I2V-14B-480P weights (online, frontal only) -------------------
"$PY" - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="Wan-AI/Wan2.1-I2V-14B-480P", local_dir="$WEIGHTS_DIR")
print("[weights]", path)
EOF

# ---- 4. import check (CUDA not required) -------------------------------------
"$PY" - <<EOF
import sys
sys.path.insert(0, "$REPO_DIR")
import torch, flash_attn
from wan import WanI2V  # noqa: F401
print("commit    :", "$(git -C "$REPO_DIR" rev-parse HEAD)")
print("patch     :", "applied" if "R66: was 81 / 49" in open("$REPO_DIR/wan/image2video.py").read() else "MISSING")
print("torch     :", torch.__version__)
print("flash-attn:", flash_attn.__version__)
print("import check OK")
EOF
