#!/usr/bin/env bash
# R65: build the isolated `anyv2v` conda env, clone AnyV2V at a pinned commit and
# fetch the I2VGen-XL weights. Run on the lab frontal (needs internet); compute
# nodes use HF_HUB_OFFLINE=1 afterwards.
#
# AnyV2V's environment.yml pins only diffusers, so the other versions are pinned
# here. huggingface_hub is held at 0.25.2: diffusers 0.26 imports cached_download,
# removed in hub 0.26. numpy / opencv are held below 2 because torch 2.2.2 is
# built against NumPy 1.x.
#
# SAFETY: ~/.bash_profile puts streamgve/bin on PATH several times, and
# `conda activate` does not move the new env's bin in front of it. Every python /
# pip call below therefore uses the env's absolute interpreter, never PATH, and the
# script aborts unless that interpreter's sys.prefix is the anyv2v env. Nothing here
# may ever install into streamgve / five-bench.
#
# Usage: bash evaluation/r65_make_anyv2v_env.sh [--check-only]
#   --check-only   run the env-identity guard and print it, install nothing
set -euo pipefail

ENV_NAME="anyv2v"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
PY="$ENV_PREFIX/bin/python"
REPO_URL="https://github.com/TIGER-AI-Lab/AnyV2V.git"
REPO_DIR="${ANYV2V_DIR:-$HOME/Code/AnyV2V}"
COMMIT="bc540befac"
WEIGHTS_DIR="${I2VGEN_DIR:-$HOME/Data/weights/i2vgen-xl}"
FREEZE_OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/r65_anyv2v_env_freeze.txt"
CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

# Keep the env hermetic: no ~/.local site-packages, no inherited PYTHONPATH / PYTHONHOME.
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PIP_REQUIRE_VIRTUALENV 2>/dev/null || true

# ---- guard: the interpreter we are about to use must live in the anyv2v env ----
assert_env() {
  [[ -x "$PY" ]] || { echo "ERROR: $PY missing (create the env first)." >&2; exit 1; }
  local prefix
  prefix="$("$PY" -c 'import os, sys; print(os.path.realpath(sys.prefix))')"
  if [[ "$prefix" != "$(realpath "$ENV_PREFIX")" ]]; then
    echo "ERROR: $PY has sys.prefix=$prefix, expected $ENV_PREFIX; refusing to install." >&2
    exit 1
  fi
  local pyver
  pyver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  [[ "$pyver" == "3.9" ]] || { echo "ERROR: anyv2v python is $pyver, expected 3.9." >&2; exit 1; }
  echo "[guard] OK  python=$PY  sys.prefix=$prefix  version=$pyver"
}

# ---- 0. conda env (created if missing; the python path is then fixed) ----------
if [[ ! -x "$PY" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda create -y -n "$ENV_NAME" python=3.9
fi
assert_env
if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "[guard] --check-only: nothing installed."
  exit 0
fi

pip_install() { "$PY" -m pip install "$@"; }

# ---- 1. AnyV2V clone at the pinned commit ---------------------------------
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

# ---- 2. pinned packages (all via the env's own pip) ------------------------
assert_env
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
printf 'numpy==1.26.4\nopencv-python==4.9.0.80\n' > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

pip_install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip_install \
  numpy==1.26.4 \
  diffusers==0.26.3 \
  transformers==4.38.2 \
  accelerate==0.27.2 \
  huggingface_hub==0.25.2 \
  omegaconf opencv-python==4.9.0.80 moviepy imageio-ffmpeg lpips av

assert_env
"$PY" -m pip freeze > "$FREEZE_OUT"
echo "[env] pip freeze -> $FREEZE_OUT"

# ---- 3. I2VGen-XL weights (online, frontal only) ---------------------------
"$PY" - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="ali-vilab/i2vgen-xl", local_dir="$WEIGHTS_DIR")
print("[weights]", path)
EOF

# ---- 4. import check with AnyV2V's own modules (CUDA not required) ---------
assert_env
(
  cd "$REPO_DIR/i2vgen-xl"
  "$PY" - <<EOF
import sys
import torch, torchvision, diffusers, transformers, numpy
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline  # noqa: F401
import pnp_utils  # noqa: F401
import utils  # noqa: F401
assert sys.prefix == "$ENV_PREFIX", sys.prefix
assert "streamgve" not in torch.__file__, torch.__file__
print("commit      :", "$(git -C "$REPO_DIR" rev-parse HEAD)")
print("torch       :", torch.__version__, torch.__file__)
print("torchvision :", torchvision.__version__)
print("diffusers   :", diffusers.__version__)
print("transformers:", transformers.__version__)
print("numpy       :", numpy.__version__)
print("import check OK")
EOF
)
