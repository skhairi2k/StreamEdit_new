#!/bin/bash
# R57 make-env: build the conda env `editreward` that runs EditReward's scoring stage.
#
# EditReward pins transformers==4.57.0 and trl==0.8.6 and does not import under five-bench's transformers 5.3.0, so it
# gets its own env (envs are never merged). Versions follow EditReward's README / pyproject at the commit pinned in
# fetch_weights.sh (python 3.10, torch 2.5.1 with the cu121 build five-bench already runs on the L40S nodes).
#
# Run on a node with internet access (PyPI, download.pytorch.org), after fetch_weights.sh (needs ~/Code/EditReward).
# Idempotent: when the env exists and the pins match, nothing is installed.
#
# Writes evaluation/eval_suite/freeze/editreward.txt (`pip list --format=freeze`), used by the Jean Zay overlay.
# Usage: bash evaluation/eval_suite/make_editreward_env.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_NAME=editreward
EDITREWARD_DIR=${EDITREWARD_DIR:-$HOME/Code/EditReward}
CONDA_BASE=${CONDA_BASE:-$HOME/anaconda3}
FREEZE="$REPO/evaluation/eval_suite/freeze/editreward.txt"
MANIFEST="$REPO/evaluation/eval_suite/weights_manifest.json"
TORCH_INDEX=https://download.pytorch.org/whl/cu121
FLASH_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

say() { echo "[make_editreward_env] $*"; }
die() { say "FATAL: $*"; exit 1; }

[ -x "$CONDA_BASE/bin/conda" ] || die "no conda at $CONDA_BASE (set CONDA_BASE)"
[ -f "$EDITREWARD_DIR/pyproject.toml" ] || die "$EDITREWARD_DIR missing -- run evaluation/eval_suite/fetch_weights.sh first"
if [ -f "$MANIFEST" ]; then
  PIN=$(python - "$MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("editreward_code", {}).get("revision", ""))
PY
)
  HEAD=$(git -C "$EDITREWARD_DIR" rev-parse HEAD)
  [ -z "$PIN" ] || [ "$PIN" = "$HEAD" ] || die "$EDITREWARD_DIR is at ${HEAD:0:10}, the manifest pins ${PIN:0:10}"
fi

ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"

# pins that must hold in the finished env: package==version
PINS=(torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 transformers==4.57.0 trl==0.8.6)

check_pins() {  # exit 0 when every pin is installed at exactly that version
  "$PY" - "${PINS[@]}" <<'PY'
import sys
from importlib import metadata
bad = []
for pin in sys.argv[1:]:
    name, want = pin.split("==")
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = None
    if have != want:
        bad.append(f"{name}: want {want}, have {have}")
if bad:
    print("; ".join(bad)); sys.exit(1)
PY
}

if [ -x "$PY" ] && check_pins >/dev/null 2>&1 && "$PY" -c "from EditReward import EditRewardInferencer" >/dev/null 2>&1; then
  say "env $ENV_NAME already matches the pins -- nothing to install"
else
  if [ ! -x "$PY" ]; then
    say "creating env $ENV_NAME (python 3.10)"
    "$CONDA_BASE/bin/conda" create -y -n "$ENV_NAME" python=3.10
  else
    say "env $ENV_NAME exists but does not match the pins -- reinstalling the packages in it"
  fi
  PIP=("$PY" -m pip)
  say "torch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1 (cu121)"
  "${PIP[@]}" install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url "$TORCH_INDEX"
  say "runtime dependencies (EditReward README list, transformers==4.57.0, trl==0.8.6)"
  "${PIP[@]}" install -U datasets pillow openai megfile sentencepiece deepspeed fire omegaconf \
    matplotlib peft trl==0.8.6 tensorboard scipy transformers==4.57.0 accelerate \
    requests packaging pandas opencv-python einops timm huggingface_hub qwen-vl-utils \
    av regex safetensors tqdm
  say "pip install -e $EDITREWARD_DIR"
  "${PIP[@]}" install -e "$EDITREWARD_DIR"
fi

# flash-attn: optional (EditReward's inference code never asks for it); best effort, said loudly either way
if "$PY" -c "import flash_attn" >/dev/null 2>&1; then
  say "flash_attn already installed"
elif "$PY" -m pip install "$FLASH_WHEEL" >/dev/null 2>&1; then
  say "flash_attn installed from the prebuilt wheel"
else
  say "WARNING: flash_attn not installed (prebuilt wheel unavailable) -- continuing, inference does not need it"
fi

# verify: pins hold, the package imports, torch sees the build we asked for
check_pins || die "pins do not hold after install (see above)"
"$PY" - <<'PY'
import torch, transformers, trl
from EditReward import EditRewardInferencer
print(f"[make_editreward_env] verified: torch {torch.__version__}, transformers {transformers.__version__}, "
      f"trl {trl.__version__}, EditRewardInferencer imported")
PY

mkdir -p "$(dirname "$FREEZE")"
# the editable EditReward itself is left out: Jean Zay reaches it through PYTHONPATH (no editable install there)
"$PY" -m pip list --format=freeze | grep -v -i -E '^(editreward|-e )' > "$FREEZE"
say "freeze -> $FREEZE ($(wc -l < "$FREEZE") packages)"
say "done"
