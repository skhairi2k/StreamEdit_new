#!/bin/bash
# R63 setup: prepare the `unieditflow` env and the Wan2.1-T2V-1.3B-Diffusers weights once, on the lab cluster, and record
# exactly what is installed. Run on a node with internet access (frontal), NOT inside a job: the render jobs run with
# HF_HUB_OFFLINE=1.
#
#   pip install opencv-python-headless==4.10.0.84   into ~/anaconda3/envs/unieditflow (frame loading, as wan-edit/edit.py;
#                                                    no other package is touched)
#   hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers    at the pinned revision, default HF cache
#   evaluation/r63_unieditflow_freeze.txt           (python / torch / CUDA / diffusers versions, model revision, pip freeze)
#
# Idempotent: pip and hf skip what is already there. Exit 1 on any failure.
#
# Usage: bash slurm_scripts/five_bench/r63_setup.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_DIR=${R63_ENV_DIR:-$HOME/anaconda3/envs/unieditflow}
PY="$ENV_DIR/bin/python"
MODEL_REPO=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
MODEL_REV=${R63_MODEL_REV:-0fad780a534b6463e45facd96134c9f345acfa5b}
CV2_PIN=opencv-python-headless==4.10.0.84
FREEZE="$REPO/evaluation/r63_unieditflow_freeze.txt"

[ -x "$PY" ] || { echo "[r63_setup] MISSING: $PY (conda env unieditflow)"; exit 1; }

echo "[r63_setup] installing $CV2_PIN into $ENV_DIR"
"$PY" -m pip install "$CV2_PIN"

echo "[r63_setup] downloading $MODEL_REPO @ $MODEL_REV"
"$ENV_DIR/bin/hf" download "$MODEL_REPO" --revision "$MODEL_REV"

echo "[r63_setup] import check"
"$PY" - <<'PYEOF'
import cv2, diffusers, torch, transformers
from packaging.version import Version
assert Version(diffusers.__version__) >= Version("0.33.0.dev0"), f"diffusers {diffusers.__version__} < 0.33.0.dev0 (Wan)"
print(f"[r63_setup] torch {torch.__version__} cuda {torch.version.cuda} | diffusers {diffusers.__version__} | "
      f"transformers {transformers.__version__} | cv2 {cv2.__version__}")
PYEOF

echo "[r63_setup] verifying the model snapshot resolves offline at the pinned revision"
HF_HUB_OFFLINE=1 "$PY" - "$MODEL_REPO" "$MODEL_REV" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True)
print(f"[r63_setup] snapshot: {path}")
PYEOF

{
  echo "# R63 unieditflow env freeze, $(date +%F)"
  echo "# host: $(hostname)"
  echo "# model: $MODEL_REPO @ $MODEL_REV"
  "$PY" --version 2>&1
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  "$PY" -m pip freeze
} > "$FREEZE"
echo "[r63_setup] wrote $FREEZE"
echo "[r63_setup] done"
