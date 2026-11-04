#!/bin/bash
# R57 cf-env: build the conda env `ivebench` that runs the CF (Content Fidelity) stage, from IVEBench's own environment.
#
# Why: under five-bench's transformers 5.3 the Qwen2.5-VL-72B judge outputs NaN from its first language-model layer
# (R57 check-smoke-lab, debug job 1084924; the weights were checked finite on CPU), so CF runs in the env IVEBench was
# released with: ~/Code/IVEBench/ivebench.yml at the commit pinned in weights_manifest.json (ivebench_code). Its pins:
# python 3.12.9, torch 2.8.0 / torchvision 0.23.0 (cu128 wheels; the lab L40S driver 610.57 supports CUDA 12.8),
# transformers 4.55.0, accelerate 1.1.0, qwen-vl-utils 0.0.11, numpy 2.2.6, imageio-ffmpeg 0.6.0.
#
# How: `conda create` python 3.12.9, then the yml's pip section VERBATIM with --no-deps (it is a full freeze, so every
# package lands at exactly the authors' version and pip resolves nothing), except two entries that are not installable
# from PyPI and that CF never imports (they belong to IVEBench's other metrics): `clip` (an unversioned local install of
# OpenAI CLIP) and `groundingdino==0.1.0` (built from source). `conda env create -f ivebench.yml` is not used because
# its pip step fails on exactly those two entries and leaves a half-built env.
#
# Run on a node with internet access (PyPI). Refuses an existing env whose pins differ (remove it first). Idempotent:
# when the env exists, the pins hold and CF's module imports, nothing is installed. Never touches five-bench / editreward.
# Writes evaluation/eval_suite/freeze/ivebench.txt (`pip list --format=freeze`).
# Usage: bash evaluation/eval_suite/make_cf_env.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENV_NAME=ivebench
IVEBENCH_DIR=${IVEBENCH_DIR:-$HOME/Code/IVEBench}
CONDA_BASE=${CONDA_BASE:-$HOME/anaconda3}
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
YML="$IVEBENCH_DIR/ivebench.yml"
FREEZE="$REPO/evaluation/eval_suite/freeze/ivebench.txt"
MANIFEST="$REPO/evaluation/eval_suite/weights_manifest.json"
PY_FB=$HOME/anaconda3/envs/five-bench/bin/python        # only to read the yml / manifest (PyYAML); never modified
SKIP=(clip groundingdino)                                # not on PyPI, not imported by CF (see header)

say() { echo "[make_cf_env] $*"; }
die() { say "FATAL: $*"; exit 1; }

[ -x "$CONDA_BASE/bin/conda" ] || die "no conda at $CONDA_BASE (set CONDA_BASE)"
[ -f "$YML" ] || die "$YML missing -- run evaluation/eval_suite/fetch_weights.sh first (IVEBench clone)"
PIN=$("$PY_FB" -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('ivebench_code', {}).get('revision', ''))" "$MANIFEST")
HEAD=$(git -C "$IVEBENCH_DIR" rev-parse HEAD)
[ -z "$PIN" ] || [ "$PIN" = "$HEAD" ] || die "$IVEBENCH_DIR is at ${HEAD:0:10}, the manifest pins ${PIN:0:10}"
say "IVEBench at ${HEAD:0:10}, env -> $ENV_DIR"

# pins that must hold in the finished env (from ivebench.yml)
PINS=(torch==2.8.0 torchvision==0.23.0 transformers==4.55.0 accelerate==1.1.0 qwen-vl-utils==0.0.11 numpy==2.2.6
      imageio-ffmpeg==0.6.0 tokenizers==0.21.4 safetensors==0.6.1)

check_pins() {  # exit 0 when python is 3.12.9 and every pin is installed at exactly that version
  "$PY" - "${PINS[@]}" <<'PY'
import platform, sys
from importlib import metadata
bad = [] if platform.python_version() == "3.12.9" else [f"python: want 3.12.9, have {platform.python_version()}"]
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

import_check() {  # CF's own module imports in the env (no GPU needed): stub `fidelity` package as cf_score.py does
  "$PY" - "$IVEBENCH_DIR" <<'PY'
import sys, types
metrics = sys.argv[1] + "/metrics"
sys.path.insert(0, metrics)
stub = types.ModuleType("fidelity"); stub.__path__ = [metrics + "/fidelity"]; sys.modules["fidelity"] = stub
from fidelity.content_fidelity import QwenVLContentFidelityEvaluator  # noqa: F401
from fidelity.qwen_vl_utils import process_vision_info               # noqa: F401
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # noqa: F401
import cv2, decord, imageio_ffmpeg, torch, transformers
print(f"[make_cf_env] import check: QwenVLContentFidelityEvaluator ok, transformers {transformers.__version__}, "
      f"torch {torch.__version__} (CUDA {torch.version.cuda}), cv2 {cv2.__version__}, decord {decord.__version__}, "
      f"ffmpeg {imageio_ffmpeg.get_ffmpeg_exe()}")
PY
}

if [ -x "$PY" ]; then
  if check_pins >/dev/null 2>&1 && import_check >/dev/null 2>&1; then
    say "env $ENV_NAME already matches IVEBench's pins and imports CF -- nothing to install"
  else
    die "env $ENV_DIR exists but does not match IVEBench's pins ($(check_pins 2>&1 | tail -1)); remove it first: conda env remove -p $ENV_DIR"
  fi
else
  say "creating env $ENV_NAME (python 3.12.9, as ivebench.yml)"
  "$CONDA_BASE/bin/conda" create -y -p "$ENV_DIR" python=3.12.9
  REQ=$(mktemp -p "${TMPDIR:-/tmp}" ivebench_req.XXXXXX)
  trap 'rm -f "$REQ"' EXIT
  "$PY_FB" -I - "$YML" "$REQ" "${SKIP[@]}" <<'PY'
import re, sys, yaml
yml, out, skip = sys.argv[1], sys.argv[2], set(sys.argv[3:])
deps = yaml.safe_load(open(yml))["dependencies"]
pip = next(d["pip"] for d in deps if isinstance(d, dict) and "pip" in d)
kept, dropped = [], []
for spec in pip:
    name = re.split(r"[=<>!~ ]", str(spec), maxsplit=1)[0].strip().lower()
    (dropped if name in skip else kept).append(str(spec))
open(out, "w").write("\n".join(kept) + "\n")
print(f"[make_cf_env] ivebench.yml pip section: {len(pip)} entries, installing {len(kept)} verbatim, "
      f"skipped {len(dropped)}: {', '.join(dropped)}")
PY
  say "pip install --no-deps -r <ivebench.yml pip section> (exact versions, nothing resolved)"
  "$PY" -m pip install --no-deps -r "$REQ"
fi

# verify: pins hold, CF imports, pip's own consistency report (informative: the two skipped packages are not needed)
check_pins || die "pins do not hold after install (see above)"
import_check || die "CF's module does not import in $ENV_DIR"
say "pip check (informative):"; "$PY" -m pip check || true

mkdir -p "$(dirname "$FREEZE")"
"$PY" -m pip list --format=freeze > "$FREEZE"
say "freeze -> $FREEZE ($(wc -l < "$FREEZE") packages); env size $(du -sh "$ENV_DIR" 2>/dev/null | cut -f1)"
say "done"
