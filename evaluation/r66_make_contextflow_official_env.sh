#!/usr/bin/env bash
# R66: build the conda env `contextflow` for ContextFlow's own 2-GPU setup (torchrun, FSDP + xDiT Ulysses),
# as faithful as possible to the authors' instructions. Run on a node with internet, in tmux (takes a while).
#
# The authors publish no pinned env. Their instructions are all this env follows:
#   ContextFlow README  "follow the official Wan2.1 environment setup"; "Ensure PyTorch >= 2.4.0";
#                       pip install -r requirements.txt
#   Wan2.1 README       before every multi-GPU section: pip install "xfuser>=0.4.1"
#                       (every ContextFlow script runs torchrun --nproc_per_node=2 --dit_fsdp --ulysses_size 2)
# Rule for versions: every open-ended requirement is pinned at its stated minimum (constraints file), so the env is
# reproducible and as close in time to the code as the instructions allow:
#   torch 2.4.0 / torchvision 0.19.0 (cu121), diffusers 0.31.0, transformers 4.49.0, accelerate 1.1.1,
#   numpy 1.23.5, opencv-python 4.9.0.80, gradio 5.0.0, xfuser 0.4.1;
#   tokenizers 0.21.0: the stated minimum 0.20.3 conflicts with transformers 4.49.0 (needs >= 0.21).
# flash_attn is listed unpinned and would be built from source; its prebuilt 2.7.4.post1 wheel for torch 2.4 is
# installed first, so the verbatim requirements.txt (gradio and dashscope included) finds it satisfied.
# Not specified upstream: python (3.10, the version of the wheels). torchaudio is not listed, so not installed.
# A dry-run resolve of exactly these inputs succeeded on 2026-11-01 (131 packages).
#
# Clone: ContextFlow at a9b903edf4 with r66_contextflow_frames.patch in CF_DIR. An existing clone must already be at that
# commit and patched; an empty CF_DIR is cloned from GitHub, checked out and patched here. Weights: CF_WEIGHTS must hold
# Wan2.1-I2V-14B-480P (lab: downloaded by r66_make_contextflow_env.sh; Jean Zay: sent by jz/r66_push.sh).
# The lab's earlier single-GPU env (torch 2.5.1, no xfuser) also used the name `contextflow`; it was removed by hand on
# 2026-11-01 for disk space (its freeze: r66_contextflow_env_freeze.txt) and this env takes the name. Never touches an
# existing env: refuses if ENV_PREFIX exists (remove it yourself to rebuild). Every pip call goes through the new env's
# own python.
#
# Same script on both hosts (R66 renders on Jean Zay, 2026-11-02); everything host-specific is an override:
#   ENV_PREFIX  env location           lab: ~/anaconda3/envs/contextflow       Jean Zay: $WORK/envs/contextflow
#   CONDA_BIN   conda used to create   lab: ~/anaconda3/bin/conda              Jean Zay: `module load anaconda-py3`; $CONDA_EXE
#               (default: $CONDA_EXE if set; there `conda` is a shell function, so `which conda` prints its code, not a path)
#   CF_DIR, CF_WEIGHTS                 lab: ~/Code/ContextFlow, /work/...      Jean Zay: $WORK/code/ContextFlow, $WORK/weights/...
#   FREEZE_OUT  pip freeze             lab: evaluation/r66_contextflow_official_env_freeze.txt
#                                      Jean Zay: evaluation/r66_contextflow_official_env_freeze_jz.txt (must equal the lab one)
# On Jean Zay (idr_quota_user present) the quota is printed first (the env is ~51k files of the 500k $WORK file quota; check
# it, as for R50), the conda package cache defaults to $WORK/.conda/pkgs and pip runs without cache, so nothing lands in the
# small $HOME; the env's file count is printed at the end.
#
# Import check: xfuser (envs.py check_flash_attn: torch.cuda.get_device_name) and wan (modules/t5.py: torch.cuda.current_device()
# as a default argument) touch the GPU at import time, so they fail on a host without an NVIDIA driver such as the Jean Zay
# frontal. There the check imports only torch and flash_attn and defers wan + xfuser to the GPU job: r66_smoke_jz.sh runs the
# same imports first on the H100 node.
# CHECK_ONLY=1: create nothing; on the existing ENV_PREFIX run only steps 3-4 (pins, freeze, import check, file count), e.g. to
# finish a build whose import check stopped the script (Jean Zay, 2026-11-02).
#
# Usage (lab):      bash evaluation/r66_make_contextflow_official_env.sh 2>&1 | tee logs/r66_setup_official_env.log
# Usage (Jean Zay): module purge && module load anaconda-py3 && CONDA_BIN=$CONDA_EXE ENV_PREFIX=$WORK/envs/contextflow \
#                   CF_DIR=$WORK/code/ContextFlow CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P \
#                   FREEZE_OUT=evaluation/r66_contextflow_official_env_freeze_jz.txt \
#                   bash evaluation/r66_make_contextflow_official_env.sh 2>&1 | tee logs/r66_setup_env_jz.log
#                   (finish an existing env: same command with CHECK_ONLY=1, tee -a)
set -euo pipefail
CHECK_ONLY="${CHECK_ONLY:-0}"

ENV_NAME="contextflow"
REPO_URL="https://github.com/yyChen233/ContextFlow.git"
REPO_DIR="${CF_DIR:-$HOME/Code/ContextFlow}"
COMMIT="a9b903edf4"
WEIGHTS_DIR="${CF_WEIGHTS:-/work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-$CONDA_ROOT/bin/conda}}"
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREEZE_OUT="${FREEZE_OUT:-$EVAL_DIR/r66_contextflow_official_env_freeze.txt}"
FRAMES_PATCH="$EVAL_DIR/r66_contextflow_frames.patch"
PATCH_MARKER="R66: was 81 / 49"
ENV_PREFIX="${ENV_PREFIX:-$CONDA_ROOT/envs/$ENV_NAME}"
PY="$ENV_PREFIX/bin/python"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
PINS=(
  "torch==2.4.0" "torchvision==0.19.0" "diffusers==0.31.0" "transformers==4.49.0" "tokenizers==0.21.0"
  "accelerate==1.1.1" "numpy==1.23.5" "opencv-python==4.9.0.80" "gradio==5.0.0" "xfuser==0.4.1"
  "flash_attn==2.7.4.post1"
)

# ---- 0. guards: no existing env, patched clone at the pinned commit, weights present ----------------------------
# Only the prefix directory counts (a stale ~/.conda/environments.txt line survives a deleted env).
if [[ "$CHECK_ONLY" == 1 ]]; then
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: CHECK_ONLY=1 but $PY does not exist." >&2
    exit 1
  fi
  echo "[check-only] $ENV_PREFIX exists; nothing is created or installed"
elif [[ -e "$ENV_PREFIX" ]]; then
  echo "ERROR: conda env '$ENV_NAME' already exists ($ENV_PREFIX); refusing to modify it (CHECK_ONLY=1 to only check it)." >&2
  exit 1
fi
if [[ "$CHECK_ONLY" == 1 ]]; then
  :
elif [[ ! -x "$CONDA_BIN" && -x "${CONDA_EXE:-}" ]]; then
  echo "[conda] CONDA_BIN is not an executable file (a shell function?); using \$CONDA_EXE = $CONDA_EXE"
  CONDA_BIN="$CONDA_EXE"
fi
if [[ "$CHECK_ONLY" != 1 && ! -x "$CONDA_BIN" ]]; then
  echo "ERROR: conda executable not found (CONDA_BIN='${CONDA_BIN:0:80}'); Jean Zay: module load anaconda-py3, then CONDA_BIN=\$CONDA_EXE." >&2
  exit 1
fi
if command -v idr_quota_user >/dev/null 2>&1; then
  echo "[quota] Jean Zay (idr_quota_user; the env needs ~51k files and ~5 GB of \$WORK):"
  idr_quota_user 2>&1 | cut -c1-200 | sed 's/^/  /' || true
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORK/.conda/pkgs}"
  export PIP_NO_CACHE_DIR=1
  echo "[quota] conda package cache: $CONDA_PKGS_DIRS; pip cache off"
fi
if [[ ! -e "$REPO_DIR" && "$CHECK_ONLY" == 1 ]]; then
  echo "ERROR: CHECK_ONLY=1 but $REPO_DIR does not exist." >&2
  exit 1
elif [[ ! -e "$REPO_DIR" ]]; then
  if [[ ! -f "$FRAMES_PATCH" ]]; then
    echo "ERROR: $FRAMES_PATCH not found." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
  git -C "$REPO_DIR" checkout --quiet "$COMMIT"
  git -C "$REPO_DIR" apply "$FRAMES_PATCH"
  echo "[clone] $REPO_DIR cloned at $COMMIT and patched with $FRAMES_PATCH"
elif [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: $REPO_DIR exists and is not a git clone." >&2
  exit 1
fi
head_sha="$(git -C "$REPO_DIR" rev-parse HEAD)"
want_sha="$(git -C "$REPO_DIR" rev-parse --verify "${COMMIT}^{commit}" 2>/dev/null || true)"
if [[ -z "$want_sha" || "$head_sha" != "$want_sha" ]]; then
  echo "ERROR: $REPO_DIR is at $head_sha, not $COMMIT." >&2
  exit 1
fi
if ! grep -q "$PATCH_MARKER" "$REPO_DIR/wan/image2video.py"; then
  echo "ERROR: $REPO_DIR is not patched with evaluation/r66_contextflow_frames.patch." >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS_DIR/diffusion_pytorch_model.safetensors.index.json" ]]; then
  echo "ERROR: Wan2.1-I2V-14B-480P weights not found in $WEIGHTS_DIR." >&2
  exit 1
fi
echo "[clone] $REPO_DIR at $COMMIT, frame patch applied; weights in $WEIGHTS_DIR"

if [[ "$CHECK_ONLY" != 1 ]]; then
# ---- 1. conda env (new prefix only) ------------------------------------------------------------------------------
"$CONDA_BIN" create -y -p "$ENV_PREFIX" python=3.10
if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY missing after conda create." >&2
  exit 1
fi

CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
printf '%s\n' "${PINS[@]}" > "$CONSTRAINTS"

# ---- 2. torch, flash-attn wheel, then the official requirements verbatim + xfuser ---------------------------------
"$PY" -m pip install "torch==2.4.0" "torchvision==0.19.0" --index-url "$TORCH_INDEX"
"$PY" -m pip install "$FLASH_ATTN_WHEEL" -c "$CONSTRAINTS"
"$PY" -m pip install -r "$REPO_DIR/requirements.txt" "xfuser>=0.4.1" -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX"
fi

# ---- 3. every pin installed exactly -------------------------------------------------------------------------------
"$PY" - "${PINS[@]}" <<'EOF'
import sys
from importlib.metadata import version
bad = []
for pin in sys.argv[1:]:
    name, want = pin.split("==")
    got = version(name).split("+")[0]
    print(f"[pins] {name:16s} {got}")
    if got != want:
        bad.append(f"{name}: {got} != {want}")
if bad:
    sys.exit("ERROR: pins not honoured: " + "; ".join(bad))
EOF

"$PY" -m pip freeze > "$FREEZE_OUT"
echo "[env] pip freeze -> $FREEZE_OUT"

# ---- 4. import check (xfuser imports need an NVIDIA driver, see the header) ----------------------------------------
"$PY" - <<EOF
import socket, sys
sys.path.insert(0, "$REPO_DIR")
import torch, flash_attn
if torch.cuda.is_available():
    from wan import WanI2V  # noqa: F401
    import xfuser  # noqa: F401
    from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment  # noqa: F401
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention  # noqa: F401
    from wan.distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward  # noqa: F401
    print("wan + xfuser imports: OK")
else:
    print(f"wan + xfuser imports: deferred (no GPU on {socket.gethostname()}; r66_smoke_jz.sh checks them on the GPU node)")
print("commit    :", "$head_sha")
print("torch     :", torch.__version__)
print("flash-attn:", flash_attn.__version__)
from importlib.metadata import version
print("xfuser    :", version("xfuser"))
print("import check OK")
EOF
echo "[env] $ENV_PREFIX: $(find "$ENV_PREFIX" -type f | wc -l) files, $(du -sh "$ENV_PREFIX" | cut -f1)"
