#!/bin/bash
# R64: make the EXISTING conda env `flowdirector` (built 2026-10-09 exactly as FlowDirector's README says: python 3.12 +
# `pip install -r requirements.txt`) able to render, without rebuilding it.
#
# Why: the README calls flash-attn optional, but wan/modules/model.py calls flash_attention() directly, which asserts
# FLASH_ATTN_2_AVAILABLE. pip resolved `torch>=2.4.0` to torch 2.14.1+cu130, and flash-attn 2 only ships prebuilt wheels for
# torch 2.4-2.8 / CUDA 12. So only torch moves (2.14.1 -> 2.8.0, still inside the requirement) and the prebuilt
# flash-attn 2.8.3.post1 wheel is added: the same torch / flash-attn pair streamgve runs on L40S. Nothing else changes.
#
# Usage: bash evaluation/r64_make_env.sh        (local, network access needed; idempotent)
# Output: evaluation/r64_env/flowdirector_pre_r64_freeze.txt  (env as found, written once, never overwritten)
#         evaluation/r64_env/flowdirector_freeze.txt          (env after this script, rewritten on every run)
# Exits non-zero if any check fails.
set -euo pipefail

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
FD=~/Code/FlowDirector
FD_COMMIT=cecd56e30820b279f35acaa18597f4da25fa4f42
PY=~/anaconda3/envs/flowdirector/bin/python
OUT=$REPO/evaluation/r64_env
TORCH=2.8.0
TORCHVISION=0.23.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128
FA=2.8.3.post1

fail() { echo "[r64_make_env] FAIL: $*"; exit 1; }
mkdir -p "$OUT"
[ -x "$PY" ] || fail "no python at $PY"

# ---- 0. code: the official V1 commit, untouched
HEAD=$(git -C "$FD" rev-parse HEAD)
[ "$HEAD" = "$FD_COMMIT" ] || fail "FlowDirector at $HEAD, expected $FD_COMMIT"
[ -z "$(git -C "$FD" status --porcelain)" ] || fail "FlowDirector tree has local changes: $(git -C "$FD" status --porcelain | head -3 | tr '\n' ' ')"
echo "[r64_make_env] FlowDirector @ ${HEAD:0:7}, clean tree"

# ---- 1. freeze the env as found (first run only, so the original state is never lost)
if [ ! -f "$OUT/flowdirector_pre_r64_freeze.txt" ]; then
  "$PY" -m pip freeze > "$OUT/flowdirector_pre_r64_freeze.txt"
  echo "[r64_make_env] saved the original env -> $OUT/flowdirector_pre_r64_freeze.txt"
fi

# ---- 2. torch / torchvision -> 2.8.0 / 0.23.0 (cu128)
CUR_TORCH=$("$PY" -c "import torch; print(torch.__version__)")
CUR_TV=$("$PY" -c "import torchvision; print(torchvision.__version__)")
if [[ "$CUR_TORCH" != "$TORCH"* || "$CUR_TV" != "$TORCHVISION"* ]]; then
  echo "[r64_make_env] torch $CUR_TORCH / torchvision $CUR_TV -> $TORCH / $TORCHVISION ($TORCH_INDEX)"
  "$PY" -m pip install "torch==$TORCH" "torchvision==$TORCHVISION" --index-url "$TORCH_INDEX"
else
  echo "[r64_make_env] torch $CUR_TORCH / torchvision $CUR_TV already in place"
fi

# ---- 3. flash-attn: the prebuilt release wheel by URL, so pip can never fall back to a source build
CUR_FA=$("$PY" -c "import importlib.metadata as m; print(m.version('flash_attn'))" 2>/dev/null || echo none)
if [ "$CUR_FA" != "$FA" ]; then
  ABI=$("$PY" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  PYTAG=cp$("$PY" -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
  WHL="flash_attn-${FA}+cu12torch${TORCH%.*}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
  URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA}/${WHL//+/%2B}"
  echo "[r64_make_env] flash-attn $CUR_FA -> $FA ($WHL)"
  "$PY" -m pip install --no-deps "$URL"          # deps (torch, einops) are already in the env
else
  echo "[r64_make_env] flash-attn $CUR_FA already in place"
fi

# ---- 4. checks
"$PY" -m pip check || fail "pip check reports broken requirements"
"$PY" - "$FD" "$TORCH" "$TORCHVISION" "$FA" <<'EOF' || fail "env checks (see above)"
import importlib.metadata as md
import sys
from packaging.requirements import Requirement

fd, want_torch, want_tv, want_fa = sys.argv[1:5]
bad = []
# every line of FlowDirector's requirements.txt (lower bounds + the scipy / scikit-image / xfuser pins) is still satisfied
for line in open(f"{fd}/requirements.txt"):
    line = line.split("#", 1)[0].strip()
    if not line:
        continue
    req = Requirement(line)
    try:
        have = md.version(req.name)
    except md.PackageNotFoundError:
        bad.append(f"{req.name}: not installed")
        continue
    if not req.specifier.contains(have, prereleases=True):
        bad.append(f"{req.name}: {have} does not satisfy '{req.specifier}'")
    print(f"[r64_make_env]   {req.name:16s} {have:22s} {str(req.specifier) or '(any)'}")

import torch
import torchvision
import flash_attn
for name, have, want in (("torch", torch.__version__, want_torch), ("torchvision", torchvision.__version__, want_tv),
                         ("flash_attn", flash_attn.__version__, want_fa)):
    if not have.startswith(want):
        bad.append(f"{name}: {have}, expected {want}")
print(f"[r64_make_env] torch {torch.__version__} (CUDA {torch.version.cuda}), torchvision {torchvision.__version__}, "
      f"flash_attn {flash_attn.__version__}, python {sys.version.split()[0]}")

# FlowDirector's own flag: the one its model code asserts on
sys.path.insert(0, fd)
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE  # noqa: E402
if not FLASH_ATTN_2_AVAILABLE:
    bad.append("wan.modules.attention.FLASH_ATTN_2_AVAILABLE is False")

# one tiny flash-attn forward vs SDPA when a GPU is visible (the wheel's kernels load and agree)
if torch.cuda.is_available():
    q, k, v = (torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o_fa = flash_attn.flash_attn_func(q, k, v)
    o_ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    err = (o_fa.float() - o_ref.float()).abs().max().item()
    print(f"[r64_make_env] flash-attn vs SDPA on {torch.cuda.get_device_name(0)}: max |diff| = {err:.2e}")
    if err > 2e-2:
        bad.append(f"flash-attn output differs from SDPA by {err:.2e}")
else:
    print("[r64_make_env] no GPU visible: skipped the flash-attn forward check")

for b in bad:
    print(f"[r64_make_env] FAIL: {b}")
sys.exit(int(bool(bad)))
EOF

# ---- 5. freeze the env as used by R64
"$PY" -m pip freeze > "$OUT/flowdirector_freeze.txt"
echo "[r64_make_env] env OK -> $OUT/flowdirector_freeze.txt"
