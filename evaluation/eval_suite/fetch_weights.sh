#!/bin/bash
# R57 fetch-weights: download the extra models and code of the unified metric suite once, on the lab cluster, and record
# exactly what was fetched. Run on a node with internet access (login/frontal or any node that reaches huggingface.co
# and github.com), NOT inside a job: the evaluation jobs run with HF_HUB_OFFLINE=1.
#
#   TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508  ->  $WEIGHTS_DIR/EditReward-MiMo-VL-7B-SFT-2508      (EditReward)
#   XiaomiMiMo/MiMo-VL-7B-SFT-2508            ->  $WEIGHTS_DIR/MiMo-VL-7B-SFT-2508    (16.6 GB)   (EditReward's base model:
#                                                 its inferencer builds the model and the processor from it, then loads
#                                                 the checkpoint above over it; the checkpoint repo has no tokenizer files)
#   alibaba-pai/VideoCLIP-XL-v2               ->  $WEIGHTS_DIR/VideoCLIP-XL-v2                    (PSC)
#   Qwen/Qwen2.5-VL-72B-Instruct              ->  $WEIGHTS_DIR/Qwen2.5-VL-72B-Instruct  (147 GB)  (CF judge)
#   TIGER-AI-Lab/EditReward (code)            ->  $EDITREWARD_DIR   (git clone at a pinned commit)
#   RyanChenYN/IVEBench (code)                ->  $IVEBENCH_DIR     (git clone at a pinned commit; CF prompt + video prep)
#   evaluation/eval_suite/weights_manifest.json   (repo, revision/commit, local path, sha256 per weight file)
#
# Pins are the heads resolved on 2026-10-26; override with the *_REV variables to move them. Idempotent: a component
# whose manifest entry has the same revision and whose files all still exist with the recorded sizes is skipped (no
# re-download, no re-hash). To force a re-fetch of one component, delete its entry from the manifest.
#
# Usage: bash evaluation/eval_suite/fetch_weights.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MANIFEST="$REPO/evaluation/eval_suite/weights_manifest.json"
WEIGHTS_DIR=${WEIGHTS_DIR:-$HOME/Data/weights}
EDITREWARD_DIR=${EDITREWARD_DIR:-$HOME/Code/EditReward}
IVEBENCH_DIR=${IVEBENCH_DIR:-$HOME/Code/IVEBench}
QWEN72B_MIN_FREE_GB=170   # 147 GB of weights + the hub's partial-download slack

EDITREWARD_REPO=TIGER-Lab/EditReward-MiMo-VL-7B-SFT-2508
EDITREWARD_REV=${EDITREWARD_REV:-4a702b4c1d6b45e597ced81beddcfaea3741f675}
MIMO_BASE_REPO=XiaomiMiMo/MiMo-VL-7B-SFT-2508
MIMO_BASE_REV=${MIMO_BASE_REV:-d5281c5ee88b4d696f6e0f3e73b2ae9efc1d9da3}
VIDEOCLIP_REPO=alibaba-pai/VideoCLIP-XL-v2
VIDEOCLIP_REV=${VIDEOCLIP_REV:-552b698005f54886fb94dd534e4968fd20e4fbf2}
QWEN72B_REPO=Qwen/Qwen2.5-VL-72B-Instruct
QWEN72B_REV=${QWEN72B_REV:-89c86200743eec961a297729e7990e8f2ddbc4c5}
CODE_REPO=https://github.com/TIGER-AI-Lab/EditReward
CODE_REV=${CODE_REV:-77a93aaa461fe9187e0ff841b59ecc0d0620bb7f}
IVEBENCH_REPO=https://github.com/RyanChenYN/IVEBench
IVEBENCH_REV=${IVEBENCH_REV:-2dcf8900ee65197f43854c76f61c66a32bb8dbfc}

# a login shell in a job env may carry offline flags; downloading needs the network
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
if command -v hf >/dev/null 2>&1; then HF_DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then HF_DL=(huggingface-cli download)
else echo "[fetch_weights] FATAL: neither 'hf' nor 'huggingface-cli' on PATH"; exit 1; fi
command -v git >/dev/null 2>&1 || { echo "[fetch_weights] FATAL: git not on PATH"; exit 1; }

mkdir -p "$WEIGHTS_DIR" "$(dirname "$EDITREWARD_DIR")" "$(dirname "$IVEBENCH_DIR")"
[ -f "$MANIFEST" ] || echo '{}' > "$MANIFEST"

# manifest helpers (python stdlib only)
m_current() {   # $1=key $2=revision $3=path -> exit 0 if the manifest entry matches and every recorded file still has its size
  python - "$MANIFEST" "$1" "$2" "$3" <<'PY'
import json, os, sys
manifest, key, rev, path = sys.argv[1:5]
e = json.load(open(manifest)).get(key)
if not e or e.get("revision") != rev or e.get("path") != path or not os.path.isdir(path):
    sys.exit(1)
for rel, info in e.get("files", {}).items():
    p = os.path.join(path, rel)
    if not os.path.isfile(p) or os.path.getsize(p) != info["size"]:
        sys.exit(1)
PY
}
m_record() {    # $1=key $2=repo $3=revision $4=path $5=kind(weights|code)
  python - "$MANIFEST" "$1" "$2" "$3" "$4" "$5" <<'PY'
import hashlib, json, os, sys
manifest, key, repo, rev, path, kind = sys.argv[1:7]
files = {}
if kind == "weights":
    exts = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
    for root, _, names in os.walk(path):
        if ".cache" in root.split(os.sep):
            continue
        for n in sorted(names):
            if n.endswith(exts):
                p = os.path.join(root, n)
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 24), b""):
                        h.update(chunk)
                files[os.path.relpath(p, path)] = {"sha256": h.hexdigest(), "size": os.path.getsize(p)}
    if not files:
        sys.exit(f"[fetch_weights] FATAL: no weight file found under {path}")
m = json.load(open(manifest))
m[key] = {"repo": repo, "revision": rev, "path": path, "kind": kind, "files": files}
json.dump(m, open(manifest, "w"), indent=2, sort_keys=True); open(manifest, "a").write("\n")
PY
}

fetch_weights() {   # $1=key $2=repo $3=revision [$4=min free GB on $WEIGHTS_DIR before downloading]
  local KEY=$1 HFREPO=$2 REV=$3 MINFREE=${4:-0} DEST="$WEIGHTS_DIR/${2##*/}"
  if m_current "$KEY" "$REV" "$DEST"; then
    echo "[fetch_weights] $HFREPO@${REV:0:10} already present in $DEST -- skipped"; return
  fi
  local FREE; FREE=$(df -BG --output=avail "$WEIGHTS_DIR" | tail -n 1 | tr -dc '0-9')
  [ "$FREE" -ge "$MINFREE" ] || { echo "[fetch_weights] FATAL: $WEIGHTS_DIR has ${FREE} GB free, $HFREPO needs ${MINFREE}"; exit 1; }
  echo "[fetch_weights] $HFREPO@${REV:0:10} -> $DEST"
  "${HF_DL[@]}" "$HFREPO" --revision "$REV" --local-dir "$DEST"
  echo "[fetch_weights] hashing weight files of $KEY"
  m_record "$KEY" "$HFREPO" "$REV" "$DEST" weights
}

fetch_code() {      # $1=key $2=git url $3=commit $4=target dir: clone, then check out the pinned commit (detached)
  local KEY=$1 URL=$2 REV=$3 DEST=$4
  if m_current "$KEY" "$REV" "$DEST" && [ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null)" = "$REV" ]; then
    echo "[fetch_weights] $URL@${REV:0:10} already present in $DEST -- skipped"; return
  fi
  [ -d "$DEST/.git" ] || git clone "$URL" "$DEST"
  git -C "$DEST" fetch --quiet origin
  git -C "$DEST" checkout --quiet --detach "$REV"
  [ "$(git -C "$DEST" rev-parse HEAD)" = "$REV" ] || { echo "[fetch_weights] FATAL: checkout of $REV in $DEST failed"; exit 1; }
  m_record "$KEY" "$URL" "$REV" "$DEST" code
}

# small ones first: a failure of the 147 GB download then leaves everything else recorded
fetch_code    editreward_code        "$CODE_REPO"       "$CODE_REV"       "$EDITREWARD_DIR"
fetch_code    ivebench_code          "$IVEBENCH_REPO"   "$IVEBENCH_REV"   "$IVEBENCH_DIR"
fetch_weights editreward_mimo_vl_7b  "$EDITREWARD_REPO" "$EDITREWARD_REV"
fetch_weights mimo_vl_7b_sft_base    "$MIMO_BASE_REPO"  "$MIMO_BASE_REV"
fetch_weights videoclip_xl_v2        "$VIDEOCLIP_REPO"  "$VIDEOCLIP_REV"
fetch_weights qwen25_vl_72b_instruct "$QWEN72B_REPO"    "$QWEN72B_REV"    "$QWEN72B_MIN_FREE_GB"

echo "[fetch_weights] done -- manifest: $MANIFEST"
python - "$MANIFEST" <<'PY'
import json, sys
for k, e in json.load(open(sys.argv[1])).items():
    print(f"  {k:24s} {e['revision'][:10]}  {len(e['files'])} weight file(s)  {e['path']}")
PY
