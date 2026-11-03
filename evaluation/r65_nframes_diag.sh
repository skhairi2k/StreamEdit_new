#!/usr/bin/env bash
# R65: the smoke (2 repro clips, then 2 smoke pairs) at other frame counts, run on an interactive GPU node, to see which
# n_frames reproduces FiVE's released AnyV2V outputs (paper: 32 frames; AnyV2V repo: 16; the first smoke used 40).
# Everything but n_frames is the smoke's setting. Each N gets its own tree, so the 40-frame smoke is untouched:
#   $OUT/nframes/n<N>/{repro,smoke}_{raw512,srcres}, ..._manifest_shard0.csv, src512/, inversions/
# Case lists: evaluation/r65_cases_{repro,smoke}_n<N>.json = the 40-frame lists with n_frames = min(N, n_frames).
# Order: both repro runs first (they decide the gate), then both smoke runs.
#
# Usage: bash evaluation/r65_nframes_diag.sh [N ...]      (default: 32 16)
set -uo pipefail

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT=~/Data/dataggen/outputs/five_bench/r65_anyv2v
PY=~/anaconda3/envs/anyv2v/bin/python
NS=("$@"); [ "${#NS[@]}" -eq 0 ] && NS=(32 16)

cd "$REPO" || exit 1
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CPATH PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1

PREFIX=$("$PY" -I -c 'import sys; print(sys.prefix)')
[ "$PREFIX" = "$HOME/anaconda3/envs/anyv2v" ] || { echo "[r65_nframes] FATAL: python prefix $PREFIX is not the anyv2v env"; exit 1; }
echo "[r65_nframes] node=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) N=${NS[*]} $(date '+%F %T')"

for N in "${NS[@]}"; do
  for NAME in repro smoke; do
    "$PY" -I - "$N" "$NAME" <<'PYEOF'
import json, sys
n, name = int(sys.argv[1]), sys.argv[2]
cases = json.load(open(f"evaluation/r65_cases_{name}.json"))
for c in cases:
    c["n_frames"] = min(n, c["n_frames"])
json.dump(cases, open(f"evaluation/r65_cases_{name}_n{n}.json", "w"), indent=1, ensure_ascii=False)
PYEOF
  done
done

FAILED=0
for NAME in repro smoke; do
  for N in "${NS[@]}"; do
    echo "--- [r65_nframes] ${NAME} n=${N} $(date '+%F %T') ---"
    mkdir -p "$OUT/nframes/n${N}"
    "$PY" -I evaluation/r65_anyv2v_run.py --cases "evaluation/r65_cases_${NAME}_n${N}.json" \
      --out_root "$OUT/nframes/n${N}/${NAME}" --shard 0 --n_shards 1 \
      || { echo "[r65_nframes] FAILED ${NAME} n=${N}: exit $?"; FAILED=$((FAILED + 1)); }
    cat "$OUT/nframes/n${N}/${NAME}_manifest_shard0.csv" 2>&1
  done
done
echo "[r65_nframes] failures: ${FAILED} $(date '+%F %T')"
