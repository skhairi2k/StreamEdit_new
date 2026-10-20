#!/bin/bash
# R48 CLIP-D: directional CLIP score (r33_clip_directional.py) for EVERY unit in one job (the source embeddings are shared).
#
# Usage: sbatch slurm_scripts/five_bench/r48_clipd.sh subset     # evaluation/r48_cases.json, 17 units
#        sbatch slurm_scripts/five_bench/r48_clipd.sh full       # --fullbench (419 pairs), the units of r48_units.py --mode full
# Same units and frame stride (8) as r48_eval.sh. Wan-Edit frames (41) are paired positionally with the source and truncated
# to the shorter side (r33 uses n = min(...)). Output: evaluation/csv/r48_clip_directional_{MODE}.csv, method column r48_{unit}.
#SBATCH --job-name=r48_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/r48_clipd_%j.out
#SBATCH --error=logs/r48_clipd_%j.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json; SCOPE_ARG=(--cases "$CASES") ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; SCOPE_ARG=(--fullbench) ;;
  *) echo "[r48_clipd] usage: r48_clipd.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
PY_FB=~/anaconda3/envs/five-bench/bin/python
OUT="$REPO/evaluation/csv/r48_clip_directional_${MODE}.csv"
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
echo "[r48_clipd] job=${SLURM_JOB_ID:-local} mode=${MODE} stride=8 expect per type: ${EXPECT[*]}"
echo "[r48_clipd] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

METHODS=(); NAMES=(); BAD=0
while IFS=$'\t' read -r BASE DIR HOST CONFIG; do
  MISSING=$($PY_FB evaluation/r48_units.py --mode "$MODE" --missing "$DIR" --cases "$CASES")
  [ -n "$MISSING" ] && { echo "[r48_clipd] ${BASE}: $(echo "$MISSING" | wc -l) case clip(s) not rendered under $DIR, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; BAD=$((BAD + 1)); }
  METHODS+=("r48_${BASE}=${DIR}"); NAMES+=("r48_${BASE}")
done < <($PY_FB evaluation/r48_units.py --mode "$MODE" 2>/dev/null)
[ "${#METHODS[@]}" -gt 0 ] || { echo "[r48_clipd] FAILED -- no units"; exit 1; }
[ "$BAD" -ne 0 ] && { echo "[r48_clipd] FAILED -- renders incomplete, run the render step (wait-render) first"; exit 1; }
echo "[r48_clipd] preflight OK: ${#METHODS[@]} units, complete per type"

RUNLOG="logs/r48_clipd_${SLURM_JOB_ID:-local}.run.log"
$PY_FB evaluation/r33_clip_directional.py "${SCOPE_ARG[@]}" --methods "${METHODS[@]}" --data_root "$DATA_ROOT" \
  --frame_stride 8 -o "$OUT" 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r48_clipd] exit=${RC}"
{ [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; } && { echo "[r48_clipd] FAILED -- exit ${RC}, or $OUT not written"; exit 1; }
NMISS=$(grep -c 'MISSING' "$RUNLOG")
$PY_FB - "$OUT" "${EXPECT[@]}" -- "${NAMES[@]}" <<'PY'
import csv, sys
from collections import Counter
args = sys.argv[1:]
out, expect, names = args[0], dict(zip(range(1, 7), map(int, args[1:7]))), args[8:]
expect = {t: n for t, n in expect.items() if n}
rows = list(csv.DictReader(open(out)))
bad = 0
for m in names:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == m)
    ok = dict(got) == expect
    bad += not ok
    print(f"[r48_clipd] {m}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r48_clipd] missing_lines=${NMISS}"
{ [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; } && { echo "[r48_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"; exit 1; }
echo "[r48_clipd] OK -> $OUT"
exit 0
