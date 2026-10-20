#!/bin/bash
# R40 -- CLIP-D (directional CLIP) for the two draft-free combos over the WHOLE 419-pair
# bench, on each combo's FINAL stage-3 render (step14). Single task: the source-frame
# embeddings are cached per clip and reused across combos. Copy of r35_clipd.sh: only
# combos, roots, method names and the output CSV change.
#
# Uses r33_clip_directional.py's --methods NAME=DIR and --fullbench (added for R35).
# --frame_stride 8 matches r40_eval.sh's harness, so CLIP-D and LPIPS see the same frames.
#
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES, so a partial
# table exits 0; rows are counted per combo instead.
#
# --time=06:00:00 kept from R35 (4 x 419 method-clips); R40 is half that.
#SBATCH --job-name=r40_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r40_clipd_%j.out
#SBATCH --error=logs/r40_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r40_arms
OUT="$REPO/evaluation/csv/r40_clip_directional.csv"

COMBOS=(dino_warp lpips_warp)
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

cd "$REPO" || exit 1
# Conda: five-bench, NOT streamgve (the R2 env bug: .bashrc auto-activates streamgve and
# `conda activate five-bench` alone does not pop it).
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r40_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r40_clipd] job=${SLURM_JOB_ID} combos=${COMBOS[*]} stride=8 full bench"

# ---- preflight: every combo's render complete, per edit type (REAL pairs only --
# evaluate.py leaves a {video}_resize sibling next to every video it scores).
METHODS=()
BAD=0
for C in "${COMBOS[@]}"; do
  DIR="$OUT_ROOT/r40_${C}/step14"
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r40_clipd] ${C} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("r40_${C}=${DIR}")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r40_clipd] FAILED -- render incomplete, run /run-step R40 wait-stage3 first"
  exit 1
fi
echo "[r40_clipd] preflight OK: 2 combos x 419 real pairs"

RUNLOG="logs/r40_clipd_${SLURM_JOB_ID}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r40_clipd] exit=${RC}"

if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r40_clipd] FAILED -- exit ${RC}, or $OUT not written"
  exit 1
fi

# ---- post-run guard: 419 rows per combo, exact per-type breakdown, no MISSING lines.
NMISS=$(grep -c 'MISSING' "$RUNLOG")
python - "$OUT" "${COMBOS[@]}" <<'PY'
import csv, sys
from collections import Counter
out, combos = sys.argv[1], sys.argv[2:]
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
rows = list(csv.DictReader(open(out)))
bad = 0
for c in combos:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == f"r40_{c}")
    ok = dict(got) == expect
    bad += not ok
    print(f"[r40_clipd] r40_{c}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r40_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r40_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"
  exit 1
fi

echo "[r40_clipd] OK -> $OUT"
exit 0
