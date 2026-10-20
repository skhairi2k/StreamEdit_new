#!/bin/bash
# R41 CLIP-D (directional CLIP, R33's axis of record) for the binary arms, whole 419-pair
# bench, final stage-3 render (step14). One THETA per array task, all COMBOS per task:
# the source-frame embeddings are cached per clip and reused across the combos, and 4
# combos per task is R35's single-job load (6 h).
#
# Submit with the array size from the config:
#   sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_theta_index) \
#          slurm_scripts/five_bench/r41_clipd.sh
# Task i scores theta THETA_ORDER[i] -> evaluation/csv/r41_n{N}_clip_directional_t{..}.csv
#
# r35_clipd.sh otherwise: --fullbench, --frame_stride 8 (the stride the eval harness scored
# at, so CLIP-D and LPIPS describe the same frames), no --max_frames.
#
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES for any clip dir it
# cannot find, so a partial table exits 0. Rows are counted per method instead.
#SBATCH --job-name=r41_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r41_clipd_%A_%a.out
#SBATCH --error=logs/r41_clipd_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
# exits here if DRAFT_N is unset; R41_CONFIG is for synthetic tests only (see r41_config.sh)
source "${R41_CONFIG:-slurm_scripts/five_bench/r41_config.sh}"

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "${#THETA_ORDER[@]}" ]; then
  echo "[r41_clipd] bad array id '${TID}' (expected 0-$(( ${#THETA_ORDER[@]} - 1 )))"; exit 1
fi
TH=${THETA_ORDER[$TID]}
OUT="$REPO/evaluation/csv/r41_${TAG}_clip_directional_t${TH/./}.csv"

mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r41_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r41_clipd] job=${SLURM_ARRAY_JOB_ID}_${TID} theta=${TH} combos=${COMBOS[*]} stride=8 full bench"

# ---- preflight: every combo's render at this theta complete.
METHODS=()
NAMES=()
BAD=0
for C in "${COMBOS[@]}"; do
  NAME="r41_${TAG}_${C}_t${TH/./}"
  DIR="$ARMS_ROOT/$NAME/step14"
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r41_clipd] ${NAME} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("${NAME}=${DIR}")
  NAMES+=("$NAME")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r41_clipd] FAILED -- render incomplete, run /run-step R41 wait-stage3 first"
  exit 1
fi
echo "[r41_clipd] preflight OK: ${#COMBOS[@]} methods x 419 real pairs"

RUNLOG="logs/r41_clipd_${SLURM_ARRAY_JOB_ID}_${TID}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r41_clipd] exit=${RC}"
if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r41_clipd] FAILED -- exit ${RC}, or $OUT not written"
  exit 1
fi

NMISS=$(grep -c 'MISSING' "$RUNLOG")
python - "$OUT" "${NAMES[@]}" <<'PY'
import csv, sys
from collections import Counter
out, names = sys.argv[1], sys.argv[2:]
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
rows = list(csv.DictReader(open(out)))
bad = 0
for n in names:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == n)
    ok = dict(got) == expect
    bad += not ok
    print(f"[r41_clipd] {n}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r41_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r41_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"
  exit 1
fi
echo "[r41_clipd] OK -> $OUT"
exit 0
