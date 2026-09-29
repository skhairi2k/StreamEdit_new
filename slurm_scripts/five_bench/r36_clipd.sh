#!/bin/bash
# R36 -- CLIP-D (directional CLIP) for the 8 VP rho arms over the WHOLE 419-pair bench.
# Clone of r35_clipd.sh. Single task, not arrayed: the source-frame embeddings are cached
# per clip and reused across all 8 arms, so arraying would recompute that cache 8x.
#
# CLIP-D, not only the stored clip_similarity_target_image: R33 measured the incumbent not
# to rank renders by edit strength and chose clip_d_prompt as the achievement axis of record
# (see evaluation/r33_clip_directional.py's docstring).
#
# Uses R35's two options of r33_clip_directional.py, unchanged:
#   --methods NAME=DIR   the 8 rho arms, instead of a --which roster
#   --fullbench          all 419 pairs from edit{T}_FiVE.json
#
# --frame_stride 8, no --max_frames: must match the stride r36_eval.sh's harness scored at,
# so CLIP-D and the LPIPS column describe the same frames.
#
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES for any clip
# dir it cannot find, so a partial table exits 0. Rows are counted per arm instead.
#
# --time=06:00:00: R33's job 1002662 scored 616 method-clips in ~4.5 min at stride 8;
# this is 8 x 419 = 3352 method-clips, ~25 min at that rate.
#SBATCH --job-name=r36_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r36_clipd_%j.out
#SBATCH --error=logs/r36_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r36_rho_sweep
OUT="$REPO/evaluation/csv/r36_clip_directional.csv"

RHOS=(2 3 4 6 8 10 20 50)
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

echo "[r36_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r36_clipd] job=${SLURM_JOB_ID} rhos=${RHOS[*]} stride=8 full bench"

# ---- preflight: every arm's render complete, per edit type (REAL pairs only --
# evaluate.py leaves a {video}_resize sibling next to every video it scores).
METHODS=()
ARMS=()
BAD=0
for R in "${RHOS[@]}"; do
  ARM="r36_rho${R}_vp"
  DIR="$OUT_ROOT/$ARM"
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r36_clipd] ${ARM} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("${ARM}=${DIR}")
  ARMS+=("$ARM")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r36_clipd] FAILED -- render incomplete, run /run-step R36 wait-infer first"
  exit 1
fi
echo "[r36_clipd] preflight OK: 8 arms x 419 real pairs"

RUNLOG="logs/r36_clipd_${SLURM_JOB_ID}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r36_clipd] exit=${RC}"

if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r36_clipd] FAILED -- exit ${RC}, or $OUT not written"
  exit 1
fi

# ---- post-run guard: 419 rows per arm, exact per-type breakdown, no MISSING lines.
NMISS=$(grep -c 'MISSING' "$RUNLOG")
python - "$OUT" "${ARMS[@]}" <<'PY'
import csv, sys
from collections import Counter
out, arms = sys.argv[1], sys.argv[2:]
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
rows = list(csv.DictReader(open(out)))
bad = 0
for a in arms:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == a)
    ok = dict(got) == expect
    bad += not ok
    print(f"[r36_clipd] {a}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r36_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r36_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"
  exit 1
fi

echo "[r36_clipd] OK -> $OUT"
exit 0
