#!/bin/bash
# R38 -- CLIP-D (directional CLIP) for the N=5-draft combos over the WHOLE 419-pair bench,
# scored on each combo's FINAL stage-3 render (step14). r35_clipd.sh with R38's roots.
# Single task, not arrayed: source-frame embeddings are cached per clip and reused across
# combos.
#
#   sbatch slurm_scripts/five_bench/r38_clipd.sh [COMBO ...]
# With no arguments it scores all four N=5 combos. If gate-lowN dropped the tgt09 combos,
# pass the survivors (`dino_n5_unblended lpips_n5_unblended`) -- the all-four default would
# fail preflight on the dropped ones.
#
# R35's four N=15 arms are NOT re-scored: evaluation/csv/r35_clip_directional.csv already
# holds them (R38's N=15 points, smoke gate G1, job 1016420). Output here is a separate
# file, evaluation/csv/r38_clip_directional.csv, so R35's table is never overwritten.
#
# Same options as R35: --methods NAME=DIR (one per combo) and --fullbench (419 pairs from
# edit{T}_FiVE.json). --frame_stride 8, no --max_frames: must match the stride
# r38_eval.sh's harness scored at, so CLIP-D and the LPIPS column describe the same frames.
#
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES, so a partial
# table exits 0. Rows are counted per combo instead.
#
# --time=06:00:00: R35's 4 x 419 run fit comfortably (job 1012708).
#SBATCH --job-name=r38_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r38_clipd_%j.out
#SBATCH --error=logs/r38_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r38_arms
OUT="$REPO/evaluation/csv/r38_clip_directional.csv"

ALL_COMBOS=(dino_n5_unblended dino_n5_tgt09 lpips_n5_unblended lpips_n5_tgt09)
if [ "$#" -gt 0 ]; then COMBOS=("$@"); else COMBOS=("${ALL_COMBOS[@]}"); fi
for C in "${COMBOS[@]}"; do
  case " ${ALL_COMBOS[*]} " in
    *" $C "*) ;;
    *) echo "[r38_clipd] unknown combo '$C' (expected one of ${ALL_COMBOS[*]})"; exit 1 ;;
  esac
done
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

cd "$REPO" || exit 1
# Conda: five-bench, NOT streamgve (the R2 env bug).
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r38_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r38_clipd] job=${SLURM_JOB_ID:-local} combos=${COMBOS[*]} stride=8 full bench"

# ---- preflight: every combo's render complete, per edit type (REAL pairs only).
METHODS=()
BAD=0
for C in "${COMBOS[@]}"; do
  DIR="$OUT_ROOT/r38_${C}/step14"
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r38_clipd] ${C} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("r38_${C}=${DIR}")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r38_clipd] FAILED -- render incomplete, run /run-step R38 wait-stage3 first"
  exit 1
fi
echo "[r38_clipd] preflight OK: ${#COMBOS[@]} combos x 419 real pairs"

RUNLOG="logs/r38_clipd_${SLURM_JOB_ID:-local}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r38_clipd] exit=${RC}"

if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r38_clipd] FAILED -- exit ${RC}, or $OUT not written"
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
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == f"r38_{c}")
    ok = dict(got) == expect
    bad += not ok
    print(f"[r38_clipd] r38_{c}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r38_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r38_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"
  exit 1
fi

echo "[r38_clipd] OK -> $OUT"
exit 0
