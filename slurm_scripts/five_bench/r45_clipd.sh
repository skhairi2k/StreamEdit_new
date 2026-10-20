#!/bin/bash
# R45 -- CLIP-D (directional CLIP) for the R45 methods over the WHOLE 419-pair bench.
# Single task, not arrayed: source-frame embeddings are cached per clip and reused across
# methods. r38_clipd.sh otherwise.
#
#   sbatch slurm_scripts/five_bench/r45_clipd.sh                  # every row of the eval table
#   sbatch slurm_scripts/five_bench/r45_clipd.sh METHOD ...       # a subset, e.g. the 8 N=15
#                                                                 # sweep methods before sweepM exists
# Methods are the names of `python evaluation/r45_select.py eval --table`; each is scored
# under its CSV label (r45_{method} for the sweeps, the method itself for the routed arms) and
# appended to ONE file, evaluation/csv/r45_clip_directional.csv -- re-running with a subset
# rewrites the file with that subset only, so score everything in one call at the end.
#
# --frame_stride 8, no --max_frames: the stride r45_eval.sh's harness scores at.
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES, so a partial
# table exits 0: rows are counted per method instead.
#
# --time=24:00:00: up to 24 methods x 419 clips (R38 scored 4 x 419 inside 6 h).
#SBATCH --job-name=r45_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/r45_clipd_%j.out
#SBATCH --error=logs/r45_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT="$REPO/evaluation/csv/r45_clip_directional.csv"
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

cd "$REPO" || exit 1
# Conda: five-bench, NOT streamgve (the R2 env bug).
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

TABLE=$(python evaluation/r45_select.py eval --table) || exit 1
declare -A ROOT=()
ALL=()
while IFS=$'\t' read -r _ M D; do ROOT[$M]=$D; ALL+=("$M"); done <<< "$TABLE"
if [ "$#" -gt 0 ]; then SEL=("$@"); else SEL=("${ALL[@]}"); fi

echo "[r45_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r45_clipd] job=${SLURM_JOB_ID:-local} methods=${SEL[*]} stride=8 full bench"

# ---- preflight: every method known to the table and rendered complete (REAL pairs only).
METHODS=()
LABELS=()
BAD=0
for M in "${SEL[@]}"; do
  DIR=${ROOT[$M]}
  if [ -z "$DIR" ]; then echo "[r45_clipd] unknown method '$M' (not in r45_select.py eval --table)"; exit 1; fi
  case "$M" in r45_*) L="$M" ;; *) L="r45_${M}" ;; esac
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r45_clipd] ${M} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("${L}=${DIR}")
  LABELS+=("$L")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r45_clipd] FAILED -- render incomplete, run /run-step R45 wait-renders first"; exit 1
fi
echo "[r45_clipd] preflight OK: ${#SEL[@]} methods x 419 real pairs"

RUNLOG="logs/r45_clipd_${SLURM_JOB_ID:-local}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r45_clipd] exit=${RC}"
if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r45_clipd] FAILED -- exit ${RC}, or $OUT not written"; exit 1
fi

# ---- post-run guard: 419 rows per method, exact per-type breakdown, no MISSING lines.
NMISS=$(grep -c 'MISSING' "$RUNLOG")
python - "$OUT" "${LABELS[@]}" <<'PY'
import csv, sys
from collections import Counter
out, labels = sys.argv[1], sys.argv[2:]
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
rows = list(csv.DictReader(open(out)))
bad = 0
for m in labels:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == m)
    ok = dict(got) == expect
    bad += not ok
    print(f"[r45_clipd] {m}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r45_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r45_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"; exit 1
fi
echo "[r45_clipd] OK -> $OUT"
exit 0
