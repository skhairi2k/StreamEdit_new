#!/bin/bash
# R35 -- CLIP-D (directional CLIP) for the four (arm, regime) combos over the WHOLE
# 419-pair bench, scored on each combo's FINAL stage-3 render (step14). Single task, not
# arrayed: the source-frame embeddings are cached per clip and reused across all four
# combos, so arraying would recompute that cache 4x.
#
# The r7_visual_prompting baseline is NOT scored here -- the VP rho sweep is its own task.
#
# CLIP-D, not the stored clip_similarity_target_image: R33 measured the incumbent not to
# rank renders by edit strength and chose clip_d_prompt as the achievement axis of record
# (see evaluation/r33_clip_directional.py's docstring).
#
# Uses two options added to r33_clip_directional.py for R35 (both default off, so every
# earlier invocation is unchanged):
#   --methods NAME=DIR   the four combos, instead of a --which roster
#   --fullbench          all 419 pairs from edit{T}_FiVE.json, since cases.json covers 22
#                        and the full-bench manifest carries no prompts. src/trg prompts are
#                        identical to cases.json's on its 22 clips; the short word fields
#                        (clip_d_word only) use FiVE's raw objects, not cases.json's edits.
#
# --frame_stride 8, no --max_frames: must match the stride r35_eval.sh's harness scored
# at, so CLIP-D and the LPIPS column describe the same frames.
#
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES for any clip
# dir it cannot find, so a partial table exits 0. Rows are counted per combo instead.
#
# --time=06:00:00: R33's 28-arm x 22-clip run (616 method-clips, stride 8) fit in 2 h;
# this is 4 x 419 = 1676 method-clips, ~2.7x.
#SBATCH --job-name=r35_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r35_clipd_%j.out
#SBATCH --error=logs/r35_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r35_arms
OUT="$REPO/evaluation/csv/r35_clip_directional.csv"

COMBOS=(dino_unblended dino_first2 lpips_unblended lpips_first2)
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

echo "[r35_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r35_clipd] job=${SLURM_JOB_ID} combos=${COMBOS[*]} stride=8 full bench"

# ---- preflight: every combo's render complete, per edit type (REAL pairs only --
# evaluate.py leaves a {video}_resize sibling next to every video it scores).
METHODS=()
BAD=0
for C in "${COMBOS[@]}"; do
  DIR="$OUT_ROOT/r35_${C}/step14"
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r35_clipd] ${C} edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("r35_${C}=${DIR}")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r35_clipd] FAILED -- render incomplete, run /run-step R35 wait-stage3 first"
  exit 1
fi
echo "[r35_clipd] preflight OK: 4 combos x 419 real pairs"

RUNLOG="logs/r35_clipd_${SLURM_JOB_ID}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r35_clipd] exit=${RC}"

if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r35_clipd] FAILED -- exit ${RC}, or $OUT not written"
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
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == f"r35_{c}")
    ok = dict(got) == expect
    bad += not ok
    print(f"[r35_clipd] r35_{c}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?
echo "[r35_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r35_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"
  exit 1
fi

echo "[r35_clipd] OK -> $OUT"
exit 0
