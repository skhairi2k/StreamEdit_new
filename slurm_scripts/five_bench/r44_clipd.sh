#!/bin/bash
# R44 -- CLIP-D (directional CLIP) for R44's TEXT-ONLY renders over the WHOLE 419-pair
# bench. Merge of r36_clipd.sh (sweep) / r38_clipd.sh (arms). Single task, not arrayed:
# the source-frame embeddings are cached per clip and reused across every method.
#
# USAGE:
#   sbatch slurm_scripts/five_bench/r44_clipd.sh sweep N [RHO,RHO,...]   # N = 15 | 20 | 30
#       the 8 rho arms r44_n{N}_rho{R}_novp -> evaluation/csv/r44_clip_directional_sweep_n{N}.csv
#       the optional 3rd arg restricts to a comma-separated rho subset (e.g. "2,4,10,50") for
#       a sweep run at fewer than the standard 8 rhos; omit it for the full 8 (default).
#   sbatch slurm_scripts/five_bench/r44_clipd.sh arms
#       the selected combos r44_{combo} (evaluation/r44_selected_combos.txt, step14 renders)
#       -> evaluation/csv/r44_clip_directional_arms.csv (rewritten whole on every run)
#
# CLIP-D, not only the stored clip_similarity_target_image: R33 measured the incumbent not
# to rank renders by edit strength and chose clip_d_prompt as the achievement axis of record
# (see evaluation/r33_clip_directional.py's docstring).
#
# Uses R35's two options of r33_clip_directional.py, unchanged:
#   --methods NAME=DIR   the arms to score, instead of a --which roster
#   --fullbench          all 419 pairs from edit{T}_FiVE.json
#
# --frame_stride 8, no --max_frames: must match the stride r44_eval.sh's harness scored at,
# so CLIP-D and the LPIPS column describe the same frames.
#
# ⚠️ r33_clip_directional.py prints `MISSING {method}/{clip}` and CONTINUES for any clip
# dir it cannot find, so a partial table exits 0. Rows are counted per method instead.
#
# --time=06:00:00: R33's job 1002662 scored 616 method-clips in ~4.5 min at stride 8;
# the sweep is 8 x 419 = 3352 method-clips, ~25 min at that rate.
#SBATCH --job-name=r44_clipd
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r44_clipd_%j.out
#SBATCH --error=logs/r44_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench

RHOS=(2 3 4 6 8 10 20 50)
MODE=$1
NAMES=()
DIRS=()
case "$MODE" in
  sweep)
    N=$2
    case "$N" in 15|20|30) ;; *) echo "[r44_clipd] sweep needs N in {15,20,30}, got '${N}'"; exit 1 ;; esac
    if [ -n "$3" ]; then
      IFS=',' read -r -a SWEEP_RHOS <<< "$3"
      for R in "${SWEEP_RHOS[@]}"; do
        case " ${RHOS[*]} " in *" $R "*) ;; *) echo "[r44_clipd] '$R' is not one of the 8 rhos (${RHOS[*]})"; exit 1 ;; esac
      done
    else
      SWEEP_RHOS=("${RHOS[@]}")
    fi
    for R in "${SWEEP_RHOS[@]}"; do
      NAMES+=("r44_n${N}_rho${R}_novp")
      DIRS+=("$FIVE_ROOT/r44_rho_sweep/r44_n${N}_rho${R}_novp")
    done
    OUT="$REPO/evaluation/csv/r44_clip_directional_sweep_n${N}.csv"
    ;;
  arms)
    source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
    r44_load || exit 1
    for C in "${R44_COMBOS[@]}"; do
      NAMES+=("r44_${C}")
      DIRS+=("$FIVE_ROOT/r44_arms/r44_${C}/step14")
    done
    OUT="$REPO/evaluation/csv/r44_clip_directional_arms.csv"
    ;;
  *) echo "[r44_clipd] usage: r44_clipd.sh sweep N | arms"; exit 1 ;;
esac

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: five-bench, NOT streamgve (the R2 env bug).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r44_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r44_clipd] job=${SLURM_JOB_ID} mode=${MODE} methods=${NAMES[*]} stride=8 full bench"

# ---- preflight: every method's render complete, per edit type (real pairs only).
METHODS=()
BAD=0
for i in "${!NAMES[@]}"; do
  NAME=${NAMES[$i]}; DIR=${DIRS[$i]}
  for T in 1 2 3 4 5 6; do
    NP=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$NP" -ne "${EXPECT[$T]}" ] && { echo "[r44_clipd] ${NAME} edit${T}: ${NP} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  METHODS+=("${NAME}=${DIR}")
done
if [ "$BAD" -ne 0 ]; then
  echo "[r44_clipd] FAILED -- render incomplete, run the matching /run-step R44 wait-* step first"
  exit 1
fi
echo "[r44_clipd] preflight OK: ${#NAMES[@]} methods x 419 real pairs"

RUNLOG="logs/r44_clipd_${MODE}_${SLURM_JOB_ID:-local}.run.log"
python evaluation/r33_clip_directional.py \
  --fullbench \
  --methods "${METHODS[@]}" \
  --data_root "$DATA_ROOT" \
  --frame_stride 8 \
  -o "$OUT" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
echo "[r44_clipd] exit=${RC}"

if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r44_clipd] FAILED -- exit ${RC}, or $OUT not written"
  exit 1
fi

# ---- post-run guard: exactly 100/100/100/100/9/10 rows per method.
NMISS=$(grep -c 'MISSING' "$RUNLOG")
python - "$OUT" "${NAMES[@]}" <<'PY'
import csv, sys
from collections import Counter
out, names = sys.argv[1], sys.argv[2:]
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
rows = list(csv.DictReader(open(out)))
bad = 0
for m in names:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == m)
    ok = dict(got) == expect
    bad += not ok
    print(f"[r44_clipd] {m}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
GUARD=$?

echo "[r44_clipd] missing_lines=${NMISS}"
if [ "$GUARD" -ne 0 ] || [ "$NMISS" -ne 0 ]; then
  echo "[r44_clipd] FAILED -- incomplete table (see MISSING lines in $RUNLOG)"
  exit 1
fi
echo "[r44_clipd] OK -> $OUT"
exit 0
