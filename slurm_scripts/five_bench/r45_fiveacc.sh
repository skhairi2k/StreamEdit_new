#!/bin/bash
# R45 -- FiVE-Acc alone (evaluate.py --metrics five_acc ONLY) for every R45 method, over the
# WHOLE 419-pair bench. r38_fiveacc.sh otherwise, but SEQUENTIAL: ONE job loops over every
# row of `python evaluation/r45_select.py eval --table` (sweep N=15 x8, sweep matched-N x8,
# the selected routed combos), instead of one array task per method. FiVE-Acc loads
# Qwen2.5-VL-7B, so it runs one method at a time, and a single job is 1 queue entry (an array
# of 20 would push eval 20 + fiveacc 20 + clipd 1 = 41 past the 30-job submit cap).
#
#   sbatch slurm_scripts/five_bench/r45_fiveacc.sh                     # every table row, in order
#   sbatch slurm_scripts/five_bench/r45_fiveacc.sh 0 1 2               # only these table rows
#   sbatch --array=5 slurm_scripts/five_bench/r45_fiveacc.sh           # one row (array mode kept
#                                                                      #  for reruns of a failed row)
#
# RESUME: a method whose six per-edit-type CSVs already hold the full row count with both
# FiVE-Acc columns present, NaN-free and binary is SKIPPED, so a timed-out job is resubmitted
# as-is. A failed method does NOT stop the loop; the job exits non-zero at the end.
# ~30 min per method (R44: 29:28 for the 6 annotation files) + model load -> ~12 h for 20.
#
# CSV stem: r45_fiveacc_{method without its r45_ prefix} (r45_fiveacc_ll_rho2_n15_vp,
# r45_fiveacc_ll_dino_n5_tgt09), so it never collides with r45_eval.sh's stems.
#
# WHY FIVE_ACC ALONE (r33_fiveacc.sh): Qwen2.5-VL-7B co-resident with the rest of the metric
# table exhausted the GPU in R20. ⚠️ evaluate.py swallows per-metric exceptions: the log is
# grepped and the five_acc columns are checked for full row count, presence and binary values.
#SBATCH --job-name=r45_fiveacc
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/r45_fiveacc_%j.out
#SBATCH --error=logs/r45_fiveacc_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# guard STEM: 0 iff all six per-edit-type CSVs hold the full row count and both FiVE-Acc
# columns are present, NaN-free and binary. Prints one line per edit type.
guard () {
python - "$1" <<'PY'
import csv, sys
stem = sys.argv[1]
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
need = ("five_acc_yes_no", "five_acc_multi_choice")
bad = 0
for t, n_exp in expect.items():
    path = f"evaluation/csv/edit{t}_FiVE_{stem}_frame_stride8.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"[r45_fiveacc] edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols
                 and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r45_fiveacc] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), "
          f"missing cols={miss or 0}, non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
}

# score_one TID: one table row. Returns 0 on success or skip, 1 on failure.
score_one () {
  local TID=$1
  python evaluation/r45_select.py check "$TID" eval || return 1
  local ROW METHOD DIR STEM
  ROW=$(python evaluation/r45_select.py eval --table | awk -F'\t' -v i="$TID" '$1==i')
  METHOD=$(echo "$ROW" | cut -f2)
  DIR=$(echo "$ROW" | cut -f3)
  STEM="r45_fiveacc_${METHOD#r45_}"

  if guard "$STEM" > /dev/null 2>&1; then
    echo "[r45_fiveacc] SKIP ${STEM}: already complete"; return 0
  fi
  echo "--- [r45_fiveacc] row ${TID} method=${METHOD} stem=${STEM} $(date '+%F %T') ---"
  echo "[r45_fiveacc] scoring $DIR  (full bench, no --cases_json)"

  # ---- preflight: the render must be complete, per edit type (REAL pairs only).
  if [ ! -d "$DIR" ]; then
    echo "[r45_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"; return 1
  fi
  local BAD=0 T N
  for T in 1 2 3 4 5 6; do
    N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
    [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r45_fiveacc] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
  done
  if [ "$BAD" -ne 0 ]; then
    echo "[r45_fiveacc] FAILED ${STEM} -- render incomplete, run /run-step R45 wait-renders first"; return 1
  fi
  echo "[r45_fiveacc] preflight OK: 419 real pairs, 100/100/100/100/9/10"

  local METRICS_LOG="logs/r45_fiveacc_${SLURM_JOB_ID:-local}_row${TID}.metrics.log"
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --metrics five_acc \
    --src_image_folder "$DATA_ROOT" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --tgt_methods "$DIR" \
    --tgt_layout edit_video \
    --tmp_resize \
    --result_path "evaluation/csv/${STEM}.csv" \
    2>&1 | tee "$METRICS_LOG"
  local RC=${PIPESTATUS[0]}

  local NERR NOOM
  NERR=$(grep -c 'Error:' "$METRICS_LOG")
  NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
  echo "[r45_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"
  local FAILED=0
  if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
    echo "[r45_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"; FAILED=1
  fi
  guard "$STEM" || FAILED=1
  if [ "$FAILED" -ne 0 ]; then return 1; fi
  echo "[r45_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
  return 0
}

echo "[r45_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} job=${SLURM_JOB_ID:-local}"
nvidia-smi -L 2>&1 | head -2

# Which rows: array mode -> that one row; positional args -> those rows; default -> all.
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  ROWS=("$SLURM_ARRAY_TASK_ID")
elif [ "$#" -gt 0 ]; then
  ROWS=("$@")
else
  ROWS=($(python evaluation/r45_select.py eval --table | cut -f1)) || exit 1
fi
echo "[r45_fiveacc] rows: ${ROWS[*]}"

FAILED=0
for TID in "${ROWS[@]}"; do
  score_one "$TID" || { FAILED=$((FAILED + 1)); echo "[r45_fiveacc] row ${TID} FAILED -- continuing"; }
done
echo "[r45_fiveacc] ${#ROWS[@]} rows, failures: ${FAILED}"
exit $(( FAILED > 0 ))
