#!/bin/bash
# R45 -- FiVE-Acc alone (evaluate.py --metrics five_acc ONLY) for every R45 method, one array
# task per row of `python evaluation/r45_select.py eval --table`, over the WHOLE 419-pair
# bench. r38_fiveacc.sh otherwise.
#
#   sbatch --array=$(python evaluation/r45_select.py eval) slurm_scripts/five_bench/r45_fiveacc.sh
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
#SBATCH --time=04:00:00
#SBATCH --output=logs/r45_fiveacc_%A_%a.out
#SBATCH --error=logs/r45_fiveacc_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
TID=${SLURM_ARRAY_TASK_ID}
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

if [ -z "$TID" ]; then echo "[r45_fiveacc] no SLURM_ARRAY_TASK_ID"; exit 1; fi
python evaluation/r45_select.py check "$TID" eval || exit 1
ROW=$(python evaluation/r45_select.py eval --table | awk -F'\t' -v i="$TID" '$1==i')
METHOD=$(echo "$ROW" | cut -f2)
DIR=$(echo "$ROW" | cut -f3)
STEM="r45_fiveacc_${METHOD#r45_}"

echo "[r45_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r45_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} stem=${STEM}"
echo "[r45_fiveacc] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the render must be complete, per edit type (REAL pairs only).
if [ ! -d "$DIR" ]; then
  echo "[r45_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"; exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r45_fiveacc] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r45_fiveacc] FAILED ${STEM} -- render incomplete, run /run-step R45 wait-renders first"; exit 1
fi
echo "[r45_fiveacc] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r45_fiveacc_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r45_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"
FAILED=0
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r45_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"; FAILED=1
fi

# ---- post-run guard: every clip scored, both FiVE-Acc columns present, NaN-free, binary.
python - "$STEM" <<'PY' || FAILED=1
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

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r45_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
