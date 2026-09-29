#!/bin/bash
# R36 -- FiVE-Acc alone for the 8 VP rho arms, one array task per rho, over the WHOLE
# 419-pair bench, with evaluate.py --metrics five_acc ONLY. Clone of r35_fiveacc.sh.
#
# ARRAY LAYOUT: same rho order as r36_infer.sh / r36_eval.sh -- do not reorder one
# without the others.
#   0 rho2  1 rho3  2 rho4  3 rho6  4 rho8  5 rho10  6 rho20  7 rho50
#
# WHY FIVE_ACC ALONE: R20's GPU exhaustion was Qwen2.5-VL-7B (five_acc) co-resident with
# CoTracker and the rest of the metric table on one GPU. r36_eval.sh drops five_acc; this
# job runs it alone, so nothing co-resides and no H100 is needed.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, and a raised metric
# DROPS its column. The log is teed and grepped, and the per-edit-type CSVs are checked
# for the full row count AND for the five_acc columns being present and NaN-free.
#
# --tmp_resize (R36): resized frames go to a per-clip temp dir deleted after the clip is
# scored, not a persistent {video}_resize sibling -- keeps ~2 GB/arm off the home quota.
# Metric values unchanged (A/B on 2 smoke clips: lpips + niqe CSVs byte-identical).
# --time=04:00:00: R33's five_acc-only task scored 22 clips in 1 min 50 s end to end
# (logs/r33_fiveacc_1004302_16.metrics.log); 419 clips is well under an hour.
#SBATCH --job-name=r36_fiveacc
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/r36_fiveacc_%A_%a.out
#SBATCH --error=logs/r36_fiveacc_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r36_rho_sweep

RHOS=(2 3 4 6 8 10 20 50)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
  echo "[r36_fiveacc] bad array id ${TID} (expected 0-7)"; exit 1
fi
R=${RHOS[$TID]}
ARM="r36_rho${R}_vp"
# Distinct stem prefix so these never collide with the 9-metric CSVs r36_eval.sh writes
# under r36_rho{R}_vp (r33/r35_fiveacc precedent).
STEM="r36_fiveacc_rho${R}_vp"
DIR="$OUT_ROOT/$ARM"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

echo "[r36_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r36_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} rho=${R} stem=${STEM}"
echo "[r36_fiveacc] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the arm's render must be complete, per edit type (REAL pairs only --
# evaluate.py leaves a {video}_resize sibling next to every video it scores).
if [ ! -d "$DIR" ]; then
  echo "[r36_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r36_fiveacc] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r36_fiveacc] FAILED ${STEM} -- render incomplete, run /run-step R36 wait-infer first"
  exit 1
fi
echo "[r36_fiveacc] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r36_fiveacc_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r36_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ]; then
  echo "[r36_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

# ---- post-run guard: every clip scored, and the two FiVE-Acc columns the summary reads
# (yes/no and multiple-choice) present, NaN-free, and binary.
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
        print(f"[r36_fiveacc] edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols
                 and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r36_fiveacc] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), "
          f"missing cols={miss or 0}, non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r36_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
