#!/bin/bash
# R38 -- FiVE-Acc alone for the four (arm, regime) combos, one array task per combo,
# scored on each combo's FINAL stage-3 render (step14) over the WHOLE 419-pair bench,
# with evaluate.py --metrics five_acc ONLY.
#
# ARRAY LAYOUT: same combo order as r38_stage2/stage3/eval.sh -- do not reorder one
# without the others.
#   0 dino_n5_unblended   1 dino_n5_tgt09   2 lpips_n5_unblended   3 lpips_n5_tgt09
# If gate-lowN dropped the tgt09 combos, submit with --array=0,2.
# R35's four N=15 arms are NOT re-scored: their CSVs (r35_* stems) are R38's N=15
# points (smoke gate G1, job 1016420).
#
# WHY FIVE_ACC ALONE (r33_fiveacc.sh's reasoning, unchanged): R20's GPU exhaustion was
# Qwen2.5-VL-7B (five_acc) co-resident with CoTracker and the rest of the metric table on
# one GPU. r38_eval.sh drops five_acc; this job runs it alone, so nothing co-resides.
#
# Scores the 4 R38 N=5 arms ONLY. The r7_visual_prompting (VP) baseline is not scored
# here -- the VP rho sweep is its own task.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, and a raised metric
# DROPS its column. The log is teed and grepped, and the per-edit-type CSVs are checked
# for the full row count AND for the five_acc columns being present and NaN-free.
#
# --time=04:00:00: R33's five_acc-only task scored 22 clips in 1 min 50 s end to end
# (tqdm in logs/r33_fiveacc_1004302_16.metrics.log); 419 clips is well under an hour
# even pessimistically. R33 budgeted 6 h for 22 clips, far more than it needed.
# --tmp_resize (added 2026-09-28, as in R36): resized frames go to a per-clip temp dir
# deleted after scoring, not a persistent {video}_resize sibling -- keeps them off the home
# quota. Metric values unchanged (R36 A/B: lpips + niqe CSVs byte-identical).
#SBATCH --job-name=r38_fiveacc
#SBATCH --partition=L40S
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r38_fiveacc_%A_%a.out
#SBATCH --error=logs/r38_fiveacc_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r38_arms

COMBOS=(dino_n5_unblended dino_n5_tgt09 lpips_n5_unblended lpips_n5_tgt09)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r38_fiveacc] bad array id ${TID} (expected 0-3)"; exit 1
fi
COMBO=${COMBOS[$TID]}
# r33_fiveacc_{arm} precedent: a distinct stem prefix so these never collide with the
# 9-metric CSVs r38_eval.sh writes under r38_{combo}.
STEM="r38_fiveacc_${COMBO}"
DIR="$OUT_ROOT/r38_${COMBO}/step14"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

echo "[r38_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r38_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} stem=${STEM}"
echo "[r38_fiveacc] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the combo's render must be complete, per edit type (REAL pairs only --
# evaluate.py leaves a {video}_resize sibling next to every video it scores).
if [ ! -d "$DIR" ]; then
  echo "[r38_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r38_fiveacc] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r38_fiveacc] FAILED ${STEM} -- render incomplete, run /run-step R38 wait-stage3 first"
  exit 1
fi
echo "[r38_fiveacc] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r38_fiveacc_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r38_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r38_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

# ---- post-run guard: every clip scored, and the two FiVE-Acc columns the score step
# reads (yes/no and multiple-choice) present, NaN-free, and binary.
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
        print(f"[r38_fiveacc] edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols
                 and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r38_fiveacc] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), "
          f"missing cols={miss or 0}, non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r38_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
