#!/bin/bash
# R44 -- FiVE-Acc alone for R44's TEXT-ONLY renders, one array task per arm, over the WHOLE
# 419-pair bench, with evaluate.py --metrics five_acc ONLY. Clone of r36_fiveacc.sh
# (sweep) / r38_fiveacc.sh (arms).
#
# USAGE (same modes and index maps as r44_eval.sh -- do not reorder one without the other):
#   sbatch --array=0-7 slurm_scripts/five_bench/r44_fiveacc.sh sweep N    # N = 15 | 20 | 30
#       task i -> rho RHOS[i]; dir r44_rho_sweep/r44_n{N}_rho{R}_novp
#   sbatch --array=0-<K-1> slurm_scripts/five_bench/r44_fiveacc.sh arms
#       task i -> combo i of evaluation/r44_selected_combos.txt; dir r44_arms/r44_{combo}/step14
# STEM = r44_fiveacc_n{N}_rho{R}_novp | r44_fiveacc_{combo}.
#
# WHY FIVE_ACC ALONE: R20's GPU exhaustion was Qwen2.5-VL-7B (five_acc) co-resident with
# CoTracker and the rest of the metric table on one GPU. r44_eval.sh drops five_acc; this
# job runs it alone, so nothing co-resides and no H100 is needed.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, and a raised metric
# DROPS its column. The log is teed and grepped, and the per-edit-type CSVs are checked
# for the full row count AND for the five_acc columns being present and NaN-free.
#
# --tmp_resize (R36): resized frames go to a per-clip temp dir deleted after the clip is
# scored, not a persistent {video}_resize sibling -- keeps ~2 GB/arm off the home quota.
# --time=04:00:00: R33's five_acc-only task scored 22 clips in 1 min 50 s end to end
# (logs/r33_fiveacc_1004302_16.metrics.log); 419 clips is well under an hour.
#SBATCH --job-name=r44_fiveacc
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r44_fiveacc_%A_%a.out
#SBATCH --error=logs/r44_fiveacc_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench

RHOS=(2 3 4 6 8 10 20 50)
MODE=$1
TID=${SLURM_ARRAY_TASK_ID}
case "$MODE" in
  sweep)
    N=$2
    case "$N" in 15|20|30) ;; *) echo "[r44_fiveacc] sweep needs N in {15,20,30}, got '${N}'"; exit 1 ;; esac
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
      echo "[r44_fiveacc] bad array id '${TID}' (expected 0-7)"; exit 1
    fi
    NAME="r44_n${N}_rho${RHOS[$TID]}_novp"
    DIR="$FIVE_ROOT/r44_rho_sweep/$NAME"
    ;;
  arms)
    source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
    r44_load || exit 1
    K=${#R44_COMBOS[@]}
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "$K" ]; then
      echo "[r44_fiveacc] bad array id '${TID}' (expected 0-$((K-1)) for ${K} selected combos)"; exit 1
    fi
    NAME="r44_${R44_COMBOS[$TID]}"
    DIR="$FIVE_ROOT/r44_arms/$NAME/step14"
    ;;
  *) echo "[r44_fiveacc] usage: r44_fiveacc.sh sweep N | arms"; exit 1 ;;
esac
STEM="r44_fiveacc_${NAME#r44_}"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTHONUNBUFFERED=1

echo "[r44_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r44_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} arm=${NAME} stem=${STEM}"
echo "[r44_fiveacc] scoring $DIR  (full bench, five_acc only)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the arm's render must be complete, per edit type (real pairs only).
if [ ! -d "$DIR" ]; then
  echo "[r44_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  NP=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$NP" -ne "${EXPECT[$T]}" ] && { echo "[r44_fiveacc] edit${T}: ${NP} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r44_fiveacc] FAILED ${STEM} -- render incomplete, run the matching /run-step R44 wait-* step first"
  exit 1
fi
echo "[r44_fiveacc] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r44_fiveacc_${MODE}_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r44_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ]; then
  echo "[r44_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

# ---- post-run guard: full row count per edit type, both five_acc columns present and
# binary (a dropped or NaN column would silently bias the FiVE-Acc axis).
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
        print(f"[r44_fiveacc] edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols
                 and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r44_fiveacc] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), "
          f"missing cols={miss or 0}, non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r44_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
