#!/bin/bash
# R41 FiVE-Acc -- the VLM edit-success score for the binary arms, one ARM per array task,
# on each arm's FINAL stage-3 render (step14), whole 419-pair bench.
#
# Submit with the array size from the config:
#   sbatch --array=0-$(bash slurm_scripts/five_bench/r41_config.sh --max_index) \
#          slurm_scripts/five_bench/r41_fiveacc.sh
#
# r35_fiveacc.sh verbatim otherwise: --metrics five_acc only (Qwen2.5-VL), same resources
# and guards. evaluate.py swallows metric exceptions, so the log is grepped for `Error:`
# and every per-type CSV is checked for the row count AND for binary (0/1) yes/no and
# multi-choice cells -- a NaN row would otherwise average in silently.
#SBATCH --job-name=r41_fiveacc
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r41_fiveacc_%A_%a.out
#SBATCH --error=logs/r41_fiveacc_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
# exits here if DRAFT_N is unset; R41_CONFIG is for synthetic tests only (see r41_config.sh)
source "${R41_CONFIG:-slurm_scripts/five_bench/r41_config.sh}"

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "${#ARMS[@]}" ]; then
  echo "[r41_fiveacc] bad array id '${TID}' (expected 0-$(( ${#ARMS[@]} - 1 )))"; exit 1
fi
ARM=${ARMS[$TID]}
STEM="r41_fiveacc_${TAG}_${ARM}"
DIR="$ARMS_ROOT/r41_${TAG}_${ARM}/step14"

mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

echo "[r41_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r41_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} stem=${STEM}"
echo "[r41_fiveacc] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

if [ ! -d "$DIR" ]; then
  echo "[r41_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r41_fiveacc] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r41_fiveacc] FAILED ${STEM} -- render incomplete, run /run-step R41 wait-stage3 first"
  exit 1
fi
echo "[r41_fiveacc] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r41_fiveacc_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r41_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r41_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

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
        print(f"[r41_fiveacc] edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols
                 and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r41_fiveacc] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), "
          f"missing cols={miss or 0}, non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r41_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
