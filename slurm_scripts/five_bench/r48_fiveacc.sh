#!/bin/bash
# R48 FiVE-Acc: the VLM question-answering metric (five_acc) for one render directory ("unit") per array task.
#
# Usage: sbatch slurm_scripts/five_bench/r48_fiveacc.sh subset                           # 17 units (--array 0-16 default)
#        sbatch --array=0-$((N-1)) slurm_scripts/five_bench/r48_fiveacc.sh full          # N = `r48_units.py --mode full --count`
# Same units, index order, frame stride and Wan-Edit frame handling as r48_eval.sh. Output: evaluation/csv/r48_fiveacc_{unit}.csv
# (+ edit{T}_FiVE_r48_fiveacc_{unit}_frame_stride8.csv). Guard: every row has binary five_acc_yes_no / five_acc_multi_choice cells.
#SBATCH --job-name=r48_fiveacc
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-16
#SBATCH --output=logs/r48_fiveacc_%A_%a.out
#SBATCH --error=logs/r48_fiveacc_%A_%a.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json; CASES_ARG=(--cases_json "$CASES") ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; CASES_ARG=() ;;
  *) echo "[r48_fiveacc] usage: r48_fiveacc.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
PY_FB=~/anaconda3/envs/five-bench/bin/python
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

TID=${SLURM_ARRAY_TASK_ID}
UNIT=$($PY_FB evaluation/r48_units.py --mode "$MODE" --index "$TID" 2>/dev/null) || { echo "[r48_fiveacc] no unit for array id '${TID}' (mode ${MODE}) -- nothing to do"; exit 0; }
IFS=$'\t' read -r BASE DIR HOST CONFIG <<< "$UNIT"
STEM="r48_fiveacc_${BASE}"
read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
echo "[r48_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} unit=${HOST}/${CONFIG} stem=${STEM} expect per type: ${EXPECT[*]}"
echo "[r48_fiveacc] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2
echo "[r48_fiveacc] scoring $DIR"

ANNOTATIONS=()
for T in 1 2 3 4 5 6; do ANNOTATIONS+=("$DATA_ROOT/edit_prompt/edit${T}_FiVE.json"); done

[ -d "$DIR" ] || { echo "[r48_fiveacc] FAILED ${STEM} -- render dir does not exist: $DIR"; exit 1; }
MISSING=$($PY_FB evaluation/r48_units.py --mode "$MODE" --missing "$DIR" --cases "$CASES")
if [ -n "$MISSING" ]; then
  echo "[r48_fiveacc] FAILED ${STEM} -- $(echo "$MISSING" | wc -l) case clip(s) not rendered under $DIR, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ') -- run the render step first"
  exit 1
fi
echo "[r48_fiveacc] preflight OK: every clip of ${CASES} is rendered (the scorers filter by case list, so a full render dir can be scored on a subset)"

METRICS_LOG="logs/r48_fiveacc_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
$PY_FB evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml --metrics five_acc \
  --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" --tgt_layout edit_video --tmp_resize "${CASES_ARG[@]}" \
  --result_path "evaluation/csv/${STEM}.csv" 2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}
NERR=$(grep -c 'Error:' "$METRICS_LOG"); NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r48_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
{ [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; } && { echo "[r48_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"; FAILED=1; }
$PY_FB - "$STEM" "${EXPECT[@]}" <<'PY' || FAILED=1
import csv, sys
stem, expect = sys.argv[1], dict(zip(range(1, 7), map(int, sys.argv[2:8])))
need = ("five_acc_yes_no", "five_acc_multi_choice")
bad = 0
for t, n_exp in expect.items():
    if n_exp == 0:
        continue
    path = f"evaluation/csv/edit{t}_FiVE_{stem}_frame_stride8.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"[r48_fiveacc] edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r48_fiveacc] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), missing cols={miss or 0}, "
          f"non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
[ "$FAILED" -ne 0 ] && exit 1
echo "[r48_fiveacc] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
