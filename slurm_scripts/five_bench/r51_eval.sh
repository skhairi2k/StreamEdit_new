#!/bin/bash
# R51 eval: ONE unit of `r51_units.py --group GROUP` per array task, on the 419 pairs: FiVE harness metrics, then FiVE-Acc, then
# CLIP-D of that unit. Same evaluate.py / r33_clip_directional.py calls and row checks as r49_eval_all.sh full mode.
#
# Usage: sbatch --array=0-3 slurm_scripts/five_bench/r51_eval.sh se    # lab L40S: the 4 StreamEdit arms   (r51_se_chain.sh)
#        sbatch --array=0   slurm_scripts/five_bench/r51_eval.sh wan   # Jean Zay: wan_nosupp_s5           (r51_wan_chain.sh)
# Output: evaluation/csv/edit{T}_FiVE_r51_{base}_frame_stride8.csv, ..._r51_fiveacc_{base}_..., r51_clip_directional_{base}.csv
#SBATCH --job-name=r51_eval
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r51_eval_%A_%a.out
#SBATCH --error=logs/r51_eval_%A_%a.err

GROUP=${1:-}
case "$GROUP" in se|wan) ;; *) echo "[r51_eval] usage: sbatch --array=I r51_eval.sh se|wan"; exit 1 ;; esac
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
CASES=evaluation/r7_anchor_manifest.json
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
JOB=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}

TID=${SLURM_ARRAY_TASK_ID:-}
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "[r51_eval] needs an array task id (sbatch --array=I)"; exit 1; }
UNIT=$($PY_FB evaluation/r51_units.py --group "$GROUP" --index "$TID") || { echo "[r51_eval] no unit ${TID} in group ${GROUP}"; exit 1; }
IFS=$'\t' read -r BASE DIR HOST CONFIG <<< "$UNIT"

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
ANNOTATIONS=()
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -gt 0 ] && ANNOTATIONS+=("$DATA_ROOT/edit_prompt/edit${T}_FiVE.json")
done
echo "[r51_eval] job=${JOB} group=${GROUP} unit=${BASE} (${HOST}/${CONFIG}) dir=${DIR} expect per type: ${EXPECT[*]}"
echo "[r51_eval] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# per-type row counts of one result stem; for five_acc also the binary columns (as r49_eval_all.sh)
check_rows() {  # $1 = stem, $2 = kind (harness|fiveacc)
  $PY_FB - "$1" "$2" "${EXPECT[@]}" <<'PY'
import csv, sys
stem, kind, expect = sys.argv[1], sys.argv[2], dict(zip(range(1, 7), map(int, sys.argv[3:9])))
need = ("five_acc_yes_no", "five_acc_multi_choice") if kind == "fiveacc" else ()
bad = 0
for t, n_exp in expect.items():
    if n_exp == 0:
        continue
    path = f"evaluation/csv/edit{t}_FiVE_{stem}_frame_stride8.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"[r51_eval] {stem} edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r51_eval] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), missing cols={miss or 0}, "
          f"non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
}

# Completeness preflight, Wan-Edit units only (r48_units counts frames the Wan-Edit way; StreamEdit renders guard themselves).
if [ "$HOST" != se ]; then
  MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$DIR" --cases "$CASES")
  if [ -n "$MISSING" ]; then
    echo "[r51_eval] FAILED ${BASE} -- $(echo "$MISSING" | wc -l) case clip(s) not rendered under $DIR, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"
    echo "[r51_eval] unit failures: 1"; exit 1
  fi
fi

UNIT_FAIL=0
for KIND in harness fiveacc; do
  if [ "$KIND" = harness ]; then
    STEM="r51_${BASE}"
    METRICS=(structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part
             clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part niqe_target_image)
  else
    STEM="r51_fiveacc_${BASE}"; METRICS=(five_acc)
  fi
  LOG="logs/r51_eval_${JOB}_${BASE}_${KIND}.metrics.log"
  $PY_FB evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml --metrics "${METRICS[@]}" \
    --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --tgt_methods "$DIR" --tgt_layout edit_video --tmp_resize \
    --result_path "evaluation/csv/${STEM}.csv" > "$LOG" 2>&1
  RC=$?
  NERR=$(grep -c 'Error:' "$LOG"); NOOM=$(grep -c 'out of memory' "$LOG")
  echo "[r51_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM} (log ${LOG})"
  { [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; } && UNIT_FAIL=1
  check_rows "$STEM" "$KIND" || UNIT_FAIL=1
done

# ---- CLIP-D of this unit
NAME="r51_${BASE}"
OUT="evaluation/csv/r51_clip_directional_${BASE}.csv"
RUNLOG="logs/r51_eval_${JOB}_${BASE}_clipd.run.log"
$PY_FB evaluation/r33_clip_directional.py --fullbench --methods "${NAME}=${DIR}" --data_root "$DATA_ROOT" \
  --frame_stride 8 -o "$OUT" > "$RUNLOG" 2>&1
RC=$?
echo "[r51_eval] clipd exit=${RC} -> ${OUT}"
if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  UNIT_FAIL=1
else
  $PY_FB - "$OUT" "$NAME" "${EXPECT[@]}" <<'PY' || UNIT_FAIL=1
import csv, sys
from collections import Counter
out, name = sys.argv[1], sys.argv[2]
expect = {t: n for t, n in zip(range(1, 7), map(int, sys.argv[3:9])) if n}
got = Counter(int(r["edit_type"]) for r in csv.DictReader(open(out)) if r["method"] == name)
ok = dict(got) == expect
print(f"[r51_eval] clipd {name}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
fi
echo "[r51_eval] unit failures: ${UNIT_FAIL}"
exit "$UNIT_FAIL"
