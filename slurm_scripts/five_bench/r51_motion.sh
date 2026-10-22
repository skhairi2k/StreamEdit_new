#!/bin/bash
# R51 motion fidelity: ONE unit of `r51_units.py --group motion` per array task, on the 419 pairs -- FiVE's
# motion_fidelity_score and motion_fidelity_score_edit_part (CoTracker) ALONE: no Qwen2.5-VL in the same process, which is
# what exhausted the L40S when both ran together (R20/R26).
#
# Usage: sbatch --array=0-4 slurm_scripts/five_bench/r51_motion.sh                         A1..A4 + A0 (r51_se_chain.sh, afterok render)
#        sbatch --array=5 --nodelist=node39 slurm_scripts/five_bench/r51_motion.sh         official rho = 2 (r36_rho2_vp lives on /store)
# Output: evaluation/csv/edit{T}_FiVE_r51_motion_{base}_frame_stride8.csv
# Known degeneracy: motion_fidelity_score_edit_part raises on an empty edit mask (~1 clip in 53, ~8 over the bench; R2/R14) and
# evaluate.py writes NaN for that cell and logs an 'Error:' line. Those cells are COUNTED and reported; the unit fails only if
# motion_fidelity_score itself has a NaN, a row count is off, or edit_part has more than MAX_NAN_EDIT NaN cells in total.
#SBATCH --job-name=r51_motion
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r51_motion_%A_%a.out
#SBATCH --error=logs/r51_motion_%A_%a.err

MAX_NAN_EDIT=20
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
CASES=evaluation/r7_anchor_manifest.json
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
JOB=${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}

TID=${SLURM_ARRAY_TASK_ID:-}
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "[r51_motion] needs an array task id (sbatch --array=I)"; exit 1; }
UNIT=$($PY_FB evaluation/r51_units.py --group motion --index "$TID") || { echo "[r51_motion] no unit ${TID} in group motion"; exit 1; }
IFS=$'\t' read -r BASE DIR HOST CONFIG <<< "$UNIT"
echo "[r51_motion] job=${JOB} unit=${BASE} (${HOST}/${CONFIG}) dir=${DIR} node=$(hostname)"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: the render dir must be readable here (/store is mounted on some nodes only)
if [ ! -d "$DIR" ] || [ -z "$(ls "$DIR" 2>/dev/null | head -1)" ]; then
  echo "[r51_motion] FATAL: ${DIR} not readable on $(hostname) -- /store not mounted here, or the render is missing"
  echo "[r51_motion] unit failures: 1"; exit 1
fi

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

STEM="r51_motion_${BASE}"
LOG="logs/r51_motion_${JOB}_${BASE}.metrics.log"
$PY_FB evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml --metrics motion_fidelity_score motion_fidelity_score_edit_part \
  --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" --tgt_layout edit_video --tmp_resize \
  --result_path "evaluation/csv/${STEM}.csv" > "$LOG" 2>&1
RC=$?
NERR=$(grep -c 'Error:' "$LOG"); NOOM=$(grep -c 'out of memory' "$LOG")
echo "[r51_motion] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM} (log ${LOG})"
UNIT_FAIL=0
{ [ "$RC" -ne 0 ] || [ "$NOOM" -ne 0 ]; } && UNIT_FAIL=1

$PY_FB - "$STEM" "$MAX_NAN_EDIT" "${EXPECT[@]}" <<'PY' || UNIT_FAIL=1
import csv, math, sys
stem, max_nan_edit = sys.argv[1], int(sys.argv[2])
expect = dict(zip(range(1, 7), map(int, sys.argv[3:9])))
bad, nan_edit = 0, 0
for t, n_exp in expect.items():
    if n_exp == 0:
        continue
    path = f"evaluation/csv/edit{t}_FiVE_{stem}_frame_stride8.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"[r51_motion] {stem} edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in ("motion_fidelity_score", "motion_fidelity_score_edit_part") if c not in cols]
    def n_nan(c):
        return sum(1 for r in rows if not math.isfinite(float(r[cols[c]]))) if c in cols else 0
    nan_full, nan_e = n_nan("motion_fidelity_score"), n_nan("motion_fidelity_score_edit_part")
    nan_edit += nan_e
    ok = len(rows) == n_exp and not miss and nan_full == 0
    bad += not ok
    print(f"[r51_motion] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), missing cols={miss or 0}, "
          f"NaN full={nan_full}, NaN edit_part={nan_e} -- {'ok' if ok else 'MISMATCH'}")
print(f"[r51_motion] {stem}: NaN edit_part cells in total {nan_edit} (allowed <= {max_nan_edit}: empty-edit-mask clips)")
sys.exit(1 if bad or nan_edit > max_nan_edit else 0)
PY
echo "[r51_motion] unit failures: ${UNIT_FAIL}"
exit "$UNIT_FAIL"
