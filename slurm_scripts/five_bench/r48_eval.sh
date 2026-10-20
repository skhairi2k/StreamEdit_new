#!/bin/bash
# R48 eval: the FiVE harness metrics for one render directory ("unit") per array task.
#
# Usage: sbatch slurm_scripts/five_bench/r48_eval.sh subset                              # 17 units (--array 0-16 default)
#        sbatch --array=0-$((N-1)) slurm_scripts/five_bench/r48_eval.sh full             # N = `r48_units.py --mode full --count`
# Units (evaluation/r48_units.py; index -> unit is fixed): StreamEdit arms, the 9 Wan-Edit baselines, then the Wan-Edit arms.
# Metrics: as r38_eval.sh (structure distance, PSNR/LPIPS/MSE/SSIM on the unedited part, CLIP source/target/edit-part, NIQE),
# frame stride 8. Wan-Edit renders are 41 frames vs the 69-frame source folder: evaluate.py pairs positionally and stops at the
# shorter side, so the first 6 strided frames are scored (the Wan-Edit official numbers were produced the same way).
# Subset runs add --cases_json evaluation/r48_cases.json. Output: evaluation/csv/r48_{unit}.csv (+ _avg, edit{T}_FiVE_..._frame_stride8.csv)
#SBATCH --job-name=r48_eval
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-16
#SBATCH --output=logs/r48_eval_%A_%a.out
#SBATCH --error=logs/r48_eval_%A_%a.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json; CASES_ARG=(--cases_json "$CASES") ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; CASES_ARG=() ;;
  *) echo "[r48_eval] usage: r48_eval.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
PY_FB=~/anaconda3/envs/five-bench/bin/python
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

TID=${SLURM_ARRAY_TASK_ID}
UNIT=$($PY_FB evaluation/r48_units.py --mode "$MODE" --index "$TID" 2>/dev/null) || { echo "[r48_eval] no unit for array id '${TID}' (mode ${MODE}) -- nothing to do"; exit 0; }
IFS=$'\t' read -r BASE DIR HOST CONFIG <<< "$UNIT"
STEM="r48_${BASE}"
read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
echo "[r48_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} unit=${HOST}/${CONFIG} stem=${STEM} expect per type: ${EXPECT[*]}"
echo "[r48_eval] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2
echo "[r48_eval] scoring $DIR"

ANNOTATIONS=()
for T in 1 2 3 4 5 6; do ANNOTATIONS+=("$DATA_ROOT/edit_prompt/edit${T}_FiVE.json"); done

[ -d "$DIR" ] || { echo "[r48_eval] FAILED ${STEM} -- render dir does not exist: $DIR"; exit 1; }
MISSING=$($PY_FB evaluation/r48_units.py --mode "$MODE" --missing "$DIR" --cases "$CASES")
if [ -n "$MISSING" ]; then
  echo "[r48_eval] FAILED ${STEM} -- $(echo "$MISSING" | wc -l) case clip(s) not rendered under $DIR, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ') -- run the render step first"
  exit 1
fi
echo "[r48_eval] preflight OK: every clip of ${CASES} is rendered (the scorers filter by case list, so a full render dir can be scored on a subset)"

METRICS_LOG="logs/r48_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
$PY_FB evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part \
            clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part \
            niqe_target_image \
  --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" --tgt_layout edit_video --tmp_resize "${CASES_ARG[@]}" \
  --result_path "evaluation/csv/${STEM}.csv" 2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}
NERR=$(grep -c 'Error:' "$METRICS_LOG"); NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r48_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
[ -f "evaluation/csv/${STEM}_avg.csv" ] || { echo "[r48_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"; FAILED=1; }
{ [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; } && { echo "[r48_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"; FAILED=1; }
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  CSV="evaluation/csv/edit${T}_FiVE_${STEM}_frame_stride8.csv"
  N=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  STATUS="ok"; [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { STATUS="MISMATCH (expect ${EXPECT[$((T - 1))]})"; FAILED=1; }
  echo "[r48_eval] ${STEM} edit${T}: ${N} scored rows -- ${STATUS}"
done
[ "$FAILED" -ne 0 ] && exit 1
echo "[r48_eval] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
