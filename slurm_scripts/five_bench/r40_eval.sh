#!/bin/bash
# R40 eval -- the 9-metric FiVE-Bench table for the two draft-free combos, one array task
# per combo, on each combo's FINAL stage-3 render (step14), whole 419-pair bench.
# Copy of r35_eval.sh: same metrics, flags and guards -- only combos, roots and stems change,
# so the R40 and R35 CSVs are directly comparable.
#
# ARRAY LAYOUT: same combo order as r40_stage3.sh:   0 dino_warp   1 lpips_warp
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE (r31_eval.sh's note): config.yaml's
# `metrics:` key is vestigial, and evaluate.py's default includes five_acc and
# motion_fidelity, which exhaust an L40S. five_acc and CLIP-D are their own steps.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0; the log is teed and
# grepped for `Error:`, and per-edit-type row counts are checked.
#
# NOTE: {stem}_avg.csv is a mean of the six per-type means, not over the 419 clips --
# r40_score.py reads the per-edit-type CSVs instead (plain per-clip mean).
# --tmp_resize: resized frames in a per-clip temp dir (home-quota fix, R36).
#SBATCH --job-name=r40_eval
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/r40_eval_%A_%a.out
#SBATCH --error=logs/r40_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r40_arms

COMBOS=(dino_warp lpips_warp)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 1 ]; then
  echo "[r40_eval] bad array id ${TID} (expected 0-1)"; exit 1
fi
COMBO=${COMBOS[$TID]}
STEM="r40_${COMBO}"
# r40_stage3.sh writes {out_root}/r40_{combo}/step14/edit{T}/{video}/, and
# `--tgt_layout edit_video` expects exactly that {DIR}/edit{T}/{video}/ shape.
DIR="$OUT_ROOT/r40_${COMBO}/step14"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it,
# which crashed the R2 eval on ModuleNotFoundError: torchmetrics).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r40_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r40_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} stem=${STEM}"
echo "[r40_eval] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the combo's render must be complete, per edit type. Count REAL pairs
# only: evaluate.py writes a sibling {video}_resize dir next to every video it scores,
# so a bare count over an already-scored dir over-counts. Hard fail (r21_eval.sh only
# warned): scoring a short arm still writes a full-looking table.
if [ ! -d "$DIR" ]; then
  echo "[r40_eval] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r40_eval] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r40_eval] FAILED ${STEM} -- render incomplete, run /run-step R40 wait-stage3 first"
  exit 1
fi
echo "[r40_eval] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r40_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part \
            mse_unedit_part ssim_unedit_part clip_similarity_source_image \
            clip_similarity_target_image clip_similarity_target_image_edit_part \
            niqe_target_image \
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
echo "[r40_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r40_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  FAILED=1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r40_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

# ---- post-run guard: every clip scored, per edit type (header row excluded).
for T in 1 2 3 4 5 6; do
  CSV="evaluation/csv/edit${T}_FiVE_${STEM}_frame_stride8.csv"
  N=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  STATUS="ok"
  [ "$N" -ne "${EXPECT[$T]}" ] && { STATUS="MISMATCH (expect ${EXPECT[$T]})"; FAILED=1; }
  echo "[r40_eval] ${STEM} edit${T}: ${N} scored rows -- ${STATUS}"
done

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r40_eval] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
