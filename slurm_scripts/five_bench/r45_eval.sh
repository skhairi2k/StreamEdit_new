#!/bin/bash
# R45 eval -- the 9-metric FiVE-Bench table for every R45 method, one array task per row of
# `python evaluation/r45_select.py eval --table` (sweep N=15 x8, sweep matched-N x8, the
# selected routed combos), over the WHOLE 419-pair bench. r38_eval.sh otherwise.
#
#   sbatch --array=$(python evaluation/r45_select.py eval) slurm_scripts/five_bench/r45_eval.sh
#   sbatch --array=0-7 slurm_scripts/five_bench/r45_eval.sh      # only the N=15 sweep rows
#
# CSV stem: r45_{method} for the sweeps (r45_ll_rho2_n15_vp), the method itself for the
# routed arms (already r45_ll_...). Frame root from the table: {root}/edit{T}/{video}/ for
# both run_fivebench (sweeps) and r31_stage3 step14 (arms) -- `--tgt_layout edit_video`.
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE (config.yaml's key is vestigial; the
# argparse default includes five_acc + motion_fidelity, which exhaust an L40S). Same nine
# as r26/r31/r35/r38_eval.sh. five_acc and CLIP-D run in r45_fiveacc.sh / r45_clipd.sh.
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0: the log is grepped for
# `Error:` and the per-edit-type row counts are checked.
# The score step reads the per-edit-type CSVs (plain per-clip mean), never {stem}_avg.csv.
#SBATCH --job-name=r45_eval
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r45_eval_%A_%a.out
#SBATCH --error=logs/r45_eval_%A_%a.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

if [ -z "$TID" ]; then echo "[r45_eval] no SLURM_ARRAY_TASK_ID"; exit 1; fi
python evaluation/r45_select.py check "$TID" eval || exit 1
ROW=$(python evaluation/r45_select.py eval --table | awk -F'\t' -v i="$TID" '$1==i')
METHOD=$(echo "$ROW" | cut -f2)
DIR=$(echo "$ROW" | cut -f3)
case "$METHOD" in r45_*) STEM="$METHOD" ;; *) STEM="r45_${METHOD}" ;; esac

echo "[r45_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r45_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} stem=${STEM}"
echo "[r45_eval] scoring $DIR  (full bench, no --cases_json)"

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
  echo "[r45_eval] FAILED ${STEM} -- render dir does not exist: $DIR"; exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r45_eval] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r45_eval] FAILED ${STEM} -- render incomplete, run /run-step R45 wait-renders first"; exit 1
fi
echo "[r45_eval] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r45_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r45_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"
FAILED=0
if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r45_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"; FAILED=1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r45_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"; FAILED=1
fi
# ---- post-run guard: every clip scored, per edit type (header row excluded).
for T in 1 2 3 4 5 6; do
  CSV="evaluation/csv/edit${T}_FiVE_${STEM}_frame_stride8.csv"
  N=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  STATUS="ok"
  [ "$N" -ne "${EXPECT[$T]}" ] && { STATUS="MISMATCH (expect ${EXPECT[$T]})"; FAILED=1; }
  echo "[r45_eval] ${STEM} edit${T}: ${N} scored rows -- ${STATUS}"
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r45_eval] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
