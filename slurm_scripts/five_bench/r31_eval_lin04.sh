#!/bin/bash
# R31 eval, EXPLORATORY VARIANT -- identical 9-metric harness to r31_eval.sh, scoring
# r31_stage3_lin04.sh's renders (the linear_threshold tau mapping) instead of the
# calibrated budget_linear ones. Forked rather than parameterizing r31_eval.sh in place
# so the production script (still gating `verdict`) stays untouched.
#
# ⚠️ THE METRIC LIST IS BYTE-IDENTICAL TO r26_eval.sh / r31_eval.sh AND MUST STAY THAT
# WAY -- see r31_eval.sh's header for the full rationale (five_acc / motion_fidelity
# dropped for VRAM, --metrics must be on the command line not config.yaml, evaluate.py
# swallows per-metric exceptions).
#
# STEM here is r31_{arm}_lin04 (was r31_{arm}), so nothing under evaluation/csv/ from
# the calibrated run is overwritten.
#
# RECREATED 2026-09-15 after this file (along with r31_rho_map.py, r31_stage3_grids.py,
# r31_score.py, r31_stage3.sh, r31_stage3_lin04.sh, the R31 plan file, and daily.md) was
# found deleted from disk by an external process -- root cause unknown, no destructive
# command in any tracked shell history. Recreated verbatim from conversation context;
# the underlying render/eval DATA (r31_arms_lin04/, r31_rho_lin_t0.4/) was untouched.
#SBATCH --job-name=r31_eval_lin04
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r31_eval_lin04_%A_%a.out
#SBATCH --error=logs/r31_eval_lin04_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r31_arms_lin04   # NEW, not r31_arms
CASES="$REPO/evaluation/cases.json"

# The four divergence arms, same order as r31_stage2.sh / r31_stage3.sh / r31_stage3_lin04.sh.
ARMS=(lpips dino_patch normals latent)

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r31_eval_lin04] bad array id ${TID} (expected 0-3)"; exit 1
fi
ARM=${ARMS[$TID]}
STEM="r31_${ARM}_lin04"
DIR="$OUT_ROOT/r31_${ARM}_lin04/step14"

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r31_eval_lin04] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r31_eval_lin04] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} stem=${STEM}"
echo "[r31_eval_lin04] scoring $DIR"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

if [ ! -d "$DIR" ]; then
  echo "[r31_eval_lin04] FAILED ${STEM} -- arm dir does not exist: $DIR"
  exit 1
fi
# Count REAL pairs only: evaluate.py writes a sibling {video}_resize dir next to every
# video it scores, so a bare `ls` over an already-scored dir over-counts.
NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r31_eval_lin04] real frame dirs in $DIR: ${NDIRS}  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r31_eval_lin04] FAILED ${STEM} -- expected 22 real pairs, found ${NDIRS}."
  exit 1
fi

METRICS_LOG="logs/r31_eval_lin04_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
  --cases_json "$CASES" \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r31_eval_lin04] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r31_eval_lin04] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  exit 1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r31_eval_lin04] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  exit 1
fi

echo "[r31_eval_lin04] OK ${STEM} -> evaluation/csv/${STEM}_avg.csv"
exit 0
