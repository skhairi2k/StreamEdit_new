#!/bin/bash
# R31 eval -- the 9-metric FiVE-Bench table for the four spatial-divergence arms, one
# array task per arm, scored on each arm's FINAL stage-3 render (step14).
#
# WHY THIS EXISTS. Not in the user's stated outputs; added because R31 holds
# sampler/anchors/seed/case-set identical to R26, so the resulting
# (clip_similarity_target_image, lpips_unedit_part) points drop STRAIGHT onto
# r26_tradeoff_figure.py's stored 52-arm cloud with no recalibration -- and a P0 task
# whose only artefacts are videos is very hard to read. Cut this step if compute is
# tight; it gates none of the user-requested outputs.
#
# ⚠️ THE METRIC LIST IS BYTE-IDENTICAL TO r26_eval.sh AND MUST STAY THAT WAY. The
# overlay onto R26's panel is only meaningful if both sides computed the same columns
# the same way. The nine below drop five_acc (Qwen2.5-VL-7B) and
# motion_fidelity_score{,_edit_part} (CoTracker) -- R20's L40S GPU-exhaustion pair.
# ⚠️ motion_fidelity is the only TEMPORAL metric, and R31 is a VIDEO method whose whole
# risk is temporal: an arm that releases the source anchor per-token could flicker
# without moving any of these nine numbers. Results here are provisional on the temporal
# axis -- the qualitative grids (r31_stage3_grids.py --which arms) are the real check.
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE. The `metrics:` key in config.yaml is
# VESTIGIAL: evaluate.py:200 reads `metrics = args.metrics`, i.e. the argparse flag whose
# hardcoded default (evaluate.py:723-737) includes five_acc and motion_fidelity. R26's
# job 961342 was submitted believing a reduced config.yaml would take effect; it did not,
# both models loaded, and motion_fidelity_score OOMed. Editing a config file does NOT
# change which metrics run.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, so a clean exit code
# is NOT evidence the CSV is clean -- any `Error:` line means a metric was dropped and
# that row's later columns are shifted. The log is teed and grepped below, as r21/r26 do.
#
# STEM: evaluate.py derives evaluation/csv/{stem}_avg.csv itself (plus per-edit-type
# intermediates), so a stem ending in _avg would yield {stem}_avg_avg.csv. Stems here are
# r31_{arm}; the verdict step reads the _avg.csv files.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is the R20/R21 metric-crash fix.
#SBATCH --job-name=r31_eval
# L40S,A100 and --exclude=node52 are R26's proven pair: node52 advertises gpu:8 but
# exposes no device to batch jobs (nvidia-smi -L -> "No devices found."). With five_acc
# and motion_fidelity dropped, a 40GB A100 is ample (confirmed by R26 job 961490).
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r31_eval_%A_%a.out
#SBATCH --error=logs/r31_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r31_arms
CASES="$REPO/evaluation/cases.json"

# The four divergence arms, same order as r31_stage2.sh / r31_stage3.sh. `depth` was
# dropped 2026-09-11 (it needs an affine alignment R31 has no mask to fit). Do not
# reorder one script without the others.
ARMS=(lpips dino_patch normals latent)

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r31_eval] bad array id ${TID} (expected 0-3)"; exit 1
fi
ARM=${ARMS[$TID]}
STEM="r31_${ARM}"
# Stage 3 writes {out_root}/{method}/step{jj}/edit{T}/{video}/, and `--tgt_layout
# edit_video` expects exactly that {DIR}/edit{T}/{video}/ shape. step14 is the FINAL
# render of stage 3's 15-step rollout (stage 1's 7 steps are unrelated).
DIR="$OUT_ROOT/r31_${ARM}/step14"

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

echo "[r31_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r31_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} stem=${STEM}"
echo "[r31_eval] scoring $DIR"
echo "[r31_eval] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

if [ ! -d "$DIR" ]; then
  echo "[r31_eval] FAILED ${STEM} -- arm dir does not exist: $DIR"
  exit 1
fi
# Count REAL pairs only: evaluate.py writes a sibling {video}_resize dir next to every
# video it scores, so a bare `ls` over an already-scored dir over-counts.
NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r31_eval] real frame dirs in $DIR: ${NDIRS}  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r31_eval] FAILED ${STEM} -- expected 22 real pairs, found ${NDIRS}."
  echo "[r31_eval]   Scoring a short arm would still write a full-looking table. Stop."
  exit 1
fi

METRICS_LOG="logs/r31_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r31_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r31_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  exit 1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r31_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  exit 1
fi

echo "[r31_eval] OK ${STEM} -> evaluation/csv/${STEM}_avg.csv"
exit 0
