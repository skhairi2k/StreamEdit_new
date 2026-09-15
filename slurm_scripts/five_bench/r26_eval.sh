#!/bin/bash
# R26 eval -- the 16-metric FiVE-Bench table for the 9 (tau_bg, tau_fg) spatial-tau arms,
# one array task per arm. Restricted to the 22 cases.json pairs, read across all 6
# edit{T}_FiVE.json annotation files.
#
# ARMS must match r26_infer.sh EXACTLY, same order: tau_bg {0,1,2} x tau_fg {2,6,10},
# then tau_fg=50 at each tau_bg (tasks 9-11, added 2026-08-28), then the 40-arm
# extension to tau_bg in {0,1,2,3,4,6,8,10,20,50} x tau_fg in {2,3,4,6,8,10,20,50}
# constrained to tau_bg <= tau_fg (tasks 12-51, added 2026-09-01),
# bg-major. Task 6 is (2,2), the degenerate control every delta is read against.
#
# CSV stems are r26_taubg{X}_taufg{Y}_vp. evaluate.py derives
# evaluation/csv/{stem}_avg.csv itself (plus per-edit-type intermediates named
# edit{T}_FiVE_{stem}_frame_stride8*.csv), so a stem ending in _avg would yield
# {stem}_avg_avg.csv.
#
# --cases_json restricts scoring to the same 22 pairs the arms rendered. Without it,
# evaluate.py would walk all 419 pairs of the annotation files, find 397 missing, and
# either crash or silently average over whatever it did find.
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is the R20/R21 metric-crash fix: the
# 16-metric pass fragments the allocator badly enough to OOM on long clips without it.
#
# Upstream evaluate.py SWALLOWS per-metric exceptions and still exits 0, so a clean exit
# code is not evidence the CSV is clean -- any `Error:` line means a metric was dropped
# and that row's later columns are shifted. The log is teed and grepped below, exactly as
# r21_eval.sh does.
#
# --time=06:00:00: 22 clips x 16 metrics. R21's full-bench 419-pair evals ran ~10h; this
# is ~5% of that, so 6h is very wide margin.
# ⚠️ THE METRIC LIST MUST BE PASSED AS --metrics ON THE COMMAND LINE. The `metrics:` key
# in config.yaml is VESTIGIAL -- evaluate.py:200 reads `metrics = args.metrics`, i.e. the
# argparse flag whose hardcoded default (evaluate.py:619-633) includes five_acc and
# motion_fidelity. Job 961342 was submitted believing a reduced config.yaml would take
# effect; it did not, Qwen2.5-VL + CoTracker both loaded, and motion_fidelity_score OOMed
# on a 40GB A100 (2.52 GiB requested, 37.15 GiB already resident) followed by 14 niqe
# failures. Editing a config file does NOT change which metrics run.
#SBATCH --job-name=r26_eval
#
# L40S RETARGET (2026-08-27): moved off H100, which is unusable -- both nodes draining
# and fully allocated behind another user's 13-task array. Passes an explicit --metrics list that
# drops five_acc (Qwen2.5-VL-7B) and motion_fidelity_score{,_edit_part} (CoTracker) --
# the two models whose combined residency caused R20's L40S GPU exhaustion. The
# remaining 10 metrics fit in 46 GB. ⚠️ motion_fidelity is the only TEMPORAL metric, and
# for an arm that releases the source anchor faster than Eq.4 it is the one most likely
# to catch a preservation blowout -- results from this config are provisional on the
# temporal axis and must be re-scored on H100 with the full config before publication.
# --exclude=node52: that node advertises gpu:8 but exposes no device to batch jobs
# (nvidia-smi -L -> "No devices found."), which killed two r25_infer submissions.
# A100 ADDED alongside L40S (2026-08-27): 11 nodes vs L40S's 5, so the job cycles in
# sooner. R21 rejected A100 as a fallback because "a 40GB A100 would be worse than
# the L40S that already failed" -- that reasoning applied to the FULL 13-metric set
# with Qwen2.5-VL + CoTracker resident. With those dropped via the explicit --metrics
# list below the ceiling is far lower, so 40GB is ample and the objection no longer
# binds -- confirmed in practice: job 961490 scored cleanly on an A100-PCIE-40GB. GPU VRAM
# is still not exposed via scontrol here, so this is inference, not measurement --
# but an A100 OOM would be caught: the job greps 'out of memory' and fails.
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-51
#SBATCH --output=logs/r26_eval_%A_%a.out
#SBATCH --error=logs/r26_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau
CASES="$REPO/evaluation/cases.json"

# Same order as r26_infer.sh. Do not reorder one without the other.
ARMS=("0 2" "0 6" "0 10"
      "1 2" "1 6" "1 10"
      "2 2" "2 6" "2 10"
      "0 50" "1 50" "2 50"
      "0 3" "0 4" "0 8" "0 20"
      "1 3" "1 4" "1 8" "1 20"
      "2 3" "2 4" "2 8" "2 20"
      "3 3" "3 4" "3 6" "3 8" "3 10" "3 20" "3 50"
      "4 4" "4 6" "4 8" "4 10" "4 20" "4 50"
      "6 6" "6 8" "6 10" "6 20" "6 50"
      "8 8" "8 10" "8 20" "8 50"
      "10 10" "10 20" "10 50"
      "20 20" "20 50"
      "50 50")

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 51 ]; then
  echo "[r26_eval] bad array id ${TID} (expected 0-51)"; exit 1
fi
read -r BG FG <<< "${ARMS[$TID]}"
METHOD="taubg${BG}_taufg${FG}_vp"
STEM="r26_${METHOD}"
DIR="$OUT_ROOT/$METHOD"

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

echo "[r26_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r26_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${METHOD} stem=${STEM}"
echo "[r26_eval] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

if [ ! -d "$DIR" ]; then
  echo "[r26_eval] FAILED ${STEM} -- arm dir does not exist: $DIR"
  exit 1
fi
# Count REAL pairs only: evaluate.py writes a sibling {video}_resize dir next to every
# video it scores, so a bare `ls` over an already-scored dir over-counts.
NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r26_eval] real frame dirs in $DIR: ${NDIRS}  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r26_eval] FAILED ${STEM} -- expected 22 real pairs, found ${NDIRS}."
  echo "[r26_eval]   Scoring a short arm would still write a full-looking table. Stop."
  exit 1
fi

METRICS_LOG="logs/r26_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r26_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r26_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  exit 1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r26_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  exit 1
fi

echo "[r26_eval] OK ${STEM} -> evaluation/csv/${STEM}_avg.csv"
exit 0
