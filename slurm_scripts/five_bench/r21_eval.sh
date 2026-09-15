#!/bin/bash
# R21 evaluation -- 16-metric FiVE scoring of the four full-bench schedule arms
# (cos_third/zero x vp/pvp), the full-bench paper_pvp Eq.4 reference, and R7
# (r7_visual_prompting) as the vp Eq.4 reference.
#
# NO --cases_json anywhere: every method here is the whole FiVE-Bench (419 pairs
# over all 6 edit types), so there is nothing to restrict to. This is the
# opposite of r20_eval.sh, which HAD to pass --cases_json because its arms were
# a 22-clip subset while its references were full-bench.
#
# ARRAY, not one serial job. R20 measured ~16m44s per method for 22 clips
# (~45 s/clip); at 419 clips that is ~5.2 h per method, so six methods in series
# is ~31 h -- past the 24 h partition ceiling, i.e. the serial form would time
# out mid-way and leave a partial CSV set. One method per array task keeps each
# task at ~5 h. Scoring is fully independent per method (own result CSV, own
# niqe txt, own _resize dirs under its own output root), so the array produces
# byte-identical numbers to the serial form.
#
# H100, not L40S. In R20 (job 907650, L40S 44 GB) motion_fidelity_score CUDA
# -OOM'd 144x and niqe failed 140x. Upstream's `except: continue` appends no
# placeholder, so a raised metric DROPS a column and shifts every column after
# it -- silently corrupting five_acc. Both failure classes are the same root
# cause, GPU exhaustion in the parent process: the errors are absent from edit1
# (scored first, memory free) and concentrated in edit2/5/6 once Qwen2.5-VL +
# CoTracker are resident (parent at ~44.35/44.39 GiB). The niqe failures are
# that same exhaustion one level down -- calculate_NIQE shells out to
# inference_iqa.py, and the child could not init CUDA, so it wrote no txt and
# average_niqe_from_txt hit FileNotFoundError. Verified 2026-08-11: the exact
# failing command (edit2/0075_A_bicycle_resize/00080.png) runs clean standalone
# on a free GPU and returns niqe 3.044 -- so the IQA path in config.yaml is
# already correct and needs no change; it was never a path problem.
# A100 is deliberately NOT used as a fallback: this cluster does not expose GPU
# memory via scontrol, and a 40 GB A100 would be worse than the L40S that
# already failed. H100 (80 GB) is the one confirmed-larger card -- it is what
# R7's anchor job moved to for the same reason (job 897770).
#
# evaluate.py / metrics_calculator.py stay byte-identical to upstream: the fix
# is environmental. Because upstream swallows metric errors and still exits 0,
# each task tees its own output and greps it -- a task with any `Error:` line
# exits non-zero rather than reporting a quietly corrupted CSV as success.
#
# Uses the five-bench env, NOT streamgve: .bashrc auto-activates streamgve, and
# `conda activate five-bench` alone does not pop it (the R2 job-880402 failure,
# ModuleNotFoundError: torchmetrics). Hence the explicit deactivate.
#SBATCH --job-name=r21_eval
#SBATCH --partition=H100
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/r21_eval_%A_%a.out
#SBATCH --error=logs/r21_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r21_blend_full
REF_ROOT=/projects/dataggen/outputs/five_bench

cd
source .bashrc
conda deactivate
conda activate five-bench

cd "$REPO"
mkdir -p logs evaluation/csv

# The R20 metric-crash fix. 144 of the OOM messages themselves recommended it:
# "reserved but unallocated memory is large, try setting expandable_segments".
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# All six edit types -- 100/100/100/100/9/10 = 419 pairs, exactly the set every
# arm rendered (verified against the arm dirs in wait-infer).
ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# One entry per array task: "<method dir> <csv stem WITHOUT the _avg suffix>".
# evaluate.py derives evaluation/csv/{stem}_avg.csv itself (plus per-edit-type
# intermediates edit{T}_FiVE_{stem}_frame_stride8*.csv), so a stem ending in
# _avg would yield {stem}_avg_avg.csv.
#
# Indices 0-4 are the R21 renders; 5-6 are stored references.
# paper_pvp is a reference too, not an experiment arm -- it is what the two pvp
# arms are read against, since R20's paper_pvp is only 22 clips.
#
# Task 5 (r7_visual_prompting) is the vp Eq.4 reference. Verified config-matched
# to the R21 vp arms 2026-08-11: run_fivebench.py defaults --flow_shift to 1.0
# (R7 passes it explicitly), --vp_mode to "vp" (so R7's --first_frame_edit_dir
# alone IS vp) and --blend_sched to None = paper Eq.4; step/fg_boost/blend_power
# /seed and the anchor root are identical. R7's frames are mtime 07-22 23:06,
# i.e. the post-per-pair-seeding re-render -- same generation as the arms.
#
# Task 6 (baseline) is the un-anchored novp Eq.4 floor -- R1, full-bench 419,
# same post-fix generation (07-22 21:00 -> 07-23 00:49). Not in the original
# plan scope (which was vp/pvp only), added so the table has a no-anchoring
# reference: without it every column is vp or pvp and R21 cannot say what
# anchoring itself buys, nor be read against R1/R20's numbers.
METHOD_DIRS=(
  "$OUT_ROOT/cos_third_vp"
  "$OUT_ROOT/cos_third_pvp"
  "$OUT_ROOT/zero_vp"
  "$OUT_ROOT/zero_pvp"
  "$OUT_ROOT/paper_pvp"
  "$REF_ROOT/r7_visual_prompting"
  "$REF_ROOT/baseline"
)
CSV_STEMS=(
  "r21_cos_third_vp"
  "r21_cos_third_pvp"
  "r21_zero_vp"
  "r21_zero_pvp"
  "r21_paper_pvp"
  "r21_ref_vp"
  "r21_ref_novp"
)

TID=${SLURM_ARRAY_TASK_ID:-0}
DIR=${METHOD_DIRS[$TID]}
STEM=${CSV_STEMS[$TID]}

echo "[r21_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} stem=$STEM dir=$DIR"
echo "[r21_eval] full bench, no --cases_json; annotations=${#ANNOTATIONS[@]}"
echo "[r21_eval] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if [ ! -d "$DIR" ]; then
  echo "[r21_eval] FAILED $STEM -- method dir does not exist: $DIR"
  exit 1
fi
# Count REAL pairs only. evaluate.py writes a sibling {video}_resize dir next to
# every video it scores, so a bare `ls` over an already-scored dir over-counts:
# r7_visual_prompting reads 441 = 419 real + 22 _resize left by R20's 22-clip
# run. The R21 arms will grow their own _resize dirs during this job.
NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r21_eval] real frame dirs in $DIR: $NDIRS  # expect 419"
if [ "$NDIRS" -ne 419 ]; then
  echo "[r21_eval] WARNING $STEM -- expected 419 real pairs, found $NDIRS"
fi

# NOTE: the overall row evaluate.py writes is a mean over the six per-edit-type
# means, not a mean over the 419 clips, so edit5 (9 clips) weighs as much as
# edit1 (100). Every arm AND both references go through this same path, so the
# comparison stays internally consistent -- but do not read these numbers as a
# per-clip average. (Plan decision: "single overall mean".)
METRICS_LOG="logs/r21_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" \
  --tgt_layout edit_video \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

# Upstream swallows per-metric exceptions and still exits 0, so a clean exit
# code is not evidence the CSV is clean. Any `Error:` line means a metric was
# dropped and the row's later columns are shifted -- treat it as a failure.
NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")

echo "[r21_eval] $STEM exit=$RC error_lines=$NERR oom_lines=$NOOM"
if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r21_eval] FAILED $STEM -- evaluation/csv/${STEM}_avg.csv not written"
  exit 1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r21_eval] FAILED $STEM -- metric crashes present (see $METRICS_LOG)"
  exit 1
fi

echo "[r21_eval] OK $STEM -> evaluation/csv/${STEM}_avg.csv"
exit 0
