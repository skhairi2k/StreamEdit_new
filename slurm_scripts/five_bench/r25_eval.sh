#!/bin/bash
# R25 evaluate -- 9 FiVE metrics over one arm's 22 clips (see the --metrics list below;
# five_acc and motion_fidelity are deliberately excluded, so there is NO temporal metric).
#
# SINGLE TASK, not an array: one arm, 22 clips. At R20's measured ~45 s/clip that is
# well under 20 minutes of scoring, so splitting per edit type would cost six model
# loads to save nothing. (Contrast R21, which was an array only because it scored 7
# methods x 419 pairs against a 24h partition ceiling.)
#
# --cases_json IS passed here, unlike r21_eval.sh: the arm holds only the 22 pairs and
# evaluate.py would otherwise look for the other 397 and fail. It is passed once, to
# the single method being scored.
#
# JOIN KEY: evaluate.py assigns file_id from `enumerate` over the FULL annotation file
# and --cases_json only `continue`s past unlisted pairs (evaluate.py:229 / :268-272),
# so file_ids do NOT renumber under a subset. That is what lets r25_summarize.py join
# these rows against the stored full-bench r21_ref_vp CSVs per edit type.
#
# H100 + expandable_segments: the R21 fix for R20's crash pair. R20 ran this on L40S
# and hit GPU exhaustion in the parent (44.35/44.39 GiB) once Qwen2.5-VL + CoTracker
# were resident -- 2680 error lines in edit2 alone -- and the niqe failures were the
# same cause one level down (calculate_NIQE shells out to inference_iqa.py, the child
# cannot init CUDA, writes no txt, average_niqe_from_txt raises FileNotFoundError).
# The fix is headroom only; evaluate.py / metrics_calculator.py stay byte-identical
# to upstream.
#
# EXPECTED METRIC ERRORS: none. 0010_giant-slalom -- the empty-mask
# motion_fidelity_score_edit_part degeneracy of R2/R14/R21 -- is NOT in cases.json, so
# unlike R21 any `Error:` line here is a real failure and fails the job.
#
# conda: `conda deactivate` first -- .bashrc auto-activates streamgve and
# `conda activate five-bench` alone does not pop it (the R2 job-880402 failure).
# ⚠️ THE METRIC LIST MUST BE PASSED AS --metrics ON THE COMMAND LINE. The `metrics:` key
# in config.yaml is VESTIGIAL -- evaluate.py:200 reads `metrics = args.metrics`, i.e. the
# argparse flag whose hardcoded default (evaluate.py:619-633) includes five_acc and
# motion_fidelity. Job 961342 was submitted believing a reduced config.yaml would take
# effect; it did not, Qwen2.5-VL + CoTracker both loaded, and motion_fidelity_score OOMed
# on a 40GB A100 (2.52 GiB requested, 37.15 GiB already resident) followed by 14 niqe
# failures. Editing a config file does NOT change which metrics run.
#SBATCH --job-name=r25_eval
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
#SBATCH --output=logs/r25_eval_%j.out
#SBATCH --error=logs/r25_eval_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r25_adaptive_tau
CASES="$REPO/evaluation/cases.json"

# ARM SELECTION (phase 2). The three arms differ ONLY in which tau each pair gets;
# seed, anchors, cases_json, sampler defaults and the metric list are held identical,
# because the whole attribution argument rests on that. Unset R25_ARM reproduces the
# completed phase-1 adaptive run exactly.
#   adaptive  IoU -> tau            (phase 1)
#   reversed  same 22 tau values, pairing inverted by IoU rank -> does the PAIRING matter?
#   constant  tau = 6.57 everywhere                            -> does VARYING tau matter?
ARM=${R25_ARM:-adaptive}
case "$ARM" in
  adaptive|reversed|constant) ;;
  *) echo "[r25] bad R25_ARM='$ARM' (expected adaptive|reversed|constant)"; exit 1 ;;
esac
METHOD="r25_${ARM}_vp"
# The phase-1 map predates the naming scheme and keeps its original filename, so the
# adaptive arm must not be rewritten to r25_tau_map_adaptive.csv.
if [ "$ARM" = "adaptive" ]; then
  TAU_MAP="$REPO/evaluation/csv/r25_tau_map.csv"
else
  TAU_MAP="$REPO/evaluation/csv/r25_tau_map_${ARM}.csv"
fi
METHOD_DIR="$OUT_ROOT/$METHOD"
STEM="$METHOD"

# Activate conda by absolute path, NOT via `source .bashrc`: this job is submitted with
# --export=NONE (so it cannot inherit the submitting allocation's env), and .bashrc
# returns early in a non-interactive shell, which would leave conda uninitialised and
# kill the job at `conda activate`. r25_infer/r25_smoke use this same pattern.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 job-880402 bug)
conda activate five-bench
mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# GPU diagnostics: on 2026-08-27 node52 accepted L40S batch jobs while exposing no
# device (nvidia-smi -L -> "No devices found." with SLURM_JOB_GPUS set), killing two
# r25_infer submissions at CUDA init. This job runs unattended overnight, so print the
# GPU view up front -- a silent 4am failure with no evidence costs a whole day.
echo "[r25_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -3
python -c "import torch; print('[r25_eval] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -1

echo "[r25_eval] job=${SLURM_JOB_ID} arm=$ARM stem=$STEM dir=$METHOD_DIR"
echo "[r25_eval] cases_json=$CASES annotations=${#ANNOTATIONS[@]}"
echo "[r25_eval] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

if [ ! -d "$METHOD_DIR" ]; then
  echo "[r25_eval] FAILED -- method dir does not exist: $METHOD_DIR"
  echo "[r25_eval] run the infer step first"
  exit 1
fi

# Count REAL pairs only: evaluate.py writes a sibling {video}_resize dir next to every
# video it scores, so a bare `ls` over an already-scored tree over-counts (the R21 441
# vs 419 confusion).
NDIRS=$(ls -d "$METHOD_DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r25_eval] real frame dirs: $NDIRS  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r25_eval] FAILED -- expected 22 real pairs, found $NDIRS; do not score a"
  echo "[r25_eval] partial arm, the per-edit-type means would be silently wrong"
  exit 1
fi

METRICS_LOG="logs/r25_eval_${SLURM_JOB_ID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part \
            mse_unedit_part ssim_unedit_part clip_similarity_source_image \
            clip_similarity_target_image clip_similarity_target_image_edit_part \
            niqe_target_image \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$METHOD_DIR" \
  --tgt_layout edit_video \
  --cases_json "$CASES" \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

# Upstream swallows per-metric exceptions and still exits 0, so a clean exit code is
# NOT evidence the CSV is clean: `except: continue` drops a metric column and shifts
# every later one (how R20 corrupted five_acc). Any `Error:` line is a failure.
NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r25_eval] $STEM exit=$RC error_lines=$NERR oom_lines=$NOOM"

NAVG=$(ls evaluation/csv/edit?_FiVE_${STEM}_frame_stride8_avg.csv 2>/dev/null | wc -l)
echo "[r25_eval] per-edit-type _avg.csv written: $NAVG  # expect 6"

if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ] || [ "$NAVG" -ne 6 ]; then
  echo "[r25_eval] FAILED -- see $METRICS_LOG"
  exit 1
fi

echo "[r25_eval] OK"
