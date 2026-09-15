#!/bin/bash
# R28 step 3 -- re-score all 52 R26 arms with all THREE background families.
#
# ⚠️ THE METRIC LIST MUST BE PASSED AS --metrics ON THE COMMAND LINE. The `metrics:` key
# in config.yaml is VESTIGIAL -- evaluate.py reads `metrics = args.metrics`, i.e. the
# argparse flag whose hardcoded default includes five_acc and motion_fidelity. R26 lost a
# full 9-arm eval run to exactly this: a reduced config.yaml was written, ignored, and
# Qwen2.5-VL + CoTracker both loaded and OOMed on a 40GB A100.
#
# All 54 arms are scored: ALL 52 R26 (tau_bg, tau_fg) cells (extended 2026-09-01 from the
# original 12, see r28_dump.sh) plus both Eq.4 baselines. `r7_visual_prompting` (vp) is
# the like-for-like comparator for the vp-anchored R26 arms; `baseline` (novp) is the
# paper method without visual prompting.
#
# The FIXED union shifts when the arm set grows, so every arm here -- including the 12
# already scored under the old 14-arm union -- MUST be re-run against the union rebuilt
# from all 54 arms (r28_fixed_union.py). The old `r28_taubg*_avg.csv` files for those 12
# are stale the moment r28_fixed_union.py is re-run and must not be read after this step.
#SBATCH --job-name=r28_eval
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/r28_eval_%A_%a.out
#SBATCH --error=logs/r28_eval_%A_%a.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# SINGLE ENVIRONMENT: streamgve. GroundingDINO + SAM2 were installed into it on
# 2026-08-31 (SAM-2 1.0 from facebookresearch/sam2 @2b90b9f5, plus hydra-core, iopath,
# portalocker, qwen-vl-utils) precisely so this task does not straddle two envs.
# streamgve already carried transformers 5.12.0, which exposes the `threshold` kwarg that
# r25_iou.py calls -- transformers 4.44 (addit, DGE) calls it `box_threshold` and raises
# TypeError, which is what killed job 965789. Nothing pre-existing was upgraded: the
# install added 4 packages and changed no version, and the WAN pipeline
# (WanVAEWrapper / causal_model / edit_causal_inference) was re-imported afterwards to
# confirm it. Absolute interpreter path, so no conda activation stacking is possible.
PY=~/anaconda3/envs/streamgve/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R26_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau
FB_ROOT=/projects/dataggen/outputs/five_bench
MASK_ROOT=/projects/dataggen/outputs/five_bench/r28_tgt_masks
CASES=evaluation/cases.json

# Same order and spelling as r28_dump.sh -- FULL PATHS, byte-identical to r28_dump.sh's
# ARMS -- the two lists must stay the same set or an arm gets masks without a score, or a
# score without masks. Extended 2026-08-31 from the 12 R26 cells to 14 (both Eq.4
# baselines, dumped only to widen the fixed union). Extended AGAIN 2026-09-01 to the full
# 52-cell R26 grid (54 arms total) alongside r28_dump.sh; see that script's header for the
# tau_bg/tau_fg ranges. `r7_visual_prompting` (vp) is the like-for-like comparator for the
# vp-anchored R26 arms; `baseline` (novp) is the paper method without visual prompting.
ARMS=("$R26_ROOT/taubg0_taufg2_vp" "$R26_ROOT/taubg0_taufg3_vp" "$R26_ROOT/taubg0_taufg4_vp" "$R26_ROOT/taubg0_taufg6_vp"
      "$R26_ROOT/taubg0_taufg8_vp" "$R26_ROOT/taubg0_taufg10_vp" "$R26_ROOT/taubg0_taufg20_vp" "$R26_ROOT/taubg0_taufg50_vp"
      "$R26_ROOT/taubg1_taufg2_vp" "$R26_ROOT/taubg1_taufg3_vp" "$R26_ROOT/taubg1_taufg4_vp" "$R26_ROOT/taubg1_taufg6_vp"
      "$R26_ROOT/taubg1_taufg8_vp" "$R26_ROOT/taubg1_taufg10_vp" "$R26_ROOT/taubg1_taufg20_vp" "$R26_ROOT/taubg1_taufg50_vp"
      "$R26_ROOT/taubg2_taufg2_vp" "$R26_ROOT/taubg2_taufg3_vp" "$R26_ROOT/taubg2_taufg4_vp" "$R26_ROOT/taubg2_taufg6_vp"
      "$R26_ROOT/taubg2_taufg8_vp" "$R26_ROOT/taubg2_taufg10_vp" "$R26_ROOT/taubg2_taufg20_vp" "$R26_ROOT/taubg2_taufg50_vp"
      "$R26_ROOT/taubg3_taufg3_vp" "$R26_ROOT/taubg3_taufg4_vp" "$R26_ROOT/taubg3_taufg6_vp" "$R26_ROOT/taubg3_taufg8_vp"
      "$R26_ROOT/taubg3_taufg10_vp" "$R26_ROOT/taubg3_taufg20_vp" "$R26_ROOT/taubg3_taufg50_vp" "$R26_ROOT/taubg4_taufg4_vp"
      "$R26_ROOT/taubg4_taufg6_vp" "$R26_ROOT/taubg4_taufg8_vp" "$R26_ROOT/taubg4_taufg10_vp" "$R26_ROOT/taubg4_taufg20_vp"
      "$R26_ROOT/taubg4_taufg50_vp" "$R26_ROOT/taubg6_taufg6_vp" "$R26_ROOT/taubg6_taufg8_vp" "$R26_ROOT/taubg6_taufg10_vp"
      "$R26_ROOT/taubg6_taufg20_vp" "$R26_ROOT/taubg6_taufg50_vp" "$R26_ROOT/taubg8_taufg8_vp" "$R26_ROOT/taubg8_taufg10_vp"
      "$R26_ROOT/taubg8_taufg20_vp" "$R26_ROOT/taubg8_taufg50_vp" "$R26_ROOT/taubg10_taufg10_vp" "$R26_ROOT/taubg10_taufg20_vp"
      "$R26_ROOT/taubg10_taufg50_vp" "$R26_ROOT/taubg20_taufg20_vp" "$R26_ROOT/taubg20_taufg50_vp" "$R26_ROOT/taubg50_taufg50_vp"
      "$FB_ROOT/baseline"           "$FB_ROOT/r7_visual_prompting")

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 53 ]; then
  echo "[r28_eval] bad array id ${TID} (expected 0-53)"; exit 1
fi
DIR="${ARMS[$TID]}"
ARM=$(basename "$DIR")
STEM="r28_${ARM}"

echo "[r28_eval] node=$(hostname) task=${TID} arm=${ARM} stem=${STEM}"

if [ ! -d "$MASK_ROOT/$ARM" ]; then
  echo "[r28_eval] MISSING target masks $MASK_ROOT/$ARM -- run r28_dump.sh first"; exit 1
fi
if [ ! -d "$MASK_ROOT/_fixed_union" ]; then
  echo "[r28_eval] MISSING $MASK_ROOT/_fixed_union -- run r28_fixed_union.py first"; exit 1
fi
NM=$(ls "$MASK_ROOT/$ARM"/edit*/*.npz 2>/dev/null | wc -l)
if [ "$NM" -ne 22 ]; then
  echo "[r28_eval] FAILED ${STEM} -- ${NM} target masks, expected 22."
  echo "[r28_eval]   Scoring a partial arm would still write a full-looking table. Stop."
  exit 1
fi

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

METRICS_LOG="logs/r28_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
$PY evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part \
            mse_unedit_part ssim_unedit_part \
            psnr_unedit_union lpips_unedit_union \
            mse_unedit_union ssim_unedit_union \
            psnr_unedit_union_fixed lpips_unedit_union_fixed \
            mse_unedit_union_fixed ssim_unedit_union_fixed \
            clip_similarity_source_image clip_similarity_target_image \
            clip_similarity_target_image_edit_part \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" \
  --tgt_layout edit_video \
  --tgt_mask_dir "$MASK_ROOT/$ARM" \
  --fixed_union_dir "$MASK_ROOT/_fixed_union" \
  --cases_json "$CASES" \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r28_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r28_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"; exit 1
fi
# upstream evaluate.py swallows per-metric exceptions and still exits 0, so a clean exit
# code alone is not evidence the CSV is clean.
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r28_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"; exit 1
fi
echo "[r28_eval] OK ${STEM} -> evaluation/csv/${STEM}_avg.csv"
exit 0
