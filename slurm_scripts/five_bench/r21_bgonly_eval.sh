#!/bin/bash
# Background-preservation metrics ONLY, scored against BOTH source renditions.
#
# task 0 -> src = mp4-decoded frames (mp4_src_root): what the model was actually fed.
# task 1 -> src = images/*.jpg      (benchmark root): what FiVE-Bench scores against
#                                    and what StreamGVE Table 1 is stated on.
#
# The pair is the point. Scoring against the mp4 ALONE flatters us: the 3-clip probe
# (job 941829) measured the same render at LPIPS 53 vs mp4 but 125 vs jpg, because
# mp4 compression pre-removes the high-frequency detail our pipeline fails to keep.
# So the mp4 column is a self-consistency diagnostic, NOT a Table 1 comparison.
#
# Only 5 metrics: no CoTracker / Qwen-VL / niqe, so this is far lighter than the
# full 16-metric run (job 937950, ~6h/arm) and carries none of its OOM exposure.
# mp4_src_root/images = JPEG q100 re-encode of the decoded mp4 (cost measured at
# LPIPS 1.7e-3, ~1% of the effect); bmasks symlinked so mask lookups are unchanged.
#SBATCH --job-name=r21_bgonly
#SBATCH --partition=L40S
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/r21_bgonly_%A_%a.out
#SBATCH --error=logs/r21_bgonly_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r21_blend_full
REF_ROOT=/projects/dataggen/outputs/five_bench
MP4_ROOT=/projects/dataggen/outputs/five_bench/mp4_src_root

cd
source .bashrc
conda deactivate
conda activate five-bench
cd "$REPO"
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

TID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TID" -eq 0 ]; then SRC="$MP4_ROOT"; TAG=mp4; else SRC="$DATA_ROOT"; TAG=jpg; fi
echo "[bgonly] task=$TID reference=$TAG src_image_folder=$SRC"

ANNOTATIONS=(); for T in 1 2 3 4 5 6; do ANNOTATIONS+=("$DATA_ROOT/edit_prompt/edit${T}_FiVE.json"); done

METRICS=(structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part)

DIRS=("$OUT_ROOT/cos_third_vp" "$OUT_ROOT/cos_third_pvp" "$OUT_ROOT/zero_vp" \
      "$OUT_ROOT/zero_pvp" "$OUT_ROOT/paper_pvp" "$REF_ROOT/r7_visual_prompting" "$REF_ROOT/baseline")
STEMS=(cos_third_vp cos_third_pvp zero_vp zero_pvp paper_pvp ref_vp ref_novp)

FAILED=0
for i in "${!DIRS[@]}"; do
  STEM="r21bg_${TAG}_${STEMS[$i]}"
  echo "[bgonly] === $STEM ==="
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --src_image_folder "$SRC" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --metrics "${METRICS[@]}" \
    --tgt_methods "${DIRS[$i]}" \
    --tgt_layout edit_video \
    --result_path "evaluation/csv/${STEM}.csv" \
    || { echo "[bgonly] FAILED $STEM"; FAILED=$((FAILED+1)); }
done
echo "[bgonly] task=$TID ($TAG) failures=$FAILED"
exit $(( FAILED > 0 ))
