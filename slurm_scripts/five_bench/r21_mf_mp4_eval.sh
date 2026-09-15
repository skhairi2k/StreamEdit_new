#!/bin/bash
# The 3 remaining SOURCE-DEPENDENT metrics, re-scored against the mp4 rendition
# the model was actually fed. Verified from calculate_metric's call signatures that
# only these 3 of the other 11 read the source at all:
#   clip_similarity_source_image     (src_image, src_prompt)
#   motion_fidelity_score            (src_video_path, tgt_video_path)
#   motion_fidelity_score_edit_part  (same, masked)
# The other 8 (clip_similarity_target_image, ..._edit_part, niqe_target_image,
# five_acc x5) are target-only and would re-score byte-identically -- running them
# would burn CoTracker/Qwen-VL time for no new number.
#
# PREDICTION under test: motion_fidelity_score is -4.4% off the paper on the jpg
# reference (86.90 vs 90.93), same direction/magnitude as the preservation metrics
# were before the reference was corrected (job 951202: all 5 collapsed to <=1%).
# If the mp4 reference closes it too, that independently confirms the story; if not,
# motion fidelity is a separate issue.
#
# H100 + expandable_segments: this is the CoTracker path, which OOM'd 144x on the
# 44GB L40S in R20 (job 907650). 3 array tasks, not 7 -- QOS caps submitted entries
# at ~8 and 4 were already queued.
#
# GATE LESSON (job 937950): a bare "any Error: line = fail" gate is useless here.
# motion_fidelity_score_edit_part RAISES on the empty-edit-mask degeneracy (R2/R14)
# for 0010_giant-slalom in edit1-4 -- 4 per arm, identical across arms, now written
# as nan by the evaluate.py:487 patch. So fail on OOM / niqe / anything unrecognised,
# and treat ONLY that specific message as expected.
#SBATCH --job-name=r21_mfmp4
#SBATCH --partition=H100
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/r21_mfmp4_%A_%a.out
#SBATCH --error=logs/r21_mfmp4_%A_%a.err

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

ANNOTATIONS=(); for T in 1 2 3 4 5 6; do ANNOTATIONS+=("$DATA_ROOT/edit_prompt/edit${T}_FiVE.json"); done
METRICS=(clip_similarity_source_image motion_fidelity_score motion_fidelity_score_edit_part)

DIRS=("$OUT_ROOT/cos_third_vp" "$OUT_ROOT/cos_third_pvp" "$OUT_ROOT/zero_vp" \
      "$OUT_ROOT/zero_pvp" "$OUT_ROOT/paper_pvp" "$REF_ROOT/r7_visual_prompting" "$REF_ROOT/baseline")
STEMS=(cos_third_vp cos_third_pvp zero_vp zero_pvp paper_pvp ref_vp ref_novp)

TID=${SLURM_ARRAY_TASK_ID:-0}
NT=3
FAILED=0
for i in "${!DIRS[@]}"; do
  [ $(( i % NT )) -ne "$TID" ] && continue
  STEM="r21mf_mp4_${STEMS[$i]}"
  LOG="logs/r21_mfmp4_${SLURM_ARRAY_JOB_ID}_${TID}_${STEMS[$i]}.metrics.log"
  echo "[mfmp4] === $STEM ==="
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --src_image_folder "$MP4_ROOT" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --metrics "${METRICS[@]}" \
    --tgt_methods "${DIRS[$i]}" \
    --tgt_layout edit_video \
    --result_path "evaluation/csv/${STEM}.csv" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  NOOM=$(grep -c 'out of memory' "$LOG")
  NNIQE=$(grep -c 'Error: niqe' "$LOG")
  NEXPECTED=$(grep -c 'Error: motion_fidelity_score_edit_part: min()' "$LOG")
  NOTHER=$(grep 'Error:' "$LOG" | grep -vc 'motion_fidelity_score_edit_part: min()')
  echo "[mfmp4] $STEM exit=$RC oom=$NOOM niqe=$NNIQE expected_emptymask=$NEXPECTED other_errors=$NOTHER"
  if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
    echo "[mfmp4] FAILED $STEM -- no _avg.csv"; FAILED=$((FAILED+1)); continue
  fi
  if [ "$RC" -ne 0 ] || [ "$NOOM" -ne 0 ] || [ "$NNIQE" -ne 0 ] || [ "$NOTHER" -ne 0 ]; then
    echo "[mfmp4] FAILED $STEM -- real crashes present"; FAILED=$((FAILED+1)); continue
  fi
  echo "[mfmp4] OK $STEM"
done
echo "[mfmp4] task=$TID failures=$FAILED"
exit $(( FAILED > 0 ))
