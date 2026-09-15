#!/bin/bash
# R10 -- evaluation. Five metrics, one joined table, twenty qualitative grids.
#
# THE FIVE METRICS, AND WHY ONLY THESE
# ------------------------------------
# R10 injects the Qwen-edited first frame ("the anchor") into head subsets. Every
# prediction it makes is about the EDIT REGION, so every metric is masked to it.
#
#   anchor_sim_f0          did the anchor's appearance land?   PRIMARY discriminator
#   anchor_sim_mean        ...and stay?                        co-primary
#   anchor_sim_retention   last/f0 -- the fading comparison against the R7 §4.5
#                          baseline. Reported INSTEAD of relying on the OLS slope:
#                          the §4.5 decay is step-shaped (0.76 -> 0.46 by frame 8,
#                          then flat), so a linear fit understates it badly.
#   motion_fidelity_score_edit_part   did the subject freeze?  R10 ARMS ONLY
#   clip_similarity_target_image_edit_part   did *an* edit happen? guards the
#                          degenerate reading "never adopted => nothing to lose"
#   lpips_unedit_part      is the background untouched? plumbing; should be flat
#
# Dropped, whole-frame dilution: motion_fidelity_score, structure_distance,
# clip_similarity_target_image. Every arm injects source background keys, so the
# background is pinned to the source by construction and scores near-perfectly
# regardless of the gate; averaging it in destroyed the signal (`all` vs `none`
# scored 0.775 vs 0.773, p=0.45 on whole-frame motion fidelity).
# Dropped, no readable direction: lpips_edit_part, structure_distance_edit_part --
# they score the output against the SOURCE inside the region we deliberately
# change, so a successful edit must score badly on them.
#
# DO NOT compare motion_fidelity_score_edit_part against the r7_vp row. R10 runs
# blend_off (query blending, which the paper credits with motion preservation, is
# disabled) while R7 keeps it. That gap is the ablation, not the gate.
#
# Needs a GPU + the five-bench env (CLIP / LPIPS / DINO / CoTracker).
#SBATCH --job-name=r10_eval
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r10_eval_%j.out
#SBATCH --error=logs/r10_eval_%j.err

set -euo pipefail

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench
ARMS="none all spatial temporal rand_spatial rand_temporal"

cd
source .bashrc
conda deactivate   # .bashrc auto-activates streamgve one level deep; pop it so five-bench's python wins
conda activate five-bench

cd "$REPO"
mkdir -p logs evaluation/csv/r10 evaluation/figures/r10

# Stage 1 -- the three benchmark metrics, through the UNMODIFIED evaluator so
# R10 stays comparable to R1/R7. Writes into evaluation/csv/r10/, never the top
# level: the superseded eight-metric run wrote r10_arms.csv there and the two
# must not be merged.
echo "[r10_eval] stage 1/4 -- benchmark metrics"
python evaluation/r10_metrics.py \
  --in_root "$OUT_ROOT"/r10_vp_arms \
  --arms $ARMS \
  --r7_root "$OUT_ROOT"/r7_visual_prompting \
  --src_image_folder "$DATA_ROOT" \
  --annotation_dir "$DATA_ROOT"/edit_prompt \
  --cases_json evaluation/cases.json \
  --frame_stride 2 \
  --metrics motion_fidelity_score_edit_part \
            clip_similarity_target_image_edit_part \
            lpips_unedit_part \
  --out_csv evaluation/csv/r10 \
  --out_fig evaluation/figures/r10

# Stage 2 -- anchor-referenced metrics. No benchmark equivalent exists
# (calculate_clip_similarity is image-to-TEXT only). --r7_root is required, not
# optional: the §4.5 arm is the fading reference retention is read against.
echo "[r10_eval] stage 2/4 -- anchor similarity"
python evaluation/r10_anchor_sim.py \
  --in_root "$OUT_ROOT"/r10_vp_arms \
  --arms $ARMS \
  --r7_root "$OUT_ROOT"/r7_visual_prompting \
  --src_image_folder "$DATA_ROOT" \
  --cases_json evaluation/cases.json \
  --frame_stride 2 \
  --out_csv evaluation/csv/r10/r10_anchor_sim.csv

# Stage 3 -- one table. Keeps only the five metrics; keys on
# (arm, video_name, editing_type_id) because 0011_lucia is two clips.
echo "[r10_eval] stage 3/4 -- join"
python evaluation/r10_join.py \
  --bench_csv evaluation/csv/r10/r10_arms.csv \
  --anchor_csv evaluation/csv/r10/r10_anchor_sim.csv \
  --out evaluation/csv/r10/r10_final.csv

# Stage 4 -- qualitative grids: source / appearance anchor / VP baseline / 6 arms.
echo "[r10_eval] stage 4/4 -- grids"
python evaluation/r10_make_grids.py \
  --in_root "$OUT_ROOT"/r10_vp_arms \
  --arms $ARMS \
  --r7_root "$OUT_ROOT"/r7_visual_prompting \
  --anchor_root "$OUT_ROOT"/anchors \
  --src_root "$DATA_ROOT"/images \
  --out evaluation/figures/r10/grids

echo "[r10_eval] final table: evaluation/csv/r10/r10_final.csv"
echo "[r10_eval] rows: $(($(wc -l < evaluation/csv/r10/r10_final.csv) - 1)) (expect 140 = 7 arms x 20 clips)"
echo "[r10_eval] grids: $(ls evaluation/figures/r10/grids/*.png 2>/dev/null | wc -l) (expect 20)"
