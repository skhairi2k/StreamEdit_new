#!/bin/bash
# R2 -- evaluate R1's baseline arm and emit the shared 16-metric baseline_avg.csv (consumed by R3).
# edit_video layout: {tgt_root}/edit{T}/{video_name}/*.png. Requires R1 outputs to exist.
#SBATCH --job-name=r2_eval_baseline
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r2_eval_baseline_%j.out
#SBATCH --error=logs/r2_eval_baseline_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench


cd
source .bashrc
conda deactivate   # .bashrc auto-activates streamgve one level deep; pop it so five-bench's python wins
conda activate five-bench

cd "$REPO"
mkdir -p logs evaluation/csv

python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files \
    "$DATA_ROOT"/edit_prompt/edit1_FiVE.json \
    "$DATA_ROOT"/edit_prompt/edit2_FiVE.json \
    "$DATA_ROOT"/edit_prompt/edit3_FiVE.json \
    "$DATA_ROOT"/edit_prompt/edit4_FiVE.json \
    "$DATA_ROOT"/edit_prompt/edit5_FiVE.json \
    "$DATA_ROOT"/edit_prompt/edit6_FiVE.json \
  --tgt_methods "$OUT_ROOT"/baseline \
  --tgt_layout edit_video \
  --tgt_key baseline \
  --result_path evaluation/csv/baseline.csv
