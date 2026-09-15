#!/bin/bash
# R5 -- evaluate R5's bg_realwords arm and emit the 16-metric r5_bg_realwords_avg.csv (compared vs baseline_avg.csv).
# edit_video layout: {tgt_root}/edit{T}/{video_name}/*.png. Requires R5 outputs to exist.
#SBATCH --job-name=r5_eval
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r5_eval_%j.out
#SBATCH --error=logs/r5_eval_%j.err

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
  --tgt_methods "$OUT_ROOT"/r5_bg_realwords \
  --tgt_layout edit_video \
  --tgt_key r5_bg_realwords \
  --result_path evaluation/csv/r5_bg_realwords.csv
