#!/bin/bash
# R2 -- reproduce the reference 8-Wan-Edit numbers with the ported FiVE-Bench harness.
# Native FiVE layout ({video_name}/{save_dir}/) + the 8_Wan_Edit key so columns match the reference.
# Requires: fetch-reference (git lfs pull + unzip results/8-Wan-Edit.zip) done first.
#SBATCH --job-name=r2_eval_repro
#SBATCH --partition=A100,L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r2_eval_repro_%j.out
#SBATCH --error=logs/r2_eval_repro_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# NOTE: the released 8-Wan-Edit.zip only contains edit type 5 (9 videos), and unzip double-nests it.
WAN_EDIT=$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/results/8-Wan-Edit/8-Wan-Edit


cd
source .bashrc
conda deactivate   # .bashrc auto-activates streamgve one level deep; pop it so five-bench's python wins
conda activate five-bench

cd "$REPO"
mkdir -p logs evaluation/csv

# edit5 only -- the reference frames cover just that type; compare to edit5_..._avg.csv
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files \
    "$DATA_ROOT"/edit_prompt/edit5_FiVE.json \
  --tgt_methods "$WAN_EDIT" \
  --tgt_layout fivebench \
  --tgt_key 8_Wan_Edit \
  --result_path evaluation/csv/r2_repro.csv
