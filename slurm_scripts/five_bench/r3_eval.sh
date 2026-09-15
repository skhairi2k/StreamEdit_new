#!/bin/bash
# R3 -- evaluate the selectivity arm with R2's shared FiVE-Bench harness.
# NOTE: requires R2's evaluation/fivebench/evaluate.py to exist; the `eval` step depends on R2.
#SBATCH --job-name=r3_eval
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=24:00:00
#SBATCH --output=logs/r3_eval_%j.out
#SBATCH --error=logs/r3_eval_%j.err

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
  --tgt_methods "$OUT_ROOT"/r3_selectivity \
  --tgt_layout edit_video \
  --result_path evaluation/csv/r3_selectivity.csv
