#!/bin/bash
# Official FiVE-Bench evaluate.py on the 3-clip src-format probe (job 941829).
# Both arms scored against the OFFICIAL default reference (images/*.jpg):
#   src_video  = mp4 fed to the model  (the historical config; full bench = 213.5)
#   src_images = jpg fed to the model  (input and reference finally consistent)
# Same 3 clips both arms, so the delta is the input rendition alone.
#SBATCH --job-name=r21_srcfmt_eval
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r21_srcfmt_eval_%j.out
#SBATCH --error=logs/r21_srcfmt_eval_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT=/projects/dataggen/outputs/five_bench/r21_srcfmt

cd; source .bashrc; conda deactivate; conda activate five-bench; cd "$REPO"
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

for ARM in src_video src_images; do
  echo "[srcfmt_eval] === $ARM (official jpg reference) ==="
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --src_image_folder "$DATA_ROOT" \
    --annotation_mapping_files "$DATA_ROOT"/edit_prompt/edit1_FiVE.json \
    --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part \
    --tgt_methods "$OUT/$ARM" \
    --tgt_layout edit_video \
    --cases_json "$REPO/evaluation/cases_srcfmt.json" \
    --result_path "evaluation/csv/r21sf_${ARM}.csv" \
    || { echo "[srcfmt_eval] FAILED $ARM"; exit 1; }
done
echo "[srcfmt_eval] done"
