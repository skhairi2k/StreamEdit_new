#!/bin/bash
# (root: r59_max_full by default -- r49_se_arms is a /store link, unreadable on compute nodes; ROOT=<dir> overrides)
# R57 throw-away diagnostic: CF's 72B judge died in generate() with NaN/inf probabilities on all 3 smoke clips (job 1066767).
# Runs cf_debug.py with the small Qwen2.5-VL-7B-Instruct on ONE GPU, same code path, same 3 clips, to tell a pipeline problem
# from a 72B-specific one. Usage: sbatch slurm_scripts/five_bench/eval_suite_cf_debug.sh
#SBATCH --job-name=suite_cf_debug
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_suite_cf_debug_%j.out
#SBATCH --error=logs/eval_suite_cf_debug_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
# 72B probe (per-layer NaN report): sbatch --gres=gpu:4 --mem=256G --export=ALL,MODEL=/work/PERSO/skhairi/weights/Qwen2.5-VL-72B-Instruct <this script>
M7B=${MODEL:-$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*/ | head -1)}
cd "$REPO" || exit 1
mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
echo "[cf_debug] node=$(hostname) model=${M7B}"; nvidia-smi -L 2>&1 | head -2
FFBIN=$(mktemp -d -p "${TMPDIR:-/tmp}" suite_ffbin.XXXXXX)
trap 'rm -rf "$FFBIN"' EXIT
ln -s "$($PY_FB -I -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$FFBIN/ffmpeg"
export PATH="$FFBIN:$PATH"
$PY_FB -I evaluation/eval_suite/cf_debug.py --model_path "$M7B" \
  --root "${ROOT:-$HOME/Data/dataggen/outputs/five_bench/r59_se_arms/r59_max_full/step14}" --layout edit_video \
  --cases evaluation/r57_smoke_cases.json --out /dev/null 2>&1 | grep -E "^\[cf_debug\]|Error|Traceback|^  File .*(cf_|content_fid|generation)|Assertion"
