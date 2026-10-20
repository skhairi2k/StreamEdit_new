#!/bin/bash
# R37 alpha-study figures: 5 fixed examples x 12 R35/R36 arms -> 4 PNGs each (Fig 1a
# tolerance, 1b errors, 2 alpha curves, 3 ring curves) + README in
# evaluation/figures/r37_alpha_study/. Recomputes everything from the renders with
# bg_metrics (no dependence on the r37_bg_*.csv files), ~30 s per example on an L40S.
# Also runnable directly inside an interactive allocation: bash slurm_scripts/five_bench/r37_figures.sh
#SBATCH --job-name=r37_figures
#SBATCH --partition=L40S
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r37_figures_%j.out
#SBATCH --error=logs/r37_figures_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT_DIR=evaluation/figures/r37_alpha_study

cd "$REPO" || exit 1
# The eval env is five-bench; .bashrc auto-activates streamgve, so pop it first.
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs "$OUT_DIR"
export PYTHONUNBUFFERED=1

echo "[r37_fig] node=$(hostname) job=${SLURM_JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

python evaluation/r37_visualize_alpha.py \
  --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
  --out_root ~/Data/dataggen/outputs/five_bench \
  --out_dir "$OUT_DIR" \
  --stride 8 --tol_frames 0 16 32 48 64 --err_frame 32 \
  --alpha_max 0.5 --seed 0
RC=$?

NPNG=$(ls "$OUT_DIR"/*.png 2>/dev/null | wc -l)
if [ "$RC" -ne 0 ] || [ "$NPNG" -ne 20 ] || [ ! -f "$OUT_DIR/README.md" ]; then
  echo "[r37_fig] FAILED -- exit=${RC} png=${NPNG}/20 readme=$([ -f "$OUT_DIR/README.md" ] && echo yes || echo no)"
  exit 1
fi
echo "[r37_fig] OK -- 20 PNG + README in $OUT_DIR"
exit 0
