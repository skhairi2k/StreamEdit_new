#!/bin/bash
# R49: render the new clip (0004_car-roundabout, edit 3) under the 10 global Wan-Edit configs, so that every clip of
# evaluation/r49_cases.json has a baseline render.
#
# Usage: sbatch slurm_scripts/five_bench/r49_newclip_baselines.sh        (normally submitted by r49_chain.sh, no dependency)
# Configs: the 9 `r48_wan_sched.py --list` baselines (native_s0.95 .. native_s0.4, matched_s0.95 .. matched_s0.85) plus
# native_s0.978 (the global start at step 5). One GPU job loops them serially (the model loads once per config), one clip each.
# The other 17 clips of r49_cases.json already have these renders in the same directories, so this writes ONLY the new clip:
#   $D/r48_wan_arms/{config}_subset/edit3/0004_car-roundabout/*.png     (resumable: a finished clip is skipped)
#SBATCH --job-name=r49_newclip_baselines
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_newclip_baselines_%j.out
#SBATCH --error=logs/r49_newclip_baselines_%j.err

NEW_CLIP=evaluation/r49_new_clip.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r48_wan_arms
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

mapfile -t CONFIGS < <($PY_FB evaluation/r48_wan_sched.py --list 2>/dev/null)
[ "${#CONFIGS[@]}" -eq 9 ] || { echo "[r49_newclip] FATAL: expected 9 baseline configs, got ${#CONFIGS[@]}"; exit 1; }
CONFIGS+=(native_s0.978)
echo "[r49_newclip] job=${SLURM_JOB_ID:-local} node=$(hostname) configs (${#CONFIGS[@]}): ${CONFIGS[*]}"
nvidia-smi -L 2>&1 | head -2

FAILED=0
for CONFIG in "${CONFIGS[@]}"; do
  echo "=== [r49_newclip] ${CONFIG} $(date '+%F %T')"
  $PY_FB evaluation/r48_wan_render.py --config "$CONFIG" --mode subset --cases "$NEW_CLIP" --out_root "$OUT" \
    || { echo "[r49_newclip] r48_wan_render.py reported failures for ${CONFIG}"; FAILED=$((FAILED + 1)); }
  MISSING=$($PY_FB evaluation/r48_units.py --mode subset --missing "$OUT/${CONFIG}_subset" --cases "$NEW_CLIP")
  if [ -n "$MISSING" ]; then
    echo "[r49_newclip] ${CONFIG}: new clip has no complete render, e.g. $(echo "$MISSING" | head -1)"; FAILED=$((FAILED + 1))
  else
    echo "[r49_newclip] ${CONFIG}: new clip complete"
  fi
done
echo "[r49_newclip] failures: ${FAILED}"
exit $(( FAILED > 0 ))
