#!/bin/bash
# R65 smoke: AnyV2V (I2VGen-XL) driver on the 2 reproduction-gate clips (official FiVE prompts + the released first frame),
# then on the 2 smoke pairs (edit1/0001_bus and the 25-frame edit1/0072_dog-agility, Correct prompts + anchors).
#
# Usage: sbatch slurm_scripts/five_bench/r65_smoke.sh
# Outputs under $D/r65_anyv2v: repro_raw512 / repro_srcres, smoke_raw512 / smoke_srcres, *_manifest_shard0.csv.
# The manifests give sec/video and peak memory for the gate (peak < 44 GB; K and --time of r65_render.sh follow from sec/video).
# Offline on the node: weights are read from ~/Data/weights/i2vgen-xl (HF_HUB_OFFLINE=1). The anyv2v env's python is called by its
# absolute path -- `conda activate` does not put it in front of streamgve on this account (~/.bash_profile) -- and the CUDA path
# variables exported for streamgve are dropped, so the cu118 wheels load their own libraries. --exclude=node52 only: it advertises
# gpu:8 but exposes no device to batch jobs.
#SBATCH --job-name=r65_smoke
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/r65_smoke_%j.out
#SBATCH --error=logs/r65_smoke_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT=~/Data/dataggen/outputs/five_bench/r65_anyv2v
PY=~/anaconda3/envs/anyv2v/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CPATH PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1

echo "[r65_smoke] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
PREFIX=$("$PY" -I -c 'import sys; print(sys.prefix)')
if [ "$PREFIX" != "$HOME/anaconda3/envs/anyv2v" ]; then
  echo "[r65_smoke] FATAL: python prefix $PREFIX is not the anyv2v env"; exit 1
fi

FAILED=0
for NAME in repro smoke; do
  echo "--- [r65_smoke] ${NAME} $(date '+%F %T') ---"
  "$PY" -I evaluation/r65_anyv2v_run.py --cases "evaluation/r65_cases_${NAME}.json" \
    --out_root "$OUT/${NAME}" --shard 0 --n_shards 1 \
    || { echo "[r65_smoke] FAILED ${NAME}: exit $?"; FAILED=$((FAILED + 1)); }
  echo "[r65_smoke] ${NAME} manifest:"
  cat "$OUT/${NAME}_manifest_shard0.csv" 2>&1
done

echo "[r65_smoke] failures: ${FAILED}"
[ "$FAILED" -eq 0 ]
