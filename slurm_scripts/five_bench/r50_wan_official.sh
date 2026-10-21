#!/bin/bash
# R50: the official Wan-Edit baseline, native_s0.92 (N = 50, skip 15), on the 18 clips of evaluation/r49_cases.json, for Jean Zay.
# It is the reference of the reproduction check: jz/r50_compare.py tests the paired difference  wan_max_s5 - wan_native_s0.92  against
# the lab cluster's confidence interval, so BOTH arms have to be rendered on the same platform. The lab already has this render
# (r48_wan_arms/native_s0.92_subset); Jean Zay does not (outputs are not shipped), so it is rendered here from scratch.
#
# Usage: sbatch slurm_scripts/five_bench/r50_wan_official.sh        (submitted by jz/r50_repro_chain.sh, no dependency)
# Output: $D/r48_wan_arms/native_s0.92_subset/edit{T}/{video}/*.png   (same folder name as the lab's, so r49_units.py finds it)
# RESUMABLE: finished clips are skipped, so a timed-out job is simply resubmitted.
# Same call as r49_newclip_baselines.sh (r48_wan_render.py --mode subset --cases ...), one config.
#SBATCH --job-name=r50_wan_official
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r50_wan_official_%j.out
#SBATCH --error=logs/r50_wan_official_%j.err

CONFIG=native_s0.92
CASES=evaluation/r49_cases.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r48_wan_arms
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
echo "[r50_wan_official] job=${SLURM_JOB_ID:-local} node=$(hostname) config=${CONFIG} cases=${CASES} (${N_CASES} clips)"
nvidia-smi -L 2>&1 | head -2

FAILED=0
T0=$(date +%s)
$PY_FB evaluation/r48_wan_render.py --config "$CONFIG" --mode subset --cases "$CASES" --out_root "$OUT" \
  || { echo "[r50_wan_official] r48_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }
echo "[r50_wan_official] render wall time: $(( $(date +%s) - T0 )) s"

# ---- post-run guard: every case clip has a complete render (r48_units.py counts frames the Wan-Edit way: right for this host)
RDIR="$OUT/${CONFIG}_subset"
MISSING=$($PY_FB evaluation/r48_units.py --mode subset --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r50_wan_official] ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r50_wan_official] ${RDIR##*/}: $((N_CASES - N_MISS)) complete dirs (expect ${N_CASES})"
echo "[r50_wan_official] failures: ${FAILED}"
exit $(( FAILED > 0 ))
