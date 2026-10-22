#!/bin/bash
# R51 stage 3 (Wan-Edit, Jean Zay): the no-support arm A4 (nosupp_s5) on the 419 pairs -- same renderer and settings as R49's
# max_s5 (r49_wan_render.py: N = 50, loop start 5, per-token start map), only the start maps differ.
#
# Usage: sbatch slurm_scripts/five_bench/r51_wan_render.sh      (r51_wan_chain.sh, afterok on r51_field.sh wan)
# Input : $D/r51_field_full/wan_nosupp_s5/edit{T}/{video}.npy   (r51_field.py)
# Output: $D/r51_wan_arms/nosupp_s5_full/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a render that hits the
# time limit is simply resubmitted.
#SBATCH --job-name=r51_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r51_wan_render_%j.out
#SBATCH --error=logs/r51_wan_render_%j.err

CONFIG=nosupp_s5
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r7_anchor_manifest.json
FIELD=$D/r51_field_full
OUT=$D/r51_wan_arms
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r51_wan_render] job=${SLURM_JOB_ID:-local} config=${CONFIG} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r51_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the start maps must exist for every clip
FAILED=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD/wan_${CONFIG}/edit${T}"/*.npy 2>/dev/null | wc -l)
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r51_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run r51_field.sh wan first"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -ne 0 ] && exit 1

$PY_FB evaluation/r49_wan_render.py --config "$CONFIG" --mode full --cases "$CASES" --out_root "$OUT" --field_root "$FIELD" \
  || { echo "[r51_wan_render] r49_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render
RDIR="$OUT/${CONFIG}_full"
MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r51_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r51_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS)) dirs total (expect ${TOTAL} case clips)"
echo "[r51_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
