#!/bin/bash
# R62 stage 3 (Wan-Edit on FiVE-Bench-Correct, ON JEAN ZAY after jz/r50_port.sh): R51's no-support arm A4 (nosupp_s5) on the pairs whose Wan-Edit
# inputs changed in Correct (the 190 pairs of evaluation/r58_cases_maxs5.json). Every other pair keeps R51's A4 render (r62_assemble.py links it).
# Same renderer and settings as R49's max_s5 / R51's A4 (r49_wan_render.py: N = 50, loop start 5, per-token start map), only the start maps differ.
#
# Usage: sbatch slurm_scripts/five_bench/r62_wan_render.sh      (r62_wan_chain.sh, afterok on r62_field.sh wan)
# Input : $D/r62_field/wan_nosupp_s5/edit{T}/{video}.npy   (r51_field.py, via r62_field.sh wan)
# Output: $D/r62_wan_arms/nosupp_s5_full/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a render that hits the time limit is
# simply resubmitted. NEVER overwrites a video: the output root must be $D/r62_wan_arms (any other root is refused).
#SBATCH --job-name=r62_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r62_wan_render_%j.out
#SBATCH --error=logs/r62_wan_render_%j.err

CONFIG=nosupp_s5
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r58_cases_maxs5.json
ANNO=evaluation/five_bench_correct
FIELD=$D/r62_field
OUT=$D/r62_wan_arms
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

# no-overwrite guard: the output root is the new R62 folder, never an R48 / R49 / R51 / R58 render folder
[ "$OUT" = "$D/r62_wan_arms" ] || { echo "[r62_wan_render] FATAL: refusing output root ${OUT} (only r62_wan_arms)"; exit 1; }

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

for F in "$CASES" "$ANNO/edit1_FiVE.json" evaluation/r49_wan_render.py; do
  [ -s "$F" ] || { echo "[r62_wan_render] FATAL: missing $F (git pull? python evaluation/r58_build_cases.py?)"; exit 1; }
done

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r62_wan_render] job=${SLURM_JOB_ID:-local} config=${CONFIG} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) -> ${OUT}"
echo "[r62_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the start maps must exist for every clip
FAILED=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD/wan_${CONFIG}/edit${T}"/*.npy 2>/dev/null | wc -l)
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r62_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run r62_field.sh wan first"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -ne 0 ] && exit 1

$PY_FB evaluation/r49_wan_render.py --config "$CONFIG" --mode full --cases "$CASES" --anno_dir "$ANNO" --out_root "$OUT" --field_root "$FIELD" \
  || { echo "[r62_wan_render] r49_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render
RDIR="$OUT/${CONFIG}_full"
MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r62_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r62_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS)) dirs total (expect ${TOTAL} case clips)"
echo "[r62_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
