#!/bin/bash
# R58 stage 1 (Wan-Edit on FiVE-Bench-Correct, ON JEAN ZAY after jz/r50_port.sh): 17-step draft from step 3 (sigma 0.958) + per-token
# cross-attention recording, on the 190 pairs of evaluation/r58_cases_maxs5.json (prompt OR trigger object changed in Correct; every
# other pair keeps its R49 draft-based render). Same settings as R49 (r48_wan_draft.py: 17 steps, shift 5, skip_timesteps 3, seed 0,
# text only, attention on the first 20 blocks at sigma in [0.55, 0.9]), prompts and trigger words from evaluation/five_bench_correct.
#
# Usage: sbatch slurm_scripts/five_bench/r58_wan_draft.sh        (r58_wan_chain.sh submits it)
#
# Never overwrites an R48/R49 draft: the output root is $D/r58_wan_draft only. RESUMABLE: clips whose attention npz exists are skipped.
# Outputs ($D = ~/Data/dataggen/outputs/five_bench):
#   $D/r58_wan_draft/attn/edit{T}/{video}.npz                       per-token attention
#   $D/r58_wan_draft/frames/step00/edit{T}/{video}/*.png            draft frames (41), for r58_div
#SBATCH --job-name=r58_wan_draft
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r58_wan_draft_%j.out
#SBATCH --error=logs/r58_wan_draft_%j.err

CASES=evaluation/r58_cases_maxs5.json
ANNO=evaluation/five_bench_correct
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r58_wan_draft
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
for F in "$CASES" "$ANNO/edit1_FiVE.json" evaluation/r48_wan_draft.py; do
  [ -s "$F" ] || { echo "[r58_wan_draft] FATAL: missing $F (git pull?)"; exit 1; }
done
mkdir -p logs "$OUT/attn" "$OUT/frames"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r58_wan_draft] job=${SLURM_JOB_ID} cases=${CASES} anno=${ANNO} expect per type: ${EXPECT[*]} (total ${TOTAL}) -> ${OUT}"
echo "[r58_wan_draft] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
$PY_FB evaluation/r48_wan_draft.py --cases "$CASES" --anno_dir "$ANNO" --out_attn "$OUT/attn" --out_frames "$OUT/frames" \
  || { echo "[r58_wan_draft] r48_wan_draft.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: exact per-type counts (a total alone once hid a dropped clip plus a duplicate)
N_ATTN=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$OUT/attn/edit${T}"/*.npz 2>/dev/null | wc -l); N_ATTN=$((N_ATTN + N))
  NF=$(ls -d "$OUT/frames/step00/edit${T}"/*/ 2>/dev/null | wc -l)
  if [ "$N" -ne "${EXPECT[$((T - 1))]}" ] || [ "$NF" -ne "${EXPECT[$((T - 1))]}" ]; then
    echo "[r58_wan_draft] MISMATCH edit${T}: ${N} attn npz / ${NF} frame dirs, expected ${EXPECT[$((T - 1))]}"
    FAILED=$((FAILED + 1))
  fi
done
echo "[r58_wan_draft] attn npz total: ${N_ATTN} (expect ${TOTAL})"
echo "[r58_wan_draft] failures: ${FAILED}"
exit $(( FAILED > 0 ))
