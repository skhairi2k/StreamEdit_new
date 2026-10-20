#!/bin/bash
# R48 stage 1 (Wan-Edit): 17-step draft from step 3 (sigma 0.958) + per-token cross-attention recording.
#
# Usage: sbatch slurm_scripts/five_bench/r48_wan_draft.sh subset   # evaluation/r48_cases.json (18 clips)
#        sbatch slurm_scripts/five_bench/r48_wan_draft.sh full     # r7_anchor_manifest.json (419 pairs)
#
# evaluation/r48_wan_draft.py: sample_steps 17, shift 5, skip_timesteps 3, seeded noise (base_seed 0), text only,
# attention on the first 20 blocks at the steps with sigma in [0.55, 0.9] (7..13), written in the same npz schema as
# StreamEdit's. RESUMABLE: clips whose attention npz already exists are skipped, so a timed-out full run is simply
# re-submitted.
#
# Outputs ($D = ~/Data/dataggen/outputs/five_bench):
#   $D/r48_wan_draft/attn/edit{T}/{video}.npz                       per-token attention
#   $D/r48_wan_draft/frames/step00/edit{T}/{video}/*.png            draft frames (41), for r48_div
# Explicit interpreter path: `conda activate` does not reliably change `python` here.
#SBATCH --job-name=r48_wan_draft
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_wan_draft_%j.out
#SBATCH --error=logs/r48_wan_draft_%j.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json ;;
  full)   CASES=evaluation/r7_anchor_manifest.json ;;
  *) echo "[r48_wan_draft] usage: r48_wan_draft.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r48_wan_draft
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
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
echo "[r48_wan_draft] job=${SLURM_JOB_ID} mode=${MODE} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r48_wan_draft] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
$PY_FB evaluation/r48_wan_draft.py --cases "$CASES" --out_attn "$OUT/attn" --out_frames "$OUT/frames" \
  || { echo "[r48_wan_draft] r48_wan_draft.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: exact per-type counts (a total alone once hid a dropped clip plus a duplicate)
N_ATTN=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$OUT/attn/edit${T}"/*.npz 2>/dev/null | wc -l); N_ATTN=$((N_ATTN + N))
  NF=$(ls -d "$OUT/frames/step00/edit${T}"/*/ 2>/dev/null | wc -l)
  if [ "$N" -ne "${EXPECT[$((T - 1))]}" ] || [ "$NF" -ne "${EXPECT[$((T - 1))]}" ]; then
    echo "[r48_wan_draft] MISMATCH edit${T}: ${N} attn npz / ${NF} frame dirs, expected ${EXPECT[$((T - 1))]}"
    FAILED=$((FAILED + 1))
  fi
done
echo "[r48_wan_draft] attn npz total: ${N_ATTN} (expect ${TOTAL})"
echo "[r48_wan_draft] failures: ${FAILED}"
exit $(( FAILED > 0 ))
