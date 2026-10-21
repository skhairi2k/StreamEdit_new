#!/bin/bash
# R49 stage 3 (Wan-Edit): one render config per array task. MODE2 subset (default) = the 18 clips of evaluation/r49_cases.json,
# full = the 419 pairs of evaluation/r7_anchor_manifest.json (R49 full stage, Jean Zay: r49_wan_full_chain.sh).
#
# Usage: sbatch --array=0-1 slurm_scripts/five_bench/r49_wan_render.sh arms   # max_s9, max_s5 (adaptive; r49_chain.sh)
#        sbatch --array=0-2 slurm_scripts/five_bench/r49_wan_render.sh m59    # matched59_s0.95/0.92/0.85 (r49_matched59.sh)
#        sbatch --array=1   slurm_scripts/five_bench/r49_wan_render.sh arms full   # max_s5 on 419 pairs (the full-stage arm)
#        sbatch --array=0-3 slurm_scripts/five_bench/r49_wan_render.sh m59 full    # matched59_s0.95/0.92/0.85/0.978 on 419 pairs
# Adaptive configs read start maps from $D/r49_field/wan_{arm}/edit{T}/{video}.npy (subset) or $D/r49_field_full/... (full), from r49_field.py.
# Output: $D/r49_wan_arms/{config}_{subset|full}/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a render that
# hits the time limit is simply resubmitted (alone, with its own --array index).
#SBATCH --job-name=r49_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_wan_render_%A_%a.out
#SBATCH --error=logs/r49_wan_render_%A_%a.err

MODE=${1:-}
case "$MODE" in
  arms) CONFIGS=(max_s9 max_s5) ;;
  m59)  CONFIGS=(matched59_s0.95 matched59_s0.92 matched59_s0.85 matched59_s0.978) ;;   # index 3 only used by the full stage
  *) echo "[r49_wan_render] usage: r49_wan_render.sh arms|m59 [subset|full]"; exit 1 ;;
esac
PYMODE=${2:-subset}
case "$PYMODE" in
  subset) CASES=evaluation/r49_cases.json;          FIELD=r49_field ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; FIELD=r49_field_full ;;
  *) echo "[r49_wan_render] usage: r49_wan_render.sh arms|m59 [subset|full]"; exit 1 ;;
esac
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r49_wan_arms
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ]; then echo "[r49_wan_render] bad array id '${TID}'"; exit 1; fi
if [ "$TID" -ge "${#CONFIGS[@]}" ]; then echo "[r49_wan_render] array id ${TID} >= ${#CONFIGS[@]} configs -- nothing to do"; exit 0; fi
CONFIG=${CONFIGS[$TID]}

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r49_wan_render] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE}/${PYMODE} config=${CONFIG} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r49_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
if [ "$MODE" = arms ]; then      # preflight: the start maps must exist for every clip
  for T in 1 2 3 4 5 6; do
    N=$(ls "$D/${FIELD}/wan_${CONFIG}/edit${T}"/*.npy 2>/dev/null | wc -l)
    [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r49_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run r49_field.sh first"; FAILED=$((FAILED + 1)); }
  done
  [ "$FAILED" -ne 0 ] && exit 1
fi

$PY_FB evaluation/r49_wan_render.py --config "$CONFIG" --mode "$PYMODE" --cases "$CASES" --out_root "$OUT" --field_root "$D/$FIELD" \
  || { echo "[r49_wan_render] r49_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render under this config's own {config}_{PYMODE} dir
RDIR="$OUT/${CONFIG}_${PYMODE}"
MISSING=$($PY_FB evaluation/r48_units.py --mode "$PYMODE" --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r49_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r49_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS)) dirs total (expect ${TOTAL} case clips)"
echo "[r49_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
