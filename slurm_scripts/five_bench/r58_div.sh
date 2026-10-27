#!/bin/bash
# R58 stage 2a (ON JEAN ZAY): draft-vs-source divergence d of the Wan drafts of r58_wan_draft.sh, on the 190 pairs of
# evaluation/r58_cases_maxs5.json (30x52 token grid, R31 temporal mean per VAE frame group). Clone of the Wan tasks of r48_div.sh.
#
# Usage: sbatch --array=0-1 slurm_scripts/five_bench/r58_div.sh        (r58_wan_chain.sh submits it, afterok the draft)
# ARRAY: 0 wan_dino (arm dino_patch)   1 wan_lpips (arm lpips)       r49_field.py reads {host}_{metric}
# Input : $D/r58_wan_draft/frames/step00 (no latents -> --n_latent_frames 0 = per clip from the frame count)
# Output: $D/r58_div/{wan_dino,wan_lpips}/edit{T}/{video}.npz  (key `d` = raw divergence [F, 1560])
# Never writes into $D/r48_div: the output root is $D/r58_div only.
#SBATCH --job-name=r58_div
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r58_div_%A_%a.out
#SBATCH --error=logs/r58_div_%A_%a.err

CASES=evaluation/r58_cases_maxs5.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

NAMES=(wan_dino wan_lpips)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 1 ]; then
  echo "[r58_div] bad array id '${TID}' (expected 0-1)"; exit 1
fi
NAME=${NAMES[$TID]}; METRIC=${NAME##*_}
case "$METRIC" in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; esac
TARGET_ROOT=$D/r58_wan_draft/frames/step00
OUT=$D/r58_div

cd "$REPO" || exit 1
[ -s "$CASES" ] || { echo "[r58_div] FATAL: missing $CASES (git pull?)"; exit 1; }
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
echo "[r58_div] job=${SLURM_ARRAY_JOB_ID}_${TID} ${NAME} (arm ${ARM}) target=${TARGET_ROOT} expect per type: ${EXPECT[*]}"
echo "[r58_div] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2
if [ ! -d "$TARGET_ROOT" ]; then echo "[r58_div] FATAL: ${TARGET_ROOT} missing -- run r58_wan_draft.sh first"; exit 1; fi

FAILED=0
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r58_div] ${NAME} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
    --data_root "$DATA_ROOT" --target_root "$TARGET_ROOT" --n_latent_frames 0 --out_root "$OUT" \
    --temporal_reduce mean --no-save_native --seed 0 \
    || { echo "[r58_div] FAILED ${NAME} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guard: exact per-type counts
N_TOTAL=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$OUT/$NAME/edit${T}"/*.npz 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  if [ "$N" -ne "${EXPECT[$((T - 1))]}" ]; then
    echo "[r58_div] MISMATCH ${NAME} edit${T}: ${N} npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1))
  fi
done
echo "[r58_div] ${NAME}: ${N_TOTAL} npz total"
echo "[r58_div] failures: ${FAILED}"
exit $(( FAILED > 0 ))
