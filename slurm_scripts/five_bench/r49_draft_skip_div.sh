#!/bin/bash
# R49 side experiment: draft-vs-source divergence (DINOv3 patch features and LPIPS) for the Wan-Edit drafts started at step 2 and
# step 1 of the 17-step grid, on the 18 clips of evaluation/r49_cases.json. The step-3 draft already has its maps in r49_div.
# Usage: sbatch --dependency=afterok:<draft jobs> slurm_scripts/five_bench/r49_draft_skip_div.sh
# Input : $D/r49_wan_draft/frames/skip{2,1}/edit{T}/{video}/*.png   (r49_wan_draft_skip.sh)
# Output: $D/r49_draft_skip_div/skip{2,1}/wan_{dino,lpips}/edit{T}/{video}.npz   (key d [F, 1560], raw distance, unclipped)
# Same r31_divergence.py flags as r49_div.sh.
#SBATCH --job-name=r49_draft_skip_div
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_draft_skip_div_%j.out
#SBATCH --error=logs/r49_draft_skip_div_%j.err

CASES=evaluation/r49_cases.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
echo "[r49_draft_skip_div] job=${SLURM_JOB_ID:-local} node=$(hostname) expect per type: ${EXPECT[*]}"; nvidia-smi -L 2>&1 | head -2

FAILED=0
for SK in 2 1; do
  TARGET_ROOT=$D/r49_wan_draft/frames/skip${SK}
  OUT=$D/r49_draft_skip_div/skip${SK}
  mkdir -p "$OUT"
  if [ ! -d "$TARGET_ROOT" ]; then echo "[r49_draft_skip_div] FATAL: ${TARGET_ROOT} missing -- run r49_wan_draft_skip.sh ${SK} first"; FAILED=$((FAILED + 1)); continue; fi
  for NAME in wan_dino wan_lpips; do
    METRIC=${NAME##*_}
    case "$METRIC" in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; esac
    echo "=== [r49_draft_skip_div] skip ${SK} ${NAME} (arm ${ARM}) $(date '+%F %T')"
    for T in 1 2 3 4 5 6; do
      [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
      $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
        --data_root "$DATA_ROOT" --target_root "$TARGET_ROOT" --n_latent_frames 0 --out_root "$OUT" \
        --temporal_reduce mean --no-save_native --seed 0 \
        || { echo "[r49_draft_skip_div] FAILED skip ${SK} ${NAME} edit${T}"; FAILED=$((FAILED + 1)); }
      N=$(ls "$OUT/$NAME/edit${T}"/*.npz 2>/dev/null | wc -l)
      [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r49_draft_skip_div] MISMATCH skip ${SK} ${NAME} edit${T}: ${N} npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1)); }
    done
  done
done
echo "[r49_draft_skip_div] failures: ${FAILED}"
exit $(( FAILED > 0 ))
