#!/bin/bash
# R59 stage 2a: draft-vs-source divergence d of se_max's 5-step VP draft on FiVE-Bench-Correct, on the 30x52 token grid (R31 temporal mean per VAE frame group).
#
# Usage: sbatch slurm_scripts/five_bench/r59_div.sh        (array 0-1, r59_se_chain.sh, afterok on r59_se_stage1.sh)
# ARRAY (one metric per task; r49_field.py reads se_{metric}):  0 se_dino (DINOv3 patch 1-cos)   1 se_lpips (LPIPS alex)
# Same r31_divergence.py call as r48_div.sh's se tasks, only the roots and the case list differ:
#   draft frames $D/r59_se_stage1/frames/n5_tgt09/step04 (the final x0), latents $D/r59_se_stage1/latents, out $D/r59_div/se_{metric}/edit{T}/{video}.npz
# (key `d` = raw divergence [F, 1560]; r49_field.py normalises it with the ABSOLUTE caps). 415 pairs of evaluation/r43_anchor_manifest_correct.json.
# The divergence reads the source videos and the draft frames only (no prompts), so there is no prompt guard here; the draft's is in r59_se_stage1.sh.
#SBATCH --job-name=r59_div
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --exclude=node52
#SBATCH --output=logs/r59_div_%A_%a.out
#SBATCH --error=logs/r59_div_%A_%a.err

CASES=evaluation/r43_anchor_manifest_correct.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
PY_FB=~/anaconda3/envs/five-bench/bin/python

NAMES=(se_dino se_lpips)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 1 ]; then
  echo "[r59_div] bad array id '${TID}' (expected 0-1)"; exit 1
fi
NAME=${NAMES[$TID]}; METRIC=${NAME##*_}
case "$METRIC" in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; esac
TARGET_ROOT=$D/r59_se_stage1/frames/n5_tgt09/step04
HOST_ARGS=(--latent_dir "$D/r59_se_stage1/latents")
OUT=$D/r59_div

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
echo "[r59_div] job=${SLURM_ARRAY_JOB_ID}_${TID} ${NAME} (arm ${ARM}) target=${TARGET_ROOT} expect per type: ${EXPECT[*]}"
echo "[r59_div] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2
if [ ! -d "$TARGET_ROOT" ]; then echo "[r59_div] FATAL: ${TARGET_ROOT} missing -- run r59_se_stage1.sh first"; exit 1; fi

FAILED=0
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r59_div] ${NAME} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
    --data_root "$DATA_ROOT" --target_root "$TARGET_ROOT" "${HOST_ARGS[@]}" --out_root "$OUT" \
    --temporal_reduce mean --no-save_native --seed 0 \
    || { echo "[r59_div] FAILED ${NAME} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guard: exact per-type counts
N_TOTAL=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$OUT/$NAME/edit${T}"/*.npz 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  if [ "$N" -ne "${EXPECT[$((T - 1))]}" ]; then
    echo "[r59_div] MISMATCH ${NAME} edit${T}: ${N} npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1))
  fi
done
echo "[r59_div] ${NAME}: ${N_TOTAL} npz total"
echo "[r59_div] failures: ${FAILED}"
exit $(( FAILED > 0 ))
