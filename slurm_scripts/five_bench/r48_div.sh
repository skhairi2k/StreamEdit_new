#!/bin/bash
# R48 stage 2a: draft-vs-source divergence d, on the 30x52 token grid (R31 temporal mean per VAE frame group).
#
# Usage: sbatch slurm_scripts/five_bench/r48_div.sh subset   # evaluation/r48_cases.json (18 clips)
#        sbatch slurm_scripts/five_bench/r48_div.sh full     # r7_anchor_manifest.json (419 pairs)
#
# ARRAY (one host x metric per task; r48_field.py reads {host}_{metric}):
#   0 se_dino   1 se_lpips   2 wan_dino   3 wan_lpips
#   se : draft frames $D/r48_se_stage1/frames/n5_tgt09/step04, latents $D/r48_se_stage1/latents  (18 latent / 69 pixel frames)
#   wan: draft frames $D/r48_wan_draft/frames/step00, no latents -> --n_latent_frames 0 (auto)   (11 latent / 41 pixel frames; fewer for the 7 short videos)
# Output: $D/r48_div/{host}_{metric}/edit{T}/{video}.npz  (key `d` = raw divergence [F, 1560]; r48_field normalises it
# with the ABSOLUTE div_max, so the per-clip min/max `m` stored alongside is not used).
# Needs the two stage-1 jobs (r48_se_stage1.sh, r48_wan_draft.sh) finished for the same MODE.
#SBATCH --job-name=r48_div
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_div_%A_%a.out
#SBATCH --error=logs/r48_div_%A_%a.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json ;;
  full)   CASES=evaluation/r7_anchor_manifest.json ;;
  *) echo "[r48_div] usage: r48_div.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
PY_FB=~/anaconda3/envs/five-bench/bin/python

NAMES=(se_dino se_lpips wan_dino wan_lpips)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r48_div] bad array id '${TID}' (expected 0-3)"; exit 1
fi
NAME=${NAMES[$TID]}; HOST=${NAME%%_*}; METRIC=${NAME##*_}
case "$METRIC" in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; esac
if [ "$HOST" = se ]; then
  TARGET_ROOT=$D/r48_se_stage1/frames/n5_tgt09/step04
  HOST_ARGS=(--latent_dir "$D/r48_se_stage1/latents")
else
  TARGET_ROOT=$D/r48_wan_draft/frames/step00
  HOST_ARGS=(--n_latent_frames 0)      # 0 = per clip from the draft's frame count (11 for 41 frames, fewer for the 7 short videos)
fi
OUT=$D/r48_div

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
echo "[r48_div] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} ${NAME} (arm ${ARM}) target=${TARGET_ROOT} expect per type: ${EXPECT[*]}"
echo "[r48_div] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2
if [ ! -d "$TARGET_ROOT" ]; then echo "[r48_div] FATAL: ${TARGET_ROOT} missing -- run the stage-1 job for this host first"; exit 1; fi

FAILED=0
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r48_div] ${NAME} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
    --data_root "$DATA_ROOT" --target_root "$TARGET_ROOT" "${HOST_ARGS[@]}" --out_root "$OUT" \
    --temporal_reduce mean --no-save_native --seed 0 \
    || { echo "[r48_div] FAILED ${NAME} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guard: exact per-type counts
N_TOTAL=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$OUT/$NAME/edit${T}"/*.npz 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  if [ "$N" -ne "${EXPECT[$((T - 1))]}" ]; then
    echo "[r48_div] MISMATCH ${NAME} edit${T}: ${N} npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1))
  fi
done
echo "[r48_div] ${NAME}: ${N_TOTAL} npz total"
echo "[r48_div] failures: ${FAILED}"
exit $(( FAILED > 0 ))
