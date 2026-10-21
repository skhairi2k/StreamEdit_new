#!/bin/bash
# R49 stage 2a: draft-vs-source divergence maps (DINOv3 patch features and LPIPS) for both hosts, 18 clips.
#
# Usage: sbatch slurm_scripts/five_bench/r49_div.sh [HOSTS]    (normally submitted by r49_chain.sh, afterok on both drafts)
#   HOSTS: "se wan" (default, the lab cluster), se or wan -- only those hosts' names are computed and checked
#          (Jean Zay runs `wan` only: r50_repro_chain.sh)
# One GPU job loops the four names that r48_div.sh runs as array tasks 0-3 (se_dino, se_lpips, wan_dino, wan_lpips), with the
# same r31_divergence.py flags, so it takes one queue entry instead of four.
# Input : StreamEdit draft frames $D/r49_se_stage1/frames/n5_tgt09/step04 + latents; Wan-Edit draft frames $D/r49_wan_draft/frames/step00
# Output: $D/r49_div/{se,wan}_{dino,lpips}/edit{T}/{video}.npz   (key d [F, 1560], raw distance, unclipped)
#SBATCH --job-name=r49_div
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_div_%j.out
#SBATCH --error=logs/r49_div_%j.err

HOSTS=${1:-se wan}
read -r -a HOST_LIST <<< "$HOSTS"
for H in "${HOST_LIST[@]}"; do
  case "$H" in se|wan) ;; *) echo "[r49_div] usage: r49_div.sh [\"se wan\"|se|wan] -- bad host '$H'"; exit 1 ;; esac
done
CASES=evaluation/r49_cases.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r49_div
PY_FB=~/anaconda3/envs/five-bench/bin/python

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
echo "[r49_div] job=${SLURM_JOB_ID:-local} hosts=${HOST_LIST[*]} cases=${CASES} expect per type: ${EXPECT[*]}"
echo "[r49_div] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
for NAME in se_dino se_lpips wan_dino wan_lpips; do
  HOST=${NAME%%_*}; METRIC=${NAME##*_}
  [[ " ${HOST_LIST[*]} " == *" $HOST "* ]] || continue          # not a requested host
  case "$METRIC" in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; esac
  if [ "$HOST" = se ]; then
    TARGET_ROOT=$D/r49_se_stage1/frames/n5_tgt09/step04
    HOST_ARGS=(--latent_dir "$D/r49_se_stage1/latents")
  else
    TARGET_ROOT=$D/r49_wan_draft/frames/step00
    HOST_ARGS=(--n_latent_frames 0)      # 0 = per clip from the draft's frame count (11 for 41 frames, fewer for the short videos)
  fi
  echo "=== [r49_div] ${NAME} (arm ${ARM}) target=${TARGET_ROOT} $(date '+%F %T')"
  if [ ! -d "$TARGET_ROOT" ]; then echo "[r49_div] FATAL: ${TARGET_ROOT} missing -- run the draft job for this host first"; FAILED=$((FAILED + 1)); continue; fi
  for T in 1 2 3 4 5 6; do
    [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
    echo "--- [r49_div] ${NAME} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
    $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
      --data_root "$DATA_ROOT" --target_root "$TARGET_ROOT" "${HOST_ARGS[@]}" --out_root "$OUT" \
      --temporal_reduce mean --no-save_native --seed 0 \
      || { echo "[r49_div] FAILED ${NAME} edit${T}"; FAILED=$((FAILED + 1)); }
  done
  N_TOTAL=0
  for T in 1 2 3 4 5 6; do
    N=$(ls "$OUT/$NAME/edit${T}"/*.npz 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
    if [ "$N" -ne "${EXPECT[$((T - 1))]}" ]; then
      echo "[r49_div] MISMATCH ${NAME} edit${T}: ${N} npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1))
    fi
  done
  echo "[r49_div] ${NAME}: ${N_TOTAL} npz total"
done
echo "[r49_div] failures: ${FAILED}"
exit $(( FAILED > 0 ))
