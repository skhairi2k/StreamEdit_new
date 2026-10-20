#!/bin/bash
# R40 warp -- anchor-only, flow-warped divergence fields for the whole 419-pair FiVE-Bench.
#
# Replaces R35's stage 1 (draft render) AND stage 2 (per-frame divergence): per clip,
# r40_warp_divergence.py measures ONE DINO / LPIPS map between source frame 0 and the VP
# anchor, carries it to every frame with chained RAFT flow on the source video (warp in
# pixel space), then applies R31's token-grid + VAE-grouping reduction. Output is in R31
# stage-2 format, so r31_div_abs.py / r31_rho_map.py / r31_stage3.py run unmodified:
#   r40_div/{dino_warp,lpips_warp}/edit{T}/{video}.npz
#
# SINGLE JOB, NOT ARRAYED: the flow is the expensive part and both arms share it, so one
# pass writes both arms. Arraying by arm would compute every clip's flow twice.
#
# ⚠️ QUEUE ONLY AFTER R38/R39 RENDERS have left the queue (plan Decisions). Smoke gates
# G1-G3 (r40_smoke.sh) must have passed first.
#
# ⚠️ ENV IS `five-bench`, NOT `streamgve` -- `lpips` is not installed in streamgve (see
# r35_stage2.sh). RAFT weights are pre-cached in ~/.cache/torch/hub/checkpoints.
#
# --time=08:00:00: 2 RAFT-large calls per frame pair at 480x832 (~25k pairs bench-wide)
# plus one LPIPS + one DINOv3 forward per clip; the estimate is 3-4 h, x2 margin.
#SBATCH --job-name=r40_warp
#SBATCH --partition=L40S,A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r40_warp_%j.out
#SBATCH --error=logs/r40_warp_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors   # == r35_stage3.sh's --first_frame_edit_dir
DIV_ROOT=~/Data/dataggen/outputs/five_bench/r40_div
CASES=$REPO/evaluation/r7_anchor_manifest.json

ARMS=(dino_warp lpips_warp)
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench            # NOT streamgve -- see header note

cd "$REPO"
mkdir -p logs "$DIV_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r40_warp] job=${SLURM_JOB_ID} node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r40_warp] anchors=${ANCHOR_ROOT} -> ${DIV_ROOT}/{${ARMS[*]}}"
nvidia-smi -L 2>&1 | head -2

# Preflight: every anchor must exist BEFORE the models load.
N_ANCH=0
BAD_ANCH=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls "$ANCHOR_ROOT/edit${T}"/*.png 2>/dev/null | wc -l)
  N_ANCH=$((N_ANCH + N))
  [ "$N" -lt "${EXPECT[$T]}" ] && { echo "[r40_warp] edit${T}: ${N} anchors, expected >= ${EXPECT[$T]}"; BAD_ANCH=$((BAD_ANCH+1)); }
done
if [ "$BAD_ANCH" -ne 0 ]; then
  echo "[r40_warp] FATAL: anchors incomplete under ${ANCHOR_ROOT}"; exit 1
fi
echo "[r40_warp] preflight OK: ${N_ANCH} anchors"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r40_warp] edit${T} ---"
  python evaluation/r40_warp_divergence.py \
    --edit_type "$T" \
    --cases "$CASES" \
    --data_root "$DATA_ROOT" \
    --anchor_root "$ANCHOR_ROOT" \
    --out_root "$DIV_ROOT" \
    --temporal_reduce mean \
    --flow_iters 12 \
    --seed 0 \
    || { echo "[r40_warp] FAILED edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown for BOTH arms (a total-only count once
# hid a dropped clip and a duplicate -- the R35 anchor-manifest episode).
for A in "${ARMS[@]}"; do
  N_TOTAL=0
  BAD_TYPES=0
  for T in "${EDIT_TYPES[@]}"; do
    N=$(ls "$DIV_ROOT/$A/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + N))
    STATUS="ok"
    if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
    echo "[r40_warp] ${A} edit${T}: ${N} npz -- ${STATUS}"
  done
  echo "[r40_warp] ${A}: ${N_TOTAL} npz total (expect 419)"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi
done

echo "[r40_warp] failures: ${FAILED}"
exit $(( FAILED > 0 ))
