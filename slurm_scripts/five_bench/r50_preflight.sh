#!/bin/bash
# R50 preflight (Jean Zay H100, one short job): everything the Wan-side R49 jobs need, on a compute node, offline, before any long run.
#
# Usage: sbatch slurm_scripts/five_bench/r50_preflight.sh                      (after jz/r50_setup_dirs.sh, r50_pip_overlay.sh, r50_port.sh)
#        sbatch --qos=qos_gpu_h100-dev slurm_scripts/five_bench/r50_preflight.sh   (the dev QOS starts sooner; a command-line option wins)
#   1. the node: GPU, the CPUs / memory Slurm granted, whether the network is cut
#   2. torch / CUDA / transformers through the interpreter the jobs use ($PY_FB)
#   3. evaluation/r50_preflight.py: DINOv3, LPIPS, CLIP, the FiVE harness (CLIPScore, PSNR, LPIPS, SSIM, DINO ViT-B/8, Qwen2.5-VL with
#      flash attention), NIQE, flash-attn -- every model loaded offline the way the real jobs load it
#   4. ONE real Wan-Edit draft (r48_wan_draft.py on evaluation/r50_preflight_case.json = 0076_A_rabbit, edit 2): Wan2.1 weights, the
#      17-step draft, the attention recorder. Its wall time is the per-clip cost to extrapolate the R49 Wan full stage from.
# Output: $D/r50_preflight/<jobid>/{attn,frames} (nothing is deleted; each run has its own folder)
# Ends with "[r50_preflight] failures: N" (0 = go ahead with jz-repro).
#SBATCH --job-name=r50_preflight
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r50_preflight_%j.out
#SBATCH --error=logs/r50_preflight_%j.err

CASES=evaluation/r50_preflight_case.json
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r50_preflight/${SLURM_JOB_ID:-local}
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT/attn" "$OUT/frames"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
FAILED=0
echo "[r50_preflight] job=${SLURM_JOB_ID:-local} node=$(hostname) $(date '+%F %T') out=${OUT}"

echo "=== [r50_preflight] 1. node"
nvidia-smi -L 2>&1 | head -2
echo "[r50_preflight] granted: cpus-per-task=${SLURM_CPUS_PER_TASK:-?}, cpus on node=${SLURM_CPUS_ON_NODE:-?}, memory limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '?')"
echo "[r50_preflight] interpreter: ${PY_FB}"
[ -x "$PY_FB" ] || { echo "[r50_preflight] FATAL: ${PY_FB} not executable -- run jz/r50_pip_overlay.sh and jz/r50_port.sh first"; echo "[r50_preflight] failures: 1"; exit 1; }

echo "=== [r50_preflight] 2. torch / CUDA through the job interpreter"
$PY_FB -c "import sys, torch, transformers; print('[r50_preflight] python', sys.version.split()[0], '| torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available(), '| transformers', transformers.__version__)" \
  || { echo "[r50_preflight] FAILED: the interpreter cannot import torch / transformers"; FAILED=$((FAILED + 1)); }

echo "=== [r50_preflight] 3. models, offline"
LOG_MODELS=logs/r50_preflight_${SLURM_JOB_ID:-local}.models.log
$PY_FB evaluation/r50_preflight.py 2>&1 | tee "$LOG_MODELS"
NMODEL=$(grep -oE 'model failures: [0-9]+' "$LOG_MODELS" | tail -1 | grep -oE '[0-9]+$')
[ -z "$NMODEL" ] && { echo "[r50_preflight] r50_preflight.py printed no summary (crashed before the checks?)"; NMODEL=1; }
FAILED=$((FAILED + NMODEL))

echo "=== [r50_preflight] 4. one Wan-Edit draft: $(python3 -c "import json; c=json.load(open('$CASES'))[0]; print(c['video_name'], 'edit', c['edit_type'])" 2>/dev/null)"
T0=$(date +%s)
$PY_FB evaluation/r48_wan_draft.py --cases "$CASES" --out_attn "$OUT/attn" --out_frames "$OUT/frames" \
  || { echo "[r50_preflight] r48_wan_draft.py reported failures"; FAILED=$((FAILED + 1)); }
SECS=$(( $(date +%s) - T0 ))
N_ATTN=$(ls "$OUT"/attn/edit*/*.npz 2>/dev/null | wc -l)
N_FRAMES=$(ls "$OUT"/frames/step00/edit*/*/*.png 2>/dev/null | wc -l)
echo "[r50_preflight] draft: ${N_ATTN} attention npz (expect 1), ${N_FRAMES} frames, ${SECS} s wall for one clip (model load included)"
{ [ "$N_ATTN" -ne 1 ] || [ "$N_FRAMES" -lt 9 ]; } && { echo "[r50_preflight] MISMATCH: the draft did not produce its outputs"; FAILED=$((FAILED + 1)); }

echo "[r50_preflight] failures: ${FAILED}"
exit $(( FAILED > 0 ))
