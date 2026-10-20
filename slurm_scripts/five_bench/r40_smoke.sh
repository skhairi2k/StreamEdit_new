#!/bin/bash
# R40 smoke -- anchor-only, flow-warped divergence on 2 clips, with gates G1-G3.
#
# 2 clips: 0001_bus (edit1, single window) and 0034_cows (edit2, 93 frames / 24 latent
# frames -- the long multi-window case, so the flow chain is ~90 links long).
# r40_warp_divergence.py --smoke writes both arms' npz under r40_smoke/ and prints:
#   G1  frame-0 identity (zero-flow warp == un-warped map, both arms)
#   G2  n_latent_frames == R35's r35_div/dino_first2 npz for the same clip
#   G3  flow direction: warping source frame 0 beats not warping on >= 90% of moving frames
# plus overlay PNGs of the warped LPIPS map (frames 0 / mid / last) in
# evaluation/figures/r40_smoke/. Exit code 0 only if every gate passes.
#
# ⚠️ ENV IS `five-bench`, NOT `streamgve` -- same reason as r35_stage2.sh: `lpips` is not
# installed in streamgve. RAFT weights are pre-cached in ~/.cache/torch/hub/checkpoints
# (raft_large_C_T_SKHT_V2-ff5fadd5.pth, downloaded 2026-10-01 from the login node).
#SBATCH --job-name=r40_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r40_smoke_%j.out
#SBATCH --error=logs/r40_smoke_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
SMOKE_ROOT=$D/r40_smoke

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench            # NOT streamgve -- see header note

cd "$REPO"
mkdir -p logs "$SMOKE_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r40_smoke] job=${SLURM_JOB_ID} node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

python evaluation/r40_warp_divergence.py --smoke \
  --case_ids edit1/0001_bus edit2/0034_cows \
  --out_root "$SMOKE_ROOT" \
  --png_dir evaluation/figures/r40_smoke \
  --temporal_reduce mean \
  --seed 0
RC=$?

echo "[r40_smoke] exit=${RC}"
ls -la evaluation/figures/r40_smoke/ "$SMOKE_ROOT"/*/edit*/ 2>/dev/null
exit $RC
