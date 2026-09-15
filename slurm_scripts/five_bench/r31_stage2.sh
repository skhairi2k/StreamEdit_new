#!/bin/bash
# R31 stage 2 -- per-token, per-frame divergence between the source and stage 1's
# UNBLENDED target, for all four definitions.
#
# ARRAY LAYOUT: ONE DIVERGENCE ARM PER TASK (the task's packaging requirement), each
# looping all 6 edit types internally so the arm's model loads once per task.
#   0 lpips        LPIPS(alex, spatial=True) per-pixel map            GPU
#   1 dino_patch   DINOv3 ViT-B/16, per-position 1-cos                GPU
#   2 normals      Marigold Normals, angular error in degrees         GPU
#   3 latent       mean_C |z_trg - z_src|, NO model at all            CPU-ish
#
# ⚠ A 5th arm, `depth`, was specced and DROPPED 2026-09-11: monocular relative depth needs
# an affine alignment, and R31 has no mask to fit it on -- it collapses once the edit
# exceeds ~13% of the frame, against a MEDIAN union coverage of 0.50 on these 22 clips.
# `normals` is a DIRECT estimator and needs no alignment, so it is unaffected.
# `latent` needs no network and no GPU but runs in the same array for uniformity; it
# finishes in seconds.
#
# THE TWO REDUCTIONS (see r31_divergence.py's docstring for the derivation):
#   spatial   480x832 -> 30x52 = 1560 = the pipeline's frame_seq_length.
#             lpips/normals by exact 16x16 area-mean; dino_patch needs NONE
#             (DINOv3 patch 16 divides 480 and 832 exactly, so its patch grid IS the
#             token grid); latent by an exact 2x2 pool, the DiT's own patchify.
#   temporal  the VAE's own grouping -- frame 0 alone, then blocks of 4 -- so 81 pixel
#             frames -> 21 latent frames. `latent` skips it, already per latent frame.
#
# HF_HUB_OFFLINE=1: compute nodes have no outbound network. DINOv3
# (facebook/dinov3-vitb16-pretrain-lvd1689m, gate accepted 2026-09-11) and Marigold must
# already be in ~/.cache/huggingface. Setting this makes a cache miss
# fail loudly here rather than hang reaching for the Hub.
#
# The 12-row grids (IMAGE half: source, target, each arm's NATIVE-resolution map;
# LATENT half: z_src, z_trg as PCA->RGB, and each arm's reduced 30x52 field) are built ONCE, by task 0 only, after every arm's npz exists --
# guarded below, because a page needs all four arms and the array tasks finish out of
# order. If task 0 runs first the guard simply skips and the grids are made by
# re-running `python evaluation/r31_div_grid.py ...` locally; it needs no GPU.
#
# Cost: 22 clips x ~81 frame-pairs per arm. normals is by far the heaviest (Marigold is a
# diffusion model at ensemble_size 5, i.e. 5 x 4 = 20 UNet passes per image, x2 images,
# x81 frames, x22 clips). 6 h is sized for that arm; the others finish far sooner.
#
# --mem=64G matches every other five_bench script (the partition default host-OOMs).
#SBATCH --job-name=r31_stage2
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r31_stage2_%A_%a.out
#SBATCH --error=logs/r31_stage2_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# 2026-09-15: stage 1 is now a 7-step run measured at its FINAL index, so the target
# lives under r31_unblended_s7/step06, not r31_unblended/step07. The 15-step run's
# outputs are still on disk under the un-suffixed names if a comparison is wanted.
TARGET_ROOT=/projects/dataggen/outputs/five_bench/r31_stage1/r31_unblended_s7/step06
LATENT_DIR=/projects/dataggen/outputs/five_bench/r31_latents_s7
DIV_ROOT=/projects/dataggen/outputs/five_bench/r31_div
CASES=$REPO/evaluation/cases.json
GRID_DIR=$REPO/evaluation/figures/r31_div_grids

EDIT_TYPES=(1 2 3 4 5 6)
ARMS=(lpips dino_patch normals latent)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$ARM" ]; then
  echo "[r31_stage2] bad array id ${SLURM_ARRAY_TASK_ID} (expected 0-3)"; exit 1
fi

# ⚠️ ENV IS `five-bench`, NOT `streamgve` -- FIXED 2026-09-15 after job 993071 lost two
# arms to it. `streamgve` is the RENDER env; it cannot run this stage:
#   * `lpips` is not installed there at all      -> ModuleNotFoundError
#   * Marigold needs diffusers/transformers that agree. streamgve has diffusers 0.31.0 +
#     transformers 5.12.0, and diffusers 0.31 imports FLAX_WEIGHTS_NAME from
#     transformers.utils, which transformers 5.x removed -> ImportError on
#     `MarigoldNormalsPipeline`, so the `normals` arm dies at load.
# `five-bench` (diffusers 0.36.0 + transformers 5.3.0 + lpips) runs ALL FOUR arms, DINOv3
# included -- verified per-env before this re-run. It is also the env R30's own divergence
# stage used (r30_stage1.sh:69), which is where this script should have inherited from.
# The R2 env bug still applies: .bashrc auto-activates streamgve and `conda activate
# five-bench` alone does not pop it, so the explicit `conda deactivate` is load-bearing.
cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench

cd "$REPO"
mkdir -p logs "$DIV_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the R20/R21 metric-crash fix
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r31_stage2] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} arm=${ARM}"
echo "[r31_stage2] target=${TARGET_ROOT}"
echo "[r31_stage2] latents=${LATENT_DIR}  ->  ${DIV_ROOT}/${ARM}"

# Preflight: stage 1 must have produced both the frames and the latents. Failing here
# costs seconds; failing after a model load costs minutes x 6 edit types.
if [ ! -d "$LATENT_DIR" ]; then
  echo "[r31_stage2] FATAL: ${LATENT_DIR} does not exist -- run /run-step R31 stage1 first"
  exit 1
fi
N_NPZ=$(ls "$LATENT_DIR"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r31_stage2] stage-1 latents found: ${N_NPZ} (expect 22)"
if [ "$N_NPZ" -eq 0 ]; then
  echo "[r31_stage2] FATAL: no stage-1 latents"; exit 1
fi

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "[r31_stage2] --- ${ARM} / edit${T} ---"
  python evaluation/r31_divergence.py \
    --edit_type "$T" \
    --arm "$ARM" \
    --cases "$CASES" \
    --data_root "$DATA_ROOT" \
    --target_root "$TARGET_ROOT" \
    --latent_dir "$LATENT_DIR" \
    --out_root "$DIV_ROOT" \
    --temporal_reduce mean \
    --save_native \
    --normals_steps 4 \
    --normals_ensemble 5 \
    --seed 0 || { echo "[r31_stage2] FAILED ${ARM} / edit${T}"; FAILED=$((FAILED+1)); }
done

N_OUT=$(ls "$DIV_ROOT/$ARM"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r31_stage2] arm=${ARM}: ${N_OUT} npz (expect 22) failures=${FAILED}"

# Grids need ALL four arms, and array tasks finish out of order -- so build them only if
# every arm is already complete. Whichever task finishes last satisfies this; if none
# does (a failure), run r31_div_grid.py locally once the gap is fixed.
TOTAL=$(ls "$DIV_ROOT"/*/edit*/*.npz 2>/dev/null | wc -l)
if [ "$TOTAL" -eq 88 ]; then
  echo "[r31_stage2] all 88 npz present (4 arms x 22) -> building the 12-row image/latent grids"
  python evaluation/r31_div_grid.py \
    --div_root "$DIV_ROOT" \
    --data_root "$DATA_ROOT" \
    --target_root "$TARGET_ROOT" \
    --latent_dir "$LATENT_DIR" \
    --cases "$CASES" \
    --out_dir "$GRID_DIR" || { echo "[r31_stage2] grid build FAILED"; FAILED=$((FAILED+1)); }
else
  echo "[r31_stage2] ${TOTAL}/88 npz present -- another arm is still running; grids skipped"
fi

exit $(( FAILED > 0 ))
