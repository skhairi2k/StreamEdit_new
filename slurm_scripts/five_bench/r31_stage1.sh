#!/bin/bash
# R31 stage 1 -- the UNBLENDED target render + the latents stage 2 measures on.
#
# One 7-STEP rollout per clip with the StreamGVE Q/K blend switched off
# (--blend_sched zero => s(p)=0 => blender_rate=1.0, all three blend sites become the
# identity). R22's step_dump hook rides along and index 6 -- the LAST step -- is both the
# measured target and the final render, so one VAE decode per clip covers both.
#
# CHANGED 2026-09-15 (user's call). This was a 15-step rollout measured at its MIDDLE x0
# prediction (index 7 == len//2, the pre-injection boundary). It is now a real --step 7
# run measured at its FINAL x0, so stage 2 reads a fully denoised image: LPIPS / DINOv3 /
# Marigold expect a natural image, and on a half-denoised x0 they partly measure
# denoising artefacts rather than the edit.
#
# ⚠️ NOT A TRUNCATION. denoising_step_list = np.arange(1000, 0, -1000/step)
# (inference_edit_streamedit.py:68), so 7 steps give [1000,857,714,571,428,285,142] -- a
# coarser grid, NOT a subset of the 15-step one. Different trajectory, not a prefix.
#
# ⚠️ NOT PRE-INJECTION. Both gates are len(denoising_step_list)//2, which is 3 here:
#   * the grounding-mask union is injected into the KV cache at
#     edit_causal_inference.py:896 -- at index 3;
#   * the bg source-KV injection is gated `current_timestep_index > total//2`
#     (causal_model.py:492), so it fires at indices 4, 5 and 6.
# The measured index 6 is thus THREE STEPS AFTER both channels engage and has already
# been shaped by the R26 grounding-mask union. Accepted deliberately, but check-stage2
# must weigh it: the injection suppresses target-vs-source difference in the BACKGROUND
# specifically, so a map may localise partly BECAUSE the mask made it localise. The npz
# records this as measure_kind=post_injection__final_render.
#
# WHAT IS *NOT* DISABLED. The bg source-KV injection and the mask gathering are separate
# channels and stay ON -- deliberately (see the R10 comment in causal_model.py: without
# the injection the target branch decouples from the source video entirely). Stage 3 runs
# in that same regime, which is what makes this divergence transfer.
#
# OUTPUTS ARE SEPARATE FROM THE 15-STEP RUN. METHOD/LATENT_DIR carry an _s7 suffix so
# job 989942's 22 npz and its step07/step14 frames survive untouched for comparison.
#
# ARRAY LAYOUT: one EDIT TYPE per task (the model loads once per task).
#   task 0..5  ->  edit_type 1..6
# evaluation/cases.json spans all six: 1/13/2/2/3/1 = 22 clips. r31_stage1.py filters via
# --cases_json and fails loudly if a named clip is absent from the edit{T} json.
#
# Sampler settings other than --step are held BYTE-IDENTICAL to R26/R30 -- seed 0,
# flow_shift 1.0, fg_boost_factor 4, rollout_chunk_size 21, vp anchoring off the same
# anchor dir. The STEP COUNT is deliberately 7 here and ONLY here: stage 3 stays at 15,
# which is what keeps R30's A_disc endpoints (A_disc(2)=0.300, A_disc(50)=0.0021) valid
# without recalibration and keeps the overlay on R26's stored trade-off curve honest.
# Stage 1 is never compared to an R26 arm, so its step count is free.
#
# Cost: roughly HALF the old run -- 7 denoising steps instead of 15, and 1 VAE decode
# instead of 2. ~1 min/clip => the 4 h budget is now large headroom, kept for the
# multi-window clips (0034_cows, and the 8 vp clips that split because
# independent_first_frame shortens window 1).
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r31_stage1
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/r31_stage1_%A_%a.out
#SBATCH --error=logs/r31_stage1_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r31_stage1
# _s7 so the 15-step run's latents (job 989942) are not overwritten.
LATENT_DIR=/projects/dataggen/outputs/five_bench/r31_latents_s7
# Same anchor files R7/R10/R20/R21/R26/R30 read -- all 6 edit types verified populated.
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES=$REPO/evaluation/cases.json

# _s7 so frames land beside, not on top of, the 15-step run's r31_unblended/.
METHOD=r31_unblended_s7
STEPS=7
# 6 = the last index, i.e. the fully denoised render AND the target stage 2 measures.
# One index => one VAE decode per clip.
DUMP_STEPS=6
MEASURE_INDEX=6

T=$(( SLURM_ARRAY_TASK_ID + 1 ))
if [ "$T" -lt 1 ] || [ "$T" -gt 6 ]; then
  echo "[r31_stage1] bad array id ${SLURM_ARRAY_TASK_ID} (expected 0-5)"; exit 1
fi

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT" "$LATENT_DIR"

echo "[r31_stage1] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} edit_type=${T}"
echo "[r31_stage1] method=${METHOD} steps=${STEPS} blend_sched=zero (Q/K blend OFF)"
echo "[r31_stage1] measured index ${MEASURE_INDEX} is POST-injection (len//2=$(( STEPS / 2 )))"
echo "[r31_stage1] dump_steps=${DUMP_STEPS} measure_index=${MEASURE_INDEX}"
echo "[r31_stage1] frames -> ${OUT_ROOT}/${METHOD}   latents -> ${LATENT_DIR}/edit${T}"

# r31_stage1.py pins --blend_sched zero internally; it is not exposed as a flag, so a
# "stage 1" output can never be silently blended.
python evaluation/r31_stage1.py \
  --edit_type "$T" \
  --method "$METHOD" \
  --cases_json "$CASES" \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --latent_dir "$LATENT_DIR" \
  --vp_mode vp \
  --first_frame_edit_dir "$ANCHOR_ROOT" \
  --dump_steps "$DUMP_STEPS" \
  --measure_index "$MEASURE_INDEX" \
  --step "$STEPS" \
  --flow_shift 1.0 \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --rollout_chunk_size 21 \
  --seed 0
RC=$?

LOG=logs/r31_stage1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
N_NPZ=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
N_DIRS=$(ls -d "$OUT_ROOT/$METHOD"/step*/edit${T}/*/ 2>/dev/null | wc -l)
N_GATE=$(grep -c 'GATE1 PASS' "$LOG" 2>/dev/null)
echo "[r31_stage1] edit${T}: npz=${N_NPZ} frame_dirs=${N_DIRS} (expect npz x 1) GATE1=${N_GATE}"
echo "[r31_stage1] failures: $(( RC != 0 ))"
exit $RC
