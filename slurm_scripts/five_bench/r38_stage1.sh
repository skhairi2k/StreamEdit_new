#!/bin/bash
# R38 stage 1 -- R35's divergence-measurement TARGETS, re-rendered as a CHEAP DRAFT
# (N=5 denoising steps instead of 15) on the WHOLE 419-pair FiVE-Bench, in two regimes.
#
# ARRAY LAYOUT: one RUN (= N x regime) per task; the model loads once per task, then
# loops all 6 edit types (as r35_stage1.sh).
#   task 0  ->  n5_unblended  (--blend_sched zero,   N=5, Q/K blend OFF every step)
#   task 1  ->  n5_tgt09      (--blend_sched tgt0.9, N=5, anchored iff input t > 0.9)
# RUNS / SCHEDS / NSTEPS are a TABLE so the conditional N=7 follow-up (plan: run only if
# N=5 is clearly worse than N=15) is two added rows + an --array bump, not a rewrite.
# No --cases_json: absent = full bench via run_fivebench's own edit{T}_FiVE.json.
#
# THE N=15 POINTS ARE NOT RE-RENDERED. R38 smoke gate G1 (job 1016420) showed tgt0.9 at
# N=15 is byte-identical to R35's first2, and G2 that zero is unchanged, so R35's
# r35_stage1/{unblended,first2} trees ARE R38's N=15 points.
#
# GRID: np.arange(1000, 0, -200) = 1000, 800, 600, 400, 200 (warp is the identity at
# flow_shift 1.0). tgt0.9 anchors ONLY the t=1.0 call here (== first1, smoke gate G3) --
# a call whose current-chunk input is the same pure noise in both branches, so this
# regime may land close to unblended. That is what step `regime-gap` / `gate-lowN`
# measure before anything heavier runs on it.
#
# MEASURED LATENT: --dump_steps N-1 --measure_index N-1 = index 4, the final x0 (frames
# under step04/), the same "fully denoised draft" convention as R35's step14. Both
# injection gates are len//2 = 2: mask union at index 2, bg source-KV at indices 3-4.
#
# Everything else held IDENTICAL to r35_stage1.sh (seed 0, flow_shift 1.0, fg_boost 4,
# blend_power 2 -- inert under a named schedule --, rollout_chunk_size 21, vp + anchors).
#
# --time=20:00:00: R35's 15-step budget kept as headroom; a 5-step task should take
# roughly a third of R35's stage-1 wall clock plus the per-clip fixed cost (model
# context forwards, VAE encode/decode) -- re-check against this run's wall clock.
#SBATCH --job-name=r38_stage1
# L40S ONLY: R35's 1007492 OOM'd a 40 GB A100 on 0032_mermaid at 39.2/39.5 GiB.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r38_stage1_%A_%a.out
#SBATCH --error=logs/r38_stage1_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors

RUNS=(n5_unblended n5_tgt09)
SCHEDS=(zero tgt0.9)
NSTEPS=(5 5)
TID=${SLURM_ARRAY_TASK_ID}
MAX_TID=$(( ${#RUNS[@]} - 1 ))
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt "$MAX_TID" ]; then
  echo "[r38_stage1] bad array id ${TID} (expected 0-${MAX_TID})"; exit 1
fi
RUN=${RUNS[$TID]}
SCHED=${SCHEDS[$TID]}
STEPS=${NSTEPS[$TID]}
METHOD="$RUN"

# Frames -> r38_stage1/{run}/step{NN}/edit{T}/{video}/*.png
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r38_stage1
# Latents -> r38_latents/{run}/edit{T}/{video}.npz -- RUN folded into the path.
LATENT_DIR=~/Data/dataggen/outputs/five_bench/r38_latents/$RUN

# The last index of an N-step rollout -- the fully denoised draft AND the target stage 2
# measures. One index => one VAE decode per clip.
DUMP_STEPS=$((STEPS - 1))
MEASURE_INDEX=$((STEPS - 1))

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT" "$LATENT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r38_stage1] job=${SLURM_ARRAY_JOB_ID}_${TID} run=${RUN} blend_sched=${SCHED} steps=${STEPS}"
echo "[r38_stage1] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r38_stage1] dump_steps=${DUMP_STEPS} measure_index=${MEASURE_INDEX} (fully denoised, final x0)"
echo "[r38_stage1] no --cases_json -- full bench via run_fivebench's own edit{T}_FiVE.json"
echo "[r38_stage1] frames -> ${OUT_ROOT}/${METHOD}   latents -> ${LATENT_DIR}/edit{T}"
nvidia-smi -L 2>&1 | head -2

# r31_stage1.py echoes the per-step blender_rate resolved on the pipeline's real warped
# grid: n5_unblended must read [1,1,1,1,1], n5_tgt09 [0,1,1,1,1]. Checked at wait-stage1.
FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r38_stage1] ${RUN} / edit${T} ---"
  python evaluation/r31_stage1.py \
    --edit_type "$T" \
    --method "$METHOD" \
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
    --blend_sched "$SCHED" \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r38_stage1] FAILED ${RUN} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: 419 latents for THIS run, with the exact per-type breakdown.
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
N_TOTAL=0
BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r38_stage1] ${RUN} edit${T}: ${N} latents -- ${STATUS}"
done
echo "[r38_stage1] ${RUN}: ${N_TOTAL} latents total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r38_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
