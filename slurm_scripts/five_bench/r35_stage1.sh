#!/bin/bash
# R35 stage 1 -- the divergence-measurement TARGETS, rendered on the WHOLE 419-pair
# FiVE-Bench, in TWO alternative gating regimes.
#
# ARRAY LAYOUT: one REGIME per task (the model loads once per task, then loops all 6
# edit types internally -- unlike r31_stage1.sh, which arrays over edit type because it
# is scoped to the 22-clip --cases_json subset).
#   task 0  ->  unblended  (--blend_sched zero,   s(p)=0 for every step, Q/K blend OFF)
#   task 1  ->  first2     (--blend_sched first2, s(p)=1 for steps 0-1, then 0)
# No --cases_json is passed to either task: absent = full bench via run_fivebench's own
# edit{T}_FiVE.json, exactly as r31_stage1.py already supports.
#
# STEP COUNT: 15, matching stage 3 -- deliberately NOT R31's 7 (2026-09-15 recipe).
# R31 measured a 7-step rollout's final x0; R35 measures a 15-step rollout's final x0,
# so --dump_steps 14 --measure_index 14 (the LAST index, index 14 of 0..14) is the
# fully-denoised target, matching what r31_stage1.py's own docstring calls out as the
# ONLY regime its image-space arms (LPIPS, DINO) are calibrated for -- --measure_index 6
# (the script's default, built for --step 7) would silently yield a MID-TRAJECTORY x0 at
# --step 15 and the script only warns, it does not refuse.
#
# ⚠️ MASK-INJECTION CAVEAT, carried forward from R31 and roughly DOUBLED here. Both the
# grounding-mask union and the background source-KV injection gate on len//2 of the
# denoising_step_list:
#   R31 (7 steps):  gate at index 3, measured at index 6 -> 3 steps after injection.
#   R35 (15 steps): gate at index 7, measured at index 14 -> 7 steps after injection.
# So the measured target here has had roughly twice as many post-injection steps to
# suppress target-vs-source difference in the BACKGROUND specifically. This applies
# IDENTICALLY to both regimes and both arms -- it cannot explain a DINO-vs-LPIPS or
# unblended-vs-first2 difference, only inflate the apparent localisation of all four in
# absolute terms. Unaffected by --blend_sched: mask gathering and bg source-KV injection
# are channels separate from the Q/K blend the schedule controls, and stay on in both
# regimes (r31_stage1.py's own "WHAT 'UNBLENDED' MEANS HERE" section).
#
# Sampler settings otherwise held IDENTICAL to r31_stage3.sh's render config -- seed 0,
# flow_shift 1.0, fg_boost_factor 4, rollout_chunk_size 21, vp + anchors -- so these
# targets and the stage-3 renders sit on the same schedule.
#
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint.
# --exclude: the proven post-2026-09-22 node set (R33/R34), kept defensively.
# --time=20:00:00: 419 pairs at ~1-2 min/clip (R31's 7-step rate) plus roughly 2x work
# per clip at 15 steps: generous headroom over the ~14-28h implied, plus multi-window
# clips and queue variance -- re-check against the first real run's wall clock.
#SBATCH --job-name=r35_stage1
# L40S ONLY (2026-09-24). The first submission (job 1007492, L40S,A100) landed both tasks on
# 40 GB A100s; 0032_mermaid then hit CUDA OOM on the A100-PCIe task at 39.2/39.5 GiB. Not a
# length effect (longer 81-frame clips passed; the SXM4 task rendered mermaid fine) -- the
# 40 GB card simply sits too close to the ceiling. Cancelled after ~1 h and resubmitted here.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r35_stage1_%A_%a.out
#SBATCH --error=logs/r35_stage1_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors

REGIMES=(unblended first2)
SCHEDS=(zero first2)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 1 ]; then
  echo "[r35_stage1] bad array id ${TID} (expected 0-1)"; exit 1
fi
REGIME=${REGIMES[$TID]}
SCHED=${SCHEDS[$TID]}
METHOD="$REGIME"

# Frames -> r35_stage1/{regime}/step{NN}/edit{T}/{video}/*.png
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r35_stage1
# Latents -> r35_latents/{regime}/edit{T}/{video}.npz -- REGIME folded into the path
# (not the filename) so the two regimes' latents can never collide.
LATENT_DIR=~/Data/dataggen/outputs/five_bench/r35_latents/$REGIME

STEPS=15
# 14 = the last index of a 15-step rollout -- the fully denoised render AND the target
# stage 2 measures. One index => one VAE decode per clip.
DUMP_STEPS=14
MEASURE_INDEX=14

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT" "$LATENT_DIR"

# The R20/R21 metric-crash fix, which r35_stage3.sh already sets and this script lacked:
# lets the caching allocator grow segments instead of fragmenting near the memory ceiling.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r35_stage1] job=${SLURM_ARRAY_JOB_ID}_${TID} regime=${REGIME} blend_sched=${SCHED}"
echo "[r35_stage1] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r35_stage1] steps=${STEPS} dump_steps=${DUMP_STEPS} measure_index=${MEASURE_INDEX} (fully denoised, final x0)"
echo "[r35_stage1] no --cases_json -- full bench via run_fivebench's own edit{T}_FiVE.json"
echo "[r35_stage1] frames -> ${OUT_ROOT}/${METHOD}   latents -> ${LATENT_DIR}/edit{T}"
nvidia-smi -L 2>&1 | head -2

# r31_stage1.py's own startup log resolves --blend_sched through the SAME function the
# bridge calls and echoes the per-step blender_rate -- for regime=unblended it must read
# all 1s, for regime=first2 it must read [0,0,1,1,...,1]. Checked per T at wait-stage1,
# not duplicated here.
FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r35_stage1] ${REGIME} / edit${T} ---"
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
    || { echo "[r35_stage1] FAILED ${REGIME} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: 419 latents for THIS regime, with the exact per-type breakdown.
# A short-by-one count here is exactly what the cases_r35_fullbench.json / manifest
# episode was about catching -- one silently-dropped clip still looks like "it ran".
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
N_TOTAL=0
BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r35_stage1] ${REGIME} edit${T}: ${N} latents -- ${STATUS}"
done
echo "[r35_stage1] ${REGIME}: ${N_TOTAL} latents total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r35_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
