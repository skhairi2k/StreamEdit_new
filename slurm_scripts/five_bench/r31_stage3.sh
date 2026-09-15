#!/bin/bash
# R31 stage 3 -- the SPATIALLY-GATED render. Full 15-step rollout per clip with Eq. 4's
# scalar release exponent replaced by a CONTINUOUS PER-TOKEN field built offline by
# r31_rho_map.py from stage 2's divergence maps. One rollout per clip; step_dump rides
# along so the final AND every intermediate video come from the SAME run.
#
# ARRAY LAYOUT: ONE DIVERGENCE ARM PER TASK (same order as r31_stage2.sh /
# r31_eval.sh -- do not reorder one without the others), each looping all 6 edit types
# internally so the diffusion model loads once per arm.
#   0 lpips        1 dino_patch        2 normals        3 latent
# `depth` was dropped 2026-09-11 (see r31_stage2.sh) so this stays --array=0-3.
#
# EACH TASK FIRST BUILDS ITS OWN RHO FIELD (r31_rho_map.py), THEN RENDERS. Cheap and
# CPU-only (milliseconds per clip via vectorized bisection), so it is folded into this
# same job rather than a separate step -- no reason to burn a GPU allocation waiting on
# it. Preflight requires all 22 stage-2 npz for this arm to exist first.
#
# --step 15 IS PINNED, NOT stage 1's 7. r31_rho_map.py inverts the source budget on the
# 15-step t_next grid (R30's A_disc(2)=0.300, A_disc(50)=0.0021), and the overlay onto
# r26_tradeoff_figure.py's stored curve only holds because sampler/anchors/seed/cases are
# byte-identical to R26/R30. r31_stage3.py itself refuses a rho field built for another
# schedule (checked against the npz's own recorded step/flow_shift).
#
# --dump_steps all: the user asked for the final AND every mid-step video, and
# step_dump yields every intermediate from the SAME rollout -- 1 rollout + 15 VAE
# decodes per clip, NOT 15 separate rollouts.
#
# --time=08:00:00: 15 VAE decodes per clip on top of the rollout (R22's ~5.3 min/clip at
# a single decode, roughly doubled here), and normals-adjacent arms are no heavier at
# render time since stage 3 never touches Marigold -- it only consumes stage 2's `m`.
# --mem=64G matches every other five_bench render script (the partition default host-OOMs
# loading UMT5-XXL + the checkpoint, the R9 job-900404 failure).
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is the R20/R21 metric-crash fix,
# carried here defensively even though this stage renders rather than scores.
#
# RECREATED 2026-09-15 after this file was found deleted from disk by an external
# process (root cause unknown, no destructive command in any tracked shell history) --
# recreated verbatim from conversation context. Its job (993124/993164) already
# completed cleanly before deletion; the render output (r31_arms/) was untouched.
#SBATCH --job-name=r31_stage3
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r31_stage3_%A_%a.out
#SBATCH --error=logs/r31_stage3_%A_%a.err

# NOTE: no set -u -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at conda activate (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DIV_ROOT=/projects/dataggen/outputs/five_bench/r31_div
RHO_ROOT=/projects/dataggen/outputs/five_bench/r31_rho
OUT_ROOT=/projects/dataggen/outputs/five_bench/r31_arms
# Same anchor files R7/R10/R20/R21/R26/R30/R31-stage1 read -- held byte-identical.
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES=$REPO/evaluation/cases.json

# The four divergence arms, same order as r31_stage2.sh / r31_eval.sh.
ARMS=(lpips dino_patch normals latent)
STEP=15
FLOW_SHIFT=1.0
TAU_MIN=2
TAU_MAX=50

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r31_stage3] bad array id ${TID} (expected 0-3)"; exit 1
fi
ARM=${ARMS[$TID]}
METHOD="r31_${ARM}"

# Conda: streamgve -- this stage RENDERS (load_pipe / rollout_inference), unlike stage 2
# which scores in five-bench. .bashrc auto-activates streamgve (the R2 env bug), so the
# explicit deactivate/activate pair is kept even though it looks like a no-op here.
cd
source .bashrc
conda deactivate
conda activate streamgve

cd "$REPO"
mkdir -p logs "$RHO_ROOT" "$OUT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r31_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} method=${METHOD}"
echo "[r31_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4

# ---- preflight: this arm's stage-2 divergence npz must be complete BEFORE building the
# rho field or loading the model. r31_stage2.sh's array tasks finish out of order (e.g.
# normals -- Marigold ensemble_size 5 -- is far slower than latent), so a partial
# arm here would silently route a short/degenerate field.
N_DIV=$(ls "$DIV_ROOT/$ARM"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r31_stage3] stage-2 npz for ${ARM}: ${N_DIV} (expect 22)"
if [ "$N_DIV" -ne 22 ]; then
  echo "[r31_stage3] FATAL: ${ARM} stage-2 output incomplete (${N_DIV}/22) under $DIV_ROOT/$ARM"
  echo "[r31_stage3]   -- wait for r31_stage2.sh's array task for this arm to finish."
  exit 1
fi

# ---- build this arm's exponent field, budget-linear per token. Cheap and CPU-only
# (vectorized bisection, milliseconds per clip); folded into this job so a GPU
# allocation is never held idle waiting on it.
echo "[r31_stage3] building rho field: ${DIV_ROOT}/${ARM} -> ${RHO_ROOT}/${ARM}"
python evaluation/r31_rho_map.py \
  --div_root "$DIV_ROOT/$ARM" \
  -o "$RHO_ROOT/$ARM" \
  --tau_min "$TAU_MIN" --tau_max "$TAU_MAX" \
  --step "$STEP" --flow_shift "$FLOW_SHIFT"
RC=$?
N_RHO=$(ls "$RHO_ROOT/$ARM"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r31_stage3] rho field: ${N_RHO} npz (expect 22) rc=${RC}"
if [ "$RC" -ne 0 ] || [ "$N_RHO" -ne 22 ]; then
  echo "[r31_stage3] FATAL: rho field build failed or incomplete for ${ARM}"
  exit 1
fi

# ---- render all 6 edit types, model loaded once per arm. r31_stage3.py's own
# preflight re-validates every rho npz (shape, frame count, schedule) before the model
# loads, so a bad field is still caught before a ~20-minute load, not just here.
FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r31_stage3] ${METHOD} edit${T} ---"
  python evaluation/r31_stage3.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --rho_dir "$RHO_ROOT/$ARM" \
    --cases_json "$CASES" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps all \
    --step "$STEP" \
    --flow_shift "$FLOW_SHIFT" \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r31_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

LOG=logs/r31_stage3_${SLURM_ARRAY_JOB_ID}_${TID}.out
N_GATE=$(grep -c 'GATE1 PASS' "$LOG" 2>/dev/null)
N_STEP14=$(ls -d "$OUT_ROOT/$METHOD/step14"/edit*/*/ 2>/dev/null | wc -l)
N_STEP00=$(ls -d "$OUT_ROOT/$METHOD/step00"/edit*/*/ 2>/dev/null | wc -l)
echo "[r31_stage3] ${METHOD}: GATE1=${N_GATE} (expect 22) step14 dirs=${N_STEP14} step00 dirs=${N_STEP00} (expect 22 each)"

N_BAD=$(cat "$OUT_ROOT/$METHOD"/_manifest_edit*.csv 2>/dev/null \
       | awk -F, 'NR==1 || $3!="ok"' | awk -F, '$3!="status" && $3!="ok"' | wc -l)
echo "[r31_stage3] ${METHOD}: non-ok manifest rows: ${N_BAD}"
if [ "$N_BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r31_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
