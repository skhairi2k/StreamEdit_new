#!/bin/bash
# R31 stage 3, EXPLORATORY VARIANT -- same spatially-gated render as r31_stage3.sh, but
# with r31_rho_map.py's 2026-09-15 --mapping linear_threshold instead of the calibrated
# budget_linear default. User-requested comparison: tau=0 below m=0.4 (source fully
# pinned), then LINEAR IN THE EXPONENT from 0 to 50 for m in [0.4, 1] -- the R26-style
# linear-in-exponent mapping the module's own docstring says hides 34-97% fictitious
# released mass, built to see what it looks like next to the calibrated curve, NOT to
# replace it. Forked rather than parameterizing r31_stage3.sh in place so the production
# script (still gating `verdict`) stays untouched.
#
# REUSES r31_div/ (stage 2's `m` fields) READ-ONLY -- the divergence measurement is
# mapping-independent, so stage 2 does not need to be recomputed; only DIV_ROOT is read,
# never written, by either this script or r31_rho_map.py.
#
# EVERY OUTPUT PATH IS NEW so nothing from the calibrated run is touched:
#   RHO_ROOT -> r31_rho_lin_t0.4   (already built locally, CPU-only, 2026-09-15; this
#               script rebuilds it anyway for idempotency -- identical values, milliseconds)
#   OUT_ROOT -> r31_arms_lin04     (was r31_arms)
#   METHOD   -> r31_${ARM}_lin04   (was r31_${ARM})
#
# Same array layout, same schedule (--step 15, stage 1's 7 is irrelevant here), same
# render cost as r31_stage3.sh -- see that script's header for the full rationale on
# --time / --mem / --dump_steps all. Measured real elapsed time for the calibrated run
# (993124/993164, 2026-09-15): ~1-1.5h wall clock across all 4 arms. This job (993384)
# itself took ~2h10m, a bit longer, consistent with cluster load rather than the mapping
# (render cost is mapping-independent; only the CPU-side rho values differ). It COMPLETED
# CLEANLY (GATE1=22/22, step14=22/22, 0 failures per arm) before this file was found
# deleted from disk by an external process -- recreated verbatim from conversation
# context 2026-09-15; the render output itself (r31_arms_lin04/) was untouched.
#SBATCH --job-name=r31_stage3_lin04
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r31_stage3_lin04_%A_%a.out
#SBATCH --error=logs/r31_stage3_lin04_%A_%a.err

# NOTE: no set -u -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at conda activate (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DIV_ROOT=/projects/dataggen/outputs/five_bench/r31_div          # READ-ONLY, unchanged
RHO_ROOT=/projects/dataggen/outputs/five_bench/r31_rho_lin_t0.4  # NEW, not r31_rho
OUT_ROOT=/projects/dataggen/outputs/five_bench/r31_arms_lin04    # NEW, not r31_arms
# Same anchor files R7/R10/R20/R21/R26/R30/R31 read -- held byte-identical.
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES=$REPO/evaluation/cases.json

# The four divergence arms, same order as r31_stage2.sh / r31_eval.sh / r31_stage3.sh.
ARMS=(lpips dino_patch normals latent)
STEP=15
FLOW_SHIFT=1.0
THRESH=0.4
TAU_MAX=50

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r31_stage3_lin04] bad array id ${TID} (expected 0-3)"; exit 1
fi
ARM=${ARMS[$TID]}
METHOD="r31_${ARM}_lin04"

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

echo "[r31_stage3_lin04] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} method=${METHOD}"
echo "[r31_stage3_lin04] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4

# ---- preflight: this arm's stage-2 divergence npz must be complete BEFORE building the
# rho field or loading the model.
N_DIV=$(ls "$DIV_ROOT/$ARM"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r31_stage3_lin04] stage-2 npz for ${ARM}: ${N_DIV} (expect 22)"
if [ "$N_DIV" -ne 22 ]; then
  echo "[r31_stage3_lin04] FATAL: ${ARM} stage-2 output incomplete (${N_DIV}/22) under $DIV_ROOT/$ARM"
  exit 1
fi

# ---- build this arm's exponent field with the EXPLORATORY mapping. Cheap and CPU-only
# (milliseconds per clip, no schedule/bisection needed at all for this mapping); folded
# into this job so a GPU allocation is never held idle waiting on it, matching
# r31_stage3.sh's own convention.
echo "[r31_stage3_lin04] building rho field (linear_threshold): ${DIV_ROOT}/${ARM} -> ${RHO_ROOT}/${ARM}"
python evaluation/r31_rho_map.py \
  --div_root "$DIV_ROOT/$ARM" \
  -o "$RHO_ROOT/$ARM" \
  --mapping linear_threshold --thresh "$THRESH" --tau_max "$TAU_MAX" \
  --step "$STEP" --flow_shift "$FLOW_SHIFT"
RC=$?
N_RHO=$(ls "$RHO_ROOT/$ARM"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r31_stage3_lin04] rho field: ${N_RHO} npz (expect 22) rc=${RC}"
if [ "$RC" -ne 0 ] || [ "$N_RHO" -ne 22 ]; then
  echo "[r31_stage3_lin04] FATAL: rho field build failed or incomplete for ${ARM}"
  exit 1
fi

# ---- render all 6 edit types, model loaded once per arm. r31_stage3.py's own preflight
# re-validates every rho npz (shape, frame count, schedule) before the model loads.
FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r31_stage3_lin04] ${METHOD} edit${T} ---"
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
    || { echo "[r31_stage3_lin04] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

LOG=logs/r31_stage3_lin04_${SLURM_ARRAY_JOB_ID}_${TID}.out
N_GATE=$(grep -c 'GATE1 PASS' "$LOG" 2>/dev/null)
N_STEP14=$(ls -d "$OUT_ROOT/$METHOD/step14"/edit*/*/ 2>/dev/null | wc -l)
N_STEP00=$(ls -d "$OUT_ROOT/$METHOD/step00"/edit*/*/ 2>/dev/null | wc -l)
echo "[r31_stage3_lin04] ${METHOD}: GATE1=${N_GATE} (expect 22) step14 dirs=${N_STEP14} step00 dirs=${N_STEP00} (expect 22 each)"

N_BAD=$(cat "$OUT_ROOT/$METHOD"/_manifest_edit*.csv 2>/dev/null \
       | awk -F, 'NR==1 || $3!="ok"' | awk -F, '$3!="status" && $3!="ok"' | wc -l)
echo "[r31_stage3_lin04] ${METHOD}: non-ok manifest rows: ${N_BAD}"
if [ "$N_BAD" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r31_stage3_lin04] failures: ${FAILED}"
exit $(( FAILED > 0 ))
