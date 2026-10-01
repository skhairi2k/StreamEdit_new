#!/bin/bash
# R38 stage 3 -- the SPATIALLY-GATED 15-step render for the four N=5-draft COMBOS, on the
# whole 419-pair FiVE-Bench. r35_stage3.sh with R38's roots and combo table; the render
# itself is UNCHANGED from R35 -- only the rho fields (built from an N=5 draft) differ.
#
# ARRAY LAYOUT: one COMBO per task, same order as r38_stage2.sh (do not reorder one
# without the other):
#   0 dino_n5_unblended   1 dino_n5_tgt09   2 lpips_n5_unblended   3 lpips_n5_tgt09
# If gate-lowN drops the tgt09 combos, submit with --array=0,2.
#
# The N=15 renders are NOT redone: R35's r35_arms/r35_{combo} are R38's N=15 points
# (smoke gate G1, job 1016420).
#
# Does NOT build rho: r38_absnorm.sh (this step's `wait_for`) built r38_div_abs/{combo}
# and r38_rho/{combo}; preflight checks both are complete before the model loads.
#
# --step 15 IS PINNED (the RENDER stays at 15 steps; only the stage-1 draft is N=5):
# r31_rho_map.py inverted the source budget on the 15-step t_next grid, and a different
# --step here would silently invalidate every exponent in the field.
# --dump_steps 14: final only (R35 Decision; ~15 GB per combo vs ~230 GB for `all`).
#
# Render config identical to r35_stage3.sh: vp + anchors, fg_boost 4, blend_power 2,
# rollout_chunk_size 21, seed 0, flow_shift 1.0.
#
# ⚠ DISK: ~15 GB per combo, ~60 GB for all four, under ~/Data -- R35's stage 3 hit
# Errno 122 (disk quota) once (dino_first2 edit1 0050_slackline). The manifest guard below
# catches such a clip; re-render it alone with --cases_json as R35 did (job 1012688).
#SBATCH --job-name=r38_stage3
# L40S ONLY: same model and clip lengths as R35 stage 1, which OOM'd a 40 GB A100.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r38_stage3_%A_%a.out
#SBATCH --error=logs/r38_stage3_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DIV_ABS_ROOT=~/Data/dataggen/outputs/five_bench/r38_div_abs
RHO_ROOT=~/Data/dataggen/outputs/five_bench/r38_rho
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r38_arms
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors

COMBOS=(dino_n5_unblended dino_n5_tgt09 lpips_n5_unblended lpips_n5_tgt09)
STEP=15
FLOW_SHIFT=1.0

TID=${SLURM_ARRAY_TASK_ID}
MAX_TID=$(( ${#COMBOS[@]} - 1 ))
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt "$MAX_TID" ]; then
  echo "[r38_stage3] bad array id ${TID} (expected 0-${MAX_TID})"; exit 1
fi
COMBO=${COMBOS[$TID]}
METHOD="r38_${COMBO}"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

# Conda: streamgve -- this stage RENDERS.
cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r38_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} method=${METHOD}"
echo "[r38_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r38_stage3] free on \$HOME/Data: $(df -h ~/Data | awk 'NR==2{print $4}')"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: BOTH r38_div_abs and r38_rho complete for this combo BEFORE the model loads.
FAILED=0
for ROOT_NAME in "div_abs:$DIV_ABS_ROOT" "rho:$RHO_ROOT"; do
  LABEL=${ROOT_NAME%%:*}; ROOT=${ROOT_NAME#*:}
  N_TOTAL=0
  BAD=0
  for T in "${EDIT_TYPES[@]}"; do
    N=$(ls "$ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + N))
    [ "$N" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r38_stage3] ${COMBO} ${LABEL}: ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r38_stage3] FATAL: ${COMBO} ${LABEL} incomplete under ${ROOT}/${COMBO}"
    echo "[r38_stage3]   -- run /run-step R38 absnorm first"
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi

# ---- render all 6 edit types, model loaded once per combo. r31_stage3.py re-validates
# every rho npz (shape, frame count, schedule) before the model loads.
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r38_stage3] ${METHOD} edit${T} ---"
  python evaluation/r31_stage3.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --rho_dir "$RHO_ROOT/$COMBO" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 14 \
    --step "$STEP" \
    --flow_shift "$FLOW_SHIFT" \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r38_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown of step14 dirs, plus manifest status.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r38_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- ${STATUS}"
done
echo "[r38_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

# Per FILE, not `cat | awk`: R35's guard piped 6 CSVs through one awk stream, so FNR==1
# skipped only the first header and the other 5 counted as non-ok rows (false positive
# found at R35 wait-stage3). Passing the files to awk directly resets FNR per file.
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r38_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED+1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r38_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    awk -F, 'FNR==1{next} $3!="ok"{print "[r38_stage3]   non-ok: " FILENAME ": " $0}' "${MANIFESTS[@]}"
    FAILED=$((FAILED+1))
  fi
fi

echo "[r38_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
