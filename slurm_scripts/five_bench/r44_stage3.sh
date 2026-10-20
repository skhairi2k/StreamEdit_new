#!/bin/bash
# R44 stage 3 -- the SPATIALLY-GATED 15-step TEXT-ONLY render for the user's selected
# combos, on the whole 419-pair FiVE-Bench. r38_stage3.sh with --vp_mode novp, no anchors,
# R44's roots and the r44_combos.sh selection; only the conditioning (no visual prompt)
# and the rho fields (built from text-only drafts) differ from R35/R38.
#
# USAGE: sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --combos | wc -l) - 1 )) \
#          slurm_scripts/five_bench/r44_stage3.sh
# ARRAY LAYOUT: task i -> combo i of evaluation/r44_selected_combos.txt (file order) --
# same index map as r44_stage2 / r44_absnorm / r44_eval / r44_fiveacc. METHOD r44_{combo}.
#
# Invocation identical to r44_smoke.sh's `routed` call, which gate G1 compared (on a
# constant rho = 2 field) against the uniform render the R44 sweep uses: --vp_mode novp,
# NO --first_frame_edit_dir, fg_boost 4, blend_power 2, rollout_chunk_size 21, seed 0,
# flow_shift 1.0 -- so the routed arms and the R44 uniform curve share one sampler.
#
# Does NOT build rho: r44_absnorm.sh (this step's `wait_for`) built r44_div_abs/{combo}
# and r44_rho/{combo}; preflight checks both are complete before the model loads.
#
# --step 15 IS PINNED (the RENDER stays at 15 steps whatever the draft N): r31_rho_map.py
# inverted the source budget on the 15-step t_next grid, and a different --step here would
# silently invalidate every exponent in the field.
# --dump_steps 14: final only (R35 Decision; ~15 GB per combo vs ~230 GB for `all`).
#
# ⚠ DISK: ~15 GB per combo under ~/Data -- R35's stage 3 hit Errno 122 (disk quota) once
# (dino_first2 edit1 0050_slackline). The manifest guard below catches such a clip;
# re-render it alone with --cases_json as R35 did (job 1012688).
#SBATCH --job-name=r44_stage3
# L40S ONLY: same model and clip lengths as R35 stage 1, which OOM'd a 40 GB A100.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r44_stage3_%A_%a.out
#SBATCH --error=logs/r44_stage3_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
DIV_ABS_ROOT=$FIVE_ROOT/r44_div_abs
RHO_ROOT=$FIVE_ROOT/r44_rho
OUT_ROOT=$FIVE_ROOT/r44_arms

STEP=15
FLOW_SHIFT=1.0

source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
r44_load || exit 1
TID=${SLURM_ARRAY_TASK_ID}
K=${#R44_COMBOS[@]}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "$K" ]; then
  echo "[r44_stage3] bad array id '${TID}' (expected 0-$((K-1)) for ${K} selected combos)"; exit 1
fi
COMBO=${R44_COMBOS[$TID]}
METHOD="r44_${COMBO}"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

# Conda: streamgve -- this stage RENDERS. `cd; source .bashrc` killed R25 array job 960685
# when submitted from inside a GPU allocation; this is the r26/r36/r39_smoke pattern.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r44_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} method=${METHOD} (text only)"
echo "[r44_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r44_stage3] free on \$HOME/Data: $(df -h ~/Data | awk 'NR==2{print $4}')"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: BOTH r44_div_abs and r44_rho complete for this combo BEFORE the model loads.
FAILED=0
for ROOT_NAME in "div_abs:$DIV_ABS_ROOT" "rho:$RHO_ROOT"; do
  LABEL=${ROOT_NAME%%:*}; ROOT=${ROOT_NAME#*:}
  N_TOTAL=0
  BAD=0
  for T in "${EDIT_TYPES[@]}"; do
    NF=$(ls "$ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
    N_TOTAL=$((N_TOTAL + NF))
    [ "$NF" -ne "${EXPECT[$T]}" ] && BAD=$((BAD+1))
  done
  echo "[r44_stage3] ${COMBO} ${LABEL}: ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r44_stage3] FATAL: ${COMBO} ${LABEL} incomplete under ${ROOT}/${COMBO}"
    echo "[r44_stage3]   -- run /run-step R44 absnorm first"
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi

# Created only after preflight, so a failed preflight leaves no empty output tree behind.
mkdir -p "$OUT_ROOT"

# ---- render all 6 edit types, model loaded once per combo. r31_stage3.py re-validates
# every rho npz (shape, frame count, schedule) before the model loads.
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r44_stage3] ${METHOD} edit${T} ---"
  python evaluation/r31_stage3.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --rho_dir "$RHO_ROOT/$COMBO" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --vp_mode novp \
    --dump_steps 14 \
    --step "$STEP" \
    --flow_shift "$FLOW_SHIFT" \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r44_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown of step14 dirs, plus manifest status.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  NF=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + NF))
  STATUS="ok"
  if [ "$NF" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r44_stage3] ${METHOD} edit${T}: ${NF} step14 dirs -- ${STATUS}"
done
echo "[r44_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

# Per FILE, not `cat | awk`: R35's guard piped 6 CSVs through one awk stream, so FNR==1
# skipped only the first header and the other 5 counted as non-ok rows (false positive
# found at R35 wait-stage3). Passing the files to awk directly resets FNR per file.
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r44_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED+1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r44_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    awk -F, 'FNR==1{next} $3!="ok"{print "[r44_stage3]   non-ok: " FILENAME ": " $0}' "${MANIFESTS[@]}"
    FAILED=$((FAILED+1))
  fi
fi

echo "[r44_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
