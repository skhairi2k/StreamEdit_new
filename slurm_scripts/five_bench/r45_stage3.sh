#!/bin/bash
# R45 stage 3 -- the SPATIALLY-GATED 15-step LongLive render for the SELECTED combos, on the
# whole 419-pair FiVE-Bench. r38_stage3.sh with R45's roots, combo table and --backbone ll.
#
#   sbatch --array=$(python evaluation/r45_select.py stage3) slurm_scripts/five_bench/r45_stage3.sh
#
# ARRAY LAYOUT: one COMBO per task, same order as r45_stage2.sh / r45_select.py COMBOS:
#   0 ll_dino_n15_unblended   1 ll_dino_n15_tgt09   2 ll_lpips_n15_unblended   3 ll_lpips_n15_tgt09
#   4 ll_dino_n5_unblended    5 ll_dino_n5_tgt09    6 ll_lpips_n5_unblended    7 ll_lpips_n5_tgt09
# An unselected combo refuses to start (r45_select.py check).
#
# Does NOT build rho: r45_absnorm.sh built r45_div_abs/{combo} and r45_rho/{combo}; preflight
# checks both are complete before the model loads.
# --step 15 IS PINNED (the rho field was inverted on the 15-step t_next grid).
# --dump_steps 14: final only. Render: --backbone ll (tripled VP frame), fg_boost 2,
# blend_power 2, seed 0, flow_shift 1.0 -- the same sampler as r45_sweep.sh at N=15.
#
# ⚠ DISK: ~15 GB per combo under ~/Data -- R35's stage 3 hit Errno 122 (disk quota) once.
#SBATCH --job-name=r45_stage3
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r45_stage3_%A_%a.out
#SBATCH --error=logs/r45_stage3_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
DIV_ABS_ROOT=~/Data/dataggen/outputs/five_bench/r45_div_abs
RHO_ROOT=~/Data/dataggen/outputs/five_bench/r45_rho
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r45_arms
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors

COMBOS=(ll_dino_n15_unblended ll_dino_n15_tgt09 ll_lpips_n15_unblended ll_lpips_n15_tgt09
        ll_dino_n5_unblended  ll_dino_n5_tgt09  ll_lpips_n5_unblended  ll_lpips_n5_tgt09)
STEP=15
FLOW_SHIFT=1.0

TID=${SLURM_ARRAY_TASK_ID}
MAX_TID=$(( ${#COMBOS[@]} - 1 ))
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt "$MAX_TID" ]; then
  echo "[r45_stage3] bad array id ${TID} (expected 0-${MAX_TID})"; exit 1
fi
COMBO=${COMBOS[$TID]}
METHOD="r45_${COMBO}"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve
cd "$REPO"
python evaluation/r45_select.py check "$TID" stage3 || exit 1
mkdir -p logs "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r45_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} method=${METHOD} backbone=ll"
echo "[r45_stage3] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r45_stage3] free on \$HOME/Data: $(df -h ~/Data | awk 'NR==2{print $4}')"
nvidia-smi -L 2>&1 | head -2

# ---- preflight: BOTH r45_div_abs and r45_rho complete for this combo BEFORE the model loads.
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
  echo "[r45_stage3] ${COMBO} ${LABEL}: ${N_TOTAL} npz (expect 419), mismatched types=${BAD}"
  if [ "$N_TOTAL" -ne 419 ] || [ "$BAD" -ne 0 ]; then
    echo "[r45_stage3] FATAL: ${COMBO} ${LABEL} incomplete under ${ROOT}/${COMBO} -- run /run-step R45 absnorm first"
    FAILED=$((FAILED+1))
  fi
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi

# ---- render all 6 edit types, model loaded once per combo. r31_stage3.py re-validates
# every rho npz (shape, frame count, schedule) before the model loads.
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r45_stage3] ${METHOD} edit${T} ---"
  python evaluation/r31_stage3.py --backbone ll \
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
    --fg_boost_factor 2 \
    --blend_power 2 \
    --seed 0 \
    || { echo "[r45_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown of step14 dirs, plus manifest status.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r45_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- ${STATUS}"
done
echo "[r45_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi
# Per FILE (FNR resets per file), not `cat | awk` -- R35's false-positive lesson.
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r45_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED+1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r45_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    awk -F, 'FNR==1{next} $3!="ok"{print "[r45_stage3]   non-ok: " FILENAME ": " $0}' "${MANIFESTS[@]}"
    FAILED=$((FAILED+1))
  fi
fi
echo "[r45_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
