#!/bin/bash
# R45 stage 2 -- per-token, per-frame divergence on the LongLive drafts, for the SELECTED
# combos, on the whole 419-pair FiVE-Bench. r38_stage2.sh with R45's roots and combo table.
#
#   sbatch --array=$(python evaluation/r45_select.py stage2) slurm_scripts/five_bench/r45_stage2.sh
#
# ARRAY LAYOUT: one COMBO per task, the order of r45_select.py COMBOS (and r45_stage3.sh):
#   0 ll_dino_n15_unblended   1 ll_dino_n15_tgt09   2 ll_lpips_n15_unblended   3 ll_lpips_n15_tgt09
#   4 ll_dino_n5_unblended    5 ll_dino_n5_tgt09    6 ll_lpips_n5_unblended    7 ll_lpips_n5_tgt09
# An unselected combo refuses to start (r45_select.py check).
#
# TARGET: the draft's FINAL x0 -- step{N-1} of r45_stage1/{run}. Metric code unchanged:
# r31_divergence.py works on decoded frames + latents and is backbone-agnostic.
# --arm stays the TRUE metric name (npz `arm` field); --arm_dir_name {combo}; --no-save_native.
#
# ⚠️ ENV IS `five-bench`, NOT `streamgve` (`lpips` is not installed in streamgve); the
# `conda deactivate` before it is load-bearing (.bashrc auto-activates streamgve).
#SBATCH --job-name=r45_stage2
#SBATCH --partition=L40S,A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r45_stage2_%A_%a.out
#SBATCH --error=logs/r45_stage2_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
STAGE1_ROOT=~/Data/dataggen/outputs/five_bench/r45_stage1
LATENTS_ROOT=~/Data/dataggen/outputs/five_bench/r45_latents
DIV_ROOT=~/Data/dataggen/outputs/five_bench/r45_div
CASES=$REPO/evaluation/r7_anchor_manifest.json   # 419/419 verified 2026-09-24

COMBOS=(ll_dino_n15_unblended ll_dino_n15_tgt09 ll_lpips_n15_unblended ll_lpips_n15_tgt09
        ll_dino_n5_unblended  ll_dino_n5_tgt09  ll_lpips_n5_unblended  ll_lpips_n5_tgt09)
TID=${SLURM_ARRAY_TASK_ID}
MAX_TID=$(( ${#COMBOS[@]} - 1 ))
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt "$MAX_TID" ]; then
  echo "[r45_stage2] bad array id ${TID} (expected 0-${MAX_TID})"; exit 1
fi
COMBO=${COMBOS[$TID]}
# Explicit lookup, not string-splitting: dino_patch itself contains an underscore.
case "$COMBO" in
  ll_dino_n15_unblended)  ARM=dino_patch; RUN=n15_unblended; N=15 ;;
  ll_dino_n15_tgt09)      ARM=dino_patch; RUN=n15_tgt09;     N=15 ;;
  ll_lpips_n15_unblended) ARM=lpips;      RUN=n15_unblended; N=15 ;;
  ll_lpips_n15_tgt09)     ARM=lpips;      RUN=n15_tgt09;     N=15 ;;
  ll_dino_n5_unblended)   ARM=dino_patch; RUN=n5_unblended;  N=5  ;;
  ll_dino_n5_tgt09)       ARM=dino_patch; RUN=n5_tgt09;      N=5  ;;
  ll_lpips_n5_unblended)  ARM=lpips;      RUN=n5_unblended;  N=5  ;;
  ll_lpips_n5_tgt09)      ARM=lpips;      RUN=n5_tgt09;      N=5  ;;
  *) echo "[r45_stage2] unknown combo ${COMBO}"; exit 1 ;;
esac
TARGET_ROOT=$STAGE1_ROOT/$RUN/step$(printf %02d $((N - 1)))
LATENT_DIR=$LATENTS_ROOT/$RUN

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench            # NOT streamgve -- see header note
cd "$REPO"
python evaluation/r45_select.py check "$TID" stage2 || exit 1
mkdir -p logs "$DIV_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the R20/R21 metric-crash fix
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r45_stage2] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} arm=${ARM} run=${RUN} N=${N}"
echo "[r45_stage2] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r45_stage2] target=${TARGET_ROOT}  latents=${LATENT_DIR}  -> ${DIV_ROOT}/${COMBO}"
nvidia-smi -L 2>&1 | head -2

# Preflight: this run's stage-1 output must be complete BEFORE the model loads.
if [ ! -d "$LATENT_DIR" ] || [ ! -d "$TARGET_ROOT" ]; then
  echo "[r45_stage2] FATAL: ${LATENT_DIR} or ${TARGET_ROOT} does not exist -- run /run-step R45 stage1 first"
  exit 1
fi
N_LAT_TOTAL=0
for T in "${EDIT_TYPES[@]}"; do
  NL=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  NF=$(ls -d "$TARGET_ROOT/edit${T}"/*/ 2>/dev/null | wc -l)
  N_LAT_TOTAL=$((N_LAT_TOTAL + NL))
  if [ "$NL" -ne "${EXPECT[$T]}" ] || [ "$NF" -ne "${EXPECT[$T]}" ]; then
    echo "[r45_stage2] FATAL: ${RUN} edit${T} has ${NL} latents / ${NF} frame dirs, expected ${EXPECT[$T]}"
    exit 1
  fi
done
echo "[r45_stage2] preflight OK: ${N_LAT_TOTAL} stage-1 latents + frame dirs for run=${RUN} (expect 419)"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r45_stage2] ${COMBO} / edit${T} ---"
  python evaluation/r31_divergence.py \
    --edit_type "$T" \
    --arm "$ARM" \
    --arm_dir_name "$COMBO" \
    --cases "$CASES" \
    --data_root "$DATA_ROOT" \
    --target_root "$TARGET_ROOT" \
    --latent_dir "$LATENT_DIR" \
    --out_root "$DIV_ROOT" \
    --temporal_reduce mean \
    --no-save_native \
    --seed 0 \
    || { echo "[r45_stage2] FAILED ${COMBO} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown.
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  NO=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + NO))
  STATUS="ok"
  if [ "$NO" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r45_stage2] ${COMBO} edit${T}: ${NO} npz -- ${STATUS}"
done
echo "[r45_stage2] ${COMBO}: ${N_TOTAL} npz total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi
echo "[r45_stage2] failures: ${FAILED}"
exit $(( FAILED > 0 ))
