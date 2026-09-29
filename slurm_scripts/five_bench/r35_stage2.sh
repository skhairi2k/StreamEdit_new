#!/bin/bash
# R35 stage 2 -- per-token, per-frame divergence for the four (arm, regime) COMBOS,
# on the whole 419-pair FiVE-Bench.
#
# ARRAY LAYOUT: one COMBO per task (the task's packaging requirement), each looping all
# 6 edit types internally so the arm's model loads once per task -- same pattern as
# r31_stage2.sh, but the array indexes (arm, regime) pairs instead of arm alone, because
# R35 crosses 2 arms x 2 stage-1 gating regimes:
#   0 dino_unblended   DINOv3 ViT-B/16, per-position 1-cos  x  unblended stage-1 target
#   1 dino_first2      DINOv3 ViT-B/16, per-position 1-cos  x  first2 stage-1 target
#   2 lpips_unblended  LPIPS(alex, spatial=True)            x  unblended stage-1 target
#   3 lpips_first2     LPIPS(alex, spatial=True)            x  first2 stage-1 target
# (SLURM priority order per the plan's Decisions table.)
#
# Each combo writes into its OWN tree, r35_div/{combo}/edit{T}/{video}.npz, via
# r31_divergence.py's new --arm_dir_name override (2026-09-24): --arm stays the TRUE
# metric name (dino_patch / lpips), so the npz's own `arm` field is unaffected --
# --arm_dir_name only redirects where the four combos land on disk.
#
# --no-save_native (per the plan's Decisions, MUST): at 419 clips x 2 arms x 2 regimes,
# --save_native's default True would write ~27 GB of pre-reduction LPIPS/DINO maps that
# only R31's native-resolution figure ever consumed -- not part of R35's deliverables.
#
# ⚠️ ENV IS `five-bench`, NOT `streamgve` -- same reason as r31_stage2.sh: `lpips` is not
# installed in streamgve at all (ModuleNotFoundError), and streamgve's diffusers/
# transformers pairing breaks Marigold-adjacent imports. `five-bench` is what actually
# runs these two arms; the explicit `conda deactivate` before it is load-bearing because
# .bashrc auto-activates streamgve first (the R2 env bug) and `conda activate five-bench`
# alone does not pop that.
#
# --mem=64G matches every other five_bench script (the partition default host-OOMs).
#SBATCH --job-name=r35_stage2
#SBATCH --partition=L40S,A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r35_stage2_%A_%a.out
#SBATCH --error=logs/r35_stage2_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
STAGE1_ROOT=~/Data/dataggen/outputs/five_bench/r35_stage1
LATENTS_ROOT=~/Data/dataggen/outputs/five_bench/r35_latents
DIV_ROOT=~/Data/dataggen/outputs/five_bench/r35_div
CASES=$REPO/evaluation/r7_anchor_manifest.json   # corrected 2026-09-24, 419/419 verified

COMBOS=(dino_unblended dino_first2 lpips_unblended lpips_first2)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r35_stage2] bad array id ${TID} (expected 0-3)"; exit 1
fi
COMBO=${COMBOS[$TID]}

# Explicit lookup, not string-splitting: dino_patch itself contains an underscore, so
# COMBO.split("_") would not recover it reliably.
case "$COMBO" in
  dino_unblended)  ARM=dino_patch; REGIME=unblended ;;
  dino_first2)     ARM=dino_patch; REGIME=first2    ;;
  lpips_unblended) ARM=lpips;      REGIME=unblended ;;
  lpips_first2)    ARM=lpips;      REGIME=first2    ;;
  *) echo "[r35_stage2] unknown combo ${COMBO}"; exit 1 ;;
esac

# 14 = the last index of stage 1's 15-step rollout (r35_stage1.sh: --dump_steps 14
# --measure_index 14) -- the fully-denoised target, matching r31_divergence.py's own
# --target_root convention (".../step06" for R31's 7-step run; ".../step14" here).
TARGET_ROOT=$STAGE1_ROOT/$REGIME/step14
LATENT_DIR=$LATENTS_ROOT/$REGIME

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench            # NOT streamgve -- see header note

cd "$REPO"
mkdir -p logs "$DIV_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the R20/R21 metric-crash fix
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r35_stage2] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} arm=${ARM} regime=${REGIME}"
echo "[r35_stage2] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r35_stage2] target=${TARGET_ROOT}  latents=${LATENT_DIR}  -> ${DIV_ROOT}/${COMBO}"
nvidia-smi -L 2>&1 | head -2

# Preflight: this regime's stage-1 output must be complete BEFORE the model loads.
# Failing here costs seconds; failing after a model load costs minutes x 6 edit types.
if [ ! -d "$LATENT_DIR" ] || [ ! -d "$TARGET_ROOT" ]; then
  echo "[r35_stage2] FATAL: ${LATENT_DIR} or ${TARGET_ROOT} does not exist -- "
  echo "[r35_stage2]   run /run-step R35 stage1 first"
  exit 1
fi
N_LAT_TOTAL=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_LAT_TOTAL=$((N_LAT_TOTAL + N))
  if [ "$N" -ne "${EXPECT[$T]}" ]; then
    echo "[r35_stage2] FATAL: ${REGIME} edit${T} has ${N} latents, expected ${EXPECT[$T]}"
    exit 1
  fi
done
echo "[r35_stage2] preflight OK: ${N_LAT_TOTAL} stage-1 latents for regime=${REGIME} (expect 419)"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r35_stage2] ${COMBO} / edit${T} ---"
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
    || { echo "[r35_stage2] FAILED ${COMBO} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown, same discipline as r35_stage1.sh --
# a total-only count already proved insufficient once this task (the anchor-manifest
# episode: a total of 419 hid one dropped clip and one duplicate).
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  N=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + N))
  STATUS="ok"
  if [ "$N" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r35_stage2] ${COMBO} edit${T}: ${N} npz -- ${STATUS}"
done
echo "[r35_stage2] ${COMBO}: ${N_TOTAL} npz total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r35_stage2] failures: ${FAILED}"
exit $(( FAILED > 0 ))
