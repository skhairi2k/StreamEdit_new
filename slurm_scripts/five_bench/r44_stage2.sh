#!/bin/bash
# R44 stage 2 -- per-token, per-frame divergence of the TEXT-ONLY drafts, for the user's
# selected combos, on the whole 419-pair FiVE-Bench. r38_stage2.sh with R44's roots and
# the r44_combos.sh table.
#
# USAGE: sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --combos | wc -l) - 1 )) \
#          slurm_scripts/five_bench/r44_stage2.sh
# ARRAY LAYOUT: task i -> combo i of evaluation/r44_selected_combos.txt (file order), each
# looping all 6 edit types (model loads once). Same index map as r44_absnorm/stage3/eval.
#   dino_*   DINOv3 ViT-B/16, per-position 1-cos    (--arm dino_patch)
#   lpips_*  LPIPS(alex, spatial=True)              (--arm lpips)
# on the draft r44_stage1/{run} with run = n{N}_{regime} from the combo name.
#
# TARGET: the draft's FINAL x0 -- step{N-1:02d} of r44_stage1/{run} (step14 at N=15, step04
# at N=5), the same "fully denoised draft" convention as R35/R38. r31_divergence.py makes
# no step-count assumption: it only passes the npz's measure_index / measure_kind through.
# It compares the draft with the SOURCE video only -- no anchor input -- so stage 2 is
# unchanged by text-only; --cases is read for (video_name, edit_type) only.
#
# --arm_dir_name {combo}: --arm stays the TRUE metric name, so the npz's `arm` field is
# unaffected; only the output directory is combo-keyed. --no-save_native (R35 Decision).
#
# ⚠️ ENV IS `five-bench`, NOT `streamgve` (`lpips` is not installed in streamgve); the
# `conda deactivate` before it is load-bearing (.bashrc auto-activates streamgve).
# --time=10:00:00 as R38 (R38's 4 tasks, 419 clips each, finished well inside it).
#SBATCH --job-name=r44_stage2
#SBATCH --partition=L40S,A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r44_stage2_%A_%a.out
#SBATCH --error=logs/r44_stage2_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
STAGE1_ROOT=$FIVE_ROOT/r44_stage1
LATENTS_ROOT=$FIVE_ROOT/r44_latents
DIV_ROOT=$FIVE_ROOT/r44_div
CASES=$REPO/evaluation/r7_anchor_manifest.json   # 419/419 verified 2026-09-24; clip list only

source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
r44_load || exit 1
TID=${SLURM_ARRAY_TASK_ID}
K=${#R44_COMBOS[@]}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "$K" ]; then
  echo "[r44_stage2] bad array id '${TID}' (expected 0-$((K-1)) for ${K} selected combos)"; exit 1
fi
COMBO=${R44_COMBOS[$TID]}
r44_parse "$COMBO" || exit 1          # -> ARM, N, RUN (and METRIC, REGIME, SCHED)

TARGET_ROOT=$STAGE1_ROOT/$RUN/step$(printf %02d $((N - 1)))
LATENT_DIR=$LATENTS_ROOT/$RUN

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
EDIT_TYPES=(1 2 3 4 5 6)

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (R2 env bug)
conda activate five-bench            # NOT streamgve -- see header note

mkdir -p logs "$DIV_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the R20/R21 metric-crash fix
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r44_stage2] job=${SLURM_ARRAY_JOB_ID}_${TID} combo=${COMBO} arm=${ARM} run=${RUN} N=${N}"
echo "[r44_stage2] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r44_stage2] target=${TARGET_ROOT}  latents=${LATENT_DIR}  -> ${DIV_ROOT}/${COMBO}"
nvidia-smi -L 2>&1 | head -2

# Preflight: this run's stage-1 output must be complete BEFORE the model loads.
if [ ! -d "$LATENT_DIR" ] || [ ! -d "$TARGET_ROOT" ]; then
  echo "[r44_stage2] FATAL: ${LATENT_DIR} or ${TARGET_ROOT} does not exist -- "
  echo "[r44_stage2]   run /run-step R44 stage1 first"
  exit 1
fi
N_LAT_TOTAL=0
for T in "${EDIT_TYPES[@]}"; do
  NL=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  NF=$(ls -d "$TARGET_ROOT/edit${T}"/*/ 2>/dev/null | wc -l)
  N_LAT_TOTAL=$((N_LAT_TOTAL + NL))
  if [ "$NL" -ne "${EXPECT[$T]}" ] || [ "$NF" -ne "${EXPECT[$T]}" ]; then
    echo "[r44_stage2] FATAL: ${RUN} edit${T} has ${NL} latents / ${NF} frame dirs, expected ${EXPECT[$T]}"
    exit 1
  fi
done
echo "[r44_stage2] preflight OK: ${N_LAT_TOTAL} stage-1 latents + frame dirs for run=${RUN} (expect 419)"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r44_stage2] ${COMBO} / edit${T} ---"
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
    || { echo "[r44_stage2] FAILED ${COMBO} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: exact per-type breakdown (a total-only count once hid a dropped
# clip plus a duplicate -- the R35 anchor-manifest episode).
N_TOTAL=0
BAD_TYPES=0
for T in "${EDIT_TYPES[@]}"; do
  NO=$(ls "$DIV_ROOT/$COMBO/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + NO))
  STATUS="ok"
  if [ "$NO" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r44_stage2] ${COMBO} edit${T}: ${NO} npz -- ${STATUS}"
done
echo "[r44_stage2] ${COMBO}: ${N_TOTAL} npz total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r44_stage2] failures: ${FAILED}"
exit $(( FAILED > 0 ))
