#!/bin/bash
# R20 evaluation -- 16-metric FiVE scoring of the schedule arms (9 phase-1 cosine,
# +6 phase-2 const/zero once rendered), the pvp+paper
# reference arm, and the two stored paper-schedule references (R1, R7).
#
# --cases_json is passed to EVERY method, references included. That is not a
# convenience: five_bench/baseline holds 419 pairs and r7_visual_prompting 443,
# so scoring them unrestricted and comparing against 22-clip arms would be an
# unfair comparison dressed up as a baseline. Every column must be the same
# 22 clips.
#
# BLOCKED until the R1/R7 re-render lands: `baseline` and `r7_visual_prompting`
# predate the per-pair seeding change (2026-07-22) and are being re-rendered.
# Scoring the old frames against fresh arms mixes two sampling generations and
# the deltas would partly measure noise. Check their mtimes before running.
#
# Uses the five-bench env, NOT streamgve: .bashrc auto-activates streamgve, and
# `conda activate five-bench` alone does not pop it (the R2 job-880402 failure,
# ModuleNotFoundError: torchmetrics). Hence the explicit deactivate.
#SBATCH --job-name=r20_eval
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r20_eval_%j.out
#SBATCH --error=logs/r20_eval_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r20_blend_sched
REF_ROOT=/projects/dataggen/outputs/five_bench
CASES=$REPO/evaluation/cases.json

cd
source .bashrc
conda deactivate
conda activate five-bench

cd "$REPO"
mkdir -p logs evaluation/csv

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# $1 = method dir, $2 = csv stem WITHOUT the _avg suffix.
# evaluate.py derives the stable overall file itself as
#   evaluation/csv/{stem}_avg.csv
# (plus per-edit-type intermediates named edit{T}_FiVE_{stem}_frame_stride8*.csv).
# Passing a stem that already ends in _avg would yield {stem}_avg_avg.csv.
#
# NOTE: that overall row is a mean over the four per-edit-type means, not a mean
# over the 22 clips, so edit1 (1 clip) weighs as much as edit2 (16). Every arm AND
# both references go through this same path, so the comparison stays internally
# consistent -- but do not compare these numbers against a per-clip average.
score () {
  echo "[r20_eval] === $2 ($1) ==="
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --src_image_folder "$DATA_ROOT" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --tgt_methods "$1" \
    --tgt_layout edit_video \
    --cases_json "$CASES" \
    --result_path "evaluation/csv/$2.csv" \
    || { echo "[r20_eval] FAILED $2"; return 1; }
}

FAILED=0

# --- the 9 phase-1 schedule arms + the pvp paper reference ---
# Phase-2 arms (const, zero) are scored by the same loop once they exist; the
# `[ -d ]` guard keeps this script runnable before phase 2 has been launched
# rather than failing on 6 absent directories.
for SCHED in cos_full cos_half cos_third const zero; do
  for MODE in novp vp pvp; do
    if [ -d "$OUT_ROOT/${SCHED}_${MODE}" ]; then
      score "$OUT_ROOT/${SCHED}_${MODE}" "r20_${SCHED}_${MODE}" || FAILED=$((FAILED+1))
    else
      echo "[r20_eval] skip ${SCHED}_${MODE} -- not rendered yet"
    fi
  done
done
score "$OUT_ROOT/paper_pvp" "r20_paper_pvp" || FAILED=$((FAILED+1))

# --- the two stored paper-schedule references, restricted to the same 22 clips ---
score "$REF_ROOT/baseline"              "r20_ref_novp" || FAILED=$((FAILED+1))
score "$REF_ROOT/r7_visual_prompting"   "r20_ref_vp"   || FAILED=$((FAILED+1))

echo "[r20_eval] csvs: $(ls evaluation/csv/r20_*_avg.csv 2>/dev/null | wc -l) (12 after phase 1, 18 after phase 2)"
echo "[r20_eval] failures: ${FAILED}"
exit $(( FAILED > 0 ))
