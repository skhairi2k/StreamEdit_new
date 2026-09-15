#!/bin/bash
# R21 -- promote R20's two most promising blend-rate arms (cos_third, zero) to the
# WHOLE FiVE-Bench (all 6 edit types, ~419 pairs) under the two anchoring regimes
# the user cares about: vp (paper SS4.5, R7's cached first frame) and pvp (R10's
# persistent re-roped anchor bank). No novp.
#
# ARRAY LAYOUT -- one ARM per task (--array=0-4); each task loads the model once
# and loops ALL 6 edit types internally. 5 tasks total, which fits under the QOS
# submit-job-per-user cap in a single submission (the 30-task arm x edit-type
# layout was rejected: QOSMaxSubmitJobPerUserLimit).
#
# ARMS (index -> "{sched} {vp_mode}", output dir {sched}_{vp_mode}):
#   0  cos_third vp    <- experiment
#   1  cos_third pvp   <- experiment
#   2  zero      vp    <- experiment
#   3  zero      pvp   <- experiment
#   4  paper     pvp   <- REFERENCE arm (Eq.4 under persistent VP)
#
# Why arm 4 exists: the pvp Eq.4 reference does NOT exist at full-bench scale --
# R20 rendered paper_pvp on only the 22 cases.json clips, and R10b's all-heads arm
# went through a different driver. Without it the pvp cos_third/zero columns would
# have no config-matched baseline. The vp Eq.4 reference already exists on disk
# (five_bench/r7_visual_prompting, R7, re-rendered under per-pair seeding) and the
# novp one (five_bench/baseline, R1) is unused here.
#
# NO --cases_json: this is the full bench. run_fivebench.py re-seeds per pair (on
# by default since R20), so every pair is comparable to R1/R7 full-bench references.
#
# NEW --out_root (r21_blend_full): arm dirs cos_third_pvp / zero_pvp / paper_pvp
# also exist under R20's r20_blend_sched/ as 22-clip renders. The distinct out_root
# is what keeps the two generations from mixing -- do NOT point this at r20_blend_sched.
#
# Anchors: /projects/dataggen/outputs/five_bench/anchors/edit{T}/{video}.png -- the
# same R7 anchors R10/R20 read (all 6 types, 419 present), so arms differ only by
# injection mechanism.
#
# --time=20:00:00: one arm is ~419 pairs x ~85s ~= 10h; 20h leaves margin under the
# L40S 24h MaxTime. --mem=64G: the partition default (~31.5G) host-OOMs loading
# UMT5-XXL + checkpoint.
#SBATCH --job-name=r21_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/r21_infer_%A_%a.out
#SBATCH --error=logs/r21_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r21_blend_full
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors

# arm index -> "{sched} {vp_mode}"
ARMS=("cos_third vp" "cos_third pvp" "zero vp" "zero pvp" "paper pvp")
EDIT_TYPES=(1 2 3 4 5 6)

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 4 ]; then
  echo "[r21_infer] bad array id ${TID} (expected 0-4)"; exit 1
fi
read -r SCHED VP_MODE <<< "${ARMS[$TID]}"
METHOD="${SCHED}_${VP_MODE}"

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

echo "[r21_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${METHOD} sched=${SCHED} vp_mode=${VP_MODE}"
echo "[r21_infer] edit_types=${EDIT_TYPES[*]} -> ${OUT_ROOT}/${METHOD}/ (full bench, no --cases_json)"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "[r21_infer] --- ${METHOD} / edit${T} ---"
  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --blend_sched "$SCHED" --vp_mode "$VP_MODE" \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0 \
    || { echo "[r21_infer] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
done

N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/*/*/ 2>/dev/null | wc -l)
echo "[r21_infer] ${METHOD} frame dirs (all edit types): ${N_DIRS}"
echo "[r21_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
