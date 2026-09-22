#!/bin/bash
# R33 -- FiVE-Acc alone, array over R26's 16 constant-b arms + R31's 4 divergence arms
# (20 tasks), scored with evaluate.py --metrics five_acc ONLY.
#
# WHY FIVE_ACC ALONE. R20's GPU exhaustion was Qwen2.5-VL-7B (five_acc) co-resident with
# CoTracker (motion_fidelity_score) and the rest of the 9/16-metric table on one GPU --
# not five_acc's own cost. r26_eval.sh/r31_eval.sh already drop both models entirely;
# this job runs the one they drop, alone, so nothing new co-resides with it.
#
# ARM -> DIR SHAPE DIFFERS PER TASK GROUP. R26 arms sit at
# r26_spatial_tau/{method}/edit{T}/{video}/ directly. R31 arms sit one level deeper, at
# r31_arms/r31_{arm}/step14/edit{T}/{video}/ (stage 3's FINAL render; step14 is R31's
# 15-step rollout's last frame). The task dispatch below resolves the full DIR per task
# rather than a root + formula, since one formula cannot express both shapes.
#
# STEM naming follows the precedent of the (now-removed) discrete-oracle framework's
# r30_fiveacc_taubg0_taufg{b}_vp_avg.csv files: r33_fiveacc_{arm-label}. evaluate.py
# derives evaluation/csv/{stem}_avg.csv itself (plus per-edit-type intermediates named
# edit{T}_FiVE_{stem}_frame_stride8*.csv), so a stem ending in _avg would double up.
#
# evaluate.py SWALLOWS per-metric exceptions and still exits 0 -- a clean exit code is
# NOT evidence the table is clean. The log is teed and grepped below, as r26/r31_eval.sh
# do.
#SBATCH --job-name=r33_fiveacc
# /projects/dataggen SIDESTEPPED 2026-09-21: three submissions in a row (1002742,
# 1002751, 1002959) each excluded the previous round's bad node and each time a NEW
# idle node (node57, node51 -> node01 -> node02) turned out to have the same stale
# /projects/dataggen mount, while nodes already holding warm jobs (node55, node56, an
# interactive V100/node12 session) saw it fine -- exclude-and-retry was not converging.
# User provided a local copy at ~/Data/dataggen (home dir, reliably mounted everywhere
# unlike the /projects share), so R26_ROOT/R31_ROOT below point there instead. Node
# exclusions kept defensively for this run since they haven't been disproven for the
# home mount specifically, not because /projects's problem is assumed to follow.
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/r33_fiveacc_%A_%a.out
#SBATCH --error=logs/r33_fiveacc_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R26_ROOT=~/Data/dataggen/outputs/five_bench/r26_spatial_tau
R31_ROOT=~/Data/dataggen/outputs/five_bench/r31_arms
CASES="$REPO/evaluation/cases.json"

# CONST_BS, same 8 values as r30_score.py / r26_eval.sh. Tasks 0-7: R26 SPATIAL
# (taubg0_taufg{b}_vp). Tasks 8-15: R26 UNIFORM (taubg{b}_taufg{b}_vp). Tasks 16-19:
# R31's 4 divergence arms (depth already dropped upstream at R31's stage 2 -- see
# r31_eval.sh).
BS=(2 3 4 6 8 10 20 50)
R31_ARMS=(lpips dino_patch normals latent)

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 19 ]; then
  echo "[r33_fiveacc] bad array id ${TID} (expected 0-19)"; exit 1
fi

if [ "$TID" -lt 8 ]; then
  B=${BS[$TID]}
  METHOD="taubg0_taufg${B}_vp"
  STEM="r33_fiveacc_${METHOD}"
  DIR="$R26_ROOT/$METHOD"
elif [ "$TID" -lt 16 ]; then
  B=${BS[$((TID - 8))]}
  METHOD="taubg${B}_taufg${B}_vp"
  STEM="r33_fiveacc_${METHOD}"
  DIR="$R26_ROOT/$METHOD"
else
  ARM=${R31_ARMS[$((TID - 16))]}
  STEM="r33_fiveacc_r31_${ARM}"
  DIR="$R31_ROOT/r31_${ARM}/step14"
fi

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

echo "[r33_fiveacc] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r33_fiveacc] job=${SLURM_ARRAY_JOB_ID}_${TID} stem=${STEM}"
echo "[r33_fiveacc] scoring $DIR"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

if [ ! -d "$DIR" ]; then
  echo "[r33_fiveacc] FAILED ${STEM} -- arm dir does not exist: $DIR"
  echo "[r33_fiveacc]   (is ~/Data/dataggen present on this node?)"
  exit 1
fi
# Count REAL pairs only: evaluate.py writes a sibling {video}_resize dir next to every
# video it scores, so a bare `ls` over an already-scored dir over-counts.
NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r33_fiveacc] real frame dirs in $DIR: ${NDIRS}  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r33_fiveacc] FAILED ${STEM} -- expected 22 real pairs, found ${NDIRS}."
  echo "[r33_fiveacc]   Scoring a short arm would still write a full-looking table. Stop."
  exit 1
fi

METRICS_LOG="logs/r33_fiveacc_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics five_acc \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" \
  --tgt_layout edit_video \
  --cases_json "$CASES" \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r33_fiveacc] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r33_fiveacc] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  exit 1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r33_fiveacc] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  exit 1
fi

echo "[r33_fiveacc] OK ${STEM} -> evaluation/csv/${STEM}_avg.csv"
exit 0
