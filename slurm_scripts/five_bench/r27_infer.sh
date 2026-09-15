#!/bin/bash
# R27 infer -- depth-delta adaptive tau over the 22 cases.json pairs, vp mode.
#
# Two arms (depth + constant), selected by R27_ARM at submit time. Every pair
# renders with its release exponent from evaluation/csv/r27_tau_map*.csv via
# --tau_map. tau IS Eq. 4's rho (causal_model.py:334).
#
# Array over EDIT TYPES (task 0..5 -> edit1..edit6). Clip counts 1/13/2/2/3/1 = 22.
#
# COVERAGE IS A HARD ERROR: run_fivebench.py raises SystemExit if the map misses
# any pair. Do NOT pass --blend_power alongside --tau_map.
#
# Sampler defaults match R25/R26: step 15, fg_boost 4, seed 0, vp + anchors.
#
#SBATCH --job-name=r27_infer
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/r27_infer_%A_%a.out
#SBATCH --error=logs/r27_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

# sbatch --export=R27_ARM=depth    slurm_scripts/five_bench/r27_infer.sh
# sbatch --export=R27_ARM=constant slurm_scripts/five_bench/r27_infer.sh

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r27_depth_tau
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES="$REPO/evaluation/cases.json"

ARM=${R27_ARM:-depth}
case "$ARM" in
  depth|constant) ;;
  *) echo "[r27_infer] bad R27_ARM='$ARM' (expected depth|constant)"; exit 1 ;;
esac
METHOD="r27_${ARM}_vp"
if [ "$ARM" = "depth" ]; then
  TAU_MAP="$REPO/evaluation/csv/r27_tau_map.csv"
else
  TAU_MAP="$REPO/evaluation/csv/r27_tau_map_${ARM}.csv"
fi

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 5 ]; then
  echo "[r27_infer] bad array id ${TID} (expected 0-5)"; exit 1
fi
T=$(( TID + 1 ))

EXPECTED=(1 13 2 2 3 1)
EXP=${EXPECTED[$TID]}

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r27_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r27_infer] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
nvidia-smi -L 2>&1 | head -4
python -c "import torch; print('[r27_infer] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -2

if [ ! -f "$TAU_MAP" ]; then
  echo "[r27_infer] missing $TAU_MAP -- run the taumap step first"; exit 1
fi

echo "[r27_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} edit_type=${T} arm=${ARM} method=${METHOD}"
echo "[r27_infer] tau_map=${TAU_MAP} cases=${CASES} expect ${EXP} clip(s)"

FAILED=0
python evaluation/run_fivebench.py \
  --edit_type "$T" --method "$METHOD" \
  --vp_mode vp \
  --first_frame_edit_dir "$ANCHOR_ROOT" \
  --cases_json "$CASES" \
  --tau_map "$TAU_MAP" \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --step 15 --fg_boost_factor 4 --seed 0 \
  || { echo "[r27_infer] FAILED edit${T}"; FAILED=$((FAILED+1)); }

N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/"edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r27_infer] edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
if [ "$N_DIRS" -ne "$EXP" ] && [ "$FAILED" -eq 0 ]; then
  echo "[r27_infer] COUNT-MISMATCH edit${T}: ${N_DIRS} != ${EXP}"
  FAILED=$((FAILED+1))
fi

echo "[r27_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
