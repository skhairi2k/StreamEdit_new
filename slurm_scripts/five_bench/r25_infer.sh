#!/bin/bash
# R25 infer -- the IoU-driven adaptive-tau arm over the 22 cases.json pairs, vp mode.
#
# ONE arm, not a sweep: every pair renders with its own release exponent, read from
# evaluation/csv/r25_tau_map.csv and passed to run_fivebench.py as --tau_map. tau IS
# Eq. 4's rho (`blender_rate = 1 - t_next ** blend_power`, causal_model.py:334), so a
# per-pair tau is exactly a per-pair W_src = t ** tau.
#
# The array is over EDIT TYPES (task 0..5 -> edit1..edit6), not over arms: run_fivebench
# takes one --edit_type per invocation, and splitting this way loads the model once per
# type instead of once per clip. Clip counts per type are 1/13/2/2/3/1 = 22, so the
# tasks are very uneven -- edit2 carries 13 of the 22 and sets the wall time.
#
# COVERAGE IS A HARD ERROR, not a fallback: if r25_tau_map.csv misses any pair that
# survives the --cases_json filter, run_fivebench.py raises SystemExit at startup. A
# silent fall back to blend_power=2.0 would render a half-adaptive arm that still
# produces 22 clips and a full 16-metric table, and nothing downstream could detect it.
#
# Do NOT run this before `smoke` passes. Gates 1+2 there are what prove --tau_map
# actually reaches blend_power rather than being parsed and discarded.
#
# Sampler config is the run_fivebench.py default set shared by R1/R7/R20/R21 --
# step 15, flow_shift 1.0, fg_boost 4, seed 0, chunk 21, overlap 1, sink 0. Only the
# exponent varies, and only per pair. --blend_power is deliberately NOT passed: the
# tau_map supplies it for every pair, and passing both would imply a fallback exists.
#
# Per-pair reseeding (run_fivebench.py:213-226) is what makes a 22-pair subset
# bit-comparable to the stored full-bench references, so seed / anchors / cases_json
# must match the reference exactly and none of the three may drift.
#
# --time=04:00:00: 22 clips at R22's measured ~5.3 min/clip is ~2h for the WHOLE arm;
# the largest single task (edit2, 13 clips) is ~70 min. 4h is generous margin.
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r25_infer
#SBATCH --partition=L40S
# node52 advertises gpu:8 but exposes no device to batch jobs (nvidia-smi -L ->
# "No devices found."), which killed two phase-1 submissions at CUDA init.
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/r25_infer_%A_%a.out
#SBATCH --error=logs/r25_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).


# adaptive — the phase-1 arm (already rendered; this is just the default)
# sbatch --export=R25_ARM=adaptive --array=0-5 slurm_scripts/five_bench/r25_infer.sh

# reversed — same 22 tau values, pairing inverted
# sbatch --export=R25_ARM=reversed --array=0-5 slurm_scripts/five_bench/r25_infer.sh

# constant — tau = 6.57 everywhere
# sbatch --export=R25_ARM=constant --array=0-5 slurm_scripts/five_bench/r25_infer.sh


REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r25_adaptive_tau
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES="$REPO/evaluation/cases.json"

# ARM SELECTION (phase 2). The three arms differ ONLY in which tau each pair gets;
# seed, anchors, cases_json, sampler defaults and the metric list are held identical,
# because the whole attribution argument rests on that. Unset R25_ARM reproduces the
# completed phase-1 adaptive run exactly.
#   adaptive  IoU -> tau            (phase 1)
#   reversed  same 22 tau values, pairing inverted by IoU rank -> does the PAIRING matter?
#   constant  tau = 6.57 everywhere                            -> does VARYING tau matter?
ARM=${R25_ARM:-adaptive} # set ARM to the value of R25_ARM; if R25_ARM is unset or empty, use adaptive instead. This does not modify R25_ARM itself — it only picks a value for ARM
case "$ARM" in
  adaptive|reversed|constant) ;;
  *) echo "[r25] bad R25_ARM='$ARM' (expected adaptive|reversed|constant)"; exit 1 ;;
esac
METHOD="r25_${ARM}_vp"
# The phase-1 map predates the naming scheme and keeps its original filename, so the
# adaptive arm must not be rewritten to r25_tau_map_adaptive.csv.
if [ "$ARM" = "adaptive" ]; then
  TAU_MAP="$REPO/evaluation/csv/r25_tau_map.csv"
else
  TAU_MAP="$REPO/evaluation/csv/r25_tau_map_${ARM}.csv"
fi

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 5 ]; then
  echo "[r25_infer] bad array id ${TID} (expected 0-5)"; exit 1
fi
T=$(( TID + 1 ))                       # task 0..5 -> edit type 1..6

# Expected clip count per edit type in cases.json (1/13/2/2/3/1 = 22).
EXPECTED=(1 13 2 2 3 1)
EXP=${EXPECTED[$TID]}

# Conda: use the SAME pattern as r25_smoke.sh, which is the one activation proven to
# work from this machine tonight. `cd; source .bashrc` (inherited from r21_infer.sh) was
# written for submission from a LOGIN shell; array job 960685 was submitted from inside a
# GPU allocation and all 6 tasks died at torch.cuda.current_device() before any render.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

# Diagnostics: job 960685 failed with "CUDA unknown error ... available devices zero" and
# the logs carried nothing to tell an env leak from a node fault. Print the GPU view the
# job actually has, so a repeat is diagnosable instead of another guess.
echo "[r25_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r25_infer] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
nvidia-smi -L 2>&1 | head -4
python -c "import torch; print('[r25_infer] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -2

if [ ! -f "$TAU_MAP" ]; then
  echo "[r25_infer] missing $TAU_MAP -- run the taumap step first"; exit 1
fi

echo "[r25_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} edit_type=${T} arm=${ARM} method=${METHOD}"
echo "[r25_infer] tau_map=${TAU_MAP} cases=${CASES} expect ${EXP} clip(s)"

FAILED=0
python evaluation/run_fivebench.py \
  --edit_type "$T" --method "$METHOD" \
  --vp_mode vp \
  --first_frame_edit_dir "$ANCHOR_ROOT" \
  --cases_json "$CASES" \
  --tau_map "$TAU_MAP" \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --step 15 --fg_boost_factor 4 --seed 0 \
  || { echo "[r25_infer] FAILED edit${T}"; FAILED=$((FAILED+1)); }

# Frame dirs for THIS edit type, excluding the *_resize dirs evaluate.py leaves behind.
N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/"edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r25_infer] edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
if [ "$N_DIRS" -ne "$EXP" ] && [ "$FAILED" -eq 0 ]; then
  echo "[r25_infer] COUNT-MISMATCH edit${T}: ${N_DIRS} != ${EXP}"
  FAILED=$((FAILED+1))
fi

echo "[r25_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
