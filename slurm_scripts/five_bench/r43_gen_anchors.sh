#!/bin/bash
# R43 -- FiVE-Bench-Correct VP anchors with the SAME generator as R7 (job 897770):
# Qwen-Image-Edit-2511, H100, single-stage, CFG 3.0, 40 steps, seed 0. Only the instruction
# text differs from R7 (instruction_v2 + the keep-pose suffix on edit types 1-4, which is in
# the manifest's first_frame_edit_prompt; gen_anchors.py reads it from there).
#
#   sbatch slurm_scripts/five_bench/r43_gen_anchors.sh smoke
#       one task: repro manifest (3 R7 pairs, FiVE's own instruction, no suffix) -> anchor_r43_smoke/repro,
#       r43_check_repro.py against R7's anchors/, smoke manifest (26 cases) -> anchor_r43_smoke,
#       r43_contact_sheets.py --smoke
#   sbatch --array=0-5 slurm_scripts/five_bench/r43_gen_anchors.sh full
#       array task i generates edit type i+1 of the correct manifest -> anchors_r43_correct
#
# NO `conda activate`: the login .bashrc auto-activates streamgve (diffusers 0.31, no
# QwenImageEditPlusPipeline), so the qwen-edit env's python is called directly with -E -s.
# Resumable: existing PNGs are skipped. The #SBATCH log files only hold a pointer line; the
# real log is logs/r43_gen_anchors_<mode>_<jobid>_<task>.{out,err}.
#SBATCH --job-name=r43_gen_anchors
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r43_gen_anchors_%A_%a.out
#SBATCH --error=logs/r43_gen_anchors_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=$HOME/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=$HOME/Data/dataggen/outputs/five_bench
QWEN_PYTHON=/home/ids/skhairi/anaconda3/envs/qwen-edit/bin/python

MODE="${1:-}"
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
  echo "usage: sbatch [--array=0-5] r43_gen_anchors.sh smoke|full" >&2
  exit 2
fi

cd "$REPO" || exit 1
mkdir -p logs
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LOG="logs/r43_gen_anchors_${MODE}_${JOB}_${TASK}"
echo "[r43_gen_anchors] mode=${MODE} log -> ${LOG}.out / ${LOG}.err"
exec > "${LOG}.out" 2> "${LOG}.err"

# Generator flags, written out even though they equal gen_anchors.py's defaults (= R7's run).
# NO --two_stage, NO --cpu_offload: single-stage on the H100, as in R7.
GEN_FLAGS=(--true_cfg_scale 3.0 --guidance_scale 1.0 --num_inference_steps 40 --seed 0 --negative_prompt " ")

echo "[r43_gen_anchors] mode=${MODE} job=${JOB} task=${TASK} host=$(hostname)"
echo "[r43_gen_anchors] gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$QWEN_PYTHON" -E -s -c "import diffusers, transformers, torch; print('[r43_gen_anchors] diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"

# run_gen <cases.json> <out_dir> [extra gen_anchors args]; fails on any [ERROR line.
run_gen() {
  local cases="$1" out="$2"; shift 2
  local tmp; tmp=$(mktemp)
  "$QWEN_PYTHON" -u -E -s evaluation/gen_anchors.py --cases "$cases" --out "$out" \
    --data_root "$DATA_ROOT" "${GEN_FLAGS[@]}" "$@" 2>&1 | tee "$tmp"
  local rc=${PIPESTATUS[0]}
  local n_err; n_err=$(grep -c '^\[ERROR' "$tmp")
  rm -f "$tmp"
  if [ "$rc" -ne 0 ] || [ "$n_err" -ne 0 ]; then
    echo "[r43_gen_anchors] FAILED rc=${rc} errors=${n_err}"
    exit 1
  fi
}

if [ "$MODE" = "smoke" ]; then
  echo "[r43_gen_anchors] smoke 1/4: repro anchors (R7 pairs, official instruction, no suffix)"
  run_gen evaluation/r43_anchor_manifest_repro.json "$OUT_ROOT/anchor_r43_smoke/repro"
  echo "[r43_gen_anchors] smoke 2/4: generator check against R7 anchors/"
  "$QWEN_PYTHON" -E -s evaluation/r43_check_repro.py \
    --manifest evaluation/r43_anchor_manifest_repro.json \
    --r7_root "$OUT_ROOT/anchors" --out_csv evaluation/csv/r43_repro_check.csv || exit 1
  echo "[r43_gen_anchors] smoke 3/4: 26 smoke anchors (instruction_v2 + keep-pose suffix)"
  run_gen evaluation/r43_anchor_manifest_smoke.json "$OUT_ROOT/anchor_r43_smoke"
  echo "[r43_gen_anchors] smoke 4/4: contact sheets"
  "$QWEN_PYTHON" -E -s evaluation/r43_contact_sheets.py --smoke \
    --data_root "$DATA_ROOT" --r7_manifest evaluation/r7_anchor_manifest.json \
    --r7_root "$OUT_ROOT/anchors" --smoke_manifest evaluation/r43_anchor_manifest_smoke.json \
    --out_dir evaluation/figures || exit 1
else
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[r43_gen_anchors] full mode needs --array=0-5" >&2
    exit 2
  fi
  T=$((SLURM_ARRAY_TASK_ID + 1))
  echo "[r43_gen_anchors] full: edit_type=${T} -> $OUT_ROOT/anchors_r43_correct"
  run_gen evaluation/r43_anchor_manifest_correct.json "$OUT_ROOT/anchors_r43_correct" --edit_type "$T"
fi
echo "[r43_gen_anchors] finished mode=${MODE} task=${TASK}"
