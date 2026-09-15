#!/bin/bash
#SBATCH --job-name=r27_sweep_eval
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r27_sweep_eval_%j.out
#SBATCH --error=logs/r27_sweep_eval_%j.err
#
# Score the r27_tau_sweep renders of 0011_lucia_e5, one eval per tau.
#
# WHY: picking the calibration point by eye is exactly the kind of unjustified constant
# this task has already been burned by twice (the 0.005 top-slice carryover, the `auto`
# ceiling). The sweep's job is to place D_hi against a MEASURED trade-off, so it needs
# numbers next to the grid. The decision-relevant pair is:
#   clip_similarity_target_image        -- did the dog actually get added (edit strength)
#   lpips_unedit_part / psnr_unedit_part -- did the woman and background survive
# Their crossing point across tau IS the calibration curve.
#
# ⚠️ n = 1. These are single-clip numbers on one edit and carry no error bar. They are
# for locating a knee, not for claiming an effect; the 22-pair arms are what would
# support a claim. Read them WITH the grid, never instead of it.
#
# Same explicit 9-metric --metrics list as r25_eval.sh / r26: config.yaml's `metrics:`
# key is VESTIGIAL (evaluate.py:200 reads args.metrics), and a different set would break
# comparability with the stored R21/R25 references.
#
# NOTE: env is `five-bench`, NOT `streamgve` -- the eval stack is a separate env.
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh reads an unbound SYS_SYSROOT and
# aborts at `conda activate` (R20 job 907410; reproduced on this task's first sweep launch).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r27_tau_sweep
CASES="$REPO/evaluation/cases_lucia_e5.json"
TAUS="${R27_TAUS:-2 5 10 20 50 100}"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench
mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

echo "[r27_sweep_eval] node=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi -L 2>&1 | head -2
python -c "import torch; print('[r27_sweep_eval] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())" 2>&1 | tail -1
echo "[r27_sweep_eval] taus: ${TAUS}"

FAILED=0
for TAU in $TAUS; do
  STEM="r27_sweep_tau${TAU}"
  METHOD_DIR="$OUT_ROOT/$STEM"
  if [ ! -d "$METHOD_DIR" ]; then
    echo "[r27_sweep_eval] SKIP tau=${TAU} -- no render dir $METHOD_DIR"
    FAILED=$((FAILED+1)); continue
  fi

  MLOG="logs/r27_sweep_eval_${SLURM_JOB_ID:-local}_tau${TAU}.metrics.log"
  echo ""
  echo "=============== tau=${TAU}  $(date +%H:%M:%S) ==============="
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --metrics structure_distance psnr_unedit_part lpips_unedit_part \
              mse_unedit_part ssim_unedit_part clip_similarity_source_image \
              clip_similarity_target_image clip_similarity_target_image_edit_part \
              niqe_target_image \
    --src_image_folder "$DATA_ROOT" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --tgt_methods "$METHOD_DIR" \
    --tgt_layout edit_video \
    --cases_json "$CASES" \
    --result_path "evaluation/csv/${STEM}.csv" \
    2>&1 | tee "$MLOG"
  RC=${PIPESTATUS[0]}

  # Upstream swallows per-metric exceptions and still exits 0, so a clean exit code is
  # NOT evidence the CSV is clean: `except: continue` drops a metric column and shifts
  # every later one (how R20 corrupted five_acc). Any `Error:` line is a failure.
  NERR=$(grep -c 'Error:' "$MLOG")
  NOOM=$(grep -c 'out of memory' "$MLOG")
  echo "[r27_sweep_eval] tau=${TAU} exit=$RC error_lines=$NERR oom_lines=$NOOM"
  if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ]; then
    echo "[r27_sweep_eval] FAILED tau=${TAU} -- see $MLOG"; FAILED=$((FAILED+1))
  fi
done

echo ""
echo "[r27_sweep_eval] failures: ${FAILED}"
exit $(( FAILED > 0 ))
