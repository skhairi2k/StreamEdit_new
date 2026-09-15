#!/bin/bash
#SBATCH --job-name=r27_tau_sweep
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r27_tau_sweep_%j.out
#SBATCH --error=logs/r27_tau_sweep_%j.err
#
# R27 -- single-clip tau sweep on 0011_lucia_e5 ("a woman in a black dress" -> the same
# woman "followed closely by a dog": an ADDITION, and the case that motivated R27).
#
# WHY THIS EXISTS. The `taumap` step's ceiling D_hi is not derivable from r27_depth.csv.
# D_hi answers "how much depth movement means the source must be abandoned entirely",
# which is a property of the RENDER, not of the depth statistics -- so it has to be
# calibrated against renders. This sweep is that calibration, and it was listed as an
# Open question in the plan from the start.
#
# It answers two things the depth CSV cannot:
#   1. A CALIBRATION POINT. If e5 looks right at tau 5, set D_hi so its D_norm = 0.784
#      maps there; if it needs 50, set D_hi ~ 0.79 and accept that most of the set clips.
#   2. WHETHER TAU IS EVEN THE RIGHT KNOB FOR AN ADDITION. No D_hi changes e5's RANK
#      (20/22) because the budget map is monotone. If e5 fails at every tau, the premise
#      that this edit needs detachment is what is wrong, not the mapping -- which is
#      exactly the call the `verdict` gate was rewritten to require.
#
# tau reaches the model as --blend_power (run_fivebench.py:381 falls back to the scalar
# when no --tau_map is given), so no map file is needed and nothing about the R27 arms
# is touched. Every other flag is IDENTICAL to r25_infer.sh, including --src_frames
# defaulting to 'video': the point is comparability with the stored arms, and a different
# source rendition is worth LPIPS 0.173 on its own.
#
# tau = 100 is included beyond the map's tau_max = 50 deliberately: at --step 15, tau 50
# still leaves W_src = 0.031 at the first blended step, so 50 may not be saturation. If
# 50 and 100 are indistinguishable, tau_max = 50 is a safe ceiling; if 100 differs, the
# ceiling is cutting into live range.
#
# Runnable two ways -- sbatch, or directly inside an already-held allocation (which is
# how the `depth` step ran on job 965047: this session lives inside the allocation, so a
# nested srun --pty would be a job step with no tty).

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and aborts at `conda activate` (R20 job 907410, and again here on
# the first launch of this script). r25_infer.sh carries the same note.

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r27_tau_sweep
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES="$REPO/evaluation/cases_lucia_e5.json"
TAUS="${R27_TAUS:-2 5 10 20 50 100}"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r27_sweep] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r27_sweep] SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
nvidia-smi -L 2>&1 | head -4
python -c "import torch; print('[r27_sweep] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -2
echo "[r27_sweep] taus: ${TAUS}"

FAILED=0
for TAU in $TAUS; do
  METHOD="r27_sweep_tau${TAU}"
  echo ""
  echo "=============================================================="
  echo "[r27_sweep] tau=${TAU}  method=${METHOD}  $(date +%H:%M:%S)"
  echo "=============================================================="
  python evaluation/run_fivebench.py \
    --edit_type 5 --method "$METHOD" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --cases_json "$CASES" \
    --blend_power "$TAU" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --seed 0 \
    || { echo "[r27_sweep] FAILED tau=${TAU}"; FAILED=$((FAILED+1)); continue; }

  N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/edit5/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r27_sweep] tau=${TAU} frame dirs: ${N_DIRS} (expected 1)"
  [ "$N_DIRS" -ne 1 ] && { echo "[r27_sweep] COUNT-MISMATCH tau=${TAU}"; FAILED=$((FAILED+1)); }
done

echo ""
echo "[r27_sweep] failures: ${FAILED}"
exit $(( FAILED > 0 ))
