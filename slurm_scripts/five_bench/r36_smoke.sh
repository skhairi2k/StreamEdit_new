#!/bin/bash
# R36 smoke -- 2 clips x rho {2,3}, VP, then 3 sha256 gates via evaluation/r36_check_parity.py:
#
# GATE1 (IDENTICAL) rho=3 == R26 taubg3_taufg3_vp on both clips. R26's diagonal cells ran
#   the same scalar --blend_power path with the same per-pair seed, sampler and anchors, so
#   this proves the pipeline has not moved since R26 and that r36_infer.sh's flags match.
# GATE2 (IDENTICAL) rho=2 == R7 r7_visual_prompting on both clips (both are index 0 of
#   their edit{T}_FiVE.json, so the pre-reseed R7 render drew the same noise).
# GATE3 (DIFFERS)   rho=3 != rho=2 -- --blend_power actually reaches the bridge; without
#   this, GATE1/GATE2 are also satisfied by a flag that is parsed then ignored.
#
# Clips: 0001_bus (edit1), 0007_guitar-violin (edit5) -- evaluation/r36_smoke_cases.json.
# Sampler must stay byte-identical to r36_infer.sh / R26 / R7: step 15, fg_boost 4,
# flow_shift 1.0, seed 0 (chunk 21 / overlap 1 / sink 0 are run_fivebench.py defaults).
#SBATCH --job-name=r36_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r36_smoke_%j.out
#SBATCH --error=logs/r36_smoke_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors"
OUT_ROOT="$FIVE_ROOT/r36_smoke"
CASES="$REPO/evaluation/r36_smoke_cases.json"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r36_smoke] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4

for R in 2 3; do
  for T in 1 5; do
    echo "--- [r36_smoke] rho=${R} edit${T} ---"
    python evaluation/run_fivebench.py \
      --edit_type "$T" --method "r36_rho${R}_vp" \
      --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
      --cases_json "$CASES" \
      --blend_power "$R" \
      --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
      --step 15 --fg_boost_factor 4 --flow_shift 1.0 --seed 0 \
      || { echo "SMOKE-ABORT: render failed rho=${R} edit${T}"; exit 1; }
  done
done

python evaluation/r36_check_parity.py --smoke \
  --smoke_root "$OUT_ROOT" --smoke_cases "$CASES" \
  --r26_root "$FIVE_ROOT/r26_spatial_tau" \
  --r7_root "$FIVE_ROOT/r7_visual_prompting" \
  --data_root "$DATA_ROOT"
RC=$?
echo "[r36_smoke] parity exit=${RC}"
exit $RC
