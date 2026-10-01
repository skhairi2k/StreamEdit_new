#!/bin/bash
# R39 smoke -- 2 clips x {(N=15, rho=2), (N=20, rho=2), (N=20, rho=3), (N=30, rho=2)}, VP,
# then 3 sha256 gates via evaluation/r39_check_smoke.py:
#
# GATE1 (IDENTICAL) N=15 rho=2 == R36 r36_rho_sweep/r36_rho2_vp on both clips. Same flags as
#   r36_infer.sh except the method name, so this proves the R39 render path is unchanged
#   since R36 -- in particular that R38's _schedule_blend_rate edit (tgt<tau>, t_cur) left
#   the paper path (blend_sched None) untouched.
# GATE2 (DIFFERS)   N=20 rho=2 and N=30 rho=2 each != N=15 rho=2 on every frame, same frame
#   count, and N=30 != N=20 -- --step actually reaches the sampler at both budgets.
# GATE3 (DIFFERS)   N=20 rho=3 != N=20 rho=2 -- --blend_power still reaches the bridge at N=20.
#
# Clips: 0001_bus (edit1), 0007_guitar-violin (edit5) -- evaluation/r36_smoke_cases.json.
# Sampler must stay byte-identical to r36_infer.sh apart from --step: fg_boost 4,
# flow_shift 1.0, seed 0 (chunk 21 / overlap 1 / sink 0 are run_fivebench.py defaults).
# The step count actually run is visible in the tqdm totals in the .err (".../20 [", ".../30 [").
#SBATCH --job-name=r39_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/r39_smoke_%j.out
#SBATCH --error=logs/r39_smoke_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors"
OUT_ROOT="$FIVE_ROOT/r39_smoke"
CASES="$REPO/evaluation/r36_smoke_cases.json"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r39_smoke] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4

for CFG in "15 2" "20 2" "20 3" "30 2"; do
  set -- $CFG
  STEPS=$1
  R=$2
  for T in 1 5; do
    echo "--- [r39_smoke] N=${STEPS} rho=${R} edit${T} ---"
    python evaluation/run_fivebench.py \
      --edit_type "$T" --method "r39_n${STEPS}_rho${R}_vp" \
      --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
      --cases_json "$CASES" \
      --blend_power "$R" \
      --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
      --step "$STEPS" --fg_boost_factor 4 --flow_shift 1.0 --seed 0 \
      || { echo "SMOKE-ABORT: render failed N=${STEPS} rho=${R} edit${T}"; exit 1; }
  done
done

python evaluation/r39_check_smoke.py \
  --smoke_root "$OUT_ROOT" --smoke_cases "$CASES" \
  --r36_root "$FIVE_ROOT/r36_rho_sweep"
RC=$?
echo "[r39_smoke] gates exit=${RC}"
exit $RC
