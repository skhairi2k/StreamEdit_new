#!/bin/bash
# R21 completion: the one missing cell in the arm grid -- cos_third x novp.
#
# R21 was originally scoped vp/pvp only, so `paper` ended up with all three VP
# modes (novp came free from R1's `baseline`, vp from R7, pvp from the r21 render)
# while `cos_third` had only vp/pvp. Without this arm the schedule effect cannot be
# read at novp, where the two references are strongest. R20 has a cos_third_novp but
# only on its 22-clip subset -- not comparable to these 419-pair rows.
#
# NO --first_frame_edit_dir: novp ignores the anchor, and run_fivebench.py forces
# vp_mode=novp when no anchor dir is given, so the flag is passed explicitly only
# to make the intent legible in the log.
#
# Everything else is byte-for-byte the r21_infer.sh recipe (step 15, fg_boost 4,
# blend_power 2, seed 0, per-pair reseed, no --cases_json) so this row drops
# straight into the existing table.
#SBATCH --job-name=r21_novp
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/r21_novp_infer_%j.out
#SBATCH --error=logs/r21_novp_infer_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r21_blend_full

SCHED=cos_third
VP_MODE=novp
METHOD="${SCHED}_${VP_MODE}"
EDIT_TYPES=(1 2 3 4 5 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

echo "[r21_novp] job=${SLURM_JOB_ID} arm=${METHOD} sched=${SCHED} vp_mode=${VP_MODE}"
echo "[r21_novp] edit_types=${EDIT_TYPES[*]} -> ${OUT_ROOT}/${METHOD}/ (full bench, no --cases_json)"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "[r21_novp] --- ${METHOD} / edit${T} ---"
  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --blend_sched "$SCHED" --vp_mode "$VP_MODE" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0 \
    || { echo "[r21_novp] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
done

N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r21_novp] ${METHOD} real frame dirs (all edit types): ${N_DIRS}  # expect 419"
echo "[r21_novp] failures: ${FAILED}"
exit $(( FAILED > 0 ))
