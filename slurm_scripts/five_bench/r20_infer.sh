#!/bin/bash
# R20 -- blend-rate schedule sweep, 3 cosine schedules x 3 VP modes on the 22
# clips of evaluation/cases.json.
#
# ARRAY LAYOUT (one VP mode per task, all schedules for that mode inside it, so
# the model loads once per task):
# PHASE 1 -- the cosine sweep (the actual experiment):
#   0  novp  x {cos_full, cos_half, cos_third}   66 runs
#   1  vp    x {cos_full, cos_half, cos_third}   66 runs
#   2  pvp   x {cos_full, cos_half, cos_third}   66 runs
#   3  pvp   x {paper}                           22 runs  <- REFERENCE arm
# PHASE 2 -- degenerate sanity anchors, only worth running once phase 1 lands:
#   4  novp  x {const, zero}                     44 runs
#   5  vp    x {const, zero}                     44 runs
#   6  pvp   x {const, zero}                     44 runs
#
# Why task 3 exists: the paper-schedule references for novp and vp already exist
# on disk (five_bench/baseline from R1, five_bench/r7_visual_prompting from R7,
# both verified present for all 22 clips). There is NO config-matched pvp+paper
# run -- R10b's `all` arm went through r10_vp_arms.py with an all-360 head gate --
# so without task 3 the pvp column would have no baseline to be read against.
#
# What phase 2 bounds. `_schedule_blend_rate` returns the SOURCE weight s(p), and
# the bridge uses blender_rate = 1 - s(p):
#   const -> s=1 always -> blender_rate=0 -> queries and keys taken ENTIRELY from
#            the source branch at every step. Output should collapse toward a
#            reconstruction of the source video: the "no edit happened" floor.
#   zero  -> s=0 always -> blender_rate=1 -> no blending at all, though the
#            background source-KV injection still runs. This is R10's `blend_off`
#            condition, so it should land close to those arms -- a cross-check
#            against a run built by a completely different driver.
# Together they bracket the cosine schedules: no schedule should score outside
# [const, zero] on preservation, and one that does means the plumbing is wrong,
# not that the schedule is clever.
#
# Submission:
#   sbatch --array=0,1,2,3 slurm_scripts/five_bench/r20_infer.sh   # phase 1
#   sbatch --array=4,5,6   slurm_scripts/five_bench/r20_infer.sh   # phase 2
#
# The 22 clips span edit types 1, 2, 5, 6 (1/16/3/2), so each task loops those
# four types; run_fivebench.py filters to the clips via --cases_json and fails
# loudly if any named clip is absent from the edit{T} json.
#
# All sampler settings below are the run_fivebench.py defaults R1/R7 used --
# rollout_chunk_size stays at 21, which is what makes these arms comparable to
# the stored references. Windowing under that setting:
#   novp / pvp : single window everywhere EXCEPT 0034_cows (24 latent frames).
#   vp         : 8 of 22 split, because independent_first_frame shortens window 1
#                by one block, pushing every 21-latent clip over.
# 0034_cows is therefore the only clip that exercises persistent VP ACROSS a
# window boundary -- the one real test of the bank being rebuilt per window. It
# is also the clip that crashed R9 on the single-window path (cache overflow at
# 24 > 21 latent frames), which is exactly why this must NOT be "simplified" to
# rollout_chunk_size=-1.
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r20_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/r20_infer_%A_%a.out
#SBATCH --error=logs/r20_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r20_blend_sched
# Same anchor files R7 and R10 read, so arms differ only by injection mechanism.
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES=$REPO/evaluation/cases.json

# The 22 clips span exactly these four edit types.
EDIT_TYPES=(1 2 5 6)

case "${SLURM_ARRAY_TASK_ID}" in
  0) VP_MODE=novp; SCHEDS=(cos_full cos_half cos_third) ;;
  1) VP_MODE=vp;   SCHEDS=(cos_full cos_half cos_third) ;;
  2) VP_MODE=pvp;  SCHEDS=(cos_full cos_half cos_third) ;;
  3) VP_MODE=pvp;  SCHEDS=(paper) ;;
  4) VP_MODE=novp; SCHEDS=(const zero) ;;
  5) VP_MODE=vp;   SCHEDS=(const zero) ;;
  6) VP_MODE=pvp;  SCHEDS=(const zero) ;;
  *) echo "[r20_infer] bad array id ${SLURM_ARRAY_TASK_ID} (expected 0-6)"; exit 1 ;;
esac

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

echo "[r20_infer] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} vp_mode=${VP_MODE}"
echo "[r20_infer] scheds=${SCHEDS[*]} edit_types=${EDIT_TYPES[*]} -> ${OUT_ROOT}"

FAILED=0
for SCHED in "${SCHEDS[@]}"; do
  # Arm directory is {sched}_{vp_mode}; the reference arm is therefore `paper_pvp`.
  METHOD="${SCHED}_${VP_MODE}"
  for T in "${EDIT_TYPES[@]}"; do
    echo "[r20_infer] --- ${METHOD} / edit${T} ---"
    ARGS=(--edit_type "$T" --method "$METHOD" --blend_sched "$SCHED"
          --cases_json "$CASES" --data_root "$DATA_ROOT" --out_root "$OUT_ROOT"
          --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0)
    if [ "$VP_MODE" != "novp" ]; then
      ARGS+=(--vp_mode "$VP_MODE" --first_frame_edit_dir "$ANCHOR_ROOT")
    fi
    python evaluation/run_fivebench.py "${ARGS[@]}" || {
      echo "[r20_infer] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
  done
done

N_DIRS=$(ls -d "$OUT_ROOT"/*_"${VP_MODE}"/*/*/ 2>/dev/null | wc -l)
echo "[r20_infer] vp_mode=${VP_MODE} frame dirs matching *_${VP_MODE}: ${N_DIRS}"
echo "[r20_infer] (tasks 2/3/6 all write *_pvp -- 66, then 88, then 132 cumulative)"
echo "[r20_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
