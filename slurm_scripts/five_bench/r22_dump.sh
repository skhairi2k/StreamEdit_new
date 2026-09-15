#!/bin/bash
# R22 -- per-denoising-step video dump on the 22 clips of evaluation/cases.json.
#
# For every clip, one unperturbed rollout produces 15 videos: V_j = the assembly of
# each block's own step-j x0 prediction. Blocks still commit their step-14 value to
# the KV cache, so generation is untouched and V_14 IS the arm's real output --
# validated bit-for-bit by r22_smoke.sh before this ever runs.
#
# ARRAY LAYOUT (one arm per task; the model loads once per task):
#   0  paper_novp     Eq. 4, no anchor        -> reference render: five_bench/baseline (R1)
#   1  paper_vp       Eq. 4 + SS4.5 anchor    -> reference render: five_bench/r7_visual_prompting (R7)
#   2  cos_third_pvp  cos_third + R10 bank    -> reference render: r21_blend_full/cos_third_pvp (R21)
#
# Why these three arms: each has a stored FULL-BENCH render of the same
# post-seeding-fix generation (2026-07-22/23), so step14 of every clip can be
# byte-compared against it. That is what makes the matrix's bottom-right cell a
# checked fact rather than an assumption.
#
# The 22 clips span edit types 1, 2, 5, 6 (1/16/3/2), so each task loops those four;
# r22_dump_steps.py filters to the clips via --cases_json and fails loudly if any
# named clip is absent from the edit{T} json.
#
# Sampler settings are the run_fivebench.py defaults R1/R7/R20/R21 used -- in
# particular rollout_chunk_size stays 21, which is what keeps these renders
# comparable to the stored references. Under that setting:
#   novp / pvp : single window everywhere EXCEPT 0034_cows (24 latent frames).
#   vp         : 8 of 22 split, because independent_first_frame shortens window 1
#                by one block, pushing every 21-latent clip over.
# ⚠️ 0034_cows is the ONLY clip exercising the R22 overlap-slicing path
# (ret_step_list[-1][:, :, rollout_overlap:]) under novp/pvp. r22_smoke.sh used
# 0001_bus (18 latent frames, single window) and therefore could NOT cover it. A
# wrong slice axis there misaligns frames silently rather than crashing, so check
# that clip explicitly at wait-dump.
#
# Cost: the smoke measured ~4 min/clip (1 inference + 15 VAE decodes) => ~1.5 h/arm
# for 22 clips. The 6 h budget is deliberate headroom for the multi-window clips,
# which run one inference per window.
#
# Disk: ~546 MB per clip x 15 steps => ~12 GB/arm, ~36 GB total. /projects/dataggen
# had 9.9 TB free at launch, so this is not a constraint.
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r22_dump
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/r22_dump_%A_%a.out
#SBATCH --error=logs/r22_dump_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r22_step_dump
# Same anchor files R7/R10/R20/R21 read, so arms differ only by injection mechanism.
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES=$REPO/evaluation/cases.json

# The 22 clips span exactly these four edit types.
EDIT_TYPES=(1 2 5 6)
STEPS=15

# ARM   = output directory name (and the matrix's arm key)
# VPMODE= how the anchor enters (novp => no anchor at all)
# SCHED = blender-rate schedule; empty means the run_fivebench/pipeline default,
#         which is StreamGVE Eq. 4 unchanged. Do NOT pass `--blend_sched paper`
#         here just to be explicit -- it is the same code path, but leaving the
#         flag off is what the stored R1/R7 references were rendered with.
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARM=paper_novp;    VPMODE=novp; SCHED= ;;
  1) ARM=paper_vp;      VPMODE=vp;   SCHED= ;;
  2) ARM=cos_third_pvp; VPMODE=pvp;  SCHED=cos_third ;;
  *) echo "[r22_dump] bad array id ${SLURM_ARRAY_TASK_ID} (expected 0-2)"; exit 1 ;;
esac

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

echo "[r22_dump] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} arm=${ARM}"
echo "[r22_dump] vp_mode=${VPMODE} sched=${SCHED:-paper(eq4)} steps=${STEPS}"
echo "[r22_dump] edit_types=${EDIT_TYPES[*]} -> ${OUT_ROOT}/${ARM}"

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "[r22_dump] --- ${ARM} / edit${T} ---"
  ARGS=(--edit_type "$T" --method "$ARM"
        --cases_json "$CASES" --data_root "$DATA_ROOT" --out_root "$OUT_ROOT"
        --step "$STEPS" --fg_boost_factor 4 --blend_power 2 --seed 0)
  if [ "$VPMODE" != "novp" ]; then
    ARGS+=(--vp_mode "$VPMODE" --first_frame_edit_dir "$ANCHOR_ROOT")
  fi
  if [ -n "$SCHED" ]; then
    ARGS+=(--blend_sched "$SCHED")
  fi
  # r22_dump_steps.py exits 1 if ANY pair errored, so this guard actually fires.
  python evaluation/r22_dump_steps.py "${ARGS[@]}" || {
    echo "[r22_dump] FAILED ${ARM} / edit${T}"; FAILED=$((FAILED+1)); }
done

# 22 clips x 15 steps = 330 leaf frame dirs per arm.
N_DIRS=$(ls -d "$OUT_ROOT/$ARM"/step*/*/*/ 2>/dev/null | wc -l)
echo "[r22_dump] arm=${ARM} frame dirs: ${N_DIRS} (expect 330 = 22 clips x 15 steps)"
echo "[r22_dump] GATE1 lines: $(grep -c 'GATE1 PASS' logs/r22_dump_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2>/dev/null) (expect 22)"
echo "[r22_dump] failures: ${FAILED}"
exit $(( FAILED > 0 ))
