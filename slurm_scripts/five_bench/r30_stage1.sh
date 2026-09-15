#!/bin/bash
# R30 stage 1 -- see .claude/plans/r30-static-anchor-divergence_8b3f52d1.plan.md
#
# Eight STATIC anchor-vs-source divergences over the 22 cases, then eight b maps.
# SINGLE PHASE, SINGLE TASK, and NO DIFFUSION ROLLOUT AT ALL.
#
# There is no R30_PHASE=dump here on purpose. The plan's earlier shape had stage 1 run an
# unblended pass to dump masks; the revised plan reuses R26's, because r26_dump.sh already
# ran the identical spec -- --blend_sched zero --union_dump_dir --vp_mode vp
# --first_frame_edit_dir $ANCHOR_ROOT --cases_json evaluation/cases.json --step 15
# --fg_boost_factor 4 --seed 0 -- covering 22/22 cases and dumped 2026-08-27, AFTER
# cases.json's final append on 2026-08-26. So stage 1 loads no video model; it runs LPIPS,
# DINOv2, CLIP, Depth-Anything and Marigold on two still images per case.
#
# THE BUDGET-RESPONSE PRE-CHECK IS DISCARDED (2026-09-10), which is why no rollout sweep
# appears here. The plan's Open item asked whether realized edit strength is linear in
# injected budget, to be validated by sweeping one clip over b in {2, 3, 3.7, 4.5, 5.5, 7,
# 10, 16, 50}. R26's 176 stored videos already answer it at zero GPU cost: FiVE-Acc N/44
# against A_disc(b) reads 28, 30, 34, 34, 34, 35, 36, 37 at A = .300 .218 .168 .111 .080
# .061 .021 .0021, against an endpoint straight line of 28.0 30.5 32.0 33.7 34.6 35.2 36.4
# 37.0 -- a maximum departure of 2.0 on a 9-point span, 22%. That is a RAMP, not a knee,
# so budget-linear stands. Two caveats carried forward into the stage-2 read: N is flat at
# 34 across b in [4, 8], so deciles d = 0.5/0.6/0.7 buy nothing and arms differing only
# there will look alike for reasons unrelated to the measures; and edit CLIP is flat above
# b = 4 (21.38, 22.16, 22.68, then 22.40, 22.45, 22.29, 22.39, 22.21), which is why it is
# a secondary axis. Not bought: the per-clip CONTINUOUS strength curve -- the aggregate
# ramp is a mixture of 22 step functions -- but the aggregate is what decides whether
# routing can help at all.
#
#SBATCH --job-name=r30_stage1
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r30_stage1_%j.out
#SBATCH --error=logs/r30_stage1_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
MASK_ROOT=/projects/dataggen/outputs/five_bench/r26_masks
CASES="$REPO/evaluation/cases.json"

DIV_CSV="$REPO/evaluation/csv/r30_divergence.csv"
VIZ_DIR="$REPO/evaluation/figures/r30_divergence_panels"

# Must match the render step and R26's stored grid. A_disc is a mean over THIS schedule,
# so a b map built at one step count silently changes meaning at another.
STEP=15
B_MIN=2
B_MAX=50
SEED=0

cd "$REPO" || exit 1

# The eval env is five-bench, NOT streamgve (.bashrc auto-activates streamgve and
# `conda activate five-bench` alone does not pop it -- the R2 eval died on
# ModuleNotFoundError: torchmetrics). five-bench is also the only env with lpips
# installed and with diffusers >= 0.36, which Marigold needs.
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv evaluation/figures

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TORCH_HOME=~/.cache/torch
export HF_HUB_OFFLINE=1

echo "[r30_stage1] node=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L 2>&1 | head -4

# ---- Guard 0: inputs ------------------------------------------------------------------
for P in "$CASES" "$ANCHOR_ROOT" "$MASK_ROOT"; do
  if [ ! -e "$P" ]; then echo "[r30_stage1] FAILED -- missing $P"; exit 1; fi
done

# 22 masks, one per case. A short mask set would still produce a full-looking CSV for the
# cases it covers and then fail 22-pair coverage at render time, hours later.
NMASK=$(find "$MASK_ROOT" -name '*.npz' -type f | wc -l)
if [ "$NMASK" -ne 22 ]; then
  echo "[r30_stage1] FAILED -- expected 22 union masks under $MASK_ROOT, found ${NMASK}."
  exit 1
fi

# check-div gate 0, mask provenance: the masks must POSTDATE cases.json's last append,
# else they describe a different case set than the one being measured.
if [ "$CASES" -nt "$(find "$MASK_ROOT" -name '*.npz' -type f -printf '%T@ %p\n' \
     | sort -n | head -1 | cut -d' ' -f2-)" ]; then
  echo "[r30_stage1] FAILED -- cases.json is NEWER than the oldest mask in $MASK_ROOT."
  echo "[r30_stage1]   A mask from before the final cases.json append describes a"
  echo "[r30_stage1]   different case set. Re-dump with r26_dump.sh, or confirm by hand."
  exit 1
fi

# ---- Guard 1: model caches ------------------------------------------------------------
# HF_HUB_OFFLINE=1 means a cache miss is a hard failure at model load. Checking here costs
# a second and saves losing the queue slot 20 minutes in.
HUB=~/.cache/huggingface/hub
MISSING=0
check_cache () {   # $1 = repo id, $2 = arms it serves
  local dir="$HUB/models--${1//\//--}"
  if [ ! -d "$dir" ]; then
    echo "[r30_stage1]   MISSING $1   (arms: $2)"
    echo "[r30_stage1]     hf download $1"
    MISSING=1
  fi
}
check_cache "facebook/dinov2-with-registers-base" "dino_cls, dino_patch, selfsim"
check_cache "prs-eth/marigold-normals-v1-1"       "normals"
check_cache "openai/clip-vit-large-patch14"       "clip_image, clip_prompt"
check_cache "depth-anything/Depth-Anything-V2-Large-hf" "depth"
if [ ! -f ~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth ]; then
  echo "[r30_stage1]   MISSING alexnet checkpoint for LPIPS (arm: lpips)"
  MISSING=1
fi
if [ "$MISSING" -ne 0 ]; then
  echo "[r30_stage1] FAILED -- pre-fetch the above on a LOGIN node, then resubmit."
  echo "[r30_stage1]   Jobs run with HF_HUB_OFFLINE=1, so a miss cannot self-heal."
  exit 1
fi

# ---- Step 1: the eight divergences ----------------------------------------------------
echo "[r30_stage1] === divergences (8 arms x 22 cases, no rollout) ==="
python evaluation/r30_divergence.py \
  --cases "$CASES" \
  --data_root "$DATA_ROOT" \
  --anchor_root "$ANCHOR_ROOT" \
  --mask_dir "$MASK_ROOT" \
  --viz_dir "$VIZ_DIR" \
  --seed "$SEED" \
  -o "$DIV_CSV"
RC=$?

# RC != 0 here is NOT automatically fatal to this script: r30_divergence.py returns 1
# whenever ANY arm has ANY per-clip gap (e.g. depth on 0042_gym-ball, whose union mask is
# the whole frame with no background to align on) -- a failure deliberately scoped to
# ONE arm so it does not cost the other seven their run (see r30_divergence.py's
# docstring on `notes`). Aborting the whole job on that exit code would undo exactly the
# per-arm scoping the script exists to provide, and it is not hypothetical: job 988591
# (2026-09-10) hit precisely this -- depth failed on 0042_gym-ball, RC=1, and the OLD
# version of this guard threw away all 22 rows of a perfectly good CSV, writing ZERO b
# maps for the seven arms that had no problem at all.
#
# The real signal is whether the CSV was actually written with a full 22 rows -- that
# happens unconditionally at the end of r30_divergence.py's main(), before its own
# failure check, so it survives a per-arm-scoped RC=1. Only a MISSING or SHORT csv (a
# crash before the write, or at argument/model-load time) is a hard stop here.
if [ ! -f "$DIV_CSV" ]; then
  echo "[r30_stage1] FAILED -- ${DIV_CSV} was not written (r30_divergence.py exit ${RC})."
  exit 1
fi
N_DIV=$(($(wc -l < "$DIV_CSV") - 1))
if [ "$N_DIV" -ne 22 ]; then
  echo "[r30_stage1] FAILED -- ${DIV_CSV} has ${N_DIV} rows, expected 22 (r30_divergence.py exit ${RC})."
  exit 1
fi
if [ "$RC" -ne 0 ]; then
  echo "[r30_stage1] NOTE -- r30_divergence.py exited ${RC} (per-arm failure(s) scoped"
  echo "[r30_stage1]   to specific clips, see the log above). ${DIV_CSV} has all 22 rows,"
  echo "[r30_stage1]   so proceeding to the b maps -- the affected arm(s) will be refused"
  echo "[r30_stage1]   individually by r30_b_map.py below, the rest are unaffected."
fi

# ---- Step 2: one b map per arm --------------------------------------------------------
# A flat list, not a nested arm x reduction loop: only the MASKED reduction earns a
# rollout set (spatial selectivity is already carried by the stage-2 mask, so b need only
# encode per-pixel intensity), and the global column is context for reading results.
ARMS=(lpips dino_cls dino_patch clip_image clip_prompt depth normals selfsim)

FAILED=0
for ARM in "${ARMS[@]}"; do
  echo "[r30_stage1] === b map: ${ARM} ==="
  python evaluation/r30_b_map.py \
    --arm "$ARM" \
    --divergence "$DIV_CSV" \
    --reduction masked \
    --b_min "$B_MIN" --b_max "$B_MAX" \
    --step "$STEP" \
    -o "evaluation/csv/r30_b_map_${ARM}.csv"
  ARC=$?
  # A refusal here is INFORMATIVE, not a crash: r30_b_map.py exits non-zero when an arm
  # yields fewer than 3 distinct b, which means that arm carries no routing signal and is
  # not worth 22 rollouts. Keep going so the run reports every arm's verdict at once.
  [ "$ARC" -ne 0 ] && { echo "[r30_stage1]   ${ARM}: REFUSED (exit ${ARC})"; FAILED=1; }
done

# ---- Report ---------------------------------------------------------------------------
echo
echo "[r30_stage1] ---- b maps written ----"
for ARM in "${ARMS[@]}"; do
  F="evaluation/csv/r30_b_map_${ARM}.csv"
  if [ -f "$F" ]; then
    echo "[r30_stage1]   ok      ${ARM}  ($(($(wc -l < "$F") - 1)) rows)"
  else
    echo "[r30_stage1]   ABSENT  ${ARM}"
  fi
done

echo
echo "[r30_stage1] divergences -> $DIV_CSV"
echo "[r30_stage1] panels      -> $VIZ_DIR"
if [ "$FAILED" -ne 0 ]; then
  echo "[r30_stage1] one or more arms produced no usable map -- read the refusals above."
  echo "[r30_stage1] Next: /run-step R30 check-div  (the arm set may need trimming)"
  exit 1
fi
echo "[r30_stage1] DONE -- all 8 arms mapped."
echo "[r30_stage1] Next: /run-step R30 check-div"
