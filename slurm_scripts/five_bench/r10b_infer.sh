#!/bin/bash
# R10b -- the persistent anchor bank WITH blending on. ONE new arm.
#
# WHY THIS RUN EXISTS
# -------------------
# R10 answered the routing question (negative: head identity moves anchor
# adherence by +0.053 at frame 0 and by +0.0004 -- nothing -- averaged over the
# clip, versus +0.206 for simply injecting everywhere). What it could NOT answer
# is the question the design rests on: does the persistent bank destroy motion?
#
# It could not, because every R10 arm ran blend_off=True, disabling QUERY
# BLENDING -- the mechanism the paper (SS4.2) credits with structure and motion
# preservation. With it off no arm had motion transfer to lose, and the numbers
# show exactly that: edit-part motion fidelity sits at ~0.68 for
# none/all/spatial/temporal alike while R7 (SS4.5, blending ON) scores 0.837.
# That gap is what blending supplies; the anchor never got the chance to damage it.
#
# ONLY ONE ARM IS RENDERED
# ------------------------
# The two references already exist and were produced with the SAME sampler
# settings (--step 15 --fg_boost_factor 4 --blend_power 2 --seed 0) and the same
# {root}/edit{T}/{video} layout:
#   no anchor, blending on   -> five_bench/baseline            (r1_infer.sh)
#   SS4.5 anchor, blending on -> five_bench/r7_visual_prompting (r7_infer.sh)
# Rendering a `none` arm here would just re-make `baseline`. So this run adds the
# single missing cell: persistent bank + blending on.
#
# With blending on, all three finally share a base config and differ only in the
# injection mechanism -- none / cached-initial-latent / persistent re-roped bank.
# That is the first controlled comparison of the three; in R10 the r7 arm was
# confounded on BOTH motion and appearance because it alone kept blending.
#
# WHAT DECIDES WHAT (score against baseline and r7_visual_prompting)
#   appearance  anchor_sim_mean -- does the bank still hold the anchor once
#               blending pulls the output toward the source? (R10, blend_off:
#               all 0.817 vs none 0.618, r7 0.628)
#   motion      motion_fidelity_score_edit_part -- does it fall below baseline?
#               (R10: -0.019, p=0.37, but with no motion present to lose)
#
#   holds appearance AND motion -> working method; routing was a detour
#   holds appearance, kills motion -> tension is real and localised; build the
#                                     warped/propagated anchor
#   loses appearance -> R10's persistence gain was a blend_off artifact
#
# DRIVER-EQUIVALENCE CHECK (stage 2, 3 clips)
# -------------------------------------------
# `baseline` comes from a DIFFERENT driver (run_fivebench.py) than this run
# (r10_vp_arms.py). Sampler flags match, but the drivers could still differ in
# truncation or rollout handling, which would make baseline-vs-bank a driver
# comparison rather than a mechanism one. Stage 2 renders `none` through THIS
# driver on 3 clips; if those frames match `baseline`, the substitution is sound.
# Cheap insurance -- ~3 minutes against a silent confound.
#
# NOTE: r1_infer.sh does not pass --flow_shift while r7_infer.sh passes 1.0.
# Verify that is run_fivebench.py's default before treating the two references
# as sampler-identical.
#
# Separate OUT_ROOT: R10's frames are the blend_off half of this comparison and
# must not be overwritten.
#SBATCH --job-name=r10b_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/r10b_infer_%j.out
#SBATCH --error=logs/r10b_infer_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r10b_blend_on
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors

cd
source .bashrc
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

# Same 20 clips as R10 and as baseline/r7, so every comparison is paired
# clip-for-clip.
CASES=(0001_bus \
       0028_kite-walk 0040_tennis 0057_dog 0011_lucia_e2 0014_burnout \
       0016_horsejump-high 0068_planes-water 0072_dog-agility 0074_rhino \
       0075_A_bicycle 0076_A_rabbit 0079_A_bus 0090_A_deer 0091_A_hawk \
       0007_guitar-violin 0069_car-turn 0011_lucia_e5 \
       0042_gym-ball 0002_girl-dog)
if [ "$#" -gt 0 ]; then CASES=("$@"); fi

echo "[r10b] job=${SLURM_JOB_ID} cases=${#CASES[@]} arm=all BLENDING=ON -> ${OUT_ROOT}"

python evaluation/r10_vp_arms.py \
  --cases "${CASES[@]}" \
  --arms all \
  --keep_blending \
  --gates evaluation/r19_head_gates.pt \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --anchor_root "$ANCHOR_ROOT" \
  --step 15 \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --seed 0

echo "[r10b] stage 2 -- driver-equivalence check vs five_bench/baseline (3 clips)"
python evaluation/r10_vp_arms.py \
  --cases 0057_dog 0001_bus 0074_rhino \
  --arms none \
  --keep_blending \
  --gates evaluation/r19_head_gates.pt \
  --data_root "$DATA_ROOT" \
  --out_root "$OUT_ROOT" \
  --anchor_root "$ANCHOR_ROOT" \
  --step 15 \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --seed 0

echo "[r10b] frame dirs: $(ls -d "$OUT_ROOT"/*/*/*/ 2>/dev/null | wc -l) (expect 23 = 20 all + 3 none)"
echo "[r10b] compare $OUT_ROOT/none/edit2/0057_dog against /projects/dataggen/outputs/five_bench/baseline/edit2/0057_dog"
