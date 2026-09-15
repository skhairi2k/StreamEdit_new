#!/bin/bash
# R22 smoke -- validate the per-step dump path on ONE clip before committing ~6h of
# GPU to the 3-arm dump.
#
# Clip: 0001_bus, edit type 1, arm paper_novp (no anchor, Eq. 4) -- i.e. exactly the
# R1 baseline configuration. cases.json holds exactly one edit_type=1 entry
# (0001_bus), so --cases_json + --edit_type 1 selects it without a bespoke json.
#
# GATE1 (latent, printed by the driver): the last denoising step's latent must BE the
#   returned rollout latent, not merely resemble it. Asserted three times -- per window
#   and after stitching inside the pipeline, then again in the driver.
#
# GATE2 (pixel, checked here -- THE SUBSTANTIVE ONE): the step14 PNGs must be
#   BIT-IDENTICAL to the stored full-bench reference render. This is what proves
#     (a) the 15 sequential VAE decodes do not carry temporal-conv state into each
#         other (if they did, step 14 would differ from a fresh single decode), and
#     (b) adding the dump did not perturb sampling.
#   The reference is five_bench/baseline (R1), which is the same post-seeding-fix
#   generation (2026-07-22) -- per-pair reseeding is what makes a 1-clip subset run
#   reproduce a full-bench render byte for byte.
#
# A GATE2 FAIL is a STOP condition: it means the step-14 render is not the arm's real
# output, so the whole bottom-right identity is unsound. Do not launch `dump`.
#
# Writes to a DEDICATED smoke root, not r22_step_dump, so a partial single-clip tree
# cannot skew wait-dump's 330-dir-per-arm count.
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r22_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r22_smoke_%j.out
#SBATCH --error=logs/r22_smoke_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
SMOKE_ROOT=/projects/dataggen/outputs/five_bench/r22_smoke
REF_DIR=/projects/dataggen/outputs/five_bench/baseline/edit1/0001_bus
CASES=$REPO/evaluation/cases.json

ARM=paper_novp
CLIP=0001_bus
STEPS=15
LAST=$(printf "step%02d" $((STEPS - 1)))

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs

echo "[r22_smoke] arm=${ARM} clip=${CLIP} steps=${STEPS} -> ${SMOKE_ROOT}"

# paper_novp == no --first_frame_edit_dir (forces novp) and no --blend_sched (Eq. 4).
# Every other sampler flag is the run_fivebench.py default R1 used.
python evaluation/r22_dump_steps.py \
  --edit_type 1 \
  --method "$ARM" \
  --cases_json "$CASES" \
  --data_root "$DATA_ROOT" \
  --out_root "$SMOKE_ROOT" \
  --step "$STEPS" \
  --fg_boost_factor 4 \
  --blend_power 2 \
  --seed 0
RC=$?

if [ $RC -ne 0 ]; then
  echo "[r22_smoke] GATE1 FAIL -- driver exited ${RC} (see the assert message above)"
  echo "[r22_smoke] GATE2 SKIPPED"
  exit 1
fi
echo "[r22_smoke] GATE1 PASS -- see the per-clip GATE1 line from the driver above"

# --- GATE2: bit-parity of the final step against the stored R1 reference ---
DUMP_DIR="${SMOKE_ROOT}/${ARM}/${LAST}/edit1/${CLIP}"

if [ ! -d "$REF_DIR" ]; then
  echo "[r22_smoke] GATE2 FAIL -- reference render missing: ${REF_DIR}"
  exit 1
fi
if [ ! -d "$DUMP_DIR" ]; then
  echo "[r22_smoke] GATE2 FAIL -- dump dir missing: ${DUMP_DIR}"
  exit 1
fi

N_DUMP=$(ls "$DUMP_DIR"/*.png 2>/dev/null | wc -l)
N_REF=$(ls "$REF_DIR"/*.png 2>/dev/null | wc -l)
if [ "$N_DUMP" -eq 0 ] || [ "$N_DUMP" -ne "$N_REF" ]; then
  echo "[r22_smoke] GATE2 FAIL -- frame count ${N_DUMP} (dump) != ${N_REF} (ref)"
  exit 1
fi

MISMATCH=0
for f in "$DUMP_DIR"/*.png; do
  b=$(basename "$f")
  cmp -s "$f" "$REF_DIR/$b" || { echo "[r22_smoke]   differs: $b"; MISMATCH=$((MISMATCH+1)); }
done

if [ "$MISMATCH" -ne 0 ]; then
  echo "[r22_smoke] GATE2 FAIL -- ${MISMATCH}/${N_DUMP} frames differ from ${REF_DIR}"
  echo "[r22_smoke] STOP: step-14 is not the arm's real output; do not launch dump."
  exit 1
fi

echo "[r22_smoke] GATE2 PASS -- all ${N_DUMP} frames bit-identical to ${REF_DIR}"
echo "[r22_smoke] ready for: /run-step R22 dump"
exit 0
