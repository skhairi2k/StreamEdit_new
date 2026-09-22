#!/bin/bash
# R34 -- CLIP-D (directional CLIP) on the FIRST CHUNK ONLY, for R26's 16 constant-b arms
# + R31's 4 divergence arms. Single task, not arrayed: 20 arms x 22 clips x 9 frames is
# ~4k ViT-L/14 image forwards, minutes of GPU time, and the source embeddings are cached
# per clip and reused across all 20 arms -- arraying it would recompute that cache 20
# times over.
#
# WHY CLIP-D AND NOT THE STORED clip_similarity_target_image. R33 measured the incumbent
# not to rank these renders by edit strength (mean Spearman(b, clip_target) = +0.098,
# positive on 11/22 clips) and chose clip_d_prompt as the achievement axis of record
# (+0.4080, 17/22, 0 flat clips). R34 computes BOTH on the new window -- the incumbent
# comes from r34_eval.sh, this script supplies CLIP-D -- so the first-chunk picture can
# be read on the axis R33 validated rather than the one it retired.
#
# THE WINDOW must match r34_eval.sh's exactly: --frame_stride 1 --max_frames 9, the first
# rollout chunk (3 latent frames = pixel frames 0-8). If these two diverge, CLIP-D and
# the LPIPS/CLIP-target columns would describe different frames and could not share a row
# in r34_arms.csv.
#
# --which r26r31 is an R34 addition to the shared script: R33's `all` also walks R30's 8
# routed arms, which are out of R34's scope (each is rendered at its own per-clip routed
# b, so they are 8 points, not a sweep, and R34's oracle is defined over the 8 SPATIAL
# b's instead).
#
# ⚠️ Read-only over the render trees: it opens frames and writes ONE csv under
# evaluation/csv/. It cannot disturb r31_arms/ or r26_spatial_tau/.
#SBATCH --job-name=r34_clipd
# RENDER ROOTS LIVE IN ~/Data, NOT /projects -- the same fix r33_fiveacc.sh landed on
# 2026-09-22 after five rounds of jobs died on a stale /projects/dataggen mount (node57,
# node51, then node01, node02; exclude-and-retry never converged because each retry's
# idlest node was a new bad one). Note r33_clip_directional.sh still names /projects: it
# succeeded before the roots were copied, on a node that happened to have the mount. Do
# not copy that part of it. With the home mount, L40S is usable again, so this takes the
# wider partition pair rather than r33_clip_directional.sh's A100-only narrowing.
# Exclusions kept defensively: node52 for the older gpu:8-but-no-device fault, the other
# three from R33's confirmed-bad list.
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
# 20 arms on 9 frames each; R33's 28-arm/11-frame run fit in 2h with room to spare.
#SBATCH --time=02:00:00
#SBATCH --output=logs/r34_clipd_%j.out
#SBATCH --error=logs/r34_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R26_ROOT=~/Data/dataggen/outputs/five_bench/r26_spatial_tau
R31_ROOT=~/Data/dataggen/outputs/five_bench/r31_arms
CASES="$REPO/evaluation/cases.json"
OUT="$REPO/evaluation/csv/r34_clip_directional_chunk1.csv"

STRIDE=1
MAXF=9          # 3 latent frames = 1 + 4 + 4 pixel frames, same window as r34_eval.sh
NARMS=20        # R26: 8 spatial + 8 uniform; R31: 4 divergence
NCLIPS=22
EXPECT_ROWS=$((NARMS * NCLIPS))

cd "$REPO" || exit 1
# Conda: five-bench, NOT streamgve (the R2 env bug: .bashrc auto-activates streamgve and
# `conda activate five-bench` alone does not pop it).
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r34_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r34_clipd] job=${SLURM_JOB_ID} stride=${STRIDE} max_frames=${MAXF}"

# Mount guard, the thing that caught R33's failures in seconds instead of hours.
for R in "$R26_ROOT" "$R31_ROOT"; do
  if [ ! -d "$R" ]; then
    echo "[r34_clipd] FAILED -- render root does not exist on this node: $R"
    echo "[r34_clipd]   (is ~/Data/dataggen present here?)"
    exit 1
  fi
done

python evaluation/r33_clip_directional.py \
  --which r26r31 \
  --cases "$CASES" \
  --data_root "$DATA_ROOT" \
  --r26_root "$R26_ROOT" \
  --r31_root "$R31_ROOT" \
  --frame_stride $STRIDE --max_frames $MAXF \
  -o "$OUT" \
  2>&1 | tee "logs/r34_clipd_${SLURM_JOB_ID}.run.log"
RC=${PIPESTATUS[0]}
echo "[r34_clipd] exit=${RC}"

if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
  echo "[r34_clipd] FAILED -- exit ${RC}, or $OUT not written"
  exit 1
fi

# The script prints `MISSING {method}/{clip}` and CONTINUES for any arm dir it cannot
# find, so a partial table exits 0 and looks complete. Count the rows instead.
NMISS=$(grep -c 'MISSING' "logs/r34_clipd_${SLURM_JOB_ID}.run.log")
NROWS=$(( $(wc -l < "$OUT") - 1 ))
NMETH=$(tail -n +2 "$OUT" | cut -d, -f1 | sort -u | wc -l)
NFR=$(tail -n +2 "$OUT" | cut -d, -f5 | sort -u | tr '\n' ' ')
echo "[r34_clipd] rows=${NROWS} (expect ${EXPECT_ROWS})  methods=${NMETH} (expect ${NARMS})  missing_lines=${NMISS}"
echo "[r34_clipd] distinct n_frames values across rows: ${NFR}  # expect just ${MAXF}"

if [ "$NROWS" -ne "$EXPECT_ROWS" ] || [ "$NMETH" -ne "$NARMS" ] || [ "$NMISS" -ne 0 ]; then
  echo "[r34_clipd] FAILED -- incomplete table (see MISSING lines in the run log)"
  exit 1
fi
# Every row must have been scored on exactly MAXF frames; a short clip would otherwise be
# averaged in as if it were a full chunk.
if [ "$(echo $NFR)" != "$MAXF" ]; then
  echo "[r34_clipd] FAILED -- some rows scored a frame count other than ${MAXF}: ${NFR}"
  exit 1
fi

echo "[r34_clipd] OK -> $OUT"
exit 0
