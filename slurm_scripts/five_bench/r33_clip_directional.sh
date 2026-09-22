#!/bin/bash
# R31 -- directional CLIP (CLIP-D) over R26's constant-b reference renders AND R31's four
# divergence arms, so both sides of the trade-off figure move onto the same new
# achievement axis in one pass.
#
# WHY THIS EXISTS. evaluate.py's clip_similarity_target_image does not rank renders by edit
# strength on this case set -- measured per clip over R26's own 8-b sweep:
# Spearman(b, clip_target) = +0.098 mean, positive on only 11/22 clips, while
# Spearman(b, niqe_target) = -0.530 (quality IMPROVES with b) and Spearman(niqe, clip) =
# -0.093 (CLIP does not track quality either). Root cause is measured too:
# cos(E_txt(src_prompt), E_txt(trg_prompt)) = 0.846 mean, up to 0.970 -- the two captions
# differ by one noun out of ~35 tokens and the shared scene description, which every render
# satisfies at every b, dominates the embedding. CLIP-D subtracts that shared part off both
# the text and the image side. See evaluation/r33_clip_directional.py's docstring.
#
# ⚠️ --frame_stride MUST stay 8. It is what the whole R26/R30/R31 program scored at
# (evaluate.py --frame_stride default), and CLIP-D is only comparable with the stored
# lpips/ssim/clip columns if it describes the same frames.
#
# ⚠️ This also silently fixes a real bug for 0040_tennis. torchmetrics 1.9.0
# (clip_score.py:145-146), which the FiVE harness calls, raw-slices input_ids[:77] on
# over-long captions and drops the EOT token that CLIP pools its text embedding at. That
# clip's prompts are 85/84 tokens, so its stored clip_similarity_* columns do not depend on
# the prompt at all (src_prompt and trg_prompt return a bit-identical 5.528). This script
# tokenises with truncation=True, which keeps EOT.
#
# ⚠️ Read-only over the render trees: it opens frames and writes ONE csv under
# evaluation/csv/. It cannot disturb r31_arms/ or r26_spatial_tau/.
#SBATCH --job-name=r33_clipd
# Same partition pair and node52 exclusion as r31_eval.sh (node52 advertises gpu:8 but
# exposes no device to batch jobs). One ViT-L/14 forward per frame -- far lighter than the
# 9-metric harness, so a single GPU is ample.
# L40S dropped 2026-09-21: r33_clipd jobs 1002579/1002591 failed instantly on node57 with
# /projects/dataggen missing; user independently confirmed the same on node51. Two of L40S's
# four nodes (node39/50/51/57) bad in one sitting reads as an L40S-partition-wide mount
# regression, not a single flaky node -- an interactive V100/node12 job sees the path fine
# in the meantime. Restricted to A100 until the L40S mount is confirmed fixed; re-add
# --partition=L40S,A100 at that point.
#SBATCH --partition=A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r33_clipd_%j.out
#SBATCH --error=logs/r33_clipd_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT
# and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R31_ROOT=/projects/dataggen/outputs/five_bench/r31_arms
R26_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau
R30_ROOT=/projects/dataggen/outputs/five_bench/r30_arms
WHICH=${WHICH:-all}
SUFFIX=${SUFFIX:-}
# 28 arms x 22 clips. R26 16 (8 spatial + 8 uniform) + R30 8 routed + R31 4 divergence.
EXPECT_ROWS=616

# NOT arrayed, deliberately. r33_clip_directional.py loops clips OUTER and methods INNER,
# so each clip's source frames are embedded once (211 total) and reused by all 28 methods:
# 6,119 image embeds for the whole job. Splitting by arm would re-embed the sources in
# every task -- 11,816 embeds, +93% GPU for a job that needs one GPU for well under an
# hour, plus 28 model loads. If this ever runs long, split by TASK GROUP (R26-spatial /
# R26-uniform / R30 / R31, --array=0-3), which keeps source reuse at 4x redundancy.

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc auto-activates
# streamgve, and `conda activate five-bench` alone does not pop it).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv
export PYTHONUNBUFFERED=1

echo "[r33_clipd] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r33_clipd] which=${WHICH} suffix='${SUFFIX}'"

# /projects/dataggen is NOT mounted on every node (it is absent on the login node this was
# written from). Fail loudly here rather than letting the script report 22 MISSING lines
# and still exit 0 with an empty CSV.
for D in "$R31_ROOT" "$R26_ROOT" "$R30_ROOT"; do
  if [ ! -d "$D" ]; then
    echo "[r33_clipd] FAILED -- render root does not exist on this node: $D"
    echo "[r33_clipd]   (is /projects/dataggen mounted here?)"
    exit 1
  fi
done

python evaluation/r33_clip_directional.py \
  --which "$WHICH" \
  --cases "$REPO/evaluation/cases.json" \
  --data_root "$DATA_ROOT" \
  --r31_root "$R31_ROOT" \
  --r26_root "$R26_ROOT" \
  --r30_root "$R30_ROOT" \
  --suffix "$SUFFIX" \
  --frame_stride 8
RC=$?

OUT="evaluation/csv/r33_clip_directional${SUFFIX}.csv"
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[r33_clipd] FAILED rc=${RC}, out=${OUT}"
  exit 1
fi
NROW=$(( $(wc -l < "$OUT") - 1 ))
echo "[r33_clipd] OK -- ${NROW} rows -> ${OUT}"
if [ "$WHICH" = "all" ] && [ "$NROW" -ne "$EXPECT_ROWS" ]; then
  echo "[r33_clipd] WARNING: expected ${EXPECT_ROWS} rows for --which all, got ${NROW}."
  echo "[r33_clipd]   Some (method, clip) pair was missing -- grep the log for MISSING."
fi
# Degeneracy guard. CLIP-D divides by ||d_img||, so a render that barely differs from its
# source still yields a confident-looking cosine fixed by noise rather than by the edit --
# the low-b end of the sweep is exactly that regime. Surface the smallest norms here so a
# degenerate reading cannot be mistaken for a real one downstream.
echo "[r33_clipd] 10 smallest |d_img| (method, case_id, norm) -- inspect before trusting:"
python - "$OUT" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("delta_img_norm")]
rows.sort(key=lambda r: float(r["delta_img_norm"]))
for r in rows[:10]:
    print(f"    {r['method']:<28} {r['case_id']:<20} {float(r['delta_img_norm']):.5f}")
n = sum(1 for r in rows if float(r["delta_img_norm"]) < 0.05)
print(f"    -> {n}/{len(rows)} rows below 0.05")
PY
