#!/bin/bash
# R39 eval -- the 9-metric FiVE-Bench table for the 16 compute-matched VP rho arms
# (N=20 and N=30 single-pass), one array task per arm, over the WHOLE 419-pair bench.
# Clone of r39_eval.sh with R39's arms and roots.
#
# ARRAY LAYOUT: same (N, rho) arm map as r39_infer.sh -- do not reorder one without the
# other(s). task i -> N = NS[i/8], rho = RHOS[i%8]:
#   0-7  = N=20, rho 2 3 4 6 8 10 20 50
#   8-15 = N=30, rho 2 3 4 6 8 10 20 50
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE: config.yaml's `metrics:` key is
# vestigial (job 961342), and evaluate.py's argparse default includes five_acc and
# motion_fidelity, which exhaust an L40S. These nine are the r26/r31/r35_eval.sh list,
# byte-identical. five_acc and CLIP-D are scored by r39_fiveacc.sh / r39_clipd.sh.
# Motion fidelity is out of scope for R39 (needs H100).
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, and a raised metric
# DROPS its column and shifts every later one. The log is teed and grepped: any `Error:`
# line fails the task. Per-edit-type CSV row counts are checked too.
#
# NOTE for summarize: {stem}_avg.csv is a mean over the SIX per-edit-type means, not over
# the 419 clips. r36_summarize.py (reused by R39 with r39 stems) reads the per-edit-type CSVs and takes a plain per-clip mean.
#
# --tmp_resize (R36): resized frames go to a per-clip temp dir deleted after the clip is
# scored, not a persistent {video}_resize sibling -- keeps ~2 GB/arm off the home quota.
# Metric values unchanged (A/B on 2 smoke clips: lpips + niqe CSVs byte-identical).
# Partition L40S,A100: with five_acc and motion_fidelity excluded a 40 GB A100 is ample
# (R26 job 961490). --time=12:00:00: R20 measured ~45 s/clip with the heavier 16-metric
# set -> ~5.2 h for 419 clips; the 9-metric set is lighter.
#SBATCH --job-name=r39_eval
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/r39_eval_%A_%a.out
#SBATCH --error=logs/r39_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r39_rho_sweep

RHOS=(2 3 4 6 8 10 20 50)
NS=(20 30)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 15 ]; then
  echo "[r39_eval] bad array id ${TID} (expected 0-15)"; exit 1
fi
N=${NS[$((TID / 8))]}
R=${RHOS[$((TID % 8))]}
STEM="r39_n${N}_rho${R}_vp"
# r39_infer.sh writes {out_root}/r39_n{N}_rho{R}_vp/edit{T}/{video}/, the shape
# `--tgt_layout edit_video` expects.
DIR="$OUT_ROOT/$STEM"

declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[r39_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r39_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} steps=${N} rho=${R} stem=${STEM}"
echo "[r39_eval] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the arm's render must be complete, per edit type. Count REAL pairs only:
# evaluate.py writes a {video}_resize sibling next to every video it scores. Hard fail:
# scoring a short arm still writes a full-looking table.
if [ ! -d "$DIR" ]; then
  echo "[r39_eval] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$N" -ne "${EXPECT[$T]}" ] && { echo "[r39_eval] edit${T}: ${N} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r39_eval] FAILED ${STEM} -- render incomplete, run /run-step R39 wait-infer first"
  exit 1
fi
echo "[r39_eval] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r39_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance psnr_unedit_part lpips_unedit_part \
            mse_unedit_part ssim_unedit_part clip_similarity_source_image \
            clip_similarity_target_image clip_similarity_target_image_edit_part \
            niqe_target_image \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" \
  --tgt_layout edit_video \
  --tmp_resize \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r39_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r39_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  FAILED=1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ]; then
  echo "[r39_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

# ---- post-run guard: every clip scored, per edit type (header row excluded).
for T in 1 2 3 4 5 6; do
  CSV="evaluation/csv/edit${T}_FiVE_${STEM}_frame_stride8.csv"
  N=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  STATUS="ok"
  [ "$N" -ne "${EXPECT[$T]}" ] && { STATUS="MISMATCH (expect ${EXPECT[$T]})"; FAILED=1; }
  echo "[r39_eval] ${STEM} edit${T}: ${N} scored rows -- ${STATUS}"
done

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r39_eval] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
