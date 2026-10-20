#!/bin/bash
# R44 eval -- the 9-metric FiVE-Bench table for R44's TEXT-ONLY renders, one array task per
# arm, over the WHOLE 419-pair bench. Clone of r36_eval.sh (sweep) / r38_eval.sh (arms).
#
# USAGE (two modes, same metric list and guards):
#   sbatch --array=0-7 slurm_scripts/five_bench/r44_eval.sh sweep N    # N = 15 | 20 | 30
#       task i -> rho RHOS[i] of r44_infer.sh's sweep at N (same index map -- do not
#       reorder one without the other); dir r44_rho_sweep/r44_n{N}_rho{R}_novp
#   sbatch --array=0-<K-1> slurm_scripts/five_bench/r44_eval.sh arms
#       task i -> combo i of evaluation/r44_selected_combos.txt (r44_combos.sh, K combos);
#       dir r44_arms/r44_{combo}/step14 (stage 3's final render)
# STEM (= CSV name) = the method name: r44_n{N}_rho{R}_novp | r44_{combo}.
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE: config.yaml's `metrics:` key is
# vestigial (job 961342), and evaluate.py's argparse default includes five_acc and
# motion_fidelity, which exhaust an L40S. These nine are the r26/r31/r35/r36_eval.sh list,
# byte-identical. five_acc and CLIP-D are scored by r44_fiveacc.sh / r44_clipd.sh.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, and a raised metric
# DROPS its column and shifts every later one. The log is teed and grepped: any `Error:`
# line fails the task. Per-edit-type CSV row counts are checked too.
#
# NOTE for summarize/score: {stem}_avg.csv is a mean over the SIX per-edit-type means, not
# over the 419 clips. r36_summarize.py / r44_score.py read the per-edit-type CSVs and take a
# plain per-clip mean.
#
# --tmp_resize (R36): resized frames go to a per-clip temp dir deleted after the clip is
# scored, not a persistent {video}_resize sibling -- keeps ~2 GB/arm off the home quota.
# Partition L40S,A100: with five_acc and motion_fidelity excluded a 40 GB A100 is ample
# (R26 job 961490). --time=12:00:00: R20 measured ~45 s/clip with the heavier 16-metric
# set -> ~5.2 h for 419 clips; the 9-metric set is lighter.
#SBATCH --job-name=r44_eval
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r44_eval_%A_%a.out
#SBATCH --error=logs/r44_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench

RHOS=(2 3 4 6 8 10 20 50)
MODE=$1
TID=${SLURM_ARRAY_TASK_ID}
case "$MODE" in
  sweep)
    N=$2
    case "$N" in 15|20|30) ;; *) echo "[r44_eval] sweep needs N in {15,20,30}, got '${N}'"; exit 1 ;; esac
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
      echo "[r44_eval] bad array id '${TID}' (expected 0-7)"; exit 1
    fi
    NAME="r44_n${N}_rho${RHOS[$TID]}_novp"
    # r44_infer.sh writes {out_root}/{method}/edit{T}/{video}/ -- the edit_video layout.
    DIR="$FIVE_ROOT/r44_rho_sweep/$NAME"
    ;;
  arms)
    source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
    r44_load || exit 1
    K=${#R44_COMBOS[@]}
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "$K" ]; then
      echo "[r44_eval] bad array id '${TID}' (expected 0-$((K-1)) for ${K} selected combos)"; exit 1
    fi
    NAME="r44_${R44_COMBOS[$TID]}"
    # r44_stage3.sh writes {out_root}/{method}/step14/edit{T}/{video}/.
    DIR="$FIVE_ROOT/r44_arms/$NAME/step14"
    ;;
  *) echo "[r44_eval] usage: r44_eval.sh sweep N | arms"; exit 1 ;;
esac
STEM="$NAME"

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

echo "[r44_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
echo "[r44_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} stem=${STEM}"
echo "[r44_eval] scoring $DIR  (full bench, no --cases_json)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# ---- preflight: the arm's render must be complete, per edit type. Count REAL pairs only:
# evaluate.py may leave a {video}_resize sibling next to a scored video. Hard fail:
# scoring a short arm still writes a full-looking table.
if [ ! -d "$DIR" ]; then
  echo "[r44_eval] FAILED ${STEM} -- render dir does not exist: $DIR"
  exit 1
fi
BAD=0
for T in 1 2 3 4 5 6; do
  NP=$(ls -d "$DIR/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  [ "$NP" -ne "${EXPECT[$T]}" ] && { echo "[r44_eval] edit${T}: ${NP} real pairs, expected ${EXPECT[$T]}"; BAD=$((BAD+1)); }
done
if [ "$BAD" -ne 0 ]; then
  echo "[r44_eval] FAILED ${STEM} -- render incomplete, run the matching /run-step R44 wait-* step first"
  exit 1
fi
echo "[r44_eval] preflight OK: 419 real pairs, 100/100/100/100/9/10"

METRICS_LOG="logs/r44_eval_${MODE}_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
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
echo "[r44_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

FAILED=0
if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r44_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  FAILED=1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ] || [ "$NOOM" -ne 0 ]; then
  echo "[r44_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  FAILED=1
fi

# ---- post-run guard: every clip scored, per edit type (header row excluded).
for T in 1 2 3 4 5 6; do
  CSV="evaluation/csv/edit${T}_FiVE_${STEM}_frame_stride8.csv"
  NR=$(( $( [ -f "$CSV" ] && wc -l < "$CSV" || echo 1 ) - 1 ))
  STATUS="ok"
  [ "$NR" -ne "${EXPECT[$T]}" ] && { STATUS="MISMATCH (expect ${EXPECT[$T]})"; FAILED=1; }
  echo "[r44_eval] ${STEM} edit${T}: ${NR} scored rows -- ${STATUS}"
done

if [ "$FAILED" -ne 0 ]; then exit 1; fi
echo "[r44_eval] OK ${STEM} -> evaluation/csv/edit{1..6}_FiVE_${STEM}_frame_stride8.csv"
exit 0
