#!/bin/bash
# R57 unified metric suite, job 3 of 3 (CPU): join the harness, FiVE-Acc, CLIP-D, EditReward, PSC and CF CSVs of one method
# into evaluation/csv/suite_{tag}_{method}.csv (one row per pair) and suite_{tag}_{method}_per_type.csv (plain per-clip
# means). Submitted by eval_suite.sh with --dependency=afterok on the metrics and cf jobs; to run it alone:
#   sbatch slurm_scripts/five_bench/eval_suite_merge.sh --method M --tag T [--root R] [--cases C] [--allow_nan COL...]
# Exit 1 on a missing file / column / pair, or on any NaN outside --allow_nan (merge_suite.py).
#SBATCH --job-name=suite_merge
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_suite_merge_%j.out
#SBATCH --error=logs/eval_suite_merge_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
# shellcheck source=/dev/null
source "$REPO/slurm_scripts/five_bench/eval_suite_common.sh"
# --root is not needed here (accepted, so the launcher can pass one argument list to every job)
SUITE_NO_ROOT=1
suite_parse_args "$@" || exit 1
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
echo "[eval_suite] merge job=${JOB} method=${METHOD} tag=${TAG} cases=${CASES:-<every pair of $EDIT_JSON_DIR>} node=$(hostname)"
ALLOW_ARG=()
[ "${#ALLOW_NAN[@]}" -gt 0 ] && ALLOW_ARG=(--allow_nan "${ALLOW_NAN[@]}")
$PY_FB -I "$SUITE_DIR/merge_suite.py" --tag "$TAG" --method "$METHOD" --edit_json_dir "$EDIT_JSON_DIR" \
  "${CASES_PY[@]}" --csv_dir evaluation/csv "${ALLOW_ARG[@]}"
RC=$?
echo "[eval_suite] merge exit=${RC}"
exit "$RC"
