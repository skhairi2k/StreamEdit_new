#!/bin/bash
# R57 unified metric suite, the merge job (CPU): join the harness, FiVE-Acc, CLIP-D, EditReward, PSC and CF CSVs of each method
# into evaluation/csv/suite_{tag}_{method}.csv (one row per pair), suite_{tag}_{method}_per_type.csv and
# suite_{tag}_{method}_overall.csv (plain per-clip means per edit type / over all pairs). Submitted by eval_suite.sh with --dependency=afterok on the metrics and cf jobs; to run it alone:
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
echo "[eval_suite] merge job=${JOB} tag=${TAG} methods=${#UNITS[@]}: ${UNITS[*]%%=*} cases=${CASES:-<every pair of $EDIT_JSON_DIR>} node=$(hostname)"
# optional merge flags: columns allowed to be NaN, and --without_cf when the launcher ran --part metrics (CF not scored yet)
MERGE_ARGS=()
[ "${#ALLOW_NAN[@]}" -gt 0 ] && MERGE_ARGS=(--allow_nan "${ALLOW_NAN[@]}")
[ "$WITHOUT_CF" -eq 1 ] && MERGE_ARGS+=(--without_cf)
F=0
for U in "${UNITS[@]}"; do
  unit_set "$U"
  echo "=== [eval_suite] merge ${METHOD}"
  $PY_FB -I "$SUITE_DIR/merge_suite.py" --tag "$TAG" --method "$METHOD" --edit_json_dir "$EDIT_JSON_DIR" \
    "${CASES_PY[@]}" --csv_dir evaluation/csv "${MERGE_ARGS[@]}"
  RC=$?
  echo "[eval_suite] merge ${METHOD} exit=${RC}"
  [ "$RC" -eq 0 ] || F=1
done
echo "[eval_suite] merge failures: ${F}"
exit "$F"
