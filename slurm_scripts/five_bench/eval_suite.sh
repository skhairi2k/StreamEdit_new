#!/bin/bash
# R57 unified metric suite -- lab launcher. Run it with bash (NOT sbatch) on the frontal: it checks everything it can, then
# submits three jobs and prints their ids:
#   metrics  (1x L40S)   harness + motion fidelity, FiVE-Acc, CLIP-D, EditReward, PSC       eval_suite_metrics.sh
#   cf       (4x L40S)   IVEBench Content Fidelity, Qwen2.5-VL-72B judge, in parallel        eval_suite_cf.sh
#   merge    (CPU)       joins everything; --dependency=afterok on metrics AND cf             eval_suite_merge.sh
#
# Usage: bash slurm_scripts/five_bench/eval_suite.sh --method NAME --root DIR --tag TAG
#            [--layout edit_video|flat] [--edit_json_dir evaluation/five_bench_correct] [--cases FILE] [--allow_nan COL...]
#   e.g. bash slurm_scripts/five_bench/eval_suite.sh --method r49_max_full \
#          --root ~/Data/dataggen/outputs/five_bench/r49_se_arms/r49_max_full/step14 --layout edit_video \
#          --cases evaluation/r57_smoke_cases.json --tag r57_smoke_lab
# Full FiVE-Bench-Correct run (415 pairs): omit --cases. --allow_nan motion_fidelity_score_edit_part for that run (NaN on
# clips with an empty edit mask, ~1 in 50); the metrics job still reports its 'Error:' line, so afterok blocks the merge
# then -- run eval_suite_merge.sh by hand once the two jobs finished.
# Arguments and output names: eval_suite_common.sh. Logs: logs/eval_suite_{metrics,cf,merge}_<jobid>.{out,err}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
# shellcheck source=/dev/null
source "$REPO/slurm_scripts/five_bench/eval_suite_common.sh"
suite_parse_args "$@" || exit 1
[ -n "$CASES" ] && CASES=$(readlink -f "$CASES")   # before the cd: relative to where the user stands
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
case "$EDIT_JSON_DIR" in /*) ;; *) EDIT_JSON_DIR="$REPO/$EDIT_JSON_DIR" ;; esac

# ---- preflight: fail here, in seconds, not after a queue wait
FAIL=0
need() { [ -e "$1" ] || { echo "[eval_suite] MISSING: $1  ($2)"; FAIL=1; }; }
need "$PY_FB" "five-bench env"
need "$PY_ER" "editreward env: bash evaluation/eval_suite/make_editreward_env.sh"
need "$WEIGHTS_DIR/EditReward-MiMo-VL-7B-SFT-2508/model.safetensors" "bash evaluation/eval_suite/fetch_weights.sh"
need "$WEIGHTS_DIR/MiMo-VL-7B-SFT-2508/config.json" "bash evaluation/eval_suite/fetch_weights.sh"
need "$WEIGHTS_DIR/VideoCLIP-XL-v2/VideoCLIP-XL-v2.bin" "bash evaluation/eval_suite/fetch_weights.sh"
need "$WEIGHTS_DIR/Qwen2.5-VL-72B-Instruct/config.json" "bash evaluation/eval_suite/fetch_weights.sh"
need ~/Code/EditReward/pyproject.toml "bash evaluation/eval_suite/fetch_weights.sh"
need ~/Code/IVEBench/metrics/fidelity/content_fidelity.py "bash evaluation/eval_suite/fetch_weights.sh"
need "$ROOT" "--root"
[ -z "$CASES" ] || need "$CASES" "--cases"
[ "$FAIL" -eq 0 ] || { echo "[eval_suite] preflight FAILED"; exit 1; }

suite_expect || exit 1
echo "[eval_suite] ${METHOD}: pairs per edit type ${EXPECT[*]}"
MISSING=$($PY_FB -I "$SUITE_DIR/suite_common.py" missing --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" "${CASES_PY[@]}")
if [ -n "$MISSING" ]; then
  echo "[eval_suite] preflight FAILED: $(echo "$MISSING" | wc -l) pair(s) without a render under $ROOT, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"
  exit 1
fi
echo "[eval_suite] every pair has a render under $ROOT"

# ---- submit (the arguments are passed on unchanged, with the resolved --cases / --edit_json_dir / --root)
ARGS=(--method "$METHOD" --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" --tag "$TAG")
[ -n "$CASES" ] && ARGS+=(--cases "$CASES")
J_METRICS=$(sbatch --parsable slurm_scripts/five_bench/eval_suite_metrics.sh "${ARGS[@]}") || { echo "[eval_suite] sbatch metrics FAILED"; exit 1; }
J_CF=$(sbatch --parsable slurm_scripts/five_bench/eval_suite_cf.sh "${ARGS[@]}") || { echo "[eval_suite] sbatch cf FAILED (metrics ${J_METRICS} is queued)"; exit 1; }
MARGS=("${ARGS[@]}")
[ "${#ALLOW_NAN[@]}" -gt 0 ] && MARGS+=(--allow_nan "${ALLOW_NAN[@]}")
J_MERGE=$(sbatch --parsable --dependency="afterok:${J_METRICS}:${J_CF}" slurm_scripts/five_bench/eval_suite_merge.sh "${MARGS[@]}") \
  || { echo "[eval_suite] sbatch merge FAILED (metrics ${J_METRICS} and cf ${J_CF} are queued)"; exit 1; }
echo "[eval_suite] submitted: metrics=${J_METRICS} cf=${J_CF} merge=${J_MERGE} (afterok ${J_METRICS},${J_CF})"
echo "[eval_suite] logs: logs/eval_suite_{metrics,cf,merge}_<jobid>.out"
echo "[eval_suite] result: evaluation/csv/suite_${TAG}_${METHOD}.csv and suite_${TAG}_${METHOD}_per_type.csv"
