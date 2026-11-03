#!/bin/bash
# R57 unified metric suite -- lab launcher. Run it with bash (NOT sbatch) on the frontal: it checks everything the chosen part
# needs, then submits the jobs and prints their ids:
#   metrics  (1x L40S)   harness + motion fidelity, FiVE-Acc, CLIP-D, EditReward, PSC       eval_suite_metrics.sh
#   cf       (4x L40S)   IVEBench Content Fidelity, Qwen2.5-VL-72B judge                     eval_suite_cf.sh
# There is no merge job: each job writes a method's tables when it finishes it -- the metrics job with CF if CF is complete
# (else with empty CF columns), the CF job if the metrics are complete -- so the last of the two writes the full table.
# (eval_suite_merge.sh stays for merging by hand.)
#
# --part all      (default) metrics + cf, in parallel
# --part metrics  metrics only (CF columns empty). Starts at once on 1 GPU: the table without waiting for 4 free GPUs
# --part cf       cf only; merges with CF for every method whose metrics are complete
# Relaunching the same command after a job hit the 24 h limit resumes it: every stage skips what is already scored.
#
# Usage: bash slurm_scripts/five_bench/eval_suite.sh --method NAME --root DIR --tag TAG [--part all|metrics|cf]
#            [--layout edit_video|flat] [--edit_json_dir evaluation/five_bench_correct] [--cases FILE] [--allow_nan COL...]
#        bash slurm_scripts/five_bench/eval_suite.sh --batch FILE --per_job K --tag TAG [--part ...] [...]   several methods
#   --batch FILE  one method per line, `NAME ROOT` (whitespace-separated; blank lines and # comments ignored); same tag,
#                 layout and cases for all of them. Can be combined with --method/--root
#   --per_job K   methods per job (default 1). Each group of K methods gets one metrics job (and/or one cf job) that scores
#                 them one after the other: 1-2 jobs per group instead of per method (the submit cap).
#                 metrics: ~7-9 h per method on 415 pairs (estimate), 24 h limit -> K=2, K=3 if a method takes <= ~7 h (a
#                 job that runs over is relaunched and resumes). cf: per-method time not measured yet
#   --dry_run     run the preflight and print the sbatch commands, submit nothing
#   e.g. bash slurm_scripts/five_bench/eval_suite.sh --method r49_max_full \
#          --root ~/Data/dataggen/outputs/five_bench/r49_se_arms/r49_max_full/step14 --layout edit_video \
#          --cases evaluation/r57_smoke_cases.json --tag r57_smoke_lab
# Full FiVE-Bench-Correct run (415 pairs): omit --cases. --allow_nan COL... exists for a column that turns out to be NaN on
# some clips (the metrics job's gate then ignores evaluate.py's 'Error: COL:' lines); no column of the suite needs it today.
# Arguments and output names: eval_suite_common.sh. Logs: logs/eval_suite_{metrics,cf,merge}_<jobid>.{out,err}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
# shellcheck source=/dev/null
source "$REPO/slurm_scripts/five_bench/eval_suite_common.sh"

# launcher-only options, taken out before the common parser sees the rest
BATCH=""; PER_JOB=1; DRY=0; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --batch) BATCH=$2; shift 2 ;;
    --per_job) PER_JOB=$2; shift 2 ;;
    --dry_run) DRY=1; shift ;;
    *) REST+=("$1"); shift ;;
  esac
done
[[ "$PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "[eval_suite] --per_job must be a positive integer"; exit 1; }
if [ -n "$BATCH" ]; then
  [ -f "$BATCH" ] || { echo "[eval_suite] --batch $BATCH: no such file"; exit 1; }
  while read -r NAME DIR EXTRA || [ -n "$NAME" ]; do   # also a last line without a newline
    [ -z "$NAME" ] || [[ "$NAME" == \#* ]] && continue
    [ -n "$DIR" ] && [ -z "$EXTRA" ] || { echo "[eval_suite] --batch $BATCH: line '$NAME $DIR $EXTRA' is not 'NAME ROOT'"; exit 1; }
    REST+=(--unit "${NAME}=${DIR}")
  done < "$BATCH"
fi
suite_parse_args "${REST[@]}" || exit 1
[ -n "$CASES" ] && CASES=$(readlink -f "$CASES")   # before the cd: relative to where the user stands
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
case "$EDIT_JSON_DIR" in /*) ;; *) EDIT_JSON_DIR="$REPO/$EDIT_JSON_DIR" ;; esac

# ---- preflight: fail here, in seconds, not after a queue wait (only what the chosen part needs)
FAIL=0
need() { [ -e "$1" ] || { echo "[eval_suite] MISSING: $1  ($2)"; FAIL=1; }; }
need "$PY_FB" "five-bench env"
if [ "$PART" != cf ]; then
  need "$PY_ER" "editreward env: bash evaluation/eval_suite/make_editreward_env.sh"
  need "$WEIGHTS_DIR/EditReward-MiMo-VL-7B-SFT-2508/model.safetensors" "bash evaluation/eval_suite/fetch_weights.sh"
  need "$WEIGHTS_DIR/MiMo-VL-7B-SFT-2508/config.json" "bash evaluation/eval_suite/fetch_weights.sh"
  need "$WEIGHTS_DIR/VideoCLIP-XL-v2/VideoCLIP-XL-v2.bin" "bash evaluation/eval_suite/fetch_weights.sh"
  need ~/Code/EditReward/pyproject.toml "bash evaluation/eval_suite/fetch_weights.sh"
  need ~/Code/IVEBench/metrics/compliance/phrase_semantic_consistency.py "bash evaluation/eval_suite/fetch_weights.sh"
fi
if [ "$PART" != metrics ]; then
  need "$QWEN72B_DIR/config.json" "bash evaluation/eval_suite/fetch_weights.sh"
  need ~/Code/IVEBench/metrics/fidelity/content_fidelity.py "bash evaluation/eval_suite/fetch_weights.sh"
fi
for U in "${UNITS[@]}"; do unit_set "$U"; need "$ROOT" "root of method ${METHOD}"; done
[ -z "$CASES" ] || need "$CASES" "--cases"
[ "$FAIL" -eq 0 ] || { echo "[eval_suite] preflight FAILED"; exit 1; }

suite_expect || exit 1
echo "[eval_suite] part=${PART}, ${#UNITS[@]} method(s), ${PER_JOB} per job; pairs per edit type ${EXPECT[*]}"
for U in "${UNITS[@]}"; do
  unit_set "$U"
  MISSING=$($PY_FB -I "$SUITE_DIR/suite_common.py" missing --root "$ROOT" --layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" "${CASES_PY[@]}")
  if [ -n "$MISSING" ]; then
    echo "[eval_suite] preflight FAILED: ${METHOD}: $(echo "$MISSING" | wc -l) pair(s) without a render under $ROOT, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"
    FAIL=1
  fi
  if [ "$PART" = cf ]; then
    # the full merge after cf needs the metrics CSVs; say now if they are not there yet
    for F in "evaluation/csv/suite_editreward_${TAG}_${METHOD}.csv" "evaluation/csv/suite_psc_${TAG}_${METHOD}.csv" \
             "evaluation/csv/suite_clipd_${TAG}_${METHOD}.csv"; do
      [ -f "$F" ] || echo "[eval_suite] WARNING: $F missing -- run --part metrics first, or the merge after cf will fail"
    done
  fi
done
[ "$FAIL" -eq 0 ] || exit 1
echo "[eval_suite] every pair of every method has a render"

# ---- submit, one group of --per_job methods at a time (the arguments are passed on with the resolved paths)
COMMON=(--layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" --tag "$TAG")
[ -n "$CASES" ] && COMMON+=(--cases "$CASES")
[ "${#ALLOW_NAN[@]}" -gt 0 ] && COMMON+=(--allow_nan "${ALLOW_NAN[@]}")
submit() {  # sbatch "$@" -> job id; --dry_run prints the command and returns a placeholder id
  if [ "$DRY" -eq 1 ]; then
    echo "[eval_suite] DRY RUN: sbatch --parsable $*" >&2; echo "DRY$((${#SUBMITTED[@]} + 1))"   # runs in $(...): count from the parent
  else
    sbatch --parsable "$@"
  fi
}
NG=0; SUBMITTED=()
for ((i = 0; i < ${#UNITS[@]}; i += PER_JOB)); do
  GROUP=("${UNITS[@]:i:PER_JOB}"); NG=$((NG + 1))
  UARGS=(); for U in "${GROUP[@]}"; do UARGS+=(--unit "$U"); done
  J_METRICS=""; J_CF=""
  if [ "$PART" != cf ]; then
    J_METRICS=$(submit slurm_scripts/five_bench/eval_suite_metrics.sh "${UARGS[@]}" "${COMMON[@]}") \
      || { echo "[eval_suite] sbatch metrics FAILED at group ${NG}; already queued: ${SUBMITTED[*]:-none}"; exit 1; }
    SUBMITTED+=("$J_METRICS")
  fi
  if [ "$PART" != metrics ]; then
    J_CF=$(submit slurm_scripts/five_bench/eval_suite_cf.sh "${UARGS[@]}" "${COMMON[@]}") \
      || { echo "[eval_suite] sbatch cf FAILED at group ${NG}; already queued: ${SUBMITTED[*]:-none}"; exit 1; }
    SUBMITTED+=("$J_CF")
  fi
  echo "[eval_suite] group ${NG} (${GROUP[*]%%=*}): ${J_METRICS:+metrics=${J_METRICS} }${J_CF:+cf=${J_CF}}"
done
echo "[eval_suite] $([ "$DRY" -eq 1 ] && echo 'DRY RUN, nothing submitted' || echo submitted): part=${PART}, ${#UNITS[@]} method(s) in ${NG} group(s), ${#SUBMITTED[@]} job(s)"
echo "[eval_suite] logs: logs/eval_suite_{metrics,cf}_<jobid>.out"
echo "[eval_suite] results: evaluation/csv/suite_${TAG}_<method>.csv and suite_${TAG}_<method>_per_type.csv$([ "$PART" = metrics ] && echo ' (CF columns empty until --part cf)')"
