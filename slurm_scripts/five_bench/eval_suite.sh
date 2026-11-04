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
#   --root_base DIR  a ROOT in the batch file that does not start with / or ~ is taken relative to DIR (e.g.
#                 /scratch/PERSO/skhairi/five_bench), so a task's batch file stays valid wherever its renders are staged
#   --per_job K   methods per job (default 1). Each group of K methods gets one metrics job (and/or one cf job) that scores
#                 them one after the other: 1-2 jobs per group instead of per method (the submit cap).
#                 metrics: ~7-9 h per method on 415 pairs (estimate), 24 h limit -> K=2, K=3 if a method takes <= ~7 h (a
#                 job that runs over is relaunched and resumes). cf: per-method time not measured yet
#   --dry_run     run the preflight and print the sbatch commands, submit nothing
#   --all_methods also submit methods whose part is already complete (by default they are left out, so relaunching the
#                 same command after a job hit 24 h submits only the methods that are not finished)
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
BATCH=""; PER_JOB=1; DRY=0; ROOT_BASE=""; KEEP_DONE=0; REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --batch) BATCH=$2; shift 2 ;;
    --root_base) ROOT_BASE=${2%/}; shift 2 ;;
    --per_job) PER_JOB=$2; shift 2 ;;
    --dry_run) DRY=1; shift ;;
    --all_methods) KEEP_DONE=1; shift ;;
    *) REST+=("$1"); shift ;;
  esac
done
[[ "$PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "[eval_suite] --per_job must be a positive integer"; exit 1; }
if [ -n "$BATCH" ]; then
  [ -f "$BATCH" ] || { echo "[eval_suite] --batch $BATCH: no such file"; exit 1; }
  while read -r NAME DIR EXTRA || [ -n "$NAME" ]; do   # also a last line without a newline
    [ -z "$NAME" ] || [[ "$NAME" == \#* ]] && continue
    [ -n "$DIR" ] && [ -z "$EXTRA" ] || { echo "[eval_suite] --batch $BATCH: line '$NAME $DIR $EXTRA' is not 'NAME ROOT'"; exit 1; }
    case "$DIR" in
      /*|~*) ;;
      *) [ -n "$ROOT_BASE" ] || { echo "[eval_suite] --batch $BATCH: '$NAME' has a relative root '$DIR' -- give --root_base"; exit 1; }
         DIR="${ROOT_BASE}/${DIR}" ;;
    esac
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
  need "$PY_CF" "ivebench env (CF): bash evaluation/eval_suite/make_cf_env.sh"
  need "$QWEN72B_DIR/config.json" "bash evaluation/eval_suite/fetch_weights.sh"
  need ~/Code/IVEBench/metrics/fidelity/content_fidelity.py "bash evaluation/eval_suite/fetch_weights.sh"
fi
for U in "${UNITS[@]}"; do unit_set "$U"; need "$ROOT" "root of method ${METHOD}"; done
[ -z "$CASES" ] || need "$CASES" "--cases"
[ "$FAIL" -eq 0 ] || { echo "[eval_suite] preflight FAILED"; exit 1; }

suite_expect || exit 1
echo "[eval_suite] part=${PART}, ${#UNITS[@]} method(s), ${PER_JOB} per job, src_window=${SRC_WINDOW}; pairs per edit type ${EXPECT[*]}"
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

# ---- relaunch-friendly: drop the methods whose part is already complete, so a relaunch submits only what is left
# (metrics: every metrics stage complete; cf: the CF CSV complete; all: both). --all_methods keeps them (they then skip
# everything inside the job).
# never two jobs on the same method and part at once: leave out methods already queued or running for this tag and part
# (each job carries --comment "eval_suite|TAG|PART|m1,m2,...")
QUEUED=$(squeue -u "$USER" -h -o '%i %k' 2>/dev/null | awk -v t="$TAG" '{ split($2, f, "|"); if (f[1] == "eval_suite" && f[2] == t) print $1, f[3], f[4] }')
queued_job() {  # $1 = part (metrics|cf), $2 = method -> job id, or nothing
  echo "$QUEUED" | awk -v p="$1" -v m="$2" '$2 == p { n = split($3, ms, ","); for (i = 1; i <= n; i++) if (ms[i] == m) { print $1; exit } }'
}
KEEP=(); BUSY=()
for U in "${UNITS[@]}"; do
  unit_set "$U"; J=""
  [ "$PART" != cf ] && J=$(queued_job metrics "$METHOD")
  [ -z "$J" ] && [ "$PART" != metrics ] && J=$(queued_job cf "$METHOD")
  if [ -n "$J" ]; then BUSY+=("${METHOD}(job ${J})"); else KEEP+=("$U"); fi
done
if [ "${#BUSY[@]}" -gt 0 ]; then
  echo "[eval_suite] already queued or running for part=${PART}, not submitted again (${#BUSY[@]}): ${BUSY[*]}"
  UNITS=("${KEEP[@]}")
  [ "${#UNITS[@]}" -gt 0 ] || { echo "[eval_suite] nothing to do: every method is already queued or running"; exit 0; }
fi

if [ "$KEEP_DONE" -eq 0 ]; then
  TODO_UNITS=(); DONE_NAMES=()
  for U in "${UNITS[@]}"; do
    unit_set "$U"
    case "$PART" in
      metrics) metrics_done && { DONE_NAMES+=("$METHOD"); continue; } ;;
      cf)      cf_done && { DONE_NAMES+=("$METHOD"); continue; } ;;
      all)     metrics_done && cf_done && { DONE_NAMES+=("$METHOD"); continue; } ;;
    esac
    TODO_UNITS+=("$U")
  done
  if [ "${#DONE_NAMES[@]}" -gt 0 ]; then
    echo "[eval_suite] already complete for part=${PART}, not submitted (${#DONE_NAMES[@]}): ${DONE_NAMES[*]}"
  fi
  UNITS=("${TODO_UNITS[@]}")
  if [ "${#UNITS[@]}" -eq 0 ]; then
    echo "[eval_suite] nothing to do: every method is complete for part=${PART}"
    exit 0
  fi
fi

# ---- submit, one group of --per_job methods at a time (the arguments are passed on with the resolved paths)
COMMON=(--layout "$LAYOUT" --edit_json_dir "$EDIT_JSON_DIR" --tag "$TAG" --src_window "$SRC_WINDOW")
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
    J_METRICS=$(submit --comment "eval_suite|${TAG}|metrics|$(IFS=,; echo "${GROUP[*]%%=*}")" slurm_scripts/five_bench/eval_suite_metrics.sh "${UARGS[@]}" "${COMMON[@]}") \
      || { echo "[eval_suite] sbatch metrics FAILED at group ${NG}; already queued: ${SUBMITTED[*]:-none}"; exit 1; }
    SUBMITTED+=("$J_METRICS")
  fi
  if [ "$PART" != metrics ]; then
    J_CF=$(submit --comment "eval_suite|${TAG}|cf|$(IFS=,; echo "${GROUP[*]%%=*}")" slurm_scripts/five_bench/eval_suite_cf.sh "${UARGS[@]}" "${COMMON[@]}") \
      || { echo "[eval_suite] sbatch cf FAILED at group ${NG}; already queued: ${SUBMITTED[*]:-none}"; exit 1; }
    SUBMITTED+=("$J_CF")
  fi
  echo "[eval_suite] group ${NG} (${GROUP[*]%%=*}): ${J_METRICS:+metrics=${J_METRICS} }${J_CF:+cf=${J_CF}}"
done
echo "[eval_suite] $([ "$DRY" -eq 1 ] && echo 'DRY RUN, nothing submitted' || echo submitted): part=${PART}, ${#UNITS[@]} method(s) in ${NG} group(s), ${#SUBMITTED[@]} job(s)"
echo "[eval_suite] logs: logs/eval_suite_{metrics,cf}_<jobid>.out"
echo "[eval_suite] results: evaluation/csv/suite_${TAG}_<method>.csv (per clip), suite_${TAG}_<method>_per_type.csv and suite_${TAG}_<method>_overall.csv$([ "$PART" = metrics ] && echo ' (CF columns empty until --part cf)')"
