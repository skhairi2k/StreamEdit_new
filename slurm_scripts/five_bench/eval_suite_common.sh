#!/bin/bash
# R57 unified metric suite -- shared by eval_suite.sh (launcher) and eval_suite_{metrics,cf,merge}.sh. Sourced, never run.
#
# Arguments (same for the launcher and every job):
#   --method NAME          label of the method (CSV column key, file names)
#   --root DIR             method render root; frames at DIR/edit{T}/{video}/ (--layout edit_video) or DIR/{video}/ (flat)
#   --unit NAME=DIR        one method, repeatable: a job scores its units one after the other (same tag, layout, cases).
#                          --method/--root is one unit; the launcher builds units from --batch FILE and groups them
#                          --per_job K to a job
#   --layout edit_video|flat           (default edit_video)
#   --edit_json_dir DIR    FiVE-Bench-Correct annotations (default evaluation/five_bench_correct)
#   --cases FILE           optional subset, [{"video_name": ..., "edit_type": ...}]; default: every pair of --edit_json_dir
#   --tag TAG              run tag, part of every output name
#   --allow_nan COL...     metric columns allowed to be NaN (none needed by default): the merge
#                          leaves them out of the NaN check, and the metrics job's 'Error:' gate ignores evaluate.py's
#                          "Error: <COL>: ..." lines for them (any other 'Error:' line still fails the job)
#   --part all|metrics|cf  (launcher) which GPU job(s) to submit, default all. metrics = the 1-GPU job + a merge without CF,
#                          cf = the 4-GPU CF job + a full merge (needs the metrics CSVs from an earlier --part metrics run)
#   --without_cf           (merge) build the tables without CF: its columns are left empty, not NaN (set by --part metrics)
#   --src_window full|render   source frames seen by PSC and CF (default full = the whole source clip). render: the source
#                          cut to the render's frame count, the window the harness / CLIP-D / EditReward already use (they
#                          pair source and render frames after truncating to the shorter). R58-R66 use render (R57 plan,
#                          "Common evaluation procedure")
#
# Output names (evaluation/csv/):
#   edit{T}_FiVE_suite_{tag}_{method}_frame_stride8.csv         harness, one file per edit type
#   edit{T}_FiVE_suite_fiveacc_{tag}_{method}_frame_stride8.csv FiVE-Acc
#   suite_clipd_{tag}_{method}.csv  suite_editreward_...csv  suite_psc_...csv  suite_cf_...csv
#   suite_{tag}_{method}.csv  suite_{tag}_{method}_per_type.csv  suite_{tag}_{method}_overall.csv   (merge: per clip,
#                          per edit type, overall; plain per-clip means)

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
WEIGHTS_DIR=~/Data/weights
# the 147 GB CF judge lives on /work (home quota); /work is readable from the L40S nodes (probe 2026-10-30, node57)
QWEN72B_DIR=${QWEN72B_DIR:-/work/PERSO/skhairi/weights/Qwen2.5-VL-72B-Instruct}
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
PY_ER=/home/ids/skhairi/anaconda3/envs/editreward/bin/python
# CF runs in IVEBench's own env (its ivebench.yml: transformers 4.55.0, torch 2.8.0): under five-bench's transformers 5.3
# the 72B judge outputs NaN from its first language-model layer (R57, 2026-11-04). Built by evaluation/eval_suite/make_cf_env.sh
PY_CF=/home/ids/skhairi/anaconda3/envs/ivebench/bin/python
SUITE_DIR=evaluation/eval_suite

METHOD=""; ROOT=""; LAYOUT=edit_video; EDIT_JSON_DIR=evaluation/five_bench_correct; CASES=""; TAG=""; ALLOW_NAN=()
PART=all; WITHOUT_CF=0; UNITS=(); SRC_WINDOW=full

suite_parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --method) METHOD=$2; shift 2 ;;
      --root) ROOT=$2; shift 2 ;;
      --layout) LAYOUT=$2; shift 2 ;;
      --edit_json_dir) EDIT_JSON_DIR=$2; shift 2 ;;
      --cases) CASES=$2; shift 2 ;;
      --tag) TAG=$2; shift 2 ;;
      --allow_nan) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ALLOW_NAN+=("$1"); shift; done ;;
      --part) PART=$2; shift 2 ;;
      --without_cf) WITHOUT_CF=1; shift ;;
      --unit) UNITS+=("$2"); shift 2 ;;
      --src_window) SRC_WINDOW=$2; shift 2 ;;
      *) echo "[eval_suite] unknown argument: $1"; return 1 ;;
    esac
  done
  case "$PART" in all|metrics|cf) ;; *) echo "[eval_suite] --part must be all, metrics or cf"; return 1 ;; esac
  case "$SRC_WINDOW" in full|render) ;; *) echo "[eval_suite] --src_window must be full or render"; return 1 ;; esac
  SRC_WINDOW_PY=(--src_window "$SRC_WINDOW")
  local c
  for c in "${ALLOW_NAN[@]}"; do
    [[ "$c" =~ ^[a-z0-9_]+$ ]] || { echo "[eval_suite] --allow_nan: bad column name '$c'"; return 1; }
  done
  [ -n "$METHOD" ] && UNITS=("${METHOD}=${ROOT}" "${UNITS[@]}")
  [ -n "$TAG" ] || { echo "[eval_suite] --tag is required"; return 1; }
  [[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "[eval_suite] --tag: letters, digits, _ . - only"; return 1; }
  [ "${#UNITS[@]}" -gt 0 ] || [ -n "${SUITE_NO_UNITS:-}" ] || { echo "[eval_suite] give --method and --root, or --unit NAME=DIR"; return 1; }
  local u seen=" "
  for u in "${UNITS[@]}"; do
    unit_set "$u"
    [[ "$METHOD" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "[eval_suite] method '$METHOD': letters, digits, _ . - only"; return 1; }
    [ -n "$ROOT" ] || [ -n "${SUITE_NO_ROOT:-}" ] || { echo "[eval_suite] method '$METHOD' has no root"; return 1; }
    [[ "$seen" != *" $METHOD "* ]] || { echo "[eval_suite] method '$METHOD' given twice"; return 1; }
    seen="${seen}${METHOD} "
  done
  case "$LAYOUT" in edit_video|flat) ;; *) echo "[eval_suite] --layout must be edit_video or flat"; return 1 ;; esac
  CASES_ARG=(); CASES_PY=()
  if [ -n "$CASES" ]; then CASES_ARG=(--cases_json "$CASES"); CASES_PY=(--cases "$CASES"); fi
  JOB=${SLURM_JOB_ID:-local}
}

# METHOD / ROOT of one unit "NAME=DIR" (split at the first '=')
unit_set() {
  METHOD=${1%%=*}
  ROOT=${1#*=}
  [ "$ROOT" = "$1" ] && ROOT=""
  ROOT=${ROOT/#\~/$HOME}
}

# EXPECT = pair count per edit type (six integers) and ANNOTATIONS = the edit{T}_FiVE.json files of the types with pairs
suite_expect() {
  read -r -a EXPECT <<< "$($PY_FB -I "$SUITE_DIR/suite_common.py" counts --edit_json_dir "$EDIT_JSON_DIR" "${CASES_PY[@]}")" || return 1
  [ "${#EXPECT[@]}" -eq 6 ] || { echo "[eval_suite] could not count the pairs"; return 1; }
  ANNOTATIONS=()
  for T in 1 2 3 4 5 6; do
    [ "${EXPECT[$((T - 1))]}" -gt 0 ] && ANNOTATIONS+=("$EDIT_JSON_DIR/edit${T}_FiVE.json")
  done
}

# per-type row counts of one harness result stem; for five_acc also the binary columns (as r49_eval_all.sh)
check_harness_rows() {  # $1 = stem, $2 = harness|fiveacc
  $PY_FB -I - "$1" "$2" "${EXPECT[@]}" <<'PY'
import csv, sys
stem, kind, expect = sys.argv[1], sys.argv[2], dict(zip(range(1, 7), map(int, sys.argv[3:9])))
need = ("five_acc_yes_no", "five_acc_multi_choice") if kind == "fiveacc" else ()
bad = 0
for t, n_exp in expect.items():
    if n_exp == 0:
        continue
    path = f"evaluation/csv/edit{t}_FiVE_{stem}_frame_stride8.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"[eval_suite] {stem} edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[eval_suite] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), missing cols={miss or 0}, "
          f"non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
}

# per-type row counts of a keyed suite CSV (CLIP-D, EditReward, PSC, CF); $3 = optional method to filter on (CLIP-D)
check_keyed_rows() {  # $1 = csv, $2 = name, $3 = method
  $PY_FB -I - "$1" "$2" "${3:-}" "${EXPECT[@]}" <<'PY'
import csv, sys
from collections import Counter
path, name, method, expect = sys.argv[1], sys.argv[2], sys.argv[3], dict(zip(range(1, 7), map(int, sys.argv[4:10])))
expect = {t: n for t, n in expect.items() if n}
try:
    rows = [r for r in csv.DictReader(open(path)) if not method or r["method"] == method]
except FileNotFoundError:
    print(f"[eval_suite] {name}: MISSING {path}"); sys.exit(1)
got = dict(sorted(Counter(int(r["edit_type"]) for r in rows).items()))
ok = got == expect
print(f"[eval_suite] {name}: {len(rows)} rows {got} (expect {expect}) -- {'ok' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
}

# gate of one stage: exit code, 'Error:' lines of its log, optional row-count check run by the caller
stage_gate() {  # $1 = name, $2 = exit code, $3 = log
  # 'Error:' lines of the --allow_nan columns (evaluate.py prints "Error: <metric>: ...") are counted apart and do not fail
  local NERR NALLOW=0 NOOM RE
  NERR=$(grep -c 'Error:' "$3"); NOOM=$(grep -c 'out of memory' "$3")
  if [ "${#ALLOW_NAN[@]}" -gt 0 ]; then
    RE="^Error: ($(IFS='|'; echo "${ALLOW_NAN[*]}")):"
    NALLOW=$(grep -c -E "$RE" "$3")
  fi
  echo "[eval_suite] $1 exit=$2 error_lines=$((NERR - NALLOW)) allowed_nan_error_lines=${NALLOW} oom_lines=${NOOM} (log $3)"
  { [ "$2" -ne 0 ] || [ "$((NERR - NALLOW))" -ne 0 ]; } && return 1
  return 0
}

# ---- completeness (stage skip on resume, and which merge to run), for the current METHOD
suite_py() { $PY_FB -I "$SUITE_DIR/suite_common.py" "$@" --edit_json_dir "$EDIT_JSON_DIR" "${CASES_PY[@]}"; }

harness_done() {  # $1 = stem: every pair has a row
  local TMPJ N
  TMPJ=$(mktemp -p "${TMPDIR:-/tmp}" suite_miss.XXXXXX)
  N=$(suite_py harness_missing --stem "$1" --out "$TMPJ" ${ALLOW_NAN[@]:+--allow_nan "${ALLOW_NAN[@]}"} | tail -1)
  rm -f "$TMPJ"
  [ "$N" = 0 ]
}

cf_done() { suite_py keyed_complete --csv "evaluation/csv/suite_cf_${TAG}_${METHOD}.csv" --cols cf; }

metrics_done() {  # every stage of the metrics job is complete for METHOD
  harness_done "suite_${TAG}_${METHOD}" && harness_done "suite_fiveacc_${TAG}_${METHOD}" \
    && suite_py keyed_complete --csv "evaluation/csv/suite_clipd_${TAG}_${METHOD}.csv" --cols clip_d_prompt,clip_d_word --method "$METHOD" \
    && suite_py keyed_complete --csv "evaluation/csv/suite_editreward_${TAG}_${METHOD}.csv" --cols editreward,editreward_if,editreward_vq \
    && suite_py keyed_complete --csv "evaluation/csv/suite_psc_${TAG}_${METHOD}.csv" --cols psc
}

# merge METHOD's tables: the full merge when its CF is complete, else --without_cf (CF columns empty). Run at the end of a
# setting by the metrics job and by the CF job, so whichever of the two finishes last writes the full table.
merge_unit() {
  local MA=() KIND=full LOG="logs/eval_suite_${JOB}_${METHOD}_merge.log" RC
  [ "${#ALLOW_NAN[@]}" -gt 0 ] && MA=(--allow_nan "${ALLOW_NAN[@]}")
  if ! cf_done; then MA+=(--without_cf); KIND=without_cf; fi
  $PY_FB -I "$SUITE_DIR/merge_suite.py" --tag "$TAG" --method "$METHOD" --edit_json_dir "$EDIT_JSON_DIR" \
    "${CASES_PY[@]}" --csv_dir evaluation/csv "${MA[@]}" > "$LOG" 2>&1
  RC=$?
  echo "[eval_suite] ${METHOD} merge (${KIND}) exit=${RC}: $(tail -1 "$LOG") (log ${LOG})"
  return "$RC"
}
