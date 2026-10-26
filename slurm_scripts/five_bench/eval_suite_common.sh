#!/bin/bash
# R57 unified metric suite -- shared by eval_suite.sh (launcher) and eval_suite_{metrics,cf,merge}.sh. Sourced, never run.
#
# Arguments (same for the launcher and every job):
#   --method NAME          label of the method (CSV column key, file names)
#   --root DIR             method render root; frames at DIR/edit{T}/{video}/ (--layout edit_video) or DIR/{video}/ (flat)
#   --layout edit_video|flat           (default edit_video)
#   --edit_json_dir DIR    FiVE-Bench-Correct annotations (default evaluation/five_bench_correct)
#   --cases FILE           optional subset, [{"video_name": ..., "edit_type": ...}]; default: every pair of --edit_json_dir
#   --tag TAG              run tag, part of every output name
#   --allow_nan COL...     (merge only) metric columns allowed to be NaN, e.g. motion_fidelity_score_edit_part on a full run
#
# Output names (evaluation/csv/):
#   edit{T}_FiVE_suite_{tag}_{method}_frame_stride8.csv         harness, one file per edit type
#   edit{T}_FiVE_suite_fiveacc_{tag}_{method}_frame_stride8.csv FiVE-Acc
#   suite_clipd_{tag}_{method}.csv  suite_editreward_...csv  suite_psc_...csv  suite_cf_...csv
#   suite_{tag}_{method}.csv  suite_{tag}_{method}_per_type.csv     (merge)

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
WEIGHTS_DIR=~/Data/weights
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
PY_ER=/home/ids/skhairi/anaconda3/envs/editreward/bin/python
SUITE_DIR=evaluation/eval_suite

METHOD=""; ROOT=""; LAYOUT=edit_video; EDIT_JSON_DIR=evaluation/five_bench_correct; CASES=""; TAG=""; ALLOW_NAN=()

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
      *) echo "[eval_suite] unknown argument: $1"; return 1 ;;
    esac
  done
  [ -n "$METHOD" ] && { [ -n "$ROOT" ] || [ -n "${SUITE_NO_ROOT:-}" ]; } && [ -n "$TAG" ] \
    || { echo "[eval_suite] --method, --root and --tag are required"; return 1; }
  [[ "$METHOD" =~ ^[A-Za-z0-9_.-]+$ && "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "[eval_suite] --method/--tag: letters, digits, _ . - only"; return 1; }
  case "$LAYOUT" in edit_video|flat) ;; *) echo "[eval_suite] --layout must be edit_video or flat"; return 1 ;; esac
  ROOT=${ROOT/#\~/$HOME}
  CASES_ARG=(); CASES_PY=()
  if [ -n "$CASES" ]; then CASES_ARG=(--cases_json "$CASES"); CASES_PY=(--cases "$CASES"); fi
  JOB=${SLURM_JOB_ID:-local}
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
  local NERR NOOM
  NERR=$(grep -c 'Error:' "$3"); NOOM=$(grep -c 'out of memory' "$3")
  echo "[eval_suite] $1 exit=$2 error_lines=${NERR} oom_lines=${NOOM} (log $3)"
  { [ "$2" -ne 0 ] || [ "$NERR" -ne 0 ]; } && return 1
  return 0
}
