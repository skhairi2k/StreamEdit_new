#!/bin/bash
# R44 combo table -- the ONE place the 8 text-only (novp) divergence-routing combos and
# the user's selection are defined. Sourced by every combo-dependent R44 script
# (stage1/stage2/absnorm/stage3/eval/fiveacc/clipd); run directly to size sbatch arrays.
#
# COMBO NAME: {metric}_n{N}_{regime}
#   metric  dino | lpips          -> r31_divergence.py --arm dino_patch | lpips
#   N       5 | 15                -> stage-1 draft step count (stage 3 always renders at 15)
#   regime  tgt09 | unblended     -> --blend_sched tgt0.9 | zero
# Stage-1 RUN = n{N}_{regime}: one draft per RUN, shared by both metrics.
#
# SELECTION: evaluation/r44_selected_combos.txt, one combo per line, written by the user
# at the `choose-config` gate. File order = array-index order in every script (so put the
# highest-priority combo first). Hard-fails on: missing/empty file, blank or comment lines,
# unknown combo, duplicates. Override the path with R44_SELECTION (smoke tests only).
#
# USAGE
#   source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
#   r44_load || exit 1                 # sets R44_COMBOS and R44_RUNS (selection order)
#   COMBO=${R44_COMBOS[$TID]}; r44_parse "$COMBO" || exit 1
#   echo "$METRIC $ARM $N $REGIME $SCHED $RUN"
#
#   bash slurm_scripts/five_bench/r44_combos.sh --combos   # selected combos, one per line
#   bash slurm_scripts/five_bench/r44_combos.sh --runs     # stage-1 runs they need
#   bash slurm_scripts/five_bench/r44_combos.sh --all      # all 8 combos
#
# Functions return non-zero instead of calling `exit`, so sourcing this file into an
# interactive shell is safe; the calling sbatch script decides to exit.

R44_ALL_COMBOS=(dino_n15_tgt09 dino_n15_unblended lpips_n15_tgt09 lpips_n15_unblended
                dino_n5_tgt09  dino_n5_unblended  lpips_n5_tgt09  lpips_n5_unblended)

# Repo root from this file's own location, so the default path also works when the file
# is executed from anywhere (sbatch spools the CALLING script, not this sourced one).
R44_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
R44_SELECTION=${R44_SELECTION:-$R44_REPO/evaluation/r44_selected_combos.txt}

# r44_parse COMBO -> sets METRIC ARM N REGIME SCHED RUN; returns 1 on an unknown combo.
r44_parse() {
  local combo=$1 ntag
  case " ${R44_ALL_COMBOS[*]} " in
    *" $combo "*) ;;
    *) echo "[r44_combos] unknown combo '${combo}' (valid: ${R44_ALL_COMBOS[*]})" >&2; return 1 ;;
  esac
  IFS=_ read -r METRIC ntag REGIME <<< "$combo"
  N=${ntag#n}
  RUN="${ntag}_${REGIME}"
  case $METRIC in dino) ARM=dino_patch ;; lpips) ARM=lpips ;; *) return 1 ;; esac
  case $REGIME in tgt09) SCHED=tgt0.9 ;; unblended) SCHED=zero ;; *) return 1 ;; esac
  case $N in 5|15) ;; *) return 1 ;; esac
}

# r44_selected -> prints the selected combos (file order); returns 1 on any invalid file.
r44_selected() {
  local line n=0 seen=" " bad=0
  if [ ! -s "$R44_SELECTION" ]; then
    echo "[r44_combos] selection file missing or empty: ${R44_SELECTION}" >&2
    echo "[r44_combos]   -- run the choose-config step first (/run-step R44 choose-config)" >&2
    return 1
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    line=${line%$'\r'}
    if [ -z "${line//[[:space:]]/}" ] || [[ $line == \#* ]]; then
      echo "[r44_combos] ${R44_SELECTION}:${n}: blank or comment line not allowed" >&2
      bad=1; continue
    fi
    if [[ $line =~ [[:space:]] ]]; then
      echo "[r44_combos] ${R44_SELECTION}:${n}: whitespace in '${line}'" >&2
      bad=1; continue
    fi
    r44_parse "$line" || { bad=1; continue; }
    case $seen in
      *" $line "*) echo "[r44_combos] ${R44_SELECTION}:${n}: duplicate '${line}'" >&2; bad=1; continue ;;
    esac
    seen="${seen}${line} "
  done < "$R44_SELECTION"
  [ "$bad" -eq 0 ] || return 1
  # Second pass prints only once the whole file is known valid (no partial output).
  grep -v '^$' "$R44_SELECTION" | tr -d '\r'
}

# r44_runs -> prints the unique stage-1 RUNs of the selection, first-appearance order.
r44_runs() {
  local combos c seen=" "
  combos=$(r44_selected) || return 1
  for c in $combos; do
    r44_parse "$c" || return 1
    case $seen in *" $RUN "*) continue ;; esac
    seen="${seen}${RUN} "
    echo "$RUN"
  done
}

# r44_load -> sets the arrays R44_COMBOS and R44_RUNS; returns 1 if the selection is invalid.
r44_load() {
  R44_COMBOS=($(r44_selected)) || return 1
  R44_RUNS=($(r44_runs)) || return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case ${1:-} in
    --combos) r44_selected ;;
    --runs)   r44_runs ;;
    --all)    printf '%s\n' "${R44_ALL_COMBOS[@]}" ;;
    *) echo "usage: $0 --combos | --runs | --all" >&2; exit 2 ;;
  esac
fi
