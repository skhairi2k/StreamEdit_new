#!/bin/bash
# R41 config -- the ONE place the draft length N, the combos and the binary cuts are set.
#
# R41 compares the continuous (budget-linear) m -> rho mapping with a binary one
# (rho = 2 if m < theta else 50) on the FINAL method's stage-1 draft length N. Binary and
# continuous arms must share N, so every R41 script reads N from here and nowhere else.
#
# SET AT gate-draft (after R38 is analyzed), NOT BEFORE:
#   DRAFT_N   5 or 15. Left EMPTY on purpose: every R41 script refuses to run until it is set.
#   COMBOS    {dino, lpips} x {unblended, tgt09}. tgt09 is the tau = 0.9 regime; at N=15 it IS
#             R35's `first2` (byte-identical, R38 smoke G1). Default: all four. At N=5 remove
#             the tgt09 pair only if R38's gate-lowN dropped it.
#   THETAS    binary cuts on m, read as a curve (no single cut is defended).
#
# Upstream folder names (R35 `*_first2`, R38 `*_n5_*`) appear ONLY in SRC_NAME below; every
# R41 path, stem, figure label and CSV row uses the four combo names.
#
# TESTS ONLY: R41_CONFIG=<path> makes the R41 scripts source another config file (this one
# keeps DRAFT_N empty), and R41_TEST_D=<dir> moves every data root under <dir>.
#
# Sourced by the R41 Slurm scripts (`source slurm_scripts/five_bench/r41_config.sh`), which
# then have: DRAFT_N TAG COMBOS THETAS THETA_ORDER ARMS ARM_COMBO ARM_THETA SRC_NAME
# DIV_ABS_ROOT CONT_RHO_ROOT CONT_ARMS_ROOT CONT_PREFIX CONT_CLIPD_CSV RHO_ROOT ARMS_ROOT
# SMOKE_ROOT EXPECT, and the banner on stdout. Also runnable:
#   bash r41_config.sh                     banner (fails if DRAFT_N is unset)
#   bash r41_config.sh --draft_n           prints DRAFT_N (fails if unset)
#   bash r41_config.sh --max_index         len(ARMS)-1, for `sbatch --array=0-<this>`
#   bash r41_config.sh --max_theta_index   len(THETAS)-1, for r41_clipd.sh
#   bash r41_config.sh --json [N]          resolved config as JSON for the Python scripts.
#                                          N overrides DRAFT_N -- ONLY for gate-draft's
#                                          --precheck, which compares both N before one is set.

DRAFT_N=5            # set at gate-draft, 2026-10-12: final routing config (2026-10-07) is
                     # DINO * tau=0.9 at N=5 -- ONE combo, not the original four
COMBOS=(dino_tgt09)
THETAS=(0.3 0.5 0.7) # theta=0.7 leaves <0.5% FG in 36% of dino_tgt09@N=5 clips (precheck,
                     # evaluation/csv/r41_precheck_n5.csv) -- kept anyway, not "most" clips;
                     # read it as the under-editing end of the binary curve, not a typical point

# ---------------------------------------------------------------------------------------
# R41_TEST_D: synthetic tests ONLY -- moves every input AND output root under a scratch dir.
R41_D=${R41_TEST_D:-$HOME/Data/dataggen/outputs/five_bench}
ALL_COMBOS=(dino_unblended dino_tgt09 lpips_unblended lpips_tgt09)
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)

# r41_resolve <N>: set every N-dependent name. Returns non-zero (with a message) on any
# invalid setting, so a sourcing script can exit before touching data.
r41_resolve () {
  local N=$1 c t
  case "$N" in
    15) DIV_ABS_ROOT=$R41_D/r35_div_abs; CONT_RHO_ROOT=$R41_D/r35_rho
        CONT_ARMS_ROOT=$R41_D/r35_arms;  CONT_PREFIX=r35
        declare -gA SRC_NAME=([dino_unblended]=dino_unblended [dino_tgt09]=dino_first2
                              [lpips_unblended]=lpips_unblended [lpips_tgt09]=lpips_first2) ;;
    5)  DIV_ABS_ROOT=$R41_D/r38_div_abs; CONT_RHO_ROOT=$R41_D/r38_rho
        CONT_ARMS_ROOT=$R41_D/r38_arms;  CONT_PREFIX=r38
        declare -gA SRC_NAME=([dino_unblended]=dino_n5_unblended [dino_tgt09]=dino_n5_tgt09
                              [lpips_unblended]=lpips_n5_unblended [lpips_tgt09]=lpips_n5_tgt09) ;;
    "") echo "[r41] DRAFT_N is not set -- run gate-draft first: N=5 or N=15?" >&2; return 1 ;;
    *)  echo "[r41] DRAFT_N='$N' -- must be 5 or 15" >&2; return 1 ;;
  esac
  CONT_CLIPD_CSV=evaluation/csv/${CONT_PREFIX}_clip_directional.csv
  TAG=n$N
  RHO_ROOT=$R41_D/r41_rho/$TAG
  ARMS_ROOT=$R41_D/r41_arms/$TAG
  SMOKE_ROOT=$R41_D/r41_smoke/$TAG

  [ "${#COMBOS[@]}" -gt 0 ] || { echo "[r41] COMBOS is empty" >&2; return 1; }
  for c in "${COMBOS[@]}"; do
    [[ " ${ALL_COMBOS[*]} " == *" $c "* ]] \
      || { echo "[r41] unknown combo '$c' (expected one of ${ALL_COMBOS[*]})" >&2; return 1; }
  done
  [ "${#THETAS[@]}" -gt 0 ] || { echo "[r41] THETAS is empty" >&2; return 1; }
  for t in "${THETAS[@]}"; do
    awk -v t="$t" 'BEGIN { exit !(t > 0 && t < 1) }' \
      || { echo "[r41] theta '$t' must be in (0, 1)" >&2; return 1; }
  done

  # Theta order: 0.5 first (the headline cut renders first), then THETAS as listed.
  THETA_ORDER=()
  for t in "${THETAS[@]}"; do [ "$t" = "0.5" ] && THETA_ORDER+=("$t"); done
  for t in "${THETAS[@]}"; do [ "$t" != "0.5" ] && THETA_ORDER+=("$t"); done

  # ARMS: theta-major in THETA_ORDER, combos in COMBOS order. Suffix t03 / t05 / t07.
  ARMS=(); declare -gA ARM_COMBO=() ARM_THETA=()
  local arm
  for t in "${THETA_ORDER[@]}"; do
    for c in "${COMBOS[@]}"; do
      arm="${c}_t${t/./}"
      ARMS+=("$arm"); ARM_COMBO[$arm]=$c; ARM_THETA[$arm]=$t
    done
  done
  return 0
}

r41_banner () {
  echo "[r41] DRAFT_N=${DRAFT_N} fields=${DIV_ABS_ROOT} continuous=${CONT_PREFIX}_* " \
       "combos=${COMBOS[*]} thetas=${THETA_ORDER[*]} arms=${#ARMS[@]}"
}

r41_json () {
  local first=1 a
  printf '{"draft_n": %s, "tag": "%s", "combos": [' "$1" "$TAG"
  printf '%s' "$(printf '"%s", ' "${COMBOS[@]}" | sed 's/, $//')"
  printf '], "thetas": [%s], ' "$(printf '%s, ' "${THETA_ORDER[@]}" | sed 's/, $//')"
  printf '"src_name": {%s}, ' "$(for c in "${COMBOS[@]}"; do printf '"%s": "%s", ' "$c" "${SRC_NAME[$c]}"; done | sed 's/, $//')"
  printf '"arms": ['
  for a in "${ARMS[@]}"; do
    [ $first -eq 1 ] || printf ', '
    first=0
    printf '{"arm": "%s", "combo": "%s", "theta": %s}' "$a" "${ARM_COMBO[$a]}" "${ARM_THETA[$a]}"
  done
  printf '], "div_abs_root": "%s", "cont_rho_root": "%s", "cont_arms_root": "%s", ' \
    "$DIV_ABS_ROOT" "$CONT_RHO_ROOT" "$CONT_ARMS_ROOT"
  printf '"cont_prefix": "%s", "cont_clipd_csv": "%s", "rho_root": "%s", "arms_root": "%s"}\n' \
    "$CONT_PREFIX" "$CONT_CLIPD_CSV" "$RHO_ROOT" "$ARMS_ROOT"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # Sourced by an R41 script: resolve the configured N or stop that script here.
  r41_resolve "$DRAFT_N" || exit 1
  r41_banner
else
  case "$1" in
    "")                r41_resolve "$DRAFT_N" || exit 1; r41_banner ;;
    --draft_n)         r41_resolve "$DRAFT_N" || exit 1; echo "$DRAFT_N" ;;
    --max_index)       r41_resolve "$DRAFT_N" || exit 1; echo $(( ${#ARMS[@]} - 1 )) ;;
    --max_theta_index) r41_resolve "$DRAFT_N" || exit 1; echo $(( ${#THETAS[@]} - 1 )) ;;
    --json)            N=${2:-$DRAFT_N}; r41_resolve "$N" || exit 1; r41_json "$N" ;;
    *) echo "usage: bash $0 [--draft_n | --max_index | --max_theta_index | --json [N]]" >&2
       exit 2 ;;
  esac
fi
