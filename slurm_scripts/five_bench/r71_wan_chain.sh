#!/bin/bash
# R71 Wan-Edit track (ON JEAN ZAY, H100, in the clone, after r50_port.sh and r50_scratch_links.sh): the range-normalized ControlLift mapping
# m = round(49 q), s = 50 - m (m_max = 49, s_min = 1) on all 415 FiVE-Bench-Correct pairs, plus the equal-NFE curve it needs (63 active steps).
#
#   field   r71_field.sh                     gate vs the s_min 5 fields, remap q -> s = 50 - round(49 q)  -> r71_field/wan_{max,nosupp,nodiv,dino,lpips}_s1
#     -> render  r71_wan_render.sh --array=0-4   max_s1, nosupp_s1, nodiv_s1, dino_s1, lpips_s1 (N = 50, loop start 1)   -> r71_wan_arms  afterok field
#   render  r71_wan_render.sh --array=5-10     matched63_s{0.996,0.978,0.95,0.92,0.85,0.5} (global start, 63 active steps)  -> r71_wan_arms  no dependency
# No eval stage: every config is scored on the lab (R57 suite) after jz/r71_fetch_renders.sh.
# No upstream job: the s_min 5 fields finished long ago. The driver checks that every source root of r71_field.sh is on $SCRATCH (a missing
# root is FATAL with its /store archive path: push it back first). Every R71 render is new (nothing reused from R49 / R58), so R58's
# repro-check is not a gate here.
# Writes "<stage> <id>" lines to logs/r71_wan_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists;
# delete it deliberately to resubmit. Renders are RESUMABLE: a timed-out config is resubmitted alone (sbatch --array=<i> slurm_scripts/five_bench/r71_wan_render.sh).
#
# Usage: bash slurm_scripts/five_bench/r71_wan_chain.sh [--dry-run]    (--dry-run: run every guard, print the sbatch lines, submit nothing)
#        R50_ENV=<file> overrides jz/r50_jz.env (tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r71_wan_chain] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r71_wan_chain] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: ported scripts, code, case files
SB=slurm_scripts/five_bench
SCRIPTS=($SB/r71_field.sh $SB/r71_wan_render.sh)
for F in "${SCRIPTS[@]}" evaluation/r71_remap_field.py evaluation/r58_cases_maxs5.json evaluation/r68_cases_reuse225.json \
         evaluation/r43_anchor_manifest_correct.json evaluation/five_bench_correct/edit1_FiVE.json; do
  [ -s "$F" ] || { echo "[r71_wan_chain] FATAL: missing $F (git pull?)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r71_wan_chain] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
for C in max_s1 nosupp_s1 nodiv_s1 dino_s1 lpips_s1; do
  grep -q "\"${C}\"" evaluation/r49_wan_render.py || { echo "[r71_wan_chain] FATAL: r49_wan_render.py lacks config ${C} (git pull?)"; exit 1; }
done
grep -q '^MATCHED63' evaluation/r49_wan_render.py || { echo "[r71_wan_chain] FATAL: r49_wan_render.py lacks MATCHED63 (git pull?)"; exit 1; }
grep -q -- '--anno_dir' evaluation/r49_wan_render.py || { echo "[r71_wan_chain] FATAL: r49_wan_render.py lacks --anno_dir (git pull?)"; exit 1; }
N_190=$(grep -c '"case_id"' evaluation/r58_cases_maxs5.json)        # one "case_id" key per entry (no python on the frontal needed)
N_225=$(grep -c '"case_id"' evaluation/r68_cases_reuse225.json)
[ "$N_190" -eq 190 ] && [ "$N_225" -eq 225 ] || { echo "[r71_wan_chain] FATAL: case files hold ${N_190} / ${N_225} pairs (expect 190 / 225)"; exit 1; }

# ---- outputs go to $SCRATCH, never overwritten
D=$HOME/Data/dataggen/outputs/five_bench
if [ "${R71_ALLOW_WORK_OUTPUT:-0}" != 1 ]; then
  for X in r71_field r71_wan_arms; do
    [ -L "$D/$X" ] || { echo "[r71_wan_chain] FATAL: $D/$X is not a symlink to \$SCRATCH -- run: bash jz/r50_scratch_links.sh (R71_ALLOW_WORK_OUTPUT=1 overrides, at the risk of the \$WORK file quota)"; exit 1; }
  done
fi
for A in max nosupp nodiv dino lpips; do
  [ -n "$(ls -A "$D/r71_field/wan_${A}_s1" 2>/dev/null)" ] && { echo "[r71_wan_chain] REFUSE: $D/r71_field/wan_${A}_s1 already holds files -- R71 fields are never overwritten"; exit 1; }
done

# ---- source roots of r71_field.sh, on $SCRATCH (archived on /store, push back if purged)
STORE=/store/PERSO/skhairi/Outputs/jz_five_bench
BAD=0
check() {  # $1 = dir under $D, $2 = minimum npz count
  local n; n=$(ls "$D/$1"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r71_wan_chain] $1: ${n} npz (need >= $2)"
  [ "$n" -ge "$2" ] || { echo "[r71_wan_chain]   MISSING: push it back from the archive ${STORE}/$1 (lab frontal)"; BAD=$((BAD + 1)); }
}
check r58_field/wan_max_s5 190;     check r49_field_full/wan_max_s5 225
check r62_field/wan_nosupp_s5 190;  check r51_field_full/wan_nosupp_s5 225
check r68_field/wan_nodiv_s5 415;   check r68_field/wan_dino_s5 415;   check r68_field/wan_lpips_s5 415
[ "$BAD" -ne 0 ] && { echo "[r71_wan_chain] FATAL: ${BAD} source root(s) incomplete"; exit 1; }

JOBS=logs/r71_wan_chain_jobs.txt
if [ -e "$JOBS" ] && [ "$DRY" -eq 0 ]; then
  echo "[r71_wan_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r71_wan_chain] delete it deliberately to resubmit"; exit 1
fi
[ "$DRY" -eq 0 ] && echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname) | no upstream job" > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch arguments; prints the id, records it, aborts if refused
  local stage=$1; shift
  if [ "$DRY" -eq 1 ]; then
    echo "[r71_wan_chain] dry-run ${stage}: sbatch $*" >&2; echo "9$(printf '%s' "$stage" | cksum | cut -c1-5)"; return
  fi
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r71_wan_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r71_wan_chain] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r71_wan_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r71_wan_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

# optional walltime overrides (e.g. to fit before a maintenance reservation; the render resumes afterwards):
#   R71_FIELD_TIME=00:15:00 R71_RENDER_TIME=08:00:00 bash slurm_scripts/five_bench/r71_wan_chain.sh
FT=(); [ -n "${R71_FIELD_TIME:-}" ] && FT=(--time="$R71_FIELD_TIME")
RT=(); [ -n "${R71_RENDER_TIME:-}" ] && RT=(--time="$R71_RENDER_TIME")
[ "$DRY" -eq 0 ] && [ $(( ${#FT[@]} + ${#RT[@]} )) -gt 0 ] && echo "# walltime overrides: field ${R71_FIELD_TIME:-default} render ${R71_RENDER_TIME:-default}" >> "$JOBS"
FD=$(submit field "${FT[@]}" $SB/r71_field.sh) || exit 1
RA=$(submit render_adapt "${RT[@]}" --dependency=afterok:${FD} --array=0-4 $SB/r71_wan_render.sh) || exit 1
RG=$(submit render_global "${RT[@]}" --array=5-10 $SB/r71_wan_render.sh) || exit 1

if [ "$DRY" -eq 1 ]; then
  echo "[r71_wan_chain] DRY RUN, nothing submitted (the ids above are placeholders): field ${FD} -> render_adapt ${RA} (array 0-4); render_global ${RG} (array 5-10)"
else
  echo "[r71_wan_chain] submitted: field ${FD} -> render_adapt ${RA} (array 0-4: max_s1 nosupp_s1 nodiv_s1 dino_s1 lpips_s1); render_global ${RG} (array 5-10: matched63 x6)"
  echo "[r71_wan_chain] ids in ${JOBS}; follow with: squeue -u \$USER"
fi
