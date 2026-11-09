#!/bin/bash
# R68 Wan-Edit track (ON JEAN ZAY, H100, in the clone, after r50_port.sh and r50_scratch_links.sh): R51's demand ablations A1-A3 on
# FiVE-Bench-Correct, all 415 pairs (never rendered before, nothing reused).
#
#   field   r68_field.sh wan                 A0 gate vs r58_field/wan_max_s5 (190/190) and vs r49_field_full/wan_max_s5 (225/225)
#                                            + A1-A3 start maps                                       -> r68_field/wan_{nodiv,dino,lpips}_s5
#     -> render  r68_wan_render.sh --array=0-2   nodiv_s5, dino_s5, lpips_s5 (N = 50, loop start 5), 415 pairs each -> r68_wan_arms  afterok field
# No eval stage: every arm is scored on the lab (R57 suite) after jz/r68_fetch_renders.sh.
# No upstream job: R58's and R49's fields finished long ago. The driver instead checks that every input of r68_field.sh wan is on $SCRATCH:
#   190 pairs (r58_cases_maxs5.json):    r58_wan_draft/attn, r58_div/wan_{dino,lpips} (>= 190 npz), r58_field/wan_max_s5 (190 .npy)
#   225 pairs (r68_cases_reuse225.json): r48_wan_draft/attn, r48_div/wan_{dino,lpips} (>= 225 npz), r49_field_full/wan_max_s5 (>= 225 .npy)
# A missing root is FATAL with its /store archive path (/store/PERSO/skhairi/Outputs/jz_five_bench/<root>, push it back first).
# A0 on the 225 unchanged pairs is R49's render (R58's assembly links it), so A1-A3 vs A0 compares renders made at different times: R58's
# reproducibility gate (evaluation/csv/r58_repro_check.csv) must pass, as for R62.
# Writes "<stage> <id>" lines to logs/r68_wan_chain_jobs.txt, headed by the git commit, and REFUSES to run if that file exists;
# delete it deliberately to resubmit. Renders are RESUMABLE: a timed-out arm is resubmitted alone (sbatch --array=<i> slurm_scripts/five_bench/r68_wan_render.sh).
#
# Usage: bash slurm_scripts/five_bench/r68_wan_chain.sh [--dry-run]    (--dry-run: run every guard, print the sbatch lines, submit nothing)
#        R50_ENV=<file> overrides jz/r50_jz.env (tests)
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
cd "$REPO" || exit 1
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r68_wan_chain] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
[ -d "$WORK" ] || { echo "[r68_wan_chain] FATAL: $WORK does not exist on $(hostname) -- this driver is for Jean Zay only"; exit 1; }
mkdir -p logs

# ---- guards: ported scripts, code, case files
SB=slurm_scripts/five_bench
SCRIPTS=($SB/r68_field.sh $SB/r68_wan_render.sh)
for F in "${SCRIPTS[@]}" evaluation/r51_field.py evaluation/r58_cases_maxs5.json evaluation/r68_cases_reuse225.json \
         evaluation/r43_anchor_manifest_correct.json evaluation/five_bench_correct/edit1_FiVE.json; do
  [ -s "$F" ] || { echo "[r68_wan_chain] FATAL: missing $F (git pull? r68_cases_reuse225.json: python evaluation/r68_build_cases.py on the lab, commit)"; exit 1; }
done
for F in "${SCRIPTS[@]}"; do
  grep -q '^# \[r50_port\]' "$F" || { echo "[r68_wan_chain] FATAL: $F is not ported to Jean Zay -- run: bash jz/r50_port.sh"; exit 1; }
done
for C in nodiv_s5 dino_s5 lpips_s5; do
  grep -q "\"${C}\"" evaluation/r49_wan_render.py || { echo "[r68_wan_chain] FATAL: r49_wan_render.py lacks config ${C} (git pull?)"; exit 1; }
done
grep -q -- '--anno_dir' evaluation/r49_wan_render.py || { echo "[r68_wan_chain] FATAL: r49_wan_render.py lacks --anno_dir (git pull?)"; exit 1; }
N_190=$(grep -c '"case_id"' evaluation/r58_cases_maxs5.json)        # one "case_id" key per entry (no python on the frontal needed)
N_225=$(grep -c '"case_id"' evaluation/r68_cases_reuse225.json)
[ "$N_190" -eq 190 ] && [ "$N_225" -eq 225 ] || { echo "[r68_wan_chain] FATAL: case files hold ${N_190} / ${N_225} pairs (expect 190 / 225)"; exit 1; }

# ---- outputs go to $SCRATCH, never overwritten
D=$HOME/Data/dataggen/outputs/five_bench
if [ "${R68_ALLOW_WORK_OUTPUT:-0}" != 1 ]; then
  for X in r68_field r68_wan_arms; do
    [ -L "$D/$X" ] || { echo "[r68_wan_chain] FATAL: $D/$X is not a symlink to \$SCRATCH -- run: bash jz/r50_scratch_links.sh (R68_ALLOW_WORK_OUTPUT=1 overrides, at the risk of the \$WORK file quota)"; exit 1; }
  done
fi
for X in r68_field/wan_nodiv_s5 r68_field/wan_dino_s5 r68_field/wan_lpips_s5; do
  [ -n "$(ls -A "$D/$X" 2>/dev/null)" ] && { echo "[r68_wan_chain] REFUSE: $D/$X already holds files -- R68 fields are never overwritten"; exit 1; }
done

# ---- inputs of r68_field.sh wan, on $SCRATCH (archived on /store, push back if purged)
STORE=/store/PERSO/skhairi/Outputs/jz_five_bench
BAD=0
check() {  # $1 = dir under $D, $2 = extension, $3 = minimum count
  local n; n=$(ls "$D/$1"/edit*/*."$2" 2>/dev/null | wc -l)
  echo "[r68_wan_chain] $1: ${n} .$2 (need >= $3)"
  [ "$n" -ge "$3" ] || { echo "[r68_wan_chain]   MISSING: push it back from the archive ${STORE}/$1 (lab frontal)"; BAD=$((BAD + 1)); }
}
check r58_wan_draft/attn npz 190; check r58_div/wan_dino npz 190; check r58_div/wan_lpips npz 190; check r58_field/wan_max_s5 npy 190
check r48_wan_draft/attn npz 225; check r48_div/wan_dino npz 225; check r48_div/wan_lpips npz 225; check r49_field_full/wan_max_s5 npy 225
[ "$BAD" -ne 0 ] && { echo "[r68_wan_chain] FATAL: ${BAD} input root(s) incomplete"; exit 1; }

REPRO=evaluation/csv/r58_repro_check.csv
[ -s "$REPRO" ] || { echo "[r68_wan_chain] FATAL: $REPRO missing -- R58's repro-check has not run"; exit 1; }
if awk -F, 'NR > 1 && $NF != 1 { bad++ } END { exit (bad > 0 || NR < 5) }' "$REPRO"; then :; else
  echo "[r68_wan_chain] FATAL: $REPRO has a failing row or fewer than 4 rows -- new renders do NOT reproduce R49's, so A0 on the 225 unchanged pairs is not comparable"; exit 1
fi

JOBS=logs/r68_wan_chain_jobs.txt
if [ -e "$JOBS" ] && [ "$DRY" -eq 0 ]; then
  echo "[r68_wan_chain] REFUSE: $JOBS exists -- the chain was already submitted:"; cat "$JOBS"
  echo "[r68_wan_chain] delete it deliberately to resubmit"; exit 1
fi
[ "$DRY" -eq 0 ] && echo "# commit $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo') | modified files: $(git status --porcelain 2>/dev/null | wc -l) | $(date '+%F %T') | $(hostname) | no upstream job" > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch arguments; prints the id, records it, aborts if refused
  local stage=$1; shift
  if [ "$DRY" -eq 1 ]; then
    echo "[r68_wan_chain] dry-run ${stage}: sbatch $*" >&2; echo "9$(printf '%s' "$stage" | cksum | cut -c1-5)"; return
  fi
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r68_wan_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r68_wan_chain] already submitted (scancel them before retrying): $(grep -v '^#' "$JOBS" | paste -sd' ')" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r68_wan_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r68_wan_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

# optional walltime overrides (e.g. to fit before a maintenance reservation; the render resumes afterwards):
#   R68_FIELD_TIME=00:15:00 R68_RENDER_TIME=08:00:00 bash slurm_scripts/five_bench/r68_wan_chain.sh
FT=(); [ -n "${R68_FIELD_TIME:-}" ] && FT=(--time="$R68_FIELD_TIME")
RT=(); [ -n "${R68_RENDER_TIME:-}" ] && RT=(--time="$R68_RENDER_TIME")
[ "$DRY" -eq 0 ] && [ $(( ${#FT[@]} + ${#RT[@]} )) -gt 0 ] && echo "# walltime overrides: field ${R68_FIELD_TIME:-default} render ${R68_RENDER_TIME:-default}" >> "$JOBS"
FD=$(submit field "${FT[@]}" $SB/r68_field.sh wan) || exit 1
RD=$(submit render "${RT[@]}" --dependency=afterok:${FD} --array=0-2 $SB/r68_wan_render.sh) || exit 1

if [ "$DRY" -eq 1 ]; then
  echo "[r68_wan_chain] DRY RUN, nothing submitted (the ids above are placeholders): field ${FD} -> render ${RD} (array 0-2)"
else
  echo "[r68_wan_chain] submitted: field ${FD} -> render ${RD} (array 0-2: nodiv_s5, dino_s5, lpips_s5)"
  echo "[r68_wan_chain] ids in ${JOBS}; follow with: squeue -u \$USER"
fi
