#!/bin/bash
# R60 render -- TEXT-ONLY (novp) uniform-rho StreamEdit on FiVE-Bench-Correct (R44's sampler, Correct annotations).
#
# USAGE:
#   sbatch slurm_scripts/five_bench/r60_render.sh correct             (--array=0-15 is in the header)
#   sbatch --array=0-1 slurm_scripts/five_bench/r60_render.sh repro   (reproducibility gate)
# MODE correct, ARRAY LAYOUT: 16 tasks, task i -> (N, rho, case file); each task loops the edit types of its case file.
#   0-7   N=15  rho 2 3 4 6 8 10 20 50   evaluation/r60_cases_regen.json (190 pairs, prompt / trigger word changed;
#                                         the other 225 reuse R44's renders)
#   8-11  N=20  rho 2 4 10 50            evaluation/r60_cases_regen.json (190)
#   12-15 N=20  rho 3 6 8 20             evaluation/r43_anchor_manifest_correct.json (415; R44 never rendered these)
#   out: $D/r60_rho_sweep/r60_n{N}_rho{R}_novp/edit{T}/{video}/*.png
# MODE repro, ARRAY LAYOUT: 2 tasks, 0 = N=15 rho 2, 1 = N=20 rho 2, on evaluation/r60_cases_repro.json (2 unchanged pairs);
#   out: $D/r60_repro/r60_n{N}_rho2_novp. r60_repro_check.py compares them with R44's renders (gate for the reuse).
#
# Sampler identical to r44_infer.sh: --vp_mode novp, NO --first_frame_edit_dir, fg_boost 4, flow_shift 1.0,
# rollout_chunk_size 21, seed 0. Added: --edit_json_dir evaluation/five_bench_correct, --cases_json.
# Expected clip count per edit type is read from the case file.
#
# --time=24:00:00 (partition max): R39 estimates ~14 h for 419 clips at N=20 (tasks 12-15 render 415).
# RESUME: an edit type already complete (dirs + all-ok manifest) is skipped, but run_fivebench.py has no per-clip skip, so a
# timed-out task re-renders the edit type it was in: resubmit just that index. A failed edit type does NOT stop the loop;
# the task exits non-zero at the end.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404).
# --exclude=node52: exposes no GPU to batch jobs.
#SBATCH --job-name=r60_render
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/r60_render_%A_%a.out
#SBATCH --error=logs/r60_render_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANNO_DIR=evaluation/five_bench_correct
CASES_REGEN=evaluation/r60_cases_regen.json
CASES_REPRO=evaluation/r60_cases_repro.json
CASES_ALL=evaluation/r43_anchor_manifest_correct.json

MODE=${1:-}
TID=${SLURM_ARRAY_TASK_ID}
case "$MODE" in
  correct)
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 15 ]; then
      echo "[r60_render] bad array id '${TID}' (expected 0-15)"; exit 1
    fi
    OUT_ROOT="$D/r60_rho_sweep"
    RHOS_15=(2 3 4 6 8 10 20 50)
    RHOS_20A=(2 4 10 50)
    RHOS_20B=(3 6 8 20)
    if [ "$TID" -le 7 ]; then
      STEPS=15; R=${RHOS_15[$TID]}; CASES=$CASES_REGEN
    elif [ "$TID" -le 11 ]; then
      STEPS=20; R=${RHOS_20A[$((TID - 8))]}; CASES=$CASES_REGEN
    else
      STEPS=20; R=${RHOS_20B[$((TID - 12))]}; CASES=$CASES_ALL
    fi
    ;;
  repro)
    if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 1 ]; then
      echo "[r60_render] bad array id '${TID}' for repro (expected 0-1; submit with --array=0-1)"; exit 1
    fi
    OUT_ROOT="$D/r60_repro"
    STEPS=$(( TID == 0 ? 15 : 20 )); R=2; CASES=$CASES_REPRO
    ;;
  *)
    echo "[r60_render] usage: sbatch slurm_scripts/five_bench/r60_render.sh correct|repro"; exit 1 ;;
esac
METHOD="r60_n${STEPS}_rho${R}_novp"

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

# Expected clips per edit type, from the case file (index 0 = edit1 ... 5 = edit6).
read -r -a EXPECTED <<< "$(python -c "
import json, sys
c = json.load(open('$CASES'))
print(' '.join(str(sum(int(x['edit_type']) == t for x in c)) for t in range(1, 7)))")"
TOTAL_EXPECTED=0; for n in "${EXPECTED[@]}"; do TOTAL_EXPECTED=$((TOTAL_EXPECTED + n)); done
if [ "${#EXPECTED[@]}" -ne 6 ] || [ "$TOTAL_EXPECTED" -eq 0 ]; then
  echo "[r60_render] cannot read per-type counts from ${CASES}"; exit 1
fi

echo "[r60_render] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r60_render] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} method=${METHOD} steps=${STEPS} rho=${R} cases=${CASES} expect=${EXPECTED[*]} (${TOTAL_EXPECTED})"
# (no "vp_mode=" in this echo: the wait step greps for run_fivebench.py's own vp_mode=novp line)

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  if [ "$EXP" -eq 0 ]; then
    echo "--- [r60_render] ${METHOD} edit${T}: no case in ${CASES}, skipped ---"
    continue
  fi
  echo "--- [r60_render] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished. Complete = EXP real clip dirs AND a
  # manifest with exactly EXP rows, all ok. Anything less is re-rendered whole, overwriting
  # clips cut off mid-write.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r60_render] SKIP ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
    continue
  fi

  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode novp \
    --blend_power "$R" \
    --data_root "$DATA_ROOT" --edit_json_dir "$ANNO_DIR" --cases_json "$CASES" --out_root "$OUT_ROOT" \
    --step "$STEPS" --fg_boost_factor 4 --flow_shift 1.0 --rollout_chunk_size 21 --seed 0 \
    || { echo "[r60_render] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r60_render] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r60_render] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
    FAILED=$((FAILED + 1))
  fi

  # A per-pair error is recorded in the manifest rather than raised, so a clean exit code
  # is not evidence the edit type is complete.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  if [ -f "$MAN" ]; then
    N_BAD=$(awk -F, 'NR>1 && $3!="ok"' "$MAN" | wc -l)
  else
    N_BAD=-1
  fi
  echo "[r60_render] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r60_render] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r60_render] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
echo "[r60_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
