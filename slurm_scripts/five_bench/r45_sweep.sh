#!/bin/bash
# R45 sweep -- LongLive (LongLive-native config) + visual prompting rho sweep over the WHOLE
# 419-pair FiVE-Bench: rho = --blend_power in {2,3,4,6,8,10,20,50}, at a step count given as
# the first argument:
#   sbatch --array=0-7 slurm_scripts/five_bench/r45_sweep.sh 15                 # R36 analog
#   sbatch --array=0-7 slurm_scripts/five_bench/r45_sweep.sh $(python evaluation/r45_select.py sweepM)
#                                                       # R39 analog, 15 + draft N (20 or 30)
# Any N other than 15 must equal r45_method.json's matched_n (set at method-gate), so the
# compute-matched sweep cannot be rendered at the wrong N.
#
# ARRAY LAYOUT: 8 tasks, one per rho (task i -> RHOS[i]); each task loops ALL 6 edit types
# (100/100/100/100/9/10 = 419 pairs). r45_eval.sh / r45_fiveacc.sh index the method table of
# `r45_select.py eval --table`, whose sweep rows follow the same rho order.
#
# Sampler: run_fivebench.py --backbone ll (LongLive-native: tripled VP frame, LoRA, sink 3 +
# 12-frame local window) with fg_boost 2, flow_shift 1.0, seed 0. The SF sweeps (R36/R39)
# differ ONLY in backbone and its native fg_boost (4); everything else is identical.
#
# --time=24:00:00 (partition max) for every N: an #SBATCH line cannot depend on $1. SF took
# 11.5-12.8 h at N=15 and ~17.5-19 h at N=30 (R39); LL's per-step cost is measured by smoke
# G4 -- re-check this budget against it before launching N=30.
# RESUME: an edit type already complete (dirs + all-ok manifest) is skipped; run_fivebench.py
# has no per-clip skip, so a timed-out task re-renders the edit type it was in: resubmit just
# that index (sbatch --array=<i> ... <N>).
# A failed edit type does NOT stop the loop: the remaining types still render, and the task
# exits non-zero at the end so the failure is visible.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404).
#SBATCH --job-name=r45_sweep
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/r45_sweep_%A_%a.out
#SBATCH --error=logs/r45_sweep_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors"
RHOS=(2 3 4 6 8 10 20 50)
EXPECTED=(100 100 100 100 9 10)
TOTAL_EXPECTED=419

STEPS=$1
case "$STEPS" in
  15|20|30) ;;
  *) echo "[r45_sweep] first argument must be the step count 15, 20 or 30 (got '${STEPS}')"; exit 1 ;;
esac
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
  echo "[r45_sweep] bad array id '${TID}' (expected 0-7)"; exit 1
fi
R=${RHOS[$TID]}
METHOD="ll_rho${R}_n${STEPS}_vp"
OUT_ROOT="$FIVE_ROOT/r45_sweep/n${STEPS}"

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

if [ "$STEPS" -ne 15 ]; then
  MATCHED=$(python evaluation/r45_select.py sweepM) || exit 1
  if [ "$STEPS" -ne "$MATCHED" ]; then
    echo "[r45_sweep] N=${STEPS} but r45_method.json's matched_n is ${MATCHED}; refusing."; exit 1
  fi
fi

mkdir -p logs "$OUT_ROOT"
echo "[r45_sweep] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r45_sweep] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} steps=${STEPS} rho=${R} expect=${TOTAL_EXPECTED}"

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r45_sweep] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"
  # Skip an edit type an earlier run already finished. Complete = EXP real clip dirs AND a
  # manifest with exactly EXP rows, all ok. Anything less is re-rendered whole.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r45_sweep] SKIP ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
    continue
  fi

  python evaluation/run_fivebench.py --backbone ll \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --blend_power "$R" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step "$STEPS" --fg_boost_factor 2 --flow_shift 1.0 --seed 0 \
    || { echo "[r45_sweep] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r45_sweep] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r45_sweep] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
    FAILED=$((FAILED + 1))
  fi

  # A per-pair error is recorded in the manifest rather than raised, so a clean exit code
  # is not evidence the edit type is complete.
  if [ -f "$MAN" ]; then
    N_BAD=$(awk -F, 'NR>1 && $3!="ok"' "$MAN" | wc -l)
  else
    N_BAD=-1
  fi
  echo "[r45_sweep] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r45_sweep] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r45_sweep] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
echo "[r45_sweep] failures: ${FAILED}"
exit $(( FAILED > 0 ))
