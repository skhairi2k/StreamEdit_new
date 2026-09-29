#!/bin/bash
# R36 infer -- StreamEdit + visual prompting (paper §4.5, Eq. 4 scalar blend) over the
# WHOLE 419-pair FiVE-Bench, swept over R26's uniform-baseline grid
# rho = --blend_power in {2,3,4,6,8,10,20,50}. rho=2 is re-rendered, not reused from R7.
#
# ARRAY LAYOUT: 8 tasks, one per rho (task i -> RHOS[i]); each task loops ALL 6 edit
# types (100/100/100/100/9/10 = 419 pairs). 8 tasks stay well under the per-user
# QOSMaxSubmitJobPerUserLimit (27-32 jobs) that rejected the earlier 48-task layout.
#
# Sampler MUST stay byte-identical to r36_smoke.sh / R26 / R7: step 15, fg_boost 4,
# flow_shift 1.0, seed 0 (chunk 21 / overlap 1 / sink 0 are run_fivebench.py defaults).
# r36_smoke.sh job 1007548 proved this exact invocation reproduces R26's diagonal and R7
# sha256-identically; check-parity re-proves it on all 22 cases.json clips x 8 rho.
#
# --time=20:00:00 (partition max 24h): R35 stage1 job 1007492 renders the full bench at
# ~1.5 min/clip incl. model loads (37 clips in 54 min) -> ~10.5 h for 419 clips. Same
# limit as R21/R35 full-bench tasks.
# NO RESUME: run_fivebench.py re-renders every pair of an edit type, so a timed-out task
# must be resubmitted whole (sbatch --array=<i>).
# A failed edit type does NOT stop the loop: the remaining types still render, and the
# task exits non-zero at the end so the failure is visible.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404).
#SBATCH --job-name=r36_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/r36_infer_%A_%a.out
#SBATCH --error=logs/r36_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors"
OUT_ROOT="$FIVE_ROOT/r36_rho_sweep"

RHOS=(2 3 4 6 8 10 20 50)
EXPECTED=(100 100 100 100 9 10)
TOTAL_EXPECTED=419

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
  echo "[r36_infer] bad array id '${TID}' (expected 0-7)"; exit 1
fi
R=${RHOS[$TID]}
METHOD="r36_rho${R}_vp"

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r36_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r36_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} rho=${R} expect=${TOTAL_EXPECTED}"

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r36_infer] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished (the 2026-09-25 quota abort left
  # several complete). Complete = EXP real clip dirs AND a manifest with exactly EXP rows,
  # all ok. Anything less is re-rendered whole, overwriting clips cut off mid-write.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r36_infer] SKIP ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
    continue
  fi

  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --blend_power "$R" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --flow_shift 1.0 --seed 0 \
    || { echo "[r36_infer] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r36_infer] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r36_infer] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
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
  echo "[r36_infer] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r36_infer] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r36_infer] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
echo "[r36_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
