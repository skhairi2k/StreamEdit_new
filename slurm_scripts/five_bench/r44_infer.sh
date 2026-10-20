#!/bin/bash
# R44 infer -- TEXT-ONLY (novp) counterpart of R36's VP rho sweep: StreamEdit with the
# plain Eq. 4 scalar blend, rho = --blend_power in {2,3,4,6,8,10,20,50}, over the WHOLE
# 419-pair FiVE-Bench, NO visual prompt (no anchor). This is the uniform reference curve
# R44's text-only routing arms are scored against. rho=2 is re-rendered, not reused from
# R1 (old code, on /projects) -- same call as R36 made for R7.
#
# USAGE: sbatch --array=0-7 slurm_scripts/five_bench/r44_infer.sh [N]
#   N = 15 (default: the secondary reference curve), or 20 / 30 for the MANDATORY
#   equal-NFE pass (N = 15 + the chosen config's draft steps; the headline reference;
#   which N is chosen at plan step choose-matched).
# ARRAY LAYOUT: 8 tasks, task i -> RHOS[i]; each task loops ALL 6 edit types
# (100/100/100/100/9/10 = 419 pairs). r44_eval.sh / r44_fiveacc.sh `sweep N` use the
# same index map.
#
# Sampler identical to r44_smoke.sh's `uniform` call, which gate G1 compares against the
# routed render: --vp_mode novp, NO --first_frame_edit_dir, step N, fg_boost 4,
# flow_shift 1.0, rollout_chunk_size 21, seed 0. Everything else as r36_infer.sh.
#
# --time=24:00:00 (partition max): R36's N=15 tasks took 11.5-12.8 h for 419 clips (job
# 1007603); R39 estimates ~14 h at N=20 and ~17.5-19 h at N=30 -- one limit covers all N.
# RESUME: an edit type already complete (dirs + all-ok manifest) is skipped, but
# run_fivebench.py has no per-clip skip, so a timed-out task re-renders the edit type it
# was in: resubmit just that index (sbatch --array=<i> ... r44_infer.sh N).
# A failed edit type does NOT stop the loop: the remaining types still render, and the
# task exits non-zero at the end so the failure is visible.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404).
# --exclude: node52 exposes no GPU to batch jobs (R39 job 1016629; re-probed still broken
# 2026-10-08, job 1028156). node01/51/57 re-probed OK 2026-10-08 (node_probe.sh), re-allowed.
#SBATCH --job-name=r44_infer
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/r44_infer_%A_%a.out
#SBATCH --error=logs/r44_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
OUT_ROOT="$FIVE_ROOT/r44_rho_sweep"

RHOS=(2 3 4 6 8 10 20 50)
EXPECTED=(100 100 100 100 9 10)
TOTAL_EXPECTED=419

STEPS=${1:-15}
case "$STEPS" in
  15|20|30) ;;
  *) echo "[r44_infer] bad N '${STEPS}' (expected 15, 20 or 30)"; exit 1 ;;
esac

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
  echo "[r44_infer] bad array id '${TID}' (expected 0-7)"; exit 1
fi
R=${RHOS[$TID]}
METHOD="r44_n${STEPS}_rho${R}_novp"

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r44_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r44_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} steps=${STEPS} rho=${R} expect=${TOTAL_EXPECTED}"
# (no "vp_mode=" in this echo: wait-sweep greps for run_fivebench.py's own vp_mode=novp line)

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r44_infer] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished. Complete = EXP real clip dirs AND a
  # manifest with exactly EXP rows, all ok. Anything less is re-rendered whole, overwriting
  # clips cut off mid-write.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r44_infer] SKIP ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
    continue
  fi

  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode novp \
    --blend_power "$R" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step "$STEPS" --fg_boost_factor 4 --flow_shift 1.0 --rollout_chunk_size 21 --seed 0 \
    || { echo "[r44_infer] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r44_infer] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r44_infer] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
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
  echo "[r44_infer] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r44_infer] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r44_infer] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
echo "[r44_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
