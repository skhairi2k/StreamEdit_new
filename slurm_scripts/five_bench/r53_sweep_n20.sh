#!/bin/bash
# R53 sweep -- the TEXT-ONLY (novp) uniform-rho reference at N = 20 (= se_max_novp's 5 + 15 NFE) for the four rho values R44's N = 20
# sweep lacks: rho in {3, 6, 8, 20}. Together with R44's rho in {2, 4, 10, 50} (r44_n20_rho{R}_novp) it gives the equal-NFE text-only curve
# the same 8 points as R39's VP curve. Over the WHOLE 419-pair FiVE-Bench, NO visual prompt (no anchor).
#
# USAGE: sbatch slurm_scripts/five_bench/r53_sweep_n20.sh        (--array=0-3 is in the header; r53_se_chain.sh)
# ARRAY LAYOUT: 4 tasks, task i -> RHOS[i] = (3 6 8 20); each task loops ALL 6 edit types (100/100/100/100/9/10 = 419 pairs).
#
# Sampler identical to r44_infer.sh (a copy of it fixed at N = 20): --vp_mode novp, NO --first_frame_edit_dir, step 20, fg_boost 4,
# flow_shift 1.0, rollout_chunk_size 21, seed 0. Only the rho list, N, the method name and the output root differ.
# Output: $FIVE_ROOT/r53_rho_sweep/r53_n20_rho{R}_novp/edit{T}/{video}/*.png  (on /home, not /store)
#
# --time=24:00:00 (partition max): R39 estimates ~14 h at N = 20. RESUME: an edit type already complete (dirs + all-ok manifest) is skipped,
# but run_fivebench.py has no per-clip skip, so a timed-out task re-renders the edit type it was in: resubmit just that index
# (sbatch --array=<i> slurm_scripts/five_bench/r53_sweep_n20.sh). A failed edit type does NOT stop the loop; the task exits non-zero at the end.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404). --exclude=node52: exposes no GPU to batch jobs.
#SBATCH --job-name=r53_sweep
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r53_sweep_%A_%a.out
#SBATCH --error=logs/r53_sweep_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
OUT_ROOT="$FIVE_ROOT/r53_rho_sweep"

RHOS=(3 6 8 20)
EXPECTED=(100 100 100 100 9 10)
TOTAL_EXPECTED=419

STEPS=20

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 3 ]; then
  echo "[r53_sweep] bad array id '${TID}' (expected 0-3)"; exit 1
fi
R=${RHOS[$TID]}
METHOD="r53_n${STEPS}_rho${R}_novp"

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r53_sweep] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r53_sweep] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} steps=${STEPS} rho=${R} expect=${TOTAL_EXPECTED}"
# (no "vp_mode=" in this echo: wait-sweep greps for run_fivebench.py's own vp_mode=novp line)

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r53_sweep] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished. Complete = EXP real clip dirs AND a
  # manifest with exactly EXP rows, all ok. Anything less is re-rendered whole, overwriting
  # clips cut off mid-write.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r53_sweep] SKIP ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
    continue
  fi

  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode novp \
    --blend_power "$R" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step "$STEPS" --fg_boost_factor 4 --flow_shift 1.0 --rollout_chunk_size 21 --seed 0 \
    || { echo "[r53_sweep] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r53_sweep] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r53_sweep] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
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
  echo "[r53_sweep] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r53_sweep] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r53_sweep] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
echo "[r53_sweep] failures: ${FAILED}"
exit $(( FAILED > 0 ))
