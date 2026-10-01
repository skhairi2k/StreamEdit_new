#!/bin/bash
# R39 infer -- R36's StreamEdit + visual prompting rho sweep (paper §4.5, Eq. 4 scalar
# blend, rho = --blend_power in {2,3,4,6,8,10,20,50}) over the WHOLE 419-pair FiVE-Bench,
# re-rendered in a single pass at a compute-matched step count:
#   N=20 = R38's 15+5 routing arms (5-step draft + 15-step render),
#   N=30 = R35's 15+15 routing arms.
#
# ARRAY LAYOUT: 16 tasks, one per (N, rho) arm: task i -> N = NS[i/8], rho = RHOS[i%8]
# (0-7 = N=20, 8-15 = N=30); each task loops ALL 6 edit types (100/100/100/100/9/10 =
# 419 pairs). r39_eval.sh / r39_fiveacc.sh use the same index map.
#
# Sampler MUST stay byte-identical to r36_infer.sh apart from --step: fg_boost 4,
# flow_shift 1.0, seed 0 (chunk 21 / overlap 1 / sink 0 are run_fivebench.py defaults).
# r39_smoke.sh job 1016517 proved this invocation at --step 15 reproduces R36 r36_rho2_vp
# sha256-identically, and that --step 20 / 30 change every frame.
#
# --time=24:00:00 (partition max). R36's N=15 tasks took 11.5-12.8 h for 419 clips
# (job 1007603); the smoke adds ~2-4 s per extra step per clip on a fixed per-clip cost,
# so ~14 h at N=20 and ~17.5-19 h at N=30 -- not the ~24 h a linear N/15 scaling predicts.
# RESUME: an edit type already complete (dirs + all-ok manifest) is skipped, but
# run_fivebench.py has no per-clip skip, so a timed-out task re-renders the edit type it
# was in (<= ~5 h at N=30): resubmit just that index (sbatch --array=<i>).
# A failed edit type does NOT stop the loop: the remaining types still render, and the
# task exits non-zero at the end so the failure is visible.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404).
#SBATCH --job-name=r39_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/r39_infer_%A_%a.out
#SBATCH --error=logs/r39_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors"
OUT_ROOT="$FIVE_ROOT/r39_rho_sweep"

RHOS=(2 3 4 6 8 10 20 50)
NS=(20 30)
EXPECTED=(100 100 100 100 9 10)
TOTAL_EXPECTED=419

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 15 ]; then
  echo "[r39_infer] bad array id '${TID}' (expected 0-15)"; exit 1
fi
STEPS=${NS[$((TID / 8))]}
R=${RHOS[$((TID % 8))]}
METHOD="r39_n${STEPS}_rho${R}_vp"

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

echo "[r39_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r39_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} steps=${STEPS} rho=${R} expect=${TOTAL_EXPECTED}"

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r39_infer] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished. Complete = EXP real clip dirs AND a
  # manifest with exactly EXP rows, all ok. Anything less is re-rendered whole, overwriting
  # clips cut off mid-write.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r39_infer] SKIP ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
    continue
  fi

  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --blend_power "$R" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step "$STEPS" --fg_boost_factor 4 --flow_shift 1.0 --seed 0 \
    || { echo "[r39_infer] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r39_infer] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r39_infer] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
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
  echo "[r39_infer] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r39_infer] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r39_infer] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
echo "[r39_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
