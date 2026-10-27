#!/bin/bash
# R61 sweep -- StreamEdit on LONGLIVE (VP) rho sweep on FiVE-Bench-Correct: rho = --blend_power in
# {2,3,4,6,8,10,20,50}, at a step count given as the first argument (15 or 20):
#   sbatch --array=0-7 slurm_scripts/five_bench/r61_sweep.sh 15   # native + N=15 reference curve
#   sbatch --array=0-7 slurm_scripts/five_bench/r61_sweep.sh 20   # equal-NFE (5 draft + 15 render) reference curve
#
# ARRAY LAYOUT: 8 tasks, one per rho (task i -> RHOS[i]); each task loops ALL 6 edit types
# (98/100/99/99/9/10 = 415 pairs, counts read from the Correct JSONs). Index 0 (rho=2) is the
# native baseline at N=15 and the equal-NFE baseline at N=20.
#
# Sampler: run_fivebench.py --backbone ll (LongLive-native: tripled VP frame, LoRA, sink 3 + 12-frame
# local window) with fg_boost 2, flow_shift 1.0, seed 0, same inputs as se_max_ll (r61_ll_stage1.sh /
# r61_ll_stage3.sh): prompts + trigger words from evaluation/five_bench_correct (--edit_json_dir,
# absolute) and anchors from anchors_r43_correct. Every anchor changed, so nothing from R45 is reused.
# PROMPT GUARD: run_fivebench.py writes edit{T}/_prompts.json (the strings the pipeline receives);
# r59_check_prompts.py compares them to the Correct JSONs after every edit type (also for a SKIPPED,
# already-complete type) and a mismatch counts as a failure.
# RESUME: an edit type already complete (dirs + all-ok manifest) is skipped; run_fivebench.py has no
# per-clip skip, so a timed-out task re-renders the edit type it was in: resubmit just that index
# (sbatch --array=<i> ... <N>). A failed edit type does NOT stop the loop; the task exits non-zero
# at the end so the failure is visible.
# --time=24:00:00 (partition max) for every N -- an #SBATCH line cannot depend on $1. R45's LongLive
# sweep took 11.5-12.8 h at N=15 (R39 SF analog); re-check this budget against r61_smoke's G2 timing.
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint (R9 job 900404).
#SBATCH --job-name=r61_sweep
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/r61_sweep_%A_%a.out
#SBATCH --error=logs/r61_sweep_%A_%a.err
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors_r43_correct"
EDIT_JSON_DIR="$REPO/evaluation/five_bench_correct"
RHOS=(2 3 4 6 8 10 20 50)
STEPS=$1
case "$STEPS" in
  15|20) ;;
  *) echo "[r61_sweep] first argument must be the step count 15 or 20 (got '${STEPS}')"; exit 1 ;;
esac
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
  echo "[r61_sweep] bad array id '${TID}' (expected 0-7)"; exit 1
fi
R=${RHOS[$TID]}
METHOD="r61_ll_rho${R}_n${STEPS}"
OUT_ROOT="$FIVE_ROOT/r61_sweep/n${STEPS}"

cd "$REPO" || exit 1
# freeze check: the Correct benchmark files must not have changed under us
if ! git diff --quiet HEAD -- evaluation/five_bench_correct evaluation/r43_anchor_manifest_correct.json 2>/dev/null; then
  echo "[r61_sweep] REFUSE: evaluation/five_bench_correct or r43_anchor_manifest_correct.json has uncommitted changes -- commit them first"
  exit 1
fi
echo "[r61_sweep] commit=$(git rev-parse HEAD 2>/dev/null)"
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

# expected clips per edit type, from the Correct JSONs themselves
read -r -a EXPECTED <<< "$(python -I - "$EDIT_JSON_DIR" <<'PYEOF'
import json, sys
print(" ".join(str(len(json.load(open(f"{sys.argv[1]}/edit{t}_FiVE.json")))) for t in range(1, 7)))
PYEOF
)"
TOTAL_EXPECTED=0; for n in "${EXPECTED[@]}"; do TOTAL_EXPECTED=$((TOTAL_EXPECTED + n)); done
if [ "${#EXPECTED[@]}" -ne 6 ] || [ "$TOTAL_EXPECTED" -ne 415 ]; then
  echo "[r61_sweep] FATAL: Correct JSON counts '${EXPECTED[*]}' (total ${TOTAL_EXPECTED}), expected 6 types / 415 pairs"; exit 1
fi

echo "[r61_sweep] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r61_sweep] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} steps=${STEPS} rho=${R} expect=${EXPECTED[*]} (${TOTAL_EXPECTED}) edit_json_dir=${EDIT_JSON_DIR} anchors=${ANCHOR_ROOT}"

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r61_sweep] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished: EXP real clip dirs AND a manifest with exactly EXP rows, all ok.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r61_sweep] SKIP render ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
  else
    python evaluation/run_fivebench.py --backbone ll \
      --edit_type "$T" --method "$METHOD" \
      --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
      --blend_power "$R" \
      --data_root "$DATA_ROOT" --edit_json_dir "$EDIT_JSON_DIR" --out_root "$OUT_ROOT" \
      --step "$STEPS" --fg_boost_factor 2 --flow_shift 1.0 --seed 0 \
      || { echo "[r61_sweep] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }
  fi

  # prompts / trigger words / anchors actually used must be the Correct ones (also for a skipped type)
  python evaluation/r59_check_prompts.py --sidecar "$OUT_ROOT/$METHOD/edit${T}/_prompts.json" \
    --edit_json_dir "$EDIT_JSON_DIR" --anchor_dir "$ANCHOR_ROOT" --expect_n "$EXP" \
    || { echo "[r61_sweep] PROMPT-MISMATCH ${METHOD} edit${T}"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r61_sweep] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r61_sweep] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
    FAILED=$((FAILED + 1))
  fi

  # A per-pair error is recorded in the manifest rather than raised, so a clean exit code is not evidence the type is complete.
  if [ -f "$MAN" ]; then
    N_BAD=$(awk -F, 'NR>1 && $3!="ok"' "$MAN" | wc -l)
  else
    N_BAD=-1
  fi
  echo "[r61_sweep] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r61_sweep] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r61_sweep] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
python evaluation/r59_check_prompts.py --sidecar "$OUT_ROOT/$METHOD/edit*/_prompts.json" \
  --edit_json_dir "$EDIT_JSON_DIR" --anchor_dir "$ANCHOR_ROOT" --expect_n "$TOTAL_EXPECTED" \
  || { echo "[r61_sweep] PROMPT-MISMATCH ${METHOD} (all types)"; FAILED=$((FAILED + 1)); }
echo "[r61_sweep] failures: ${FAILED}"
exit $(( FAILED > 0 ))
