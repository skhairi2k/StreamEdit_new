#!/bin/bash
# R59 render-baselines -- StreamEdit (Self-Forcing) + visual prompting, uniform rho, on the 415 FiVE-Bench-Correct pairs.
#
# ARRAY LAYOUT: 9 tasks, one per arm (each loops ALL 6 edit types: 98/100/99/99/9/10 = 415 pairs, counts read from the JSONs):
#   task 0    -> native baseline       N = 15, rho = 2     (method r59_n15_rho2_vp)
#   tasks 1-8 -> equal-NFE curve       N = 20, rho in {2,3,4,6,8,10,20,50}  (method r59_n20_rho{R}_vp; N=20 = se_max's 5 + 15 steps;
#                                      rho = 2 is the equal-NFE baseline, the 8-rho curve is R39's, re-rendered on Correct)
# Sampler byte-identical to r39_infer.sh apart from the inputs: fg_boost 4, flow_shift 1.0, seed 0 (chunk 21 / overlap 1 / sink 0 are
# run_fivebench.py defaults). INPUTS differ: prompts + trigger words from evaluation/five_bench_correct (--edit_json_dir, absolute: the
# runner resolves it after its chdir) and anchors from anchors_r43_correct. Every anchor changed, so nothing from R39 is reused.
# PROMPT GUARD: run_fivebench.py writes edit{T}/_prompts.json (the strings the pipeline receives); r59_check_prompts.py compares them to
# the Correct JSONs after every edit type (also for a SKIPPED, already complete type) and a mismatch counts as a failure.
# RESUME: an edit type already complete (dirs + all-ok manifest) is skipped, but run_fivebench.py has no per-clip skip, so a timed-out task
# re-renders the edit type it was in: resubmit just that index (sbatch --array=<i>).
# A failed edit type does NOT stop the loop; the task exits non-zero at the end. --mem=64G: the partition default host-OOMs loading
# UMT5-XXL + the checkpoint (R9 job 900404). --exclude=node52 only: it advertises gpu:8 but exposes no device to batch jobs.
# Expected time on L40S: ~12 h at N=15, ~14 h at N=20 (R36 / R39), under the 24 h partition max.
#SBATCH --job-name=r59_infer
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/r59_infer_%A_%a.out
#SBATCH --error=logs/r59_infer_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT and would abort the job at `conda activate`.

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT="$FIVE_ROOT/anchors_r43_correct"
EDIT_JSON_DIR="$REPO/evaluation/five_bench_correct"
OUT_ROOT="$FIVE_ROOT/r59_rho_sweep"

RHOS=(2 3 4 6 8 10 20 50)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 8 ]; then
  echo "[r59_infer] bad array id '${TID}' (expected 0-8)"; exit 1
fi
if [ "$TID" -eq 0 ]; then STEPS=15; R=2; else STEPS=20; R=${RHOS[$((TID - 1))]}; fi
METHOD="r59_n${STEPS}_rho${R}_vp"

cd "$REPO" || exit 1
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
  echo "[r59_infer] FATAL: Correct JSON counts '${EXPECTED[*]}' (total ${TOTAL_EXPECTED}), expected 6 types / 415 pairs"; exit 1
fi

echo "[r59_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r59_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} steps=${STEPS} rho=${R} expect=${EXPECTED[*]} (${TOTAL_EXPECTED}) edit_json_dir=${EDIT_JSON_DIR} anchors=${ANCHOR_ROOT}"

FAILED=0
for T in 1 2 3 4 5 6; do
  EXP=${EXPECTED[$((T - 1))]}
  echo "--- [r59_infer] ${METHOD} edit${T} (expect ${EXP}) $(date '+%F %T') ---"

  # Skip an edit type an earlier run already finished: EXP real clip dirs AND a manifest with exactly EXP rows, all ok.
  MAN="$OUT_ROOT/$METHOD/edit${T}/_manifest.csv"
  N_PRE=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  N_OK=$( [ -f "$MAN" ] && awk -F, 'NR>1 && $3=="ok"' "$MAN" | wc -l || echo 0)
  N_ROWS=$( [ -f "$MAN" ] && awk 'NR>1' "$MAN" | wc -l || echo 0)
  if [ "$N_PRE" -eq "$EXP" ] && [ "$N_ROWS" -eq "$EXP" ] && [ "$N_OK" -eq "$EXP" ]; then
    echo "[r59_infer] SKIP render ${METHOD} edit${T}: already complete (${N_PRE} dirs, ${N_OK}/${EXP} ok)"
  else
    python evaluation/run_fivebench.py \
      --edit_type "$T" --method "$METHOD" \
      --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
      --blend_power "$R" \
      --data_root "$DATA_ROOT" --edit_json_dir "$EDIT_JSON_DIR" --out_root "$OUT_ROOT" \
      --step "$STEPS" --fg_boost_factor 4 --flow_shift 1.0 --seed 0 \
      || { echo "[r59_infer] FAILED ${METHOD} edit${T}: run_fivebench.py exit $?"; FAILED=$((FAILED + 1)); }
  fi

  # prompts / trigger words / anchors actually used must be the Correct ones (also for a skipped type)
  python evaluation/r59_check_prompts.py --sidecar "$OUT_ROOT/$METHOD/edit${T}/_prompts.json" \
    --edit_json_dir "$EDIT_JSON_DIR" --anchor_dir "$ANCHOR_ROOT" --expect_n "$EXP" \
    || { echo "[r59_infer] PROMPT-MISMATCH ${METHOD} edit${T}"; FAILED=$((FAILED + 1)); }

  # Real frame dirs only -- evaluate.py leaves *_resize siblings behind on later runs.
  N_DIRS=$(ls -d "$OUT_ROOT/$METHOD/edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r59_infer] ${METHOD} edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
  if [ "$N_DIRS" -ne "$EXP" ]; then
    echo "[r59_infer] COUNT-MISMATCH ${METHOD} edit${T}: ${N_DIRS} != ${EXP}"
    FAILED=$((FAILED + 1))
  fi

  # A per-pair error is recorded in the manifest rather than raised, so a clean exit code is not evidence the type is complete.
  if [ -f "$MAN" ]; then
    N_BAD=$(awk -F, 'NR>1 && $3!="ok"' "$MAN" | wc -l)
  else
    N_BAD=-1
  fi
  echo "[r59_infer] ${METHOD} edit${T} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ]; then
    echo "[r59_infer] MANIFEST-ERRORS ${METHOD} edit${T} (-1 = manifest missing)"
    FAILED=$((FAILED + 1))
  fi
done

N_TOTAL=$(ls -d "$OUT_ROOT/$METHOD"/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r59_infer] ${METHOD} total frame dirs: ${N_TOTAL} (expected ${TOTAL_EXPECTED}) $(date '+%F %T')"
python evaluation/r59_check_prompts.py --sidecar "$OUT_ROOT/$METHOD/edit*/_prompts.json" \
  --edit_json_dir "$EDIT_JSON_DIR" --anchor_dir "$ANCHOR_ROOT" --expect_n "$TOTAL_EXPECTED" \
  || { echo "[r59_infer] PROMPT-MISMATCH ${METHOD} (all types)"; FAILED=$((FAILED + 1)); }
echo "[r59_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
