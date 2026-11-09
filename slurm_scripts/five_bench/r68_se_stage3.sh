#!/bin/bash
# R68 stage 3 (StreamEdit): the 15-step render of R52's B1 (se_max's demand, linear map rho = 2 + 48 q) on FiVE-Bench-Correct (415 pairs).
#
# Usage: sbatch slurm_scripts/five_bench/r68_se_stage3.sh        (r68_se_chain.sh, afterok on r68_field.sh se; no array, one arm)
# Input : $SC/r68_field/se_max_linrho/edit{T}/{video}.npz   (key `rho`, step 15, flow_shift 1.0 -- r31_stage3 validates both)
# Output: $SC/r68_se_arms/r68_max_linrho_full/step14/edit{T}/{video}/*.png   (+ _manifest_edit{T}.csv, _prompts_edit{T}.json)
#         $SC = lab /scratch/PERSO/skhairi/five_bench: read directly by the R57 evaluation; archived to /store by step store-copy.
# CPUs: 4 with --hint=nomultithread = 8 logical (user rule, 2026-11-09), not R62's 16.
# r62_se_stage3.sh unchanged apart from the field (B1's linear rho instead of an A1-A4 arm): VP, fg_boost 4, blend_power 2 -- inert under a rho
# field --, rollout_chunk_size 21, seed 0, and the Correct inputs: prompts + trigger words from evaluation/five_bench_correct (--edit_json_dir,
# absolute), cases from r43_anchor_manifest_correct.json (--cases_json), anchors from anchors_r43_correct. Total NFE = 5 (R59 draft) + 15 = 20.
# PROMPT GUARD: r31_stage3.py writes $OUT_ROOT/$METHOD/_prompts_edit{T}.json; r59_check_prompts.py compares them to the Correct JSONs at the
# end and a mismatch fails the job. NO-OVERWRITE GUARD: the output root must be $SC/r68_se_arms.
#SBATCH --job-name=r68_se_stage3
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r68_se_stage3_%j.out
#SBATCH --error=logs/r68_se_stage3_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench       # anchors only (a real directory, not a /store link)
SC=/scratch/PERSO/skhairi/five_bench       # field in, render out: Slurm jobs read and write lab /scratch only (R58 storage policy)
CASES=evaluation/r43_anchor_manifest_correct.json
EDIT_JSON_DIR=$REPO/evaluation/five_bench_correct
FIELD_DIR=$SC/r68_field/se_max_linrho
OUT_ROOT=$SC/r68_se_arms
ANCHOR_ROOT=$D/anchors_r43_correct
METHOD=r68_max_linrho_full
PY_SG=/home/ids/skhairi/anaconda3/envs/streamgve/bin/python
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

# no-overwrite guard: the output root is the new R68 folder, never an R49 / R51 / R52 / R59 / R62 render folder
[ "$OUT_ROOT" = "$SC/r68_se_arms" ] || { echo "[r68_se_stage3] FATAL: refusing output root ${OUT_ROOT} (only $SC/r68_se_arms)"; exit 1; }

cd "$REPO" || exit 1
mkdir -p logs "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r68_se_stage3] job=${SLURM_JOB_ID:-local} method=${METHOD} field=${FIELD_DIR} cases=${CASES} edit_json_dir=${EDIT_JSON_DIR} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r68_se_stage3] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the field must be complete
FAILED=0; N_RHO=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD_DIR/edit${T}"/*.npz 2>/dev/null | wc -l); N_RHO=$((N_RHO + N))
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r68_se_stage3] FATAL: edit${T} has ${N} rho npz, need ${EXPECT[$((T - 1))]} -- run r68_field.sh se first"; FAILED=$((FAILED + 1)); }
done
echo "[r68_se_stage3] ${N_RHO} rho npz under ${FIELD_DIR}"
[ "$FAILED" -ne 0 ] && exit 1

for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r68_se_stage3] ${METHOD} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage3.py --edit_type "$T" --method "$METHOD" --rho_dir "$FIELD_DIR" \
    --data_root "$DATA_ROOT" --edit_json_dir "$EDIT_JSON_DIR" --out_root "$OUT_ROOT" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --cases_json "$CASES" --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
    --rollout_chunk_size 21 --seed 0 \
    || { echo "[r68_se_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: exact per-type counts, manifests all ok, Correct prompts / trigger words / anchors
N_TOTAL=0; BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r68_se_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- MISMATCH (expect ${EXPECT[$((T - 1))]})"; BAD_TYPES=$((BAD_TYPES + 1)); }
done
echo "[r68_se_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect ${TOTAL})"
{ [ "$N_TOTAL" -ne "$TOTAL" ] || [ "$BAD_TYPES" -ne 0 ]; } && FAILED=$((FAILED + 1))
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r68_se_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED + 1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r68_se_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  [ "$N_BAD" -ne 0 ] && FAILED=$((FAILED + 1))
fi
$PY_FB evaluation/r59_check_prompts.py --sidecar "$OUT_ROOT/$METHOD/_prompts_edit*.json" --edit_json_dir "$EDIT_JSON_DIR" \
  --anchor_dir "$ANCHOR_ROOT" --expect_n "$TOTAL" || { echo "[r68_se_stage3] PROMPT-MISMATCH"; FAILED=$((FAILED + 1)); }
echo "[r68_se_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
