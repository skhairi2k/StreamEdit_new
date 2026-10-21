#!/bin/bash
# R49 stage 3 (StreamEdit): the adaptive 15-step render with the inverse-response rho field, arm `max`.
#
# Usage: sbatch slurm_scripts/five_bench/r49_se_stage3.sh [subset]   18 clips of evaluation/r49_cases.json (r49_chain.sh, afterok on r49_field.sh)
#        sbatch slurm_scripts/five_bench/r49_se_stage3.sh full       419 pairs of evaluation/r7_anchor_manifest.json (R49 se-full-render)
# Input : subset $D/r49_field/se_max/edit{T}/{video}.npz, full $D/r49_field_full/se_max/...   (key `rho`, step 15, flow_shift 1.0 --
#         r31_stage3 validates both)
# Output: $D/r49_se_arms/r49_max_{subset,full}/step14/edit{T}/{video}/*.png   (+ _manifest_edit{T}.csv)
# Full mode passes no --cases_json: r31_stage3 renders the full bench from run_fivebench's own edit{T}_FiVE.json, as R38 / r48_se_stage3.sh.
# Flags identical to r48_se_stage3.sh / r38_stage3.sh (VP + Qwen anchors, fg_boost 4, blend_power 2 -- inert under a rho
# field --, rollout_chunk_size 21, seed 0). Total NFE = 5 (R49 draft) + 15 (this render) = 20.
#SBATCH --job-name=r49_se_stage3
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_se_stage3_%j.out
#SBATCH --error=logs/r49_se_stage3_%j.err

MODE=${1:-subset}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
case "$MODE" in
  subset) CASES=evaluation/r49_cases.json;          CASES_ARG=(--cases_json "$CASES"); FIELD_DIR=$D/r49_field/se_max ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; CASES_ARG=();                      FIELD_DIR=$D/r49_field_full/se_max ;;
  *) echo "[r49_se_stage3] usage: r49_se_stage3.sh [subset|full]"; exit 1 ;;
esac
OUT_ROOT=$D/r49_se_arms
ANCHOR_ROOT=$D/anchors
PY_SG=/home/ids/skhairi/anaconda3/envs/streamgve/bin/python
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
METHOD="r49_max_${MODE}"

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
echo "[r49_se_stage3] job=${SLURM_JOB_ID:-local} mode=${MODE} method=${METHOD} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r49_se_stage3] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the field must be complete
FAILED=0; N_RHO=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD_DIR/edit${T}"/*.npz 2>/dev/null | wc -l); N_RHO=$((N_RHO + N))
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r49_se_stage3] FATAL: edit${T} has ${N} rho npz, need ${EXPECT[$((T - 1))]} -- run r49_field.sh first"; FAILED=$((FAILED + 1)); }
done
echo "[r49_se_stage3] ${N_RHO} rho npz under ${FIELD_DIR}"
[ "$FAILED" -ne 0 ] && exit 1

for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r49_se_stage3] ${METHOD} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage3.py --edit_type "$T" --method "$METHOD" --rho_dir "$FIELD_DIR" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    "${CASES_ARG[@]}" --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
    --rollout_chunk_size 21 --seed 0 \
    || { echo "[r49_se_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: exact per-type counts, manifests all ok
N_TOTAL=0; BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r49_se_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- MISMATCH (expect ${EXPECT[$((T - 1))]})"; BAD_TYPES=$((BAD_TYPES + 1)); }
done
echo "[r49_se_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect ${TOTAL})"
{ [ "$N_TOTAL" -ne "$TOTAL" ] || [ "$BAD_TYPES" -ne 0 ]; } && FAILED=$((FAILED + 1))
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r49_se_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED + 1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r49_se_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  [ "$N_BAD" -ne 0 ] && FAILED=$((FAILED + 1))
fi
echo "[r49_se_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
