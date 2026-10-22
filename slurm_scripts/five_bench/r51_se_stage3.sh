#!/bin/bash
# R51 stage 3 (StreamEdit): the 15-step render of the demand-component ablation arms on the 419 pairs, one arm per array task.
#
# Usage: sbatch --array=0-3 slurm_scripts/five_bench/r51_se_stage3.sh      (r51_se_chain.sh, afterok on r51_field.sh se)
#   task 0 nodiv (A1)   1 dino (A2)   2 lpips (A3)   3 nosupp (A4)
# Input : $D/r51_field_full/se_{arm}/edit{T}/{video}.npz   (key `rho`, step 15, flow_shift 1.0 -- r31_stage3 validates both)
# Output: $D/r51_se_arms/r51_{arm}_full/step14/edit{T}/{video}/*.png   (+ _manifest_edit{T}.csv)
# Flags identical to r49_se_stage3.sh full (VP + Qwen anchors, fg_boost 4, blend_power 2 -- inert under a rho field --,
# rollout_chunk_size 21, seed 0; no --cases_json: the full bench of run_fivebench's own edit{T}_FiVE.json). NFE 5 + 15 = 20.
#SBATCH --job-name=r51_se_stage3
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r51_se_stage3_%A_%a.out
#SBATCH --error=logs/r51_se_stage3_%A_%a.err

ARMS=(nodiv dino lpips nosupp)
TID=${SLURM_ARRAY_TASK_ID:-}
[[ "$TID" =~ ^[0-3]$ ]] || { echo "[r51_se_stage3] bad array id '${TID}' (expected 0-3)"; exit 1; }
ARM=${ARMS[$TID]}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r7_anchor_manifest.json
FIELD_DIR=$D/r51_field_full/se_${ARM}
OUT_ROOT=$D/r51_se_arms
ANCHOR_ROOT=$D/anchors
PY_SG=/home/ids/skhairi/anaconda3/envs/streamgve/bin/python
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
METHOD="r51_${ARM}_full"

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
echo "[r51_se_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${ARM} method=${METHOD} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r51_se_stage3] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the field must be complete
FAILED=0; N_RHO=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD_DIR/edit${T}"/*.npz 2>/dev/null | wc -l); N_RHO=$((N_RHO + N))
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r51_se_stage3] FATAL: edit${T} has ${N} rho npz, need ${EXPECT[$((T - 1))]} -- run r51_field.sh se first"; FAILED=$((FAILED + 1)); }
done
echo "[r51_se_stage3] ${N_RHO} rho npz under ${FIELD_DIR}"
[ "$FAILED" -ne 0 ] && exit 1

for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r51_se_stage3] ${METHOD} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage3.py --edit_type "$T" --method "$METHOD" --rho_dir "$FIELD_DIR" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
    --rollout_chunk_size 21 --seed 0 \
    || { echo "[r51_se_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: exact per-type counts, manifests all ok
N_TOTAL=0; BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r51_se_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- MISMATCH (expect ${EXPECT[$((T - 1))]})"; BAD_TYPES=$((BAD_TYPES + 1)); }
done
echo "[r51_se_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect ${TOTAL})"
{ [ "$N_TOTAL" -ne "$TOTAL" ] || [ "$BAD_TYPES" -ne 0 ]; } && FAILED=$((FAILED + 1))
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r51_se_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED + 1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r51_se_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  [ "$N_BAD" -ne 0 ] && FAILED=$((FAILED + 1))
fi
echo "[r51_se_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
