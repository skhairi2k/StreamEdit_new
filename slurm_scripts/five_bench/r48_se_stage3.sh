#!/bin/bash
# R48 stage 3 (StreamEdit): the adaptive 15-step render, one arm per array task.
#
# Usage: sbatch slurm_scripts/five_bench/r48_se_stage3.sh subset                  # 4 arms x 18 clips   (--array 0-3 default)
#        sbatch --array=0-$((N-1)) slurm_scripts/five_bench/r48_se_stage3.sh full # the N `se` arms listed in
#                                                                                  evaluation/r48_selected_arms.txt, 419 pairs
# Arms (r48_field.py): dino_SD, lpips_SD, dino_D (no support), S (no divergence). r48_selected_arms.txt has one
# "<host> <arm>" per line (host = se|wan); only the `se` lines are used here, in file order.
#
# Input : $D/r48_field/se_{arm}/edit{T}/{video}.npz   (key `rho`, step 15, flow_shift 1.0 -- r31_stage3 validates both)
# Output: $D/r48_se_arms/r48_{arm}_{MODE}/step14/edit{T}/{video}/*.png   (+ _manifest_edit{T}.csv)
# Flags identical to r38_stage3.sh (VP + Qwen anchors, fg_boost 4, blend_power 2 -- inert under a rho field --,
# rollout_chunk_size 21, seed 0). Total NFE = 5 (draft, r48_se_stage1) + 15 (this render) = 20.
#SBATCH --job-name=r48_se_stage3
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_se_stage3_%A_%a.out
#SBATCH --error=logs/r48_se_stage3_%A_%a.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json; CASES_ARG=(--cases_json "$CASES") ;;
  full)   CASES=evaluation/r7_anchor_manifest.json; CASES_ARG=() ;;   # full bench via run_fivebench's own edit{T}_FiVE.json, as R38
  *) echo "[r48_se_stage3] usage: r48_se_stage3.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
FIELD_ROOT=$D/r48_field
OUT_ROOT=$D/r48_se_arms
ANCHOR_ROOT=$D/anchors
PY_SG=~/anaconda3/envs/streamgve/bin/python
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$MODE" = subset ]; then
  ARMS=(dino_SD lpips_SD dino_D S)
else
  [ -s evaluation/r48_selected_arms.txt ] || { echo "[r48_se_stage3] FATAL: evaluation/r48_selected_arms.txt missing/empty -- write it at gate-decision"; exit 1; }
  mapfile -t ARMS < <(awk '$1=="se"{print $2}' evaluation/r48_selected_arms.txt)
  [ "${#ARMS[@]}" -gt 0 ] || { echo "[r48_se_stage3] no se arms selected -- nothing to render"; exit 0; }
fi
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ]; then echo "[r48_se_stage3] bad array id '${TID}'"; exit 1; fi
if [ "$TID" -ge "${#ARMS[@]}" ]; then echo "[r48_se_stage3] array id ${TID} >= ${#ARMS[@]} arms -- nothing to do"; exit 0; fi
ARM=${ARMS[$TID]}
case "$ARM" in dino_SD|lpips_SD|dino_D|S) ;; *) echo "[r48_se_stage3] unknown arm '${ARM}'"; exit 1 ;; esac
METHOD="r48_${ARM}_${MODE}"

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r48_se_stage3] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} arm=${ARM} method=${METHOD} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r48_se_stage3] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the field must be complete for this arm
FAILED=0; N_RHO=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD_ROOT/se_${ARM}/edit${T}"/*.npz 2>/dev/null | wc -l); N_RHO=$((N_RHO + N))
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r48_se_stage3] FATAL: edit${T} has ${N} rho npz, need ${EXPECT[$((T - 1))]} -- run the field step first"; FAILED=$((FAILED + 1)); }
done
echo "[r48_se_stage3] ${ARM}: ${N_RHO} rho npz under ${FIELD_ROOT}/se_${ARM}"
[ "$FAILED" -ne 0 ] && exit 1

for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r48_se_stage3] ${METHOD} edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage3.py --edit_type "$T" --method "$METHOD" --rho_dir "$FIELD_ROOT/se_${ARM}" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    "${CASES_ARG[@]}" --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
    --rollout_chunk_size 21 --seed 0 \
    || { echo "[r48_se_stage3] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: exact per-type counts, manifests all ok
N_TOTAL=0; BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  N=$(ls -d "$OUT_ROOT/$METHOD/step14/edit${T}"/*/ 2>/dev/null | wc -l); N_TOTAL=$((N_TOTAL + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r48_se_stage3] ${METHOD} edit${T}: ${N} step14 dirs -- MISMATCH (expect ${EXPECT[$((T - 1))]})"; BAD_TYPES=$((BAD_TYPES + 1)); }
done
echo "[r48_se_stage3] ${METHOD}: ${N_TOTAL} step14 dirs total (expect ${TOTAL})"
[ "$N_TOTAL" -ne "$TOTAL" ] || [ "$BAD_TYPES" -ne 0 ] && FAILED=$((FAILED + 1))
MANIFESTS=("$OUT_ROOT/$METHOD"/_manifest_edit*.csv)
if [ ! -e "${MANIFESTS[0]}" ]; then
  echo "[r48_se_stage3] ${METHOD}: no manifest CSVs found"; FAILED=$((FAILED + 1))
else
  N_BAD=$(awk -F, 'FNR==1{next} $3!="ok"' "${MANIFESTS[@]}" | wc -l)
  echo "[r48_se_stage3] ${METHOD}: ${#MANIFESTS[@]} manifests, non-ok rows: ${N_BAD}"
  [ "$N_BAD" -ne 0 ] && FAILED=$((FAILED + 1))
fi
echo "[r48_se_stage3] failures: ${FAILED}"
exit $(( FAILED > 0 ))
