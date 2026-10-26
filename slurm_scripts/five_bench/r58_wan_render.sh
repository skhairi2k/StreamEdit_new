#!/bin/bash
# R58 stage 3 (Wan-Edit on FiVE-Bench-Correct, ON JEAN ZAY after jz/r50_port.sh): one render config per array task, ONLY the pairs whose
# Wan-Edit inputs changed in Correct (evaluation/r58_cases_*.json, built by r58_build_cases.py). Every other pair keeps its R49 render.
#
# Usage: sbatch --array=0-10 slurm_scripts/five_bench/r58_wan_render.sh correct   # 11 baselines on the 47 pairs of r58_cases_base.json
#        sbatch --array=11   slurm_scripts/five_bench/r58_wan_render.sh correct   # max_s5 on the 190 pairs of r58_cases_maxs5.json
#        sbatch -A omt@h100 --qos=qos_gpu_h100-dev --time=01:00:00 --array=0-1 slurm_scripts/five_bench/r58_wan_render.sh repro
#                                                                                 # reproducibility gate: native_s0.92 + matched59_s0.92 on r58_cases_repro.json
# Array index -> config (renderer):
#   0-5  native_s0.95 s0.92 s0.85 s0.75 s0.6 s0.4   (r48_wan_render.py)    6  native_s0.978   (r48_wan_render.py)
#   7-10 matched59_s0.95 s0.92 s0.85 s0.978         (r49_wan_render.py)    11 max_s5          (r49_wan_render.py, start maps from $D/r58_field)
#   repro: 0 native_s0.92 (r48)   1 matched59_s0.92 (r49)
# All with --anno_dir evaluation/five_bench_correct --mode full. NEVER overwrites a video: output only to $D/r58_wan_arms (correct) or
# $D/r58_repro (repro) -- any other root is refused -- and the renderers skip a clip whose PNGs already exist (RESUMABLE after a time limit).
# Output: $OUT/{config}_full/edit{T}/{video}/*.png
#SBATCH --job-name=r58_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r58_wan_render_%A_%a.out
#SBATCH --error=logs/r58_wan_render_%A_%a.err

MODE=${1:-}
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
ANNO=evaluation/five_bench_correct
case "$MODE" in
  correct) CONFIGS=(native_s0.95 native_s0.92 native_s0.85 native_s0.75 native_s0.6 native_s0.4 native_s0.978
                    matched59_s0.95 matched59_s0.92 matched59_s0.85 matched59_s0.978 max_s5)
           OUT=$D/r58_wan_arms ;;
  repro)   CONFIGS=(native_s0.92 matched59_s0.92)
           OUT=$D/r58_repro ;;
  *) echo "[r58_wan_render] usage: r58_wan_render.sh correct|repro"; exit 1 ;;
esac
# no-overwrite guard: the output root is one of the two new R58 folders, never an R48/R49/R51 render folder
case "$OUT" in
  "$D/r58_wan_arms"|"$D/r58_repro") ;;
  *) echo "[r58_wan_render] FATAL: refusing output root ${OUT} (only r58_wan_arms / r58_repro)"; exit 1 ;;
esac

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ]; then echo "[r58_wan_render] bad array id '${TID}'"; exit 1; fi
if [ "$TID" -ge "${#CONFIGS[@]}" ]; then echo "[r58_wan_render] array id ${TID} >= ${#CONFIGS[@]} configs -- nothing to do"; exit 0; fi
CONFIG=${CONFIGS[$TID]}
case "$MODE/$CONFIG" in
  repro/*)        CASES=evaluation/r58_cases_repro.json ;;
  correct/max_s5) CASES=evaluation/r58_cases_maxs5.json ;;
  correct/*)      CASES=evaluation/r58_cases_base.json ;;
esac
case "$CONFIG" in
  native_*) RENDERER=evaluation/r48_wan_render.py ;;
  *)        RENDERER=evaluation/r49_wan_render.py ;;
esac
for F in "$CASES" "$RENDERER" "$ANNO/edit1_FiVE.json"; do
  [ -s "$F" ] || { echo "[r58_wan_render] FATAL: missing $F (git pull? python evaluation/r58_build_cases.py?)"; exit 1; }
done

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r58_wan_render] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} config=${CONFIG} renderer=${RENDERER} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) -> ${OUT}"
echo "[r58_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
EXTRA=()
if [ "$CONFIG" = max_s5 ]; then      # preflight: the start maps of the 190 pairs must exist
  EXTRA=(--field_root "$D/r58_field")
  for T in 1 2 3 4 5 6; do
    N=$(ls "$D/r58_field/wan_max_s5/edit${T}"/*.npy 2>/dev/null | wc -l)
    [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r58_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run r58_field.sh first"; FAILED=$((FAILED + 1)); }
  done
  [ "$FAILED" -ne 0 ] && exit 1
fi

$PY_FB "$RENDERER" --config "$CONFIG" --mode full --cases "$CASES" --anno_dir "$ANNO" --out_root "$OUT" "${EXTRA[@]}" \
  || { echo "[r58_wan_render] ${RENDERER} reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render under this config's own {config}_full dir
RDIR="$OUT/${CONFIG}_full"
MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r58_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r58_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS)) dirs total (expect ${TOTAL} case clips)"
echo "[r58_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
