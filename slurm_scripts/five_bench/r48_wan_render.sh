#!/bin/bash
# R48 stage 3 (Wan-Edit): one render config per array task -- the global baselines and the adaptive arms.
#
# Usage: sbatch --array=0-8 slurm_scripts/five_bench/r48_wan_render.sh subset       # the 9 baselines on the 18 clips (wan-baselines)
#        sbatch --array=9-12 slurm_scripts/five_bench/r48_wan_render.sh subset      # the 4 adaptive arms on the 18 clips (wan-render)
#        sbatch --array=0-$((8+N)) slurm_scripts/five_bench/r48_wan_render.sh full   # 9 baselines + the N `wan` arms of evaluation/r48_selected_arms.txt, 419 pairs
#        sbatch --array=0-1 slurm_scripts/five_bench/r48_wan_render.sh explore      # R48 v2a: the 2 global starts above step 9 (native_s0.978, native_s1) on the 18 clips
#        sbatch --array=0   slurm_scripts/five_bench/r48_wan_render.sh explore_full # R49 full stage: native_s0.978 (index 0 of --list_explore) on the 419 pairs
#        sbatch --array=0-5 slurm_scripts/five_bench/r48_wan_render.sh full         # R49 full stage, Jean Zay: the 6 native configs native_s0.95 .. native_s0.4 (0.92 = official)
# MODES explore / explore_full use `r48_wan_sched.py --list_explore` (index 0..1) instead of the 9-baseline list and write {config}_subset / {config}_full.
# Array index -> config:  0..8 = `r48_wan_sched.py --list` (native_s0.95 .. native_s0.4, matched_s0.95 .. matched_s0.85),
#                         9.. = adapt_{arm}  (subset: dino_SD lpips_SD dino_D S; full: `wan` lines of r48_selected_arms.txt, file order)
# Adaptive configs read $D/r48_field/wan_{arm}/edit{T}/{video}.npy (start step per token, from r48_field.py).
# Output: $D/r48_wan_arms/{config}_{MODE}/edit{T}/{video}/*.png (41 frames per clip), every config in its own dir per mode
# RESUMABLE: finished clips are skipped, so a timed-out full run is simply re-submitted (a full config is ~420 clips x 3-4 min).
#SBATCH --job-name=r48_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-12
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_wan_render_%A_%a.out
#SBATCH --error=logs/r48_wan_render_%A_%a.err

MODE=${1:-}
case "$MODE" in
  subset)  CASES=evaluation/r48_cases.json;        PYMODE=subset ;;
  full)    CASES=evaluation/r7_anchor_manifest.json; PYMODE=full ;;
  explore) CASES=evaluation/r48_cases.json;        PYMODE=subset ;;   # v2a explore configs, on the subset clips
  explore_full) CASES=evaluation/r7_anchor_manifest.json; PYMODE=full ;;   # R49 full stage: native_s0.978 on the 419 pairs
  *) echo "[r48_wan_render] usage: r48_wan_render.sh subset|full|explore|explore_full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT=$D/r48_wan_arms
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ]; then echo "[r48_wan_render] bad array id '${TID}'"; exit 1; fi
if [ "$MODE" = explore ] || [ "$MODE" = explore_full ]; then
  mapfile -t CONFIGS < <($PY_FB evaluation/r48_wan_sched.py --list_explore 2>/dev/null)
  [ "${#CONFIGS[@]}" -eq 2 ] || { echo "[r48_wan_render] FATAL: expected 2 explore configs, got ${#CONFIGS[@]}"; exit 1; }
else
  mapfile -t CONFIGS < <($PY_FB evaluation/r48_wan_sched.py --list 2>/dev/null)
  [ "${#CONFIGS[@]}" -eq 9 ] || { echo "[r48_wan_render] FATAL: expected 9 baseline configs, got ${#CONFIGS[@]}"; exit 1; }
fi
# The adaptive arms (ids 9+) are only listed when this task needs one: the 9 baselines (ids 0-8) depend on neither the field
# nor the gate, so a full-mode baseline job must not require evaluation/r48_selected_arms.txt.
if [ "$MODE" != explore ] && [ "$MODE" != explore_full ] && [ "$TID" -ge 9 ]; then
  if [ "$MODE" = subset ]; then
    for A in dino_SD lpips_SD dino_D S; do CONFIGS+=("adapt_${A}"); done
  else
    [ -s evaluation/r48_selected_arms.txt ] || { echo "[r48_wan_render] FATAL: evaluation/r48_selected_arms.txt missing/empty -- write it at gate-decision"; exit 1; }
    while read -r H A; do [ "$H" = wan ] && CONFIGS+=("adapt_${A}"); done < evaluation/r48_selected_arms.txt
  fi
fi
if [ "$TID" -ge "${#CONFIGS[@]}" ]; then echo "[r48_wan_render] array id ${TID} >= ${#CONFIGS[@]} configs -- nothing to do"; exit 0; fi
CONFIG=${CONFIGS[$TID]}

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r48_wan_render] job=${SLURM_ARRAY_JOB_ID}_${TID} mode=${MODE} config=${CONFIG} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r48_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
if [[ "$CONFIG" == adapt_* ]]; then      # preflight: the start maps must exist for every clip
  ARM=${CONFIG#adapt_}
  for T in 1 2 3 4 5 6; do
    N=$(ls "$D/r48_field/wan_${ARM}/edit${T}"/*.npy 2>/dev/null | wc -l)
    [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r48_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run the field step first"; FAILED=$((FAILED + 1)); }
  done
  [ "$FAILED" -ne 0 ] && exit 1
fi

$PY_FB evaluation/r48_wan_render.py --config "$CONFIG" --mode "$PYMODE" --cases "$CASES" --out_root "$OUT" \
  || { echo "[r48_wan_render] r48_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every clip of the case list has a complete (41-frame) render under this config's own {config}_{PYMODE} dir
RDIR="$OUT/${CONFIG}_${PYMODE}"
MISSING=$($PY_FB evaluation/r48_units.py --mode "$PYMODE" --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r48_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r48_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS)) dirs total (expect ${TOTAL} case clips)"
echo "[r48_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
