#!/bin/bash
# R71 stage 2 (Wan-Edit on FiVE-Bench-Correct, ON JEAN ZAY after jz/r50_port.sh): the range-normalized mapping m = round(49 q), s = 50 - m
# (m_max = 49, s_min = 1) and the equal-NFE curve it needs. All 415 Correct pairs per config, nothing reused. One config per array task:
#   task 0 max_s1 (A0)   1 nosupp_s1 (A4)   2 nodiv_s1 (A1)   3 dino_s1 (A2)   4 lpips_s1 (A3)      per-token start maps from r71_field
#   task 5-10 matched63_s{0.996,0.978,0.95,0.92,0.85,0.5}   global start sigma, N / skip chosen so exactly 63 steps are active
#                                                           (= the adaptive budget: 14 draft + 49 final): 64/1, 70/7, 80/17, 90/27, 119/56, 378/315
# Same renderer and settings as R58 / R68 (r49_wan_render.py: N = 50 for the adaptive arms, seed 0, 41 frames).
#
# Usage: sbatch --array=0-4  slurm_scripts/five_bench/r71_wan_render.sh   (after r71_field.sh, afterok; r71_wan_chain.sh)
#        sbatch --array=5-10 slurm_scripts/five_bench/r71_wan_render.sh   (no dependency)
#        sbatch --array=<i>  slurm_scripts/five_bench/r71_wan_render.sh   (resubmit one config that hit the time limit)
# Input : adaptive: $D/r71_field/wan_{config}/edit{T}/{video}.npy   (r71_field.sh)
# Output: $D/r71_wan_arms/{config}_full/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a task that hits the time limit is
# simply resubmitted. NEVER overwrites a video: the output root must be $D/r71_wan_arms (any other root is refused).
# ~415 clips per config: R68's arms (59 active steps) are estimated at ~10-11 h on one H100, so ~11-12 h per task at 63 steps (limit 20 h).
#SBATCH --job-name=r71_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r71_wan_render_%A_%a.out
#SBATCH --error=logs/r71_wan_render_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r43_anchor_manifest_correct.json
ANNO=evaluation/five_bench_correct
FIELD=$D/r71_field
OUT=$D/r71_wan_arms
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
CONFIGS=(max_s1 nosupp_s1 nodiv_s1 dino_s1 lpips_s1 matched63_s0.996 matched63_s0.978 matched63_s0.95 matched63_s0.92 matched63_s0.85 matched63_s0.5)
ADAPTIVE=5      # tasks 0-4 read start maps from $FIELD; tasks 5-10 are global starts
TID=${SLURM_ARRAY_TASK_ID:-}
[[ "$TID" =~ ^[0-9]+$ ]] && [ "$TID" -lt "${#CONFIGS[@]}" ] || { echo "[r71_wan_render] bad array id '${TID}' (expected 0-$((${#CONFIGS[@]} - 1)) for: ${CONFIGS[*]})"; exit 1; }
CONFIG=${CONFIGS[$TID]}

# no-overwrite guard: the output root is the new R71 folder, never an R48 / R49 / R51 / R58 / R62 / R68 / R69 render folder
[ "$OUT" = "$D/r71_wan_arms" ] || { echo "[r71_wan_render] FATAL: refusing output root ${OUT} (only r71_wan_arms)"; exit 1; }

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

for F in "$CASES" "$ANNO/edit1_FiVE.json" evaluation/r49_wan_render.py; do
  [ -s "$F" ] || { echo "[r71_wan_render] FATAL: missing $F (git pull?)"; exit 1; }
done
# adaptive configs are literal keys of ADAPT; matched63_* are generated from MATCHED63, so check the code (git pull?) via the table
if [ "$TID" -lt "$ADAPTIVE" ]; then
  grep -q "\"${CONFIG}\"" evaluation/r49_wan_render.py || { echo "[r71_wan_render] FATAL: r49_wan_render.py has no config ${CONFIG} (git pull?)"; exit 1; }
else
  grep -q "^MATCHED63" evaluation/r49_wan_render.py || { echo "[r71_wan_render] FATAL: r49_wan_render.py has no MATCHED63 (git pull?)"; exit 1; }
fi

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r71_wan_render] job=${SLURM_ARRAY_JOB_ID:-local}_${TID} config=${CONFIG} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) -> ${OUT}"
echo "[r71_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the start maps must exist for every clip
FAILED=0
if [ "$TID" -lt "$ADAPTIVE" ]; then
  for T in 1 2 3 4 5 6; do
    N=$(ls "$FIELD/wan_${CONFIG}/edit${T}"/*.npy 2>/dev/null | wc -l)
    [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r71_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run r71_field.sh first"; FAILED=$((FAILED + 1)); }
  done
fi
[ "$FAILED" -ne 0 ] && exit 1

$PY_FB evaluation/r49_wan_render.py --config "$CONFIG" --mode full --cases "$CASES" --anno_dir "$ANNO" --out_root "$OUT" --field_root "$FIELD" \
  || { echo "[r71_wan_render] r49_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render
RDIR="$OUT/${CONFIG}_full"
MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r71_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r71_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS))/${TOTAL} clips"
echo "[r71_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
