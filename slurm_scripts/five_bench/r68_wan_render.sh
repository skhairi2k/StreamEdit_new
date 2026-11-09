#!/bin/bash
# R68 stage 3 (Wan-Edit on FiVE-Bench-Correct, ON JEAN ZAY after jz/r50_port.sh): R51's demand ablations A1-A3 on Wan-Edit, never rendered
# before on any benchmark, so ALL 415 Correct pairs are rendered (no reuse). One arm per array task:
#   task 0 nodiv_s5 (A1, q = S)   1 dino_s5 (A2, q = S * m_dino)   2 lpips_s5 (A3, q = S * m_lpips)
# Same renderer and settings as R58's max_s5 (A0) / R62's nosupp_s5 (A4) (r49_wan_render.py: N = 50, loop start 5, per-token start map,
# seed 0), only the start maps differ.
#
# Usage: sbatch --array=0-2 slurm_scripts/five_bench/r68_wan_render.sh     (r68_wan_chain.sh, afterok on r68_field.sh wan)
#        sbatch --array=<i> slurm_scripts/five_bench/r68_wan_render.sh     (resubmit one arm that hit the time limit)
# Input : $D/r68_field/wan_{config}/edit{T}/{video}.npy   (r51_field.py, via r68_field.sh wan; 190 from R58's inputs + 225 from R49's)
# Output: $D/r68_wan_arms/{config}_full/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a task that hits the time limit is
# simply resubmitted. NEVER overwrites a video: the output root must be $D/r68_wan_arms (any other root is refused).
# ~415 clips per arm: R62's 190 nosupp_s5 clips took ~4-5 h on one H100, so ~10-11 h per task (limit 20 h).
#SBATCH --job-name=r68_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r68_wan_render_%A_%a.out
#SBATCH --error=logs/r68_wan_render_%A_%a.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r43_anchor_manifest_correct.json
ANNO=evaluation/five_bench_correct
FIELD=$D/r68_field
OUT=$D/r68_wan_arms
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
CONFIGS=(nodiv_s5 dino_s5 lpips_s5)
TID=${SLURM_ARRAY_TASK_ID:-}
[[ "$TID" =~ ^[0-9]+$ ]] && [ "$TID" -lt "${#CONFIGS[@]}" ] || { echo "[r68_wan_render] bad array id '${TID}' (expected 0-$((${#CONFIGS[@]} - 1)) for: ${CONFIGS[*]})"; exit 1; }
CONFIG=${CONFIGS[$TID]}

# no-overwrite guard: the output root is the new R68 folder, never an R48 / R49 / R51 / R58 / R62 render folder
[ "$OUT" = "$D/r68_wan_arms" ] || { echo "[r68_wan_render] FATAL: refusing output root ${OUT} (only r68_wan_arms)"; exit 1; }

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

for F in "$CASES" "$ANNO/edit1_FiVE.json" evaluation/r49_wan_render.py; do
  [ -s "$F" ] || { echo "[r68_wan_render] FATAL: missing $F (git pull?)"; exit 1; }
done
grep -q "\"${CONFIG}\"" evaluation/r49_wan_render.py || { echo "[r68_wan_render] FATAL: r49_wan_render.py has no config ${CONFIG} (git pull?)"; exit 1; }

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r68_wan_render] job=${SLURM_ARRAY_JOB_ID:-local}_${TID} config=${CONFIG} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) -> ${OUT}"
echo "[r68_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# ---- preflight: the start maps must exist for every clip
FAILED=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$FIELD/wan_${CONFIG}/edit${T}"/*.npy 2>/dev/null | wc -l)
  [ "$N" -lt "${EXPECT[$((T - 1))]}" ] && { echo "[r68_wan_render] FATAL: edit${T} has ${N} start maps, need ${EXPECT[$((T - 1))]} -- run r68_field.sh wan first"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -ne 0 ] && exit 1

$PY_FB evaluation/r49_wan_render.py --config "$CONFIG" --mode full --cases "$CASES" --anno_dir "$ANNO" --out_root "$OUT" --field_root "$FIELD" \
  || { echo "[r68_wan_render] r49_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render
RDIR="$OUT/${CONFIG}_full"
MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r68_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r68_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS))/${TOTAL} clips"
echo "[r68_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
