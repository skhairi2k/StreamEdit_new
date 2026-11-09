#!/bin/bash
# R69 (Wan-Edit on FiVE-Bench-Correct, ON JEAN ZAY after jz/r50_port.sh): one extra point on Wan-Edit's equal-step (matched-59) global curve,
# matched59_s0.5 (N = 354, skip 295 -> 59 active steps, start sigma 0.4997; the same 236 backbone passes per clip as R58's matched-59 configs:
# skipped steps make no model call). It is meant to be more faithful than max_s5 on PSNR / SSIM / LPIPS, so FiVE-Acc at matched background
# fidelity can be interpolated at equal steps. All 415 Correct pairs are rendered (no earlier s = 0.5 render exists to reuse).
# Same renderer and settings as R58's matched-59 configs (r49_wan_render.py: text only, seeded noise base_seed 0, n_avg 1, edit.py guide scales).
#
# Usage: sbatch slurm_scripts/five_bench/r69_wan_render.sh
# Output: $D/r69_wan_arms/matched59_s0.5_full/edit{T}/{video}/*.png. RESUMABLE: finished clips are skipped, so a render that hits the time limit is
# simply resubmitted. NEVER overwrites a video: the output root must be $D/r69_wan_arms (any other root is refused).
#SBATCH --job-name=r69_wan_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r69_wan_render_%j.out
#SBATCH --error=logs/r69_wan_render_%j.err

CONFIG=matched59_s0.5
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
CASES=evaluation/r43_anchor_manifest_correct.json
ANNO=evaluation/five_bench_correct
OUT=$D/r69_wan_arms
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

# no-overwrite guard: the output root is the new R69 folder, never an R48 / R49 / R58 / R62 render folder
[ "$OUT" = "$D/r69_wan_arms" ] || { echo "[r69_wan_render] FATAL: refusing output root ${OUT} (only r69_wan_arms)"; exit 1; }

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

for F in "$CASES" "$ANNO/edit1_FiVE.json" evaluation/r49_wan_render.py; do
  [ -s "$F" ] || { echo "[r69_wan_render] FATAL: missing $F (git pull?)"; exit 1; }
done

# ---- preflight: the config resolves to the planned grid (354 steps, skip 295 = 59 active), and the case file holds the 415 Correct pairs
GRID=$($PY_FB - <<'PYEOF'
import sys
sys.path.insert(0, "evaluation")
import r49_wan_render as R
kind, n, skip, _ = R.resolve("matched59_s0.5")
print(f"{kind} {n} {skip}")
PYEOF
)
[ "$GRID" = "global 354 295" ] || { echo "[r69_wan_render] FATAL: ${CONFIG} resolves to '${GRID}', expected 'global 354 295'"; exit 1; }

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
[ "$TOTAL" -eq 415 ] || { echo "[r69_wan_render] FATAL: ${CASES} holds ${TOTAL} pairs, expected 415"; exit 1; }
echo "[r69_wan_render] job=${SLURM_JOB_ID:-local} config=${CONFIG} (N 354, skip 295, 59 active) cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) -> ${OUT}"
echo "[r69_wan_render] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
$PY_FB evaluation/r49_wan_render.py --config "$CONFIG" --mode full --cases "$CASES" --anno_dir "$ANNO" --out_root "$OUT" \
  || { echo "[r69_wan_render] r49_wan_render.py reported failures"; FAILED=$((FAILED + 1)); }

# ---- post-run guard: every case clip has a complete render (r48_wan_common.expected_frames PNGs)
RDIR="$OUT/${CONFIG}_full"
MISSING=$($PY_FB evaluation/r48_units.py --mode full --missing "$RDIR" --cases "$CASES")
N_MISS=$([ -n "$MISSING" ] && echo "$MISSING" | wc -l || echo 0)
[ "$N_MISS" -ne 0 ] && { echo "[r69_wan_render] ${CONFIG}: ${N_MISS} case clip(s) without a complete render, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"; FAILED=$((FAILED + 1)); }
echo "[r69_wan_render] ${RDIR##*/}: $((TOTAL - N_MISS))/${TOTAL} clips complete"
echo "[r69_wan_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
