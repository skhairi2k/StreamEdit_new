#!/bin/bash
# R55 stage 1 (StreamEdit on LongLive, VP): the N=5 tgt0.9 draft with the cross-attention recorder ON, on the 419 pairs.
#
# Usage: sbatch slurm_scripts/five_bench/r55_ll_stage1.sh        (r55_chain.sh, after the smoke test passed)
# r48_se_stage1.sh full with --backbone ll and R45's LongLive settings: VP + Qwen anchors, N=5, --blend_sched tgt0.9 (only the t=1 call anchored to
# the source, target released from t=0.8 -- the same draft logic as on Self-Forcing), fg_boost 2, flow_shift 1, blend_power 2, seed 0, recorder at
# r31_stage1.py's defaults (first 20 blocks, sigma window [0.55, 0.9]). LongLive has no rollout windows (no --rollout_chunk_size). There is no
# reference draft to compare bit for bit, so the guards check the counts and that every latent is a LongLive VP draft of this schedule.
#
# Outputs ($D = ~/Data/dataggen/outputs/five_bench):
#   $D/r55_ll_stage1/frames/n5_tgt09/step04/edit{T}/{video}/*.png   draft frames (what r55_div measures against the source)
#   $D/r55_ll_stage1/latents/edit{T}/{video}.npz                    z_trg / z_src
#   $D/r55_ll_attn/edit{T}/{video}.npz                              per-token attention
# One job loops the edit types. NO RESUME: r31_stage1.py re-renders what it is given.
#SBATCH --job-name=r55_ll_stage1
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r55_ll_stage1_%j.out
#SBATCH --error=logs/r55_ll_stage1_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors
CASES=evaluation/r7_anchor_manifest.json
METHOD=n5_tgt09
SCHED=tgt0.9
STEPS=5
OUT=$D/r55_ll_stage1
ATTN=$D/r55_ll_attn
PY_SG=~/anaconda3/envs/streamgve/bin/python
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT/frames" "$OUT/latents" "$ATTN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# expected clips per edit type, from the cases file itself
read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r55_ll_stage1] job=${SLURM_JOB_ID} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) LONGLIVE, VP"
echo "[r55_ll_stage1] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r55_ll_stage1] edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage1.py --backbone ll \
    --edit_type "$T" --method "$METHOD" \
    --data_root "$DATA_ROOT" --out_root "$OUT/frames" --latent_dir "$OUT/latents" --attn_out_dir "$ATTN" \
    --cases_json "$CASES" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 4 --measure_index 4 --step "$STEPS" --flow_shift 1.0 --fg_boost_factor 2 --blend_power 2 \
    --blend_sched "$SCHED" --seed 0 \
    || { echo "[r55_ll_stage1] FAILED edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: counts per type; every latent is a LongLive VP draft of THIS schedule and step count and has its attention npz
N_ATTN=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$ATTN/edit${T}"/*.npz 2>/dev/null | wc -l); N_ATTN=$((N_ATTN + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r55_ll_stage1] MISMATCH edit${T}: ${N} attn npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1)); }
done
echo "[r55_ll_stage1] attn npz total: ${N_ATTN} (expect ${TOTAL})"
$PY_FB - "$OUT/latents" "$ATTN" "$SCHED" "$STEPS" "${EXPECT[@]}" <<'PYEOF' || FAILED=$((FAILED + 1))
import sys, numpy as np
from pathlib import Path
latents, attn, sched, steps = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
expect = dict(zip(range(1, 7), map(int, sys.argv[5:11])))
bad = []; n = 0
for t, n_exp in expect.items():
    files = sorted((latents / f"edit{t}").glob("*.npz"))
    n += len(files)
    if len(files) != n_exp:
        bad.append(f"edit{t}: {len(files)} latents, expected {n_exp}")
for f in sorted(latents.glob("edit*/*.npz")):
    z = np.load(f)
    got = (str(z["vp_mode"]), str(z["blend_sched"]), int(z["n_steps"]), str(z["backbone"]))
    if got != ("vp", sched, steps, "ll"):
        bad.append(f"{f.parent.name}/{f.stem}: vp_mode/blend_sched/n_steps/backbone = {got}")
    if not (attn / f.parent.name / f.name).exists():
        bad.append(f"{f.parent.name}/{f.stem}: attention npz missing")
print(f"[r55_ll_stage1] latents: {n} total; checked (vp, {sched}, {steps}, ll) + attention present: {len(bad)} problem(s)")
for b in bad[:20]:
    print(f"[r55_ll_stage1] BAD {b}")
sys.exit(1 if bad else 0)
PYEOF
echo "[r55_ll_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
