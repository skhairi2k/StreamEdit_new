#!/bin/bash
# R61 stage 1 (StreamEdit on LongLive, VP): se_max_ll's 5-step draft on FiVE-Bench-Correct, cross-attention recorder ON, 415 pairs.
#
# Usage: sbatch slurm_scripts/five_bench/r61_ll_stage1.sh        (r61_chain.sh)
# R55's LongLive draft (N=5, --blend_sched tgt0.9, VP + anchors, seed 0, flow_shift 1, fg_boost 2, blend_power 2, attention recorder
# at r31_stage1.py's defaults: first 20 blocks, sigma window [0.55, 0.9]; LongLive has no rollout windows, so no --rollout_chunk_size),
# re-run on the Correct inputs: prompts + trigger words from evaluation/five_bench_correct (--edit_json_dir, absolute), cases from
# r43_anchor_manifest_correct.json, anchors from anchors_r43_correct. Nothing from R45/R55 is reused (every anchor changed).
# PROMPT GUARD (as R59): r31_stage1.py writes $OUT/latents/_prompts_edit{T}.json (the strings the pipeline receives);
# r59_check_prompts.py compares them to the Correct JSONs at the end and a mismatch fails the job, so the afterok chain
# (div / field / render) stops.
#
# Outputs ($D = ~/Data/dataggen/outputs/five_bench):
#   $D/r61_ll_stage1/frames/n5_tgt09/step04/edit{T}/{video}/*.png  draft frames (what r61_div measures against the source)
#   $D/r61_ll_stage1/latents/edit{T}/{video}.npz                    z_trg / z_src   (+ _prompts_edit{T}.json)
#   $D/r61_ll_attn/edit{T}/{video}.npz                              per-token attention (src_tokens, trg_tokens, ...)
# One job loops the edit types (model loads once per r31_stage1.py call). NO RESUME: r31_stage1.py re-renders what it is given.
#SBATCH --job-name=r61_ll_stage1
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r61_ll_stage1_%j.out
#SBATCH --error=logs/r61_ll_stage1_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors_r43_correct
CASES=evaluation/r43_anchor_manifest_correct.json
EDIT_JSON_DIR=$REPO/evaluation/five_bench_correct
METHOD=n5_tgt09
SCHED=tgt0.9
STEPS=5
OUT=$D/r61_ll_stage1
ATTN=$D/r61_ll_attn
PY_SG=~/anaconda3/envs/streamgve/bin/python
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT/frames" "$OUT/latents" "$ATTN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# expected clips per edit type, from the cases file itself
read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'EOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
EOF
)"
TOTAL=0; for n in "${EXPECT[@]}"; do TOTAL=$((TOTAL + n)); done
echo "[r61_ll_stage1] job=${SLURM_JOB_ID} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL}) LONGLIVE, VP, ${STEPS}-step draft, ${SCHED}, edit_json_dir=${EDIT_JSON_DIR}"
echo "[r61_ll_stage1] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r61_ll_stage1] edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage1.py --backbone ll \
    --edit_type "$T" --method "$METHOD" \
    --data_root "$DATA_ROOT" --edit_json_dir "$EDIT_JSON_DIR" --out_root "$OUT/frames" --latent_dir "$OUT/latents" --attn_out_dir "$ATTN" \
    --cases_json "$CASES" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 4 --measure_index 4 --step "$STEPS" --flow_shift 1.0 --fg_boost_factor 2 --blend_power 2 \
    --blend_sched "$SCHED" --seed 0 \
    || { echo "[r61_ll_stage1] FAILED edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: counts per type; every latent is a LongLive VP draft of THIS schedule and step count and has its attention npz
N_ATTN=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$ATTN/edit${T}"/*.npz 2>/dev/null | wc -l); N_ATTN=$((N_ATTN + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r61_ll_stage1] MISMATCH edit${T}: ${N} attn npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1)); }
done
echo "[r61_ll_stage1] attn npz total: ${N_ATTN} (expect ${TOTAL})"
$PY_FB - "$OUT/latents" "$ATTN" "$SCHED" "$STEPS" "${EXPECT[@]}" <<'EOF' || FAILED=$((FAILED + 1))
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
print(f"[r61_ll_stage1] latents: {n} total; checked (vp, {sched}, {steps}, ll) + attention present: {len(bad)} problem(s)")
for b in bad[:20]:
    print(f"[r61_ll_stage1] BAD {b}")
sys.exit(1 if bad else 0)
EOF
# ---- prompt guard (as R59): the strings the draft was rendered with must be the Correct ones
$PY_FB evaluation/r59_check_prompts.py --sidecar "$OUT/latents/_prompts_edit*.json" --edit_json_dir "$EDIT_JSON_DIR" \
  --anchor_dir "$ANCHOR_ROOT" --expect_n "$TOTAL" || { echo "[r61_ll_stage1] PROMPT-MISMATCH"; FAILED=$((FAILED + 1)); }
echo "[r61_ll_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
