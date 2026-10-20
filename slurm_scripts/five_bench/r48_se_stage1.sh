#!/bin/bash
# R48 stage 1 (StreamEdit): the R38 n5_tgt09 draft, re-run with the cross-attention recorder ON.
#
# Usage: sbatch slurm_scripts/five_bench/r48_se_stage1.sh subset   # evaluation/r48_cases.json (18 clips)
#        sbatch slurm_scripts/five_bench/r48_se_stage1.sh full     # r7_anchor_manifest.json (419 pairs)
#
# Same flags as r38_stage1.sh task n5_tgt09 (N=5, --blend_sched tgt0.9, VP + Qwen anchors, seed 0, flow_shift 1,
# fg_boost 4, blend_power 2, rollout_chunk_size 21) plus --attn_out_dir (first 20 blocks, sigma window [0.55, 0.9]
# -- r31_stage1.py's defaults, i.e. t = 0.8 and 0.6 on this grid). The recorder is read-only: every latent must be
# BIT-IDENTICAL to r38_latents/n5_tgt09; that is checked per clip below and a mismatch fails the job.
#
# Outputs ($D = ~/Data/dataggen/outputs/five_bench):
#   $D/r48_se_stage1/frames/n5_tgt09/step04/edit{T}/{video}/*.png   draft frames (what r48_div measures against the source)
#   $D/r48_se_stage1/latents/edit{T}/{video}.npz                     z_trg / z_src
#   $D/r48_se_attn/edit{T}/{video}.npz                               per-token attention (src_tokens, trg_tokens, ...)
# One job loops the edit types (model loads once per r31_stage1.py call, as in R38).
#SBATCH --job-name=r48_se_stage1
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_se_stage1_%j.out
#SBATCH --error=logs/r48_se_stage1_%j.err

MODE=${1:-}
case "$MODE" in
  subset) CASES=evaluation/r48_cases.json ;;
  full)   CASES=evaluation/r7_anchor_manifest.json ;;
  *) echo "[r48_se_stage1] usage: r48_se_stage1.sh subset|full"; exit 1 ;;
esac

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors
R38_LATENTS=$D/r38_latents/n5_tgt09
METHOD=n5_tgt09
OUT=$D/r48_se_stage1
ATTN=$D/r48_se_attn
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
echo "[r48_se_stage1] job=${SLURM_JOB_ID} mode=${MODE} cases=${CASES} expect per type: ${EXPECT[*]} (total ${TOTAL})"
echo "[r48_se_stage1] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -eq 0 ] && continue
  echo "--- [r48_se_stage1] edit${T} (expect ${EXPECT[$((T - 1))]}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage1.py \
    --edit_type "$T" --method "$METHOD" \
    --data_root "$DATA_ROOT" --out_root "$OUT/frames" --latent_dir "$OUT/latents" --attn_out_dir "$ATTN" \
    --cases_json "$CASES" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 4 --measure_index 4 --step 5 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
    --blend_sched tgt0.9 --rollout_chunk_size 21 --seed 0 \
    || { echo "[r48_se_stage1] FAILED edit${T}"; FAILED=$((FAILED + 1)); }
done

# ---- post-run guards: counts per type, and bit-identity of every latent with R38's stored draft
N_ATTN=0
for T in 1 2 3 4 5 6; do
  N=$(ls "$ATTN/edit${T}"/*.npz 2>/dev/null | wc -l); N_ATTN=$((N_ATTN + N))
  [ "$N" -ne "${EXPECT[$((T - 1))]}" ] && { echo "[r48_se_stage1] MISMATCH edit${T}: ${N} attn npz, expected ${EXPECT[$((T - 1))]}"; FAILED=$((FAILED + 1)); }
done
echo "[r48_se_stage1] attn npz total: ${N_ATTN} (expect ${TOTAL})"
$PY_FB - "$OUT/latents" "$R38_LATENTS" "$ATTN" <<'EOF' || FAILED=$((FAILED + 1))
import sys, numpy as np
from pathlib import Path
new, ref, attn = map(Path, sys.argv[1:4])
files = sorted(new.glob("edit*/*.npz")); same = 0; bad = []
for f in files:
    r = ref / f.parent.name / f.name
    if not r.exists():
        bad.append((f.parent.name, f.stem, "no R38 latent")); continue
    if np.array_equal(np.load(f)["z_trg"], np.load(r)["z_trg"]): same += 1
    else: bad.append((f.parent.name, f.stem, "z_trg differs"))
    a = attn / f.parent.name / f.name
    if not a.exists(): bad.append((f.parent.name, f.stem, "attention npz missing"))
print(f"[r48_se_stage1] bit-identical to R38 n5_tgt09: {same}/{len(files)}")
for b in bad: print("[r48_se_stage1] BAD", *b)
sys.exit(int(bool(bad)))
EOF
echo "[r48_se_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
