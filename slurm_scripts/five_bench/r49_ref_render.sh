#!/bin/bash
# R49: render the reference arm (R48's dino_SD: linear map, rho_min 0.3, LPIPS reference 0.75) for the NEW clip only
# (0004_car-roundabout, edit 3), on both hosts, so the reference columns cover every clip of evaluation/r49_cases.json.
#
# Usage: sbatch slurm_scripts/five_bench/r49_ref_render.sh        (normally submitted by r49_chain.sh, afterok on r49_field.sh)
# Input : $D/r48_field/se_dino_SD/edit3/0004_car-roundabout.npz  and  $D/r48_field/wan_dino_SD/edit3/0004_car-roundabout.{npz,npy}
#         (written by r49_field.sh with r48_field.py on the r49 drafts)
# Output: $D/r48_se_arms/r48_dino_SD_subset/step14/edit3/0004_car-roundabout/*.png      (StreamEdit, flags as r48_se_stage3.sh)
#         $D/r48_wan_arms/adapt_dino_SD_subset/edit3/0004_car-roundabout/*.png          (Wan-Edit adaptive, loop from step 9)
# The 17 other clips of those directories come from R48. r31_stage3.py rewrites its manifest with only the clips it just
# rendered, so the existing rows of _manifest_edit3.csv are saved and merged back.
#SBATCH --job-name=r49_ref_render
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r49_ref_render_%j.out
#SBATCH --error=logs/r49_ref_render_%j.err

NEW_CLIP=evaluation/r49_new_clip.json
VIDEO=0004_car-roundabout
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
FIELD_ROOT=$D/r48_field
SE_OUT=$D/r48_se_arms
SE_METHOD=r48_dino_SD_subset
ANCHOR_ROOT=$D/anchors
PY_SG=/home/ids/skhairi/anaconda3/envs/streamgve/bin/python
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
echo "[r49_ref_render] job=${SLURM_JOB_ID:-local} node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

FAILED=0
# ---- preflight: the reference fields of the new clip must exist
for F in "$FIELD_ROOT/se_dino_SD/edit3/$VIDEO.npz" "$FIELD_ROOT/wan_dino_SD/edit3/$VIDEO.npz" "$FIELD_ROOT/wan_dino_SD/edit3/$VIDEO.npy"; do
  [ -s "$F" ] || { echo "[r49_ref_render] FATAL: missing $F -- run r49_field.sh first"; FAILED=$((FAILED + 1)); }
done
[ "$FAILED" -ne 0 ] && exit 1

# ---- StreamEdit dino_SD on the new clip
MAN="$SE_OUT/$SE_METHOD/_manifest_edit3.csv"
SAVED=$(mktemp)
[ -f "$MAN" ] && cp "$MAN" "$SAVED" || : > "$SAVED"
echo "=== [r49_ref_render] StreamEdit ${SE_METHOD} edit3/${VIDEO} $(date '+%F %T')"
$PY_SG evaluation/r31_stage3.py --edit_type 3 --method "$SE_METHOD" --rho_dir "$FIELD_ROOT/se_dino_SD" \
  --data_root "$DATA_ROOT" --out_root "$SE_OUT" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
  --cases_json "$NEW_CLIP" --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
  --rollout_chunk_size 21 --seed 0 \
  || { echo "[r49_ref_render] FAILED StreamEdit render"; FAILED=$((FAILED + 1)); }
# merge: saved rows for the other clips + the row r31_stage3 just wrote for the new clip
$PY_FB - "$SAVED" "$MAN" "$VIDEO" <<'PYEOF' || FAILED=$((FAILED + 1))
import csv, sys
saved, man, video = sys.argv[1:4]
old = list(csv.reader(open(saved))); new = list(csv.reader(open(man)))
header = new[0] if new else (old[0] if old else None)
rows = [r for r in old[1:] if r and r[1] != video] + [r for r in new[1:] if r and r[1] == video]
with open(man, "w", newline="") as f:
    w = csv.writer(f); w.writerow(header); w.writerows(sorted(rows, key=lambda r: r[1]))
print(f"[r49_ref_render] manifest edit3 merged: {len(rows)} rows ({[r[2] for r in rows if r[1] == video]} for the new clip)")
PYEOF
rm -f "$SAVED"
N_PNG=$(ls "$SE_OUT/$SE_METHOD/step14/edit3/$VIDEO"/*.png 2>/dev/null | wc -l)
echo "[r49_ref_render] StreamEdit new clip: ${N_PNG} frames"
[ "$N_PNG" -gt 0 ] || { echo "[r49_ref_render] FAILED: no StreamEdit frames for the new clip"; FAILED=$((FAILED + 1)); }

# ---- Wan-Edit adapt_dino_SD on the new clip
echo "=== [r49_ref_render] Wan-Edit adapt_dino_SD edit3/${VIDEO} $(date '+%F %T')"
$PY_FB evaluation/r48_wan_render.py --config adapt_dino_SD --mode subset --cases "$NEW_CLIP" \
  || { echo "[r49_ref_render] FAILED Wan-Edit render"; FAILED=$((FAILED + 1)); }
MISSING=$($PY_FB evaluation/r48_units.py --mode subset --missing "$D/r48_wan_arms/adapt_dino_SD_subset" --cases "$NEW_CLIP")
[ -n "$MISSING" ] && { echo "[r49_ref_render] FAILED: Wan-Edit new clip has no complete render"; FAILED=$((FAILED + 1)); } \
  || echo "[r49_ref_render] Wan-Edit new clip complete"
echo "[r49_ref_render] failures: ${FAILED}"
exit $(( FAILED > 0 ))
