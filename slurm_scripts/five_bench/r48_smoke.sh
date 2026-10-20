#!/bin/bash
# R48 smoke -- gates G1-G6 on 2 cases (0001_bus edit1, 0011_lucia edit5), one GPU job.
#
#   G1  StreamEdit draft with the attention recorder ON == R38's stored n5_tgt09 latents, bit for bit
#   G2  Wan-Edit 17-step draft with the recorder hooks ON == unhooked (hooks are read-only)
#   G3  Wan-Edit --route_start_map all = 9  ==  stock --skip_timesteps 9 (50 steps)
#   G4  StreamEdit stage 3 with a constant rho = 2 field (r48_field's real writer) == the global rho = 2 run
#   G5  dino_SD stage-3 renders at rho_min 0.3 (the plan's map) and 0: seam / freeze check against the global run
#   G6  compute_E is host-agnostic: identical synthetic data down the se and the wan writer path -> identical S/m/E
#
# Every gate prints "GATE Gk PASS|FAIL"; the job fails if any gate does. Output: ~/Data/.../five_bench/r48_smoke
# Interpreters are called by path -- `conda activate` does not reliably change `python` here.
#SBATCH --job-name=r48_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r48_smoke_%j.out
#SBATCH --error=logs/r48_smoke_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors
SMOKE=$D/r48_smoke
PY_SG=~/anaconda3/envs/streamgve/bin/python     # StreamEdit stages
PY_FB=~/anaconda3/envs/five-bench/bin/python    # divergence / field / checks / Wan-Edit

cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
rm -rf "$SMOKE"; mkdir -p "$SMOKE"
CASES=$SMOKE/cases.json
cat > "$CASES" <<'EOF'
[{"video_name": "0001_bus", "edit_type": 1},
 {"video_name": "0011_lucia", "edit_type": 5}]
EOF
# r31_divergence reads the whole case dict via evaluation/r48_cases.json's schema: take those entries from it
$PY_FB - <<EOF
import json
keep = {("0001_bus", 1), ("0011_lucia", 5)}
c = [x for x in json.load(open("evaluation/r48_cases.json")) if (x["video_name"], int(x["edit_type"])) in keep]
assert len(c) == 2, len(c)
json.dump(c, open("$CASES", "w"))
EOF

echo "[r48_smoke] job=${SLURM_JOB_ID} node=$(hostname) out=$SMOKE"; nvidia-smi -L 2>&1 | head -2
FAILED=0
step () { echo; echo "=== [r48_smoke] $* ($(date '+%T')) ==="; }
chk () { "$@" || FAILED=$((FAILED+1)); }

step "G6 compute_E host-agnostic self-test"
chk $PY_FB evaluation/r48_field.py --selftest | sed 's/^\[selftest\] GATE G6/GATE G6/'

step "stage 1 (StreamEdit draft N=5, tgt0.9, attention recorder ON)"
for T in 1 5; do
  $PY_SG evaluation/r31_stage1.py --edit_type $T --method smoke \
    --data_root "$DATA_ROOT" --out_root $SMOKE/frames --latent_dir $SMOKE/lat --attn_out_dir $SMOKE/attn \
    --cases_json "$CASES" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 4 --measure_index 4 --step 5 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
    --blend_sched tgt0.9 --rollout_chunk_size 21 --seed 0 || FAILED=$((FAILED+1))
done
step "G1"; chk $PY_FB evaluation/r48_smoke_checks.py g1 --smoke_root $SMOKE

step "divergence (StreamEdit draft vs source: dino + lpips)"
for AM in dino_patch:se_dino lpips:se_lpips; do
  for T in 1 5; do
    $PY_FB evaluation/r31_divergence.py --edit_type $T --arm ${AM%%:*} --arm_dir_name ${AM##*:} --cases "$CASES" \
      --data_root "$DATA_ROOT" --target_root $SMOKE/frames/smoke/step04 --latent_dir $SMOKE/lat \
      --out_root $SMOKE/div --temporal_reduce mean --no-save_native --seed 0 || FAILED=$((FAILED+1))
  done
done

step "field (dino_SD) at rho_min 0.3 and 0"
chk $PY_FB evaluation/r48_field.py --mode subset --cases "$CASES" --hosts se --arms dino_SD --rho_min 0.3 \
    --attn_se $SMOKE/attn --div_root $SMOKE/div --out_root $SMOKE/field03
chk $PY_FB evaluation/r48_field.py --mode subset --cases "$CASES" --hosts se --arms dino_SD --rho_min 0 \
    --attn_se $SMOKE/attn --div_root $SMOKE/div --out_root $SMOKE/field0

stage3 () {  # method rho_dir out_root
  for T in 1 5; do
    $PY_SG evaluation/r31_stage3.py --edit_type $T --method "$1" --rho_dir "$2" \
      --data_root "$DATA_ROOT" --out_root "$3" --first_frame_edit_dir "$ANCHOR_ROOT" --vp_mode vp \
      --cases_json "$CASES" --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
      --rollout_chunk_size 21 --seed 0 || FAILED=$((FAILED+1))
  done
}

step "G4: constant rho=2 field vs global rho=2 run"
chk $PY_FB evaluation/r48_smoke_checks.py g4mk --smoke_root $SMOKE
stage3 r48_g4_const2 $SMOKE/g4_rho $SMOKE/g4_arm
for T in 1 5; do
  $PY_SG evaluation/run_fivebench.py --edit_type $T --method r48_g4_global2 --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" --blend_power 2 --data_root "$DATA_ROOT" --out_root $SMOKE/g4_global \
    --step 15 --fg_boost_factor 4 --flow_shift 1.0 --seed 0 --cases_json "$CASES" || FAILED=$((FAILED+1))
done
chk $PY_FB evaluation/r48_smoke_checks.py g4 --smoke_root $SMOKE

step "G5: dino_SD renders, rho_min 0.3 vs 0"
stage3 r48_g5_rmin03 $SMOKE/field03/se_dino_SD $SMOKE/g5_arm03
stage3 r48_g5_rmin0 $SMOKE/field0/se_dino_SD $SMOKE/g5_arm0
chk $PY_FB evaluation/r48_smoke_checks.py g5 --smoke_root $SMOKE

step "G2 + G3 (Wan-Edit, in process)"
chk $PY_FB evaluation/r48_smoke_checks.py g2g3 --smoke_root $SMOKE

echo
echo "[r48_smoke] summary:"; grep -h "^GATE " logs/r48_smoke_${SLURM_JOB_ID}.out | sort | uniq -c
NFAIL=$(grep -h "^GATE .* FAIL" logs/r48_smoke_${SLURM_JOB_ID}.out | wc -l)
echo "[r48_smoke] gate FAILs: ${NFAIL}  stage failures: ${FAILED}"
exit $(( (NFAIL > 0 || FAILED > 0) ? 1 : 0 ))
