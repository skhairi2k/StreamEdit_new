#!/bin/bash
# R45 smoke: the LongLive port and the --backbone drivers, before any full-bench run.
#
#   G0  run_fivebench --backbone sf, rho=2, VP, 0001_bus/edit1 == R36 r36_rho2_vp (sha256 of
#       every PNG). Licenses the driver refactor: the SF path must be byte-identical.
#   G1  stock LL == ported LL (routing off), latents torch.equal, 0021_landing/edit1 +
#       0025_baseball/edit2 (clips where stock and case-insensitive trigger matching agree).
#   G2  uniform rho=2 field == scalar blend_power 2 on LL.
#   G3  tgt0.9 == first2 at N=15 on LL; echoed vectors N=15 [0,0,1x13], N=5 [0,1,1,1,1].
#   G4  longest FiVE clip at N=30 completes on L40S; peak VRAM + s/frame logged.
#   G5  LL NFE accounting: draft(N) + render(15) vs single(15+N), N in {5, 15}.
#   G6  run_fivebench --backbone ll on 0001_bus/edit1: frame count == R36 (SF) == 69.
#
# G1-G5 run in evaluation/r45_smoke_parity.py (two processes: stock, ported). The stock
# reference is `git archive HEAD LongLive_StreamEdit` (pristine code) with the R45 text
# encoder + checkpoint-load files copied in and the weights symlinked, so G1 isolates the
# routing port.
#
#SBATCH --job-name=r45_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r45_smoke_%j.out
#SBATCH --error=logs/r45_smoke_%j.err
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors
R36_REF=$D/r36_rho_sweep/r36_rho2_vp/edit1/0001_bus
SMOKE_ROOT=$D/r45_smoke
LL=$REPO/LongLive_StreamEdit

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh root: a stale output from an earlier smoke would make a gate compare old files.
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"

echo "[r45_smoke] job=${SLURM_JOB_ID} node=$(hostname) out=${SMOKE_ROOT}"
nvidia-smi -L 2>&1 | head -2

for f in "$LL/longlive_models/models/longlive_base.pt" "$LL/longlive_models/models/lora.pt"; do
  [ -f "$f" ] || { echo "[r45_smoke] missing weights $f"; exit 1; }
done
[ -d "$R36_REF" ] || { echo "[r45_smoke] missing R36 reference $R36_REF"; exit 1; }

CASES=$SMOKE_ROOT/cases_bus.json
echo '[{"video_name": "0001_bus", "edit_type": 1}]' > "$CASES"

FAILED=0

# ---- G0: SF through the refactored driver == R36 ------------------------------------
python evaluation/run_fivebench.py --backbone sf \
  --edit_type 1 --method r45_smoke_sf_rho2 --cases_json "$CASES" \
  --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --blend_power 2 \
  --data_root "$DATA_ROOT" --out_root "$SMOKE_ROOT/fivebench" \
  --step 15 --fg_boost_factor 4 --flow_shift 1.0 --seed 0 \
  || { echo "[r45_smoke] run_fivebench sf exit $?"; FAILED=$((FAILED + 1)); }
G0_DIR=$SMOKE_ROOT/fivebench/r45_smoke_sf_rho2/edit1/0001_bus
if diff <(cd "$R36_REF" && sha256sum *.png) <(cd "$G0_DIR" 2>/dev/null && sha256sum *.png) > /dev/null; then
  echo "GATE0 PASS  sf via --backbone == R36 r36_rho2_vp on 0001_bus ($(ls "$G0_DIR" | wc -l) PNGs)"
else
  echo "GATE0 FAIL  sf via --backbone differs from R36 r36_rho2_vp on 0001_bus"; FAILED=$((FAILED + 1))
fi

# ---- G6: LL end to end through run_fivebench, frame parity with SF ------------------
python evaluation/run_fivebench.py --backbone ll \
  --edit_type 1 --method r45_smoke_ll_rho2 --cases_json "$CASES" \
  --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --blend_power 2 \
  --data_root "$DATA_ROOT" --out_root "$SMOKE_ROOT/fivebench" \
  --step 15 --fg_boost_factor 2 --flow_shift 1.0 --seed 0 \
  || { echo "[r45_smoke] run_fivebench ll exit $?"; FAILED=$((FAILED + 1)); }
N_SF=$(ls "$R36_REF"/*.png | wc -l)
N_LL=$(ls "$SMOKE_ROOT/fivebench/r45_smoke_ll_rho2/edit1/0001_bus"/*.png 2>/dev/null | wc -l)
# find_closest_num_frame (identical in both builds' inference_edit_streamedit.py), inlined
N_SRC=$(python -c "
import os
x = len(os.listdir(os.path.expanduser('$DATA_ROOT/images/0001_bus')))
m = (x + 3) // 12
while m > 0 and 12 * m - 3 > x: m -= 1
print(12 * m - 3)")
if [ "$N_LL" -eq "$N_SF" ] && [ "$N_SF" -eq "$N_SRC" ]; then
  echo "GATE6 PASS  frames LL=$N_LL SF=$N_SF find_closest_num_frame=$N_SRC on 0001_bus"
else
  echo "GATE6 FAIL  frames LL=$N_LL SF=$N_SF find_closest_num_frame=$N_SRC on 0001_bus"; FAILED=$((FAILED + 1))
fi

# ---- stock LL reference: pristine code + R45 encoder/ckpt-load files + weights -------
REF=$SMOKE_ROOT/ll_ref
mkdir -p "$REF"
git archive HEAD LongLive_StreamEdit | tar -x -C "$REF"
REF_LL=$REF/LongLive_StreamEdit
cp "$LL/wan/modules/t5.py"               "$REF_LL/wan/modules/t5.py"
cp "$LL/utils/wan_wrapper.py"            "$REF_LL/utils/wan_wrapper.py"
cp "$LL/inference_edit_streamedit.py"    "$REF_LL/inference_edit_streamedit.py"
ln -s "$LL/longlive_models" "$REF_LL/longlive_models"
ln -s "$LL/wan_models"      "$REF_LL/wan_models"
# The reference must NOT carry the routing port: its pipeline has no rho_frames kwarg.
if grep -q "rho_frames" "$REF_LL/pipeline/edit_causal_inference.py"; then
  echo "[r45_smoke] reference pipeline already carries the R45 port -- HEAD is not stock"; exit 1
fi

# ---- G1-G5 ---------------------------------------------------------------------------
python evaluation/r45_smoke_parity.py run --root "$REF_LL" --tag stock  --out "$SMOKE_ROOT/parity" \
  || { echo "[r45_smoke] parity stock exit $?"; FAILED=$((FAILED + 1)); }
python evaluation/r45_smoke_parity.py run --root "$LL"     --tag ported --out "$SMOKE_ROOT/parity" \
  || { echo "[r45_smoke] parity ported exit $?"; FAILED=$((FAILED + 1)); }
python evaluation/r45_smoke_parity.py gates --out "$SMOKE_ROOT/parity" \
  || FAILED=$((FAILED + 1))

echo "[r45_smoke] done, failures: ${FAILED}"
exit $(( FAILED > 0 ? 1 : 0 ))
