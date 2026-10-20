#!/bin/bash
# R41 smoke: 4 gates before the full-bench binary renders, at the N chosen at gate-draft.
#
#   G1  continuous path unchanged (CPU): r31_rho_map.py --mapping budget_linear, rebuilt from
#       the chosen draft's fields for EVERY combo (419 clips each, 4 in parallel), equals the
#       stored continuous rho at the same N (np.array_equal). R41 added --mapping binary to
#       that file; the continuous arms the binary ones are compared to must be untouched.
#   G2  binary maps are what they claim (CPU): every r41_rho/n{N}/{arm} npz has rho values in
#       {2, 50}, (rho == 50) == (m >= thresh) against the field it was built from, and its
#       recorded thresh / mapping match the arm.
#   G3  continuous render parity (GPU): r31_stage3.py on 2 clips with the LAST combo's stored
#       continuous rho == its stored continuous render at the same N (sha256 of every step14
#       PNG). Proves today's tree -- including R38's uncommitted pipeline edits -- reproduces
#       the continuous arm, so binary vs continuous differ only by the mapping.
#   G4  binary render (GPU): the same 2 clips with that combo's binary rho at theta 0.5 (first
#       cut in THETA_ORDER) render the same frame count as G3 and are NOT byte-identical to it.
#
# 2 clips, as r38_smoke.sh: 0001_bus (edit1, single window) and 0034_cows (edit2, 24 latent
# frames -- the multi-window clip that exercises the rollout overlap). r31_stage3.py re-seeds
# before EVERY pair, so a --cases_json subset renders exactly the full-bench frames.
#
# Render flags: r35_stage3.sh / r38_stage3.sh verbatim (identical between the two).
#SBATCH --job-name=r41_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r41_smoke_%j.out
#SBATCH --error=logs/r41_smoke_%j.err
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO" || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# exits here if DRAFT_N is unset; R41_CONFIG is for synthetic tests only (see r41_config.sh)
source "${R41_CONFIG:-slurm_scripts/five_bench/r41_config.sh}"

LAST=${COMBOS[$(( ${#COMBOS[@]} - 1 ))]}
BIN_THETA=${THETA_ORDER[0]}
BIN_ARM="${LAST}_t${BIN_THETA/./}"
CONT_METHOD="${CONT_PREFIX}_${SRC_NAME[$LAST]}"

# Fresh root: a stale PNG from an earlier smoke would make a gate compare old output.
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT/logs"
echo "[r41_smoke] job=${SLURM_JOB_ID:-local} node=$(hostname) out=${SMOKE_ROOT}"
echo "[r41_smoke] G3/G4 combo=${LAST} continuous=${CONT_METHOD} binary=${BIN_ARM}"
nvidia-smi -L 2>&1 | head -2

# ---- G1 (CPU): rebuild budget-linear rho for every combo, 4 in parallel.
G1_PIDS=()
for C in "${COMBOS[@]}"; do
  python evaluation/r31_rho_map.py \
    --div_root "$DIV_ABS_ROOT/${SRC_NAME[$C]}" \
    -o "$SMOKE_ROOT/rho_bl/$C" \
    --tau_min 2 --tau_max 50 --step 15 --flow_shift 1.0 \
    > "$SMOKE_ROOT/logs/g1_${C}.log" 2>&1 &
  G1_PIDS+=($!)
done
G1_RUN_FAIL=0
for P in "${G1_PIDS[@]}"; do wait "$P" || G1_RUN_FAIL=$((G1_RUN_FAIL+1)); done
echo "[r41_smoke] G1 rebuilds finished, ${G1_RUN_FAIL} failed"

# ---- G3 / G4 (GPU): one r31_stage3.py call per edit type (it refuses a cases json with no
# case for its --edit_type), final index only.
CASES=${TMPDIR:-/tmp}/r41_smoke_cases_${SLURM_JOB_ID:-local}.json
cat > "$CASES" <<'EOF'
[{"video_name": "0001_bus", "edit_type": 1},
 {"video_name": "0034_cows", "edit_type": 2}]
EOF

RENDER_FAIL=0
render () {
  local METHOD=$1 RHO_DIR=$2 T
  for T in 1 2; do
    echo "--- [r41_smoke] ${METHOD} (rho_dir=${RHO_DIR}) / edit${T} ---"
    python evaluation/r31_stage3.py \
      --edit_type "$T" \
      --method "$METHOD" \
      --rho_dir "$RHO_DIR" \
      --data_root "$DATA_ROOT" \
      --out_root "$SMOKE_ROOT/frames" \
      --cases_json "$CASES" \
      --vp_mode vp \
      --first_frame_edit_dir "$ANCHOR_ROOT" \
      --dump_steps 14 \
      --step 15 \
      --flow_shift 1.0 \
      --fg_boost_factor 4 \
      --blend_power 2 \
      --rollout_chunk_size 21 \
      --seed 0 \
      2>&1 | tee -a "$SMOKE_ROOT/logs/${METHOD}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[r41_smoke] FAILED ${METHOD} / edit${T}"; RENDER_FAIL=$((RENDER_FAIL+1)); }
  done
}
render continuous "$CONT_RHO_ROOT/${SRC_NAME[$LAST]}"
render binary     "$RHO_ROOT/$BIN_ARM"

# ---- gates
COMBOS_CSV=$(IFS=,; echo "${COMBOS[*]}")
SRC_CSV=$(for C in "${COMBOS[@]}"; do printf '%s,' "${SRC_NAME[$C]}"; done | sed 's/,$//')
ARMS_CSV=$(for A in "${ARMS[@]}"; do printf '%s:%s:%s,' "$A" "${ARM_COMBO[$A]}" "${ARM_THETA[$A]}"; done | sed 's/,$//')
python - "$SMOKE_ROOT" "$DIV_ABS_ROOT" "$CONT_RHO_ROOT" "$RHO_ROOT" \
         "$CONT_ARMS_ROOT/$CONT_METHOD" "$COMBOS_CSV" "$SRC_CSV" "$ARMS_CSV" \
         "$G1_RUN_FAIL" "$RENDER_FAIL" <<'PY'
import hashlib, sys
from pathlib import Path
import numpy as np

smoke, div_abs, cont_rho, rho_root, cont_render = map(Path, sys.argv[1:6])
combos, srcs = sys.argv[6].split(","), sys.argv[7].split(",")
src = dict(zip(combos, srcs))
arms = [a.split(":") for a in sys.argv[8].split(",")]
g1_run_fail, render_fail = int(sys.argv[9]), int(sys.argv[10])
EXPECT = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
CLIPS = [(1, "0001_bus"), (2, "0034_cows")]
res = {}

def tree(root):
    files = sorted(root.glob("edit*/*.npz"))
    per = {t: sum(f.parent.name == f"edit{t}" for f in files) for t in EXPECT}
    return files, per == EXPECT

# G1
ok = g1_run_fail == 0
for c in combos:
    files, complete = tree(smoke / "rho_bl" / c)
    diff = sum(not np.array_equal(np.load(f)["rho"],
                                  np.load(cont_rho / src[c] / f.parent.name / f.name)["rho"])
               for f in files)
    print(f"  G1 {c} ({src[c]}): {len(files)} clips rebuilt, {diff} differ from stored")
    ok &= complete and diff == 0
res["G1"] = ok

# G2
ok = True
for arm, combo, theta in arms:
    files, complete = tree(rho_root / arm)
    bad = 0
    for f in files:
        z = np.load(f)
        m = np.load(div_abs / src[combo] / f.parent.name / f.name)["m"]
        rho = z["rho"]
        good = (set(np.unique(rho).tolist()) <= {2.0, 50.0}
                and np.array_equal(rho == 50.0, m >= float(z["thresh"]))
                and abs(float(z["thresh"]) - float(theta)) < 1e-6
                and str(z["mapping"]) == "binary")
        bad += not good
    print(f"  G2 {arm}: {len(files)} clips, {bad} bad")
    ok &= complete and bad == 0
res["G2"] = ok

# G3 / G4
def frames(root, T, v):
    d = root / "step14" / f"edit{T}" / v
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.png"))}

g3 = g4 = render_fail == 0
for T, v in CLIPS:
    fc, fs = frames(smoke / "frames" / "continuous", T, v), frames(cont_render, T, v)
    fb = frames(smoke / "frames" / "binary", T, v)
    n_diff = sum(fc.get(k) != fs.get(k) for k in set(fc) | set(fs))
    n_bin_diff = sum(fb.get(k) != fc.get(k) for k in set(fb) | set(fc))
    print(f"  G3 edit{T}/{v}: smoke {len(fc)} vs stored {len(fs)} frames, {n_diff} differ")
    print(f"  G4 edit{T}/{v}: binary {len(fb)} vs continuous {len(fc)} frames, "
          f"{n_bin_diff} differ")
    g3 &= bool(fc) and fc == fs
    g4 &= bool(fb) and len(fb) == len(fc) and fb != fc
res["G3"], res["G4"] = g3, g4

for g, ok in res.items():
    print(f"GATE {g} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if all(res.values()) else 1)
PY
STATUS=$?
echo "[r41_smoke] gates exit=${STATUS}"
exit $STATUS
