#!/bin/bash
# R61 smoke: StreamEdit on LongLive + FiVE-Bench-Correct, 2 clips, before the full sweeps / se_max chain.
#
# Usage: sbatch slurm_scripts/five_bench/r61_smoke.sh
# Clips: 0001_bus (edit 1, anchor changed by the keep-pose suffix) and 0042_gym-ball (edit 6, target
# prompt changed -- removal by negation fixed). Everything new runs in $D/r61_smoke/job<id>/.
#   G1  benchmark freeze: per-type pair counts of evaluation/five_bench_correct == (98 100 99 99 9 10),
#       the pair set matches evaluation/r43_anchor_manifest_correct.json exactly, every listed anchor
#       PNG under $D/anchors_r43_correct exists.
#   G2  sweep path: run_fivebench.py --backbone ll rho=2 N=15 with --edit_json_dir pointed at
#       evaluation/five_bench_correct and anchors_r43_correct, on the 2 smoke clips; the rendered
#       frames differ from $D/r45_sweep/n15/ll_rho2_n15_vp (official-FiVE anchors/prompts), and the
#       Correct target prompt of 0042_gym-ball (printed by this script, not by run_fivebench.py) is
#       readable in the log next to the call that used it.
#   G3  se_max chain on the 2 pairs: r31_stage1.py (Correct draft) -> r31_divergence.py (dino, lpips)
#       -> r49_field.py -> r31_stage3.py (Correct render); the Correct target prompt of 0042_gym-ball
#       is printed next to both stage calls; the field's q is finite in [0, 1]; the render's frame
#       count matches G2's.
#   G4  default path unchanged: r31_stage1.py on 0001_bus edit1 WITHOUT --edit_json_dir (so it reads
#       data_root/edit_prompt, the official FiVE files) and the official anchors ($D/anchors) -- its
#       latents are bit-identical to R55's smoke (job 1057789, run "on"), confirming the new flag's
#       default did not change any existing behaviour.
# Prints each gate PASS/FAIL and `[r61_smoke] failures: N`.
#SBATCH --job-name=r61_smoke
#SBATCH --partition=L40S
#SBATCH --nodelist=node39
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/r61_smoke_%j.out
#SBATCH --error=logs/r61_smoke_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
CORRECT_DIR=evaluation/five_bench_correct
MANIFEST=evaluation/r43_anchor_manifest_correct.json
ANCHOR_ROOT=$D/anchors_r43_correct
OFFICIAL_ANCHOR_ROOT=$D/anchors
CASES=evaluation/r61_smoke_cases.json
R55_ON_REF=$D/r55_smoke/job1057789/on/latents
R45_REF=$D/r45_sweep/n15/ll_rho2_n15_vp
RUN=$D/r61_smoke/job${SLURM_JOB_ID:-local}
PY_SG=~/anaconda3/envs/streamgve/bin/python
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$RUN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "[r61_smoke] job=${SLURM_JOB_ID:-local} node=$(hostname) run=${RUN}"; nvidia-smi -L 2>&1 | head -2

FAILED=0
gate() {  # $1 = gate name, $2 = exit code of its check
  if [ "$2" -eq 0 ]; then echo "[r61_smoke] $1 PASS"; else echo "[r61_smoke] $1 FAIL"; FAILED=$((FAILED + 1)); fi
}

# ---- preflight: r45_sweep is a /store link; this job is pinned to node39 for it
if [ ! -d "$R45_REF/edit1" ]; then
  echo "[r61_smoke] FATAL: ${R45_REF}/edit1 missing or unreadable (node=$(hostname)); /store is per-node"
  exit 1
fi

# ---- the 2 smoke clips, in the cases.json shape
echo "[r61_smoke] cases: 0001_bus/edit1, 0042_gym-ball/edit6 (from ${CASES})"

# ---- G1: benchmark freeze
$PY_FB - "$CORRECT_DIR" "$MANIFEST" "$ANCHOR_ROOT" <<'EOF'
import json, sys
from pathlib import Path
correct_dir, manifest_path, anchor_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).expanduser()
EXPECTED = {1: 98, 2: 100, 3: 99, 4: 99, 5: 9, 6: 10}
bad = []

json_pairs = set()
for t, exp in EXPECTED.items():
    f = correct_dir / f"edit{t}_FiVE.json"
    entries = json.loads(f.read_text())
    if len(entries) != exp:
        bad.append(f"{f}: {len(entries)} entries, expected {exp}")
    for e in entries:
        json_pairs.add((t, e["video_name"]))

manifest = json.loads(manifest_path.read_text())
manifest_pairs = {(int(c["edit_type"]), c["video_name"]) for c in manifest}
if json_pairs != manifest_pairs:
    only_json = sorted(json_pairs - manifest_pairs)[:5]
    only_manifest = sorted(manifest_pairs - json_pairs)[:5]
    bad.append(f"pair set mismatch: {len(json_pairs)} in {correct_dir}/edit*_FiVE.json vs "
               f"{len(manifest_pairs)} in {manifest_path} (only in json: {only_json}, "
               f"only in manifest: {only_manifest})")

missing_anchors = []
for c in manifest:
    p = anchor_root / f"edit{c['edit_type']}" / f"{c['video_name']}.png"
    if not p.exists():
        missing_anchors.append(str(p))
if missing_anchors:
    bad.append(f"{len(missing_anchors)} anchor PNG(s) missing, e.g. {missing_anchors[:5]}")

print(f"[r61_smoke] G1: {len(json_pairs)} Correct pairs, {len(manifest)} manifest entries, "
      f"anchors checked under {anchor_root}")
for b in bad:
    print(f"[r61_smoke] G1 PROBLEM: {b}")
sys.exit(1 if bad else 0)
EOF
gate "G1 benchmark freeze (counts, pair set, anchors)" $?

# ---- print the Correct target prompt of 0042_gym-ball once, read by G2 and G3 below
CORRECT_PROMPT=$($PY_FB -c "
import json
e = [x for x in json.load(open('$CORRECT_DIR/edit6_FiVE.json')) if x['video_name'] == '0042_gym-ball'][0]
print(e['target_prompt'])
")
echo "[r61_smoke] Correct target_prompt for edit6/0042_gym-ball: ${CORRECT_PROMPT}"

# ---- G2: sweep path, rho=2 N=15, LongLive VP, on the Correct benchmark
SWEEP_OUT=$RUN/sweep
for T in 1 6; do
  echo "--- [r61_smoke] G2 sweep ll rho=2 N=15 edit${T} (Correct prompts: ${CORRECT_PROMPT}) $(date '+%F %T') ---"
  $PY_SG evaluation/run_fivebench.py --backbone ll --edit_type "$T" --method ll_rho2_n15 \
    --edit_json_dir "$CORRECT_DIR" --cases_json "$CASES" \
    --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --blend_power 2 \
    --data_root "$DATA_ROOT" --out_root "$SWEEP_OUT" \
    --step 15 --fg_boost_factor 2 --flow_shift 1.0 --seed 0 || { echo "[r61_smoke] G2 FAILED edit${T}"; FAILED=$((FAILED + 1)); }
done
RC2=0
$PY_FB - "$SWEEP_OUT/ll_rho2_n15" "$R45_REF" <<'EOF' || RC2=1
import sys
from pathlib import Path
smoke, ref = Path(sys.argv[1]), Path(sys.argv[2])
clips = [("1", "0001_bus"), ("6", "0042_gym-ball")]
n_same = 0; n_checked = 0
for t, name in clips:
    sd, rd = smoke / f"edit{t}" / name, ref / f"edit{t}" / name
    if not sd.is_dir() or not rd.is_dir():
        print(f"  {name} edit{t}: missing render dir ({'smoke' if not sd.is_dir() else 'ref'})")
        continue
    sf, rf = sorted(sd.glob("*.png")), sorted(rd.glob("*.png"))
    n_checked += 1
    if len(sf) != len(rf):
        print(f"  {name} edit{t}: frame count differs ({len(sf)} vs {len(rf)}), as expected for a different benchmark")
        continue
    same = all(a.read_bytes() == b.read_bytes() for a, b in zip(sf, rf))
    n_same += same
    print(f"  {name} edit{t}: frames {'IDENTICAL to the official-FiVE render (unexpected)' if same else 'differ from the official-FiVE render (expected)'}")
sys.exit(1 if n_checked and n_same == n_checked else 0)
EOF
gate "G2 sweep path (Correct prompts/anchors render differently from official-FiVE)" $RC2

# ---- G3: se_max chain on the 2 pairs, Correct prompts/anchors throughout
DRAFT_OUT=$RUN/draft
ATTN_OUT=$RUN/attn
DIV_OUT=$RUN/div
FIELD_OUT=$RUN/field
ARM_OUT=$RUN/arms
RC3=0
for T in 1 6; do
  echo "--- [r61_smoke] G3 draft (Correct) edit${T} (Correct prompts: ${CORRECT_PROMPT}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage1.py --backbone ll --edit_type "$T" --method n5_tgt09 \
    --data_root "$DATA_ROOT" --edit_json_dir "$CORRECT_DIR" \
    --out_root "$DRAFT_OUT/frames" --latent_dir "$DRAFT_OUT/latents" --attn_out_dir "$ATTN_OUT" \
    --cases_json "$CASES" --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" \
    --dump_steps 4 --measure_index 4 --step 5 --flow_shift 1.0 --fg_boost_factor 2 --blend_power 2 \
    --blend_sched tgt0.9 --seed 0 || { echo "[r61_smoke] G3 FAILED draft edit${T}"; RC3=1; }
done
for NAME_ARM in "se_dino dino_patch" "se_lpips lpips"; do
  read -r NAME ARM <<< "$NAME_ARM"
  for T in 1 6; do
    $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
      --data_root "$DATA_ROOT" --target_root "$DRAFT_OUT/frames/n5_tgt09/step04" --latent_dir "$DRAFT_OUT/latents" \
      --out_root "$DIV_OUT" --temporal_reduce mean --no-save_native --seed 0 || { echo "[r61_smoke] G3 FAILED ${NAME} edit${T}"; RC3=1; }
  done
done
$PY_FB evaluation/r49_field.py --hosts se --cases "$CASES" --attn_se "$ATTN_OUT" --div_root "$DIV_OUT" --out_root "$FIELD_OUT" \
  || { echo "[r61_smoke] G3 FAILED r49_field.py"; RC3=1; }
for T in 1 6; do
  echo "--- [r61_smoke] G3 render (Correct) edit${T} (Correct prompts: ${CORRECT_PROMPT}) $(date '+%F %T') ---"
  $PY_SG evaluation/r31_stage3.py --backbone ll --edit_type "$T" --method max_ll --rho_dir "$FIELD_OUT/se_max" \
    --data_root "$DATA_ROOT" --edit_json_dir "$CORRECT_DIR" --out_root "$ARM_OUT" \
    --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --cases_json "$CASES" \
    --dump_steps 14 --step 15 --flow_shift 1.0 --fg_boost_factor 2 --blend_power 2 --seed 0 \
    || { echo "[r61_smoke] G3 FAILED render edit${T}"; RC3=1; }
done
$PY_FB - "$FIELD_OUT/se_max" "$ARM_OUT/max_ll/step14" "$SWEEP_OUT/ll_rho2_n15" <<'EOF' || RC3=1
import sys
from pathlib import Path
import numpy as np
field, arms, sweep = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
clips = [("1", "0001_bus"), ("6", "0042_gym-ball")]
bad = 0
for t, name in clips:
    f = field / f"edit{t}" / f"{name}.npz"
    if not f.exists():
        print(f"  {name} edit{t}: field npz missing"); bad += 1; continue
    q = np.load(f)["q"]
    ok = np.isfinite(q).all() and q.min() >= 0 and q.max() <= 1
    print(f"  {name} edit{t}: q in [{q.min():.3f}, {q.max():.3f}] -- {'ok' if ok else 'BAD'}")
    bad += not ok
    render_dir = arms / f"edit{t}" / name
    sweep_dir = sweep / f"edit{t}" / name
    if not render_dir.is_dir() or not sweep_dir.is_dir():
        print(f"  {name} edit{t}: missing render dir ({'arms' if not render_dir.is_dir() else 'sweep'})"); bad += 1; continue
    n_arm, n_sweep = len(list(render_dir.glob("*.png"))), len(list(sweep_dir.glob("*.png")))
    if n_arm != n_sweep:
        print(f"  {name} edit{t}: frame count {n_arm} != G2's {n_sweep}"); bad += 1
    else:
        print(f"  {name} edit{t}: frame count {n_arm} matches G2")
sys.exit(1 if bad else 0)
EOF
gate "G3 se_max chain (Correct prompts, finite q, frame count == G2)" $RC3

# ---- G4: default path unchanged -- no --edit_json_dir, official anchors, vs R55's stored smoke
G4_OUT=$RUN/g4
$PY_SG evaluation/r31_stage1.py --backbone ll --edit_type 1 --method n5_tgt09 \
  --data_root "$DATA_ROOT" --out_root "$G4_OUT/frames" --latent_dir "$G4_OUT/latents" \
  --cases_json "$CASES" --vp_mode vp --first_frame_edit_dir "$OFFICIAL_ANCHOR_ROOT" \
  --dump_steps 4 --measure_index 4 --step 5 --flow_shift 1.0 --fg_boost_factor 2 --blend_power 2 \
  --blend_sched tgt0.9 --seed 0
RC4=$?
$PY_FB - "$G4_OUT/latents" "$R55_ON_REF" <<'EOF' || RC4=1
import sys, numpy as np
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
f = a / "edit1" / "0001_bus.npz"
g = b / "edit1" / "0001_bus.npz"
if not f.exists() or not g.exists():
    print(f"  missing npz ({'this run' if not f.exists() else 'R55 reference'})"); sys.exit(1)
x, y = np.load(f), np.load(g)
bad = 0
for k in ("z_trg", "z_src"):
    eq = np.array_equal(x[k], y[k]); bad += not eq
    print(f"  0001_bus edit1 {k}: {'identical' if eq else 'DIFFERS'}")
sys.exit(1 if bad else 0)
EOF
gate "G4 default path unchanged (vs R55 smoke job 1057789/on)" $RC4

echo "[r61_smoke] failures: ${FAILED}"
exit $(( FAILED > 0 ))
