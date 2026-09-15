#!/bin/bash
# R28 smoke -- prove the union metrics are wired correctly, and that adding them left the
# existing metrics untouched. evaluate.py is shared by R1/R3/R7/R20-R26, so the
# non-regression gate is as important as the new-behaviour gates.
#
# GATE1 (NONREG) -- evaluate.py WITHOUT --tgt_mask_dir/--fixed_union_dir must reproduce a
#   stored R26 arm CSV on the same clips, to within 1e-3 RELATIVE.
#
#   Originally written as bit-for-bit, which turned out to be untestable: the stored
#   references were produced on a different node, and the evaluator is only deterministic
#   WITHIN a node. Measured 2026-08-31 -- two runs of identical code on one node are
#   bit-identical (max|d| = 0.0), while both differ from the stored reference by 3.877e-05.
#   Additivity of the R28 change was proven separately and directly, by running a REVERTED
#   copy of evaluate.py alongside the patched one on the same node: bit-identical output,
#   so the offset is pre-existing environmental drift and not this task's doing.
#   1e-3 is the tolerance the control cross-check in r26/r28_summarize.py already uses
#   (R26's own control matched the R21 reference at 9.43e-05). A real regression would be
#   orders of magnitude larger, and the observed max|d| is printed either way so drift
#   stays visible rather than hidden by the tolerance.
#
# GATE2 (FIRES) -- with --tgt_mask_dir, the *_unedit_union columns must be NON-nan.
#   Without this, gate1 is also satisfied by a flag that is parsed and then ignored.
#
# GATE3 (DIFFERS) -- *_unedit_union must DIFFER from *_unedit_part on a clip whose object
#   moves (0034_cows). Gate2 alone is also satisfied by a union that silently equals
#   M_src, which is exactly the bug this task exists to fix.
#
# GATE4 (TYPERULES) -- edit5 target mask must CONTAIN M_src (addition unions with it) and
#   the edit6 target mask must be empty (removal grounds nothing). Both are R25's
#   construction rules; a detection-based implementation would violate them silently
#   because GroundingDINO never abstains.
#
# GATE5 (SUBSET) -- the fixed-union background must be a subset of every per-arm
#   background. Holds by construction; a failure means the reduction ran on the wrong
#   axis, which no count-based check would catch (the R22/R26 class of bug).
#
#SBATCH --job-name=r28_smoke
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r28_smoke_%j.out
#SBATCH --error=logs/r28_smoke_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# SINGLE ENVIRONMENT: streamgve. GroundingDINO + SAM2 were installed into it on
# 2026-08-31 (SAM-2 1.0 from facebookresearch/sam2 @2b90b9f5, plus hydra-core, iopath,
# portalocker, qwen-vl-utils) precisely so this task does not straddle two envs.
# streamgve already carried transformers 5.12.0, which exposes the `threshold` kwarg that
# r25_iou.py calls -- transformers 4.44 (addit, DGE) calls it `box_threshold` and raises
# TypeError, which is what killed job 965789. Nothing pre-existing was upgraded: the
# install added 4 packages and changed no version, and the WAN pipeline
# (WanVAEWrapper / causal_model / edit_causal_inference) was re-imported afterwards to
# confirm it. Absolute interpreter path, so no conda activation stacking is possible.
PY=~/anaconda3/envs/streamgve/bin/python

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ARM_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg2_taufg2_vp
ARM2_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg2_taufg10_vp
WORK=/projects/dataggen/outputs/five_bench/r28_smoke
CASES=evaluation/cases.json
STRIDE=8

# 0034_cows: the object moves across the clip, so a union that equals M_src is visible.
# 0011_lucia (edit5, addition) and 0042_gym-ball (edit6, removal) exercise the type rules.
MOVER=0034_cows

echo "[r28_smoke] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
$PY -c "import torch; print('[r28_smoke] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"

rm -rf "$WORK"; mkdir -p "$WORK" logs

# ---------------------------------------------------------------- masks
echo "=== dumping target masks for 2 arms on the smoke clips ==="
for A in "$ARM_ROOT" "$ARM2_ROOT"; do
  NAME=$(basename "$A")
  $PY evaluation/r28_target_masks.py \
    --cases "$CASES" --data_root "$DATA_ROOT" \
    --tgt_root "$A" --out_dir "$WORK/masks/$NAME" \
    --frame_stride $STRIDE --seed 0 \
    || { echo "GATE0-FAIL mask dump failed for $NAME"; exit 1; }
done

echo "=== building the fixed union over both arms ==="
$PY evaluation/r28_fixed_union.py \
  --mask_root "$WORK/masks" --data_root "$DATA_ROOT" \
  --cases "$CASES" --frame_stride $STRIDE -o "$WORK/masks/_fixed_union" \
  || { echo "GATE5-FAIL fixed-union reduction failed (see the subset assertion)"; exit 1; }
echo "GATE5-OK subset: fixed union contains every per-arm union (asserted in the reducer)"

# ---------------------------------------------------------------- GATE1
echo "=== GATE1: defaults unchanged (no --tgt_mask_dir) ==="
$PY evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics structure_distance lpips_unedit_part ssim_unedit_part \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "$DATA_ROOT"/edit_prompt/edit2_FiVE.json \
  --tgt_methods "$ARM_ROOT" --tgt_layout edit_video \
  --cases_json "$CASES" --frame_stride $STRIDE \
  --result_path "$WORK/gate1_new.csv" > "$WORK/gate1.log" 2>&1
$PY - "$WORK" <<'GATE1PY'
import csv, glob, sys
work = sys.argv[1]
TOL = 1e-3   # relative; see the GATE1 note in the header
f = sorted(glob.glob(work + "/edit2_FiVE_gate1_new_frame_stride8.csv"))
if not f:
    print("GATE1-FAIL nonreg: no CSV written -- evaluate.py did not run to completion")
    raise SystemExit
def col(p, m):
    r = list(csv.reader(open(p))); h = [x.split("|")[-1] for x in r[0]]; i = h.index(m)
    return [float(x[i]) for x in r[1:]]
ref_path = "evaluation/csv/edit2_FiVE_r26_taubg2_taufg2_vp_frame_stride8.csv"
worst = 0.0
for m in ("lpips_unedit_part", "ssim_unedit_part", "structure_distance"):
    a, b = col(f[0], m), col(ref_path, m)
    if len(a) != len(b):
        print(f"GATE1-FAIL nonreg: {m} row count {len(a)} != {len(b)}"); raise SystemExit
    rel = max(abs(x - y) / max(abs(x), abs(y), 1e-12) for x, y in zip(a, b))
    worst = max(worst, rel)
    print(f"    {m}: max relative diff {rel:.3e}")
if worst <= TOL:
    print(f"GATE1-OK nonreg: existing metrics match the stored r26 control within {TOL:g} "
          f"(worst {worst:.3e}); the R28 change is additive")
else:
    print(f"GATE1-FAIL nonreg: worst relative diff {worst:.3e} > {TOL:g}. "
          "The shared evaluator moved -- every stored reference is at risk. STOP.")
GATE1PY

# ---------------------------------------------------------------- GATE2 + GATE3
echo "=== GATE2/GATE3: union metrics fire, and differ from _unedit_part ==="
$PY evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics lpips_unedit_part ssim_unedit_part \
            lpips_unedit_union ssim_unedit_union \
            lpips_unedit_union_fixed ssim_unedit_union_fixed \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "$DATA_ROOT"/edit_prompt/edit2_FiVE.json \
  --tgt_methods "$ARM_ROOT" --tgt_layout edit_video \
  --tgt_mask_dir "$WORK/masks/$(basename $ARM_ROOT)" \
  --fixed_union_dir "$WORK/masks/_fixed_union" \
  --cases_json "$CASES" --frame_stride $STRIDE \
  --result_path "$WORK/gate2.csv" > "$WORK/gate2.log" 2>&1

$PY - "$WORK" "$MOVER" <<'PY'
import csv, glob, sys, json
work, mover = sys.argv[1], sys.argv[2]
f = sorted(glob.glob(work + "/edit2_FiVE_gate2_frame_stride8.csv"))
if not f:
    print("GATE2-FAIL no CSV written"); raise SystemExit
r = list(csv.reader(open(f[0]))); hdr = [h.split('|')[-1] for h in r[0]]
rows = r[1:]
def col(name):
    i = hdr.index(name)
    return [x[i] for x in rows]
for m in ("lpips_unedit_union", "ssim_unedit_union",
          "lpips_unedit_union_fixed", "ssim_unedit_union_fixed"):
    vals = col(m)
    if all(v in ("", "nan", "N/A") for v in vals):
        print(f"GATE2-FAIL fires: {m} is nan on every clip -- the flag is parsed then ignored")
        break
else:
    print("GATE2-OK fires: all four union columns are populated")

# gate3 on the moving-object clip: find its row via the cases order
cases = [c for c in json.load(open("evaluation/cases.json")) if int(c["edit_type"]) == 2]
names = [c["video_name"] for c in cases]
if mover in names and len(rows) == len(names):
    k = names.index(mover)
    part, uni = col("lpips_unedit_part")[k], col("lpips_unedit_union")[k]
    fixed = col("lpips_unedit_union_fixed")[k]
    if part != uni and part != fixed:
        print(f"GATE3-OK differs: on {mover} part={part} union={uni} fixed={fixed}")
    else:
        print(f"GATE3-FAIL differs: union == part on {mover} "
              "-- the union silently equals M_src, which is the bug this task fixes")
else:
    print(f"GATE3-FAIL differs: could not locate {mover} ({len(rows)} rows, {len(names)} cases)")
PY

# ---------------------------------------------------------------- GATE4
echo "=== GATE4: per-type construction rules ==="
$PY - "$WORK" <<'PY'
import json, os, sys
import numpy as np
sys.path.insert(0, "evaluation")
from r28_target_masks import load_npz, load_src_masks, list_images
work = sys.argv[1]
arm = os.path.join(work, "masks", "taubg2_taufg2_vp")
data = os.path.expanduser("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
cases = json.load(open("evaluation/cases.json"))
ok = True
for c in cases:
    t, v = int(c["edit_type"]), c["video_name"]
    p = os.path.join(arm, f"edit{t}", f"{v}.npz")
    if not os.path.exists(p):
        continue
    m = load_npz(p, expect_stride=8)
    if t == 6:
        if m.any():
            print(f"GATE4-FAIL typerules: edit6 {v} target mask is not empty"); ok = False
    if t == 5:
        src_paths = list_images(os.path.join(data, "images", v))[::8][: m.shape[0]]
        from PIL import Image
        w, h = Image.open(src_paths[0]).size
        s = load_src_masks(os.path.join(data, "bmasks", v), src_paths, (h, w))
        if s is not None and not np.all(s[: m.shape[0]] <= m):
            print(f"GATE4-FAIL typerules: edit5 {v} target mask does not contain M_src"); ok = False
print("GATE4-OK typerules: edit5 contains M_src, edit6 empty" if ok else "GATE4-FAIL typerules")
PY

echo "--- r28_smoke done ---"
