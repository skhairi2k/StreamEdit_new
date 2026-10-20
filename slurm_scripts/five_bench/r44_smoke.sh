#!/bin/bash
# R44 smoke: 3 gates for the TEXT-ONLY (novp) routing pipeline before any full-bench run.
#
#   G1  routed render == uniform render, text-only. r31_stage3.py --vp_mode novp on a
#       CONSTANT rho = 2 field vs run_fivebench.py --vp_mode novp --blend_power 2:
#       sha256 of every final PNG. Licenses comparing R44's routed arms to R44's own
#       uniform rho curve (same sampler end to end). R26 job 960461 proved this only for
#       a two-level field WITH VP; the continuous R31 field path was never checked against
#       the scalar path. If frames differ the gate prints max|diff| / mean|diff| (uint8)
#       and mean LPIPS-alex and FAILS -- the user decides whether the gap is acceptable.
#   G2  r31_stage1.py --vp_mode novp echoes the expected blender_rate vector:
#       N=15 zero [1 x15], tgt0.9 [0,0,1 x13]; N=5 zero [1,1,1,1,1], tgt0.9 [0,1,1,1,1].
#   G3  no anchor leak: every run logs vp_mode=novp, every stage-1 npz records
#       vp_mode == novp, and the N=15 tgt0.9 text-only draft differs (frames AND z_trg)
#       from R35's VP first2 draft of the same clip.
#
# 2 clips: 0001_bus (edit1, single window) and 0034_cows (edit2, multi-window -- the only
# one that exercises the rollout overlap; R26 lesson). Every driver re-seeds per pair, so
# a --cases_json subset renders exactly what a full-bench run would for these clips.
# No --first_frame_edit_dir anywhere: --vp_mode novp is passed explicitly AND forced by its
# absence, so an anchor cannot enter by default.
#
#SBATCH --job-name=r44_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r44_smoke_%j.out
#SBATCH --error=logs/r44_smoke_%j.err
# --time: 6 runs x 2 edit types = 12 model loads (R38's smoke did 8 within 2 h).
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
SMOKE_ROOT=$D/r44_smoke
# Template for the constant field's shape + schedule keys (step 15, flow_shift 1.0); the
# field's VALUES are overwritten, so which R35 combo it comes from does not matter.
RHO_TEMPLATE=$D/r35_rho/dino_first2

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh root: a stale PNG from an earlier smoke would make a gate compare old output.
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT/logs"

CASES=${TMPDIR:-/tmp}/r44_smoke_cases_${SLURM_JOB_ID:-local}.json
cat > "$CASES" <<'EOF'
[{"video_name": "0001_bus", "edit_type": 1},
 {"video_name": "0034_cows", "edit_type": 2}]
EOF

echo "[r44_smoke] job=${SLURM_JOB_ID} node=$(hostname) out=${SMOKE_ROOT}"
nvidia-smi -L 2>&1 | head -2

# ---- constant rho = 2 field, one npz per smoke clip, every key copied from the template.
python - "$RHO_TEMPLATE" "$SMOKE_ROOT/rho_const2" <<'PY' || { echo "[r44_smoke] FAILED building the constant field"; exit 1; }
import sys
from pathlib import Path
import numpy as np
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for T, v in [(1, "0001_bus"), (2, "0034_cows")]:
    z = dict(np.load(src / f"edit{T}" / f"{v}.npz"))
    z["rho"] = np.full_like(z["rho"], 2.0)
    z["mapping"] = np.str_("constant_2__r44_smoke")
    (dst / f"edit{T}").mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"edit{T}" / f"{v}.npz", **z)
    print(f"[r44_smoke] constant field edit{T}/{v}: rho {z['rho'].shape} == 2.0, "
          f"step={int(z['step'])} flow_shift={float(z['flow_shift'])}")
PY

FAILED=0
# All runs share R35/R36's sampler: step 15 (stage 1: N), fg_boost 4, flow_shift 1.0,
# rollout_chunk_size 21, seed 0. One driver call per edit type (each refuses a cases json
# with no case for its --edit_type).

# uniform <method>: the R44 sweep's invocation at rho = 2 (r44_infer.sh, minus the loop).
uniform () {
  local METHOD=$1
  for T in 1 2; do
    echo "--- [r44_smoke] ${METHOD} (run_fivebench, novp, blend_power 2) / edit${T} ---"
    python evaluation/run_fivebench.py \
      --edit_type "$T" --method "$METHOD" \
      --vp_mode novp \
      --blend_power 2 \
      --data_root "$DATA_ROOT" --out_root "$SMOKE_ROOT/uniform" \
      --cases_json "$CASES" \
      --step 15 --fg_boost_factor 4 --flow_shift 1.0 --rollout_chunk_size 21 --seed 0 \
      2>&1 | tee -a "$SMOKE_ROOT/logs/${METHOD}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[r44_smoke] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
  done
}

# routed <method>: r44_stage3.sh's invocation on the constant field.
routed () {
  local METHOD=$1
  for T in 1 2; do
    echo "--- [r44_smoke] ${METHOD} (r31_stage3, novp, constant rho 2) / edit${T} ---"
    python evaluation/r31_stage3.py \
      --edit_type "$T" --method "$METHOD" \
      --rho_dir "$SMOKE_ROOT/rho_const2" \
      --data_root "$DATA_ROOT" --out_root "$SMOKE_ROOT/routed" \
      --cases_json "$CASES" \
      --vp_mode novp \
      --dump_steps 14 \
      --step 15 --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
      --rollout_chunk_size 21 --seed 0 \
      2>&1 | tee -a "$SMOKE_ROOT/logs/${METHOD}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[r44_smoke] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
  done
}

# draft <method> <blend_sched> <N>: r44_stage1.sh's invocation, final index only.
draft () {
  local METHOD=$1 SCHED=$2 N=$3
  for T in 1 2; do
    echo "--- [r44_smoke] ${METHOD} (r31_stage1, novp, blend_sched=${SCHED}, N=${N}) / edit${T} ---"
    python evaluation/r31_stage1.py \
      --edit_type "$T" --method "$METHOD" \
      --data_root "$DATA_ROOT" \
      --out_root "$SMOKE_ROOT/drafts" \
      --latent_dir "$SMOKE_ROOT/latents/$METHOD" \
      --cases_json "$CASES" \
      --vp_mode novp \
      --dump_steps $((N-1)) --measure_index $((N-1)) \
      --step "$N" --flow_shift 1.0 --fg_boost_factor 4 --blend_power 2 \
      --blend_sched "$SCHED" \
      --rollout_chunk_size 21 --seed 0 \
      2>&1 | tee -a "$SMOKE_ROOT/logs/${METHOD}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[r44_smoke] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
  done
}

uniform uniform_rho2
routed  routed_const2
draft   n15_tgt09     tgt0.9 15
draft   n15_unblended zero   15
draft   n5_tgt09      tgt0.9 5
draft   n5_unblended  zero   5

if [ "$FAILED" -ne 0 ]; then
  echo "[r44_smoke] ${FAILED} render(s) failed -- gates not evaluated"; exit 1
fi

# ---- gates. five-bench env: the only one with `lpips` (needed when G1's frames differ).
conda deactivate
conda activate five-bench
export HF_HUB_OFFLINE=1                  # compute nodes have no network

python - "$SMOKE_ROOT" "$D" <<'PY'
import hashlib, sys
from pathlib import Path
import numpy as np
from PIL import Image

smoke, D = Path(sys.argv[1]), Path(sys.argv[2])
CLIPS = [(1, "0001_bus"), (2, "0034_cows")]

def frame_dir_sha(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.png"))}

res = {}

# G1: routed (constant rho 2) vs uniform (blend_power 2), final frames.
g1 = True
for T, v in CLIPS:
    du = smoke / "uniform" / "uniform_rho2" / f"edit{T}" / v
    dr = smoke / "routed" / "routed_const2" / "step14" / f"edit{T}" / v
    fu, fr = frame_dir_sha(du), frame_dir_sha(dr)
    n_diff = sum(fu.get(k) != fr.get(k) for k in set(fu) | set(fr))
    same = bool(fu) and fu == fr
    print(f"  G1 edit{T}/{v}: frames {len(fu)} uniform vs {len(fr)} routed, {n_diff} differ "
          f"-> {'IDENTICAL' if same else 'DIFFERS'}")
    if not same and fu and set(fu) == set(fr):
        import lpips, torch
        net = lpips.LPIPS(net="alex", verbose=False)
        dmax, dmean, lp = 0, [], []
        for k in sorted(fu):
            a = np.asarray(Image.open(du / k).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(dr / k).convert("RGB"), dtype=np.int16)
            d = np.abs(a - b)
            dmax, dmean = max(dmax, int(d.max())), dmean + [float(d.mean())]
            ta, tb = (torch.from_numpy(x.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
                      for x in (a, b))
            with torch.no_grad():
                lp.append(float(net(ta, tb)))
        print(f"     max|diff| {dmax}/255, mean|diff| {np.mean(dmean):.4f}, "
              f"mean LPIPS-alex {np.mean(lp):.5f}")
    g1 &= same
res["G1"] = g1

# G2: startup echo of every stage-1 run (one line per edit type).
want = {"n15_tgt09": [0, 0] + [1] * 13, "n15_unblended": [1] * 15,
        "n5_tgt09": [0, 1, 1, 1, 1], "n5_unblended": [1] * 5}
g2 = True
for m, vec in want.items():
    lines = [l for l in (smoke / "logs" / f"{m}.log").read_text().splitlines()
             if "blender_rate per step" in l]
    good = len(lines) == 2 and all(f"= {vec}" in l for l in lines)
    print(f"  G2 {m}: {len(lines)} echo lines, expect {vec}: {'ok' if good else 'MISMATCH'}")
    for l in lines:
        print(f"     {l.strip()}")
    g2 &= good
res["G2"] = g2

# G3a: every driver call logged vp_mode=novp and none logged another mode.
g3 = True
for log in sorted((smoke / "logs").glob("*.log")):
    txt = log.read_text()
    n_novp = txt.count("vp_mode=novp")
    n_other = txt.count("vp_mode=vp ") + txt.count("vp_mode=pvp")
    ok = n_novp >= 2 and n_other == 0
    print(f"  G3 {log.stem}: vp_mode=novp x{n_novp}, other modes x{n_other}: {'ok' if ok else 'LEAK'}")
    g3 &= ok
# G3b: every stage-1 npz records novp.
for m in want:
    for T, v in CLIPS:
        mode = str(np.load(smoke / "latents" / m / f"edit{T}" / f"{v}.npz")["vp_mode"])
        ok = mode == "novp"
        print(f"  G3 npz {m} edit{T}/{v}: vp_mode={mode}: {'ok' if ok else 'LEAK'}")
        g3 &= ok
# G3c: the text-only N=15 tgt0.9 draft must differ from R35's VP first2 draft.
for T, v in CLIPS:
    ft = frame_dir_sha(smoke / "drafts" / "n15_tgt09" / "step14" / f"edit{T}" / v)
    fv = frame_dir_sha(D / "r35_stage1" / "first2" / "step14" / f"edit{T}" / v)
    zt = np.load(smoke / "latents" / "n15_tgt09" / f"edit{T}" / f"{v}.npz")["z_trg"]
    zv = np.load(D / "r35_latents" / "first2" / f"edit{T}" / f"{v}.npz")["z_trg"]
    n_same = sum(ft.get(k) == fv.get(k) for k in ft)
    differs = bool(ft) and bool(fv) and ft != fv and not (zt.shape == zv.shape and np.array_equal(zt, zv))
    print(f"  G3 vs R35 VP first2 edit{T}/{v}: {n_same}/{len(ft)} frames identical, "
          f"z_trg {'differs' if differs else 'IDENTICAL'}: {'ok' if differs else 'LEAK'}")
    g3 &= differs
res["G3"] = g3

for g, ok in res.items():
    print(f"GATE {g} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if all(res.values()) else 1)
PY
STATUS=$?
echo "[r44_smoke] gates exit=${STATUS}"
exit $STATUS
