#!/bin/bash
# R38 smoke: 4 gates for the `tgt<tau>` noise-level schedule before any full-bench run.
#
#   G1  tgt0.9 at N=15 == R35 first2       (sha256 of every step14 PNG + z_trg array_equal)
#       -- the gate that licenses REUSING R35's four arms as R38's N=15 points.
#   G2  zero   at N=15 == R35 unblended    (regression: the new t_cur plumbing must leave
#                                           every pre-R38 schedule byte-identical)
#   G3  tgt0.9 at N=5  == first1 at N=5    (threshold logic == step-count logic where they
#                                           must coincide; same frames + z_trg)
#   G4  echoed blender_rate vectors: N=15 tgt0.9 [0,0,1,...], N=5 tgt0.9 [0,1,1,1,1]
#
# 2 clips: 0001_bus (edit1, single window) and 0034_cows (edit2, 24 latent frames -- the
# multi-window clip, the only one that exercises the rollout overlap; R26 lesson).
# r31_stage1.py re-seeds before EVERY pair, so a --cases_json subset renders exactly the
# frames the full-bench R35 run produced for these clips.
#
#SBATCH --job-name=r38_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r38_smoke_%j.out
#SBATCH --error=logs/r38_smoke_%j.err
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors
SMOKE_ROOT=$D/r38_smoke

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh root: a stale PNG from an earlier smoke would make a gate compare old output.
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT/logs"

CASES=${TMPDIR:-/tmp}/r38_smoke_cases_${SLURM_JOB_ID:-local}.json
cat > "$CASES" <<'EOF'
[{"video_name": "0001_bus", "edit_type": 1},
 {"video_name": "0034_cows", "edit_type": 2}]
EOF

echo "[r38_smoke] job=${SLURM_JOB_ID} node=$(hostname) out=${SMOKE_ROOT}"
nvidia-smi -L 2>&1 | head -2

# run <method> <blend_sched> <N>: one r31_stage1.py call per edit type (it refuses a
# cases json with no case for its --edit_type), final index only, R35's stage-1 flags.
FAILED=0
run () {
  local METHOD=$1 SCHED=$2 N=$3
  for T in 1 2; do
    echo "--- [r38_smoke] ${METHOD} (blend_sched=${SCHED}, N=${N}) / edit${T} ---"
    python evaluation/r31_stage1.py \
      --edit_type "$T" \
      --method "$METHOD" \
      --data_root "$DATA_ROOT" \
      --out_root "$SMOKE_ROOT/frames" \
      --latent_dir "$SMOKE_ROOT/latents/$METHOD" \
      --cases_json "$CASES" \
      --vp_mode vp \
      --first_frame_edit_dir "$ANCHOR_ROOT" \
      --dump_steps $((N-1)) \
      --measure_index $((N-1)) \
      --step "$N" \
      --flow_shift 1.0 \
      --fg_boost_factor 4 \
      --blend_power 2 \
      --blend_sched "$SCHED" \
      --rollout_chunk_size 21 \
      --seed 0 \
      2>&1 | tee -a "$SMOKE_ROOT/logs/${METHOD}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[r38_smoke] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
  done
}

run tgt09_n15  tgt0.9 15
run zero_n15   zero   15
run tgt09_n5   tgt0.9 5
run first1_n5  first1 5

if [ "$FAILED" -ne 0 ]; then
  echo "[r38_smoke] ${FAILED} render(s) failed -- gates not evaluated"; exit 1
fi

# ---- gates: frames by sha256 over the full file list, z_trg by np.array_equal.
python - "$SMOKE_ROOT" "$D" <<'PY'
import hashlib, sys
from pathlib import Path
import numpy as np

smoke, D = Path(sys.argv[1]), Path(sys.argv[2])
CLIPS = [(1, "0001_bus"), (2, "0034_cows")]

def frames(root, step, T, v):
    d = root / f"step{step:02d}" / f"edit{T}" / v
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.png"))}

def compare(tag, a_frames, a_lat, b_frames, b_lat, step):
    ok = True
    for T, v in CLIPS:
        fa, fb = frames(a_frames, step, T, v), frames(b_frames, step, T, v)
        za = np.load(a_lat / f"edit{T}" / f"{v}.npz")["z_trg"]
        zb = np.load(b_lat / f"edit{T}" / f"{v}.npz")["z_trg"]
        same_f = bool(fa) and fa == fb
        same_z = za.shape == zb.shape and np.array_equal(za, zb)
        n_diff = sum(fa.get(k) != fb.get(k) for k in set(fa) | set(fb))
        print(f"  {tag} edit{T}/{v}: frames {len(fa)} vs {len(fb)}, {n_diff} differ; "
              f"z_trg {'IDENTICAL' if same_z else 'DIFFERS'}")
        ok &= same_f and same_z
    return ok

res = {}
res["G1"] = compare("G1", smoke / "frames" / "tgt09_n15", smoke / "latents" / "tgt09_n15",
                    D / "r35_stage1" / "first2", D / "r35_latents" / "first2", 14)
res["G2"] = compare("G2", smoke / "frames" / "zero_n15", smoke / "latents" / "zero_n15",
                    D / "r35_stage1" / "unblended", D / "r35_latents" / "unblended", 14)
res["G3"] = compare("G3", smoke / "frames" / "tgt09_n5", smoke / "latents" / "tgt09_n5",
                    smoke / "frames" / "first1_n5", smoke / "latents" / "first1_n5", 4)

# G4: the startup echo of every tgt0.9 call (2 edit types per run) must be the expected vector.
want = {"tgt09_n15": "[0, 0" + ", 1" * 13 + "]", "tgt09_n5": "[0, 1, 1, 1, 1]"}
g4 = True
for m, vec in want.items():
    lines = [l for l in (smoke / "logs" / f"{m}.log").read_text().splitlines()
             if "blender_rate per step" in l]
    good = len(lines) == 2 and all(f"= {vec}" in l for l in lines)
    print(f"  G4 {m}: {len(lines)} echo lines, expect {vec}: {'ok' if good else 'MISMATCH'}")
    for l in lines:
        print(f"     {l.strip()}")
    g4 &= good
res["G4"] = g4

for g, ok in res.items():
    print(f"GATE {g} {'PASS' if ok else 'FAIL'}")
sys.exit(0 if all(res.values()) else 1)
PY
STATUS=$?
echo "[r38_smoke] gates exit=${STATUS}"
exit $STATUS
