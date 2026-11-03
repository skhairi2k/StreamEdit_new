#!/bin/bash
# R65 full render: AnyV2V (I2VGen-XL), FiVE's published setting at 32 frames, on the 415 FiVE-Bench-Correct pairs.
#
# ARRAY LAYOUT: K = 10 shards (--array=0-9); video j of the sorted video list goes to shard j mod K, so each task renders 10 of the 100
# videos (all their edit types, one inversion per video). K must match --array: change both together.
# Time per video at 32 frames ~ 600 s (inversion ~280 s + 4.15 pairs x ~77 s, A100 diagnostic 2026-10-28; the L40S is about as fast), so a task
# needs ~1.7 h; with the plan's 1.3x factor ~2.2 h. --time=06:00:00 leaves a ~3x margin for a slow node.
# Settings (the driver defaults): t_idx 0, 500 inversion + 50 edit steps, CFG 9.0, PnP 0.2 / 0.2 / 0.5, seed 8888, AnyV2V's negative prompt.
# Cases: evaluation/r65_cases_correct.json (n_frames = min(32, source frames); 0072_dog-agility has 25).
# Outputs under $D/r65_anyv2v: correct_raw512, correct_srcres (what the R57 suite scores), correct_manifest_shard<i>.csv, src512/, inversions/.
# RESUME: a pair whose two frame dirs hold n frames is skipped, so a timed-out task is resubmitted with `sbatch --array=<i>`.
# GUARD: after the driver, every pair of this shard must hold n frames in both _raw512 and _srcres, else the task fails.
# The anyv2v env's python is called by its absolute path (`conda activate` does not put it first on PATH on this account) and the CUDA
# path variables exported for streamgve are dropped. --exclude=node52 only: it advertises gpu:8 but exposes no device to batch jobs.
#SBATCH --job-name=r65_render
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/r65_render_%A_%a.out
#SBATCH --error=logs/r65_render_%A_%a.err

N_SHARDS=10
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
OUT=~/Data/dataggen/outputs/five_bench/r65_anyv2v
CASES=evaluation/r65_cases_correct.json
PY=~/anaconda3/envs/anyv2v/bin/python

TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "$N_SHARDS" ]; then
  echo "[r65_render] bad array id '${TID}' (expected 0-$((N_SHARDS - 1)))"; exit 1
fi

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CPATH PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1

echo "[r65_render] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} shard=${TID}/${N_SHARDS} $(date '+%F %T')"
nvidia-smi -L 2>&1 | head -4
PREFIX=$("$PY" -I -c 'import sys; print(sys.prefix)')
if [ "$PREFIX" != "$HOME/anaconda3/envs/anyv2v" ]; then
  echo "[r65_render] FATAL: python prefix $PREFIX is not the anyv2v env"; exit 1
fi

FAILED=0
"$PY" -I evaluation/r65_anyv2v_run.py --cases "$CASES" --out_root "$OUT/correct" --shard "$TID" --n_shards "$N_SHARDS" \
  || { echo "[r65_render] FAILED driver: exit $?"; FAILED=$((FAILED + 1)); }

# every pair of this shard must be complete in both frame roots
"$PY" -I - "$CASES" "$OUT" "$TID" "$N_SHARDS" <<'PYEOF'
import json, sys
from pathlib import Path
cases, out, shard, k = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pairs = json.load(open(cases))
videos = sorted({c["video_name"] for c in pairs})
mine = {v for j, v in enumerate(videos) if j % k == shard}
bad = 0
for c in pairs:
    if c["video_name"] not in mine:
        continue
    for root in ("correct_raw512", "correct_srcres"):
        d = out / root / f"edit{c['edit_type']}" / c["video_name"]
        n = len(list(d.glob("*.png"))) if d.is_dir() else -1
        if n != c["n_frames"]:
            print(f"[r65_render] INCOMPLETE {root}/edit{c['edit_type']}/{c['video_name']}: {n} frames, expected {c['n_frames']}")
            bad += 1
print(f"[r65_render] shard {shard}: {sum(c['video_name'] in mine for c in pairs)} pairs checked, {bad} incomplete dirs")
sys.exit(1 if bad else 0)
PYEOF
[ $? -ne 0 ] && FAILED=$((FAILED + 1))

echo "[r65_render] failures: ${FAILED} $(date '+%F %T')"
[ "$FAILED" -eq 0 ]
