#!/bin/bash
# R35 -- HARD-STEP blend schedule: full source anchoring for the first k denoising
# steps, then blending disabled for every remaining step.
#
# User request (2026-09-24): "the scheduler of the blend rate is equal to 1. during
# 1 step, 2 steps and 3 steps, then 0. for the rest of steps (disable blending)",
# on three clips only.
#
# SIGN CONVENTION -- the one thing to get right here. `_schedule_blend_rate` returns
# the SOURCE-ANCHORING weight s(p), and the bridge sets blender_rate = 1 - s(p):
#   s = 1 -> blender_rate = 0 -> keys/queries taken from the SOURCE branch (blend ON)
#   s = 0 -> blender_rate = 1 -> identity, no blending at all (blend OFF, == R20 `zero`)
# So `first<k>` is s = 1 for step_idx < k, else 0 -- which is exactly "1 during k
# steps, then 0 (disable blending)". Verified locally before this job was written:
#   first1 -> s = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#   first2 -> s = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
#   first3 -> s = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
# and the degenerate ends agree with the existing named schedules (first0 == zero,
# first15 == const), which is what proves the indexing is a COUNT and not a fraction.
#
# Indexed on step_idx, NOT on the normalised p = i/(N-1): "during 2 steps" must mean
# two steps at any --step, not 2/15 of the schedule.
#
# ARRAY LAYOUT -- one k per task, so the three arms render in parallel:
#   0  first1      1 step  of source anchoring
#   1  first2      2 steps
#   2  first3      3 steps
# Each task loads the model once and loops the three edit types below.
#
# CLIPS (5 pairs, evaluation/cases_blendfirst.json -- the CANONICAL set):
#   edit2  0011_lucia         "Replace the woman with a lion."
#   edit2  0090_A_deer        "Replace the deer with a snowmobile..."      (added 2026-09-24)
#   edit3  0017_kid-football  "Change the color of the boy's cap from red to yellow."
#   edit5  0011_lucia         "Add a dog following the woman"
#   edit5  0069_car-turn      "Add a giant inflatable flamingo to the roof..." (added 2026-09-24)
# 0011_lucia appears under BOTH edit2 and edit5, and 0017_kid-football appears in the
# edit2 and edit3 benchmark jsons -- run_fivebench.py filters on (video_name, edit_type)
# together, so a task renders exactly the pairs listed, not every name match.
#
# INCREMENTAL RUNS: set R35_CASES to a subset json to render only those pairs into the
# SAME method dirs (e.g. cases_blendfirst_new2.json, the 2 clips added 2026-09-24, so the
# first three do not get re-rendered for nothing). The edit types looped are derived FROM
# that json -- hardcoding them would make run_fivebench.py SystemExit on an edit type the
# subset does not contain. The post-run guard always checks the CANONICAL 5, whatever was
# rendered this run, so an incremental job still fails if it leaves the set incomplete.
#
# REFERENCE: no baseline arm is rendered here. The config-matched paper-schedule vp
# run already exists on disk for all three pairs (five_bench/r7_visual_prompting,
# 69/57/69 frames), so comparing against it costs no GPU.
#
# Sampler settings match r31_stage3.sh's render call (step 15, fg_boost 4,
# blend_power 2, rollout_chunk_size 21, seed 0, vp + anchors), so these arms sit on
# the same axis as the R31/R26 renders.
#
# --mem=64G: the partition default host-OOMs loading UMT5-XXL + the checkpoint.
# --exclude: the proven post-2026-09-22 node set (R33/R34), kept defensively.
#SBATCH --job-name=r35_blendfirst
#SBATCH --partition=L40S,A100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r35_blendfirst_%A_%a.out
#SBATCH --error=logs/r35_blendfirst_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
# ~/Data, not /projects: the /projects share is unreliable per node (2026-09-22 fix).
OUT_ROOT=~/Data/dataggen/outputs/five_bench/r35_blendfirst
ANCHOR_ROOT=~/Data/dataggen/outputs/five_bench/anchors
CANON=$REPO/evaluation/cases_blendfirst.json
CASES=${R35_CASES:-$CANON}
STEP=15

# Edit types to loop = exactly those present in the json being rendered.
EDIT_TYPES=($(python -c "
import json,sys
print(' '.join(sorted({str(c['edit_type']) for c in json.load(open(sys.argv[1]))})))
" "$CASES")) || { echo "[r35] FATAL: could not read edit types from $CASES"; exit 1; }

SCHEDS=(first1 first2 first3)
TID=${SLURM_ARRAY_TASK_ID}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -gt 2 ]; then
  echo "[r35] bad array id ${TID} (expected 0-2)"; exit 1
fi
SCHED=${SCHEDS[$TID]}
METHOD="blend_${SCHED}_vp"

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r35] job=${SLURM_ARRAY_JOB_ID}_${TID} sched=${SCHED} method=${METHOD}"
echo "[r35] rendering cases=${CASES} edit_types=${EDIT_TYPES[*]}"
echo "[r35] guard set   =${CANON}"
echo "[r35] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

# ---- GATE: print the schedule this job will actually run, from the SAME function the
# pipeline calls. A silently-wrong step count would still produce 3 clips and a full
# frame tree with nothing downstream able to tell -- this is the only cheap check.
python - "$SCHED" "$STEP" <<'PYGATE' || { echo "[r35] FATAL: schedule gate failed"; exit 1; }
import sys
sys.path.insert(0, "Self-Forcing_StreamEdit")
from pipeline.utils import _schedule_blend_rate as S
sched, n = sys.argv[1], int(sys.argv[2])
k = int(sched[len("first"):])
s = [S(sched, i, n) for i in range(n)]
print(f"[r35][gate] {sched}: s(p)         = {[int(x) for x in s]}")
print(f"[r35][gate] {sched}: blender_rate = {[int(1-x) for x in s]}  (0=blend ON, 1=OFF)")
assert s[:k] == [1.0]*k, f"first {k} steps are not fully anchored: {s[:k]}"
assert set(s[k:]) <= {0.0}, f"steps after {k} are not zero: {s[k:]}"
print(f"[r35][gate] PASS: source anchored for exactly {k} of {n} steps, then disabled")
PYGATE

FAILED=0
for T in "${EDIT_TYPES[@]}"; do
  echo "--- [r35] ${METHOD} / edit${T} ---"
  python evaluation/run_fivebench.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --blend_sched "$SCHED" \
    --cases_json "$CASES" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --step "$STEP" \
    --flow_shift 1.0 \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r35] FAILED ${METHOD} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard: every pair in the CANONICAL set must have a non-empty frame dir.
# Checked pair-by-pair rather than by counting directories, so an incremental run that
# silently rendered the wrong clip cannot pass on a correct total.
GUARD_OUT=$(python - "$CANON" "$OUT_ROOT/$METHOD" <<'PYGUARD'
import json, os, sys
canon, root = sys.argv[1], os.path.expanduser(sys.argv[2])
bad = 0
for c in json.load(open(canon)):
    d = os.path.join(root, f"edit{c['edit_type']}", c["video_name"])
    n = len(os.listdir(d)) if os.path.isdir(d) else -1
    print(f"[r35]   edit{c['edit_type']} {c['video_name']}: "
          + (f"{n} frames" if n > 0 else ("MISSING DIR" if n < 0 else "EMPTY")))
    if n <= 0:
        bad += 1
print(f"[r35] guard: {bad} incomplete pair(s) of {len(json.load(open(canon)))}")
sys.exit(1 if bad else 0)
PYGUARD
)
GUARD_RC=$?
echo "$GUARD_OUT"
if [ "$GUARD_RC" -ne 0 ]; then FAILED=$((FAILED+1)); fi

echo "[r35] failures: ${FAILED}"
exit $(( FAILED > 0 ))
