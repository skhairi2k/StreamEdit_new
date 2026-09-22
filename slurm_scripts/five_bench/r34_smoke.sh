#!/bin/bash
# R34 smoke -- prove the first-chunk window is what R34 thinks it is, and that adding
# --max_frames left the shared evaluator untouched.
#
# evaluate.py is shared by R1/R3/R7/R20-R33 and every stored reference CSV in
# evaluation/csv/ was produced by it, so the non-regression gate matters at least as much
# as the new-behaviour gates. R28's smoke made the same argument; GATE2 below is its
# GATE1, retargeted at R34's two metrics.
#
# THE CLIP. edit1 holds exactly ONE case in evaluation/cases.json (0001_bus), so
# --annotation_mapping_files edit1_FiVE.json + --cases_json scores a single clip with no
# custom subset file. 0001_bus's source is 80 frames; find_closest_num_frame(80, a=4, b=3)
# = 12*6-3 = 69, so its render is 69 pixel frames = 18 latent frames = 6 chunks of 3.
# Clip length varies across the bench, but the FIRST CHUNK DOES NOT: 3 latent frames are
# always pixel frames 0-8 (1 + 4 + 4), which is why --max_frames 9 is a constant and not
# a per-clip quantity.
#
# GATE0 (PRECOND) -- the render tree exists, holds >= 9 frames, and its frame count
#   satisfies (n - 1) % 4 == 0. A render that fails the modulus is not on a latent-frame
#   grid at all, and "first 9 frames" would not be "first chunk" for it.
#
# GATE1 (WINDOW) -- the --frame_stride 1 --max_frames 9 run's per-frame CSV must hold
#   frame_idx exactly {0..8} for EACH metric. This is the gate the whole task rests on:
#   everything downstream assumes those 9 frames and nothing else.
#
# GATE2 (NONREG) -- a run WITHOUT --max_frames at stride 8 must reproduce the stored
#   edit1_FiVE_r26_taubg0_taufg2_vp_frame_stride8.csv to within 1e-3 RELATIVE.
#   Not bit-for-bit, and deliberately so: R28 measured this evaluator to be deterministic
#   only WITHIN a node (two runs on one node bit-identical at max|d| = 0.0, both offset
#   from the stored reference by 3.877e-05 produced on a different node). 1e-3 is the
#   tolerance r26/r28_summarize.py already use; a real regression is orders of magnitude
#   larger. The observed worst diff is printed either way so drift stays visible.
#
# GATE3 (DIFFERS) -- the chunk-1 averages must DIFFER from the whole-video ones. Without
#   this, GATE2 is also satisfied by a flag that is parsed and then ignored -- which is
#   precisely the failure mode r31_eval.sh's header records for config.yaml's vestigial
#   `metrics:` key.
#
# GATE4 (ALIGNED) -- frame 0's per-frame value in the chunk-1 run must EQUAL frame 0's in
#   the whole-video run. Same source frame, same render frame, same mask, so any
#   difference means the truncation desynchronised src_image_names / tgt_image_names /
#   masks, which the positional zip at evaluate.py:549 would then silently score as
#   mismatched pairs. GATE1 and GATE3 both pass under that bug; this is the only gate
#   that catches it.
#
#SBATCH --job-name=r34_smoke
# RENDER ROOTS LIVE IN ~/Data, NOT /projects. R33 burned five rounds of jobs on a
# /projects/dataggen mount regression: node51/node57 (L40S), then node01/node02 (A100)
# each in turn absorbed the bulk of an array and failed instantly on a missing mount,
# while nodes already holding warm jobs saw it fine. Exclude-and-retry never converged
# (2026-09-21, R33 run log). Fixed 2026-09-22 by using the home-dir copy, which is
# reliably mounted everywhere. Do not point this back at /projects.
# node52 is excluded for an unrelated, older reason: it advertises gpu:8 but exposes no
# device to batch jobs (nvidia-smi -L -> "No devices found."). The other three are the
# nodes R33 confirmed bad; kept defensively, not yet disproven for the home mount.
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r34_smoke_%j.out
#SBATCH --error=logs/r34_smoke_%j.err

# No `set -u`: conda's activate-gcc_linux-64.sh hook reads an unbound SYS_SYSROOT and
# would abort at `conda activate` (R20 job 907410). Absolute interpreter path below
# sidesteps activation entirely -- the eval env is five-bench, NOT streamgve (.bashrc
# auto-activates streamgve and `conda activate five-bench` alone does not pop it, which
# crashed the R2 eval on ModuleNotFoundError: torchmetrics).
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

PY=~/anaconda3/envs/five-bench/bin/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ARM=taubg0_taufg2_vp
ARM_ROOT=~/Data/dataggen/outputs/five_bench/r26_spatial_tau/$ARM
WORK=~/Data/dataggen/outputs/five_bench/r34_smoke
CASES=evaluation/cases.json
CLIP=0001_bus
REF=evaluation/csv/edit1_FiVE_r26_${ARM}_frame_stride8.csv
METRICS="lpips_unedit_part clip_similarity_target_image"

echo "[r34_smoke] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2
$PY -c "import torch; print('[r34_smoke] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"

rm -rf "$WORK"; mkdir -p "$WORK" logs

# ---------------------------------------------------------------- GATE0
echo "=== GATE0: render tree is on a latent-frame grid and long enough ==="
if [ ! -d "$ARM_ROOT/edit1/$CLIP" ]; then
  echo "GATE0-FAIL precond: $ARM_ROOT/edit1/$CLIP does not exist"; exit 1
fi
NFR=$(ls "$ARM_ROOT/edit1/$CLIP" | grep -cE '\.(png|jpg|jpeg)$')
if [ "$NFR" -lt 9 ]; then
  echo "GATE0-FAIL precond: only ${NFR} frames rendered, need >= 9 for one chunk"; exit 1
fi
if [ $(( (NFR - 1) % 4 )) -ne 0 ]; then
  echo "GATE0-FAIL precond: ${NFR} frames, (n-1)%4 != 0 -- not a latent-frame grid, so"
  echo "GATE0        'first 9 pixel frames' is not 'first chunk' for this render. STOP."
  exit 1
fi
echo "GATE0-OK precond: ${NFR} frames = $(( (NFR - 1) / 4 + 1 )) latent frames, chunk 1 = pixel 0-8"

# ---------------------------------------------------------------- runs
echo "=== RUN A: chunk-1 window (--frame_stride 1 --max_frames 9) ==="
$PY evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics $METRICS \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "$DATA_ROOT"/edit_prompt/edit1_FiVE.json \
  --tgt_methods "$ARM_ROOT" \
  --tgt_layout edit_video \
  --cases_json "$CASES" \
  --frame_stride 1 --max_frames 9 --per_frame \
  --result_path "$WORK/chunk1.csv" > "$WORK/runA.log" 2>&1
RC_A=$?
echo "[r34_smoke] RUN A exit=${RC_A} errors=$(grep -c 'Error:' "$WORK/runA.log")"

echo "=== RUN B: whole video, no --max_frames (non-regression reference) ==="
$PY evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics $METRICS \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "$DATA_ROOT"/edit_prompt/edit1_FiVE.json \
  --tgt_methods "$ARM_ROOT" \
  --tgt_layout edit_video \
  --cases_json "$CASES" \
  --frame_stride 8 --per_frame \
  --result_path "$WORK/whole.csv" > "$WORK/runB.log" 2>&1
RC_B=$?
echo "[r34_smoke] RUN B exit=${RC_B} errors=$(grep -c 'Error:' "$WORK/runB.log")"

# evaluate.py swallows per-metric exceptions and still exits 0, so a clean code is NOT
# evidence of a clean CSV -- any `Error:` line means a metric was dropped and that row's
# later columns are shifted.
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ] || \
   [ "$(grep -c 'Error:' "$WORK/runA.log")" -ne 0 ] || \
   [ "$(grep -c 'Error:' "$WORK/runB.log")" -ne 0 ]; then
  echo "GATES-ABORT: a run crashed or dropped a metric. See $WORK/runA.log, $WORK/runB.log"
  tail -20 "$WORK/runA.log" "$WORK/runB.log"
  exit 1
fi

# ---------------------------------------------------------------- GATE1-4
$PY - "$WORK" "$REF" "$CLIP" <<'PY'
import csv, os, sys

work, ref_path, clip = sys.argv[1], sys.argv[2], sys.argv[3]
METRICS = ("lpips_unedit_part", "clip_similarity_target_image")
TOL = 1e-3          # relative; see the GATE2 note in the header
EXPECTED = set(range(9))

chunk_pf = os.path.join(work, "chunk1_frame_stride1_per_frame.csv")
whole_pf = os.path.join(work, "whole_frame_stride8_per_frame.csv")
chunk_avg = os.path.join(work, "edit1_FiVE_chunk1_frame_stride1.csv")
whole_avg = os.path.join(work, "edit1_FiVE_whole_frame_stride8.csv")

failed = []


def load_per_frame(path):
    """{metric: {frame_idx: value}} for the single smoke clip."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["metric"], {})[int(r["frame_idx"])] = float(r["value"])
    return out


def load_avg(path):
    """{metric: value} from a per-edit-type CSV holding exactly one data row."""
    if not os.path.exists(path):
        return None
    rows = list(csv.reader(open(path)))
    hdr = [h.split("|")[-1] for h in rows[0]]
    if len(rows) != 2:
        return None
    return {m: float(rows[1][hdr.index(m)]) for m in METRICS if m in hdr}


# ------------------------------------------------------------------ GATE1
print("=== GATE1: the window is exactly pixel frames 0-8 ===")
chunk = load_per_frame(chunk_pf)
if chunk is None:
    print(f"GATE1-FAIL window: no per-frame CSV at {chunk_pf}")
    failed.append("GATE1")
else:
    for m in METRICS:
        idx = set(chunk.get(m, {}))
        if idx == EXPECTED:
            print(f"GATE1-OK   window: {m} scored frames {sorted(idx)}")
        else:
            print(f"GATE1-FAIL window: {m} scored {sorted(idx)}, expected {sorted(EXPECTED)}")
            failed.append("GATE1")

# ------------------------------------------------------------------ GATE2
print("=== GATE2: the shared evaluator is unchanged without --max_frames ===")
new = load_avg(whole_avg)
old = load_avg(ref_path)
if new is None or old is None:
    print(f"GATE2-FAIL nonreg: could not read {whole_avg} or {ref_path}")
    failed.append("GATE2")
else:
    worst = 0.0
    for m in METRICS:
        a, b = new[m], old[m]
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        worst = max(worst, rel)
        print(f"    {m}: new={a:.6f} stored={b:.6f} rel={rel:.3e}")
    if worst <= TOL:
        print(f"GATE2-OK   nonreg: matches the stored R26 reference within {TOL:g} "
              f"(worst {worst:.3e}); --max_frames is additive")
    else:
        print(f"GATE2-FAIL nonreg: worst relative diff {worst:.3e} > {TOL:g}. The shared "
              f"evaluator moved -- every stored reference is at risk. STOP.")
        failed.append("GATE2")

# ------------------------------------------------------------------ GATE3
print("=== GATE3: the window actually changes the score ===")
c_avg = load_avg(chunk_avg)
if c_avg is None or new is None:
    print("GATE3-FAIL differs: missing an averaged CSV")
    failed.append("GATE3")
else:
    for m in METRICS:
        if c_avg[m] != new[m]:
            print(f"GATE3-OK   differs: {m} chunk1={c_avg[m]:.6f} whole={new[m]:.6f}")
        else:
            print(f"GATE3-FAIL differs: {m} identical ({c_avg[m]:.6f}) -- --max_frames "
                  f"was parsed and then ignored")
            failed.append("GATE3")

# ------------------------------------------------------------------ GATE4
print("=== GATE4: frame 0 is the same frame pair in both windows ===")
whole = load_per_frame(whole_pf)
if chunk is None or whole is None:
    print("GATE4-FAIL aligned: missing a per-frame CSV")
    failed.append("GATE4")
else:
    for m in METRICS:
        a = chunk.get(m, {}).get(0)
        b = whole.get(m, {}).get(0)
        if a is None or b is None:
            print(f"GATE4-FAIL aligned: {m} has no frame 0 in one of the runs")
            failed.append("GATE4")
        elif a == b:
            print(f"GATE4-OK   aligned: {m} frame 0 = {a:.6f} in both runs")
        else:
            print(f"GATE4-FAIL aligned: {m} frame 0 chunk1={a:.8f} whole={b:.8f}. The "
                  f"truncation desynchronised src/tgt/mask -- evaluate.py:549 is zipping "
                  f"mismatched pairs. STOP.")
            failed.append("GATE4")

print()
if failed:
    print(f"[r34_smoke] FAILED: {sorted(set(failed))}")
    sys.exit(1)
print("[r34_smoke] ALL GATES PASSED -- ready to submit r34_eval.sh")
PY
RC=$?

echo "--- r34_smoke done (exit ${RC}) ---"
exit $RC
