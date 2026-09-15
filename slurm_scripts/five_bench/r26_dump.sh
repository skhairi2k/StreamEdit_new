#!/bin/bash
# R26 pass 1 -- dump the grounding-mask union M_f = M^src_f | M^trg_f for the 22
# cases.json pairs, vp mode. This is the ORACLE the whole task rests on: pass 2 reads
# these masks back and turns Eq. 4's scalar rho into a spatial field.
#
# No new mask maths. M_f is the tensor the pipeline ALREADY computes at denoising index
# len//2 = 7 of 15 (t_inj = 0.5), at edit_causal_inference.py's `inloop_trg_fg_mask =
# inloop_trg_fg_mask_bin | src_fg_mask_bin`; --union_dump_dir only writes it out, packed
# with np.packbits, one row per latent frame of the FINISHED video (i.e. after the SS4.5
# anchor prefix is trimmed and the rollout overlap is dropped, so row f is latent frame f
# of the render pass 2 will produce).
#
# --blend_sched zero is deliberate: W^src == 0 => blender_rate == 1 => pure target Q/K,
# so the target grounding mask is measured without source blending pulling it back toward
# the source. Background source-KV injection at t > t_inj is NOT disabled by this -- it is
# a separate channel and stays on.
#
# The array is over EDIT TYPES (task 0..5 -> edit1..edit6), not over clips: run_fivebench
# takes one --edit_type per invocation, and splitting this way loads the model once per
# type instead of once per clip. Clip counts per type are 1/13/2/2/3/1 = 22 (verified
# against cases.json), so the tasks are very uneven -- edit2 carries 13 of the 22 and sets
# the wall time.
#
# This run also produces ordinary frames, into a SEPARATE root (r26_masks_render). They
# are a by-product -- a zero-blend render is not an arm and must never be scored or
# mixed into the r26_spatial_tau tree.
#
# Sampler config is the run_fivebench.py default set shared by R1/R7/R20/R21/R25 --
# step 15, flow_shift 1.0, fg_boost 4, seed 0, chunk 21, overlap 1, sink 0 -- and MUST
# stay identical to r26_infer.sh and r26_smoke.sh. Per-pair reseeding
# (run_fivebench.py) is what makes a 22-pair subset comparable at all, so seed /
# anchors / cases_json may not drift between the two passes.
#
# Prerequisite: r26_smoke.sh must have passed. Job 960461 (2026-08-26) passed all 6
# gates -- in particular gate5/5b, which prove the per-window union is stitched with the
# right overlap and concat axis on a multi-window clip (0034_cows, 24 latent frames).
# Without that, every mask row after the first window could be silently misaligned.
#
# --time=04:00:00: 22 clips at R22's measured ~5.3 min/clip is ~2h for the WHOLE pass;
# the largest single task (edit2, 13 clips) is ~70 min. 4h is generous margin.
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r26_dump
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/r26_dump_%A_%a.out
#SBATCH --error=logs/r26_dump_%A_%a.err
# TRANSIENT WORKAROUND (2026-08-27) -- remove once node52 is fixed. On job 960708 that
# node handed out DUPLICATE GPUs: tasks 1 and 5 both received SLURM_JOB_GPUS=5 / UUID
# GPU-b9d75409, tasks 0 and 4 both received GPU 1 and `nvidia-smi -L` reported "No
# devices found". All 5 tasks that landed there died at torch.cuda.current_device()
# with "CUDA unknown error ... Setting the available devices to be zero"; task 3, on
# node39, reported cuda_ok True and rendered normally with the SAME script and env.
# R25 job 960685 failed with the identical signature earlier the same night, so this is
# a node-level GRES/cgroup fault, not an environment or activation problem.
#SBATCH --exclude=node52

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r26_masks_render
MASK_ROOT=/projects/dataggen/outputs/five_bench/r26_masks
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES="$REPO/evaluation/cases.json"
METHOD=r26_dump

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 5 ]; then
  echo "[r26_dump] bad array id ${TID} (expected 0-5)"; exit 1
fi
T=$(( TID + 1 ))                       # task 0..5 -> edit type 1..6

# Expected clip count per edit type in cases.json (1/13/2/2/3/1 = 22).
EXPECTED=(1 13 2 2 3 1)
EXP=${EXPECTED[$TID]}

# Conda: use the SAME pattern as r26_smoke.sh (job 960461, all 6 gates passed), which is
# the activation proven to work from this machine. `cd; source .bashrc` -- inherited from
# r21_infer.sh and written for submission from a LOGIN shell -- is what killed R25 array
# job 960685: submitted from inside a GPU allocation, all 6 tasks died at
# torch.cuda.current_device() before rendering a single frame.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT" "$MASK_ROOT"

# Diagnostics: job 960685 failed with "CUDA unknown error ... available devices zero" and
# the logs carried nothing to tell an env leak from a node fault. Print the GPU view the
# job actually has, so a repeat is diagnosable instead of another guess.
echo "[r26_dump] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r26_dump] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
nvidia-smi -L 2>&1 | head -4
python -c "import torch; print('[r26_dump] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -2

if [ ! -f "$CASES" ]; then
  echo "[r26_dump] missing $CASES"; exit 1
fi
if [ ! -d "$ANCHOR_ROOT/edit${T}" ]; then
  echo "[r26_dump] missing anchor dir $ANCHOR_ROOT/edit${T}"; exit 1
fi

echo "[r26_dump] job=${SLURM_ARRAY_JOB_ID}_${TID} edit_type=${T} method=${METHOD}"
echo "[r26_dump] cases=${CASES} masks->${MASK_ROOT}/edit${T}/ expect ${EXP} clip(s)"

FAILED=0
python evaluation/run_fivebench.py \
  --edit_type "$T" --method "$METHOD" \
  --vp_mode vp \
  --first_frame_edit_dir "$ANCHOR_ROOT" \
  --cases_json "$CASES" \
  --blend_sched zero \
  --union_dump_dir "$MASK_ROOT" \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --step 15 --fg_boost_factor 4 --seed 0 \
  || { echo "[r26_dump] FAILED edit${T}"; FAILED=$((FAILED+1)); }

# Frame dirs for THIS edit type, excluding the *_resize dirs evaluate.py leaves behind.
N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/"edit${T}"/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r26_dump] edit${T} frame dirs: ${N_DIRS} (expected ${EXP})"
if [ "$N_DIRS" -ne "$EXP" ] && [ "$FAILED" -eq 0 ]; then
  echo "[r26_dump] COUNT-MISMATCH edit${T}: ${N_DIRS} != ${EXP}"
  FAILED=$((FAILED+1))
fi

# The npz files are the actual deliverable -- the frames are a by-product. A render can
# succeed while the dump silently writes nothing, so count and validate them separately.
N_NPZ=$(ls "$MASK_ROOT"/"edit${T}"/*.npz 2>/dev/null | wc -l)
echo "[r26_dump] edit${T} mask npz: ${N_NPZ} (expected ${EXP})"
if [ "$N_NPZ" -ne "$EXP" ] && [ "$FAILED" -eq 0 ]; then
  echo "[r26_dump] NPZ-MISMATCH edit${T}: ${N_NPZ} != ${EXP}"
  FAILED=$((FAILED+1))
fi

# Shape/content sanity, per npz: rows must equal the clip's latent frame count, the mask
# must not be empty or entirely full, and it must VARY across frames. A frame-0-only
# field broadcast to every frame would look healthy everywhere downstream except here.
if [ "$FAILED" -eq 0 ]; then
  python - "$MASK_ROOT/edit${T}" <<'PYEOF' || FAILED=$((FAILED+1))
import sys, pathlib, numpy as np
d = pathlib.Path(sys.argv[1])
bad = 0
warn = 0
for f in sorted(d.glob("*.npz")):
    with np.load(f) as z:
        F, L = (int(v) for v in z["shape"])
        M = np.unpackbits(z["M"], axis=-1, count=L).astype(bool)
    if M.shape != (F, L):
        print(f"  BAD {f.name}: unpacked {M.shape} != {(F, L)}"); bad += 1; continue
    frac = M.mean(axis=1)
    n_distinct = len({r.tobytes() for r in M})
    flag = ""
    if F < 2:
        flag = "  <-- BAD: fewer than 2 latent frames"; bad += 1
    elif frac.max() == 0.0:
        flag = "  <-- BAD: mask entirely empty"; bad += 1
    elif frac.min() == 1.0:
        # WARN, not BAD. Verified on job 960715 (2026-08-27): 0042_gym-ball, the single
        # edit6 (removal) pair, genuinely grounds to all-ones. `union_out` initialises to
        # ZEROS, so this cannot be an uninitialised buffer -- it is real output. The type-6
        # phrase convention ("with a heavy gym ball" -> "without a heavy gym ball") plus the
        # loose `mask_bin = mask_soft > 0` threshold makes a diffuse removal phrase ground
        # everywhere. R25 flagged the same clip (iou_gt_src 6.2e-05, IoU exactly 0 by
        # construction). Consequence: tau(f,p) == tau_fg everywhere for this clip, i.e. a
        # GLOBAL rho carrying no spatial signal -- still valid data, but it cannot speak to
        # R26's question and must not be counted as evidence for or against spatial tau.
        flag = "  <-- WARN: mask entirely full -- degenerate, contributes NO spatial signal"
        warn += 1
    elif n_distinct < 2:
        flag = "  <-- BAD: identical across every latent frame (collapsed to frame 0)"
        bad += 1
    print(f"  {f.name:<32} shape=({F},{L}) distinct={n_distinct}/{F} "
          f"fg_frac min={frac.min():.3f} med={np.median(frac):.3f} max={frac.max():.3f}{flag}")
print(f"[r26_dump] npz sanity: {bad} bad, {warn} degenerate(all-ones, non-fatal)")
sys.exit(1 if bad else 0)
PYEOF
fi

echo "[r26_dump] failures: ${FAILED}"
exit $(( FAILED > 0 ))
