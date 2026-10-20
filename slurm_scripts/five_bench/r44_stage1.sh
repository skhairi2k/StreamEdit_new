#!/bin/bash
# R44 stage 1 -- the TEXT-ONLY (novp) divergence-measurement DRAFTS on the WHOLE 419-pair
# FiVE-Bench: r38_stage1.sh with no visual prompt, for the stage-1 RUNS the user's
# combo selection needs.
#
# USAGE: sbatch --array=0-$(( $(bash slurm_scripts/five_bench/r44_combos.sh --runs | wc -l) - 1 )) \
#          slurm_scripts/five_bench/r44_stage1.sh
# ARRAY LAYOUT: task i -> RUN i of `r44_combos.sh --runs` (unique n{N}_{regime} of
# evaluation/r44_selected_combos.txt, first-appearance order). One RUN is shared by both
# metrics (stage 2 measures DINO and LPIPS on the same draft). The model loads once per
# task, then loops all 6 edit types (as r35/r38_stage1.sh).
#   n15_unblended  --blend_sched zero,   N=15   n15_tgt09  --blend_sched tgt0.9, N=15
#   n5_unblended   --blend_sched zero,   N=5    n5_tgt09   --blend_sched tgt0.9, N=5
# No --cases_json: absent = full bench via run_fivebench's own edit{T}_FiVE.json.
#
# TEXT ONLY: --vp_mode novp AND no --first_frame_edit_dir (whose absence forces novp too),
# so an anchor cannot enter by default. The post-run guard also checks that every latent
# npz records vp_mode == novp (smoke gate G3 checks the same on 2 clips).
#
# REGIMES: tgt0.9 anchors every call whose warped input t > 0.9 -- t = 1.0, 0.933 at N=15
# (== R35's first2, R38 gate G1), t = 1.0 only at N=5. zero = Q/K blend OFF every step.
# The startup echo resolves the schedule on the pipeline's real grid and must read:
#   n15_unblended [1 x15]   n15_tgt09 [0,0,1 x13]   n5_unblended [1 x5]   n5_tgt09 [0,1,1,1,1]
# Checked at wait-stage1 (and by smoke gate G2).
#
# MEASURED LATENT: --dump_steps N-1 --measure_index N-1, the final x0 (frames under
# step{N-1:02d}/), the same "fully denoised draft" convention as R35 (step14) / R38 (step04).
#
# Everything else held IDENTICAL to r35/r38_stage1.sh: seed 0, flow_shift 1.0, fg_boost 4,
# blend_power 2 (inert under a named schedule), rollout_chunk_size 21.
#
# --time=20:00:00: R35's N=15 VP stage 1 ran ~1.5 min/clip (~10.5 h for 419 clips); N=5
# takes roughly a third plus the per-clip fixed cost. NO RESUME: r31_stage1.py re-renders
# every pair of an edit type, so a timed-out task is resubmitted whole (--array=<i>).
#SBATCH --job-name=r44_stage1
# L40S ONLY: R35's 1007492 OOM'd a 40 GB A100 on 0032_mermaid at 39.2/39.5 GiB.
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r44_stage1_%A_%a.out
#SBATCH --error=logs/r44_stage1_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
FIVE_ROOT=~/Data/dataggen/outputs/five_bench

source "$REPO/slurm_scripts/five_bench/r44_combos.sh"
r44_load || exit 1
TID=${SLURM_ARRAY_TASK_ID}
K=${#R44_RUNS[@]}
if [ -z "$TID" ] || [ "$TID" -lt 0 ] || [ "$TID" -ge "$K" ]; then
  echo "[r44_stage1] bad array id '${TID}' (expected 0-$((K-1)) for ${K} runs: ${R44_RUNS[*]})"; exit 1
fi
RUN=${R44_RUNS[$TID]}
# A RUN is metric-free; parse it through any combo that carries it to get N and SCHED.
r44_parse "dino_${RUN}" || exit 1
STEPS=$N
METHOD="$RUN"

# Frames -> r44_stage1/{run}/step{NN}/edit{T}/{video}/*.png
OUT_ROOT=$FIVE_ROOT/r44_stage1
# Latents -> r44_latents/{run}/edit{T}/{video}.npz -- RUN folded into the path.
LATENT_DIR=$FIVE_ROOT/r44_latents/$RUN

# The last index of an N-step rollout -- the fully denoised draft AND the target stage 2
# measures. One index => one VAE decode per clip.
DUMP_STEPS=$((STEPS - 1))
MEASURE_INDEX=$((STEPS - 1))

# Conda: `cd; source .bashrc` killed R25 array job 960685 when submitted from inside a GPU
# allocation; this is the pattern proven by r26_smoke/r36_smoke/r39_smoke.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT" "$LATENT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[r44_stage1] job=${SLURM_ARRAY_JOB_ID}_${TID} run=${RUN} blend_sched=${SCHED} steps=${STEPS} (text only)"
echo "[r44_stage1] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r44_stage1] dump_steps=${DUMP_STEPS} measure_index=${MEASURE_INDEX} (fully denoised, final x0)"
echo "[r44_stage1] no --cases_json -- full bench via run_fivebench's own edit{T}_FiVE.json"
echo "[r44_stage1] frames -> ${OUT_ROOT}/${METHOD}   latents -> ${LATENT_DIR}/edit{T}"
nvidia-smi -L 2>&1 | head -2

FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r44_stage1] ${RUN} / edit${T} ---"
  python evaluation/r31_stage1.py \
    --edit_type "$T" \
    --method "$METHOD" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --latent_dir "$LATENT_DIR" \
    --vp_mode novp \
    --dump_steps "$DUMP_STEPS" \
    --measure_index "$MEASURE_INDEX" \
    --step "$STEPS" \
    --flow_shift 1.0 \
    --fg_boost_factor 4 \
    --blend_power 2 \
    --blend_sched "$SCHED" \
    --rollout_chunk_size 21 \
    --seed 0 \
    || { echo "[r44_stage1] FAILED ${RUN} / edit${T}"; FAILED=$((FAILED+1)); }
done

# ---- post-run guard 1: 419 latents for THIS run, with the exact per-type breakdown.
declare -A EXPECT=([1]=100 [2]=100 [3]=100 [4]=100 [5]=9 [6]=10)
N_TOTAL=0
BAD_TYPES=0
for T in 1 2 3 4 5 6; do
  NL=$(ls "$LATENT_DIR/edit${T}"/*.npz 2>/dev/null | wc -l)
  N_TOTAL=$((N_TOTAL + NL))
  STATUS="ok"
  if [ "$NL" -ne "${EXPECT[$T]}" ]; then STATUS="MISMATCH (expect ${EXPECT[$T]})"; BAD_TYPES=$((BAD_TYPES+1)); fi
  echo "[r44_stage1] ${RUN} edit${T}: ${NL} latents -- ${STATUS}"
done
echo "[r44_stage1] ${RUN}: ${N_TOTAL} latents total (expect 419)"
if [ "$N_TOTAL" -ne 419 ] || [ "$BAD_TYPES" -ne 0 ]; then FAILED=$((FAILED+1)); fi

# ---- post-run guard 2: every npz is a text-only draft of THIS run's schedule and step
# count (a VP latent here would make R44's "without VP" comparison meaningless).
python - "$LATENT_DIR" "$SCHED" "$STEPS" <<'PY' || FAILED=$((FAILED+1))
import sys
from pathlib import Path
import numpy as np
root, sched, steps = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
bad = []
for p in sorted(root.glob("edit*/*.npz")):
    z = np.load(p)
    got = (str(z["vp_mode"]), str(z["blend_sched"]), int(z["n_steps"]))
    if got != ("novp", sched, steps):
        bad.append(f"{p.parent.name}/{p.stem}: vp_mode/blend_sched/n_steps = {got}")
print(f"[r44_stage1] npz check: {len(bad)} of {len(list(root.glob('edit*/*.npz')))} "
      f"not (novp, {sched}, {steps})")
for b in bad[:20]:
    print(f"[r44_stage1]   {b}")
sys.exit(1 if bad else 0)
PY

echo "[r44_stage1] failures: ${FAILED}"
exit $(( FAILED > 0 ))
