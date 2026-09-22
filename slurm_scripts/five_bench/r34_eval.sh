#!/bin/bash
# R34 -- LPIPS + CLIP-target on the FIRST CHUNK ONLY, array over R26's 16 constant-b arms
# + R31's 4 divergence arms (20 tasks). Same arm roster and dispatch as r33_fiveacc.sh;
# the only substantive difference is the frame window.
#
# THE WINDOW. num_frame_per_block=3 (configs/self_forcing_dmd.yaml:48), so the first
# rollout chunk is 3 latent frames = pixel frames 0-8 (1 + 4 + 4). Clip LENGTH varies --
# find_closest_num_frame(x, a=4, b=3) returns 12m-3, so an 80-frame source renders 69
# pixel frames, not 81 -- but the first chunk is 9 pixel frames for EVERY clip, which is
# why --max_frames 9 is a constant here and not a per-clip lookup. Verified on 0001_bus
# by r34_smoke.sh (job 1004514, GATE0: 69 frames = 18 latent = 6 chunks).
#
# WHY --frame_stride 1 AND NOT 8. The stored whole-video CSVs were scored at stride 8,
# whose grid inside this window is just {0, 8} -- two frames, far too few for LPIPS or
# CLIP. Stride 1 over 9 frames costs about what stride 8 over a whole clip did (11
# frames). The consequence is that these numbers are NOT comparable frame-for-frame with
# the stored stride-8 tables; r34_score.py joins the stored numbers as a separate
# whole-video column and reports the delta, rather than pretending one is a subset of
# the other.
#
# WHY ONLY TWO METRICS. R34's stated scope is LPIPS, CLIP-target and CLIP-D. The first
# two come from here; CLIP-D is a separate model/script and runs concurrently in
# r34_clip_directional.sh. Deliberately NOT byte-identical to r26_eval.sh/r31_eval.sh's
# 9-metric list -- R34 does not overlay onto R26's stored panel, it rebuilds its own on a
# different frame window, so matching that list would only buy runtime.
#
# --per_frame is on: the per-frame values are free here (calculate_mean would otherwise
# discard them) and let any sub-window question be answered later without re-running the
# models.
#
# ⚠️ --metrics MUST BE PASSED ON THE COMMAND LINE. The `metrics:` key in config.yaml is
# VESTIGIAL: evaluate.py reads `metrics = args.metrics`, i.e. the argparse flag whose
# hardcoded default includes five_acc and motion_fidelity. R26's job 961342 was submitted
# believing a reduced config.yaml would take effect; it did not, both models loaded, and
# motion_fidelity_score OOMed. Editing a config file does NOT change which metrics run.
#
# ⚠️ evaluate.py SWALLOWS per-metric exceptions and still exits 0, so a clean exit code is
# NOT evidence the CSV is clean -- any `Error:` line means a metric was dropped and that
# row's later columns are shifted. The log is teed and grepped below, as r26/r31/r33 do.
#
# STEM: evaluate.py derives evaluation/csv/{stem}_avg.csv itself (plus per-edit-type
# intermediates edit{T}_FiVE_{stem}_frame_stride1.csv), so a stem ending in _avg would
# yield {stem}_avg_avg.csv. Stems here are r34_{method}_chunk1 -- the window is IN the
# stem so these can never be confused with a whole-video run of the same arm.
#SBATCH --job-name=r34_eval
# RENDER ROOTS LIVE IN ~/Data, NOT /projects. R33 lost five rounds of jobs (2026-09-21,
# jobs 1002742/1002751/1002959 and the clip-d attempts before them) to a stale
# /projects/dataggen mount: each retry excluded the previous round's bad node and each
# time a NEW idle node (node57, node51 -> node01 -> node02) had the same problem, while
# nodes already holding warm jobs saw it fine. Exclude-and-retry never converged. Fixed
# 2026-09-22 with the home-dir copy, which is reliably mounted everywhere. Do not point
# this back at /projects. Node exclusions kept defensively (node52 for the older
# gpu:8-but-no-device fault; the other three are R33's confirmed-bad list).
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
# 3h, not r31_eval.sh's 6h: two light metrics (LPIPS + CLIPScore) over 9 frames x 22
# clips, with neither Qwen2.5-VL-7B nor CoTracker loaded.
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/r34_eval_%A_%a.out
#SBATCH --error=logs/r34_eval_%A_%a.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R26_ROOT=~/Data/dataggen/outputs/five_bench/r26_spatial_tau
R31_ROOT=~/Data/dataggen/outputs/five_bench/r31_arms
CASES="$REPO/evaluation/cases.json"

STRIDE=1
MAXF=9          # 3 latent frames = 1 + 4 + 4 pixel frames; see THE WINDOW above

# CONST_BS, same 8 values as r30_score.py / r26_eval.sh. Tasks 0-7: R26 SPATIAL
# (taubg0_taufg{b}_vp, background fully released, blending in the fg region only).
# Tasks 8-15: R26 UNIFORM (taubg{b}_taufg{b}_vp, tau_bg == tau_fg, the plain scalar
# Eq. 4 rho path). Tasks 16-19: R31's 4 divergence arms (depth dropped upstream at
# R31's stage 2 -- see r31_eval.sh).
BS=(2 3 4 6 8 10 20 50)
R31_ARMS=(lpips dino_patch normals latent)

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 19 ]; then
  echo "[r34_eval] bad array id ${TID} (expected 0-19)"; exit 1
fi

# ARM -> DIR SHAPE DIFFERS PER TASK GROUP. R26 arms sit at
# r26_spatial_tau/{method}/edit{T}/{video}/ directly; R31 arms sit one level deeper at
# r31_arms/r31_{arm}/step14/edit{T}/{video}/ (stage 3's FINAL render of its 15-step
# rollout). One formula cannot express both shapes, so the dispatch resolves the full
# DIR per task.
if [ "$TID" -lt 8 ]; then
  B=${BS[$TID]}
  METHOD="taubg0_taufg${B}_vp"
  DIR="$R26_ROOT/$METHOD"
elif [ "$TID" -lt 16 ]; then
  B=${BS[$((TID - 8))]}
  METHOD="taubg${B}_taufg${B}_vp"
  DIR="$R26_ROOT/$METHOD"
else
  ARM=${R31_ARMS[$((TID - 16))]}
  METHOD="r31_${ARM}"
  DIR="$R31_ROOT/r31_${ARM}/step14"
fi
STEM="r34_${METHOD}_chunk1"

# Conda: the eval env is five-bench, NOT streamgve (the R2 env bug: .bashrc
# auto-activates streamgve, and `conda activate five-bench` alone does not pop it,
# which crashed the R2 eval on ModuleNotFoundError: torchmetrics).
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # R20/R21 metric-crash fix
export PYTHONUNBUFFERED=1

echo "[r34_eval] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
echo "[r34_eval] job=${SLURM_ARRAY_JOB_ID}_${TID} method=${METHOD} stem=${STEM}"
echo "[r34_eval] scoring $DIR  (stride ${STRIDE}, first ${MAXF} frames)"

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

if [ ! -d "$DIR" ]; then
  echo "[r34_eval] FAILED ${STEM} -- arm dir does not exist: $DIR"
  echo "[r34_eval]   (is ~/Data/dataggen present on this node?)"
  exit 1
fi
# Count REAL pairs only: evaluate.py writes a sibling {video}_resize dir next to every
# video it scores, so a bare `ls` over an already-scored dir over-counts.
NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r34_eval] real frame dirs in $DIR: ${NDIRS}  # expect 22"
if [ "$NDIRS" -ne 22 ]; then
  echo "[r34_eval] FAILED ${STEM} -- expected 22 real pairs, found ${NDIRS}."
  echo "[r34_eval]   Scoring a short arm would still write a full-looking table. Stop."
  exit 1
fi

# R34-SPECIFIC GUARD. evaluate.py truncates to the first MAXF frames silently: a clip
# rendered short would be scored on whatever it has and averaged in as if it were a full
# chunk. Nothing downstream could tell the difference, so check the shortest clip here.
MINFR=$(for d in $(ls -d "$DIR"/*/*/ 2>/dev/null | grep -v '_resize/$'); do
          ls "$d" 2>/dev/null | grep -cE '\.(png|jpg|jpeg)$'
        done | sort -n | head -1)
echo "[r34_eval] shortest clip in $DIR: ${MINFR} frames  # need >= ${MAXF}"
if [ -z "$MINFR" ] || [ "$MINFR" -lt "$MAXF" ]; then
  echo "[r34_eval] FAILED ${STEM} -- a clip has only ${MINFR} frames, fewer than the"
  echo "[r34_eval]   ${MAXF}-frame chunk. It would be silently scored on a short window."
  exit 1
fi

METRICS_LOG="logs/r34_eval_${SLURM_ARRAY_JOB_ID}_${TID}.metrics.log"
python evaluation/fivebench/evaluate.py \
  --config_path evaluation/fivebench/config.yaml \
  --metrics lpips_unedit_part clip_similarity_target_image \
  --src_image_folder "$DATA_ROOT" \
  --annotation_mapping_files "${ANNOTATIONS[@]}" \
  --tgt_methods "$DIR" \
  --tgt_layout edit_video \
  --cases_json "$CASES" \
  --frame_stride $STRIDE --max_frames $MAXF --per_frame \
  --result_path "evaluation/csv/${STEM}.csv" \
  2>&1 | tee "$METRICS_LOG"
RC=${PIPESTATUS[0]}

NERR=$(grep -c 'Error:' "$METRICS_LOG")
NOOM=$(grep -c 'out of memory' "$METRICS_LOG")
echo "[r34_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM}"

if [ ! -f "evaluation/csv/${STEM}_avg.csv" ]; then
  echo "[r34_eval] FAILED ${STEM} -- evaluation/csv/${STEM}_avg.csv not written"
  exit 1
fi
if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then
  echo "[r34_eval] FAILED ${STEM} -- metric crashes present (see $METRICS_LOG)"
  exit 1
fi

# The window is what this whole task rests on, so assert it landed rather than trusting
# the flag: the per-frame CSV must carry exactly MAXF distinct frame indices. r34_smoke.sh
# proved this for one arm before the array was submitted; this re-checks it per arm.
PF="evaluation/csv/${STEM}_frame_stride${STRIDE}_per_frame.csv"
if [ ! -f "$PF" ]; then
  echo "[r34_eval] FAILED ${STEM} -- per-frame CSV not written: $PF"
  exit 1
fi
NIDX=$(tail -n +2 "$PF" | cut -d, -f5 | sort -u | wc -l)
echo "[r34_eval] distinct frame indices in $PF: ${NIDX}  # expect ${MAXF}"
if [ "$NIDX" -ne "$MAXF" ]; then
  echo "[r34_eval] FAILED ${STEM} -- scored ${NIDX} distinct frames, expected ${MAXF}."
  exit 1
fi

echo "[r34_eval] OK ${STEM} -> evaluation/csv/${STEM}_avg.csv"
exit 0
