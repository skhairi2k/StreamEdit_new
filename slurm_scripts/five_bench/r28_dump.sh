#!/bin/bash
# R28 step 1 -- dump per-arm TARGET masks for the 54 arms feeding the union metrics.
#
# 54 arms, one per array task: ALL 52 R26 (tau_bg, tau_fg) cells (extended 2026-09-01
# from the original 12 to tau_bg in {0,1,2,3,4,6,8,10,20,50} x tau_fg in
# {2,3,4,6,8,10,20,50}, constrained to tau_bg <= tau_fg) plus `baseline` (novp Eq.4) and
# `r7_visual_prompting` (vp Eq.4). The two extra arms are dumped ONLY to widen the FIXED
# cross-arm union -- they are never scored by r28_eval.sh. Both were verified 22/22 on the
# same edit{T}/{video} layout as the R26 arms.
#
# Extending to 52 cells means the FIXED union region shifts for every arm, old and new
# alike (it can only grow, never shrink -- r28_fixed_union.py's docstring), so
# r28_fixed_union.py and r28_eval.sh must both be re-run over the FULL 54-arm set, not
# just the 40 new cells -- a partial rebuild would score old arms against a union that no
# longer matches what r28_summarize.py's cached numbers assumed.
#
# ARMS order is load-bearing for nothing here (each task is independent and writes to its
# own arm dir), but it MUST stay a superset of r28_eval.sh's list or the fixed union would
# be built from arms that are never scored, and vice versa.
#
# --exclude=node52: that node advertises gpu:8 but exposes no device to batch jobs
# (nvidia-smi -L -> "No devices found."), which killed two r25_infer submissions.
#SBATCH --job-name=r28_dump
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-53
#SBATCH --output=logs/r28_dump_%A_%a.out
#SBATCH --error=logs/r28_dump_%A_%a.err

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

DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R26_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau
FB_ROOT=/projects/dataggen/outputs/five_bench
OUT_ROOT=/projects/dataggen/outputs/five_bench/r28_tgt_masks
CASES=evaluation/cases.json
STRIDE=8

ARMS=("$R26_ROOT/taubg0_taufg2_vp" "$R26_ROOT/taubg0_taufg3_vp" "$R26_ROOT/taubg0_taufg4_vp" "$R26_ROOT/taubg0_taufg6_vp"
      "$R26_ROOT/taubg0_taufg8_vp" "$R26_ROOT/taubg0_taufg10_vp" "$R26_ROOT/taubg0_taufg20_vp" "$R26_ROOT/taubg0_taufg50_vp"
      "$R26_ROOT/taubg1_taufg2_vp" "$R26_ROOT/taubg1_taufg3_vp" "$R26_ROOT/taubg1_taufg4_vp" "$R26_ROOT/taubg1_taufg6_vp"
      "$R26_ROOT/taubg1_taufg8_vp" "$R26_ROOT/taubg1_taufg10_vp" "$R26_ROOT/taubg1_taufg20_vp" "$R26_ROOT/taubg1_taufg50_vp"
      "$R26_ROOT/taubg2_taufg2_vp" "$R26_ROOT/taubg2_taufg3_vp" "$R26_ROOT/taubg2_taufg4_vp" "$R26_ROOT/taubg2_taufg6_vp"
      "$R26_ROOT/taubg2_taufg8_vp" "$R26_ROOT/taubg2_taufg10_vp" "$R26_ROOT/taubg2_taufg20_vp" "$R26_ROOT/taubg2_taufg50_vp"
      "$R26_ROOT/taubg3_taufg3_vp" "$R26_ROOT/taubg3_taufg4_vp" "$R26_ROOT/taubg3_taufg6_vp" "$R26_ROOT/taubg3_taufg8_vp"
      "$R26_ROOT/taubg3_taufg10_vp" "$R26_ROOT/taubg3_taufg20_vp" "$R26_ROOT/taubg3_taufg50_vp" "$R26_ROOT/taubg4_taufg4_vp"
      "$R26_ROOT/taubg4_taufg6_vp" "$R26_ROOT/taubg4_taufg8_vp" "$R26_ROOT/taubg4_taufg10_vp" "$R26_ROOT/taubg4_taufg20_vp"
      "$R26_ROOT/taubg4_taufg50_vp" "$R26_ROOT/taubg6_taufg6_vp" "$R26_ROOT/taubg6_taufg8_vp" "$R26_ROOT/taubg6_taufg10_vp"
      "$R26_ROOT/taubg6_taufg20_vp" "$R26_ROOT/taubg6_taufg50_vp" "$R26_ROOT/taubg8_taufg8_vp" "$R26_ROOT/taubg8_taufg10_vp"
      "$R26_ROOT/taubg8_taufg20_vp" "$R26_ROOT/taubg8_taufg50_vp" "$R26_ROOT/taubg10_taufg10_vp" "$R26_ROOT/taubg10_taufg20_vp"
      "$R26_ROOT/taubg10_taufg50_vp" "$R26_ROOT/taubg20_taufg20_vp" "$R26_ROOT/taubg20_taufg50_vp" "$R26_ROOT/taubg50_taufg50_vp"
      "$FB_ROOT/baseline"           "$FB_ROOT/r7_visual_prompting")

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 53 ]; then
  echo "[r28_dump] bad array id ${TID} (expected 0-53)"; exit 1
fi
ARM_ROOT="${ARMS[$TID]}"
NAME=$(basename "$ARM_ROOT")

# Idempotent skip: grounding is per-arm and independent of every other arm, so an arm
# whose masks already exist from a prior dump (e.g. the original 14 R28 arms, before the
# 2026-09-01 extension to 54) needs no re-grounding -- its M_tgt does not change when
# OTHER arms are added. This is what makes it safe and cheap to resubmit the FULL
# --array=0-53 after extending the arm set, rather than hand-computing which indices are
# new against a differently-ordered ARMS list. No GPU/model load happens for a skip.
N_EXISTING=$(ls "$OUT_ROOT/$NAME"/edit*/*.npz 2>/dev/null | wc -l)
if [ "$N_EXISTING" -eq 22 ]; then
  echo "[r28_dump] task=${TID} arm=${NAME}: 22/22 npz already present, skipping (idempotent)"
  echo "[r28_dump] failures: 0"
  exit 0
fi

echo "[r28_dump] node=$(hostname) task=${TID} arm=${NAME}"
nvidia-smi -L 2>&1 | head -2
$PY -c "import torch; print('[r28_dump] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available())"

if [ ! -d "$ARM_ROOT" ]; then
  echo "[r28_dump] MISSING arm root $ARM_ROOT"; exit 1
fi

$PY evaluation/r28_target_masks.py \
  --cases "$CASES" \
  --data_root "$DATA_ROOT" \
  --tgt_root "$ARM_ROOT" \
  --out_dir "$OUT_ROOT/$NAME" \
  --frame_stride $STRIDE \
  --seed 0
RC=$?

N=$(ls "$OUT_ROOT/$NAME"/edit*/*.npz 2>/dev/null | wc -l)
echo "[r28_dump] ${NAME}: ${N} npz written (expected 22)"
if [ "$N" -ne 22 ] || [ "$RC" -ne 0 ]; then
  echo "[r28_dump] failures: 1"
  echo "[r28_dump] FAILED ${NAME} (rc=${RC}, npz=${N})"
  exit 1
fi
echo "[r28_dump] failures: 0"
