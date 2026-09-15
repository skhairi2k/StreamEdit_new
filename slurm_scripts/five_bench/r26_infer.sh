#!/bin/bash
# R26 pass 2 -- the 9-cell (tau_bg, tau_fg) spatial-tau sweep over the 22 cases.json
# pairs, vp mode. Each task reads the pass-1 grounding-mask union M_f and builds a
# per-token release exponent tau(f,p) = tau_bg + (tau_fg - tau_bg) * M_f(p), replacing
# Eq. 4's single scalar rho.
#
# ARMS = tau_bg {0,1,2} x tau_fg {2,6,10}, bg-major (tasks 0-8), plus tau_fg=50 at each
# tau_bg (tasks 9-11, added 2026-08-28), plus the 40-arm extension to tau_bg in
# {0,1,2,3,4,6,8,10,20,50} x tau_fg in {2,3,4,6,8,10,20,50} constrained to tau_bg <=
# tau_fg (tasks 12-51, added 2026-09-01). Task 6 is (2,2): the DEGENERATE
# CONTROL, a uniform field that must reproduce the global blend_power=2 schedule exactly
# -- r26_smoke.sh job 960461 gate1 proved it sha256-identical on three clips, including
# one multi-window clip. Every delta in the final table is read against that cell.
#
# EVERY tau_bg == tau_fg cell is degenerate the same way (2,2) is: a uniform field is
# mathematically identical to the scalar Eq. 4 baseline, so these arms (2,2 / 3,3 / 4,4 /
# 6,6 / 8,8 / 10,10 / 20,20 / 50,50) skip the mask/tau-field pass-2 path entirely and run
# the plain --blend_power scalar path instead -- there is nothing left to re-verify at
# each diagonal cell that gate1 did not already prove once at (2,2).
#
# tau_bg = 0 is an extreme ORACLE arm, not a schedule: W^src = t**0 == 1 at EVERY step
# including the final t = 0, so background tokens stay pinned to the source for the whole
# rollout and never release. Expect it to score well on background metrics while
# potentially looking wrong (seams, freezing) -- the verdict step must check the grids,
# not just the numbers.
#
# The array is over ARMS (task 0..11), and each task loops all 6 edit types INTERNALLY so
# the model loads once per arm rather than once per type. Clip counts per type are
# 1/13/2/2/3/1 = 22.
#
# COVERAGE IS A HARD ERROR, not a fallback. run_fivebench.py raises FileNotFoundError on
# a missing mask, but that raise sits INSIDE the per-pair try/except, so it is recorded
# as an error row and the loop CONTINUES -- a partial dump would silently yield a short
# arm that still writes a manifest and still produces a full 16-metric table, and nothing
# downstream could tell. The preflight below counts the npz for every edit type and
# refuses to load the model unless all 22 are present.
#
# Sampler config, seed, anchors and cases_json MUST be identical to r26_dump.sh and
# r26_smoke.sh -- step 15, flow_shift 1.0, fg_boost 4, seed 0, chunk 21, overlap 1,
# sink 0. Per-pair reseeding (run_fivebench.py) is what makes a 22-pair subset comparable
# at all, so none of the three may drift between the two passes.
#
# --blend_power is deliberately NOT passed: the tau field supplies the exponent for every
# token, and passing both would imply a fallback exists. run_fivebench.py rejects
# --union_mask_dir together with --blend_sched or --tau_map for the same reason.
#
# --time=06:00:00: 22 clips at R22's measured ~5.3 min/clip is ~2h per arm, plus 6 model
# loads. 6h is wide margin.
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
# tau_fg=50 (tasks 9-11, 2026-08-28): an EXTREME endpoint for the tau_fg axis, not a
# candidate setting. W^src(t) = t**50 is 3.1e-2 at the first blend (t_next=0.933) and
# <1e-3 from the second step on, so the edit region is effectively released from the
# source for the whole rollout -- close to blend_off INSIDE the mask, while the
# background keeps its tau_bg schedule. Added because tau_fg 6 -> 10 already flattened
# (CLIP-T 28.4938 -> 28.4050 at tau_bg=2) and a paired test cannot separate any
# tau_fg>=6 cell; these three say whether the axis saturates or finally breaks.
#SBATCH --job-name=r26_infer
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-51
#SBATCH --output=logs/r26_infer_%A_%a.out
#SBATCH --error=logs/r26_infer_%A_%a.err
# TRANSIENT WORKAROUND (2026-08-27) -- remove once node52 is fixed. On r26_dump job
# 960708 that node handed out DUPLICATE GPUs (two tasks given the same UUID, two others
# given a GPU number with no device behind it) and all 5 tasks that landed there died at
# torch.cuda.current_device(). R25 job 960685 failed identically. See r26_dump.sh.
#SBATCH --exclude=node52

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
OUT_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau
MASK_ROOT=/projects/dataggen/outputs/five_bench/r26_masks
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES="$REPO/evaluation/cases.json"

# tau_bg {0,1,2} x tau_fg {2,6,10}, bg-major. Task 6 = (2,2) = the control.
# Tasks 12-51 (added 2026-09-01): extended to tau_bg in {0,1,2,3,4,6,8,10,20,50} x
# tau_fg in {2,3,4,6,8,10,20,50}, CONSTRAINED to tau_bg <= tau_fg only (the inverted
# region -- background released faster than the edit region -- is outside the method's
# intended semantics and was deliberately excluded; see the plan's Decisions table).
# 0-11 keep their original mapping and outputs (directories are keyed by METHOD name,
# not array index, so appending is safe); 12-51 are the 40 new arms, bg-major, fg
# ascending, skipping any (bg,fg) pair already present in 0-11.
ARMS=("0 2" "0 6" "0 10"
      "1 2" "1 6" "1 10"
      "2 2" "2 6" "2 10"
      "0 50" "1 50" "2 50"
      "0 3" "0 4" "0 8" "0 20"
      "1 3" "1 4" "1 8" "1 20"
      "2 3" "2 4" "2 8" "2 20"
      "3 3" "3 4" "3 6" "3 8" "3 10" "3 20" "3 50"
      "4 4" "4 6" "4 8" "4 10" "4 20" "4 50"
      "6 6" "6 8" "6 10" "6 20" "6 50"
      "8 8" "8 10" "8 20" "8 50"
      "10 10" "10 20" "10 50"
      "20 20" "20 50"
      "50 50")

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -gt 51 ]; then
  echo "[r26_infer] bad array id ${TID} (expected 0-51)"; exit 1
fi
read -r BG FG <<< "${ARMS[$TID]}"
METHOD="taubg${BG}_taufg${FG}_vp"

# Expected clip count per edit type in cases.json (1/13/2/2/3/1 = 22).
EXPECTED=(1 13 2 2 3 1)
TOTAL_EXPECTED=22

# Conda: the pattern proven to work from this machine (r26_smoke.sh job 960461).
# `cd; source .bashrc` is written for a LOGIN shell and killed R25 array job 960685
# when submitted from inside a GPU allocation.
cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null         # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p logs "$OUT_ROOT"

# Diagnostics: makes a node-level GPU fault diagnosable in one pass instead of a guess.
echo "[r26_infer] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[r26_infer] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi -L 2>&1 | head -4
python -c "import torch; print('[r26_infer] torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), 'n', torch.cuda.device_count())" 2>&1 | tail -2

echo "[r26_infer] job=${SLURM_ARRAY_JOB_ID}_${TID} arm=${METHOD} tau_bg=${BG} tau_fg=${FG}"
DEGENERATE=0
if [ "$BG" = "$FG" ]; then
  DEGENERATE=1
  echo "[r26_infer] NOTE this is a DEGENERATE cell (tau_bg == tau_fg == ${BG}): a uniform"
  echo "[r26_infer]      field is mathematically identical to the plain scalar Eq. 4 baseline"
  echo "[r26_infer]      (--blend_power ${BG}), so this arm skips the mask/tau-field pass-2"
  echo "[r26_infer]      machinery entirely and runs the single-pass scalar path instead --"
  echo "[r26_infer]      r26_smoke.sh job 960461 gate1 already proved the two paths agree"
  echo "[r26_infer]      sha256-identically at (2,2), so there is nothing left to verify by"
  echo "[r26_infer]      re-deriving every diagonal cell through the two-pass field."
fi

# ---- preflight: full mask coverage, BEFORE the model loads (skipped for degenerate
# cells, which never touch the mask/tau-field path) ----
if [ "$DEGENERATE" -eq 0 ]; then
  MISSING=0
  for T in 1 2 3 4 5 6; do
    EXP=${EXPECTED[$((T-1))]}
    N=$(ls "$MASK_ROOT"/"edit${T}"/*.npz 2>/dev/null | wc -l)
    echo "[r26_infer] masks edit${T}: ${N} (expected ${EXP})"
    if [ "$N" -ne "$EXP" ]; then MISSING=$((MISSING+1)); fi
  done
  if [ "$MISSING" -ne 0 ]; then
    echo "[r26_infer] ABORT -- pass-1 masks incomplete in $MASK_ROOT (${MISSING} edit type(s) short)."
    echo "[r26_infer]   run_fivebench.py would record a per-pair error and CONTINUE, producing a"
    echo "[r26_infer]   short arm with a full-looking metrics table. Re-run r26_dump.sh first."
    exit 1
  fi
fi
if [ ! -f "$CASES" ]; then echo "[r26_infer] missing $CASES"; exit 1; fi

# ---- render all 6 edit types with the model loaded once per invocation ----
# Degenerate cells (tau_bg == tau_fg) use --blend_power and skip --union_mask_dir/
# --tau_bg/--tau_fg entirely -- run_fivebench.py's own cross-validation (SystemExit on
# --tau_bg/--tau_fg without --union_mask_dir) would otherwise reject a half-specified
# call, and there is no reason to specify it here since the scalar path is exact.
FAILED=0
for T in 1 2 3 4 5 6; do
  echo "--- [r26_infer] ${METHOD} edit${T} (expect ${EXPECTED[$((T-1))]} clip(s)) ---"
  if [ "$DEGENERATE" -eq 1 ]; then
    python evaluation/run_fivebench.py \
      --edit_type "$T" --method "$METHOD" \
      --vp_mode vp \
      --first_frame_edit_dir "$ANCHOR_ROOT" \
      --cases_json "$CASES" \
      --blend_power "$BG" \
      --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
      --step 15 --fg_boost_factor 4 --seed 0 \
      || { echo "[r26_infer] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
    continue
  fi
  python evaluation/run_fivebench.py \
    --edit_type "$T" --method "$METHOD" \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --cases_json "$CASES" \
    --union_mask_dir "$MASK_ROOT" \
    --tau_bg "$BG" --tau_fg "$FG" \
    --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --seed 0 \
    || { echo "[r26_infer] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
done

# Frame dirs across ALL edit types for this arm, excluding the *_resize dirs
# evaluate.py leaves behind.
N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/*/*/ 2>/dev/null | grep -vc '_resize/$')
echo "[r26_infer] ${METHOD} total frame dirs: ${N_DIRS} (expected ${TOTAL_EXPECTED})"
if [ "$N_DIRS" -ne "$TOTAL_EXPECTED" ] && [ "$FAILED" -eq 0 ]; then
  echo "[r26_infer] COUNT-MISMATCH ${METHOD}: ${N_DIRS} != ${TOTAL_EXPECTED}"
  FAILED=$((FAILED+1))
fi

# A per-pair error is recorded in the manifest rather than raised, so a clean exit code
# is not evidence the arm is complete. Count the non-ok rows directly.
N_BAD=$(cat "$OUT_ROOT"/"${METHOD}"/edit*/_manifest.csv 2>/dev/null \
        | awk -F, 'NR>1 && $3!="ok" && $3!="status"' | wc -l)
echo "[r26_infer] ${METHOD} non-ok manifest rows: ${N_BAD}"
if [ "$N_BAD" -ne 0 ] && [ "$FAILED" -eq 0 ]; then
  echo "[r26_infer] MANIFEST-ERRORS ${METHOD}: ${N_BAD} pair(s) did not render"
  FAILED=$((FAILED+1))
fi

echo "[r26_infer] failures: ${FAILED}"
exit $(( FAILED > 0 ))
