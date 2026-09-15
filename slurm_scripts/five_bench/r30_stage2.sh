#!/bin/bash
# R30 stage 2 -- see .claude/plans/r30-static-anchor-divergence_8b3f52d1.plan.md
#
# R30_PHASE=acc    : FiVE-Acc (Qwen2.5-VL-7B) over R26's 176 STORED videos
#                    (taubg0_taufg{2,3,4,6,8,10,20,50}_vp x 22 clips). Builds the
#                    constant-b operating curve and the per-clip oracle. This is the
#                    GO/NO-GO input: it renders NOTHING and depends on NOTHING from
#                    stage 1 -- if the oracle does not clear N_curve(P_oracle), no
#                    per-clip routing can win and the 176 R30 rollouts are pointless.
#                    SUBMIT: R30_PHASE=acc sbatch slurm_scripts/five_bench/r30_stage2.sh
#
# R30_PHASE=render : array 0-7 over the eight arms, each task rendering ALL 22 clips at
#                    that arm's continuous b via run_fivebench.py --union_mask_dir
#                    --tau_bg 0 --tau_map evaluation/csv/r30_b_map_{arm}.csv. #SBATCH
#                    pragmas are static text read at submit time and cannot branch on
#                    R30_PHASE, so --array MUST be supplied on the sbatch command line --
#                    and so must --output/--error: the file's own #SBATCH --output uses
#                    %j, which for an ARRAY job resolves to the SAME value across every
#                    task, so all 8 tasks silently wrote to ONE shared file and clobbered
#                    each other (job 988820, 2026-09-10 -- only task 7's stdout survived;
#                    ground truth had to come from the rendered directories instead of the
#                    logs). %j stays correct for PHASE=acc/eval, which are single-task, so
#                    the fix is a per-submission override, not a file-wide default change:
#                    SUBMIT: R30_PHASE=render sbatch --array=0-7 \
#                              --output=logs/r30_stage2_%A_%a.out \
#                              --error=logs/r30_stage2_%A_%a.err \
#                              slurm_scripts/five_bench/r30_stage2.sh
#                    Masks are R26's r26_masks (NOT a separate r30_masks -- there is no
#                    such directory; stage 1 reuses r26_masks unchanged and render must
#                    read the identical masks it measured divergence over). An arm whose
#                    b map is missing or short is SKIPPED by its own task with a clear
#                    message, not silently rendered short. depth was disqualified this way
#                    on 0042_gym-ball through job 988820 (2026-09-10) -- its union covered
#                    100% of the frame, no background left to fit depth's affine alignment
#                    on -- but the mask itself was corrected the same day (see
#                    r30_divergence.py's DEGENERATE-TARGET FALLBACK: 0042_gym-ball is a
#                    REMOVAL edit whose trg_word is a negation with no visual referent, so
#                    trg grounding never localizes and saturates to exactly 1.0 on every
#                    frame -- confirmed by dumping M_src/M_trg separately, a small
#                    edit_causal_inference.py patch, run once for this one clip). depth's b
#                    map now covers all 22 clips; its render followed up as a standalone
#                    --array=5-5 resubmission (job 988841) after the other 7 arms had
#                    already finished under 988820, rather than re-rendering everything.
#
# R30_PHASE=eval   : FiVE-Acc + the 9-metric harness (the same 9 metrics r26_eval.sh
#                    scores: structure_distance, psnr/lpips/mse/ssim_unedit_part, both
#                    CLIP similarities, clip_similarity_target_image_edit_part, niqe)
#                    over the 176 rendered r30_arms/{arm}/edit{T}/ videos -- ONE task,
#                    looping over the eight arms sequentially, same shape as PHASE=acc
#                    (not an array: the plan's own step command has no --array, and each
#                    evaluate.py call is a fresh subprocess regardless of loop vs array,
#                    so nothing is gained by parallelising this one). An arm with no
#                    rendered directory (e.g. depth, disqualified at stage 1) is SKIPPED,
#                    not failed -- it legitimately has nothing to score.
#                    SUBMIT: R30_PHASE=eval sbatch slurm_scripts/five_bench/r30_stage2.sh
#                    NO TODO SPECCED THIS PHASE (stage2-eval has no matching-id todo in
#                    the plan, unlike acc/render); implemented by inference from the
#                    plan's "FiVE-Acc + 9-metric harness" wording, r26_eval.sh's metric
#                    list, and PHASE=acc's own join pattern (2026-09-10).
#
# five_acc is whole-clip and expands to FIVE csv columns per method
# (evaluate.py:332): five_acc_yes_no, _multi_choice, _union, _inter, and the
# combined five_acc. yn_acc is the primary binary (the plan's Significance row):
# the two booleans are correlated, so pooling them to 44 overstates significance.
#
#SBATCH --job-name=r30_stage2
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/r30_stage2_%j.out
#SBATCH --error=logs/r30_stage2_%j.err
# --time widened 10h -> 16h (2026-09-10) for PHASE=eval: acc alone (five_acc only, 8
# arms x 22 clips) fit comfortably under 10h, but eval adds the 9-metric harness on TOP
# of five_acc for each of the 8 arms -- r26_eval.sh budgeted 6h for that harness ALONE
# on ONE arm (and reported using only ~5% of it), so this is a generous, undemonstrated
# estimate, not a measurement. #SBATCH pragmas are shared across phases (static text,
# cannot branch on R30_PHASE); 16h costs acc nothing since it already finishes in a
# fraction of 10h.

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (R20 job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
R26_ROOT=/projects/dataggen/outputs/five_bench/r26_spatial_tau
CASES="$REPO/evaluation/cases.json"
PHASE=${R30_PHASE:-acc}

cd "$REPO" || exit 1

if [ "$PHASE" != "acc" ] && [ "$PHASE" != "render" ] && [ "$PHASE" != "eval" ]; then
  echo "[r30_stage2] unknown R30_PHASE=${PHASE} (expected acc, render, or eval)"
  exit 2
fi

# ========================================================================================
# PHASE=render -- array 0-7 over the eight arms. Own conda env (the diffusion pipeline,
# not five-bench), own body, exits at the end of this block. Everything below this block
# is PHASE=acc only.
# ========================================================================================
if [ "$PHASE" = "render" ]; then
  ARMS=(lpips dino_cls dino_patch clip_image clip_prompt depth normals selfsim)

  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[r30_stage2] FAILED -- PHASE=render must be submitted as an ARRAY JOB:"
    echo "[r30_stage2]   R30_PHASE=render sbatch --array=0-7 slurm_scripts/five_bench/r30_stage2.sh"
    exit 1
  fi
  TID=${SLURM_ARRAY_TASK_ID}
  if [ "$TID" -lt 0 ] || [ "$TID" -gt 7 ]; then
    echo "[r30_stage2] FAILED -- bad array id ${TID} (expected 0-7, one per arm)"
    exit 1
  fi
  ARM=${ARMS[$TID]}
  BMAP="$REPO/evaluation/csv/r30_b_map_${ARM}.csv"
  MASK_ROOT=/projects/dataggen/outputs/five_bench/r26_masks
  ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
  OUT_ROOT=/projects/dataggen/outputs/five_bench/r30_arms
  METHOD="$ARM"

  echo "[r30_stage2] === render task ${TID}: arm=${ARM} ==="

  # Preflight, BEFORE the model loads: refuse unless this arm's b map exists and covers
  # all 22 pairs. r30_b_map.py itself REFUSES TO WRITE a map for an arm missing any
  # clip's divergence (e.g. depth on 0042_gym-ball, whose union mask is the whole frame
  # with no background to align the affine fit on) -- a missing file here means "this
  # arm cannot be rendered", not a transient error to retry.
  if [ ! -f "$BMAP" ]; then
    echo "[r30_stage2] SKIP ${ARM} -- no b map at ${BMAP}."
    echo "[r30_stage2]   r30_b_map.py refused to write one for this arm (see"
    echo "[r30_stage2]   logs/r30_stage1_*.out); it does not cover all 22 clips."
    exit 1
  fi
  N_BMAP=$(($(wc -l < "$BMAP") - 1))
  if [ "$N_BMAP" -ne 22 ]; then
    echo "[r30_stage2] FAILED ${ARM} -- b map has ${N_BMAP} rows, expected 22."
    exit 1
  fi
  if [ ! -d "$MASK_ROOT" ] || [ ! -f "$CASES" ]; then
    echo "[r30_stage2] FAILED ${ARM} -- missing $MASK_ROOT or $CASES."
    exit 1
  fi

  # streamgve, not five-bench: this phase runs the diffusion pipeline, matching R25/
  # R26/R27's render passes -- five-bench (below, for PHASE=acc) has no Wan/Self-Forcing
  # build on its path and would fail at import.
  source ~/anaconda3/etc/profile.d/conda.sh
  conda deactivate 2>/dev/null
  conda activate streamgve

  mkdir -p logs "$OUT_ROOT"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  echo "[r30_stage2] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi -L 2>&1 | head -4

  # Expected clip count per edit type in cases.json (1/13/2/2/3/1 = 22), same table
  # r26_infer.sh uses -- the case set has not changed since R26's dump.
  EXPECTED=(1 13 2 2 3 1)
  TOTAL_EXPECTED=22
  FAILED=0
  for T in 1 2 3 4 5 6; do
    echo "--- [r30_stage2] ${METHOD} edit${T} (expect ${EXPECTED[$((T-1))]} clip(s)) ---"
    python evaluation/run_fivebench.py \
      --edit_type "$T" --method "$METHOD" \
      --vp_mode vp \
      --first_frame_edit_dir "$ANCHOR_ROOT" \
      --cases_json "$CASES" \
      --union_mask_dir "$MASK_ROOT" \
      --tau_bg 0 \
      --tau_map "$BMAP" \
      --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
      --step 15 --fg_boost_factor 4 --seed 0 \
      || { echo "[r30_stage2] FAILED ${METHOD} edit${T}"; FAILED=$((FAILED+1)); }
  done

  # Frame dirs across ALL edit types for this arm, excluding the *_resize dirs
  # evaluate.py leaves behind.
  N_DIRS=$(ls -d "$OUT_ROOT"/"${METHOD}"/*/*/ 2>/dev/null | grep -vc '_resize/$')
  echo "[r30_stage2] ${METHOD} total frame dirs: ${N_DIRS} (expected ${TOTAL_EXPECTED})"
  if [ "$N_DIRS" -ne "$TOTAL_EXPECTED" ] && [ "$FAILED" -eq 0 ]; then
    echo "[r30_stage2] COUNT-MISMATCH ${METHOD}: ${N_DIRS} != ${TOTAL_EXPECTED}"
    FAILED=$((FAILED+1))
  fi

  # A per-pair error is recorded in the manifest rather than raised, so a clean exit code
  # is not evidence the arm is complete. Count the non-ok rows directly.
  N_BAD=$(cat "$OUT_ROOT"/"${METHOD}"/edit*/_manifest.csv 2>/dev/null \
          | awk -F, 'NR>1 && $3!="ok" && $3!="status"' | wc -l)
  echo "[r30_stage2] ${METHOD} non-ok manifest rows: ${N_BAD}"
  if [ "$N_BAD" -ne 0 ] && [ "$FAILED" -eq 0 ]; then
    echo "[r30_stage2] MANIFEST-ERRORS ${METHOD}: ${N_BAD} pair(s) did not render"
    FAILED=$((FAILED+1))
  fi

  echo "[r30_stage2] ${ARM} failures: ${FAILED}"
  exit $(( FAILED > 0 ))
fi

# ========================================================================================
# PHASE=eval -- FiVE-Acc + the 9-metric harness over the 8 rendered arms, one task,
# looping sequentially (same shape as PHASE=acc). Own env setup (five-bench, shared with
# acc below), duplicated rather than merged because eval must run and exit BEFORE the
# acc-only b-grid join, and reads different inputs (r30_arms/, not R26's stored grid).
# ========================================================================================
if [ "$PHASE" = "eval" ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda deactivate 2>/dev/null
  conda activate five-bench

  mkdir -p logs evaluation/csv
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONUNBUFFERED=1
  export HF_HUB_OFFLINE=1

  echo "[r30_stage2] node=$(hostname) job=${SLURM_JOB_ID} phase=${PHASE}"
  nvidia-smi -L 2>&1 | head -4

  # Same guard as PHASE=acc: HF_HUB_OFFLINE=1 means a cache miss is a hard failure, and
  # metrics_calculator.py:687 SWALLOWS a load error and reports nan for every clip.
  VLM_CACHE=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct
  if [ ! -d "$VLM_CACHE" ]; then
    echo "[r30_stage2] FAILED -- Qwen2.5-VL-7B not cached at $VLM_CACHE"
    echo "[r30_stage2]   Pre-fetch on a LOGIN node, then resubmit."
    exit 1
  fi

  ANNOTATIONS=(
    "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
    "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
    "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
    "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
    "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
    "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
  )

  ARM_ROOT=/projects/dataggen/outputs/five_bench/r30_arms
  ARMS=(lpips dino_cls dino_patch clip_image clip_prompt depth normals selfsim)
  # The 9-metric harness, byte-identical to r26_eval.sh's --metrics list, plus five_acc
  # in front: this phase needs BOTH ("FiVE-Acc + 9-metric harness" per the plan), unlike
  # r26_eval.sh which split them across two jobs to fit L40S memory -- PHASE=eval instead
  # shares PHASE=acc's H100-class allocation (same #SBATCH header).
  METRICS=(five_acc structure_distance psnr_unedit_part lpips_unedit_part
           mse_unedit_part ssim_unedit_part clip_similarity_source_image
           clip_similarity_target_image clip_similarity_target_image_edit_part
           niqe_target_image)

  FAILED=0
  SCORED_ARMS=()
  for ARM in "${ARMS[@]}"; do
    DIR="$ARM_ROOT/$ARM"
    STEM="r30_eval_${ARM}"

    if [ ! -d "$DIR" ]; then
      echo "[r30_stage2] SKIP ${ARM} -- no rendered dir at $DIR (not rendered, or"
      echo "[r30_stage2]   disqualified at stage 1 -- e.g. depth on 0042_gym-ball,"
      echo "[r30_stage2]   whose union mask is the whole frame). Not a failure."
      continue
    fi
    # evaluate.py writes a sibling {video}_resize dir next to every video it scores, so a
    # bare ls over an already-scored dir over-counts.
    NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
    if [ "$NDIRS" -ne 22 ]; then
      echo "[r30_stage2] FAILED ${ARM} -- expected 22 real pairs, found ${NDIRS}."
      echo "[r30_stage2]   Scoring a short arm would still write a full-looking table."
      FAILED=1; continue
    fi

    echo "[r30_stage2] === eval arm=${ARM} (${NDIRS} pairs) ==="
    LOG="logs/r30_stage2_${SLURM_JOB_ID}_eval_${ARM}.metrics.log"
    python evaluation/fivebench/evaluate.py \
      --config_path evaluation/fivebench/config.yaml \
      --metrics "${METRICS[@]}" \
      --src_image_folder "$DATA_ROOT" \
      --annotation_mapping_files "${ANNOTATIONS[@]}" \
      --tgt_methods "$DIR" \
      --tgt_layout edit_video \
      --cases_json "$CASES" \
      --result_path "evaluation/csv/${STEM}.csv" \
      2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}

    # metrics_calculator.py:687 prints this and returns nan for every clip rather than
    # raising. Treat it as fatal: a silent all-nan table is worse than no table.
    if grep -q "five_acc will report nan" "$LOG"; then
      echo "[r30_stage2] FAILED ${ARM} -- the VLM did not load; five_acc is all nan."
      FAILED=1; continue
    fi
    NERR=$(grep -c 'Error:' "$LOG")
    echo "[r30_stage2] ${ARM} exit=${RC} error_lines=${NERR}"
    if [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; then FAILED=1; continue; fi
    SCORED_ARMS+=("$ARM")
  done

  if [ "$FAILED" -ne 0 ]; then
    echo "[r30_stage2] one or more ATTEMPTED arms failed -- NOT writing the joined csv."
    exit 1
  fi
  if [ "${#SCORED_ARMS[@]}" -eq 0 ]; then
    echo "[r30_stage2] FAILED -- no arm had a rendered directory to score."
    echo "[r30_stage2]   Run PHASE=render first."
    exit 1
  fi

  # ---- join per-arm per-edit-type tables + each arm's routed b into r30_fiveacc_arms.csv
  # Same file_id -> video_name -> case_id join as PHASE=acc's grid (file_id indexes the
  # FULL edit{T}_FiVE.json, not the cases.json subset), keyed additionally by arm.
  python - "${SCORED_ARMS[@]}" <<'PYEOF'
import csv, json, os, sys

REPO = "/home/ids/skhairi/Code/StreamEdit_bigchantier"
DATA = os.path.expanduser("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
ARMS = sys.argv[1:]
OUT = f"{REPO}/evaluation/csv/r30_fiveacc_arms.csv"
METRIC_COLS = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part",
               "mse_unedit_part", "ssim_unedit_part", "clip_similarity_source_image",
               "clip_similarity_target_image", "clip_similarity_target_image_edit_part",
               "niqe_target_image"]

cases = json.load(open(f"{REPO}/evaluation/cases.json"))
by_vid = {}
for c in cases:
    by_vid.setdefault((c["edit_type"], c["video_name"]), c["case_id"])

idx2vid = {}
for T in range(1, 7):
    d = json.load(open(f"{DATA}/edit_prompt/edit{T}_FiVE.json"))
    items = d if isinstance(d, list) else list(d.values())
    for i, x in enumerate(items):
        idx2vid[(T, str(i))] = x["video_name"]

rows, missing = [], []
for arm in ARMS:
    # Per-arm routed b, keyed by case_id -- the SAME map PHASE=render read.
    bmap_path = f"{REPO}/evaluation/csv/r30_b_map_{arm}.csv"
    b_by_case = {}
    if os.path.exists(bmap_path):
        for r in csv.DictReader(open(bmap_path)):
            b_by_case[r["case_id"]] = r["tau"]
    else:
        missing.append(f"missing b map for scored arm {arm}: {bmap_path}")

    for T in range(1, 7):
        f = f"{REPO}/evaluation/csv/edit{T}_FiVE_r30_eval_{arm}_frame_stride8.csv"
        if not os.path.exists(f):
            missing.append(f); continue
        for r in csv.DictReader(open(f)):
            fid = r["file_id"]
            vid = idx2vid.get((T, fid))
            if vid is None:
                missing.append(f"unmapped file_id {fid} in edit{T} for {arm}"); continue
            case_id = by_vid.get((T, vid))
            if case_id is None:
                continue
            row = {
                "case_id": case_id, "video_name": vid, "edit_type": T, "arm": arm,
                "b": b_by_case.get(case_id, ""),
                "yn_acc": r.get(f"{arm}|five_acc_yes_no", ""),
                "mc_acc": r.get(f"{arm}|five_acc_multi_choice", ""),
                "union":  r.get(f"{arm}|five_acc_union", ""),
                "inter":  r.get(f"{arm}|five_acc_inter", ""),
            }
            for m in METRIC_COLS:
                row[m] = r.get(f"{arm}|{m}", "")
            rows.append(row)

if missing:
    print("[r30_stage2] join problems:", *missing[:10], sep="\n  ")
    sys.exit(1)

want = 22 * len(ARMS)
if len(rows) != want:
    print(f"[r30_stage2] FAILED join -- got {len(rows)} rows, expected {want} "
          f"(22 clips x {len(ARMS)} scored arm(s)). Refusing to write a partial table.")
    sys.exit(1)

fieldnames = ["case_id", "video_name", "edit_type", "arm", "b", "yn_acc", "mc_acc",
              "union", "inter"] + METRIC_COLS
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
print(f"[r30_stage2] wrote {OUT} ({len(rows)} rows, {len(ARMS)} arm(s): "
      f"{', '.join(ARMS)})")
PYEOF
  JRC=$?
  [ "$JRC" -ne 0 ] && { echo "[r30_stage2] FAILED -- join step exit ${JRC}"; exit 1; }

  echo "[r30_stage2] DONE -- evaluation/csv/r30_fiveacc_arms.csv ready."
  echo "[r30_stage2] Next: /run-step R30 stage2-final"
  exit 0
fi

# ========================================================================================
# PHASE=acc -- everything below this point.
# ========================================================================================

# The eval env is five-bench, NOT streamgve (.bashrc auto-activates streamgve and
# `conda activate five-bench` alone does not pop it -- the R2 eval died on
# ModuleNotFoundError: torchmetrics).
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate five-bench

mkdir -p logs evaluation/csv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

echo "[r30_stage2] node=$(hostname) job=${SLURM_JOB_ID} phase=${PHASE}"
nvidia-smi -L 2>&1 | head -4

# Guard 0: the VLM must be in the local cache -- HF_HUB_OFFLINE=1 means a miss is a
# hard failure, and metrics_calculator.py:687 SWALLOWS a load error and reports nan
# for every clip, which would look like a completed run with no edits detected.
VLM_CACHE=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct
if [ ! -d "$VLM_CACHE" ]; then
  echo "[r30_stage2] FAILED -- Qwen2.5-VL-7B not cached at $VLM_CACHE"
  echo "[r30_stage2]   Pre-fetch on a LOGIN node, then resubmit."
  exit 1
fi

ANNOTATIONS=(
  "$DATA_ROOT"/edit_prompt/edit1_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit2_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit3_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit4_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit5_FiVE.json
  "$DATA_ROOT"/edit_prompt/edit6_FiVE.json
)

# The eight constant-b arms. tau_bg is 0 throughout: background pinned to the source
# for the whole rollout, which is R30's background rule verbatim. "Constant" here means
# constant ACROSS CLIPS, not b_bg == b_fg -- R26's taubg2_taufg2 (its is_control arm)
# is the uniform Eq.4 case with no fg/bg split and is deliberately NOT in this set.
BS=(2 3 4 6 8 10 20 50)

FAILED=0
for B in "${BS[@]}"; do
  METHOD="taubg0_taufg${B}_vp"
  DIR="$R26_ROOT/$METHOD"
  STEM="r30_fiveacc_${METHOD}"

  if [ ! -d "$DIR" ]; then
    echo "[r30_stage2] FAILED b=${B} -- arm dir missing: $DIR"; FAILED=1; continue
  fi
  # evaluate.py writes a sibling {video}_resize dir next to every video it scores, so a
  # bare ls over an already-scored dir over-counts.
  NDIRS=$(ls -d "$DIR"/*/*/ 2>/dev/null | grep -vc '_resize/$')
  if [ "$NDIRS" -ne 22 ]; then
    echo "[r30_stage2] FAILED b=${B} -- expected 22 real pairs, found ${NDIRS}."
    echo "[r30_stage2]   Scoring a short arm would still write a full-looking table. Stop."
    FAILED=1; continue
  fi

  echo "[r30_stage2] === b=${B} arm=${METHOD} (${NDIRS} pairs) ==="
  LOG="logs/r30_stage2_${SLURM_JOB_ID}_b${B}.metrics.log"
  python evaluation/fivebench/evaluate.py \
    --config_path evaluation/fivebench/config.yaml \
    --metrics five_acc \
    --src_image_folder "$DATA_ROOT" \
    --annotation_mapping_files "${ANNOTATIONS[@]}" \
    --tgt_methods "$DIR" \
    --tgt_layout edit_video \
    --cases_json "$CASES" \
    --result_path "evaluation/csv/${STEM}.csv" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}

  # metrics_calculator.py:687 prints this and then returns nan for every clip rather
  # than raising. Treat it as fatal: a silent all-nan table is worse than no table.
  if grep -q "five_acc will report nan" "$LOG"; then
    echo "[r30_stage2] FAILED b=${B} -- the VLM did not load; five_acc is all nan."
    FAILED=1; continue
  fi
  echo "[r30_stage2] b=${B} exit=${RC} error_lines=$(grep -c 'Error:' "$LOG")"
  [ "$RC" -ne 0 ] && FAILED=1
done

if [ "$FAILED" -ne 0 ]; then
  echo "[r30_stage2] one or more arms failed -- NOT writing the grid csv."
  exit 1
fi

# ---- join the 6 x 8 per-(edit_type, arm) tables into one long-format grid ----
# file_id indexes the FULL edit{T}_FiVE.json, NOT the cases.json subset: in edit5 the
# ids 0/2/7 are guitar-violin/lucia/car-turn while cases.json lists only three pairs in
# a different order. Mapping positionally against cases.json silently misattributes
# every row, so the map goes through the annotation file's own order.
python - <<'PYEOF'
import csv, json, os, glob, sys

REPO = "/home/ids/skhairi/Code/StreamEdit_bigchantier"
DATA = os.path.expanduser("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
BS = [2, 3, 4, 6, 8, 10, 20, 50]
OUT = f"{REPO}/evaluation/csv/r30_fiveacc_grid.csv"

cases = json.load(open(f"{REPO}/evaluation/cases.json"))
by_vid = {}
for c in cases:
    by_vid.setdefault((c["edit_type"], c["video_name"]), c["case_id"])

# (edit_type, file_id) -> video_name, straight from the annotation file order
idx2vid = {}
for T in range(1, 7):
    d = json.load(open(f"{DATA}/edit_prompt/edit{T}_FiVE.json"))
    items = d if isinstance(d, list) else list(d.values())
    for i, x in enumerate(items):
        idx2vid[(T, str(i))] = x["video_name"]

rows, missing = [], []
for b in BS:
    method = f"taubg0_taufg{b}_vp"
    for T in range(1, 7):
        f = (f"{REPO}/evaluation/csv/edit{T}_FiVE_r30_fiveacc_"
             f"{method}_frame_stride8.csv")
        if not os.path.exists(f):
            missing.append(f); continue
        for r in csv.DictReader(open(f)):
            fid = r["file_id"]
            vid = idx2vid.get((T, fid))
            if vid is None:
                missing.append(f"unmapped file_id {fid} in edit{T}"); continue
            case_id = by_vid.get((T, vid))
            if case_id is None:
                continue          # scored but not one of the 22 cases
            rows.append({
                "case_id": case_id, "video_name": vid, "edit_type": T, "b": b,
                "yn_acc": r.get(f"{method}|five_acc_yes_no", ""),
                "mc_acc": r.get(f"{method}|five_acc_multi_choice", ""),
                "union":  r.get(f"{method}|five_acc_union", ""),
                "inter":  r.get(f"{method}|five_acc_inter", ""),
            })

if missing:
    print("[r30_stage2] join problems:", *missing[:10], sep="\n  ")
    sys.exit(1)

want = 22 * len(BS)
if len(rows) != want:
    print(f"[r30_stage2] FAILED join -- got {len(rows)} rows, expected {want} "
          f"(22 clips x {len(BS)} b). Refusing to write a partial grid.")
    sys.exit(1)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"[r30_stage2] wrote {OUT} ({len(rows)} rows)")

ncase = len({r["case_id"] for r in rows})
print(f"[r30_stage2] {ncase} distinct cases x {len(BS)} b values")
PYEOF
JRC=$?
[ "$JRC" -ne 0 ] && { echo "[r30_stage2] FAILED -- join step exit ${JRC}"; exit 1; }

echo "[r30_stage2] DONE -- evaluation/csv/r30_fiveacc_grid.csv ready."
echo "[r30_stage2] Next: /run-step R30 check-oracle  (the GO/NO-GO)"
