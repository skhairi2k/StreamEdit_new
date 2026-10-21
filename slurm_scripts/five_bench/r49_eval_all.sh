#!/bin/bash
# R49 eval: one job per GROUP scores every unit of `r49_units.py --group GROUP` serially (FiVE harness metrics, then
# FiVE-Acc), then CLIP-D over the whole group. Serial on purpose: one queue entry per group (per-user submit cap).
#
# Usage: sbatch slurm_scripts/five_bench/r49_eval_all.sh main    # 3 units: se_max, wan_max_s9, wan_max_s5      (r49_chain.sh)
#        sbatch slurm_scripts/five_bench/r49_eval_all.sh base    # 12 units: 9 Wan baselines, native_s0.978, R48 dino_SD x2  (r49_chain.sh)
#        sbatch slurm_scripts/five_bench/r49_eval_all.sh m59     # 3 matched-59 units                       (r49_matched59.sh)
#        sbatch slurm_scripts/five_bench/r49_eval_all.sh wan_pair   # 2 units: wan_max_s5 + the official wan_native_s0.92 (Jean Zay reproduction, r50_repro_chain.sh;
#                                                                   # CLIP-D -> r49_clip_directional_wan_pair.csv)
#        sbatch slurm_scripts/five_bench/r49_eval_all.sh GROUP full   # 419 pairs; GROUP = se | wan | base | m59 (r49_units.py --mode full):
#                                                                     # se on the lab cluster, wan / base / m59 on Jean Zay
# Same evaluate.py calls as r48_eval.sh / r48_fiveacc.sh / r48_clipd.sh. subset: the 18 clips of evaluation/r49_cases.json
# (--cases_json), only the annotation files of edit types that have cases (types 1-5; no type-6 clip). full: no --cases_json
# (the full bench of every edit{T}_FiVE.json, counts from evaluation/r7_anchor_manifest.json), CLIP-D --fullbench.
# Output: evaluation/csv/edit{T}_FiVE_r49_{base}_frame_stride8.csv, ..._r49_fiveacc_{base}_..., r49_clip_directional_{GROUP}.csv
#         (full: unit bases end in _full, CLIP-D r49_clip_directional_{GROUP}_full.csv)
#SBATCH --job-name=r49_eval_all
#SBATCH --partition=L40S
#SBATCH --exclude=node52
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r49_eval_all_%j.out
#SBATCH --error=logs/r49_eval_all_%j.err

GROUP=${1:-}
MODE=${2:-subset}
case "$MODE/$GROUP" in
  subset/main|subset/base|subset/m59|subset/wan_pair)
    CASES=evaluation/r49_cases.json; CASES_ARG=(--cases_json "$CASES"); CLIPD_SCOPE=(--cases "$CASES"); SUFFIX="" ;;
  full/se|full/wan|full/base|full/m59)
    CASES=evaluation/r7_anchor_manifest.json; CASES_ARG=(); CLIPD_SCOPE=(--fullbench); SUFFIX="_full" ;;
  *) echo "[r49_eval] usage: r49_eval_all.sh main|base|m59|wan_pair [subset]  or  r49_eval_all.sh se|wan|base|m59 full"; exit 1 ;;
esac
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
cd "$REPO" || exit 1
mkdir -p logs evaluation/csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
JOB=${SLURM_JOB_ID:-local}

read -r -a EXPECT <<< "$($PY_FB - "$CASES" <<'PYEOF'
import json, sys, collections
c = collections.Counter(int(x["edit_type"]) for x in json.load(open(sys.argv[1])))
print(" ".join(str(c.get(t, 0)) for t in range(1, 7)))
PYEOF
)"
ANNOTATIONS=()
for T in 1 2 3 4 5 6; do
  [ "${EXPECT[$((T - 1))]}" -gt 0 ] && ANNOTATIONS+=("$DATA_ROOT/edit_prompt/edit${T}_FiVE.json")
done
echo "[r49_eval] annotation files: ${#ANNOTATIONS[@]} (edit types with cases)"
echo "[r49_eval] job=${JOB} group=${GROUP} mode=${MODE} cases=${CASES} expect per type: ${EXPECT[*]}"
echo "[r49_eval] node=$(hostname)"; nvidia-smi -L 2>&1 | head -2

# per-type row counts of one result stem; for five_acc also the binary columns (as r48_fiveacc.sh)
check_rows() {  # $1 = stem, $2 = kind (harness|fiveacc)
  $PY_FB - "$1" "$2" "${EXPECT[@]}" <<'PY'
import csv, sys
stem, kind, expect = sys.argv[1], sys.argv[2], dict(zip(range(1, 7), map(int, sys.argv[3:9])))
need = ("five_acc_yes_no", "five_acc_multi_choice") if kind == "fiveacc" else ()
bad = 0
for t, n_exp in expect.items():
    if n_exp == 0:
        continue
    path = f"evaluation/csv/edit{t}_FiVE_{stem}_frame_stride8.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"[r49_eval] {stem} edit{t}: MISSING {path}"); bad += 1; continue
    cols = {k.split("|")[-1]: k for k in (rows[0].keys() if rows else [])}
    miss = [c for c in need if c not in cols]
    nonbin = sum(1 for r in rows for c in need if c in cols and r[cols[c]] not in ("0", "1", "0.0", "1.0"))
    ok = len(rows) == n_exp and not miss and nonbin == 0
    bad += not ok
    print(f"[r49_eval] {stem} edit{t}: {len(rows)} rows (expect {n_exp}), missing cols={miss or 0}, "
          f"non-binary/NaN cells={nonbin} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
}

UNIT_FAIL=0; METHODS=(); NAMES=()
while IFS=$'\t' read -r BASE DIR HOST CONFIG; do
  echo "=== [r49_eval] unit ${BASE} (${HOST}/${CONFIG}) $(date '+%F %T') -- ${DIR}"
  # Completeness preflight, Wan-Edit units only: it counts frames the way a Wan-Edit render does (min(41, n) -> 4k+1), which is
  # wrong for StreamEdit's 3-latent-frame blocks (the 4 shortest videos render 33 / 21 frames: false "not rendered"). The
  # StreamEdit render already guards itself (dirs, manifests, gates) and the row counts are checked after scoring.
  MISSING=""
  [ "$HOST" != se ] && MISSING=$($PY_FB evaluation/r48_units.py --mode "$MODE" --missing "$DIR" --cases "$CASES")
  if [ -n "$MISSING" ]; then
    echo "[r49_eval] FAILED ${BASE} -- $(echo "$MISSING" | wc -l) case clip(s) not rendered under $DIR, e.g. $(echo "$MISSING" | head -3 | tr '\n' ' ')"
    UNIT_FAIL=$((UNIT_FAIL + 1)); continue
  fi
  METHODS+=("r49_${BASE}=${DIR}"); NAMES+=("r49_${BASE}")
  F=0
  for KIND in harness fiveacc; do
    if [ "$KIND" = harness ]; then
      STEM="r49_${BASE}"
      METRICS=(structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part
               clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part niqe_target_image)
    else
      STEM="r49_fiveacc_${BASE}"; METRICS=(five_acc)
    fi
    LOG="logs/r49_eval_${JOB}_${BASE}_${KIND}.metrics.log"
    $PY_FB evaluation/fivebench/evaluate.py \
      --config_path evaluation/fivebench/config.yaml --metrics "${METRICS[@]}" \
      --src_image_folder "$DATA_ROOT" --annotation_mapping_files "${ANNOTATIONS[@]}" \
      --tgt_methods "$DIR" --tgt_layout edit_video --tmp_resize "${CASES_ARG[@]}" \
      --result_path "evaluation/csv/${STEM}.csv" > "$LOG" 2>&1
    RC=$?
    NERR=$(grep -c 'Error:' "$LOG"); NOOM=$(grep -c 'out of memory' "$LOG")
    echo "[r49_eval] ${STEM} exit=${RC} error_lines=${NERR} oom_lines=${NOOM} (log ${LOG})"
    { [ "$RC" -ne 0 ] || [ "$NERR" -ne 0 ]; } && F=1
    check_rows "$STEM" "$KIND" || F=1
  done
  [ "$F" -ne 0 ] && { echo "[r49_eval] FAILED unit ${BASE}"; UNIT_FAIL=$((UNIT_FAIL + 1)); }
done < <($PY_FB evaluation/r49_units.py --group "$GROUP" --mode "$MODE" 2>/dev/null)

# ---- CLIP-D over the group's rendered units
OUT="evaluation/csv/r49_clip_directional_${GROUP}${SUFFIX}.csv"
if [ "${#METHODS[@]}" -gt 0 ]; then
  RUNLOG="logs/r49_eval_${JOB}_clipd.run.log"
  $PY_FB evaluation/r33_clip_directional.py "${CLIPD_SCOPE[@]}" --methods "${METHODS[@]}" --data_root "$DATA_ROOT" \
    --frame_stride 8 -o "$OUT" > "$RUNLOG" 2>&1
  RC=$?
  echo "[r49_eval] clipd exit=${RC} -> ${OUT}"
  if [ "$RC" -ne 0 ] || [ ! -f "$OUT" ]; then
    UNIT_FAIL=$((UNIT_FAIL + 1))
  else
    $PY_FB - "$OUT" "${EXPECT[@]}" -- "${NAMES[@]}" <<'PY' || UNIT_FAIL=$((UNIT_FAIL + 1))
import csv, sys
from collections import Counter
args = sys.argv[1:]
out, expect, names = args[0], dict(zip(range(1, 7), map(int, args[1:7]))), args[8:]
expect = {t: n for t, n in expect.items() if n}
rows = list(csv.DictReader(open(out)))
bad = 0
for m in names:
    got = Counter(int(r["edit_type"]) for r in rows if r["method"] == m)
    ok = dict(got) == expect
    bad += not ok
    print(f"[r49_eval] clipd {m}: {sum(got.values())} rows {dict(sorted(got.items()))} -- {'ok' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
PY
  fi
fi
echo "[r49_eval] unit failures: ${UNIT_FAIL}"
exit $(( UNIT_FAIL > 0 ))
