#!/bin/bash
# R20 smoke -- two gates.
#
# GATE 1 (parity, index-0 pairs only).
# run_fivebench.py now re-seeds before every pair (2026-07-22). The stored
# references five_bench/baseline (R1) and five_bench/r7_visual_prompting (R7)
# were rendered on 2026-07-18/19, BEFORE that change, when a pair's noise
# depended on its position in the edit{T} json. Only pairs at index 0 of their
# json are unaffected, so only those can be byte-compared today:
#     0001_bus       edit1 index 0  -> hard gate
#     0002_girl-dog  edit6 index 0  -> hard gate
#     0042_gym-ball  edit6 index 4  -> EXPECTED to differ, reported not asserted
# Once R1/R7 are re-rendered under per-pair seeding, gym-ball should match too and
# this classification can be dropped. If a reference is ABSENT (they are being
# re-rendered), that pair is SKIPPED -- gate 1 then proves nothing and gate 2
# carries the run.
#
# GATE 2 (subset independence -- the property the whole sweep rests on).
# Renders edit6 twice: restricted to the 2 cases.json pairs, and unrestricted (all
# 10). 0042_gym-ball must be bit-identical across the two. That is exactly what
# failed before per-pair seeding, and it is what makes R20 comparable to a
# full-bench reference at all.
#
# Then stage 3 renders cos_full under all three VP modes as a plumbing check.
#
# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r20_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/r20_smoke_%j.out
#SBATCH --error=logs/r20_smoke_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
REF_ROOT=/projects/dataggen/outputs/five_bench
OUT_ROOT=/projects/dataggen/outputs/five_bench/r20_smoke
ANCHOR_ROOT=/projects/dataggen/outputs/five_bench/anchors
CASES=$REPO/evaluation/cases.json
# Parity runs on edit types 1 and 6 only -- cases.json holds 1 clip of edit1 and 2
# of edit6, so that is 3 clips without needing any subset plumbing. 0002_girl-dog
# (edit6, 21 latent frames) splits into two rollout windows under --vp_mode vp,
# so the multi-window path is covered too.
PARITY_TYPES=(1 6)

cd
source .bashrc
conda deactivate                     # .bashrc auto-activates streamgve (R2 env bug)
conda activate streamgve

cd "$REPO"
mkdir -p logs "$OUT_ROOT"

echo "[r20_smoke] job=${SLURM_JOB_ID:-local} out=${OUT_ROOT}"
echo "[r20_smoke] ===== stage 1: bit-parity against R1 / R7 ====="

# --- stage 1a: defaults must reproduce R1 `baseline` ---
for T in "${PARITY_TYPES[@]}"; do
  python evaluation/run_fivebench.py \
    --edit_type "$T" \
    --method parity_novp \
    --cases_json "$CASES" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0
done

# --- stage 1b: --vp_mode vp must reproduce R7 `r7_visual_prompting` ---
for T in "${PARITY_TYPES[@]}"; do
  python evaluation/run_fivebench.py \
    --edit_type "$T" \
    --method parity_vp \
    --vp_mode vp \
    --first_frame_edit_dir "$ANCHOR_ROOT" \
    --cases_json "$CASES" \
    --data_root "$DATA_ROOT" \
    --out_root "$OUT_ROOT" \
    --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0
done

echo "[r20_smoke] ----- comparing against stored references -----"
python - <<'PY'
import hashlib, json, sys
from pathlib import Path

REPO = Path("/home/ids/skhairi/Code/StreamEdit_bigchantier")
DATA = Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark").expanduser()
OUT  = Path("/projects/dataggen/outputs/five_bench/r20_smoke")
REF  = Path("/projects/dataggen/outputs/five_bench")
cases = [c for c in json.loads((REPO / "evaluation" / "cases.json").read_text())
         if c["edit_type"] in (1, 6)]

def json_index(edit_type, video_name):
    ents = json.loads((DATA / "edit_prompt" / f"edit{edit_type}_FiVE.json").read_text())
    ents = ents if isinstance(ents, list) else list(ents.values())
    return [e["video_name"] for e in ents].index(video_name)

def digest(d):
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        return None
    h = hashlib.sha256()
    for p in pngs:
        h.update(p.read_bytes())
    return f"{len(pngs)}:{h.hexdigest()}"

bad = 0
for arm, ref in [("parity_novp", "baseline"), ("parity_vp", "r7_visual_prompting")]:
    for c in cases:
        sub = f"edit{c['edit_type']}/{c['video_name']}"
        idx = json_index(c["edit_type"], c["video_name"])
        got, want = digest(OUT / arm / sub), digest(REF / ref / sub)
        gated = (idx == 0)
        if got is None:
            print(f"MISSING  {arm}/{sub} -- OUR render is absent (real failure)")
            bad += 1
        elif want is None:
            # The R1/R7 references are being re-rendered under per-pair seeding;
            # until they exist there is nothing to compare against. Skipping is
            # correct, but it means gate 1 proves NOTHING on this run -- gate 2 is
            # the only parity-like evidence available.
            print(f"SKIPPED  {arm}/{sub} -- reference absent (re-render pending)")
        elif got == want:
            print(f"IDENTICAL {arm}/{sub} (json idx {idx}) vs {ref}")
        elif gated:
            print(f"DIFFERS  {arm}/{sub} (json idx 0 -- MUST match) vs {ref}\n"
                  f"    new={got}\n    ref={want}")
            bad += 1
        else:
            print(f"expected-differ {arm}/{sub} (json idx {idx}; reference predates "
                  f"per-pair seeding) vs {ref}")
print(f"[parity] {bad} mismatch(es) among index-0 pairs "
      f"(SKIPPED lines mean no reference existed -- gate 1 proved nothing there)")
sys.exit(1 if bad else 0)
PY
PARITY_RC=$?
if [ $PARITY_RC -ne 0 ]; then
  echo "[r20_smoke] PARITY FAILED -- the R1/R7 references are not reproducible with the"
  echo "[r20_smoke] current run_fivebench.py. Do NOT launch r20_infer.sh. Investigate first."
  exit 1
fi

echo "[r20_smoke] ===== gate 2: subset independence (edit6 subset vs full) ====="
# Restricted to the 2 cases.json pairs...
python evaluation/run_fivebench.py \
  --edit_type 6 --method subset_2 --cases_json "$CASES" \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0 || exit 1
# ...and unrestricted (all 10 pairs of edit6).
python evaluation/run_fivebench.py \
  --edit_type 6 --method subset_all \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" \
  --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0 || exit 1

python - <<'PY2'
import hashlib, json, sys
from pathlib import Path
REPO = Path("/home/ids/skhairi/Code/StreamEdit_bigchantier")
OUT  = Path("/projects/dataggen/outputs/five_bench/r20_smoke")

def digest(d):
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        return None
    h = hashlib.sha256()
    for p in pngs:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]

cases = [c for c in json.loads((REPO / "evaluation" / "cases.json").read_text())
         if c["edit_type"] == 6]
bad = 0
for c in cases:
    v = c["video_name"]
    a = digest(OUT / "subset_2" / "edit6" / v)
    b = digest(OUT / "subset_all" / "edit6" / v)
    if a is None or b is None:
        print(f"MISSING  {v}: subset={a} full={b}"); bad += 1
    elif a == b:
        print(f"SUBSET-INDEPENDENT {v}: {a}")
    else:
        print(f"SUBSET-DEPENDENT   {v}: subset={a} full={b}  <-- per-pair seeding is not working")
        bad += 1
print(f"[subset] {bad} mismatch(es)")
sys.exit(1 if bad else 0)
PY2
if [ $? -ne 0 ]; then
  echo "[r20_smoke] GATE 2 FAILED -- a pair's output still depends on how many pairs"
  echo "[r20_smoke] preceded it, so subset runs are NOT comparable to full-bench"
  echo "[r20_smoke] references. Do NOT launch r20_infer.sh."
  exit 1
fi

echo "[r20_smoke] ===== stage 3: cos_full renders on all 3 VP modes (edit1, 1 clip) ====="
for MODE in novp vp pvp; do
  echo "[r20_smoke] --- vp_mode=${MODE} sched=cos_full ---"
  ARGS=(--edit_type 1 --method "sched_cos_full_${MODE}" --blend_sched cos_full
        --cases_json "$CASES" --data_root "$DATA_ROOT" --out_root "$OUT_ROOT"
        --step 15 --fg_boost_factor 4 --blend_power 2 --seed 0)
  if [ "$MODE" != "novp" ]; then
    ARGS+=(--vp_mode "$MODE" --first_frame_edit_dir "$ANCHOR_ROOT")
  fi
  python evaluation/run_fivebench.py "${ARGS[@]}" || {
    echo "[r20_smoke] STAGE 3 FAILED for vp_mode=${MODE}"; exit 1; }
done

echo "[r20_smoke] frame dirs: $(ls -d "$OUT_ROOT"/*/*/*/ 2>/dev/null | wc -l)"
echo "[r20_smoke] done -- parity clean (index-0), subset-independent, all 3 VP modes render."
