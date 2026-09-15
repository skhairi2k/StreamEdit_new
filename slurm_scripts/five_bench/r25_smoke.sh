#!/bin/bash
# R25 smoke -- prove that --tau_map actually reaches Eq. 4's exponent before committing
# the 22-clip adaptive arm.
#
# Clip: 0001_bus, edit type 1, vp mode. cases.json holds exactly one edit_type=1 entry
# (0001_bus), so --cases_json + --edit_type 1 selects it without a bespoke json.
#
# GATE1 (IDENTICAL) -- override reachability: a tau_map forcing tau=2.0 for every pair
#   must render sha256-identical to a plain `--blend_power 2` run. tau IS blend_power
#   (causal_model.py:334), so if the map reaches the bridge these are the same
#   computation. A mismatch means the plumbing perturbed sampling.
#
# GATE2 (DIFFERS) -- non-degeneracy: the same clip at tau=8 must DIFFER from tau=2.
#   GATE1 alone is ALSO satisfied by a map that is parsed and then silently discarded
#   (both runs would fall back to blend_power=2 and match). GATE2 is what rules that out.
#   Both gates are needed; neither is sufficient.
#
# GATE3 (COVERAGE-OK) -- a tau_map that does not cover the pair must SystemExit rather
#   than render. A silent fall back to 2.0 would produce a half-adaptive arm that still
#   yields 22 clips and a full metrics table, and nothing downstream could detect it.
#
# Cheap: 3 renders of one clip (~5 min each) plus one instant failure.
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r25_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r25_smoke_%j.out
#SBATCH --error=logs/r25_smoke_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHORS=/projects/dataggen/outputs/five_bench/anchors
CASES="$REPO/evaluation/cases.json"
OUT_ROOT=/projects/dataggen/outputs/five_bench/r25_smoke
WORK="$OUT_ROOT/_maps"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p "$WORK"

# Purpose-built maps. Written here rather than reused from r25_tau_map.csv so the gates
# test the PLUMBING at known constants, independent of whatever the real map holds.
printf 'video_name,edit_type,iou,tau\n0001_bus,1,0.5,2.0\n'  > "$WORK/tau2.csv"
printf 'video_name,edit_type,iou,tau\n0001_bus,1,0.5,8.0\n'  > "$WORK/tau8.csv"
printf 'video_name,edit_type,iou,tau\n0099_absent,1,0.5,2.0\n' > "$WORK/truncated.csv"

COMMON=(--edit_type 1
        --data_root "$DATA_ROOT"
        --cases_json "$CASES"
        --first_frame_edit_dir "$ANCHORS"
        --vp_mode vp
        --out_root "$OUT_ROOT"
        --seed 0)

run_arm () {   # $1 = method dir name, rest = extra flags
  local method="$1"; shift
  echo "--- rendering $method ---"
  python evaluation/run_fivebench.py "${COMMON[@]}" --method "$method" "$@"
}

# Reference: the ordinary Eq. 4 path, no map involved.
run_arm baseline_rho2 --blend_power 2
# GATE1 arm: same exponent, delivered through the map.
run_arm map_tau2      --tau_map "$WORK/tau2.csv"
# GATE2 arm: a different exponent through the same path.
run_arm map_tau8      --tau_map "$WORK/tau8.csv"

# sha256 over the frame PNGs, in sorted order, ignoring the *_resize dirs.
hash_clip () {
  find "$OUT_ROOT/$1/edit1/0001_bus" -name '*.png' | sort | xargs sha256sum \
    | awk '{print $1}' | sha256sum | awk '{print $1}'
}

H_REF=$(hash_clip baseline_rho2)
H_T2=$(hash_clip map_tau2)
H_T8=$(hash_clip map_tau8)
echo "baseline_rho2 $H_REF"
echo "map_tau2      $H_T2"
echo "map_tau8      $H_T8"

if [ "$H_REF" = "$H_T2" ]; then
  echo "IDENTICAL gate1: tau_map(tau=2) == --blend_power 2"
else
  echo "GATE1-FAIL gate1: tau_map(tau=2) != --blend_power 2 -- the override does not"
  echo "  reach blend_power cleanly, or the plumbing perturbed sampling. STOP."
fi

if [ "$H_T2" != "$H_T8" ]; then
  echo "DIFFERS gate2: tau=8 differs from tau=2"
else
  echo "GATE2-FAIL gate2: tau=8 produced the SAME pixels as tau=2 -- the map is being"
  echo "  read and then discarded, which gate1 cannot detect. STOP."
fi

# GATE3: must exit non-zero WITHOUT rendering.
echo "--- coverage check (expected to fail) ---"
if python evaluation/run_fivebench.py "${COMMON[@]}" --method cov_check \
     --tau_map "$WORK/truncated.csv"; then
  echo "GATE3-FAIL gate3: a tau_map missing 0001_bus rendered anyway instead of"
  echo "  raising SystemExit -- a partial map would silently fall back to 2.0. STOP."
else
  echo "COVERAGE-OK gate3: incomplete tau_map raised SystemExit"
fi

echo "--- r25_smoke done ---"
