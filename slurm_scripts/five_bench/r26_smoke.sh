#!/bin/bash
# R26 smoke -- prove the SPATIAL blend field reaches the bridge, and that adding it did
# not perturb the scalar path, before committing 9 arms x 22 clips.
#
# Clips (edit type 5, both anchored): 0069_car-turn (a moving object -- the per-frame
# gate needs one) and 0007_guitar-violin. Plus one multi-window clip, 0034_cows at edit
# type 2 (24 latent frames > the 18-frame first window), because the union dump is
# stitched across rollout windows with the same overlap rule as the latents and NO
# single-window clip can catch a wrong overlap slice or concat axis -- the R22 lesson,
# restated in the R26 plan.
#
# GATE1 (IDENTICAL) -- degenerate control: tau_bg = tau_fg = 2 must render
#   sha256-identical to a plain Eq. 4 run at --blend_power 2. A uniform field IS the
#   global schedule, so this proves BOTH that the cache-aligned field reaches the three
#   blend sites AND that rewriting them (float32 rate, per-product rounding) left the
#   scalar case bit-for-bit. THIS IS THE GATE. If it fails, stop.
#
# GATE2 (IDENTICAL) -- non-regression on the untouched default path: a defaults run
#   (novp, no R26 flag) must equal the stored five_bench/baseline render.
#   CAVEAT: that reference predates the 2026-07-22 per-pair reseeding for any pair past
#   index 0 of its edit{T} json (see run_fivebench.py's warning). A GATE2-FAIL is
#   therefore only conclusive together with GATE1; read it as "investigate", not "stop".
#   Reported as GATE2-SKIP when the reference dir is absent.
#
# GATE3 (DIFFERS) -- field-reaches-bridge: tau_bg = tau_fg = 1 (uniform, but NOT 2) must
#   DIFFER from the Eq. 4 reference. Without this, GATE1 is ALSO satisfied by a field
#   that is silently ignored and falls back to blend_power=2. Neither gate alone suffices.
#
# GATE4 (PERFRAME-OK) -- the dumped M_f must VARY across latent frames on 0069_car-turn.
#   A field collapsed to frame 0 and broadcast would look perfectly healthy in gates 1-3.
#
# GATE5 (MULTIWINDOW-OK) -- on 0034_cows the dumped mask must have exactly as many rows
#   as the clip has latent frames, and tau(2,2) must still match Eq. 4 there. This is the
#   only gate that exercises the window-boundary path.
#
# Cost: ~11 renders of short clips plus 2 instant failures.
#
# --mem=64G: the partition default (8 x 3936M ~ 31.5G) host-OOMs while loading
# UMT5-XXL + the checkpoint (the R9 job-900404 failure).
#SBATCH --job-name=r26_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r26_smoke_%j.out
#SBATCH --error=logs/r26_smoke_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
ANCHORS=/projects/dataggen/outputs/five_bench/anchors
BASELINE_REF=/projects/dataggen/outputs/five_bench/baseline
OUT_ROOT=/projects/dataggen/outputs/five_bench/r26_smoke
WORK="$OUT_ROOT/_work"
MASKS="$OUT_ROOT/_masks"

cd "$REPO" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null    # .bashrc auto-activates streamgve (the R2 env bug)
conda activate streamgve

mkdir -p "$WORK" "$MASKS"

# Purpose-built case subsets. Written here rather than reusing evaluation/cases.json so
# the gates stay pinned to these clips whatever the real case set grows into.
python - "$WORK" <<'PYEOF'
import json, sys, pathlib
work = pathlib.Path(sys.argv[1])
cases = json.loads((pathlib.Path("evaluation/cases.json")).read_text())
by = {(c["video_name"], int(c["edit_type"])): c for c in cases}
(work / "smoke_e5.json").write_text(json.dumps(
    [by[("0069_car-turn", 5)], by[("0007_guitar-violin", 5)]], indent=2))
(work / "smoke_e2.json").write_text(json.dumps([by[("0034_cows", 2)]], indent=2))
print("[smoke] wrote smoke_e5.json (2 clips) and smoke_e2.json (1 multi-window clip)")
PYEOF
[ -f "$WORK/smoke_e5.json" ] || { echo "SMOKE-ABORT: could not build the case subsets"; exit 1; }

COMMON5=(--edit_type 5
         --data_root "$DATA_ROOT"
         --cases_json "$WORK/smoke_e5.json"
         --first_frame_edit_dir "$ANCHORS"
         --vp_mode vp
         --out_root "$OUT_ROOT"
         --seed 0)
COMMON2=(--edit_type 2
         --data_root "$DATA_ROOT"
         --cases_json "$WORK/smoke_e2.json"
         --first_frame_edit_dir "$ANCHORS"
         --vp_mode vp
         --out_root "$OUT_ROOT"
         --seed 0)

# sha256 over the frame PNGs, in sorted order, ignoring the *_resize dirs.
hash_clip () {   # $1 = dir holding the PNGs
  find "$1" -maxdepth 1 -name '*.png' | sort | xargs sha256sum \
    | awk '{print $1}' | sha256sum | awk '{print $1}'
}

# ---------------------------------------------------------------- pass 1: dump M_f
echo "=== pass 1: dump the grounding-mask union (--blend_sched zero) ==="
python evaluation/run_fivebench.py "${COMMON5[@]}" --method dump_e5 \
  --blend_sched zero --union_dump_dir "$MASKS" || exit 1
python evaluation/run_fivebench.py "${COMMON2[@]}" --method dump_e2 \
  --blend_sched zero --union_dump_dir "$MASKS" || exit 1

# ---------------------------------------------------------------- reference + arms
echo "=== reference and spatial arms ==="
python evaluation/run_fivebench.py "${COMMON5[@]}" --method eq4_vp   --blend_power 2 || exit 1
python evaluation/run_fivebench.py "${COMMON5[@]}" --method tau_2_2 \
  --union_mask_dir "$MASKS" --tau_bg 2 --tau_fg 2 || exit 1
python evaluation/run_fivebench.py "${COMMON5[@]}" --method tau_1_1 \
  --union_mask_dir "$MASKS" --tau_bg 1 --tau_fg 1 || exit 1

python evaluation/run_fivebench.py "${COMMON2[@]}" --method eq4_vp   --blend_power 2 || exit 1
python evaluation/run_fivebench.py "${COMMON2[@]}" --method tau_2_2 \
  --union_mask_dir "$MASKS" --tau_bg 2 --tau_fg 2 || exit 1

# defaults arm for GATE2: novp, no R26 flag, nothing but today's code path
python evaluation/run_fivebench.py --edit_type 5 --data_root "$DATA_ROOT" \
  --cases_json "$WORK/smoke_e5.json" --vp_mode novp --out_root "$OUT_ROOT" \
  --method defaults --seed 0 || exit 1

# ---------------------------------------------------------------- GATE1
echo "=== GATE1: uniform tau(2,2) == Eq. 4 blend_power=2 ==="
G1_FAIL=0
for CLIP in 0069_car-turn 0007_guitar-violin; do
  H_REF=$(hash_clip "$OUT_ROOT/eq4_vp/edit5/$CLIP")
  H_T22=$(hash_clip "$OUT_ROOT/tau_2_2/edit5/$CLIP")
  echo "  $CLIP  eq4=$H_REF  tau22=$H_T22"
  [ "$H_REF" = "$H_T22" ] || G1_FAIL=1
done
if [ "$G1_FAIL" = "0" ]; then
  echo "IDENTICAL gate1: tau_bg=tau_fg=2 reproduces Eq. 4 bit-for-bit on both clips"
else
  echo "GATE1-FAIL gate1: the uniform field does NOT reproduce the global schedule."
  echo "  Either the field never reaches the three blend sites, or the rewrite of those"
  echo "  sites perturbed the scalar case (check the float32 rate and the per-product"
  echo "  rounding back to the key dtype). STOP -- do not launch 9 arms."
fi

# ---------------------------------------------------------------- GATE2
echo "=== GATE2: defaults still match the stored baseline ==="
G2_SEEN=0; G2_FAIL=0
for CLIP in 0069_car-turn 0007_guitar-violin; do
  REF="$BASELINE_REF/edit5/$CLIP"
  [ -d "$REF" ] || continue
  G2_SEEN=1
  H_B=$(hash_clip "$REF")
  H_D=$(hash_clip "$OUT_ROOT/defaults/edit5/$CLIP")
  echo "  $CLIP  baseline=$H_B  defaults=$H_D"
  [ "$H_B" = "$H_D" ] || G2_FAIL=1
done
if [ "$G2_SEEN" = "0" ]; then
  echo "GATE2-SKIP gate2: no stored baseline render for these clips under $BASELINE_REF"
elif [ "$G2_FAIL" = "0" ]; then
  echo "IDENTICAL gate2: defaults == five_bench/baseline"
else
  echo "GATE2-FAIL gate2: the defaults path moved. NOTE the stored baseline predates the"
  echo "  2026-07-22 per-pair reseeding for any pair past index 0 of edit5_FiVE.json, so"
  echo "  this can be a stale reference rather than a regression -- but only conclude that"
  echo "  if gate1 passed. Investigate before launching 9 arms."
fi

# ---------------------------------------------------------------- GATE3
echo "=== GATE3: uniform tau(1,1) must DIFFER from Eq. 4 ==="
G3_FAIL=0
for CLIP in 0069_car-turn 0007_guitar-violin; do
  H_REF=$(hash_clip "$OUT_ROOT/eq4_vp/edit5/$CLIP")
  H_T11=$(hash_clip "$OUT_ROOT/tau_1_1/edit5/$CLIP")
  echo "  $CLIP  eq4=$H_REF  tau11=$H_T11"
  [ "$H_REF" != "$H_T11" ] || G3_FAIL=1
done
if [ "$G3_FAIL" = "0" ]; then
  echo "DIFFERS gate3: tau=1 changes the render, so the field is not being ignored"
else
  echo "GATE3-FAIL gate3: tau=1 produced the SAME pixels as tau=2 -- the field is parsed"
  echo "  and then discarded, which gate1 CANNOT detect. STOP."
fi

# ---------------------------------------------------------------- GATE4 + GATE5
echo "=== GATE4/GATE5: dumped masks are per-frame and window-complete ==="
python - "$MASKS" <<'PYEOF'
import sys, pathlib, numpy as np

masks = pathlib.Path(sys.argv[1])

def load(p):
    with np.load(p) as z:
        shape = tuple(int(v) for v in z["shape"])
        M = np.unpackbits(z["M"], axis=-1, count=shape[1]).astype(bool)
    assert M.shape == shape, (M.shape, shape)
    return M

# GATE4 -- per-frame variation on a moving object.
p4 = masks / "edit5" / "0069_car-turn.npz"
if not p4.exists():
    print(f"GATE4-FAIL gate4: {p4} was not written by pass 1. STOP.")
else:
    M = load(p4)
    n_distinct = len({r.tobytes() for r in M})
    frac = M.mean(axis=1)
    print(f"  0069_car-turn M shape={M.shape} distinct_rows={n_distinct} "
          f"fg_fraction min={frac.min():.4f} max={frac.max():.4f}")
    if M.shape[0] < 2:
        print("GATE4-FAIL gate4: only one latent frame dumped. STOP.")
    elif n_distinct < 2:
        print("GATE4-FAIL gate4: every latent frame carries the SAME mask -- the field is"
              " collapsed to frame 0 and broadcast, which gates 1-3 cannot see. STOP.")
    elif frac.max() == 0.0:
        print("GATE4-FAIL gate4: the dumped mask is entirely empty. STOP.")
    elif frac.min() == 1.0:
        print("GATE4-FAIL gate4: the dumped mask is entirely full -- likely inverted or"
              " defaulted to ones. STOP.")
    else:
        print(f"PERFRAME-OK gate4: M_f varies across {n_distinct}/{M.shape[0]} latent frames")

# GATE5 -- the multi-window clip dumped a complete, per-frame mask.
p5 = masks / "edit2" / "0034_cows.npz"
if not p5.exists():
    print(f"GATE5-FAIL gate5: {p5} was not written by pass 1. STOP.")
else:
    M = load(p5)
    n_distinct = len({r.tobytes() for r in M})
    print(f"  0034_cows M shape={M.shape} distinct_rows={n_distinct}")
    # 104 source frames -> find_closest_num_frame(104) = 93 -> (93-1)//4 + 1 = 24 latent.
    if M.shape[0] != 24:
        print(f"GATE5-FAIL gate5: expected 24 latent frames on 0034_cows, got {M.shape[0]}"
              " -- the per-window union is being stitched with the wrong overlap or axis."
              " Every pass-2 field row after the first window would be misaligned. STOP.")
    elif n_distinct < 2:
        print("GATE5-FAIL gate5: 0034_cows dumped identical rows across windows. STOP.")
    else:
        print("MULTIWINDOW-OK gate5: 24 latent frames stitched across the window boundary")
PYEOF

# GATE5 second half: the control must also hold on the multi-window clip.
H_REF2=$(hash_clip "$OUT_ROOT/eq4_vp/edit2/0034_cows")
H_T222=$(hash_clip "$OUT_ROOT/tau_2_2/edit2/0034_cows")
echo "  0034_cows  eq4=$H_REF2  tau22=$H_T222"
if [ "$H_REF2" = "$H_T222" ]; then
  echo "IDENTICAL gate5b: tau(2,2) reproduces Eq. 4 across a window boundary too"
else
  echo "GATE5B-FAIL gate5b: the control holds on single-window clips but breaks on"
  echo "  0034_cows -- the cache-aligned field is misaligned at the window boundary. STOP."
fi

# ---------------------------------------------------------------- GATE6: cross-validation
echo "=== GATE6: incompatible flag combinations must SystemExit without rendering ==="
G6_FAIL=0
if python evaluation/run_fivebench.py "${COMMON5[@]}" --method xcheck1 \
     --union_mask_dir "$MASKS" --tau_bg 1 --tau_fg 2 --blend_sched cos_half; then
  echo "  --union_mask_dir + --blend_sched RENDERED instead of exiting"; G6_FAIL=1
fi
if python evaluation/run_fivebench.py "${COMMON5[@]}" --method xcheck2 \
     --tau_bg 1 --tau_fg 2; then
  echo "  --tau_bg/--tau_fg without --union_mask_dir RENDERED instead of exiting"; G6_FAIL=1
fi
if [ "$G6_FAIL" = "0" ]; then
  echo "CROSSVAL-OK gate6: both incompatible combinations raised SystemExit"
else
  echo "GATE6-FAIL gate6: a combination that silently discards half the configuration was"
  echo "  accepted. STOP."
fi

echo "--- r26_smoke done ---"
