#!/bin/bash
# R55 smoke: the cross-attention recorder ported to LongLive, on 2 clips, before the 419-pair chain.
#
# Usage: sbatch slurm_scripts/five_bench/r55_smoke.sh
# Clips: 0001_bus (edit 1) and 0011_lucia (edit 5). Everything runs in a fresh $D/r55_smoke/job<id>/ (nothing is overwritten).
#   G1  recorder is read-only: LongLive draft (N=5 tgt0.9, VP, fg_boost 2) with --attn_out_dir ON vs OFF -> z_trg and z_src bit-identical.
#       If on != off, a second OFF run decides whether the draft is merely non-deterministic (control off == off2 fails) or the recorder
#       really changes the render; either way G1 fails, the message says which.
#   G2  the attention npz of the ON run: keys, shapes [nT, F, 30, 52] / [F, 30, 52] with F == the clip's latent-frame count, finite, non-negative,
#       and not all zero (the recorder actually accumulated).
#   G3  Self-Forcing regression: the SF recorder on 0001_bus edit 1 with R48's exact flags reproduces $D/r48_se_attn/edit1/0001_bus.npz
#       (every array np.array_equal) -- the shared files (evaluation/backbone.py) still serve SF unchanged.
#   G4  downstream: r31_divergence.py (dino, lpips) on the LongLive draft frames + latents, then r49_field.py on the smoke roots -> every field npz
#       has finite q in [0, 1] and rho [F, 1560].
# Prints each gate PASS/FAIL and `[r55_smoke] failures: N`.
#SBATCH --job-name=r55_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=node52
#SBATCH --output=logs/r55_smoke_%j.out
#SBATCH --error=logs/r55_smoke_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
DATA_ROOT=~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark
D=~/Data/dataggen/outputs/five_bench
ANCHOR_ROOT=$D/anchors
MANIFEST=evaluation/r7_anchor_manifest.json
RUN=$D/r55_smoke/job${SLURM_JOB_ID:-local}
CASES=$RUN/cases.json
METHOD=n5_tgt09
PY_SG=~/anaconda3/envs/streamgve/bin/python
PY_FB=~/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$RUN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "[r55_smoke] job=${SLURM_JOB_ID:-local} node=$(hostname) run=${RUN}"; nvidia-smi -L 2>&1 | head -2

FAILED=0
gate() {  # $1 = gate name, $2 = exit code of its check
  if [ "$2" -eq 0 ]; then echo "[r55_smoke] $1 PASS"; else echo "[r55_smoke] $1 FAIL"; FAILED=$((FAILED + 1)); fi
}

# ---- the 2 smoke clips, in the manifest's own format
$PY_FB - "$MANIFEST" "$CASES" <<'EOF' || { echo "[r55_smoke] FATAL: smoke clips not in the manifest"; exit 1; }
import json, sys
want = {("0001_bus", 1), ("0011_lucia", 5)}
cases = [c for c in json.load(open(sys.argv[1])) if (c["video_name"], int(c["edit_type"])) in want]
assert {(c["video_name"], int(c["edit_type"])) for c in cases} == want, cases
json.dump(cases, open(sys.argv[2], "w"), indent=1)
print(f"[r55_smoke] cases: {[(c['video_name'], c['edit_type']) for c in cases]}")
EOF
TYPES="1 5"

# ---- one LongLive draft run (all smoke clips); $1 = tag, $2 = 1 -> recorder ON
run_ll() {
  local tag=$1 attn=$2 T
  for T in $TYPES; do
    local ATTN_ARG=(); [ "$attn" = 1 ] && ATTN_ARG=(--attn_out_dir "$RUN/$tag/attn")
    echo "--- [r55_smoke] LongLive draft ${tag} (recorder $([ "$attn" = 1 ] && echo ON || echo OFF)) edit${T} $(date '+%F %T') ---"
    $PY_SG evaluation/r31_stage1.py --backbone ll --edit_type "$T" --method "$METHOD" --data_root "$DATA_ROOT" \
      --out_root "$RUN/$tag/frames" --latent_dir "$RUN/$tag/latents" "${ATTN_ARG[@]}" --cases_json "$CASES" \
      --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --dump_steps 4 --measure_index 4 --step 5 --flow_shift 1.0 \
      --fg_boost_factor 2 --blend_power 2 --blend_sched tgt0.9 --seed 0 || return 1
  done
}
same_latents() {  # $1, $2 = tags: every smoke latent npz has bit-identical z_trg and z_src
  $PY_FB - "$RUN/$1/latents" "$RUN/$2/latents" <<'EOF'
import sys, numpy as np
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
fa = sorted(a.glob("edit*/*.npz")); fb = sorted(b.glob("edit*/*.npz")); bad = 0
assert [f.parent.name + "/" + f.name for f in fa] == [f.parent.name + "/" + f.name for f in fb] and fa, "different clip sets"
for f in fa:
    x, y = np.load(f), np.load(b / f.parent.name / f.name)
    for k in ("z_trg", "z_src"):
        eq = np.array_equal(x[k], y[k]); bad += not eq
        if not eq:
            print(f"  {f.parent.name}/{f.stem} {k}: max |diff| = {float(np.abs(x[k].astype(np.float64) - y[k].astype(np.float64)).max()):.3e}")
print(f"  {len(fa)} clips x (z_trg, z_src): {'identical' if not bad else str(bad) + ' differ'}")
sys.exit(int(bad > 0))
EOF
}

# ---- G1
run_ll on 1 || { echo "[r55_smoke] FAILED the recorder-ON LongLive run"; FAILED=$((FAILED + 1)); }
run_ll off 0 || { echo "[r55_smoke] FAILED the recorder-OFF LongLive run"; FAILED=$((FAILED + 1)); }
same_latents on off; RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[r55_smoke] G1: on != off -- running a second OFF run as the determinism control"
  run_ll off2 0 && { same_latents off off2 && echo "[r55_smoke] G1 control: off == off2 (deterministic), so the RECORDER changes the render" \
                                           || echo "[r55_smoke] G1 control: off != off2 -- the LongLive draft is NOT deterministic run to run; G1 cannot be decided this way"; }
fi
gate "G1 recorder is read-only (latents on == off)" $RC

# ---- G2
$PY_FB - "$RUN/on/attn" "$RUN/on/latents" <<'EOF'
import sys, numpy as np
from pathlib import Path
attn, lat = Path(sys.argv[1]), Path(sys.argv[2]); bad = 0
files = sorted(attn.glob("edit*/*.npz"))
if len(files) != 2: print(f"  expected 2 attention npz, found {len(files)}"); bad += 1
for f in files:
    a = np.load(f); z = np.load(lat / f.parent.name / f.name)
    F = z["z_trg"].shape[0]; probs = []
    need = {"src_tokens", "trg_tokens", "src_prompt", "trg_prompt", "idx_src", "idx_trg", "n_prompt_src", "n_prompt_trg"}
    if not need <= set(a.files): probs.append(f"missing keys {sorted(need - set(a.files))}")
    else:
        for k, nt in (("src_tokens", len(a["idx_src"])), ("trg_tokens", len(a["idx_trg"]))):
            if a[k].shape != (nt, F, 30, 52): probs.append(f"{k} {a[k].shape} != {(nt, F, 30, 52)}")
        for k in ("src_prompt", "trg_prompt"):
            if a[k].shape != (F, 30, 52): probs.append(f"{k} {a[k].shape} != {(F, 30, 52)}")
        for k in ("src_tokens", "trg_tokens", "src_prompt", "trg_prompt"):
            v = a[k]
            if not np.isfinite(v).all(): probs.append(f"{k} not finite")
            elif v.min() < 0: probs.append(f"{k} has negative mass {v.min():.3g}")
            elif not v.any(): probs.append(f"{k} is all zero (the recorder never accumulated)")
    print(f"  {f.parent.name}/{f.stem}: F={F} tokens src/trg={len(a['idx_src']) if 'idx_src' in a.files else '?'}/{len(a['idx_trg']) if 'idx_trg' in a.files else '?'} "
          f"-- {'ok' if not probs else '; '.join(probs)}")
    bad += bool(probs)
sys.exit(int(bad > 0))
EOF
gate "G2 attention npz: keys, shapes, finite, non-zero" $?

# ---- G3: Self-Forcing regression against R48's stored attention of 0001_bus edit 1
SFC=$RUN/cases_sf.json
$PY_FB - "$CASES" "$SFC" <<'EOF'
import json, sys
json.dump([c for c in json.load(open(sys.argv[1])) if c["video_name"] == "0001_bus" and int(c["edit_type"]) == 1], open(sys.argv[2], "w"), indent=1)
EOF
echo "--- [r55_smoke] Self-Forcing draft with the recorder ON (R48's flags), 0001_bus edit1 $(date '+%F %T') ---"
$PY_SG evaluation/r31_stage1.py --edit_type 1 --method "$METHOD" --data_root "$DATA_ROOT" \
  --out_root "$RUN/sf/frames" --latent_dir "$RUN/sf/latents" --attn_out_dir "$RUN/sf/attn" --cases_json "$SFC" \
  --vp_mode vp --first_frame_edit_dir "$ANCHOR_ROOT" --dump_steps 4 --measure_index 4 --step 5 --flow_shift 1.0 \
  --fg_boost_factor 4 --blend_power 2 --blend_sched tgt0.9 --rollout_chunk_size 21 --seed 0
RC_SF=$?
$PY_FB - "$RUN/sf/attn/edit1/0001_bus.npz" "$D/r48_se_attn/edit1/0001_bus.npz" <<'EOF'
import sys, numpy as np
new, ref = np.load(sys.argv[1]), np.load(sys.argv[2]); bad = 0
for k in ("src_tokens", "trg_tokens", "src_prompt", "trg_prompt", "idx_src", "idx_trg", "n_prompt_src", "n_prompt_trg"):
    eq = np.array_equal(new[k], ref[k]); bad += not eq
    print(f"  {k}: {'identical' if eq else 'DIFFERS (max |diff| %.3e)' % float(np.abs(new[k].astype(np.float64) - ref[k].astype(np.float64)).max())}")
sys.exit(int(bad > 0))
EOF
RC=$?; [ "$RC_SF" -ne 0 ] && RC=1
gate "G3 Self-Forcing recorder == R48's stored attention (0001_bus edit1)" $RC

# ---- G4: divergence + R49's field on the LongLive smoke roots
RC4=0
for NAME_ARM in "se_dino dino_patch" "se_lpips lpips"; do
  read -r NAME ARM <<< "$NAME_ARM"
  for T in $TYPES; do
    $PY_FB evaluation/r31_divergence.py --edit_type "$T" --arm "$ARM" --arm_dir_name "$NAME" --cases "$CASES" \
      --data_root "$DATA_ROOT" --target_root "$RUN/on/frames/$METHOD/step04" --latent_dir "$RUN/on/latents" --out_root "$RUN/div" \
      --temporal_reduce mean --no-save_native --seed 0 || RC4=1
  done
done
$PY_FB evaluation/r49_field.py --hosts se --cases "$CASES" --attn_se "$RUN/on/attn" --div_root "$RUN/div" --out_root "$RUN/field" || RC4=1
$PY_FB - "$RUN/field/se_max" <<'EOF' || RC4=1
import sys, numpy as np
from pathlib import Path
files = sorted(Path(sys.argv[1]).glob("edit*/*.npz")); bad = 0
if len(files) != 2: print(f"  expected 2 field npz, found {len(files)}"); bad += 1
for f in files:
    z = np.load(f); q = z["q"]; rho = z["rho"]
    ok = np.isfinite(q).all() and q.min() >= 0 and q.max() <= 1 and rho.shape == (q.shape[0], 1560) and np.isfinite(rho).all()
    print(f"  {f.parent.name}/{f.stem}: q in [{q.min():.3f}, {q.max():.3f}], mean {q.mean():.3f}; rho {rho.shape} in [{rho.min():.2f}, {rho.max():.2f}] -- {'ok' if ok else 'BAD'}")
    bad += not ok
sys.exit(int(bad > 0))
EOF
gate "G4 divergence + field build on the LongLive draft" $RC4

echo "[r55_smoke] failures: ${FAILED}"
exit $(( FAILED > 0 ))
