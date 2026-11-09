#!/bin/bash
# R68 stage 2 (CPU): the ablation fields that complete the FiVE-Bench-Correct ablation table beyond R62.
#
# Usage: sbatch slurm_scripts/five_bench/r68_field.sh se     # lab: B1 = R52's linear map rho = 2 + 48 q of se_max's demand, 415 pairs  (r68_se_chain.sh)
#        sbatch slurm_scripts/five_bench/r68_field.sh wan    # Jean Zay: A1 nodiv, A2 dino, A3 lpips at s_min 5, 415 pairs             (r68_wan_chain.sh)
# Runs `r51_field.py --selftest`, then the field. r51_field.py first recomputes A0 (q = S * max, exposure inverse) for every pair and requires
# bit-equality with the A0 field; on any mismatch it writes nothing and exits 1.
#   se : ONE call, on lab /scratch ($SC, the copies staged from /store by step stage-r59-inputs). attention $SC/r59_se_attn, divergence
#        $SC/r59_div/se_{dino,lpips}, gate $SC/r59_field_full/se_max, cases r43_anchor_manifest_correct.json (415), --mapping linear --arms max
#        -> $SC/r68_field/se_max_linrho (+ 'linear check PASS 415/415').
#   wan: TWO calls into the same output root, the split of A0 = R58 max_s5's own assembly:
#        (1) the 190 pairs of r58_cases_maxs5.json   : attention $D/r58_wan_draft/attn, divergence $D/r58_div,  gate $D/r58_field/wan_max_s5
#        (2) the 225 pairs of r68_cases_reuse225.json: attention $D/r48_wan_draft/attn, divergence $D/r48_div,  gate $D/r49_field_full/wan_max_s5
#            (Correct inputs = official inputs for these, checked by r68_build_cases.py)
# Output: se $SC/r68_field/se_max_linrho | wan $D/r68_field/{wan_nodiv_s5, wan_dino_s5, wan_lpips_s5}, /edit{T}/{video}.npz (+ .npy start maps
# for wan), 415 per arm dir.
#SBATCH --job-name=r68_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r68_field_%j.out
#SBATCH --error=logs/r68_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench       # Jean Zay: links into $SCRATCH (r50_scratch_links.sh)
SC=/scratch/PERSO/skhairi/five_bench       # lab: Slurm jobs read and write lab /scratch only (R58 storage policy; ~/Data/.../r59_* -> /store, frontal-only)
HOST=${1:-}
MANIFEST=evaluation/r43_anchor_manifest_correct.json
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
# one "CASES|ATTN|DIV|GATE" entry per r51_field.py call
case "$HOST" in
  se)  MAPPING=linear;  ARMS="max";              PRE=se_;  POST=_linrho; GATE_SUB=se_max;     GATE_EXT=npz; OUT=$SC/r68_field
       CALLS=("$MANIFEST|$SC/r59_se_attn|$SC/r59_div|$SC/r59_field_full") ;;   # staged from /store by step stage-r59-inputs
  wan) MAPPING=inverse; ARMS="nodiv dino lpips"; PRE=wan_; POST=_s5;     GATE_SUB=wan_max_s5; GATE_EXT=npy; OUT=$D/r68_field
       CALLS=("evaluation/r58_cases_maxs5.json|$D/r58_wan_draft/attn|$D/r58_div|$D/r58_field"
              "evaluation/r68_cases_reuse225.json|$D/r48_wan_draft/attn|$D/r48_div|$D/r49_field_full") ;;
  *) echo "[r68_field] usage: r68_field.sh se|wan"; exit 1 ;;
esac
DIRS=(); for A in $ARMS; do DIRS+=("${PRE}${A}${POST}"); done

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r68_field] job=${SLURM_JOB_ID:-local} node=$(hostname) host=${HOST} mapping=${MAPPING} arms=${ARMS} out=${OUT} -> ${DIRS[*]}"

ncases() { $PY_FB -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$1"; }

# ---- preflight: every call's case file and upstream inputs complete, gate present; the calls together cover the manifest
N_MANIFEST=$(ncases "$MANIFEST")
FAILED=0; N_SUM=0
for C in "${CALLS[@]}"; do
  IFS='|' read -r CASES ATTN DIV GATE <<< "$C"
  [ -s "$CASES" ] || { echo "[r68_field] FATAL: $CASES missing (python evaluation/r68_build_cases.py? git pull?)"; FAILED=$((FAILED + 1)); continue; }
  N_CASES=$(ncases "$CASES"); N_SUM=$((N_SUM + N_CASES))
  for X in "$ATTN" "$DIV/${PRE}dino" "$DIV/${PRE}lpips"; do
    N=$(ls "$X"/edit*/*.npz 2>/dev/null | wc -l)
    echo "[r68_field] ${CASES}: ${X}: ${N} npz (need >= ${N_CASES})"
    [ "$N" -lt "$N_CASES" ] && FAILED=$((FAILED + 1))
  done
  N=$(ls "$GATE/$GATE_SUB"/edit*/*."$GATE_EXT" 2>/dev/null | wc -l)
  echo "[r68_field] ${CASES}: gate ${GATE}/${GATE_SUB}: ${N} .${GATE_EXT} (need >= ${N_CASES})"
  [ "$N" -lt "$N_CASES" ] && FAILED=$((FAILED + 1))
done
[ "$N_SUM" -ne "$N_MANIFEST" ] && { echo "[r68_field] FATAL: the calls cover ${N_SUM} pairs, the manifest holds ${N_MANIFEST}"; FAILED=$((FAILED + 1)); }
[ "$FAILED" -ne 0 ] && { echo "[r68_field] FATAL: inputs incomplete -- nothing run"; echo "[r68_field] failures: ${FAILED}"; exit 1; }

$PY_FB evaluation/r51_field.py --selftest || { echo "[r68_field] FAILED selftest"; echo "[r68_field] failures: 1"; exit 1; }
RC=0
for C in "${CALLS[@]}"; do
  IFS='|' read -r CASES ATTN DIV GATE <<< "$C"
  echo "--- [r68_field] ${CASES} $(date '+%F %T') ---"
  $PY_FB evaluation/r51_field.py --host "$HOST" --mapping "$MAPPING" --arms $ARMS --cases "$CASES" \
    --attn_root "$ATTN" --div_root "$DIV" --gate_ref "$GATE" --out_root "$OUT" \
    || { echo "[r68_field] FAILED r51_field.py on ${CASES}"; RC=1; }
done

# ---- post-run guard: one npz per manifest pair in each arm dir (and one .npy start map per pair for wan)
FAILED=$RC
for DIR in "${DIRS[@]}"; do
  N=$(ls "$OUT/$DIR"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r68_field] ${DIR}: ${N} npz (expect ${N_MANIFEST})"
  [ "$N" -ne "$N_MANIFEST" ] && FAILED=$((FAILED + 1))
  if [[ "$DIR" == wan_* ]]; then
    NN=$(ls "$OUT/$DIR"/edit*/*.npy 2>/dev/null | wc -l)
    echo "[r68_field] ${DIR}: ${NN} start maps (expect ${N_MANIFEST})"
    [ "$NN" -ne "$N_MANIFEST" ] && FAILED=$((FAILED + 1))
  fi
done
echo "[r68_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
