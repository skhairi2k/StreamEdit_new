#!/bin/bash
# R51 stage 2 (CPU): demand-component ablation fields on the 419 pairs, from R49's full-stage inputs.
#
# Usage: sbatch slurm_scripts/five_bench/r51_field.sh se     # lab: A1 nodiv, A2 dino, A3 lpips, A4 nosupp  (r51_se_chain.sh)
#        sbatch slurm_scripts/five_bench/r51_field.sh wan    # Jean Zay: A4 nosupp at s_min 5                 (r51_wan_chain.sh)
#        sbatch slurm_scripts/five_bench/r51_field.sh se linear max $D/r52_field_full   # R52's B1: S * max, rho = 2 + 48 q  (r52_se_chain.sh)
# Optional arguments after HOST, all with the R51 behaviour as default:  MAPPING (inverse | linear), ARMS (quoted list, default per host),
# OUT (default $D/r51_field_full). `linear` is StreamEdit only and writes {host}_{arm}_linrho dirs.
# Runs `r51_field.py --selftest`, then the field. r51_field.py first recomputes A0 (R49's own q = S * max) for every pair and
# requires bit-equality with $D/r49_field_full (se_max rho / wan_max_s5 start maps); on any mismatch it writes nothing and exits 1.
# Inputs: attention $D/r48_se_attn | $D/r48_wan_draft/attn, divergence $D/r48_div/{se,wan}_{dino,lpips} (R49's full-stage roots).
# Output: $D/r51_field_full/{se_nodiv, se_dino, se_lpips, se_nosupp | wan_nosupp_s5}/edit{T}/{video}.npz (+ .npy start maps for wan)
#SBATCH --job-name=r51_field
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r51_field_%j.out
#SBATCH --error=logs/r51_field_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
HOST=${1:-}
MAPPING=${2:-inverse}
ARMS=${3:-}
OUT=${4:-$D/r51_field_full}
case "$HOST" in
  se)  DEFAULT_ARMS="nodiv dino lpips nosupp"; SUF=${MAPPING/#linear/_linrho}; SUF=${SUF/#inverse/}; PRE=se_;  POST="$SUF" ;;
  wan) DEFAULT_ARMS="nosupp";                  PRE=wan_; POST=_s5 ;;
  *) echo "[r51_field] usage: r51_field.sh se|wan [inverse|linear [ARMS [OUT]]]"; exit 1 ;;
esac
case "$MAPPING" in inverse|linear) ;; *) echo "[r51_field] bad MAPPING '${MAPPING}' (inverse|linear)"; exit 1 ;; esac
[ "$MAPPING" = linear ] && [ "$HOST" = wan ] && { echo "[r51_field] --mapping linear is StreamEdit only"; exit 1; }
DIRS=(); for A in ${ARMS:-$DEFAULT_ARMS}; do DIRS+=("${PRE}${A}${POST}"); done
CASES=evaluation/r7_anchor_manifest.json
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
echo "[r51_field] job=${SLURM_JOB_ID:-local} node=$(hostname) host=${HOST} mapping=${MAPPING} arms=${ARMS:-$DEFAULT_ARMS} cases=${CASES} out=${OUT}"

$PY_FB evaluation/r51_field.py --selftest || { echo "[r51_field] FAILED selftest"; exit 1; }
ARMS_ARG=(); [ -n "$ARMS" ] && ARMS_ARG=(--arms $ARMS)
$PY_FB evaluation/r51_field.py --host "$HOST" --mapping "$MAPPING" "${ARMS_ARG[@]}" --cases "$CASES" --div_root "$D/r48_div" --gate_ref "$D/r49_field_full" --out_root "$OUT"
RC=$?

# ---- post-run guard: one npz per pair in each arm dir (and one .npy start map per pair for wan)
N_CASES=$($PY_FB -c "import json; print(len(json.load(open('$CASES'))))")
FAILED=$(( RC != 0 ))
for DIR in "${DIRS[@]}"; do
  N=$(ls "$OUT/$DIR"/edit*/*.npz 2>/dev/null | wc -l)
  echo "[r51_field] ${DIR}: ${N} npz (expect ${N_CASES})"
  [ "$N" -ne "$N_CASES" ] && FAILED=$((FAILED + 1))
  if [[ "$DIR" == wan_* ]]; then
    NN=$(ls "$OUT/$DIR"/edit*/*.npy 2>/dev/null | wc -l)
    [ "$NN" -ne "$N_CASES" ] && { echo "[r51_field] ${DIR}: ${NN} start maps (expect ${N_CASES})"; FAILED=$((FAILED + 1)); }
  fi
done
echo "[r51_field] failures: ${FAILED}"
exit $(( FAILED > 0 ))
