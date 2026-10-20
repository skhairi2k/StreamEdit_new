#!/bin/bash
# R49 post (CPU): scores + figures + grids, after the eval job(s). Submitted twice: by r49_chain.sh after the main eval, and
# by r49_matched59.sh after the matched-59 eval (r49_score.py adds the matched-59 curve once its CSVs exist).
#
# Usage: sbatch slurm_scripts/five_bench/r49_post.sh
# Output: evaluation/csv/r49_points.csv, r49_deltas.csv, evaluation/figures/r49_pareto/, evaluation/figures/r49_grids/
#SBATCH --job-name=r49_post
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r49_post_%j.out
#SBATCH --error=logs/r49_post_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
PY_FB=/home/ids/skhairi/anaconda3/envs/five-bench/bin/python
cd "$REPO" || exit 1
mkdir -p logs
echo "[r49_post] job=${SLURM_JOB_ID:-local} node=$(hostname)"

FAILED=0
$PY_FB evaluation/r49_score.py || { echo "[r49_post] FAILED r49_score.py"; FAILED=$((FAILED + 1)); }
$PY_FB evaluation/r49_grids.py || { echo "[r49_post] FAILED r49_grids.py"; FAILED=$((FAILED + 1)); }

for F in evaluation/csv/r49_points.csv evaluation/csv/r49_deltas.csv \
         evaluation/figures/r49_pareto/se_five_acc.pdf evaluation/figures/r49_pareto/wan_five_acc.pdf \
         evaluation/figures/r49_grids/frames_se.png evaluation/figures/r49_grids/frames_wan.png; do
  [ -s "$F" ] || { echo "[r49_post] MISSING ${F}"; FAILED=$((FAILED + 1)); }
done
N_PAGES=$(ls evaluation/figures/r49_grids/fields/*/*.png 2>/dev/null | wc -l)
echo "[r49_post] field pages: ${N_PAGES} (expect 36)"
[ "$N_PAGES" -ne 36 ] && FAILED=$((FAILED + 1))
echo "[r49_post] failures: ${FAILED}"
exit $(( FAILED > 0 ))
