#!/bin/bash
# R42 scores -- Grounded-SAM object masks + two structural divergences (DINOv3 per-patch
# 1-cos, DINO v1 key self-similarity) between source frame 0 and the VP anchor, inside
# M = M_src | M_trg, for all 419 FiVE-Bench pairs. One job, one GPU, no rollout.
#
# Writes evaluation/csv/r42_shape_scores.csv (419 rows) and one npz per pair under
# ~/Data/dataggen/outputs/five_bench/r42_scores/edit{T}/{video}.npz.
# Post-run GUARD: 419 rows = 100/100/100/100/9/10 per edit type, 0 [ERROR] lines.
#
# Two ways to run:
#   sbatch slurm_scripts/five_bench/r42_scores.sh            # queued, log logs/r42_scores_<jobid>.out
#   bash   slurm_scripts/five_bench/r42_scores.sh            # on the current node's GPU (e.g. an
#                                                           # interactive L40S allocation)
# Either way the python output is also tee'd to logs/r42_scores_<run_id>.py.log, which is
# what the [ERROR] count reads, so the guard does not depend on Slurm's log naming.
#
# ⚠️ ENV IS `streamgve`, NOT `five-bench`: five-bench (py3.11, torch 2.5.1) has no `sam2`,
# so Grounded-SAM cannot load there (local run 2026-10-01: ModuleNotFoundError). The
# --smoke validation ran in streamgve (py3.12, torch 2.8). No lpips is needed here.
# All weights are in the local caches (HF: grounding-dino-base, sam2-hiera-large,
# dinov3-vitb16; torch hub: dino_vitbase8_pretrain.pth).
#SBATCH --job-name=r42_scores
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=node01,node51,node52,node57
#SBATCH --output=logs/r42_scores_%j.out
#SBATCH --error=logs/r42_scores_%j.err

# NOTE: no `set -u` -- conda's activate-gcc_linux-64.sh hook reads an unbound
# SYS_SYSROOT and would abort the job at `conda activate` (job 907410).

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
D=~/Data/dataggen/outputs/five_bench
OUT_DIR=$D/r42_scores
CSV=evaluation/csv/r42_shape_scores.csv

cd
source .bashrc
conda deactivate
conda activate streamgve             # NOT five-bench -- see header note

cd "$REPO"
mkdir -p logs "$OUT_DIR"

# sbatch sets SLURM_JOB_NAME=r42_scores; inside an interactive allocation SLURM_JOB_ID is
# the allocation's, so a direct `bash` run gets a timestamped id instead.
if [ "${SLURM_JOB_NAME:-}" = "r42_scores" ]; then RUN_ID=${SLURM_JOB_ID}; else RUN_ID=local_$(date +%Y%m%d_%H%M%S); fi
PYLOG=logs/r42_scores_${RUN_ID}.py.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1                                   # compute nodes have no network

echo "[r42_scores] run=${RUN_ID} node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -2

python evaluation/r42_shape_scores.py \
  --manifest evaluation/r7_anchor_manifest.json \
  --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
  --anchor_root "$D/anchors" \
  --prompts evaluation/r42_grounding_prompts.json \
  --out_dir "$OUT_DIR" \
  -o "$CSV" \
  --seed 0 2>&1 | tee "$PYLOG"
RC=${PIPESTATUS[0]}
echo "[r42_scores] python exit=${RC}"

# ---- post-run guard: exact per-type coverage ---------------------------------------
python - "$CSV" <<'PY'
import csv, sys
from collections import Counter
expect = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
rows = list(csv.DictReader(open(sys.argv[1])))
got = Counter(int(r["edit_type"]) for r in rows)
ok = len(rows) == 419 and all(got[t] == n for t, n in expect.items())
print(f"GUARD {'PASS' if ok else 'FAIL'}: {len(rows)} rows (expect 419) "
      + " ".join(f"edit{t}={got[t]}/{n}" for t, n in expect.items()))
sys.exit(0 if ok else 1)
PY
GUARD=$?

N_ERR=$(grep -c "^\[ERROR\]" "$PYLOG")
echo "[r42_scores] [ERROR] lines: ${N_ERR}"
[ "$RC" -eq 0 ] && [ "$GUARD" -eq 0 ] && [ "$N_ERR" -eq 0 ] && exit 0 || exit 1
