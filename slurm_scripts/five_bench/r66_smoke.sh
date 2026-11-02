#!/bin/bash
# R66 smoke: ContextFlow (Wan2.1-I2V-14B-480P, frame-count patch) on the 4 smoke pairs of evaluation/r66_cases_smoke.json
# (edit2/0002_girl-dog, edit3/0002_girl-dog, edit5/0009_bear, edit6/0002_girl-dog: swap, swap/colour, insert, delete),
# with the authors' setup: 2 L40S, torchrun --nproc_per_node=2, FSDP + Ulysses 2, t5_cpu, offload_model (env `contextflow`).
#
# Usage: sbatch slurm_scripts/five_bench/r66_smoke.sh
# Runs r66_contextflow_run.py twice: --frames 41 (the render setting) then --frames 81 (diagnostic of the patch at the authors'
# length). Each call loads the model once, inverts each video once (latents/n{F}/) and edits its pairs. An OOM at one frame count
# is a result (manifest status 'oom', driver code 2), not a failure; any other error fails the job. torchrun does not pass the
# driver's code through, so it is read from exitcode_n{F}_shard0.txt (written by rank 0 at the end; missing = the run died).
#
# node51 excluded: its GPU GPU-f298644f-ba38-3874-3c7f-c5469e0e514f raises 'uncorrectable ECC error' at the first CUDA call
# (jobs 1072482 on 2026-10-29 and 1080609 on 2026-11-01).
# Resources: --mem=192G because each FSDP rank loads the fp32 checkpoint (61 GiB) and T5 on the CPU (11 GB) before sharding;
# OMP_NUM_THREADS=8 per rank (torchrun's default is 1, which would starve the CPU T5).
#
# Outputs ($R66_ROOT/smoke):
#   n41/edit{T}/{video}/*.png, n81/edit{T}/{video}/*.png     ContextFlow renders
#   latents/n41/{video}.pt, latents/n81/{video}.pt          RF-Solver inversion latents (reused on resubmit)
#   manifest_n41_shard0.csv, manifest_n81_shard0.csv        per pair: n_frames, sec_inversion, sec_edit, peak_mem_gb (max over
#                                                           the 2 GPUs), status
#   nvidia_smi_mem_n{F}.txt                                 index, memory.used of both GPUs sampled every 5 s (peak per GPU printed)
# Read by evaluation/r66_smoke_check.py (contact sheet + timing + projected L40S-hours).
#SBATCH --job-name=r66_smoke
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --exclude=node52,node51
#SBATCH --output=logs/r66_smoke_%j.out
#SBATCH --error=logs/r66_smoke_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
R66_ROOT=${R66_ROOT:-/work/PERSO/skhairi/five_bench/r66_contextflow}
OUT=$R66_ROOT/smoke
CASES=evaluation/r66_cases_smoke.json
PY_CF=~/anaconda3/envs/contextflow/bin/python

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# PCIe peer-to-peer between the L40S hangs NCCL's first collective (smoke job 1080618, probe job 1080823 on node39:
# P2P/CUMEM init completes, all-reduce never returns); host shared memory works (~13.5 GB/s for a 256 MB all-reduce).
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

echo "[r66_smoke] job=${SLURM_JOB_ID} node=$(hostname) cases=${CASES} out=${OUT}"
nvidia-smi -L 2>&1 | head -4
if [[ ! -x "$PY_CF" ]]; then
  echo "[r66_smoke] ERROR: $PY_CF missing (run evaluation/r66_make_contextflow_official_env.sh first)" >&2
  exit 1
fi
if [[ "$(nvidia-smi -L | wc -l)" -ne 2 ]]; then
  echo "[r66_smoke] ERROR: expected 2 GPUs, got $(nvidia-smi -L | wc -l)" >&2
  exit 1
fi

FAILED=0
for F in 41 81; do
  MEMLOG=$OUT/nvidia_smi_mem_n${F}.txt
  CODEFILE=$OUT/exitcode_n${F}_shard0.txt
  rm -f "$CODEFILE"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > "$MEMLOG" &
  SMI_PID=$!
  "$PY_CF" -m torch.distributed.run --standalone --nproc_per_node=2 \
    evaluation/r66_contextflow_run.py --cases "$CASES" --frames "$F" --out_root "$OUT" --shard 0 --n_shards 1
  RC=$?
  kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null
  PEAKS=$(awk -F', ' '{if ($2 > p[$1]) p[$1] = $2} END {for (g in p) printf "GPU%s %s MiB  ", g, p[g]}' "$MEMLOG")
  CODE=$(cat "$CODEFILE" 2>/dev/null || echo "none")
  case $CODE in
    0) echo "[r66_smoke] n${F}: all pairs ok; nvidia-smi peak ${PEAKS}" ;;
    2) echo "[r66_smoke] n${F}: OOM on some pairs (recorded as a result); nvidia-smi peak ${PEAKS}" ;;
    *) echo "[r66_smoke] n${F}: FAILED (driver code ${CODE}, torchrun exit ${RC}); nvidia-smi peak ${PEAKS}" >&2
       FAILED=$((FAILED + 1)) ;;
  esac
  echo "---- manifest n${F} ----"
  cat "$OUT/manifest_n${F}_shard0.csv" 2>/dev/null || echo "(no manifest)"
done

echo "[r66_smoke] failures: ${FAILED}"
exit $(( FAILED > 0 ))
