#!/bin/bash
# R66 smoke ON JEAN ZAY: ContextFlow (Wan2.1-I2V-14B-480P, frame-count patch) on the 4 smoke pairs of evaluation/r66_cases_smoke.json
# (edit2/0002_girl-dog, edit3/0002_girl-dog, edit5/0009_bear, edit6/0002_girl-dog: swap, swap/colour, insert, delete), with the
# authors' setup: 2 H100, torchrun --nproc_per_node=2, FSDP + Ulysses 2, t5_cpu, offload_model (env $WORK/envs/contextflow).
# Moved from the lab L40S on 2026-11-02: 46 GB was too small for this setup (job 1080831: 45.4 GB of 46, 3 of 4 pairs OOM).
#
# Usage (Jean Zay, in the clone): sbatch slurm_scripts/five_bench/r66_smoke_jz.sh
# Jean Zay header written by hand (r50_port.sh only rewrites its own SCRIPTS list and would force --gres=gpu:1).
# Runs r66_contextflow_run.py twice: --frames 41 (the render setting) then --frames 81 (diagnostic of the patch at the authors'
# length). The driver stops a run at its first failed pair (FSDP cannot recover from an exception in its forward); an OOM at one
# frame count is a result (driver code 2), any other error fails the job. torchrun does not pass the driver's code through, so it
# is read from exitcode_n{F}_shard0.txt (written by rank 0 at the end; missing = the run died).
#
# Resources: 48 CPUs = 24 per GPU (gpu_p6 nodes: 4 H100 + 96 cores; memory follows the CPU count, each FSDP rank needs ~72 GB of RAM
# for the fp32 checkpoint + T5 before sharding); OMP_NUM_THREADS=24 per rank (torchrun's default 1 would starve the CPU T5).
# NVLink between the H100: no NCCL_P2P_DISABLE (that was the L40S PCIe workaround).
#
# Paths: repo = this clone; CF_DIR=$WORK/code/ContextFlow, CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P (jz-env / jz-push);
# case paths relative to --data_dir $WORK/Data. Outputs: ~/Data/dataggen/outputs/five_bench/r66_contextflow/smoke, where
# $WORK/Data/dataggen/outputs/five_bench/r66_contextflow is a symlink to $SCRATCH/five_bench_out/r66_contextflow (made here if
# missing, as jz/r50_scratch_links.sh does for the Wan folders), so renders and latents never count against the $WORK quota:
#   n41/edit{T}/{video}/*.png, n81/edit{T}/{video}/*.png     ContextFlow renders
#   latents/n41/{video}.pt, latents/n81/{video}.pt          RF-Solver inversion latents (reused on resubmit)
#   manifest_n41_shard0.csv, manifest_n81_shard0.csv        per pair: n_frames, sec_inversion, sec_edit, peak_mem_gb (max over the
#                                                           2 GPUs), status
#   nvidia_smi_mem_n{F}.txt                                 index, memory.used of both GPUs every 5 s (peak per GPU printed)
# Read by evaluation/r66_smoke_check.py (run on the Jean Zay frontal).
#SBATCH --job-name=r66_smoke_jz
#SBATCH -A omt@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/r66_smoke_jz_%j.out
#SBATCH --error=logs/r66_smoke_jz_%j.err

REPO=${SLURM_SUBMIT_DIR:-$PWD}
CASES=evaluation/r66_cases_smoke.json
PY_CF=$WORK/envs/contextflow/bin/python
export CF_DIR=$WORK/code/ContextFlow
export CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P
DATA_DIR=$WORK/Data
LINK=$WORK/Data/dataggen/outputs/five_bench/r66_contextflow
TARGET=$SCRATCH/five_bench_out/r66_contextflow
OUT=$LINK/smoke

cd "$REPO" || exit 1
[ -f "$CASES" ] || { echo "[r66_smoke_jz] ERROR: run sbatch from the clone ($CASES not found in $REPO)" >&2; exit 1; }
module purge
mkdir -p logs
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

echo "[r66_smoke_jz] job=${SLURM_JOB_ID} node=$(hostname) repo=${REPO} cases=${CASES} out=${OUT}"
nvidia-smi -L 2>&1 | head -4
for X in "$PY_CF" "$CF_DIR/wan/image2video.py" "$CF_WEIGHTS/diffusion_pytorch_model.safetensors.index.json" \
         "$DATA_DIR/FiVE-Fine-Grained-Video-Editing-Benchmark/images" "$DATA_DIR/dataggen/outputs/five_bench/anchors_r43_correct"; do
  [ -e "$X" ] || { echo "[r66_smoke_jz] ERROR: $X missing (jz-env / jz-push / R50 staging)" >&2; exit 1; }
done
if [[ "$(nvidia-smi -L | wc -l)" -ne 2 ]]; then
  echo "[r66_smoke_jz] ERROR: expected 2 GPUs, got $(nvidia-smi -L | wc -l)" >&2
  exit 1
fi
# outputs on $SCRATCH behind a symlink in $WORK/Data
if [ -L "$LINK" ]; then
  [ "$(readlink "$LINK")" = "$TARGET" ] || { echo "[r66_smoke_jz] ERROR: $LINK links to $(readlink "$LINK"), not $TARGET" >&2; exit 1; }
elif [ -e "$LINK" ]; then
  echo "[r66_smoke_jz] ERROR: $LINK exists and is not a symlink to $TARGET (move its content to $TARGET first)" >&2; exit 1
else
  mkdir -p "$TARGET" "$(dirname "$LINK")" && ln -s "$TARGET" "$LINK" && echo "[r66_smoke_jz] linked $LINK -> $TARGET"
fi
mkdir -p "$TARGET" "$OUT"

FAILED=0
for F in 41 81; do
  MEMLOG=$OUT/nvidia_smi_mem_n${F}.txt
  CODEFILE=$OUT/exitcode_n${F}_shard0.txt
  rm -f "$CODEFILE"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 > "$MEMLOG" &
  SMI_PID=$!
  "$PY_CF" -m torch.distributed.run --standalone --nproc_per_node=2 \
    evaluation/r66_contextflow_run.py --cases "$CASES" --frames "$F" --out_root "$OUT" --shard 0 --n_shards 1 --data_dir "$DATA_DIR"
  RC=$?
  kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null
  PEAKS=$(awk -F', ' '{if ($2 > p[$1]) p[$1] = $2} END {for (g in p) printf "GPU%s %s MiB  ", g, p[g]}' "$MEMLOG")
  CODE=$(cat "$CODEFILE" 2>/dev/null || echo "none")
  case $CODE in
    0) echo "[r66_smoke_jz] n${F}: all pairs ok; nvidia-smi peak ${PEAKS}" ;;
    2) echo "[r66_smoke_jz] n${F}: stopped at an OOM (recorded as a result); nvidia-smi peak ${PEAKS}" ;;
    *) echo "[r66_smoke_jz] n${F}: FAILED (driver code ${CODE}, torchrun exit ${RC}); nvidia-smi peak ${PEAKS}" >&2
       FAILED=$((FAILED + 1)) ;;
  esac
  echo "---- manifest n${F} ----"
  cat "$OUT/manifest_n${F}_shard0.csv" 2>/dev/null || echo "(no manifest)"
done

echo "[r66_smoke_jz] failures: ${FAILED}"
exit $(( FAILED > 0 ))
