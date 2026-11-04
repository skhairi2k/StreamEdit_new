#!/bin/bash
# R66 render ON JEAN ZAY: ContextFlow (Wan2.1-I2V-14B-480P, frame-count patch) on the 415 FiVE-Bench-Correct pairs of
# evaluation/r66_cases_correct.json at 41 frames (first min(41, n) source frames, 4k+1), with the authors' setup: 2 H100 per task,
# torchrun --nproc_per_node=2, FSDP + Ulysses 2, t5_cpu, offload_model (env $WORK/envs/contextflow).
#
# Array of K = 4 video shards (set at the gate, 2026-11-04, from smoke job 1902592: inversion 145 s, edit 380 s per pair, 4.15 pairs
# per video -> 25 videos x 1720 s x 1.3 = 15.6 h per shard, < 18 h). Task i renders the videos j with j mod 4 = i, in the case-file
# order (the driver's rule), every pair of each video: 105 / 103 / 104 / 103 pairs. Each video's RF-Solver inversion runs once, on its
# first pair, and is reused by its other pairs.
#
# Usage (Jean Zay, in the clone):  sbatch slurm_scripts/five_bench/r66_render_jz.sh            all 4 shards
#                                  sbatch --array=2 slurm_scripts/five_bench/r66_render_jz.sh  resubmit one shard
# Resume: the driver skips every pair whose folder already holds its n41 PNGs and reuses saved latents, so a resubmit (after the 20 h
# limit, an OOM or any error) only does what is missing. The driver stops a shard at its first failed pair (FSDP cannot recover from
# an exception inside its forward); the guard below then reports the missing pairs and the task exits 1: resubmit that shard.
# Jean Zay header written by hand, as r66_smoke_jz.sh (r50_port.sh only rewrites its own SCRIPTS list).
#
# Resources per task: 48 CPUs = 24 per GPU (memory follows the CPU count; each FSDP rank needs ~72 GB of RAM for the fp32 checkpoint +
# T5 before sharding), OMP_NUM_THREADS=24 per rank (CPU T5); NVLink, so no NCCL_P2P_DISABLE.
#
# Paths: repo = this clone; CF_DIR=$WORK/code/ContextFlow, CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P; case paths relative to
# --data_dir $WORK/Data. Outputs under $WORK/Data/dataggen/outputs/five_bench/r66_contextflow/correct, where r66_contextflow is the
# symlink to $SCRATCH/five_bench_out/r66_contextflow made by r66_smoke_jz.sh (made here if missing; the 4 tasks may start together):
#   n41/edit{T}/{video}/00000.png ...       ContextFlow renders (fetched to the lab /scratch by jz/r66_fetch_renders.sh)
#   latents/n41/{video}.pt                  RF-Solver inversion latents (reused on resubmit; stay on Jean Zay)
#   manifest_n41_shard{i}.csv               per pair: n_frames, sec_inversion, sec_edit, peak_mem_gb, status (appended on resubmit)
#   exitcode_n41_shard{i}.txt               driver code (0 ok, 2 OOM, 1 other error; missing = the run died)
#   nvidia_smi_mem_shard{i}_{jobid}.txt     index, memory.used of both GPUs every 30 s
# Ends with "[r66_render_jz] shard i: failures: N" (N = pairs of the shard without their n41 frames, plus 1 if the driver failed).
#SBATCH --job-name=r66_render_jz
#SBATCH -A omt@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/r66_render_jz_%A_%a.out
#SBATCH --error=logs/r66_render_jz_%A_%a.err

N_SHARDS=4
FRAMES=41
SHARD=${SLURM_ARRAY_TASK_ID:?run as an array job (sbatch, or sbatch --array=<i>)}
REPO=${SLURM_SUBMIT_DIR:-$PWD}
CASES=evaluation/r66_cases_correct.json
PY_CF=$WORK/envs/contextflow/bin/python
export CF_DIR=$WORK/code/ContextFlow
export CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P
DATA_DIR=$WORK/Data
LINK=$WORK/Data/dataggen/outputs/five_bench/r66_contextflow
TARGET=$SCRATCH/five_bench_out/r66_contextflow
OUT=$LINK/correct

cd "$REPO" || exit 1
[ -f "$CASES" ] || { echo "[r66_render_jz] ERROR: run sbatch from the clone ($CASES not found in $REPO)" >&2; exit 1; }
module purge
mkdir -p logs
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=24
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

echo "[r66_render_jz] job=${SLURM_ARRAY_JOB_ID}_${SHARD} node=$(hostname) shard=${SHARD}/${N_SHARDS} frames=${FRAMES} repo=${REPO} out=${OUT}"
nvidia-smi -L 2>&1 | head -4
for X in "$PY_CF" "$CF_DIR/wan/image2video.py" "$CF_WEIGHTS/diffusion_pytorch_model.safetensors.index.json" \
         "$DATA_DIR/FiVE-Fine-Grained-Video-Editing-Benchmark/images" "$DATA_DIR/dataggen/outputs/five_bench/anchors_r43_correct"; do
  [ -e "$X" ] || { echo "[r66_render_jz] ERROR: $X missing (jz-env / jz-push / R50 staging)" >&2; exit 1; }
done
if [[ "$(nvidia-smi -L | wc -l)" -ne 2 ]]; then
  echo "[r66_render_jz] ERROR: expected 2 GPUs, got $(nvidia-smi -L | wc -l)" >&2
  exit 1
fi
# imports that need a GPU at import time (wan t5.py, xfuser envs.py), as in r66_smoke_jz.sh
"$PY_CF" - "$CF_DIR" <<'EOF' || { echo "[r66_render_jz] ERROR: wan / xfuser import check failed on $(hostname)" >&2; exit 1; }
import sys
sys.path.insert(0, sys.argv[1])
from wan import WanI2V  # noqa: F401
from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment  # noqa: F401
from xfuser.core.long_ctx_attention import xFuserLongContextAttention  # noqa: F401
from wan.distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward  # noqa: F401
print("[r66_render_jz] wan + xfuser imports OK")
EOF
# outputs on $SCRATCH behind the symlink in $WORK/Data; the 4 tasks may get here together, so a lost ln race is fine if the link is right
if [ ! -e "$LINK" ] && [ ! -L "$LINK" ]; then
  mkdir -p "$TARGET" "$(dirname "$LINK")"
  ln -s "$TARGET" "$LINK" 2>/dev/null && echo "[r66_render_jz] linked $LINK -> $TARGET"
fi
if [ ! -L "$LINK" ] || [ "$(readlink "$LINK")" != "$TARGET" ]; then
  echo "[r66_render_jz] ERROR: $LINK is not a symlink to $TARGET (move its content to $TARGET first)" >&2; exit 1
fi
mkdir -p "$OUT"

MEMLOG=$OUT/nvidia_smi_mem_shard${SHARD}_${SLURM_ARRAY_JOB_ID}.txt
CODEFILE=$OUT/exitcode_n${FRAMES}_shard${SHARD}.txt
rm -f "$CODEFILE"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 30 > "$MEMLOG" &
SMI_PID=$!
# each array task gets its own rendezvous port (two tasks may share a node)
"$PY_CF" -m torch.distributed.run --standalone --nproc_per_node=2 \
  evaluation/r66_contextflow_run.py --cases "$CASES" --frames "$FRAMES" --out_root "$OUT" \
  --shard "$SHARD" --n_shards "$N_SHARDS" --data_dir "$DATA_DIR"
RC=$?
kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null
PEAKS=$(awk -F', ' '{if ($2 > p[$1]) p[$1] = $2} END {for (g in p) printf "GPU%s %s MiB  ", g, p[g]}' "$MEMLOG")
CODE=$(cat "$CODEFILE" 2>/dev/null || echo "none")
DRIVER_FAIL=0
case $CODE in
  0) echo "[r66_render_jz] shard ${SHARD}: driver ok; nvidia-smi peak ${PEAKS}" ;;
  2) echo "[r66_render_jz] shard ${SHARD}: driver stopped at an OOM (resubmit: sbatch --array=${SHARD} ...); nvidia-smi peak ${PEAKS}" >&2
     DRIVER_FAIL=1 ;;
  *) echo "[r66_render_jz] shard ${SHARD}: driver FAILED (code ${CODE}, torchrun exit ${RC}); nvidia-smi peak ${PEAKS}" >&2
     DRIVER_FAIL=1 ;;
esac

# guard: every pair of this shard (same rule as the driver: video j -> shard j mod N_SHARDS) holds exactly its n41 PNGs
GUARD=$("$PY_CF" - "$CASES" "$OUT" "$SHARD" "$N_SHARDS" "$FRAMES" <<'EOF'
import json, sys
from pathlib import Path
cases_path, out, shard, n_shards, frames = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cases = json.loads(Path(cases_path).read_text())
videos = list(dict.fromkeys(c["video_name"] for c in cases))
mine = {v for j, v in enumerate(videos) if j % n_shards == shard}
mine_cases = [c for c in cases if c["video_name"] in mine]
missing = []
for c in mine_cases:
    d = out / f"n{frames}" / f"edit{c['edit_type']}" / c["video_name"]
    got = len(list(d.glob("*.png"))) if d.is_dir() else 0
    if got != int(c[f"n{frames}"]):
        missing.append(f"{c['case_id']} ({got}/{c[f'n{frames}']} frames)")
print(f"[r66_render_jz] shard {shard}: {len(mine)} videos, {len(mine_cases)} pairs, {len(mine_cases) - len(missing)} complete")
for m in missing[:20]:
    print(f"[r66_render_jz]   missing: {m}")
if len(missing) > 20:
    print(f"[r66_render_jz]   ... and {len(missing) - 20} more")
print(len(missing))
EOF
)
echo "$GUARD" | sed '$d'
MISSING=$(echo "$GUARD" | tail -1)
[[ "$MISSING" =~ ^[0-9]+$ ]] || { echo "[r66_render_jz] ERROR: guard did not run" >&2; MISSING=1; }
echo "---- manifest shard ${SHARD} (last 5 rows) ----"
tail -5 "$OUT/manifest_n${FRAMES}_shard${SHARD}.csv" 2>/dev/null || echo "(no manifest)"
FAILED=$((MISSING + DRIVER_FAIL))
echo "[r66_render_jz] shard ${SHARD}: failures: ${FAILED}"
exit $(( FAILED > 0 ))
