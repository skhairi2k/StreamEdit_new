#!/bin/bash
# Node health probe: is a GPU actually usable from a batch job on this node?
# USAGE: sbatch --nodelist=node52 slurm_scripts/five_bench/node_probe.sh
# Checks: nvidia-smi, torch CUDA init + matmul + 90%-VRAM alloc in streamgve, data/repo mounts readable.
#SBATCH --job-name=node_probe
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/node_probe_%j.out
#SBATCH --error=logs/node_probe_%j.err

cd /home/ids/skhairi/Code/StreamEdit_bigchantier || { echo "PROBE FAIL repo"; exit 1; }
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda activate streamgve

echo "[probe] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>&1 | head -4
python - <<'PY'
import time, torch
ok = torch.cuda.is_available()
print("[probe] torch", torch.__version__, "cuda_available", ok, "count", torch.cuda.device_count())
if not ok:
    raise SystemExit("PROBE FAIL cuda")
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): y = x @ x
torch.cuda.synchronize()
print("[probe] device", torch.cuda.get_device_name(0), f"matmul {20*2*8192**3/(time.time()-t)/1e12:.1f} TFLOPS")
big = torch.empty(int(0.9 * torch.cuda.get_device_properties(0).total_memory), dtype=torch.uint8, device="cuda")  # 90% of VRAM
print(f"[probe] {big.numel() / 2**30:.1f} GiB alloc ok")
PY
[ $? -eq 0 ] || { echo "PROBE FAIL python"; exit 1; }
for d in ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark ~/Data/dataggen/outputs/five_bench /projects; do
  timeout 20 ls "$d" >/dev/null 2>&1 && echo "[probe] mount ok $d" || echo "[probe] MOUNT FAIL $d"
done
echo "PROBE PASS $(hostname)"
