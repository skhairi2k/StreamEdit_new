#!/bin/bash
# R66 NCCL probe: smoke job 1080618 (node39, 2 L40S) hung on its first NCCL collective (1-element all-reduce, 30 min timeout).
# This 2-GPU probe prints the GPU topology, then runs evaluation/r66_nccl_probe.py under torchrun (a) with NCCL defaults and
# (b) with NCCL_P2P_DISABLE=1 (traffic through host memory instead of PCIe peer-to-peer), each under a 3 min timeout,
# with NCCL_DEBUG=INFO. Look for 'NCCL PROBE PASS' / 'HUNG' per variant in the .out.
#
# Usage: sbatch slurm_scripts/five_bench/r66_nccl_probe.sh        (optionally --nodelist=nodeXX to probe one node)
#SBATCH --job-name=r66_nccl_probe
#SBATCH --partition=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=node52,node51
#SBATCH --output=logs/r66_nccl_probe_%j.out
#SBATCH --error=logs/r66_nccl_probe_%j.err

REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
PY_CF=~/anaconda3/envs/contextflow/bin/python
cd "$REPO" || exit 1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=2

echo "[nccl_probe] job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi -L
echo "---- nvidia-smi topo -m ----"
nvidia-smi topo -m
echo "---- NCCL / torch ----"
"$PY_CF" -c "import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"

run_variant() {
  local name=$1; shift
  echo "==== variant ${name} ($*) ===="
  env "$@" timeout 180 "$PY_CF" -m torch.distributed.run --standalone --nproc_per_node=2 evaluation/r66_nccl_probe.py
  local rc=$?
  if [[ $rc -eq 124 ]]; then
    echo "[nccl_probe] ${name}: HUNG (killed after 180 s)"
  else
    echo "[nccl_probe] ${name}: exit ${rc}"
  fi
}

run_variant default NCCL_P2P_DISABLE=0
run_variant p2p_disabled NCCL_P2P_DISABLE=1
