"""R66 NCCL probe: can 2 GPUs of one L40S node talk over NCCL? (launched by slurm_scripts/five_bench/r66_nccl_probe.sh)

Runs under torchrun --nproc_per_node=2: process group init (nccl), a 1-element all-reduce (the collective the smoke job 1080618
hung on), then a 256 MB all-reduce with its bandwidth; rank 0 prints 'NCCL PROBE PASS'. A hang is caught by the caller's timeout.
"""
from __future__ import annotations

import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=2))
    small = torch.ones(1, device="cuda")
    dist.all_reduce(small)
    torch.cuda.synchronize()
    big = torch.ones(64 * 2**20, device="cuda")          # 64 Mi float32 = 256 MB
    dist.all_reduce(big)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    sec = (time.time() - t0) / 5
    if rank == 0:
        ok = small.item() == 2.0 and big[0].item() == 2.0 ** 6     # 1 + 5 all-reduces of ones over 2 ranks
        print(f"[nccl_probe] small all-reduce {small.item():.0f} (expect 2); 256 MB value {big[0].item():.0f} (expect 64); "
              f"256 MB all-reduce {sec * 1e3:.0f} ms "
              f"(~{0.25 / sec:.1f} GB/s algbw); P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'unset')}")
        print("NCCL PROBE PASS" if ok else "NCCL PROBE WRONG RESULT")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
