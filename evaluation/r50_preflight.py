"""R50 preflight: does every model the Wan-side R49 jobs use load and run OFFLINE in this environment?

Run on a Jean Zay compute node by slurm_scripts/five_bench/r50_preflight.sh (and runnable anywhere with a GPU). Each check uses the SAME
loader the real job uses (the divergence module's own loaders, the FiVE harness' MetricsCalculator, the harness' NIQE command), so a pass
means the real run will load the same weights the same way:

  env         versions of python / torch / transformers / diffusers / huggingface_hub and the file each is imported from
              (our pinned overlay must win over the module's older copies)
  cuda        a GPU is visible; a bf16 matmul runs
  flash_attn  imports and a tiny flash_attn_func runs on the GPU (Wan-Edit asserts it; the FiVE-Acc Qwen load asks for it)
  dinov3      r31_divergence.load_dinov3 (facebook/dinov3-vitb16-pretrain-lvd1689m) + a forward on a 480x832 frame
  lpips_alex  r30_divergence.load_lpips (AlexNet LPIPS, spatial) + a forward
  clip_d      r33_clip_directional.Clip (openai/clip-vit-large-patch14) text + image embeddings
  harness     evaluation/fivebench MetricsCalculator exactly as evaluate.py builds it: CLIPScore, PSNR, LPIPS(squeeze), MSE, SSIM, the DINO ViT-B/8
              structure distance (torch.hub, cached repo) and the Qwen2.5-VL FiVE-Acc model. The constructor SWALLOWS a failed Qwen load
              (five_acc would silently be NaN), so this check asserts five_acc_calculator is not None and runs a short generation.
  niqe        the harness' own `inference_iqa.py -m niqe` subprocess (pyiqa from IQA_PyTorch_model_path in config.yaml)

Everything runs with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1. Ends with "[r50_preflight] model failures: N" (exit 1 if N > 0).
usage: python evaluation/r50_preflight.py [--device cuda|cpu] [--only a,b] [--skip a,b]
       --device cpu: GPU-only checks (cuda, flash_attn) are skipped and Qwen loads without flash attention (local debugging only)
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIG = HERE / "fivebench" / "config.yaml"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "fivebench"))     # evaluate.py does `from metrics_calculator import ...`

CHECKS = ("cuda", "flash_attn", "dinov3", "lpips_alex", "clip_d", "harness", "niqe")


def origin(mod) -> str:
    return str(Path(getattr(mod, "__file__", "?")).parent)


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------------------------------- checks
def check_cuda(dev: str) -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (no GPU granted, or a CPU-only torch build)")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    b = (a @ a).float().abs().mean().item()
    free, total = torch.cuda.mem_get_info()
    return f"{name} sm_{cap[0]}{cap[1]}, {free / 2**30:.0f}/{total / 2**30:.0f} GiB free, bf16 matmul ok ({b:.2f})"


def check_flash_attn(dev: str) -> str:
    import torch
    import flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, q, q, causal=False)
    torch.cuda.synchronize()
    return f"flash_attn {flash_attn.__version__} from {origin(flash_attn)}; flash_attn_func -> {tuple(out.shape)}"


def check_dinov3(dev: str) -> str:
    import torch
    import r31_divergence as D
    model, n_prefix = D.load_dinov3(D.DINOV3_MODEL_ID, dev)
    x = torch.randn(1, 3, D.PIXEL_H, D.PIXEL_W, device=dev)
    with torch.no_grad():
        h = model(pixel_values=x).last_hidden_state
    want = n_prefix + D.GRID_H * D.GRID_W
    if h.shape[1] != want:
        raise RuntimeError(f"DINOv3 returned {h.shape[1]} tokens for a {D.PIXEL_H}x{D.PIXEL_W} frame, expected {want} ({n_prefix} prefix + {D.GRID_H}x{D.GRID_W})")
    return f"{D.DINOV3_MODEL_ID}: {n_prefix} prefix + {D.GRID_H}x{D.GRID_W} patch tokens"


def check_lpips_alex(dev: str) -> str:
    import torch
    import r30_divergence as D30
    net = D30.load_lpips(dev)
    a = torch.rand(1, 3, 480, 832, device=dev) * 2 - 1
    b = torch.rand(1, 3, 480, 832, device=dev) * 2 - 1
    with torch.no_grad():
        m = net(a, b)
    return f"LPIPS alex spatial map {tuple(m.shape)}"


def check_clip_d(dev: str) -> str:
    from PIL import Image
    import r33_clip_directional as C
    c = C.Clip(device=dev)
    t = c.text("Change the rabbit to a drone.")
    i = c.image(Image.new("RGB", (256, 256), (120, 90, 60)))
    return f"{C.MODEL}: text {tuple(t.shape)}, image {tuple(i.shape)}"


def check_harness(dev: str) -> str:
    import torch
    from omegaconf import OmegaConf
    import metrics_calculator as MC                 # also catches a missing qwen_vl_utils / torchmetrics / einops
    cfg = OmegaConf.load(CONFIG)
    calc = MC.MetricsCalculator(dev, config=cfg)
    notes = []
    if calc.five_acc_calculator is None:
        raise RuntimeError("MetricsCalculator built WITHOUT the Qwen FiVE-Acc model (its load failure is swallowed: five_acc would be NaN). "
                           "Look for 'FiVEAcc_Qwen_VL unavailable' above: flash-attn, the Qwen cache or qwen_vl_utils")
    with torch.no_grad():
        px = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8, device=dev)
        s = float(calc.clip_metric_calculator(px, "a drone in a meadow"))
        x = torch.rand(1, 3, 128, 128, device=dev)
        y = torch.rand(1, 3, 128, 128, device=dev)
        l = float(calc.lpips_metric_calculator(x * 2 - 1, y * 2 - 1))
        p = float(calc.psnr_metric_calculator(x, y))
        ss = float(calc.ssim_metric_calculator(x, y))
        notes.append(f"CLIPScore {s:.1f}, LPIPS(squeeze) {l:.3f}, PSNR {p:.1f}, SSIM {ss:.3f}")
    q = calc.five_acc_calculator
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Answer with one word, yes or no: is water wet?"}]}]
    text = q.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = q.processor(text=[text], return_tensors="pt").to(q.model.device)
    with torch.no_grad():
        out = q.model.generate(**inputs, max_new_tokens=4, do_sample=False)
    ans = q.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    impl = getattr(q.model.config, "_attn_implementation", "?")
    notes.append(f"Qwen {cfg.five_acc_vlm_model_id} ({impl}) answered {ans!r}")
    notes.append("motion fidelity: " + ("loaded" if calc.motion_fidelity_score_calculator is not None else "unavailable (optional, unused by R49)"))
    del calc, q
    return "; ".join(notes)


def check_niqe(dev: str) -> str:
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    cfg = OmegaConf.load(CONFIG)
    iqa = Path(str(cfg.IQA_PyTorch_model_path))
    if not (iqa / "inference_iqa.py").is_file():
        raise FileNotFoundError(f"{iqa}/inference_iqa.py not found (IQA_PyTorch_model_path in {CONFIG.name}; run jz/r50_port.sh)")
    with tempfile.TemporaryDirectory(prefix="r50_niqe_") as tmp:
        img = Path(tmp) / "x.png"
        rng = np.random.RandomState(0)
        base = np.kron(rng.randint(0, 255, (16, 16, 3), dtype=np.uint8), np.ones((16, 16, 1), dtype=np.uint8))
        Image.fromarray(base).save(img)
        save = Path(tmp) / "niqe.txt"
        cmd = [sys.executable, str(iqa / "inference_iqa.py"), "-m", "niqe", "-t", str(img), "--save_file", str(save)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0 or not save.is_file() or save.stat().st_size == 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            raise RuntimeError(f"inference_iqa.py exit {r.returncode}: {' | '.join(tail)}")
        return f"NIQE via {iqa.name}/inference_iqa.py -> {save.read_text().strip()[:60]!r}"


FUNCS: Dict[str, Callable[[str], str]] = {"cuda": check_cuda, "flash_attn": check_flash_attn, "dinov3": check_dinov3, "lpips_alex": check_lpips_alex,
                                          "clip_d": check_clip_d, "harness": check_harness, "niqe": check_niqe}
GPU_ONLY = {"cuda", "flash_attn"}


def report_env(dev: str) -> None:
    import torch
    import transformers
    import diffusers
    import huggingface_hub
    import site
    print(f"[r50_preflight] host={os.uname().nodename} python={sys.version.split()[0]} cwd={os.getcwd()}")
    for m in (torch, transformers, diffusers, huggingface_hub):
        print(f"[r50_preflight]   {m.__name__:<16} {m.__version__:<14} <- {origin(m)}")
    print(f"[r50_preflight]   torch cuda build {torch.version.cuda}, device request '{dev}', cuda available {torch.cuda.is_available()}")
    print(f"[r50_preflight]   user site {site.getusersitepackages()}  PYTHONUSERBASE={os.environ.get('PYTHONUSERBASE')}")
    print(f"[r50_preflight]   HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')} "
          f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        print("[r50_preflight]   network: huggingface.co REACHABLE (the offline flags are what keep this run honest)")
    except OSError:
        print("[r50_preflight]   network: huggingface.co unreachable (a compute node, as expected)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--only", default="", help="comma-separated check names")
    ap.add_argument("--skip", default="", help="comma-separated check names")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x]
    skip = {x for x in a.skip.split(",") if x}
    bad = [x for x in only + sorted(skip) if x not in FUNCS]
    if bad:
        ap.error(f"unknown check(s) {bad}; choose from {', '.join(CHECKS)}")
    report_env(a.device)
    results: List[Tuple[str, str, float, str]] = []
    for name in (only or CHECKS):
        if name in skip:
            results.append((name, "SKIP", 0.0, "--skip"))
            continue
        if a.device == "cpu" and name in GPU_ONLY:
            results.append((name, "SKIP", 0.0, "--device cpu"))
            continue
        t0 = time.time()
        try:
            detail = FUNCS[name](a.device)
            results.append((name, "OK", time.time() - t0, detail))
        except Exception as e:                      # noqa: BLE001 - report every failure, keep going
            traceback.print_exc()
            results.append((name, "FAIL", time.time() - t0, f"{type(e).__name__}: {e}"))
        free_gpu()
        print(f"[r50_preflight] {results[-1][1]:<4} {name:<11} {results[-1][2]:6.1f} s  {results[-1][3][:300]}", flush=True)
    print("\n[r50_preflight] summary")
    for name, st, dt, detail in results:
        print(f"[r50_preflight]   {st:<4} {name:<11} {dt:6.1f} s  {detail[:200]}")
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    print(f"[r50_preflight] model failures: {n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
