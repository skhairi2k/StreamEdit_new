"""R45 smoke, LongLive side: gates G1-G5 (G0 / G6 run in r45_smoke.sh through run_fivebench.py).

Two subcommands, run in SEPARATE processes (both builds import as top-level ``pipeline`` /
``utils`` / ``wan``, so only one can be live per process):

  run   --root <LongLive build> --tag stock|ported --out <dir>
        Loads the build through evaluation/backbone.py and renders latents with VP (tripled
        first frame, LL-native fg_boost 2, flow_shift 1.0, seed 0 re-set before every call).
        ``stock``  : routing off on the two G1 clips, calling ``inference`` with STOCK kwargs only.
        ``ported`` : the same two calls (G1), plus
                     G2  uniform rho == 2 field on 0021_landing          (vs the routing-off call)
                     G3  blend_sched tgt0.9 and first2 at N=15 on 0021_landing
                     G4  longest FiVE clip at N=30: completes, peak VRAM, s/frame
                     G5  generator-forward counts (batch 2 vs batch 1) on 0001_bus for a draft at
                         N=5 and N=15, the N=15 render, and single passes at N=20 and N=30
  gates --out <dir>
        Compares the saved tensors and prints ``GATE<k> PASS|FAIL`` lines; exit 1 on any FAIL.

G1 clips (0021_landing/edit1, 0025_baseball/edit2) are ones where stock (case-sensitive) and
ported (case-insensitive) trigger matching return the same non-empty tokens, so G1 isolates
the routing port. Step counts other than the loaded 15 are set by rebuilding
``pipeline.denoising_step_list`` with the formula of ``EditCausalInferencePipeline.__init__``
(checked against the loaded N=15 grid first).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backbone as bb  # noqa: E402

DATA_ROOT = Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark").expanduser()
ANCHOR_ROOT = Path("~/Data/dataggen/outputs/five_bench/anchors").expanduser()
G1_CLIPS = [("0021_landing", 1), ("0025_baseball", 2)]
G2G3_CLIP = ("0021_landing", 1)
G5_CLIP = ("0001_bus", 1)
SEED = 0
FG_BOOST = 2.0          # LL-native (paper: omega = 2 outside Ours(SF))
FLOW_SHIFT = 1.0


def entry(video_name: str, edit_type: int) -> dict:
    rows = json.loads((DATA_ROOT / "edit_prompt" / f"edit{edit_type}_FiVE.json").read_text())
    hits = [e for e in rows if e["video_name"] == video_name]
    if len(hits) != 1:
        raise SystemExit(f"{video_name} appears {len(hits)}x in edit{edit_type}_FiVE.json")
    return hits[0]


def set_steps(pipeline, n: int) -> None:
    """Rebuild the warped grid exactly as EditCausalInferencePipeline.__init__ does."""
    lst = torch.tensor(np.arange(1000, 0, -1000 / n).astype(int).tolist(), dtype=torch.long)
    timesteps = torch.cat((pipeline.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    pipeline.denoising_step_list = timesteps[1000 - lst]


class Clip:
    """Encoded inputs for one FiVE pair, built like run_fivebench.py (--src_frames video)."""

    def __init__(self, h, pipeline, device, video_name, edit_type):
        from diffusers.utils import load_video
        from PIL import Image
        from torchvision import transforms
        self.e = entry(video_name, edit_type)
        self.name = video_name
        tf = transforms.Compose([transforms.Resize((480, 832)), transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])
        src = load_video(str(DATA_ROOT / "videos" / f"{video_name}.mp4"))
        src = src[: h.find_closest_num_frame(len(src))]
        self.n_frames = len(src)
        x = torch.stack([tf(im) for im in src], dim=1).unsqueeze(0)
        self.lat = pipeline.vae.encode_to_latent(
            x.to(device=device, dtype=torch.bfloat16)).to(device=device, dtype=torch.bfloat16)
        pipeline.vae.model.clear_cache()
        anchor = Image.open(ANCHOR_ROOT / f"edit{edit_type}" / f"{video_name}.png").convert("RGB")
        self.src_vp, self.trg_vp, self.flags = bb.encode_vp("ll", pipeline, tf, src[0], anchor, device)
        if not self.flags["triple_first_frame"]:
            raise SystemExit("expected LongLive's tripled VP")


def render(h, pipeline, clip: Clip, low_memory: bool, decode: bool = False, **extra):
    """One LL edit with STOCK kwargs, plus `extra` (new R45 kwargs; empty for the stock tag)."""
    h.set_seed(SEED)
    e = clip.e
    video, latent = pipeline.inference(
        src_video=clip.lat, src_prompts=e["source_prompt"], trg_prompts=e["target_prompt"],
        src_trigger_words=e["source_object"], trg_trigger_words=e["target_object"],
        return_latents=True, wo_video_decode=not decode, profile=False, low_memory=low_memory,
        independent_first_frame=clip.flags["independent_first_frame"],
        triple_first_frame=clip.flags["triple_first_frame"],
        src_initial_latent=clip.src_vp, trg_initial_latent=clip.trg_vp,
        fg_boost_factor=FG_BOOST, blend_power=2.0, **extra)
    pipeline.vae.model.clear_cache()
    return video, latent.cpu()


def cmd_run(a) -> None:
    out = Path(a.out).expanduser().resolve() / a.tag
    out.mkdir(parents=True, exist_ok=True)
    h = bb.activate("ll", Path(a.root))
    args = types.SimpleNamespace(config_path=bb.BACKBONES["ll"]["config_path"], step=15,
                                 flow_shift=FLOW_SHIFT, use_infinite_attention=False, seed=SEED)
    pipeline, low_memory, device = bb.load_backbone("ll", args)
    print(f"[r45_smoke] {a.tag}: root={h.root} low_memory={low_memory}", flush=True)
    grid15 = pipeline.denoising_step_list.clone()
    set_steps(pipeline, 15)
    if not torch.equal(grid15, pipeline.denoising_step_list):
        raise SystemExit("set_steps(15) does not reproduce the loaded grid")

    # G1 (both tags): routing off, stock kwargs only.
    clips = {}
    for name, t in G1_CLIPS:
        clips[name] = Clip(h, pipeline, device, name, t)
        _, lat = render(h, pipeline, clips[name], low_memory)
        torch.save(lat, out / f"g1_{name}.pt")
        print(f"[r45_smoke] {a.tag} G1 {name}: latent {tuple(lat.shape)}", flush=True)
    if a.tag == "stock":
        return

    from pipeline.utils import _schedule_blend_rate
    c = clips[G2G3_CLIP[0]]
    n_lat = c.lat.shape[1]
    # G2: uniform rho == 2 field vs the scalar path (blend_power 2, already saved as G1).
    _, lat = render(h, pipeline, c, low_memory,
                    rho_frames=torch.full((n_lat, pipeline.frame_seq_length), 2.0))
    torch.save(lat, out / "g2_rho2.pt")
    # G3: tgt0.9 vs first2 at N=15, plus the echoed vectors at N=15 and N=5.
    for sched in ("tgt0.9", "first2"):
        _, lat = render(h, pipeline, c, low_memory, blend_sched=sched)
        torch.save(lat, out / f"g3_{sched}.pt")
    echo = {}
    for n in (15, 5):
        set_steps(pipeline, n)
        g = [float(t) / 1000 for t in pipeline.denoising_step_list]
        echo[n] = [int(1.0 - _schedule_blend_rate("tgt0.9", i, n, t_cur=g[i])) for i in range(n)]
    set_steps(pipeline, 15)
    (out / "g3_echo.json").write_text(json.dumps(echo))

    # G4: longest FiVE clip at N=30 (with decode, as a real sweep point renders).
    longest = None
    for t in range(1, 7):
        for e in json.loads((DATA_ROOT / "edit_prompt" / f"edit{t}_FiVE.json").read_text()):
            n = len(list((DATA_ROOT / "images" / e["video_name"]).iterdir()))
            if longest is None or n > longest[0]:
                longest = (n, e["video_name"], t)
    set_steps(pipeline, 30)
    cl = Clip(h, pipeline, device, longest[1], longest[2])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    video, _ = render(h, pipeline, cl, low_memory, decode=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    g4 = dict(clip=longest[1], edit_type=longest[2], src_frames=longest[0],
              out_frames=int(video.shape[1]), seconds=dt, s_per_frame=dt / int(video.shape[1]),
              peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
    (out / "g4.json").write_text(json.dumps(g4))
    print(f"[r45_smoke] G4 {g4}", flush=True)

    # G5: generator-forward counts by batch size.
    counts = {"b1": 0, "b2": 0}

    def _hook(_mod, _args, kwargs):
        b = kwargs["noisy_image_or_video"].shape[0]
        counts["b2" if b == 2 else "b1"] += 1
    handle = pipeline.generator.register_forward_pre_hook(_hook, with_kwargs=True)
    cb = Clip(h, pipeline, device, *G5_CLIP)
    g5 = {}
    for label, n in (("draft5", 5), ("draft15", 15), ("render15", 15), ("single20", 20),
                     ("single30", 30)):
        set_steps(pipeline, n)
        counts.update(b1=0, b2=0)
        render(h, pipeline, cb, low_memory)
        g5[label] = dict(counts)
        print(f"[r45_smoke] G5 {label}: {counts}", flush=True)
    handle.remove()
    set_steps(pipeline, 15)
    (out / "g5.json").write_text(json.dumps(g5))


def cmd_gates(a) -> None:
    out = Path(a.out).expanduser().resolve()
    st, po = out / "stock", out / "ported"
    ok = True

    def gate(k, passed, msg):
        nonlocal ok
        ok &= bool(passed)
        print(f"GATE{k} {'PASS' if passed else 'FAIL'}  {msg}", flush=True)

    for name, _ in G1_CLIPS:
        x, y = torch.load(st / f"g1_{name}.pt"), torch.load(po / f"g1_{name}.pt")
        gate(1, torch.equal(x, y), f"stock LL == ported LL (routing off) on {name} "
                                   f"(max |d| = {(x.float() - y.float()).abs().max().item():.3g})")
    off = torch.load(po / f"g1_{G2G3_CLIP[0]}.pt")
    r2 = torch.load(po / "g2_rho2.pt")
    gate(2, torch.equal(off, r2), f"uniform rho=2 field == scalar blend_power 2 on {G2G3_CLIP[0]}")
    a3, b3 = torch.load(po / "g3_tgt0.9.pt"), torch.load(po / "g3_first2.pt")
    gate(3, torch.equal(a3, b3), f"tgt0.9 == first2 at N=15 on {G2G3_CLIP[0]}")
    echo = {int(k): v for k, v in json.loads((po / "g3_echo.json").read_text()).items()}
    gate(3, echo[15] == [0, 0] + [1] * 13 and echo[5] == [0, 1, 1, 1, 1],
         f"echo N=15 {echo[15]}  N=5 {echo[5]}")
    g4 = json.loads((po / "g4.json").read_text())
    gate(4, g4["out_frames"] > 0, f"longest clip {g4['clip']} ({g4['src_frames']} src frames) at "
                                  f"N=30: {g4['out_frames']} frames, {g4['s_per_frame']:.2f} s/frame, "
                                  f"peak_vram {g4['peak_vram_gb']:.1f} GB")
    g5 = json.loads((po / "g5.json").read_text())
    for n, single in ((5, "single20"), (15, "single30")):
        d, r, s = g5[f"draft{n}"], g5["render15"], g5[single]
        b2_ok = d["b2"] + r["b2"] == s["b2"]
        b1_extra = d["b1"] + r["b1"] - s["b1"]
        gate(5, b2_ok and b1_extra == d["b1"],
             f"draft{n}+render15 vs {single}: dual (batch-2) {d['b2']}+{r['b2']} vs {s['b2']}; "
             f"extra batch-1 = {b1_extra} (draft's own clean-context forwards: {d['b1']})")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--tag", choices=("stock", "ported"), required=True)
    r.add_argument("--out", required=True)
    g = sub.add_parser("gates")
    g.add_argument("--out", required=True)
    a = p.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_gates(a)


if __name__ == "__main__":
    main()
