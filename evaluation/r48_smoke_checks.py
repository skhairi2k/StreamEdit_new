"""R48 smoke gates -- the checks that need Python (r48_smoke.sh sequences the GPU stages around them).

  g1    StreamEdit hooked draft (attn recorder ON) == R38's stored n5_tgt09 latents, bit for bit, per case.
  g2g3  Wan-Edit, in process: G2 hooked 17-step draft == unhooked; G3 --route_start_map all-9 == stock skip 9
        (50 steps). Tensors compared, not decoded mp4s.
  g4mk  write a constant rho = 2 field with r48_field's real StreamEdit writer (for r31_stage3.py --rho_dir).
  g4    that stage-3 render == the global rho = 2 run (run_fivebench.py --blend_power 2), frame for frame.
  g5    seam / freeze check of the dino_SD renders at rho_min 0.3 (the plan's map) and 0 (never releases),
        against the global run: far-background temporal change relative to the source's own, and the
        biggest frame-to-frame jump. rho_min 0.3 must stay within a band of the global run; the rho_min 0
        numbers are logged as the record of why 0 was not used.
Every gate prints ``GATE Gk PASS|FAIL`` and the script exits non-zero on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
D = Path.home() / "Data/dataggen/outputs/five_bench"
FIVE = Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
CASES = [("0001_bus", 1), ("0011_lucia", 5)]
fails = 0


def gate(name: str, ok: bool, msg: str = "") -> None:
    global fails
    fails += int(not ok)
    print(f"GATE {name} {'PASS' if ok else 'FAIL'}  {msg}", flush=True)


def load_pngs(d: Path) -> np.ndarray:
    fs = sorted(d.glob("*.png"))
    return np.stack([cv2.imread(str(f)) for f in fs])


# ----------------------------------------------------------------------------------------------------- G1
def g1(smoke: Path) -> None:
    for vid, t in CASES:
        r = np.load(D / "r38_latents/n5_tgt09" / f"edit{t}" / f"{vid}.npz")["z_trg"]
        h = np.load(smoke / "lat" / f"edit{t}" / f"{vid}.npz")["z_trg"]
        a = np.load(smoke / "attn" / f"edit{t}" / f"{vid}.npz")
        ok = np.array_equal(r, h) and np.isfinite(a["src_tokens"]).all() and a["src_tokens"].shape[1] == r.shape[0]
        gate("G1", ok, f"{vid}: hooked == R38 stored latents {np.array_equal(r, h)}, attention frames "
                       f"{a['src_tokens'].shape[1]} == latent frames {r.shape[0]}")


# ------------------------------------------------------------------------------------------------- G2 / G3
def g2g3(smoke: Path) -> None:
    import torch
    import r48_wan_common as C
    import r48_wan_draft as WD
    cli = C.load_cli()
    se = C.load_se_utils()
    pipe, d = C.build_pipeline(cli)
    tk = pipe.text_encoder.tokenizer.tokenizer
    cases = {(c["video_name"], c["edit_type"]): c for c in C.load_cases(str(HERE / "r48_cases.json"))}
    for vid, t in CASES:
        c = cases[(vid, t)]
        video = cli.load_frames(str(C.FIVE / "videos" / f"{vid}.mp4"))
        base = dict(size=cli.SIZE_CONFIGS[C.SIZE], frame_num=41, shift=5.0, sample_solver=d.sample_solver,
                    guide_scale=d.sample_guide_scale, tgt_guide_scale=d.tgt_guide_scale, n_avg=1,
                    seed_noise=True, seed=0, offload_model=False)
        # G2: hooks are read-only
        kw = dict(base, sampling_steps=WD.SAMPLE_STEPS, skip_timesteps=WD.SKIP)
        a = pipe.edit(video, c["source_prompt"], c["target_prompt"], **kw)
        i_s = se.find_phrase_token_indices(tk, c["source_prompt"], c["source_object"])[0]
        i_t = se.find_phrase_token_indices(tk, c["target_prompt"], c["target_object"])[0]
        n_s = len(tk(c["source_prompt"], add_special_tokens=True)["input_ids"])
        n_t = len(tk(c["target_prompt"], add_special_tokens=True)["input_ids"])
        rec = WD.Recorder(pipe.model, i_s, i_t, n_s, n_t, C.n_latent_frames(), WD.window_steps())
        b = pipe.edit(video, c["source_prompt"], c["target_prompt"], **kw)
        rec.close()
        gate("G2", torch.equal(a, b) and rec.calls == 4 * (WD.SAMPLE_STEPS - WD.SKIP),
             f"{vid}: hooked == unhooked draft {torch.equal(a, b)}, calls {rec.calls}")
        # G3: an all-9 start map is stock skip 9
        kw = dict(base, sampling_steps=50, skip_timesteps=9)
        s9 = pipe.edit(video, c["source_prompt"], c["target_prompt"], **kw)
        mp = smoke / f"map9_{vid}.npy"
        np.save(mp, np.full((C.n_latent_frames(), *C.GRID), 9, dtype=np.int64))
        m9 = pipe.edit(video, c["source_prompt"], c["target_prompt"],
                       route_start_map=cli.load_route_start_map(str(mp), 0, 9), **kw)
        gate("G3", torch.equal(s9, m9), f"{vid}: all-9 start map == skip 9, max|diff| {(s9 - m9).abs().max().item():.3g}")
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------- G4
def g4mk(smoke: Path) -> None:
    import r48_field as R
    for vid, t in CASES:
        n_lat = np.load(smoke / "lat" / f"edit{t}" / f"{vid}.npz")["z_trg"].shape[0]
        e2 = np.full((n_lat, *R.GRID), (2.0 - R.RHO_MIN) / (R.RHO_MAX - R.RHO_MIN))
        rho = R.write_se_npz(smoke / "g4_rho" / f"edit{t}" / f"{vid}.npz", e2)
        assert np.allclose(rho, 2.0), rho.min()
    print("[g4mk] constant rho=2 fields written with r48_field.write_se_npz")


def g4(smoke: Path) -> None:
    for vid, t in CASES:
        a = load_pngs(smoke / "g4_arm" / "r48_g4_const2" / "step14" / f"edit{t}" / vid)
        b = load_pngs(smoke / "g4_global" / "r48_g4_global2" / f"edit{t}" / vid)
        same = a.shape == b.shape and np.array_equal(a, b)
        gate("G4", same, f"{vid}: const rho=2 field render == global rho=2 run {same} "
                         f"(shapes {a.shape} vs {b.shape}, max|diff| "
                         f"{np.abs(a.astype(int) - b.astype(int)).max() if a.shape == b.shape else 'n/a'})")


# ----------------------------------------------------------------------------------------------------- G5
def _metrics(frames: np.ndarray, src: np.ndarray, far: np.ndarray) -> dict:
    """far: bool [T, H, W] far-background mask (S < 0.08 at the frame's latent index)."""
    f = frames.astype(np.float32)
    s = src.astype(np.float32)
    dt = np.abs(f[1:] - f[:-1]).mean(-1)
    ds = np.abs(s[1:] - s[:-1]).mean(-1)
    m = far[1:] & far[:-1]
    per = np.array([dt[k][m[k]].mean() if m[k].any() else np.nan for k in range(len(dt))])
    per_s = np.array([ds[k][m[k]].mean() if m[k].any() else np.nan for k in range(len(dt))])
    return dict(bg_tchange_vs_src=float(np.nanmean(per) / max(np.nanmean(per_s), 1e-6)),
                max_jump_ratio=float(np.nanmax(per) / max(np.nanmedian(per), 1e-6)))


def g5(smoke: Path) -> None:
    for vid, t in CASES:
        S = np.load(smoke / "field03" / "se_dino_SD" / f"edit{t}" / f"{vid}.npz")["S"]
        T = 4 * (S.shape[0] - 1) + 1
        src = np.stack([cv2.resize(fr, (832, 480)) for fr in _read(FIVE / "videos" / f"{vid}.mp4", T)])
        lat_of = np.array([0] + [(k - 1) // 4 + 1 for k in range(1, T)])
        far = np.stack([cv2.resize((S[lat_of[k]] < 0.08).astype(np.uint8), (832, 480),
                                   interpolation=cv2.INTER_NEAREST).astype(bool) for k in range(T)])
        runs = {"global": smoke / "g4_global" / "r48_g4_global2" / f"edit{t}" / vid,
                "rho_min=0.3": smoke / "g5_arm03" / "r48_g5_rmin03" / "step14" / f"edit{t}" / vid,
                "rho_min=0": smoke / "g5_arm0" / "r48_g5_rmin0" / "step14" / f"edit{t}" / vid}
        res = {k: _metrics(load_pngs(v), src, far) for k, v in runs.items()}
        for k, v in res.items():
            print(f"  {vid} {k:12s} far-bg temporal change / source's: {v['bg_tchange_vs_src']:.3f}   "
                  f"max/median frame jump: {v['max_jump_ratio']:.2f}", flush=True)
        g, r = res["global"], res["rho_min=0.3"]
        ok = (0.7 * g["bg_tchange_vs_src"] <= r["bg_tchange_vs_src"] <= 1.5 * g["bg_tchange_vs_src"]
              and r["max_jump_ratio"] <= 1.5 * g["max_jump_ratio"])
        gate("G5", ok, f"{vid}: rho_min=0.3 render within [0.7, 1.5]x of the global run's far-background motion, "
                       f"jump <= 1.5x")


def _read(p: Path, n: int):
    cap, out = cv2.VideoCapture(str(p)), []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=("g1", "g2g3", "g4mk", "g4", "g5"))
    ap.add_argument("--smoke_root", required=True)
    a = ap.parse_args()
    smoke = Path(a.smoke_root).expanduser()
    {"g1": g1, "g2g3": g2g3, "g4mk": g4mk, "g4": g4, "g5": g5}[a.gate](smoke)
    return int(fails > 0)


if __name__ == "__main__":
    sys.exit(main())
