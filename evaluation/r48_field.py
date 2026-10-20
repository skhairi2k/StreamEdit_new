"""R48 stage 2: the host-agnostic edit-demand field E = S * m, and its two thin host writers.

``compute_E`` is ONE pure function on plain arrays. It has no host argument and no host branch; whichever host's
attention npz is fed in, the same bytes come out (smoke gate G6 feeds identical synthetic data down both writer
paths and asserts bit-identity). The ONLY host-specific code is the two writers at the bottom, each a few lines:

  StreamEdit : rho = RHO_MIN + (RHO_MAX - RHO_MIN) * E                      -> r31_stage3.py --rho_dir npz
  Wan-Edit   : start sigma = SIGMA_FLOOR + SIGMA_SCALE * E, nearest step of the
               50-step shift-5 grid (clipped to [9, 49])                     -> int map for --route_start_map

Support S (per branch, window-averaged attention A [nT, F, 30, 52] from stage 1):
  w_p   = 1 - H(A_p / sum A_p) / log N                  selectivity of token p over the whole clip (N = F*30*52)
  A_T   = sum_p w_p A_p / sum_p w_p                      selectivity-weighted trigger map
  M     = A_T > (prompt_mass - sum_p A_p) / (n_prompt - nT)      StreamEdit Eq. 7: trigger vs mean of other tokens
  M     = M_src | M_trg                                  union of the two branches
  S     = 1 on M, exp(-LAM * d) outside (d = Euclidean distance in tokens, per frame), 0 on frames with empty M
Divergence: m = clip(d, 0, div_max) / div_max.  Field: E = S * m.

Arms (both hosts): dino_SD = S*m_dino, lpips_SD = S*m_lpips, dino_D = m_dino (no support), S = S (no divergence).

Inputs  (per host, per clip): attention npz (stage 1) and divergence npz (r31_divergence.py, key ``d`` [F, 1560]).
Outputs {out_root}/{host}_{arm}/edit{T}/{video}.npz  (+ .npy start map for wan): S, d, m, E and the mapping metadata.
usage: python evaluation/r48_field.py --mode subset --cases evaluation/r48_cases.json
       python evaluation/r48_field.py --selftest
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

GRID = (30, 52)
FRAME_SEQ_LENGTH = GRID[0] * GRID[1]
LAM = 0.5                                   # soft-support decay per token
DIV_MAX = {"dino": 0.55, "lpips": 0.75}     # DINO ~ 91st pct of in-object d on the StreamEdit draft; LPIPS percentile-matched
# host mappings (Decisions: "Mapping, StreamEdit" / "Mapping, Wan-Edit" / "Floor matching rule")
RHO_MIN, RHO_MAX = 0.3, 50.0
SE_STEP, SE_FLOW_SHIFT = 15, 1.0            # the 15-step render the rho field is inverted for (as r38_rho)
SIGMA_FLOOR, SIGMA_SCALE = 0.1, 0.86        # E=0 -> sigma 0.1, E=1 -> sigma 0.96
WAN_N, WAN_SHIFT = 50, 5.0
WAN_MIN_STEP, WAN_MAX_STEP = 9, 49
ARMS = ("dino_SD", "lpips_SD", "dino_D", "S")
OUT_ROOT = Path.home() / "Data/dataggen/outputs/five_bench/r48_field"
ATTN_ROOT = {"se": Path.home() / "Data/dataggen/outputs/five_bench/r48_se_attn",
             "wan": Path.home() / "Data/dataggen/outputs/five_bench/r48_wan_draft/attn"}
DIV_ROOT = Path.home() / "Data/dataggen/outputs/five_bench/r48_div"


# ---------------------------------------------------------------------------------------------- core (host-free)

def selectivity(a: np.ndarray) -> float:
    """1 - normalised entropy of a non-negative map: 0 for a uniform map, 1 for a single spot."""
    s = float(a.sum())
    if s <= 0:
        return 0.0
    p = a.ravel() / s
    return 1.0 - float(-(p * np.log(p + 1e-30)).sum() / np.log(p.size))


def _branch_mask(attn: np.ndarray, prompt: np.ndarray, n_prompt: int) -> np.ndarray:
    """Eq. 7 on the selectivity-weighted trigger map -> bool [F, 30, 52]."""
    n_t = attn.shape[0]
    if n_prompt <= n_t:
        raise ValueError(f"n_prompt {n_prompt} <= number of trigger tokens {n_t}")
    w = np.array([selectivity(a) for a in attn])
    a_t = np.tensordot(w, attn, 1) / w.sum() if w.sum() > 0 else attn.mean(0)
    other = (prompt - attn.sum(0)) / (n_prompt - n_t)
    return a_t > other


def soft_support(mask: np.ndarray, lam: float = LAM) -> np.ndarray:
    """bool [F, 30, 52] -> float [F, 30, 52]: 1 on the mask, exp(-lam*dist) outside, 0 on empty frames."""
    out = np.zeros(mask.shape, np.float64)
    for f in range(mask.shape[0]):
        if mask[f].any():
            out[f] = np.exp(-lam * distance_transform_edt(~mask[f]))
    return out


def compute_E(attn_src: np.ndarray, attn_trg: np.ndarray, prompt_src: np.ndarray, prompt_trg: np.ndarray,
              idx_src, idx_trg, d: np.ndarray, div_max: float,
              n_prompt_src: int, n_prompt_trg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, m, E), each float64 [F, 30, 52]. ``d`` is raw divergence [F, 30, 52]. idx_* are only checked
    against the map count (the maps are already one per trigger token)."""
    if attn_src.shape[0] != len(idx_src) or attn_trg.shape[0] != len(idx_trg):
        raise ValueError("attention map count != trigger-token count")
    if not (attn_src.shape[1:] == attn_trg.shape[1:] == prompt_src.shape == prompt_trg.shape == d.shape):
        raise ValueError(f"shape mismatch: {attn_src.shape} {attn_trg.shape} {prompt_src.shape} "
                         f"{prompt_trg.shape} {d.shape}")
    if not np.isfinite(d).all():
        raise ValueError("divergence holds a non-finite value")
    mask = _branch_mask(attn_src.astype(np.float64), prompt_src.astype(np.float64), int(n_prompt_src)) | \
        _branch_mask(attn_trg.astype(np.float64), prompt_trg.astype(np.float64), int(n_prompt_trg))
    S = soft_support(mask)
    m = np.clip(d.astype(np.float64), 0.0, div_max) / div_max
    return S, m, S * m


def arm_field(arm: str, S: np.ndarray, m_dino: np.ndarray, m_lpips: np.ndarray) -> np.ndarray:
    return {"dino_SD": S * m_dino, "lpips_SD": S * m_lpips, "dino_D": m_dino, "S": S}[arm]


# ------------------------------------------------------------------------------------------ host writers (thin)

def wan_sigma_grid() -> np.ndarray:
    t = (WAN_N - np.arange(WAN_N)) / WAN_N
    return WAN_SHIFT * t / (1 + (WAN_SHIFT - 1) * t)


def se_rho(E: np.ndarray) -> np.ndarray:
    return (RHO_MIN + (RHO_MAX - RHO_MIN) * E).reshape(E.shape[0], -1).astype(np.float32)


def write_se_npz(out: Path, field: np.ndarray, **meta) -> np.ndarray:
    """The StreamEdit writer: E -> rho npz in the format r31_stage3.py --rho_dir reads (plus metadata).
    Shared by field_case and the smoke gate G4, so the gate exercises the real writer."""
    rho = se_rho(field)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rho=rho, rho_min=np.float32(RHO_MIN), rho_max=np.float32(RHO_MAX),
                        step=np.int32(SE_STEP), flow_shift=np.float32(SE_FLOW_SHIFT), **meta)
    return rho


def wan_start_step(E: np.ndarray) -> np.ndarray:
    sig = SIGMA_FLOOR + SIGMA_SCALE * E
    step = np.abs(sig[..., None] - wan_sigma_grid()).argmin(-1)
    return np.clip(step, WAN_MIN_STEP, WAN_MAX_STEP).astype(np.int64)


# ---------------------------------------------------------------------------------------------------- per clip

def _load_d(path: Path, n_frames: int) -> np.ndarray:
    z = np.load(path)
    d = z["d"]
    if d.shape != (n_frames, FRAME_SEQ_LENGTH):
        raise ValueError(f"{path}: d {d.shape} != ({n_frames}, {FRAME_SEQ_LENGTH})")
    return d.reshape(n_frames, *GRID)


def field_case(host: str, t: int, vid: str, attn_root: Path, div_root: Path, out_root: Path) -> Dict[str, float]:
    a = np.load(attn_root / f"edit{t}" / f"{vid}.npz")
    n_frames = a["src_tokens"].shape[1]
    stats: Dict[str, float] = {}
    comps = {}
    for metric in ("dino", "lpips"):
        d = _load_d(div_root / f"{host}_{metric}" / f"edit{t}" / f"{vid}.npz", n_frames)
        S, m, E = compute_E(a["src_tokens"], a["trg_tokens"], a["src_prompt"], a["trg_prompt"],
                            a["idx_src"], a["idx_trg"], d, DIV_MAX[metric],
                            int(a["n_prompt_src"]), int(a["n_prompt_trg"]))
        comps[metric] = (S, d, m, E)
    S = comps["dino"][0]
    assert np.array_equal(S, comps["lpips"][0]), "S depends only on attention, not on the metric"
    m_d, m_l = comps["dino"][2], comps["lpips"][2]
    for arm in ARMS:
        E = arm_field(arm, S, m_d, m_l)
        metric = "lpips" if arm.startswith("lpips") else "dino"
        meta = dict(S=S.astype(np.float32), d=comps[metric][1].astype(np.float32), m=comps[metric][2].astype(np.float32),
                    E=E.astype(np.float32), arm=np.str_(arm), host=np.str_(host), video_name=np.str_(vid),
                    edit_type=np.int32(t), div_max_dino=np.float32(DIV_MAX["dino"]),
                    div_max_lpips=np.float32(DIV_MAX["lpips"]), lam=np.float32(LAM),
                    mapping=np.str_("linear"), n_latent_frames=np.int32(n_frames),
                    grid_h=np.int32(GRID[0]), grid_w=np.int32(GRID[1]), frame_seq_length=np.int32(FRAME_SEQ_LENGTH))
        out = out_root / f"{host}_{arm}" / f"edit{t}" / f"{vid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        if host == "se":
            rho = write_se_npz(out, E, **meta)
            stats[f"{arm}_rho_mean"] = float(rho.mean())
        else:
            start = wan_start_step(E)
            np.savez_compressed(out, start_step=start, sigma_floor=np.float32(SIGMA_FLOOR),
                                sigma_scale=np.float32(SIGMA_SCALE), wan_n=np.int32(WAN_N),
                                wan_shift=np.float32(WAN_SHIFT), **meta)
            np.save(out.with_suffix(".npy"), start)
            stats[f"{arm}_start_mean"] = float(start.mean())
    stats["S_cover"] = float((S > 0.5).mean())
    stats["m_dino_outside_S"] = float(((1 - (S > 0.5)) * m_d).sum() / max(m_d.sum(), 1e-12))
    return stats


# --------------------------------------------------------------------------------------------------- self test

def selftest() -> int:
    """G6: the same synthetic (attention, divergence) data written down the se and the wan path must give
    bit-identical S, m, E. Also checks the writers' endpoints (E=0/1) and the Eq. 7 sanity of S."""
    rng = np.random.default_rng(0)
    F = 5
    attn = np.abs(rng.normal(0, 1e-3, (2, F, *GRID))) + 1e-4
    attn[:, :, 10:18, 20:32] += 0.05                       # a bright blob every token attends to
    prompt = attn.sum(0) + 0.1
    d = np.abs(rng.normal(0.2, 0.1, (F, *GRID)))
    n_prompt = 12
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for host in ("se", "wan"):
            (tmp / "attn" / host / "edit1").mkdir(parents=True)
            np.savez(tmp / "attn" / host / "edit1" / "v.npz", src_tokens=attn, trg_tokens=attn,
                     src_prompt=prompt, trg_prompt=prompt, idx_src=np.array([0, 1]), idx_trg=np.array([0, 1]),
                     n_prompt_src=n_prompt, n_prompt_trg=n_prompt)
            for metric in ("dino", "lpips"):
                p = tmp / "div" / f"{host}_{metric}" / "edit1"
                p.mkdir(parents=True)
                np.savez(p / "v.npz", d=d.reshape(F, -1))
            field_case(host, 1, "v", tmp / "attn" / host, tmp / "div", tmp / "out")
        ok = True
        for arm in ARMS:
            s = np.load(tmp / "out" / f"se_{arm}" / "edit1" / "v.npz")
            w = np.load(tmp / "out" / f"wan_{arm}" / "edit1" / "v.npz")
            same = all(np.array_equal(s[k], w[k]) for k in ("S", "m", "E", "d"))
            print(f"[selftest] G6 {arm}: S/m/E/d bit-identical across hosts: {same}")
            ok &= same
        S = np.load(tmp / "out" / "se_S" / "edit1" / "v.npz")["S"]
        inside, outside = S[:, 10:18, 20:32].mean(), np.delete(S.reshape(F, -1), 0, axis=1).mean()
        print(f"[selftest] S inside blob {inside:.2f} (expect ~1) vs overall {outside:.2f}")
        ok &= inside > 0.9
    ok &= bool(np.allclose(se_rho(np.array([0.0, 1.0])).ravel(), [RHO_MIN, RHO_MAX]))
    st = wan_start_step(np.array([0.0, 1.0]))
    print(f"[selftest] wan start step at E=0/1: {st.tolist()} (expect [49, 9])")
    ok &= st.tolist() == [49, 9]
    print("[selftest] GATE G6", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------- main

def main() -> int:
    global RHO_MIN, ARMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=("subset", "full"))
    ap.add_argument("--cases")
    ap.add_argument("--hosts", nargs="+", default=["se", "wan"], choices=["se", "wan"])
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--attn_se", default=str(ATTN_ROOT["se"]))
    ap.add_argument("--attn_wan", default=str(ATTN_ROOT["wan"]))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--rho_min", type=float, default=RHO_MIN,
                    help="StreamEdit floor (smoke G5 compares 0.3 with 0, the degenerate never-release setting)")
    ap.add_argument("--div_root", default=str(DIV_ROOT))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.mode and a.cases):
        ap.error("--mode and --cases are required (or --selftest)")
    import json
    RHO_MIN, ARMS = a.rho_min, tuple(a.arms)
    attn_root = {"se": Path(a.attn_se), "wan": Path(a.attn_wan)}
    cases = [(c["video_name"], int(c["edit_type"])) for c in json.loads(Path(a.cases).read_text())]
    n_fail = 0
    for host in a.hosts:
        agg: List[Dict[str, float]] = []
        for vid, t in cases:
            try:
                agg.append(field_case(host, t, vid, attn_root[host], Path(a.div_root), Path(a.out_root)))
            except Exception as e:                       # noqa: BLE001 -- report per clip, keep going
                n_fail += 1
                print(f"[r48_field] FAILED {host} {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        if agg:
            print(f"[r48_field] {host}: {len(agg)}/{len(cases)} clips; S>0.5 covers {np.mean([s['S_cover'] for s in agg]):.1%} "
                  f"of tokens; share of DINO m outside S {np.mean([s['m_dino_outside_S'] for s in agg]):.1%}", flush=True)
    print(f"[r48_field] failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
