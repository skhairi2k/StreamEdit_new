"""R49 stage 2: inverse-response per-token controls from an attention-supported, max-combined divergence.

Inputs: the attention npz of both drafts and the raw DINO / LPIPS divergence npz (R49's own roots), and R48's support S
(``r48_field.compute_E``: selectivity-weighted Eq. 7, src|trg union, exp(-0.5 * dist) decay). Only the demand and the
mapping are new:

  div = max(clip(D_dino, 0, 0.55) / 0.55, clip(D_lpips, 0, 1.0) / 1.0),     q = S * div  in [0, 1]

Each host's control is the inverse of its exposure response U_H(c) = (Q(c) - Q(c_min)) / (Q(c_max) - Q(c_min)),
Q(c) = mean_i a_H(c, i), with a_H the per-step target weight read from the host's code:

  StreamEdit  a(rho, i) = 1 - t_i^rho (Q/K blend), rho in [2, 50] on the real 15-step t_next grid
              -> B(rho) = (1 - q) B(2) + q B(50), B(rho) = mean_i t_i^rho  == r31_rho_map.budget_linear_tau(q, t_next, 2, 50)
  Wan-Edit    a(n, i) = 1[i >= 50 - n] (velocity gate)  -> U = n / 50 exactly, n = round(50 q) (half to even),
              start s = 50 - n clipped to s >= s_min; s = 50 = never edited.  Two floors: s_min 9 and 5.

Outputs {out_root}/se_max/edit{T}/{video}.npz            rho [F, 1560] (r31_stage3 --rho_dir format) + fields
        {out_root}/wan_max_s{9,5}/edit{T}/{video}.npz/.npy start_step int [F, 30, 52] (.npy = --route_start_map) + fields
Fields stored with every output: d_dino, d_lpips (raw), m_dino, m_lpips, div, S, q, and the mapping constants.
usage: python evaluation/r49_field.py --cases evaluation/r48_cases.json
       python evaluation/r49_field.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_field as R48  # noqa: E402
from r30_b_map import t_next_schedule  # noqa: E402
from r31_rho_map import budget_linear_tau  # noqa: E402

RHO_MIN, RHO_MAX = 2.0, 50.0                 # StreamEdit native range; rho = 2 is the stock setting (q = 0)
SE_STEP, SE_FLOW_SHIFT = R48.SE_STEP, R48.SE_FLOW_SHIFT
WAN_N = R48.WAN_N
S_MINS = (9, 5)                              # Wan-Edit floors: the draft's own start (sigma 0.958) and step 5 (sigma 0.978)
DIV_MAX = {"dino": 0.55, "lpips": 1.0}         # fixed absolute references (R49's own; r48_field's LPIPS reference is 0.75)
D = Path.home() / "Data/dataggen/outputs/five_bench"
OUT_ROOT = D / "r49_field"
ATTN_ROOT = {"se": D / "r49_se_attn", "wan": D / "r49_wan_draft/attn"}      # R49's own drafts
DIV_ROOT = D / "r49_div"


def demand(S: np.ndarray, m_dino: np.ndarray, m_lpips: np.ndarray) -> tuple:
    """(div, q): max of the two clipped/normalised divergences, times the support."""
    div = np.maximum(m_dino, m_lpips)
    return div, np.clip(S * div, 0.0, 1.0)


def se_rho(q: np.ndarray, t_next: np.ndarray) -> np.ndarray:
    """Inverse StreamEdit exposure response over [RHO_MIN, RHO_MAX] (exact, bisection)."""
    return budget_linear_tau(q, t_next, RHO_MIN, RHO_MAX)


def wan_start(q: np.ndarray, s_min: int) -> np.ndarray:
    """Inverse Wan-Edit exposure response: n = round(N q) (numpy: half to even), start = N - n, clipped to >= s_min."""
    n = np.round(WAN_N * np.clip(q, 0.0, 1.0)).astype(np.int64)
    return np.clip(WAN_N - n, s_min, WAN_N).astype(np.int64)


def field_case(host: str, t: int, vid: str, attn_root: Path, div_root: Path, out_root: Path,
               t_next: np.ndarray) -> Dict[str, float]:
    a = np.load(attn_root / f"edit{t}" / f"{vid}.npz")
    n_frames = a["src_tokens"].shape[1]
    comps = {}
    for metric in ("dino", "lpips"):
        d = R48._load_d(div_root / f"{host}_{metric}" / f"edit{t}" / f"{vid}.npz", n_frames)
        S, m, _ = R48.compute_E(a["src_tokens"], a["trg_tokens"], a["src_prompt"], a["trg_prompt"],
                                a["idx_src"], a["idx_trg"], d, DIV_MAX[metric],
                                int(a["n_prompt_src"]), int(a["n_prompt_trg"]))
        comps[metric] = (S, d, m)
    S = comps["dino"][0]
    assert np.array_equal(S, comps["lpips"][0]), "S depends only on attention, not on the metric"
    div, q = demand(S, comps["dino"][2], comps["lpips"][2])
    meta = dict(S=S.astype(np.float32), d_dino=comps["dino"][1].astype(np.float32), d_lpips=comps["lpips"][1].astype(np.float32),
                m_dino=comps["dino"][2].astype(np.float32), m_lpips=comps["lpips"][2].astype(np.float32),
                div=div.astype(np.float32), q=q.astype(np.float32), host=np.str_(host), video_name=np.str_(vid),
                edit_type=np.int32(t), div_max_dino=np.float32(DIV_MAX["dino"]), div_max_lpips=np.float32(DIV_MAX["lpips"]),
                lam=np.float32(R48.LAM), combine=np.str_("max"), mapping=np.str_("inverse_exposure_response"),
                n_latent_frames=np.int32(n_frames), grid_h=np.int32(R48.GRID[0]), grid_w=np.int32(R48.GRID[1]),
                frame_seq_length=np.int32(R48.FRAME_SEQ_LENGTH))
    stats = {"q_obj": float(q[S >= 1 - 1e-9].mean()) if (S >= 1 - 1e-9).any() else float("nan"), "S_cover": float((S > 0.5).mean())}
    if host == "se":
        rho = se_rho(q, t_next).reshape(n_frames, -1).astype(np.float32)
        out = out_root / "se_max" / f"edit{t}" / f"{vid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, rho=rho, rho_min=np.float32(RHO_MIN), rho_max=np.float32(RHO_MAX),
                            step=np.int32(SE_STEP), flow_shift=np.float32(SE_FLOW_SHIFT), arm=np.str_("max"), **meta)
        stats["rho_mean"] = float(rho.mean())
    else:
        for s_min in S_MINS:
            start = wan_start(q, s_min)
            out = out_root / f"wan_max_s{s_min}" / f"edit{t}" / f"{vid}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, start_step=start, s_min=np.int32(s_min), wan_n=np.int32(WAN_N),
                                wan_shift=np.float32(R48.WAN_SHIFT), arm=np.str_(f"max_s{s_min}"), **meta)
            np.save(out.with_suffix(".npy"), start)
            stats[f"start_mean_s{s_min}"] = float(start.mean())
    return stats


def selftest() -> int:
    t_next = t_next_schedule(SE_STEP, SE_FLOW_SHIFT)
    qs = np.linspace(0, 1, 101)
    rho = se_rho(qs, t_next)
    assert abs(rho[0] - RHO_MIN) < 1e-6 and abs(rho[-1] - RHO_MAX) < 1e-6, (rho[0], rho[-1])
    assert np.all(np.diff(rho) > 0), "rho must increase strictly with q"
    # the inverse really inverts the exposure response: U(rho(q)) == q
    B = lambda r: np.mean(np.power(t_next, r))
    U = lambda r: (B(RHO_MIN) - B(r)) / (B(RHO_MIN) - B(RHO_MAX))
    err = max(abs(U(r) - q) for r, q in zip(rho, qs))
    assert err < 1e-8, f"U(rho(q)) != q, max err {err}"
    for s_min in S_MINS:
        s = wan_start(qs, s_min)
        assert s[0] == WAN_N and s[-1] == s_min, (s_min, s[0], s[-1])
        assert np.all(np.diff(s) <= 0), "start step must not increase with q"
        assert s.min() >= s_min
    assert wan_start(np.array([0.5]), 0)[0] == 25 and wan_start(np.array([0.7]), 9)[0] == 15
    S = np.array([1.0, 0.5, 0.0]); md = np.array([0.2, 1.0, 1.0]); ml = np.array([0.9, 0.1, 1.0])
    div, q = demand(S, md, ml)
    assert np.allclose(div, [0.9, 1.0, 1.0]) and np.allclose(q, [0.9, 0.5, 0.0])
    print(f"[r49_field] selftest PASS (rho: q=0 -> {rho[0]:.2f}, q=0.5 -> {se_rho(np.array([0.5]), t_next)[0]:.2f}, "
          f"q=1 -> {rho[-1]:.2f}, max |U(rho(q)) - q| = {err:.1e}; wan start q=0.5 -> 25, floors {S_MINS})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cases")
    ap.add_argument("--hosts", nargs="+", default=["se", "wan"], choices=["se", "wan"])
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--attn_se", default=str(ATTN_ROOT["se"]))
    ap.add_argument("--attn_wan", default=str(ATTN_ROOT["wan"]))
    ap.add_argument("--div_root", default=str(DIV_ROOT))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cases:
        ap.error("--cases is required (or --selftest)")
    t_next = t_next_schedule(SE_STEP, SE_FLOW_SHIFT)
    attn_root = {"se": Path(a.attn_se), "wan": Path(a.attn_wan)}
    cases = [(c["video_name"], int(c["edit_type"])) for c in json.loads(Path(a.cases).read_text())]
    n_fail = 0
    for host in a.hosts:
        agg: List[Dict[str, float]] = []
        for vid, t in cases:
            try:
                agg.append(field_case(host, t, vid, attn_root[host], Path(a.div_root), Path(a.out_root), t_next))
            except Exception as e:                       # noqa: BLE001 -- report per clip, keep going
                n_fail += 1
                print(f"[r49_field] FAILED {host} {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        if agg:
            extra = (f"mean rho {np.mean([s['rho_mean'] for s in agg]):.2f}" if host == "se" else
                     "mean start " + ", ".join(f"s_min {m}: {np.mean([s[f'start_mean_s{m}'] for s in agg]):.1f}" for m in S_MINS))
            print(f"[r49_field] {host}: {len(agg)}/{len(cases)} clips; S>0.5 covers {np.mean([s['S_cover'] for s in agg]):.1%}; "
                  f"mean q on the mask {np.nanmean([s['q_obj'] for s in agg]):.3f}; {extra}", flush=True)
    print(f"[r49_field] failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
