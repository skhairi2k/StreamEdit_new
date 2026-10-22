"""R51 stage 2: demand-component ablation fields -- only the demand q changes, everything else is R49's full stage.

Inputs are R49's full-stage inputs (attention r48_se_attn / r48_wan_draft/attn, divergence r48_div/{host}_{dino,lpips}), the
support S and the clipped divergences m_dino = clip(d, 0, 0.55)/0.55, m_lpips = clip(d, 0, 1.0)/1.0 exactly as r49_field.py.
Arms (q, clipped to [0, 1]):

  max     S * max(m_dino, m_lpips)     A0, R49's method -- recomputed for the gate only, never written
  nodiv   S                            A1  no divergence
  dino    S * m_dino                   A2  DINO only
  lpips   S * m_lpips                  A3  LPIPS only
  nosupp  max(m_dino, m_lpips)         A4  no support

Mapping unchanged (r49_field): StreamEdit rho = budget-linear inverse on [2, 50] (15-step t_next), Wan-Edit start = 50 - round(50 q)
clipped to s >= 5.

A0 GATE: before anything is written, A0 is recomputed with this code for every pair and must be bit-equal to R49's field
(--gate_ref: se_max/.../{video}.npz key `rho`, or wan_max_s5/.../{video}.npy). Any mismatch or unreadable pair exits 1 with
nothing written, so a wrong input root or cap can never reach a render.

Outputs {out_root}/se_{arm}/edit{T}/{video}.npz            rho [F, 1560] (r31_stage3 --rho_dir format) + q, S, m_dino, m_lpips
        {out_root}/wan_{arm}_s5/edit{T}/{video}.npz/.npy   start_step int [F, 30, 52] (.npy = --route_start_map) + the same fields
usage: python evaluation/r51_field.py --host se --cases evaluation/r7_anchor_manifest.json
       python evaluation/r51_field.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_field as R48  # noqa: E402
import r49_field as R49  # noqa: E402
from r30_b_map import t_next_schedule  # noqa: E402

D = R49.D
OUT_ROOT = D / "r51_field_full"
GATE_REF = D / "r49_field_full"                                    # R49's full-stage field (A0)
ATTN_ROOT = {"se": D / "r48_se_attn", "wan": D / "r48_wan_draft/attn"}  # R49 full stage reads the R48 full drafts
DIV_ROOT = D / "r48_div"
WAN_S_MIN = 5                                                       # max_s5's floor, the only Wan-Edit floor in the full stage
ARMS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "max": lambda S, md, ml: S * np.maximum(md, ml),
    "nodiv": lambda S, md, ml: S,
    "dino": lambda S, md, ml: S * md,
    "lpips": lambda S, md, ml: S * ml,
    "nosupp": lambda S, md, ml: np.maximum(md, ml),
}
DEFAULT_ARMS = {"se": ("nodiv", "dino", "lpips", "nosupp"), "wan": ("nosupp",)}


def demand(arm: str, S: np.ndarray, m_dino: np.ndarray, m_lpips: np.ndarray) -> np.ndarray:
    return np.clip(ARMS[arm](S, m_dino, m_lpips), 0.0, 1.0)


def load_inputs(host: str, t: int, vid: str, attn_root: Path, div_root: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """(n_frames, S, m_dino, m_lpips) exactly as r49_field.field_case computes them."""
    a = np.load(attn_root / f"edit{t}" / f"{vid}.npz")
    n_frames = a["src_tokens"].shape[1]
    S, m = None, {}
    for metric in ("dino", "lpips"):
        d = R48._load_d(div_root / f"{host}_{metric}" / f"edit{t}" / f"{vid}.npz", n_frames)
        S_m, m[metric], _ = R48.compute_E(a["src_tokens"], a["trg_tokens"], a["src_prompt"], a["trg_prompt"],
                                          a["idx_src"], a["idx_trg"], d, R49.DIV_MAX[metric],
                                          int(a["n_prompt_src"]), int(a["n_prompt_trg"]))
        if S is None:
            S = S_m
        elif not np.array_equal(S, S_m):
            raise ValueError("S depends only on attention, not on the metric")
    return n_frames, S, m["dino"], m["lpips"]


def control(host: str, q: np.ndarray, n_frames: int, t_next: np.ndarray) -> np.ndarray:
    """R49's inverse mapping, with R49's own dtype / shape conventions (so A0 can be compared bit for bit)."""
    if host == "se":
        return R49.se_rho(q, t_next).reshape(n_frames, -1).astype(np.float32)
    return R49.wan_start(q, WAN_S_MIN)


def gate_one(host: str, t: int, vid: str, attn_root: Path, div_root: Path, gate_ref: Path, t_next: np.ndarray) -> bool:
    n_frames, S, md, ml = load_inputs(host, t, vid, attn_root, div_root)
    a0 = control(host, demand("max", S, md, ml), n_frames, t_next)
    if host == "se":
        ref = np.load(gate_ref / "se_max" / f"edit{t}" / f"{vid}.npz")["rho"]
    else:
        ref = np.load(gate_ref / f"wan_max_s{WAN_S_MIN}" / f"edit{t}" / f"{vid}.npy")
    return a0.dtype == ref.dtype and np.array_equal(a0, ref)


def write_arm(host: str, arm: str, t: int, vid: str, out_root: Path, ctrl: np.ndarray, q: np.ndarray, S: np.ndarray,
              m_dino: np.ndarray, m_lpips: np.ndarray, n_frames: int) -> Path:
    meta = dict(q=q.astype(np.float32), S=S.astype(np.float32), m_dino=m_dino.astype(np.float32),
                m_lpips=m_lpips.astype(np.float32), arm=np.str_(arm), host=np.str_(host), video_name=np.str_(vid),
                edit_type=np.int32(t), div_max_dino=np.float32(R49.DIV_MAX["dino"]),
                div_max_lpips=np.float32(R49.DIV_MAX["lpips"]), lam=np.float32(R48.LAM),
                mapping=np.str_("inverse_exposure_response"), n_latent_frames=np.int32(n_frames),
                grid_h=np.int32(R48.GRID[0]), grid_w=np.int32(R48.GRID[1]), frame_seq_length=np.int32(R48.FRAME_SEQ_LENGTH))
    if host == "se":
        out = out_root / f"se_{arm}" / f"edit{t}" / f"{vid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, rho=ctrl, rho_min=np.float32(R49.RHO_MIN), rho_max=np.float32(R49.RHO_MAX),
                            step=np.int32(R49.SE_STEP), flow_shift=np.float32(R49.SE_FLOW_SHIFT), **meta)
    else:
        out = out_root / f"wan_{arm}_s{WAN_S_MIN}" / f"edit{t}" / f"{vid}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, start_step=ctrl, s_min=np.int32(WAN_S_MIN), wan_n=np.int32(R49.WAN_N),
                            wan_shift=np.float32(R48.WAN_SHIFT), **meta)
        np.save(out.with_suffix(".npy"), ctrl)
    return out


def selftest() -> int:
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 1, (3, 30, 52)); md = rng.uniform(0, 1, S.shape); ml = rng.uniform(0, 1, S.shape)
    q = {arm: demand(arm, S, md, ml) for arm in ARMS}
    for arm, v in q.items():
        assert v.shape == S.shape and np.all((v >= 0) & (v <= 1)), arm
    assert np.array_equal(q["nodiv"], S)
    assert np.array_equal(q["dino"], S * md) and np.array_equal(q["lpips"], S * ml)
    assert np.all(q["nosupp"] >= q["max"]), "S <= 1, so dropping the support can only raise q"
    _, q_r49 = R49.demand(S, md, ml)
    assert np.array_equal(q["max"], q_r49), "the A0 arm must reproduce r49_field.demand bit for bit"
    t_next = t_next_schedule(R49.SE_STEP, R49.SE_FLOW_SHIFT)
    rho = control("se", q["max"], 3, t_next)
    assert rho.shape == (3, 30 * 52) and rho.dtype == np.float32
    start = control("wan", q["max"], 3, t_next)
    assert start.shape == S.shape and start.min() >= WAN_S_MIN and start.max() <= R49.WAN_N
    print(f"[r51_field] selftest PASS (arms {', '.join(ARMS)}; nosupp >= max; max == r49_field.demand)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cases")
    ap.add_argument("--host", choices=("se", "wan"))
    ap.add_argument("--arms", nargs="+", default=None, choices=[a for a in ARMS if a != "max"])
    ap.add_argument("--attn_root", default=None, help="default: R49's full-stage attention for --host")
    ap.add_argument("--div_root", default=str(DIV_ROOT))
    ap.add_argument("--gate_ref", default=str(GATE_REF))
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.cases and a.host):
        ap.error("--cases and --host are required (or --selftest)")
    arms = a.arms or list(DEFAULT_ARMS[a.host])
    attn_root = Path(a.attn_root) if a.attn_root else ATTN_ROOT[a.host]
    div_root, gate_ref, out_root = Path(a.div_root), Path(a.gate_ref), Path(a.out_root)
    t_next = t_next_schedule(R49.SE_STEP, R49.SE_FLOW_SHIFT)
    cases = [(c["video_name"], int(c["edit_type"])) for c in json.loads(Path(a.cases).read_text())]
    print(f"[r51_field] host={a.host} arms={arms} cases={len(cases)} attn={attn_root} div={div_root} gate_ref={gate_ref} "
          f"out={out_root}", flush=True)

    # ---- A0 gate: every pair, before anything is written
    bad: List[str] = []
    for vid, t in cases:
        try:
            if not gate_one(a.host, t, vid, attn_root, div_root, gate_ref, t_next):
                bad.append(f"{t}/{vid}: differs from the R49 field")
        except Exception as e:                       # noqa: BLE001 -- an unreadable pair fails the gate too
            bad.append(f"{t}/{vid}: {type(e).__name__}: {e}")
    if bad:
        for b in bad[:10]:
            print(f"[r51_field] A0 gate MISMATCH {b}", flush=True)
        print(f"[r51_field] A0 gate FAIL {len(cases) - len(bad)}/{len(cases)} -- nothing written")
        print(f"[r51_field] failures: {len(bad)}")
        return 1
    print(f"[r51_field] A0 gate PASS {len(cases)}/{len(cases)}", flush=True)

    # ---- the ablation arms
    n_fail = 0
    stats: Dict[str, List[float]] = {arm: [] for arm in arms}
    for vid, t in cases:
        try:
            n_frames, S, md, ml = load_inputs(a.host, t, vid, attn_root, div_root)
            for arm in arms:
                q = demand(arm, S, md, ml)
                ctrl = control(a.host, q, n_frames, t_next)
                write_arm(a.host, arm, t, vid, out_root, ctrl, q, S, md, ml, n_frames)
                stats[arm].append(float(ctrl.mean()))
        except Exception as e:                       # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r51_field] FAILED {a.host} {t}/{vid}: {type(e).__name__}: {e}", flush=True)
    unit = "rho" if a.host == "se" else "start step"
    for arm in arms:
        if stats[arm]:
            print(f"[r51_field] {a.host}_{arm}: {len(stats[arm])}/{len(cases)} clips, mean {unit} {np.mean(stats[arm]):.3f}")
    print(f"[r51_field] failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
