"""R54 stage 2: draft-free rho field -- R40's flow-warped anchor maps, combined by R49's max, mapped by R49's exposure inverse.

R40 measured ONE divergence map between source frame 0 and the VP anchor and carried it to every frame by RAFT optical flow on the source
video (r40_warp_divergence.py), writing the raw per-token map d [F, 1560] per metric ({dino_warp, lpips_warp}). Those maps went through the
same DINOv3 / LPIPS functions, the same 16x16 token-grid reduction and the same VAE temporal mean as R49's per-frame divergence, so they are the
warped counterparts of R49's d_dino / d_lpips. R54 takes them exactly where R49 takes its own:

  m_dino = clip(d_dino, 0, 0.55) / 0.55,   m_lpips = clip(d_lpips, 0, 1.0) / 1.0,   q = max(m_dino, m_lpips)   (per token, after the reduction)
  rho = r49_field.se_rho(q)   (budget-linear inverse on [2, 50], 15-step t_next)

No attention support S: a draft-free arm has no draft, hence no cross-attention. Its exact draft-based counterpart is R51's A4 (nosupp: the same
q with a per-frame draft), so R54 vs A4 isolates "per-frame draft vs flow-warped anchor". The max is NOT taken on warped maps: the two metrics live
on different grids at frame 0 (DINO per 16x16 patch, LPIPS per pixel), and a max before the token-grid reduction would not equal R49's q.

CHECK: every pair's latent-frame count must equal R49's (read from r49_field_full/se_max/{video}.npz, n_latent_frames); a mismatch fails that
pair (the whole job then fails) instead of being resampled.

Output {out_root}/se_warp_max/edit{T}/{video}.npz   rho [F, 1560] (r31_stage3 --rho_dir format) + q, m_dino, m_lpips, d_dino, d_lpips
usage: python evaluation/r54_field.py --cases evaluation/r7_anchor_manifest.json
       python evaluation/r54_field.py --selftest
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
import r49_field as R49  # noqa: E402
import r51_field as R51  # noqa: E402
from r30_b_map import t_next_schedule  # noqa: E402

D = R49.D
WARP_ROOT = D / "r40_div"                                  # R40's flow-warped maps (a /store link on the lab: read on a node that mounts it)
REF_ROOT = D / "r49_field_full"                            # R49's field: the reference latent-frame count per pair
OUT_ROOT = D / "r54_field_full"
WARP_ARM = {"dino_warp": "dino_patch", "lpips_warp": "lpips"}   # the `arm` each R40 file must carry (guards against swapped folders)


def warp_demand(d_dino: np.ndarray, d_lpips: np.ndarray):
    """(m_dino, m_lpips, q): R49's caps, then the per-token max."""
    m_dino = np.clip(d_dino.astype(np.float64), 0.0, R49.DIV_MAX["dino"]) / R49.DIV_MAX["dino"]
    m_lpips = np.clip(d_lpips.astype(np.float64), 0.0, R49.DIV_MAX["lpips"]) / R49.DIV_MAX["lpips"]
    return m_dino, m_lpips, np.clip(np.maximum(m_dino, m_lpips), 0.0, 1.0)


def load_warp(warp_root: Path, name: str, t: int, vid: str, n_frames: int) -> np.ndarray:
    path = warp_root / name / f"edit{t}" / f"{vid}.npz"
    arm = str(np.load(path)["arm"])
    if arm != WARP_ARM[name]:
        raise ValueError(f"{path}: arm {arm!r} != {WARP_ARM[name]!r}")
    d = R48._load_d(path, n_frames)                          # raises unless d is [n_frames, 1560]
    if not np.isfinite(d).all():
        raise ValueError(f"{path}: non-finite divergence")
    return d


def field_case(t: int, vid: str, warp_root: Path, ref_root: Path, out_root: Path, t_next: np.ndarray) -> Dict[str, float]:
    n_frames = int(np.load(ref_root / "se_max" / f"edit{t}" / f"{vid}.npz")["n_latent_frames"])
    d_dino = load_warp(warp_root, "dino_warp", t, vid, n_frames)
    d_lpips = load_warp(warp_root, "lpips_warp", t, vid, n_frames)
    m_dino, m_lpips, q = warp_demand(d_dino, d_lpips)
    rho = R49.se_rho(q, t_next).reshape(n_frames, -1).astype(np.float32)
    out = out_root / "se_warp_max" / f"edit{t}" / f"{vid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rho=rho, rho_min=np.float32(R49.RHO_MIN), rho_max=np.float32(R49.RHO_MAX), step=np.int32(R49.SE_STEP),
                        flow_shift=np.float32(R49.SE_FLOW_SHIFT), arm=np.str_("warp_max"), host=np.str_("se"), video_name=np.str_(vid),
                        edit_type=np.int32(t), q=q.astype(np.float32), m_dino=m_dino.astype(np.float32), m_lpips=m_lpips.astype(np.float32),
                        d_dino=d_dino.astype(np.float32), d_lpips=d_lpips.astype(np.float32),
                        div_max_dino=np.float32(R49.DIV_MAX["dino"]), div_max_lpips=np.float32(R49.DIV_MAX["lpips"]),
                        mapping=np.str_("inverse_exposure_response"), source=np.str_(str(warp_root)), n_latent_frames=np.int32(n_frames),
                        grid_h=np.int32(R48.GRID[0]), grid_w=np.int32(R48.GRID[1]), frame_seq_length=np.int32(R48.FRAME_SEQ_LENGTH))
    return {"q_mean": float(q.mean()), "rho_mean": float(rho.mean())}


def selftest() -> int:
    rng = np.random.default_rng(0)
    md_raw = rng.uniform(0, 1.2, (3, 30, 52)); ml_raw = rng.uniform(0, 1.5, md_raw.shape)
    m_dino, m_lpips, q = warp_demand(md_raw, ml_raw)
    assert np.all((q >= 0) & (q <= 1)) and np.all(q >= m_dino) and np.all(q >= m_lpips)
    assert np.allclose(m_dino, np.clip(md_raw, 0, 0.55) / 0.55) and np.allclose(m_lpips, np.clip(ml_raw, 0, 1.0))
    # R54's demand is R51's A4 (no support) on the same maps: the pair R54 vs A4 differs only in where the maps come from
    assert np.array_equal(q, R51.demand("nosupp", np.ones_like(q), m_dino, m_lpips))
    t_next = t_next_schedule(R49.SE_STEP, R49.SE_FLOW_SHIFT)
    rho = R49.se_rho(q, t_next)
    assert rho.min() >= R49.RHO_MIN - 1e-6 and rho.max() <= R49.RHO_MAX + 1e-6
    ends = R49.se_rho(np.array([0.0, 1.0]), t_next)
    assert abs(ends[0] - R49.RHO_MIN) < 1e-6 and abs(ends[1] - R49.RHO_MAX) < 1e-6
    print(f"[r54_field] selftest PASS (q = max of the capped maps == R51 nosupp; rho in [{rho.min():.2f}, {rho.max():.2f}], endpoints 2 / 50)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cases")
    ap.add_argument("--warp_root", default=str(WARP_ROOT))
    ap.add_argument("--ref_root", default=str(REF_ROOT))
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.cases:
        ap.error("--cases is required (or --selftest)")
    t_next = t_next_schedule(R49.SE_STEP, R49.SE_FLOW_SHIFT)
    cases = [(c["video_name"], int(c["edit_type"])) for c in json.loads(Path(a.cases).read_text())]
    print(f"[r54_field] cases={len(cases)} warp_root={a.warp_root} ref_root={a.ref_root} out_root={a.out_root}", flush=True)
    agg: List[Dict[str, float]] = []
    n_fail = 0
    for vid, t in cases:
        try:
            agg.append(field_case(t, vid, Path(a.warp_root), Path(a.ref_root), Path(a.out_root), t_next))
        except Exception as e:                       # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r54_field] FAILED {t}/{vid}: {type(e).__name__}: {e}", flush=True)
    if agg:
        print(f"[r54_field] se_warp_max: {len(agg)}/{len(cases)} clips; mean q {np.mean([s['q_mean'] for s in agg]):.3f}, "
              f"mean rho {np.mean([s['rho_mean'] for s in agg]):.3f}")
    print(f"[r54_field] frame-count check {'PASS' if n_fail == 0 else 'FAIL'} {len(agg)}/{len(cases)}")
    print(f"[r54_field] failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
