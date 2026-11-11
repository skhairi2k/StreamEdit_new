"""R71: Wan-Edit start maps under the range-normalized ControlLift mapping, remapped from the stored demand q.

R49 / R58 / R62 / R68 map the demand to a start step as  s = clip(50 - round(50 q), s_min = 5, 50)  (normalized over the full
50-step gate, then capped). R71 normalizes the exposure over the operating range m in {0, ..., m_max = 49}, like StreamEdit:

    m = round(49 q),   s = 50 - m          (q = 0 -> s = 50, never edited;  q = 1 -> s = 1, start sigma 0.9957)

q is stored in every field npz, so no draft, attention or divergence is recomputed. Per arm and pair of the 415-pair manifest:
  GATE   the stored s_min 5 start map must equal clip(50 - round(50 q), 5, 50), computed in float64 from the stored (float32) q.
         The only tolerated difference: a token whose 50 q lies within TIE_TOL of x.5 (a float32 / float64 rounding tie, about 1 token
         in 10^7); those are counted and printed. Any other mismatch = the stored q is not the q that was rendered -> nothing is
         written for that arm, exit 1.
  WRITE  {out_root}/wan_{arm}_s1/edit{T}/{video}.npz (every source key copied; start_step, s_min = 1, m_max = 49, arm '{arm}_s1',
         mapping 'inverse_exposure_normalized') and the .npy start map (int64, same shape as the source map).

Sources (roots under --d_root = ~/Data/dataggen/outputs/five_bench; the 190 / 225 split is A0 = R58 max_s5's own assembly):
  max                 r58_field/wan_max_s5 (190 pairs of r58_cases_maxs5.json)   + r49_field_full/wan_max_s5 (225, r68_cases_reuse225.json)
  nosupp  (A4)        r62_field/wan_nosupp_s5 (190)                               + r51_field_full/wan_nosupp_s5 (225)
  nodiv / dino / lpips (A1-A3)   r68_field/wan_{arm}_s5 (all 415)

usage: python evaluation/r71_remap_field.py --selftest
       python evaluation/r71_remap_field.py --out_root $D/r71_field [--arms max nosupp nodiv dino lpips]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D_ROOT = Path.home() / "Data/dataggen/outputs/five_bench"
WAN_N = 50
S_MIN_OLD = 5            # the stored maps
M_MAX = 49               # R71: operating range m in {0, ..., 49}, s_min = 1
TIE_TOL = 1e-3           # |50 q - (n + 0.5)| below this: a float32 / float64 rounding tie
ARMS = ("max", "nosupp", "nodiv", "dino", "lpips")
# arm -> [(source root, case file giving the pairs it serves; None = every manifest pair)]
SOURCES: Dict[str, List[Tuple[str, str]]] = {
    "max": [("r58_field/wan_max_s5", "evaluation/r58_cases_maxs5.json"),
            ("r49_field_full/wan_max_s5", "evaluation/r68_cases_reuse225.json")],
    "nosupp": [("r62_field/wan_nosupp_s5", "evaluation/r58_cases_maxs5.json"),
               ("r51_field_full/wan_nosupp_s5", "evaluation/r68_cases_reuse225.json")],
    "nodiv": [("r68_field/wan_nodiv_s5", "")],
    "dino": [("r68_field/wan_dino_s5", "")],
    "lpips": [("r68_field/wan_lpips_s5", "")],
}
MANIFEST = "evaluation/r43_anchor_manifest_correct.json"


def old_start(q: np.ndarray) -> np.ndarray:
    """R49's mapping (r49_field.wan_start) in float64: clip(50 - round(50 q), 5, 50)."""
    n = np.round(WAN_N * np.clip(q.astype(np.float64), 0.0, 1.0)).astype(np.int64)
    return np.clip(WAN_N - n, S_MIN_OLD, WAN_N).astype(np.int64)


def new_start(q: np.ndarray) -> np.ndarray:
    """R71 mapping: m = round(49 q) (numpy, half to even), s = 50 - m."""
    m = np.round(M_MAX * np.clip(q.astype(np.float64), 0.0, 1.0)).astype(np.int64)
    return (WAN_N - m).astype(np.int64)


def gate(q: np.ndarray, stored: np.ndarray) -> Tuple[bool, int]:
    """(passes, number of tolerated tie tokens): stored == old_start(q) except float32 / float64 rounding ties."""
    diff = old_start(q) != stored
    if not diff.any():
        return True, 0
    x = WAN_N * np.clip(q.astype(np.float64), 0.0, 1.0)
    tie = np.abs(x - (np.floor(x) + 0.5)) < TIE_TOL
    ok = bool(np.all(tie[diff]) and np.all(np.abs(old_start(q) - stored)[diff] == 1))
    return ok, int(diff.sum())


def load_pairs(path: Path) -> List[Tuple[int, str]]:
    return [(int(c["edit_type"]), str(c["video_name"])) for c in json.load(open(path))]


def plan(arm: str, d_root: Path) -> List[Tuple[int, str, Path]]:
    """(edit type, video, source npz) for the 415 manifest pairs of an arm; a pair served by no source is an error."""
    manifest = load_pairs(REPO / MANIFEST)
    src: Dict[Tuple[int, str], Path] = {}
    for root, cases in SOURCES[arm]:
        pairs = load_pairs(REPO / cases) if cases else manifest
        for t, v in pairs:
            if (t, v) in src:
                raise SystemExit(f"[r71_remap] {arm}: pair {t}/{v} served by two sources")
            src[(t, v)] = d_root / root / f"edit{t}" / f"{v}.npz"
    missing = [k for k in manifest if k not in src]
    if missing or len(src) != len(manifest):
        raise SystemExit(f"[r71_remap] {arm}: sources cover {len(src)} pairs, the manifest holds {len(manifest)} (missing e.g. {missing[:3]})")
    return [(t, v, src[(t, v)]) for t, v in manifest]


def run_arm(arm: str, d_root: Path, out_root: Path) -> int:
    items = plan(arm, d_root)
    out_dir = out_root / f"wan_{arm}_s1"
    # ---- pass 1: gate every pair, write nothing
    bad, ties, s_old_sum, s_new_sum, ntok = [], 0, 0.0, 0.0, 0
    for t, v, p in items:
        if not p.exists():
            bad.append(f"{t}/{v}: missing {p}")
            continue
        f = np.load(p)
        q, st = f["q"], f["start_step"]
        if st.shape != q.shape:
            bad.append(f"{t}/{v}: start_step {st.shape} != q {q.shape}")
            continue
        ok, nt = gate(q, st)
        ties += nt
        if not ok:
            bad.append(f"{t}/{v}: stored start map != clip(50 - round(50 q), 5, 50) on {int((old_start(q) != st).sum())} token(s) beyond rounding ties")
            continue
        s_old_sum += float(st.sum()); s_new_sum += float(new_start(q).sum()); ntok += st.size
    if bad:
        print(f"[r71_remap] {arm}: GATE FAIL on {len(bad)}/{len(items)} pair(s), nothing written:")
        for b in bad[:10]:
            print("   ", b)
        return 1
    print(f"[r71_remap] {arm}: gate PASS {len(items)}/{len(items)} (tolerated float32 rounding ties: {ties} token(s)); "
          f"mean start {s_old_sum / ntok:.3f} (s_min 5) -> {s_new_sum / ntok:.3f} (s_min 1)")
    # ---- pass 2: write
    for t, v, p in items:
        out = out_dir / f"edit{t}" / f"{v}.npz"
        if out.exists() or out.with_suffix(".npy").exists():
            print(f"[r71_remap] REFUSE: {out} exists (R71 fields are never overwritten)")
            return 1
        f = np.load(p)
        keep = {k: f[k] for k in f.files}
        s1 = new_start(f["q"])
        assert s1.shape == f["start_step"].shape and s1.min() >= 1 and s1.max() <= WAN_N
        keep.update(start_step=s1, s_min=np.int32(1), m_max=np.int32(M_MAX), arm=np.str_(f"{arm}_s1"),
                    mapping=np.str_("inverse_exposure_normalized"))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **keep)
        np.save(out.with_suffix(".npy"), s1)
    print(f"[r71_remap] {arm}: wrote {len(items)} npz + {len(items)} .npy -> {out_dir}")
    return 0


def selftest() -> int:
    q = np.array([0.0, 1.0, 0.5, 0.9, 0.95, 0.01, 0.02, 1.2, -0.3])
    s1, s0 = new_start(q), old_start(q)
    assert s1[0] == 50 and s1[1] == 1 and s1[2] in (25, 26) and s1[7] == 1 and s1[8] == 50, s1
    assert s0[0] == 50 and s0[1] == 5 and s0[3] == 5 and s0[4] == 5, s0           # the old cap: every q >= 0.9 -> 5
    assert s1[3] == 6 and s1[4] == 3, s1                                        # new: q = 0.9 -> 6, 0.95 -> 3
    g = np.linspace(0, 1, 5001)
    assert np.all(np.diff(new_start(g)) <= 0) and np.all(np.diff(old_start(g)) <= 0), "start must be non-increasing in q"
    assert new_start(g).min() == 1 and new_start(g).max() == 50
    ok, n = gate(np.float32([0.78999996, 0.5, 0.0]), old_start(np.float32([0.78999996, 0.5, 0.0])))
    assert ok and n == 0
    ok, n = gate(np.float32([0.78999996]), np.array([10]))                       # the float32 tie of 0033_horsejump-low edit3
    assert ok and n == 1, (ok, n)
    ok, _ = gate(np.float32([0.5]), np.array([20]))                              # 25 expected, 20 stored: a real mismatch
    assert not ok
    print("[r71_remap] selftest PASS (q 0 -> s 50, q 1 -> s 1, q 0.9 -> 6, q 0.95 -> 3; old cap 5; monotone; tie gate)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--d_root", default=str(D_ROOT))
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out_root:
        ap.error("--out_root is required")
    rc = 0
    for arm in a.arms:
        rc |= run_arm(arm, Path(a.d_root).expanduser(), Path(a.out_root).expanduser())
    print(f"[r71_remap] failures: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
