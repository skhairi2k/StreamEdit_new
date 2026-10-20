"""R48 qualitative grids: what the field and its host mapping actually look like, per clip, per arm, per host.

The R48 analogue of r31_stage3_grids.py's ``--which tau`` page, for both hosts. One PNG per (host, arm, clip), K evenly
subsampled latent frames as columns and five field rows (plus the source frame on top):

  d        raw, UNCLIPPED draft-source divergence, drawn on [0, div_max] so saturation shows as a flat plateau
  m        clip(d, 0, div_max) / div_max
  S        soft attention support
  E        S * m (or m / S alone for the dino_D / S ablation arms) -- the field that is actually mapped
  control  the host's mapped value: rho (StreamEdit) or the START SIGMA of the step each token starts at (Wan-Edit)

The control row's title states its value at E = 0.5, computed from the npz's OWN stored mapping constants
(rho_min/rho_max, sigma_floor/sigma_scale/wan_n/wan_shift) -- never a hardcoded number, so a changed mapping cannot
silently mislabel the figure (the same reason r31_stage3_grids has ``_tau_at_m_half``). Reads npz written by
r48_field.py; runs no model.

Output: evaluation/figures/r48_tau_grids/{host}/{arm}/e{T}_{video}.png
usage: python evaluation/r48_tau_grids.py --mode subset --cases evaluation/r48_cases.json [--host both] [--n_frames 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_field as R  # noqa: E402

FIVE = Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
FIG_ROOT = HERE / "figures" / "r48_tau_grids"
ARM_METRIC = {"dino_SD": "dino", "dino_D": "dino", "lpips_SD": "lpips", "S": "dino"}


def source_frame(vid: str, f: int) -> np.ndarray:
    """RGB source frame at the first pixel frame of latent frame f (pixel 0 for f = 0, 4f otherwise)."""
    cap = cv2.VideoCapture(str(FIVE / "videos" / f"{vid}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0 if f == 0 else 4 * f)
    ok, fr = cap.read()
    return cv2.cvtColor(cv2.resize(fr, (832, 480)), cv2.COLOR_BGR2RGB) if ok else np.zeros((480, 832, 3), np.uint8)


def control_info(z) -> tuple:
    """(control array [F,30,52], label, vmin, vmax, value at E=0.5) from the npz's own stored constants."""
    if str(z["host"]) == "se":
        lo, hi = float(z["rho_min"]), float(z["rho_max"])
        rho = z["rho"].reshape(z["E"].shape)
        return rho, "rho", lo, hi, lo + (hi - lo) * 0.5
    n, shift = int(z["wan_n"]), float(z["wan_shift"])
    t = (n - np.arange(n)) / n
    grid = shift * t / (1 + (shift - 1) * t)
    start = z["start_step"]
    sig = grid[start]
    target = float(z["sigma_floor"]) + float(z["sigma_scale"]) * 0.5
    step = int(np.abs(grid - target).argmin())
    return sig, "start sigma", float(grid[start].min()), float(grid[start].max()), f"{target:.3f} (step {step})"


def page(z, host: str, arm: str, vid: str, t: int, n_frames: int, out: Path) -> None:
    F = z["E"].shape[0]
    fs = sorted({int(round(x)) for x in np.linspace(0, F - 1, n_frames)})
    dmax = float(z["div_max_lpips"] if ARM_METRIC[arm] == "lpips" else z["div_max_dino"])
    ctrl, clabel, cmin, cmax, half = control_info(z)
    rows = [("source", None, None, None),
            (f"d raw (colour range 0..div_max={dmax:g})", z["d"], 0.0, dmax),
            ("m = clip(d)/div_max", z["m"], 0.0, 1.0),
            ("S soft support", z["S"], 0.0, 1.0),
            (f"E ({arm})", z["E"], 0.0, 1.0),
            (f"{clabel}   [value at E=0.5: {half if isinstance(half, str) else f'{half:.3g}'}]", ctrl, cmin, cmax)]
    fig, axes = plt.subplots(len(rows), len(fs), figsize=(2.6 * len(fs), 1.7 * len(rows)), squeeze=False)
    for r, (name, arr, lo, hi) in enumerate(rows):
        for c, f in enumerate(fs):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if arr is None:
                ax.imshow(source_frame(vid, f))
            else:
                im = ax.imshow(arr[f], cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
            if r == 0:
                ax.set_title(f"latent frame {f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
        if arr is not None:
            cb = fig.colorbar(im, ax=axes[r].tolist(), fraction=0.012, pad=0.01)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"R48 {host} / {arm} / edit{t} / {vid}", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("subset", "full"), required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--host", choices=("se", "wan", "both"), default="both")
    ap.add_argument("--field_root", default=str(R.OUT_ROOT))
    ap.add_argument("--out_dir", default=None,
                    help="default: evaluation/figures/r48_tau_grids (subset) or r48_tau_grids_full (full), so the two never overwrite")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--selected", action="store_true",
                    help="only the (host, arm) pairs listed in --selected_file (the arms that passed the gate) instead of all 4 arms x both hosts")
    ap.add_argument("--selected_file", default=str(HERE / "r48_selected_arms.txt"))
    a = ap.parse_args()
    if a.out_dir is None:
        a.out_dir = str(FIG_ROOT if a.mode == "subset" else FIG_ROOT.with_name(FIG_ROOT.name + "_full"))
    cases = [(c["video_name"], int(c["edit_type"])) for c in json.loads(Path(a.cases).read_text())]
    hosts = ("se", "wan") if a.host == "both" else (a.host,)
    pairs = [(h, arm) for h in hosts for arm in R.ARMS]
    if a.selected:
        sf = Path(a.selected_file)
        if not sf.is_file() or not sf.read_text().strip():
            raise SystemExit(f"[r48_tau_grids] --selected: {sf} missing or empty -- write the passing (host, arm) pairs at gate-decision")
        chosen = {tuple(l.split()[:2]) for l in sf.read_text().splitlines() if l.strip() and not l.startswith("#")}
        bad = [c for c in chosen if c[0] not in ("se", "wan") or c[1] not in R.ARMS]
        if bad:
            raise SystemExit(f"[r48_tau_grids] bad lines in {sf}: {bad}")
        pairs = [(h, arm) for h, arm in pairs if (h, arm) in chosen]
        print(f"[r48_tau_grids] --selected: {len(pairs)} (host, arm) pair(s): {pairs}", flush=True)
    n_ok = n_fail = 0
    for host, arm in pairs:
        if True:
            for vid, t in cases:
                npz = Path(a.field_root) / f"{host}_{arm}" / f"edit{t}" / f"{vid}.npz"
                try:
                    z = np.load(npz)
                    page(z, host, arm, vid, t, a.n_frames, Path(a.out_dir) / host / arm / f"e{t}_{vid}.png")
                    n_ok += 1
                except Exception as e:                       # noqa: BLE001 -- report per page, keep going
                    n_fail += 1
                    print(f"[r48_tau_grids] FAILED {host}/{arm} {t}/{vid}: {type(e).__name__}: {e}", flush=True)
    print(f"[r48_tau_grids] {a.mode}: {n_ok} pages -> {a.out_dir}  failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
