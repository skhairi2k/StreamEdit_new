"""R49 qualitative grids: the official baseline next to the R49 variants, and the fields behind each R49 control map.

(a) Frame grids, one PNG per host (rows = the 18 clips of r49_cases.json; each cell = middle | last frame side by side, labelled with
    the clip's bg LPIPS / FiVE-Acc when that method's per-clip CSVs exist):
      StreamEdit  source | official (rho 2, N 15) | R48 dino_SD | R49 max
      Wan-Edit    source | official (N 50, skip 15) | R48 dino_SD | R49 max s_min 9 | R49 max s_min 5
(b) Field pages, one PNG per host and clip, over 6 evenly spaced latent frames: source, d_dino (0..0.55), d_lpips (0..1.0),
    div, S, q, and the control map (StreamEdit rho; Wan-Edit start sigma for s_min 9 and 5), annotated with the control value
    at q = 0.5 computed from the npz's own constants.
Reads renders, r49_field npz and per-clip CSVs only; runs no model.

Outputs evaluation/figures/r49_grids/frames_{se,wan}.png, evaluation/figures/r49_grids/fields/{se,wan}/e{T}_{video}.png
usage: python evaluation/r49_grids.py [--field_root DIR] [--out_dir DIR] [--cases FILE]
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
import r48_score as R  # noqa: E402
from r30_b_map import t_next_schedule  # noqa: E402
from r31_rho_map import budget_linear_tau  # noqa: E402

D = Path.home() / "Data/dataggen/outputs/five_bench"
FIVE = Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
# column -> (render root with edit{T}/{video}/*.png, harness stem, five_acc stem, clipd csv, clipd method) -- stems for the labels
COLUMNS = {
    "se": [("official: rho 2, N 15", D / "r36_rho_sweep/r36_rho2_vp", "r36_rho2_vp", "r36_fiveacc_rho2_vp"),
           ("R48 dino_SD", D / "r48_se_arms/r48_dino_SD_subset/step14", "r49_se_r48_dino_SD", "r49_fiveacc_se_r48_dino_SD"),
           ("R49 max", D / "r49_se_arms/r49_max_subset/step14", "r49_se_max", "r49_fiveacc_se_max")],
    "wan": [("official: N 50, skip 15", D / "r48_wan_arms/native_s0.92_subset", "r49_wan_native_s0p92", "r49_fiveacc_wan_native_s0p92"),
            ("R48 dino_SD", D / "r48_wan_arms/adapt_dino_SD_subset", "r49_wan_r48_dino_SD", "r49_fiveacc_wan_r48_dino_SD"),
            ("R49 max, s_min 9", D / "r49_wan_arms/max_s9_subset", "r49_wan_max_s9", "r49_fiveacc_wan_max_s9"),
            ("R49 max, s_min 5", D / "r49_wan_arms/max_s5_subset", "r49_wan_max_s5", "r49_fiveacc_wan_max_s5")],
}


def per_clip(csv_dir: Path, harness: str, fiveacc: str, scope, names):
    """{clip: (bg LPIPS, FiVE-Acc)} or {} if the CSVs are not there (labels are optional)."""
    try:
        h = R.load_harness(csv_dir, harness, (R.Y,), scope, names)[R.Y]
        f = R.load_harness(csv_dir, fiveacc, ("five_acc",), scope, names)["five_acc"]
        return {k: (h[k], f[k]) for k in h if k in f}
    except (SystemExit, KeyError, IndexError):
        return {}


def frames(dirp: Path):
    return sorted(dirp.glob("*.png")) if dirp.is_dir() else []


def rgb(p) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def source_frame(vid: str, idx: int, size) -> np.ndarray:
    cap = cv2.VideoCapture(str(FIVE / "videos" / f"{vid}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), size) if ok else np.zeros((size[1], size[0], 3), np.uint8)


def frame_grid(host: str, cases, csv_dir: Path, out: Path, scope, names) -> None:
    cols = COLUMNS[host]
    labels = [per_clip(csv_dir, h, f, scope, names) for _, _, h, f in cols]
    fig, axes = plt.subplots(len(cases), len(cols) + 1, figsize=(4.2 * (len(cols) + 1), 1.3 * len(cases)), squeeze=False)
    for r, (t, vid) in enumerate(cases):
        ref = frames(cols[0][1] / f"edit{t}" / vid)                  # official render fixes the frame indices of the row
        n = len(ref) if ref else 41
        idx = (n // 2, n - 1)
        size = (rgb(ref[0]).shape[1], rgb(ref[0]).shape[0]) if ref else (832, 480)
        cells = [np.hstack([source_frame(vid, i, size) for i in idx])]
        for _, root, _, _ in cols:
            fs = frames(root / f"edit{t}" / vid)
            if not fs:
                cells.append(None)
                continue
            cells.append(np.hstack([cv2.resize(rgb(fs[min(i, len(fs) - 1)]), size) for i in idx]))
        for c, im in enumerate(cells):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if im is None:
                ax.set_facecolor("#dddddd"); ax.text(0.5, 0.5, "not rendered", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            else:
                ax.imshow(im)
            if c == 0:
                ax.set_ylabel(f"e{t}\n{vid}", fontsize=6, rotation=0, ha="right", va="center")
            else:
                lab = labels[c - 1].get((t, vid))
                if lab:
                    ax.set_xlabel(f"bg LPIPS {lab[0]:.3f} · FiVE-Acc {lab[1]:.2f}", fontsize=6, labelpad=1)
            if r == 0:
                ax.set_title("source" if c == 0 else cols[c - 1][0], fontsize=8)
    fig.suptitle(f"R49 {'StreamEdit' if host == 'se' else 'Wan-Edit'} — each cell: middle | last frame", fontsize=10)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def wan_sigma(start: np.ndarray, n: int, shift: float) -> np.ndarray:
    t = (n - np.arange(n + 1)) / n                                    # step n (= never edited) -> sigma 0
    grid = shift * t / (1 + (shift - 1) * t)
    return grid[start]


def field_page(host: str, t: int, vid: str, field_root: Path, out: Path, n_frames: int = 6) -> None:
    if host == "se":
        z = np.load(field_root / "se_max" / f"edit{t}" / f"{vid}.npz")
        F = z["q"].shape[0]
        tn = t_next_schedule(int(z["step"]), float(z["flow_shift"]))
        half = float(budget_linear_tau(np.array([0.5]), tn, float(z["rho_min"]), float(z["rho_max"]))[0])
        controls = [(f"rho [value at q=0.5: {half:.2f}]", z["rho"].reshape(z["q"].shape), float(z["rho_min"]), float(z["rho_max"]))]
    else:
        zs = {m: np.load(field_root / f"wan_max_s{m}" / f"edit{t}" / f"{vid}.npz") for m in (9, 5)}
        z = zs[9]
        F = z["q"].shape[0]
        controls = []
        for m, zz in zs.items():
            n, sh = int(zz["wan_n"]), float(zz["wan_shift"])
            s_half = max(n - int(np.round(n * 0.5)), m)
            controls.append((f"start sigma, s_min {m} [q=0.5: step {s_half}, sigma {wan_sigma(np.array(s_half), n, sh):.3f}]",
                             wan_sigma(zz["start_step"], n, sh), 0.0, 1.0))
    fs = sorted({int(round(x)) for x in np.linspace(0, F - 1, n_frames)})
    rows = [("source", None, 0, 1), ("d DINO (0..0.55)", z["d_dino"], 0.0, float(z["div_max_dino"])),
            (f"d LPIPS (0..{float(z['div_max_lpips']):g})", z["d_lpips"], 0.0, float(z["div_max_lpips"])), ("div = max(m)", z["div"], 0, 1),
            ("S support", z["S"], 0, 1), ("q = S·div", z["q"], 0, 1), *controls]
    fig, axes = plt.subplots(len(rows), len(fs), figsize=(2.5 * len(fs), 1.6 * len(rows)), squeeze=False)
    for r, (name, arr, lo, hi) in enumerate(rows):
        for c, f in enumerate(fs):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if arr is None:
                ax.imshow(source_frame(vid, 0 if f == 0 else 4 * f, (832, 480)))
            else:
                im = ax.imshow(arr[f], cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
            if r == 0:
                ax.set_title(f"latent frame {f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
        if arr is not None:
            cb = fig.colorbar(im, ax=axes[r].tolist(), fraction=0.012, pad=0.01)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"R49 {host} / edit{t} / {vid}", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "r49_cases.json"))
    ap.add_argument("--field_root", default=str(D / "r49_field"))
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--out_dir", default=str(HERE / "figures" / "r49_grids"))
    a = ap.parse_args()
    cases = sorted({(int(c["edit_type"]), c["video_name"]) for c in json.loads(Path(a.cases).read_text())})
    scope, names, out = set(cases), R.clip_names(), Path(a.out_dir)
    n_fail = 0
    for host in ("se", "wan"):
        frame_grid(host, cases, Path(a.csv_dir), out / f"frames_{host}.png", scope, names)
        for t, vid in cases:
            try:
                field_page(host, t, vid, Path(a.field_root), out / "fields" / host / f"e{t}_{vid}.png")
            except Exception as e:                       # noqa: BLE001 -- report per page, keep going
                n_fail += 1
                print(f"[r49_grids] FAILED field page {host} {t}/{vid}: {type(e).__name__}: {e}", flush=True)
    print(f"[r49_grids] frames_se.png, frames_wan.png + {2 * len(cases) - n_fail} field pages -> {out}  failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
