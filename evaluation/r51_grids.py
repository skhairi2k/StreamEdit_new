"""R51 qualitative frame grid (StreamEdit): the official baseline, R49's se_max (A0) and the four demand-ablation arms.

Rows = the 18 clips of r49_cases.json (fixed, not cherry-picked). Columns: source | official (rho 2, N 15) | A0 full | A1 no divergence |
A2 DINO only | A3 LPIPS only | A4 no support. Each cell = middle | last frame, labelled with the clip's bg LPIPS / FiVE-Acc taken from the
419-pair per-clip CSVs. Reads renders and CSVs only; runs no model. The official render lives on /store, so run this where it is
mounted (gpu-gw, node39): an unreadable column FAILS the script instead of drawing a grey "not rendered" column.

Output evaluation/figures/r51_grids/frames_se.png
usage: python evaluation/r51_grids.py [--cases FILE] [--csv_dir DIR] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_score as R  # noqa: E402
import r49_grids as G49  # noqa: E402

D = G49.D
# column -> (title, render root with edit{T}/{video}/*.png, harness stem, five_acc stem) -- stems of the 419-pair per-clip CSVs
COLUMNS = [
    ("official: rho 2, N 15", D / "r36_rho_sweep/r36_rho2_vp", "r36_rho2_vp", "r36_fiveacc_rho2_vp"),
    ("A0 full (R49 se_max)", D / "r49_se_arms/r49_max_full/step14", "r49_se_max_full", "r49_fiveacc_se_max_full"),
    ("A1 no divergence", D / "r51_se_arms/r51_nodiv_full/step14", "r51_se_nodiv_full", "r51_fiveacc_se_nodiv_full"),
    ("A2 DINO only", D / "r51_se_arms/r51_dino_full/step14", "r51_se_dino_full", "r51_fiveacc_se_dino_full"),
    ("A3 LPIPS only", D / "r51_se_arms/r51_lpips_full/step14", "r51_se_lpips_full", "r51_fiveacc_se_lpips_full"),
    ("A4 no support", D / "r51_se_arms/r51_nosupp_full/step14", "r51_se_nosupp_full", "r51_fiveacc_se_nosupp_full"),
]


def preflight(cases, columns) -> int:
    """Every column must have a render for every case clip; fail loudly otherwise."""
    bad = 0
    for title, root, _h, _f in columns:
        missing = [(t, v) for t, v in cases if not G49.frames(root / f"edit{t}" / v)]
        if missing:
            bad += 1
            print(f"[r51_grids] FATAL column '{title}': no frames for {len(missing)}/{len(cases)} clips under {root} "
                  f"(e.g. {missing[0]}) -- render not finished, or /store not mounted on {socket.gethostname()}")
    return bad


def frame_grid(cases, columns, csv_dir: Path, out: Path, scope, names) -> None:
    labels = [G49.per_clip(csv_dir, h, f, scope, names) for _t, _r, h, f in columns]
    n_lab = [len(x) for x in labels]
    if any(n != len(cases) for n in n_lab):
        print(f"[r51_grids] WARNING: per-clip labels found for {n_lab} of {len(cases)} clips per column (cells without one stay unlabelled)")
    fig, axes = plt.subplots(len(cases), len(columns) + 1, figsize=(4.2 * (len(columns) + 1), 1.3 * len(cases)), squeeze=False)
    for r, (t, vid) in enumerate(cases):
        ref = G49.frames(columns[0][1] / f"edit{t}" / vid)              # the official render fixes the frame indices of the row
        n = len(ref)
        idx = (n // 2, n - 1)
        size = (G49.rgb(ref[0]).shape[1], G49.rgb(ref[0]).shape[0])
        cells = [np.hstack([G49.source_frame(vid, i, size) for i in idx])]
        for _title, root, _h, _f in columns:
            fs = G49.frames(root / f"edit{t}" / vid)
            cells.append(np.hstack([cv2.resize(G49.rgb(fs[min(i, len(fs) - 1)]), size) for i in idx]))
        for c, im in enumerate(cells):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(im)
            if c == 0:
                ax.set_ylabel(f"e{t}\n{vid}", fontsize=6, rotation=0, ha="right", va="center")
            else:
                lab = labels[c - 1].get((t, vid))
                if lab:
                    ax.set_xlabel(f"bg LPIPS {lab[0]:.3f} · FiVE-Acc {lab[1]:.2f}", fontsize=6, labelpad=1)
            if r == 0:
                ax.set_title("source" if c == 0 else columns[c - 1][0], fontsize=8)
    fig.suptitle("R51 StreamEdit demand ablation — each cell: middle | last frame", fontsize=10)
    fig.tight_layout(pad=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "r49_cases.json"))
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--out_dir", default=str(HERE / "figures" / "r51_grids"))
    a = ap.parse_args()
    cases = sorted({(int(c["edit_type"]), c["video_name"]) for c in json.loads(Path(a.cases).read_text())})
    if preflight(cases, COLUMNS):
        print("[r51_grids] failures: 1")
        return 1
    out = Path(a.out_dir) / "frames_se.png"
    frame_grid(cases, COLUMNS, Path(a.csv_dir), out, set(cases), R.clip_names())
    print(f"[r51_grids] wrote {out} ({len(cases)} clips x {len(COLUMNS) + 1} columns)  failures: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
