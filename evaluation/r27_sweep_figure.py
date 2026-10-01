#!/usr/bin/env python
"""R27 -- read the single-clip tau sweep on 0011_lucia_e5 as one grid.

Rows are the swept tau values plus the source and the Qwen anchor; columns are frames
sampled at fixed NORMALIZED times. The normalization is not cosmetic: the VAE changes
the frame count (80 source frames vs 69 rendered on this clip), so indexing both by
raw frame number would drift about 14% by the end of the shot and every row would show
a different moment. Sampling by t in [0, 1] keeps the columns comparable.

The anchor row is the Qwen first-frame edit -- the target the rollout is asked to
propagate. It occupies only the t = 0 column; the rest of that row is left blank
rather than tiled, because there is no anchor for later frames and tiling one would
imply a temporal signal that does not exist.

What to look for, in the order that decides the calibration:
  1. Does the dog appear AT ALL, at any tau? If not, the founding premise of R27 --
     that this edit fails because tau is too small -- is wrong, and no ceiling fixes it.
  2. If it appears, at which tau does it first hold across the whole clip? That tau is
     the calibration point: set D_hi so this case's D_norm = 0.784 maps to it.
  3. Do tau 50 and tau 100 differ? At --step 15 tau 50 still leaves W_src = 0.031 at
     the first blended step, so 50 may not be saturation. If they are indistinguishable
     the map's tau_max = 50 is a safe ceiling; if not, it is cutting into live range.
  4. Does the WOMAN survive as tau rises? Large tau releases the source early, so the
     failure mode at the top of the range is losing the unedited content -- which is
     the trade the whole adaptive-tau idea is trying to place.

Usage
-----
    python evaluation/r27_sweep_figure.py \\
        --sweep_root /projects/dataggen/outputs/five_bench/r27_tau_sweep \\
        -o evaluation/figures/r27_tau_sweep_lucia_e5.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def frame_paths(d: Path) -> List[Path]:
    """Sorted frame files in a render dir, ignoring evaluate.py's *_resize leftovers."""
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))


def sample_by_time(paths: Sequence[Path], ts: Sequence[float]) -> List[Path]:
    """Pick the frame nearest each normalized time t in [0, 1]."""
    n = len(paths)
    return [paths[min(n - 1, max(0, int(round(t * (n - 1)))))] for t in ts]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r27_tau_sweep"))
    p.add_argument("--video_name", type=str, default="0011_lucia")
    p.add_argument("--edit_type", type=int, default=5)
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--anchor_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/anchors"))
    p.add_argument("--n_cols", type=int, default=6,
                   help="Frames per row, sampled at evenly spaced normalized times.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/figures/r27_tau_sweep_lucia_e5.png"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ts = np.linspace(0.0, 1.0, args.n_cols)

    # ---- collect the rows ---------------------------------------------------------
    rows: List[Tuple[str, List[Optional[Path]]]] = []

    src_dir = (args.data_root / "images" / args.video_name).expanduser()
    if not src_dir.is_dir():
        raise SystemExit(f"source frames not found: {src_dir}")
    src = frame_paths(src_dir)
    rows.append((f"source\n({len(src)} frames)", list(sample_by_time(src, ts))))

    anchor = (args.anchor_root / f"edit{args.edit_type}" /
              f"{args.video_name}.png").expanduser()
    if anchor.is_file():
        # t = 0 only; the remaining columns stay blank on purpose (see the docstring).
        rows.append(("Qwen anchor\n(first frame)",
                     [anchor] + [None] * (args.n_cols - 1)))

    # Numeric sort, so tau=100 does not land between 10 and 20.
    taus: List[Tuple[float, Path]] = []
    for d in args.sweep_root.glob(f"r27_sweep_tau*/edit{args.edit_type}/{args.video_name}"):
        m = re.search(r"r27_sweep_tau([0-9.]+)", str(d))
        if m and d.is_dir():
            taus.append((float(m.group(1)), d))
    if not taus:
        raise SystemExit(f"no renders under {args.sweep_root}")
    for tau, d in sorted(taus):
        fp = frame_paths(d)
        tag = "  = Eq. 4" if abs(tau - 2.0) < 1e-6 else ""
        rows.append((f"tau = {tau:g}{tag}\n({len(fp)} frames)",
                     list(sample_by_time(fp, ts))))

    # ---- draw ---------------------------------------------------------------------
    nr, nc = len(rows), args.n_cols
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 1.7 * nr))
    axes = np.atleast_2d(axes)
    for r, (label, paths) in enumerate(rows):
        for c in range(nc):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.75")
            if paths[c] is not None:
                ax.imshow(np.asarray(Image.open(paths[c]).convert("RGB")))
            else:
                ax.set_facecolor("0.96")
            if r == 0:
                ax.set_title(f"t = {ts[c]:.1f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=8)

    fig.suptitle(
        f"R27 tau sweep -- {args.video_name}_e{args.edit_type}  "
        f"(add a dog;  D_norm = 0.784, rank 20/22)\n"
        f"columns sampled at equal NORMALIZED time (source and renders differ in frame count)",
        fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}  ({nr} rows x {nc} cols)", flush=True)
    for label, _ in rows:
        print(f"  row: {label.splitlines()[0]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
