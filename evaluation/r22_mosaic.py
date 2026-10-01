#!/usr/bin/env python3
"""R22 frame mosaics: what the video actually looks like across (frame, step).

The metric surfaces in ``r22_figures.py`` say *when* the numbers stop moving.
This says what the pixels are doing while they move. One page per (video, arm):

    rows = pixel frames with ``i % --frame_mod == 0``   (default 10 -> 0,10,20,...)
    cols = denoising steps ``--steps``                  (default 0 2 5 8 11 14)
    cell = that frame of that clip decoded at that step

Read straight off the step dump written by ``evaluation/r22_dump_steps.py`` --
no inference, no re-decode, nothing regenerated:

    {dump_root}/{arm}/step{jj}/edit{T}/{video}/{iiiii}.png

Frame files are 0-indexed, so ``i % 10 == 0`` selects ``00000.png``,
``00010.png``, ... and the row label is the frame index as it appears on disk.

Two details that are load-bearing rather than cosmetic
------------------------------------------------------
1. **``_e{T}`` in the filename.** ``0011_lucia`` appears under edit types 2 AND
   5 in ``cases.json``; a ``{video}.pdf`` name would silently overwrite one
   mosaic with the other. Same keying trap as ``r22_build_matrix.py``.
2. **Tiles are downsampled** to ``--tile_width`` (default 208 px = 832/4).
   Embedding 60 frames at native 832x480 makes a single page ~30 MB and the
   66-page set ~2 GB, which is not a figure set anyone can open.

Missing frames are reported and left blank rather than aborting the run, so one
bad clip cannot cost the other 65 pages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_STEPS = [0, 2, 5, 8, 11, 14]


def frame_path(dump_root: Path, arm: str, step: int, edit_type: int,
               video: str, frame: int) -> Path:
    return dump_root / arm / f"step{step:02d}" / f"edit{edit_type}" / video / f"{frame:05d}.png"


def load_tile(p: Path, tile_width: int) -> np.ndarray | None:
    """Load one frame downsampled to tile_width, or None if it is not there."""
    if not p.exists():
        return None
    with Image.open(p) as im:
        im = im.convert("RGB")
        w, h = im.size
        if tile_width and tile_width < w:
            im = im.resize((tile_width, max(1, round(h * tile_width / w))),
                           Image.LANCZOS)
        return np.asarray(im)


def mosaic_one(dump_root: Path, arm: str, video: str, edit_type: int,
               frames: list[int], steps: list[int], tile_width: int,
               outdir: Path) -> tuple[Path, int]:
    """One page for one (video, arm). Returns (path, n_missing)."""
    nrow, ncol = len(frames), len(steps)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(1.55 * ncol, 0.95 * nrow + 0.6),
                             squeeze=False)

    missing = 0
    for r, fi in enumerate(frames):
        for c, j in enumerate(steps):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            tile = load_tile(frame_path(dump_root, arm, j, edit_type, video, fi),
                             tile_width)
            if tile is None:
                missing += 1
                ax.set_facecolor("0.9")
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=6, transform=ax.transAxes)
            else:
                ax.imshow(tile)
            for s in ax.spines.values():
                s.set_linewidth(0.3)
            if r == 0:
                ax.set_title(f"$j$={j}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"$i$={fi}", fontsize=8)

    fig.suptitle(f"{video}  (edit {edit_type})  —  {arm}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    d = outdir / arm
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{video}_e{edit_type}.pdf"          # _e{T}: 0011_lucia is in e2 AND e5
    fig.savefig(p)
    plt.close(fig)
    return p, missing


def count_frames(dump_root: Path, arm: str, edit_type: int, video: str,
                 step: int) -> int:
    d = dump_root / arm / f"step{step:02d}" / f"edit{edit_type}" / video
    return len(list(d.glob("*.png"))) if d.is_dir() else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump_root", type=Path,
                    default=Path("/projects/dataggen/outputs/five_bench/r22_step_dump"))
    ap.add_argument("--cases_json", type=Path, default=Path("evaluation/cases.json"))
    ap.add_argument("--arms", nargs="*", default=None,
                    help="default: every arm directory under --dump_root")
    ap.add_argument("--frame_mod", type=int, default=10,
                    help="keep pixel frames with i %% frame_mod == 0 (default 10)")
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--tile_width", type=int, default=208,
                    help="downsample each frame to this width before embedding")
    ap.add_argument("--outdir", type=Path, default=Path("evaluation/figures/r22_mosaic"))
    ap.add_argument("--only_video", nargs="*", default=None)
    args = ap.parse_args()

    if not args.dump_root.is_dir():
        raise SystemExit(f"[r22_mosaic] dump root not found: {args.dump_root}")
    arms = args.arms or sorted(d.name for d in args.dump_root.iterdir() if d.is_dir())
    cases = json.loads(args.cases_json.read_text())
    if args.only_video:
        keep = set(args.only_video)
        cases = [c for c in cases if c["video_name"] in keep]

    print(f"[r22_mosaic] arms={arms} | clips={len(cases)} | steps={args.steps} | "
          f"frame_mod={args.frame_mod} | tile_width={args.tile_width}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_pages = n_missing = 0
    for arm in arms:
        for c in cases:
            video, et = c["video_name"], c["edit_type"]
            nb = count_frames(args.dump_root, arm, et, video, args.steps[0])
            if nb == 0:
                print(f"[r22_mosaic] SKIP {arm}/{video} (e{et}): no frames on disk")
                continue
            frames = list(range(0, nb, args.frame_mod))
            p, miss = mosaic_one(args.dump_root, arm, video, et, frames,
                                 args.steps, args.tile_width, args.outdir)
            n_pages += 1
            n_missing += miss
            flag = f"  ({miss} missing tiles)" if miss else ""
            print(f"[r22_mosaic] {p.relative_to(args.outdir)}  "
                  f"{len(frames)}x{len(args.steps)} of {nb}f{flag}")

    print(f"[r22_mosaic] wrote {n_pages} pages -> {args.outdir}/<arm>/<video>_e<T>.pdf")
    if n_missing:
        print(f"[r22_mosaic] WARNING: {n_missing} tiles were missing on disk")


if __name__ == "__main__":
    main()
