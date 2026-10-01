#!/usr/bin/env python
"""R34 -- per-clip qualitative grid showing the FIRST CHUNK ONLY: pixel frames 0-8.

One PNG per case, ``edit{T}_{name}_chunk1.png``, 6 rows x 9 columns:

    source / baseline (uniform b=2) / r31_lpips / r31_dino_patch / r31_normals / r31_latent

WHY NOT A FLAG ON r31_stage3_grids.py
--------------------------------------
That script's ``build_arm_grid`` bakes in three assumptions R34 breaks at once: it picks
columns with ``evenly_spaced(n_have, k)`` across the WHOLE clip, it iterates R31's four
arms only, and its ``load_arm_frames`` hardcodes the ``r31_{arm}/step14/edit{T}/{name}``
tree. R26's arms sit one level shallower (``{method}/edit{T}/{name}``), and R34 needs a
fixed 0-8 window plus the R26 baseline row. Threading three flags through that function
would leave it harder to read than either script alone.

THE WINDOW. ``num_frame_per_block: 3`` (configs/self_forcing_dmd.yaml) means the first
rollout chunk is 3 latent frames = pixel frames 0-8 (1 + 4 + 4). Clip LENGTH varies --
``find_closest_num_frame(x, a=4, b=3)`` returns ``12m - 3``, so an 80-frame source
renders 69 pixel frames, not 81 -- but the first chunk is 9 pixel frames for EVERY clip.
There is deliberately no subsampling call anywhere in this file: ``idx = range(9)``.

⚠️ FRAME INDICES ARE POSITIONAL, NOT FILENAME NUMBERS. Source frames on disk are
1-indexed (``images/{name}/00001.jpg``) while renders are 0-indexed
(``.../00000.png``), and ``evaluate.py`` pairs them by POSITION after sorting, never by
name. This script uses the same positional convention, so column ``i`` here is the same
pair that produced ``frame_idx == i`` in R34's per-frame CSVs. Sourcing the rows from
``images/`` rather than ``videos/{name}.mp4`` (which r31_stage3_grids.py uses) is the
same choice for the same reason: ``images/`` is what the metrics were computed from.

Local, no GPU, no model -- every frame already exists on disk under ~/Data.

Usage
-----
    python evaluation/r34_chunk1_grids.py \\
        --cases evaluation/cases.json \\
        --r26_root ~/Data/dataggen/outputs/five_bench/r26_spatial_tau \\
        --r31_root ~/Data/dataggen/outputs/five_bench/r31_arms \\
        --out_dir evaluation/figures/r34_chunk1_grids
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

CHUNK_FRAMES = 9          # 3 latent frames = 1 + 4 + 4 pixel frames
LATENT_PER_CHUNK = 3

# (row label, source kind, method). kind None = the benchmark source frames.
ROWS: Sequence[Tuple[str, Optional[str], Optional[str]]] = (
    ("source", None, None),
    ("baseline\n(uniform b=2)", "r26", "taubg2_taufg2_vp"),
    ("r31_lpips", "r31", "lpips"),
    ("r31_dino_patch", "r31", "dino_patch"),
    ("r31_normals", "r31", "normals"),
    ("r31_latent", "r31", "latent"),
)

IMG_EXT = {".png", ".jpg", ".jpeg"}


def list_frames(d: Path) -> List[Path]:
    """Sorted image paths — the same ordering evaluate.py's positional zip relies on."""
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.suffix.lower() in IMG_EXT),
                  key=lambda p: p.name)


def frames_dir(kind: str, method: str, t: int, name: str,
               r26_root: Path, r31_root: Path) -> Path:
    """The two render trees differ in depth; one formula cannot express both."""
    if kind == "r26":
        return r26_root / method / f"edit{t}" / name
    return r31_root / f"r31_{method}" / "step14" / f"edit{t}" / name


def n_chunks(n_pixel: int) -> Optional[int]:
    """Chunk count for a clip of `n_pixel` frames, or None if it is off the latent grid."""
    if n_pixel < 1 or (n_pixel - 1) % 4:
        return None
    n_latent = (n_pixel - 1) // 4 + 1
    return n_latent // LATENT_PER_CHUNK if n_latent % LATENT_PER_CHUNK == 0 else None


def build_grid(case: dict, data_root: Path, r26_root: Path, r31_root: Path,
               out_dir: Path, dpi: int) -> Optional[Path]:
    name, t = case["video_name"], int(case["edit_type"])
    case_id = case["case_id"]

    rows: List[Tuple[str, List[Path]]] = []
    for label, kind, method in ROWS:
        d = (data_root / "images" / name if kind is None
             else frames_dir(kind, method, t, name, r26_root, r31_root))
        paths = list_frames(d)
        # Fail the clip rather than padding: a row that repeats its last frame would
        # read as a static edit, which is exactly the failure mode these grids exist
        # to reveal.
        if len(paths) < CHUNK_FRAMES:
            print(f"[r34_chunk1_grids] SKIP {case_id}: row {label.splitlines()[0]!r} has "
                  f"{len(paths)} frames (< {CHUNK_FRAMES}) under {d}")
            return None
        rows.append((label, paths))

    n_rendered = len(rows[1][1])
    nch = n_chunks(n_rendered)
    chunk_note = (f"chunk 1 of {nch}" if nch
                  else f"chunk 1; {n_rendered} rendered frames are OFF the latent grid")

    idx = list(range(CHUNK_FRAMES))          # the whole point: no subsampling
    # Row height follows the frame aspect ratio. These clips are 16:9, so square cells
    # would put ~40% dead space between every row and make a 6x9 page hard to scan.
    size = Image.open(rows[0][1][0]).size
    col_w = 1.5
    row_h = col_w * size[1] / size[0]
    fig, axes = plt.subplots(len(rows), CHUNK_FRAMES,
                             figsize=(col_w * CHUNK_FRAMES, row_h * len(rows)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    for r, (label, paths) in enumerate(rows):
        for j, i in enumerate(idx):
            im = Image.open(paths[i]).convert("RGB")
            if im.size != size:
                im = im.resize(size)
            axes[r][j].imshow(im)
            if r == 0:
                axes[r][j].set_title(f"frame {i}", fontsize=6.5, pad=2)
        axes[r][0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center",
                              labelpad=34)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.01,
                        wspace=0.02, hspace=0.06)
    fig.suptitle(
        f"R34 {case_id} (edit{t}) — first chunk, pixel frames 0-8 ({chunk_note}); "
        f"{n_rendered} frames rendered. Indices are POSITIONAL, matching "
        f"r34_*_per_frame.csv",
        fontsize=8, y=0.995)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"edit{t}_{name}_chunk1.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    data_root = args.data_root.expanduser()
    r26_root = args.r26_root.expanduser()
    r31_root = args.r31_root.expanduser()
    out_dir = args.out_dir.expanduser()

    for root in (data_root, r26_root, r31_root):
        if not root.is_dir():
            raise SystemExit(f"[r34_chunk1_grids] root does not exist: {root}")

    n_ok, failed = 0, []
    for c in cases:
        out = build_grid(c, data_root, r26_root, r31_root, out_dir, args.dpi)
        if out is None:
            failed.append(c["case_id"])
            continue
        n_ok += 1
        print(f"[r34_chunk1_grids] {c['case_id']:<22} -> {out.name}", flush=True)

    print(f"\n[r34_chunk1_grids] {n_ok}/{len(cases)} grids -> {out_dir}")
    if failed:
        print(f"[r34_chunk1_grids] FAILED {len(failed)}: {', '.join(failed)}")
        return 1
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--r26_root", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/r26_spatial_tau"),
                   help="~/Data, never /projects -- see R33's mount-regression history.")
    p.add_argument("--r31_root", type=Path,
                   default=Path("~/Data/dataggen/outputs/five_bench/r31_arms"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("evaluation/figures/r34_chunk1_grids"))
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
