#!/usr/bin/env python
"""R30 per-video diagnostic grids: one tall grid per clip, all 8 arms side by side.

Rows (17 total, fixed):

    1.    source video
    2-9.  each of the 8 arms' own rendered output, at ITS OWN routed b for this clip
          (r30_arms/{arm}/edit{T}/{video}), labelled with `m` (this arm's divergence `d`
          for this clip, MIN-MAX NORMALIZED against that arm's own dataset floor/ceiling
          -- the same `m` column r30_b_map_{arm}.csv already computed, so it lands in
          [0, 1] and is comparable ACROSS arms despite their raw `d` living in unrelated
          units: LPIPS distance, cosine distance, angular degrees, ...) and `b` (the
          inferred blend rate that `m` was mapped to, budget-linear on the real sampler
          grid -- see r30_b_map.py)
    10-17. the 8 constant-b SPATIAL reference videos (R26's taubg0_taufg{b}_vp family),
          same clip, same mechanism, for a fixed visual anchor every arm's row can be
          read against.

Columns: F frames, evenly sampled across each row's OWN length (rendered arms run a few
frames shorter than the source mp4, a fixed consequence of the model's causal windowing;
sampling by FRACTION of each row's own length keeps columns roughly time-aligned across
rows of different length).

Unlike r30_arm_grids.py (one page per (arm, clip), with the arm's raw divergence map
recomputed on GPU), this script needs no model inference at all: `m` and `b` are read
straight from r30_b_map_{arm}.csv, and every frame is an already-rendered PNG or the
source mp4. Local, no GPU.

Usage
-----
    python evaluation/r30_video_grids.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --arms_root /projects/dataggen/outputs/five_bench/r30_arms \\
        --const_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \\
        --b_map_dir evaluation/csv \\
        --n_frames 6 \\
        --out_dir evaluation/figures/r30_video_grids
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ARMS: Tuple[str, ...] = ("lpips", "dino_cls", "dino_patch", "clip_image",
                         "clip_prompt", "depth", "normals", "selfsim")
CONST_BS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 20, 50)


# --------------------------------------------------------------------------------------
# frame sources -- same conventions as r30_arm_grids.py, kept independent here so this
# script has no GPU-only import (lpips/diffusers) in its chain.
# --------------------------------------------------------------------------------------
def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_video_frame(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} of {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def png_dir_frame_count(d: Path) -> int:
    return len(list(d.glob("[0-9]" * 5 + ".png")))


def read_png_frame(d: Path, idx: int) -> np.ndarray:
    p = d / f"{idx:05d}.png"
    if not p.exists():
        raise FileNotFoundError(str(p))
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


class VideoRow:
    """One grid row's frame source: either a source .mp4 or a directory of PNG frames."""

    def __init__(self, kind: str, path: Path):
        self.kind = kind
        self.path = path
        self.n = video_frame_count(path) if kind == "mp4" else png_dir_frame_count(path)

    def frame(self, frac: float) -> Optional[np.ndarray]:
        if self.n == 0:
            return None
        idx = round(frac * (self.n - 1))
        try:
            if self.kind == "mp4":
                return read_video_frame(self.path, idx)
            return read_png_frame(self.path, idx)
        except (FileNotFoundError, RuntimeError):
            return None


# --------------------------------------------------------------------------------------
def load_b_maps(b_map_dir: Path, arms: Sequence[str]
               ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """{arm: {case_id: (m, tau)}} -- m is the min-max NORMALIZED divergence (r30_b_map.py's
    own per-arm floor/ceiling), tau is the inferred b it was mapped to."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for arm in arms:
        path = b_map_dir / f"r30_b_map_{arm}.csv"
        if not path.exists():
            print(f"[r30_video_grids] WARNING: {path} not found, arm '{arm}' will read "
                 f"as missing for every clip", flush=True)
            continue
        by_case: Dict[str, Tuple[float, float]] = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                by_case[r["case_id"]] = (float(r["m"]), float(r["tau"]))
        out[arm] = by_case
    return out


# --------------------------------------------------------------------------------------
def build_page(out_path: Path, case: Dict[str, object], n_frames: int, tile_width: int,
               source_video: Path, arm_rows: List[Tuple[str, Optional[VideoRow]]],
               const_rows: List[Tuple[str, Optional[VideoRow]]],
               title: Optional[str] = None) -> Optional[str]:
    case_id = str(case["case_id"])
    rows: List[Tuple[str, Optional[VideoRow]]] = (
        [("source", VideoRow("mp4", source_video))] + arm_rows + const_rows)

    missing = [label for label, src in rows if src is None]
    if missing:
        print(f"[r30_video_grids] {case_id}: missing frame source(s) for {missing}",
             flush=True)

    fracs = [i / (n_frames - 1) if n_frames > 1 else 0.0 for i in range(n_frames)]
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, n_frames,
                             figsize=(1.6 * n_frames, 0.95 * nrow + 0.6),
                             squeeze=False)

    for r, (label, src) in enumerate(rows):
        for c, frac in enumerate(fracs):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            frame = src.frame(frac) if src is not None else None
            if frame is not None:
                if tile_width and tile_width < frame.shape[1]:
                    h = max(1, round(frame.shape[0] * tile_width / frame.shape[1]))
                    frame = cv2.resize(frame, (tile_width, h),
                                      interpolation=cv2.INTER_AREA)
                ax.imshow(frame)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7,
                       transform=ax.transAxes)
            if c == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")

    fig.suptitle(title if title is not None
                 else f"{case_id}  (edit_type {case['edit_type']})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return None if not missing else f"{case_id}: missing {missing}"


# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    if args.only_cases:
        wanted = set(args.only_cases.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
    if not cases:
        raise SystemExit("[r30_video_grids] no cases selected")

    data_root = args.data_root.expanduser()
    b_maps = load_b_maps(args.b_map_dir, ARMS)

    # Blind re-rating support: {case_id: label}. When present, each page is NAMED and
    # TITLED by its label instead of its case_id, and the per-arm m/b numbers are dropped
    # from the row labels -- both are per-clip unique, so either one alone would let a
    # rater recognise a clip they have already rated (or look it up), which is exactly
    # what the blinding exists to prevent. Everything else about the page is unchanged,
    # so the stimulus stays as close as possible to the un-blinded version.
    label_map: Dict[str, str] = {}
    if args.label_map is not None:
        with open(args.label_map) as fh:
            label_map = {r["case_id"]: r["label"] for r in csv.DictReader(fh)}
        cases = [c for c in cases if c["case_id"] in label_map]

    n_written = 0
    failures: List[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        video_name = str(case["video_name"])
        edit_type = int(case["edit_type"])
        src_video = data_root / str(case["src_video"])

        arm_rows: List[Tuple[str, Optional[VideoRow]]] = []
        for arm in ARMS:
            m_tau = b_maps.get(arm, {}).get(case_id)
            arm_dir = args.arms_root / arm / f"edit{edit_type}" / video_name
            row_src = VideoRow("png_dir", arm_dir) if arm_dir.exists() else None
            if args.anonymize:
                label = arm
            elif m_tau is not None:
                m, tau = m_tau
                label = f"{arm}  m={m:.3f} b={tau:.2f}"
            else:
                label = f"{arm}  m=n/a b=n/a"
            arm_rows.append((label, row_src))

        const_rows: List[Tuple[str, Optional[VideoRow]]] = []
        for b in CONST_BS:
            const_dir = args.const_root / f"taubg0_taufg{b}_vp" / f"edit{edit_type}" / video_name
            const_rows.append((f"const b={b}",
                              VideoRow("png_dir", const_dir) if const_dir.exists() else None))

        label = label_map.get(case_id)
        out_path = args.out_dir / f"{label or case_id}.png"
        err = build_page(out_path, case, args.n_frames, args.tile_width, src_video,
                         arm_rows, const_rows, title=label)
        if err:
            failures.append(err)
        n_written += 1
        print(f"[r30_video_grids] wrote {out_path}", flush=True)

    print(f"\n[r30_video_grids] wrote {n_written} page(s) under {args.out_dir}, "
         f"{len(failures)} with missing row(s)", flush=True)
    return 0


# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--only_cases", type=str, default=None,
                   help="Comma-separated case_id subset, for a quick smoke test.")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--arms_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r30_arms"),
                   help="{arms_root}/{arm}/edit{T}/{video}/{iiiii}.png")
    p.add_argument("--const_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_spatial_tau"),
                   help="{const_root}/taubg0_taufg{b}_vp/edit{T}/{video}/{iiiii}.png")
    p.add_argument("--b_map_dir", type=Path, default=Path("evaluation/csv"),
                   help="Where r30_b_map_{arm}.csv live (m, tau per case_id).")
    p.add_argument("--label_map", type=Path, default=None,
                   help="Optional CSV (case_id,label) for BLIND rating: each page is "
                        "named and titled by its label instead of its case_id, and only "
                        "the listed cases are rendered. Pair with --anonymize.")
    p.add_argument("--anonymize", action="store_true",
                   help="Drop the per-arm m/b numbers from the row labels (they are "
                        "per-clip unique, so they identify a clip on their own).")
    p.add_argument("--n_frames", type=int, default=6, help="Grid columns (F).")
    p.add_argument("--tile_width", type=int, default=150)
    p.add_argument("--out_dir", type=Path, default=Path("evaluation/figures/r30_video_grids"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
