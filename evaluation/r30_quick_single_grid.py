#!/usr/bin/env python
"""Quick one-off grid for a single clip: source, grounding mask overlay, one R26
constant-rho baseline, and a subset of R30 divergence-routed arms. No GPU needed (the
mask row is the CPU-only overlay from r26_grid_figure.py's draw_mask_sanity, not the
VAE-decode row).

Usage
-----
    python evaluation/r30_quick_single_grid.py \\
        --case_id 0011_lucia_e2 --baseline_b 2 --arms lpips,normals,dino_patch \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --out evaluation/figures/r30_quick_0011_lucia_e2.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r26_grid_figure import load_mask, latent_grid, pixel_to_latent, frame_paths  # noqa: E402
from r30_video_grids import VideoRow  # noqa: E402


def mask_overlay_frame(src_path: str, mask: np.ndarray, pixel_idx: int,
                       n_pixel_frames: int) -> np.ndarray:
    """Source frame with the union grounding mask ($M_f$) tinted orange on top."""
    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    gh, gw = latent_grid(mask.shape[1], w, h)
    lf = pixel_to_latent(pixel_idx, n_pixel_frames, mask.shape[0])
    m = mask[lf].reshape(gh, gw)
    big = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
        img.size, Image.NEAREST)) > 127
    arr = np.asarray(img).astype(np.float32)
    tint = np.array([255.0, 38.0, 0.0])
    alpha = 0.45
    out = arr.copy()
    out[big] = (1 - alpha) * arr[big] + alpha * tint
    return out.astype(np.uint8)


def load_routed_rho(b_map_dir: Path, arm: str, case_id: str) -> Optional[float]:
    """Per-arm routed rho (r30_b_map.py's `tau` column) for one case_id, or None if
    unavailable -- the row label degrades gracefully rather than failing the figure."""
    path = b_map_dir / f"r30_b_map_{arm}.csv"
    if not path.exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["case_id"] == case_id:
                return float(r["tau"])
    return None


def build(case_id: str, baseline_b: int, arms: Sequence[str], n_frames: int,
         cases_path: Path, data_root: Path, masks_dir: Path, const_root: Path,
         arms_root: Path, out_path: Path, mask_path_override: Optional[Path] = None,
         baseline_dir_override: Optional[Path] = None, flat: bool = False,
         b_map_dir: Optional[Path] = None) -> None:
    cases = json.loads(cases_path.read_text())
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if case is None:
        raise SystemExit(f"case_id {case_id!r} not found in {cases_path}")
    edit_type = int(case["edit_type"])
    video_name = str(case["video_name"])
    print(f"[quick_grid] {case_id}: edit_type={edit_type} video_name={video_name} "
         f"trg_word={case.get('trg_word')!r}", flush=True)

    src_dir = data_root / "images" / video_name
    src_paths = frame_paths(str(src_dir))
    if not src_paths:
        raise SystemExit(f"no source frames under {src_dir}")
    n_src = len(src_paths)
    fracs = [i / (n_frames - 1) if n_frames > 1 else 0.0 for i in range(n_frames)]
    src_idx = [round(f * (n_src - 1)) for f in fracs]

    mask_path = mask_path_override if mask_path_override is not None \
        else masks_dir / f"edit{edit_type}" / f"{video_name}.npz"
    mask = load_mask(str(mask_path)) if mask_path.exists() else None
    if mask is None:
        print(f"[quick_grid] WARNING: no mask at {mask_path}", flush=True)

    rows: List[Tuple[str, object]] = []
    rows.append(("source", ("src", None)))

    baseline_dir = baseline_dir_override if baseline_dir_override is not None \
        else const_root / f"taubg0_taufg{baseline_b}_vp" / f"edit{edit_type}" / video_name
    rows.append((f"baseline\n$\\rho$={baseline_b}",
                ("row", VideoRow("png_dir", baseline_dir) if baseline_dir.exists() else None)))

    for arm in arms:
        arm_dir = (arms_root / arm / video_name) if flat \
            else (arms_root / arm / f"edit{edit_type}" / video_name)
        rho = load_routed_rho(b_map_dir, arm, case_id) if b_map_dir is not None else None
        label = f"{arm}\n$\\rho$={rho:.1f}" if rho is not None else arm
        rows.append((label,
                    ("row", VideoRow("png_dir", arm_dir) if arm_dir.exists() else None)))

    # mask row inserted after source, before baseline (matches r26's grid ordering intent:
    # source -> what-the-model-sees -> renders)
    rows.insert(1, ("$M_f$ mask\n(grounding)", ("mask", None)))

    nrow = len(rows)
    fig, axes = plt.subplots(nrow, n_frames, figsize=(1.9 * n_frames, 1.5 * nrow + 0.6),
                             squeeze=False)
    missing = []
    for r, (label, (kind, src)) in enumerate(rows):
        for c, frac in enumerate(fracs):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            frame = None
            if kind == "src":
                frame = np.asarray(Image.open(src_paths[src_idx[c]]).convert("RGB"))
            elif kind == "mask":
                if mask is not None:
                    frame = mask_overlay_frame(src_paths[src_idx[c]], mask, src_idx[c], n_src)
            elif kind == "row":
                if src is not None:
                    frame = src.frame(frac)
            if frame is not None:
                ax.imshow(frame)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7,
                       transform=ax.transAxes)
                if kind == "row":
                    missing.append(label.replace("\n", " "))
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            if r == 0:
                ax.set_title(f"t={frac:.2f}", fontsize=7)

    fig.suptitle(f"{case_id}  (edit_type {edit_type}, {video_name})  "
                f"'{case.get('src_word')}' -> '{case.get('trg_word')}'", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    uniq_missing = sorted(set(missing))
    print(f"[quick_grid] wrote {out_path}"
         + (f"  (missing row(s): {uniq_missing})" if uniq_missing else ""), flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case_id", default="0011_lucia_e2")
    p.add_argument("--baseline_b", type=int, default=2)
    p.add_argument("--arms", default="lpips,normals,dino_patch")
    p.add_argument("--n_frames", type=int, default=6)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--masks_dir", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_masks"))
    p.add_argument("--const_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_spatial_tau"))
    p.add_argument("--arms_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r30_arms"))
    p.add_argument("--out", type=Path,
                   default=Path("evaluation/figures/r30_quick_grid.png"))
    p.add_argument("--mask_path", type=Path, default=None,
                   help="direct path to the .npz mask, overrides --masks_dir/edit{T}/{video}.npz")
    p.add_argument("--baseline_dir", type=Path, default=None,
                   help="direct path to the baseline frame dir, overrides the "
                        "--const_root/taubg0_taufgB_vp/edit{T}/{video} convention")
    p.add_argument("--flat", action="store_true",
                   help="arm dirs are --arms_root/{arm}/{video} with no edit{T} subfolder")
    p.add_argument("--b_map_dir", type=Path, default=Path("evaluation/csv"),
                   help="where r30_b_map_{arm}.csv live, for each arm's routed rho label; "
                        "pass a non-existent dir to omit the rho label entirely")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    build(args.case_id, args.baseline_b, args.arms.split(","), args.n_frames,
         args.cases, args.data_root.expanduser(), args.masks_dir, args.const_root,
         args.arms_root, args.out, mask_path_override=args.mask_path,
         baseline_dir_override=args.baseline_dir, flat=args.flat, b_map_dir=args.b_map_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
