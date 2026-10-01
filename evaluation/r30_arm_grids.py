#!/usr/bin/env python
"""R30 diagnostic grids: for each arm, one page per clip showing what the divergence
measure actually saw next to what it actually routed.

Rows (10, or 11 for a SPATIAL arm):

    1. source video
    2. [SPATIAL arms only] the raw divergence map (lpips/dino_patch/depth/normals) --
       time-invariant (divergence is measured once, from the source-vs-anchor frame
       pair, never per output frame), so the same map image is repeated in every column
    3. this arm's own rendered output, at ITS OWN routed b for this clip (r30_arms/{arm})
    4-11. the 8 constant-b SPATIAL reference videos (R26's taubg0_taufg{b}_vp family),
       for a same-clip, same-mechanism visual anchor -- "is this arm's routed b behaving
       like the constant it's closest to, or like something else?"

Columns: F frames, evenly sampled across each row's OWN length (rendered arms run to a
few frames shorter than the source mp4, a fixed consequence of the model's causal
windowing -- sampling by FRACTION of each row's own length keeps the columns roughly
time-aligned across rows of different length, rather than by matching absolute frame
index).

This is a visual tool for the question the run's own diagnostics can't fully answer on
their own: for an arm whose divergence-vs-oracle rank correlation is mediocre, is the
divergence MEASURE looking at the wrong thing (the map lights up the wrong region, or
doesn't distinguish a barely-edited clip from a heavily-edited one), or is a reasonable
measure being MIS-MAPPED to b (the map looks right, but the rendered output over- or
under-shoots what the constant-b references show for that amount of divergence)?

Non-spatial arms (dino_cls, clip_image, clip_prompt, selfsim) have no map to draw --
their divergence is a single global number by construction, not a region -- so they get
10 rows, not 11; the number itself is printed in the source row's caption instead.

Model loading is real: the 4 spatial arms' divergence maps are RECOMPUTED here (they are
never persisted by r30_divergence.py, only their scalar reductions are), by importing and
calling that script's own measure_case()/model loaders directly -- same code path, same
seed, so the map matches the number already in r30_divergence.csv bit for bit. This is
therefore a GPU script (LPIPS + DINOv2 + Depth-Anything + Marigold Normals), unlike every
other r30_*.py script, which is local/no-GPU by design.

Usage
-----
    python evaluation/r30_arm_grids.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --anchor_root /projects/dataggen/outputs/five_bench/anchors \\
        --mask_dir /projects/dataggen/outputs/five_bench/r26_masks \\
        --split_mask_dir /projects/dataggen/outputs/five_bench/r30_split_masks \\
        --arms_root /projects/dataggen/outputs/five_bench/r30_arms \\
        --const_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \\
        --divergence_csv evaluation/csv/r30_divergence.csv \\
        --fiveacc_arms_csv evaluation/csv/r30_fiveacc_arms.csv \\
        --n_frames 6 \\
        --out_dir evaluation/figures/r30_arm_grids
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r30_divergence as rd  # noqa: E402

CONST_BS: Tuple[int, ...] = (2, 3, 4, 6, 8, 10, 20, 50)


# --------------------------------------------------------------------------------------
# frame sources
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


def sample_indices(n_total: int, n_frames: int) -> List[int]:
    """n_frames evenly spaced indices in [0, n_total - 1], inclusive of both ends."""
    if n_total <= 1:
        return [0] * n_frames
    return [round(i * (n_total - 1) / (n_frames - 1)) for i in range(n_frames)]


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
# divergence maps (spatial arms only) -- recomputed via r30_divergence.py's own code
# --------------------------------------------------------------------------------------
def load_spatial_models(device: str, args: argparse.Namespace) -> Dict[str, object]:
    models: Dict[str, object] = {
        "lpips": rd.load_lpips(device),
        "dinov2": rd.load_dinov2(args.dinov2_model_id, device),
        "depth": rd.load_depth(args.depth_model_id, device),
        "normals": rd.load_normals(args.normals_model_id, device, torch.float32),
    }
    return models


def load_csv_by_case(path: Path, key: str = "case_id") -> Dict[str, Dict[str, str]]:
    with open(path) as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def load_arms_by_case_arm(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    with open(path) as fh:
        return {(r["case_id"], r["arm"]): r for r in csv.DictReader(fh)}


def normalize_map_for_display(m: np.ndarray) -> np.ndarray:
    lo, hi = float(np.percentile(m, 1)), float(np.percentile(m, 99))
    if hi <= lo:
        return np.zeros_like(m)
    return np.clip((m - lo) / (hi - lo), 0.0, 1.0)


# --------------------------------------------------------------------------------------
def build_page(out_path: Path, arm: str, case: Dict[str, object], n_frames: int,
               source_video: Path, arm_video_dir: Path, const_dirs: Dict[int, Path],
               div_map: Optional[np.ndarray], d_value: Optional[float],
               routed_b: Optional[float], tile_width: int) -> Optional[str]:
    case_id = str(case["case_id"])
    is_spatial = arm in rd.SPATIAL_ARMS

    rows: List[Tuple[str, object]] = [("source", VideoRow("mp4", source_video))]
    if is_spatial:
        rows.append(("divmap", None))  # placeholder, handled specially below
    rows.append((f"{arm} (routed)", VideoRow("png_dir", arm_video_dir)
                if arm_video_dir.exists() else None))
    for b in CONST_BS:
        d = const_dirs.get(b)
        rows.append((f"const b={b}", VideoRow("png_dir", d) if d is not None and d.exists()
                    else None))

    missing = [label for label, src in rows if label != "divmap" and src is None]
    if missing:
        return f"{case_id}/{arm}: missing frame source(s) for {missing}"

    fracs = [i / (n_frames - 1) if n_frames > 1 else 0.0 for i in range(n_frames)]
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, n_frames,
                             figsize=(1.6 * n_frames, 1.05 * nrow + 0.6),
                             squeeze=False)

    disp_map = normalize_map_for_display(div_map) if div_map is not None else None
    for r, (label, src) in enumerate(rows):
        for c, frac in enumerate(fracs):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if label == "divmap":
                if disp_map is not None:
                    ax.imshow(disp_map, cmap="magma", vmin=0.0, vmax=1.0)
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7,
                           transform=ax.transAxes)
            else:
                frame = src.frame(frac)
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

    d_str = f"{d_value:.4f}" if d_value is not None else "n/a"
    b_str = f"{routed_b:.2f}" if routed_b is not None else "n/a"
    fig.suptitle(f"{case_id} -- arm={arm}  d_{arm}_masked={d_str}  routed b={b_str}",
                fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return None


# --------------------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.expanduser().read_text())
    if args.only_cases:
        wanted = set(args.only_cases.split(","))
        cases = [c for c in cases if c["case_id"] in wanted]
    if not cases:
        raise SystemExit("[r30_arm_grids] no cases selected")

    div_by_case = load_csv_by_case(args.divergence_csv)
    arms_by_case_arm = load_arms_by_case_arm(args.fiveacc_arms_csv)

    args.data_root = args.data_root.expanduser()
    args.anchor_root = args.anchor_root.expanduser()
    args.mask_dir = args.mask_dir.expanduser()
    args.split_mask_dir = args.split_mask_dir.expanduser() if args.split_mask_dir else None

    print(f"[r30_arm_grids] loading 4 spatial-arm models on {args.device} ...", flush=True)
    models = load_spatial_models(args.device, args)
    print("[r30_arm_grids] models loaded", flush=True)

    failures: List[str] = []
    n_written = 0
    for case in cases:
        case_id = str(case["case_id"])
        video_name = str(case["video_name"])
        edit_type = int(case["edit_type"])
        src_video = args.data_root / str(case["src_video"])

        for arm in rd.ARMS:
            div_row = div_by_case.get(case_id, {})
            d_str = div_row.get(f"d_{arm}_masked", "")
            d_value = float(d_str) if d_str not in ("", None) else None

            arm_row = arms_by_case_arm.get((case_id, arm))
            routed_b = float(arm_row["b"]) if arm_row and arm_row.get("b") not in ("", None) \
                else None

            arm_video_dir = args.arms_root / arm / f"edit{edit_type}" / video_name
            const_dirs = {b: args.const_root / f"taubg0_taufg{b}_vp" / f"edit{edit_type}"
                         / video_name for b in CONST_BS}

            div_map = None
            if arm in rd.SPATIAL_ARMS:
                div_map = _recompute_one_map(case, models, args, arm)

            out_path = args.out_dir / arm / f"{case_id}.png"
            err = build_page(out_path, arm, case, args.n_frames, src_video, arm_video_dir,
                            const_dirs, div_map, d_value, routed_b, args.tile_width)
            if err:
                failures.append(err)
                print(f"[r30_arm_grids] [SKIP] {err}", flush=True)
            else:
                n_written += 1
                print(f"[r30_arm_grids] wrote {out_path}", flush=True)

    print(f"\n[r30_arm_grids] wrote {n_written} page(s) under {args.out_dir}, "
         f"{len(failures)} failure(s)", flush=True)
    return 0


def _recompute_one_map(case: Dict[str, object], models: Dict[str, object],
                       args: argparse.Namespace, arm: str) -> Optional[np.ndarray]:
    """Recomputes just one spatial arm's map, reusing r30_divergence's own per-arm code
    by calling measure_case with arms=[arm] and reaching into its local `maps` dict via
    a light monkeypatch-free re-derivation: measure_case does not return `maps`, so this
    duplicates its map-producing branch bodies at the smallest possible scope instead of
    forking the whole function. Kept arm-by-arm (not batched) so a single arm's failure
    (e.g. depth's background-fraction gate) does not cost the other three their map.
    """
    data_root, anchor_root, mask_dir = args.data_root, args.anchor_root, args.mask_dir
    case_id = str(case["case_id"])
    video_name = str(case["video_name"])
    edit_type = int(case["edit_type"])
    device = args.device

    src_path = data_root / "images" / video_name / "00001.jpg"
    anchor_path = anchor_root / f"edit{edit_type}" / f"{video_name}.png"
    mask_path = mask_dir / f"edit{edit_type}" / f"{video_name}.npz"
    split_mask_path = (args.split_mask_dir / f"edit{edit_type}" / f"{video_name}.npz"
                       if args.split_mask_dir is not None else None)

    src_raw = Image.open(src_path).convert("RGB")
    anchor_raw = Image.open(anchor_path).convert("RGB")
    hw = (anchor_raw.size[1], anchor_raw.size[0])
    h, w = hw
    src_img = rd.resize_to(src_raw, hw)
    anchor_img = rd.resize_to(anchor_raw, hw)
    union, lat_hw, _ = rd.load_union_frame0(mask_path, hw, args.mask_open, split_mask_path,
                                            args.trg_degenerate_frac)
    bg = ~union
    if not union.any():
        return None

    rd.set_seed(args.seed)
    if arm == "lpips":
        net = models["lpips"]
        a = rd.to_unit_tensor(src_img, hw, device) * 2.0 - 1.0
        b = rd.to_unit_tensor(anchor_img, hw, device) * 2.0 - 1.0
        with torch.no_grad():
            total, _ = net(a, b, retPerLayer=True)
        return total[0, 0].detach().cpu().numpy().astype(np.float64)

    if arm == "dino_patch":
        model = models["dinov2"]
        dh = (h // rd.DINOV2_PATCH) * rd.DINOV2_PATCH
        dw = (w // rd.DINOV2_PATCH) * rd.DINOV2_PATCH
        gh, gw = dh // rd.DINOV2_PATCH, dw // rd.DINOV2_PATCH
        _, pat_s, _ = rd.dinov2_features(model, src_img, (dh, dw), device)
        _, pat_a, _ = rd.dinov2_features(model, anchor_img, (dh, dw), device)
        per_pos = (1.0 - (pat_s * pat_a).sum(-1)).cpu().numpy().astype(np.float64)
        grid = per_pos.reshape(gh, gw)
        return np.asarray(Image.fromarray(grid.astype(np.float32)).resize((w, h),
                          Image.NEAREST), dtype=np.float64)

    if arm == "depth":
        processor, model = models["depth"]
        bg_frac = float(bg.mean())
        if bg_frac < rd.MIN_BG_FRACTION:
            return None
        d_src = rd.estimate_depth(src_img, processor, model, device, hw)
        d_anc = rd.estimate_depth(anchor_img, processor, model, device, hw)
        if float(np.std(d_anc[bg])) <= 0.0:
            return None
        a, b_ = rd.fit_affine(d_anc, d_src, bg)
        resid_abs = np.abs(d_src - (a * d_anc + b_))
        if args.depth_norm == "iqr":
            q1, q3 = np.percentile(d_src, [25, 75])
            iqr_src = float(q3 - q1)
            if iqr_src <= 0.0:
                return None
            return resid_abs / iqr_src
        return resid_abs

    if arm == "normals":
        pipe = models["normals"]
        n_src = rd.estimate_normals(pipe, src_img, hw, args.normals_steps,
                                    args.normals_ensemble, args.seed, device)
        n_anc = rd.estimate_normals(pipe, anchor_img, hw, args.normals_steps,
                                    args.normals_ensemble, args.seed, device)
        cos = np.clip((n_src * n_anc).sum(-1), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    return None


# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--only_cases", type=str, default=None,
                   help="Comma-separated case_id subset, for a quick smoke test.")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--anchor_root", type=Path, required=True)
    p.add_argument("--mask_dir", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_masks"))
    p.add_argument("--split_mask_dir", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r30_split_masks"),
                   help="Reproduces the same M_src-only fallback used for 0042_gym-ball "
                        "in the production r30_divergence.csv -- see that script's flag "
                        "of the same name. Pass empty/None to disable.")
    p.add_argument("--trg_degenerate_frac", type=float, default=0.90)
    p.add_argument("--mask_open", type=int, default=1)
    p.add_argument("--depth_norm", choices=("iqr", "raw"), default="iqr")
    p.add_argument("--normals_steps", type=int, default=4)
    p.add_argument("--normals_ensemble", type=int, default=5)
    p.add_argument("--dinov2_model_id", type=str, default=rd.DINOV2_MODEL_ID)
    p.add_argument("--depth_model_id", type=str, default=rd.DEPTH_MODEL_ID)
    p.add_argument("--normals_model_id", type=str, default=rd.NORMALS_MODEL_ID)
    p.add_argument("--arms_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r30_arms"),
                   help="{arms_root}/{arm}/edit{T}/{video}/{iiiii}.png")
    p.add_argument("--const_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_spatial_tau"),
                   help="{const_root}/taubg0_taufg{b}_vp/edit{T}/{video}/{iiiii}.png")
    p.add_argument("--divergence_csv", type=Path,
                   default=Path("evaluation/csv/r30_divergence.csv"))
    p.add_argument("--fiveacc_arms_csv", type=Path,
                   default=Path("evaluation/csv/r30_fiveacc_arms.csv"))
    p.add_argument("--n_frames", type=int, default=6, help="Grid columns (F).")
    p.add_argument("--tile_width", type=int, default=180)
    p.add_argument("--out_dir", type=Path, default=Path("evaluation/figures/r30_arm_grids"))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
