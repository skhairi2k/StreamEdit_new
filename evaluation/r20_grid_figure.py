#!/usr/bin/env python3
"""R20 qualitative grid figures.

For every (case, vp_mode) pair in evaluation/cases.json this renders ONE figure:

    rows  = Source + the 6 blend schedules (Eq.4, cos_full, cos_half, cos_third,
            const, zero)
    cols  = K frames evenly sampled over the clip (first & last always included)

=> 22 cases x 3 vp_modes = 66 PNGs. Time is the x-axis (left->right = later),
schedules are adjacent rows so ranking is a vertical scan; one file per VP mode
keeps each grid legible (the layout chosen for R20).

Directory resolution
--------------------
The "Eq.4 / paper" row is NOT one directory -- the paper schedule under each VP
mode is a different run:

    novp -> five_bench/baseline               (R1)
    vp   -> five_bench/r7_visual_prompting     (R7)
    pvp  -> r20_blend_sched/paper_pvp          (rendered here)

every other schedule is r20_blend_sched/{sched}_{vp_mode}. Frames live at
{arm}/edit{T}/{video}/{idx:05d}.png.

Source alignment
----------------
Source frames (DATA_ROOT/images/{video}/{i:05d}.jpg, 1-indexed) usually have a
DIFFERENT count than the decoded edit -- but NOT because of temporal resampling.
The pipeline prefix-truncates the source to the nearest valid length before
editing (find_closest_num_frame -> src_video[:new_len]), so the edit is exactly
the first n_out source frames, aligned 1:1: edit frame i <-> source jpg i+1.
We map the reference that way; a proportional-time stretch would desync the
Source row (worst in the last columns / high-motion clips).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# schedule key -> row label. 'paper' is the Eq.4 reference (special dir mapping).
SCHED_LABEL = {
    "paper": "Eq.4",
    "cos_full": "cos_full",
    "cos_half": "cos_half",
    "cos_third": "cos_third",
    "const": "const",
    "zero": "zero",
}
VP_TITLE = {"novp": "No VP", "vp": "VP (§4.5)", "pvp": "Persistent VP"}


def arm_dir(sched: str, vp_mode: str, out_root: Path, ref_root: Path) -> Path:
    """Map (schedule, vp_mode) to the run directory holding edit{T}/{video}/."""
    if sched == "paper":
        return {
            "novp": ref_root / "baseline",
            "vp": ref_root / "r7_visual_prompting",
            "pvp": out_root / "paper_pvp",
        }[vp_mode]
    return out_root / f"{sched}_{vp_mode}"


def load_frame(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def n_output_frames(case_dir: Path) -> int:
    return len(list(case_dir.glob("[0-9]*.png")))


def sample_indices(n: int, k: int) -> np.ndarray:
    """k evenly spaced frame indices in [0, n-1], first & last included."""
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def source_index_for(out_idx: int, n_out: int, n_src: int) -> int:
    """Source index (1-indexed jpg filename) for an output sample.

    The pipeline prefix-truncates the source to a valid length before editing
    (find_closest_num_frame -> src_video[:new_len]), so the edit IS the first
    n_out source frames, aligned 1:1: edit frame i <-> source frame i+1.
    """
    return min(out_idx + 1, n_src)  # +1: source jpgs are 1-indexed


def placeholder(ax, text: str) -> None:
    ax.set_facecolor("0.85")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=7, color="0.35",
            transform=ax.transAxes)


def render_case(case: dict, vp_mode: str, schedules: list[str], k: int,
                out_root: Path, ref_root: Path, data_root: Path,
                outdir: Path, dpi: int) -> Optional[Path]:
    edit_t = case["edit_type"]
    video = case["video_name"]
    sub = f"edit{edit_t}/{video}"

    # frame count comes from whichever schedule dir is present (all arms share it)
    n_out = 0
    for s in schedules:
        cd = arm_dir(s, vp_mode, out_root, ref_root) / sub
        if cd.is_dir():
            n_out = n_output_frames(cd)
            if n_out:
                break
    if not n_out:
        print(f"[skip] {sub} [{vp_mode}]: no rendered frames found")
        return None

    idxs = sample_indices(n_out, k)
    rows = ["source"] + schedules
    ncols = len(idxs)

    # source frames + proportional index map
    src_dir = data_root / "images" / video
    src_jpgs = sorted(src_dir.glob("[0-9]*.jpg")) if src_dir.is_dir() else []
    n_src = len(src_jpgs)

    # figure size from image aspect (fall back to 16:9)
    sample_img = None
    for s in schedules:
        cd = arm_dir(s, vp_mode, out_root, ref_root) / sub
        img = load_frame(cd / f"{idxs[0]:05d}.png")
        if img is not None:
            sample_img = img
            break
    aspect = (sample_img.shape[0] / sample_img.shape[1]) if sample_img is not None else 9 / 16
    cell_w = 2.2
    fig_w = ncols * cell_w + 1.1              # +label gutter
    fig_h = len(rows) * cell_w * aspect + 0.8  # +title
    fig, axes = plt.subplots(len(rows), ncols, figsize=(fig_w, fig_h), squeeze=False)

    for r, row in enumerate(rows):
        for c, fi in enumerate(idxs):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("0.7"); sp.set_linewidth(0.4)
            if row == "source":
                if n_src:
                    si = source_index_for(int(fi), n_out, n_src)
                    img = load_frame(src_dir / f"{si:05d}.jpg")
                else:
                    img = None
            else:
                cd = arm_dir(row, vp_mode, out_root, ref_root) / sub
                img = load_frame(cd / f"{fi:05d}.png")
            if img is None:
                placeholder(ax, "missing")
            else:
                ax.imshow(img)
            if c == 0:
                lbl = "Source" if row == "source" else SCHED_LABEL.get(row, row)
                ax.set_ylabel(lbl, fontsize=9, rotation=0, ha="right", va="center",
                              labelpad=6, fontweight="bold" if row in ("source", "paper") else "normal")
            if r == 0:
                ax.set_title(f"f{int(fi)}", fontsize=8, pad=3)

    src_word = case.get("src_word", "?"); trg_word = case.get("trg_word", "?")
    fig.suptitle(f"{video} · edit{edit_t} · {VP_TITLE.get(vp_mode, vp_mode)}"
                 f"   —   {src_word} → {trg_word}",
                 fontsize=11, y=0.997)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.93, bottom=0.01,
                        wspace=0.03, hspace=0.06)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{video}_edit{edit_t}_{vp_mode}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_json", default=str(repo / "evaluation/cases.json"))
    ap.add_argument("--out_root",
                    default="/projects/dataggen/outputs/five_bench/r20_blend_sched")
    ap.add_argument("--ref_root", default="/projects/dataggen/outputs/five_bench")
    ap.add_argument("--data_root",
                    default=os.path.expanduser(
                        "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    ap.add_argument("--outdir", default=str(repo / "figures/r20_grids"))
    ap.add_argument("-k", "--frames", type=int, default=5, help="frames per video")
    ap.add_argument("--schedules", nargs="+",
                    default=["paper", "cos_full", "cos_half", "cos_third", "const", "zero"])
    ap.add_argument("--vp_modes", nargs="+", default=["novp", "vp", "pvp"])
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--only_video", default=None,
                    help="restrict to a single video_name (debug)")
    args = ap.parse_args()

    cases = json.load(open(args.cases_json))
    out_root, ref_root = Path(args.out_root), Path(args.ref_root)
    data_root, outbase = Path(args.data_root), Path(args.outdir)

    made = 0
    for case in cases:
        if args.only_video and case["video_name"] != args.only_video:
            continue
        for vp in args.vp_modes:
            p = render_case(case, vp, args.schedules, args.frames,
                            out_root, ref_root, data_root,
                            outbase / vp, args.dpi)
            if p is not None:
                made += 1
                print(f"[ok] {p}")
    print(f"[r20_grid] wrote {made} figures under {outbase}")


if __name__ == "__main__":
    main()
