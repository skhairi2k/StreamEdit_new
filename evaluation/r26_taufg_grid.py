#!/usr/bin/env python
"""R26: per-video tau_fg sweep grids at tau_bg = 0.

One page per clip. Each page is a header band plus an ``(1 + N) x T`` image grid:

    row 0        the SOURCE clip
    rows 1..N    the ``tau_bg=0, tau_fg=<fg>`` arms, one row per swept ``tau_fg``
    columns      ``T`` frames sampled over NORMALIZED time

``tau_bg`` is held at 0 throughout -- the background is pinned to the source for the
whole rollout -- so every difference down a column is attributable to ``tau_fg`` alone,
which is the point of the figure. The 12-cell (bg, fg) picture lives in
``r26_grid_figure.py``'s heatmaps instead; this one isolates a single row of that grid
and shows what it looks like.

The header band carries the VISUAL PROMPT actually fed to the run: the anchor image
(``anchor_image`` in cases.json, i.e. the edited first frame used for ``--vp_mode vp``),
next to the source first frame it was derived from, next to the text that produced it.
Without it the reader cannot tell an arm that failed to follow the edit from an arm that
faithfully followed an anchor that was itself wrong.

Every row is TRIMMED to ``find_closest_num_frame`` before time sampling, exactly as
``r26_grid_figure.draw_grids`` does: run_fivebench.py cuts each clip to that length before
encoding, so the source directory holds MORE frames than any render (80 on disk vs 69
rendered for 0001_bus). Sampling normalized time over the untrimmed source would push the
source row's t=1.0 past the end of every arm row and the rows would drift apart at the
tail -- an artifact of the figure that reads as a temporal artifact of the method.

CPU-only: no VAE, no masks, no GPU.

Usage
-----
    python evaluation/r26_taufg_grid.py \\
        --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \\
        --cases evaluation/cases.json \\
        --n_frames 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r26_grid_figure import (  # noqa: E402  reuse the trimming/sampling used by the (b) grids
    _closest_num_frame,
    arm_name,
    frame_paths,
    sample_by_time,
)

_ARM_RE = re.compile(r"^taubg(\d+)_taufg(\d+)_vp$")


def discover_tau_fg(out_root: str, tau_bg: int) -> List[int]:
    """Every ``tau_fg`` present on disk at this ``tau_bg``, ascending.

    Derived from the arm directories rather than hardcoded: the sweep grew from
    ``tau_fg in {2, 6, 10}`` to ``{2, 3, 4, 6, 8, 10, 20, 50}`` over the course of R26,
    and a hardcoded list would silently drop the arms added last.
    """
    if not os.path.isdir(out_root):
        raise FileNotFoundError(f"{out_root} not found")
    fgs = []
    for name in os.listdir(out_root):
        m = _ARM_RE.match(name)
        if m and int(m.group(1)) == tau_bg:
            fgs.append(int(m.group(2)))
    return sorted(fgs)


def source_frames(src_root: str, video_name: str) -> List[str]:
    """Source frames TRIMMED to ``find_closest_num_frame`` -- what was actually rendered."""
    sp = frame_paths(os.path.join(src_root, "images", video_name))
    if not sp:
        return []
    n_keep = _closest_num_frame(len(sp))
    return sp[:n_keep] if n_keep else sp


def prompt_text(case: dict, width: int = 78) -> str:
    """The text side of the visual prompt: what the anchor was asked for, and the target."""
    lines = [
        f"src word : {case.get('src_word', '')}",
        f"trg word : {case.get('trg_word', '')}",
        "",
        "first-frame edit prompt (-> anchor image):",
    ]
    lines += textwrap.wrap(case.get("first_frame_edit_prompt")
                           or case.get("instruction", ""), width) or [""]
    lines += ["", "target prompt (video):"]
    lines += textwrap.wrap(case.get("trg_prompt", ""), width)
    return "\n".join(lines)


def draw_case(case: dict, out_root: str, src_root: str, tau_bg: int,
              tau_fgs: Sequence[int], n_frames: int) -> Optional[plt.Figure]:
    """Build one clip's page, or ``None`` if it has no usable row."""
    t, name = int(case["edit_type"]), case["video_name"]

    sp = source_frames(src_root, name)
    rows: List[Tuple[str, List[str]]] = []
    if sp:
        rows.append(("source", sample_by_time(sp, n_frames)))
    else:
        print(f"[r26_taufg] WARNING edit{t}/{name}: no source frames under "
              f"{os.path.join(src_root, 'images', name)}")

    missing: List[int] = []
    for fg in tau_fgs:
        d = os.path.join(out_root, arm_name(tau_bg, fg), f"edit{t}", name)
        fp = frame_paths(d)
        if not fp:
            missing.append(fg)
            continue
        if sp and len(fp) != len(sp):
            # After the trim the VAE round-trips n -> n, so a length mismatch means this
            # render is not this source's render -- the rows would not be time-aligned.
            print(f"[r26_taufg] WARNING edit{t}/{name} tau_fg={fg}: {len(fp)} rendered "
                  f"frames vs {len(sp)} trimmed source frames -- rows may not align")
        rows.append((rf"$\tau_{{fg}}$ = {fg}", sample_by_time(fp, n_frames)))
    if missing:
        print(f"[r26_taufg] edit{t}/{name}: no render for tau_fg in {missing} "
              f"(tau_bg={tau_bg}); those rows omitted")
    if len(rows) < 2:
        print(f"[r26_taufg] skip edit{t}/{name}: only {len(rows)} row(s) available")
        return None

    n_rows = len(rows)
    head_h, cell_h, cell_w = 2.4, 1.6, 2.5
    fig = plt.figure(figsize=(cell_w * n_frames, head_h + cell_h * n_rows))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[head_h, cell_h * n_rows], hspace=0.12,
        left=0.075, right=0.995, top=0.93, bottom=0.01)

    # -- header: source first frame | anchor (visual prompt) | the text behind it -------
    hgs = gs[0].subgridspec(1, 3, width_ratios=[1, 1, 2.6], wspace=0.06)
    anchor = case.get("anchor_image", "")
    for ax, (label, path) in zip(
            (fig.add_subplot(hgs[0]), fig.add_subplot(hgs[1])),
            (("source first frame", sp[0] if sp else None),
             ("visual prompt (anchor)", anchor if anchor and os.path.exists(anchor) else None))):
        ax.set_xticks([]); ax.set_yticks([])
        if path:
            ax.imshow(Image.open(path).convert("RGB"))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color="red")
        ax.set_title(label, fontsize=8)
    if anchor and not os.path.exists(anchor):
        print(f"[r26_taufg] WARNING edit{t}/{name}: anchor image {anchor} not found")
    tax = fig.add_subplot(hgs[2])
    tax.axis("off")
    tax.text(0.0, 1.0, prompt_text(case), fontsize=7, family="monospace",
             ha="left", va="top", transform=tax.transAxes)

    # -- the (1 + N) x T grid -----------------------------------------------------------
    ggs = gs[1].subgridspec(n_rows, n_frames, wspace=0.02, hspace=0.06)
    for ri, (label, items) in enumerate(rows):
        for ci in range(n_frames):
            ax = fig.add_subplot(ggs[ri, ci])
            ax.set_xticks([]); ax.set_yticks([])
            if ci < len(items):
                ax.imshow(Image.open(items[ci]).convert("RGB"))
            if ci == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
            if ri == 0:
                ax.set_title(f"t={ci / max(n_frames - 1, 1):.2f}", fontsize=7)

    fig.suptitle(f"edit{t} — {name}   |   $\\tau_{{bg}}$ = {tau_bg}, "
                 f"$\\tau_{{fg}}$ sweep   (frames sampled by normalized time)", fontsize=11)
    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root",
                    default="/projects/dataggen/outputs/five_bench/r26_spatial_tau",
                    help="root holding the taubg{X}_taufg{Y}_vp arm directories")
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--src_root", default=os.path.expanduser(
        "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"),
        help="FiVE benchmark root holding images/{video}/*.jpg")
    ap.add_argument("--out_dir", default="evaluation/figures/r26_taufg_grids")
    ap.add_argument("--tau_bg", type=int, default=0,
                    help="held fixed for every row (0 pins the background to the source)")
    ap.add_argument("--tau_fg", default="",
                    help="comma-separated tau_fg values; default = every arm found on "
                         "disk at this tau_bg")
    ap.add_argument("--n_frames", type=int, default=6)
    ap.add_argument("--ext", default="png", choices=("png", "pdf"),
                    help="per-clip file format")
    ap.add_argument("--combined", default="r26_taufg_grids.pdf",
                    help="multi-page PDF with every clip, written next to the per-clip "
                         "files; '' to skip")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args(argv)

    with open(args.cases, encoding="utf-8") as fh:
        cases = json.load(fh)
    tau_fgs = ([int(x) for x in args.tau_fg.split(",") if x.strip()]
               if args.tau_fg.strip() else discover_tau_fg(args.out_root, args.tau_bg))
    if not tau_fgs:
        print(f"[r26_taufg] no taubg{args.tau_bg}_taufg*_vp arms under {args.out_root}",
              file=sys.stderr)
        return 1
    print(f"[r26_taufg] {len(cases)} cases x tau_bg={args.tau_bg}, tau_fg={tau_fgs} "
          f"({len(tau_fgs)} rows + source) x {args.n_frames} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    pdf = (PdfPages(os.path.join(args.out_dir, args.combined))
           if args.combined.strip() else None)
    written = 0
    try:
        for case in cases:
            fig = draw_case(case, args.out_root, args.src_root, args.tau_bg,
                            tau_fgs, args.n_frames)
            if fig is None:
                continue
            p = os.path.join(
                args.out_dir,
                f"edit{int(case['edit_type'])}_{case['video_name']}_taubg{args.tau_bg}"
                f".{args.ext}")
            fig.savefig(p, dpi=args.dpi)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
            written += 1
    finally:
        if pdf is not None:
            pdf.close()
    print(f"[r26_taufg] wrote {written} page(s) -> {args.out_dir}/")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
