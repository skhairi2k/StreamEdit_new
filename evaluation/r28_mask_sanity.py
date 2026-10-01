#!/usr/bin/env python
"""R28 mask sanity: is the grounded target mask actually on the edited object?

Every number R28 produces rests on `M_tgt` being right. The metrics cannot tell a real
preservation result from a mask that grounded the wrong thing -- an over-firing mask
shrinks the background and flatters the arm, an under-firing one inflates it. This figure
is the only artifact that separates the two, and it is what caught R26's loose-mask caveat.

Drawn on the arm's OWN RENDERED frames, because that is the canvas `M_tgt` was grounded
on. Four states per pixel, with FIXED colours:

    blue    M_src only  -- GT source object the edit vacated
    orange  M_tgt only  -- the region the CURRENT metric wrongly scores as background
    white   both        -- object present in source and target
    untinted            -- background under the union, the region R28 actually scores

The orange area IS R28's thesis made visible: it is exactly the pixels `*_unedit_part`
counts as "background that should have been preserved" while some arm demonstrably edited
them. A figure with almost no orange would mean the metric change cannot matter.

Optional 5th state, drawn when ``--fixed_union_dir`` is given (purple): pixels inside the
FIXED cross-arm union but outside this arm's OWN `M_src | M_tgt`, i.e. area some OTHER
arm's grounding contributed. `arm_union ⊆ fixed_union` always (r28_fixed_union.py asserts
it), so this is never negative -- it is exactly how much MORE `*_unedit_union_fixed`
excludes from this arm's scored background versus what the arm's own `*_unedit_union`
would have excluded. A figure with lots of purple on an arm whose own mask looks tight
means the fixed family is scoring it on a noticeably smaller region than its per-arm mask
alone would suggest, which is the situation `*_unedit_union_fixed` exists to normalize.

⚠️ Colours are written as an explicit RGBA overlay with the background fully transparent,
NOT via `imshow(mask, cmap=...)`. matplotlib autoscales a single-valued array, so an
all-ones or all-zeros mask would render in whatever colour means something else in the
neighbouring panel -- the exact misreading R26's first sanity figure produced.

Clips are chosen by how far `M_tgt` extends BEYOND `M_src`, since that is the quantity the
whole task turns on, and any degenerate (empty / all-ones) mask is always kept.

Usage
-----
    python evaluation/r28_mask_sanity.py \\
        --tgt_mask_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/taubg2_taufg2_vp \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg2_taufg2_vp \\
        --fixed_union_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union \\
        --cases evaluation/cases.json \\
        -o evaluation/figures/r28_mask_sanity.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r28_target_masks import list_images, load_src_masks, load_npz  # noqa: E402
from r28_fixed_union import load_fixed_union  # noqa: E402

C_SRC = (0.20, 0.45, 1.00)    # M_src only
C_TGT = (1.00, 0.45, 0.00)    # M_tgt only  <- the pixels today's metric mis-scores
C_BOTH = (1.00, 1.00, 1.00)   # both
C_FIXED = (0.60, 0.10, 0.85)  # fixed union only -- contributed by another arm
ALPHA = 0.50


def clip_stats(case, tgt_mask_dir: str, data_root: str, stride: int):
    """(M_src, M_tgt, render frame paths) for one clip, or None if unavailable."""
    t, v = int(case["edit_type"]), case["video_name"]
    mp = os.path.join(tgt_mask_dir, f"edit{t}", f"{v}.npz")
    if not os.path.exists(mp):
        return None
    M = load_npz(mp, expect_stride=stride)
    sp = list_images(os.path.join(data_root, "images", v))[::stride][: M.shape[0]]
    if not sp:
        return None
    w, h = Image.open(sp[0]).size
    S = load_src_masks(os.path.join(data_root, "bmasks", v), sp, (h, w))
    if S is None:
        return None
    n = min(M.shape[0], S.shape[0])
    return S[:n], M[:n], (h, w)


def fixed_union_for_clip(case, fixed_union_dir: str, stride: int) -> Optional[np.ndarray]:
    """This clip's fixed cross-arm union mask, or None if not (yet) built for it."""
    t, v = int(case["edit_type"]), case["video_name"]
    p = os.path.join(fixed_union_dir, f"edit{t}", f"{v}.npz")
    if not os.path.exists(p):
        return None
    return load_fixed_union(p, expect_stride=stride)


def overlay(img: Image.Image, src: np.ndarray, tgt: np.ndarray,
            fixed: Optional[np.ndarray] = None) -> np.ndarray:
    """RGBA overlay: fixed colours, background fully transparent.

    ``fixed``, when given, is the FIXED cross-arm union for this frame. It always
    contains ``src | tgt`` (r28_fixed_union.py asserts this), so it can only ever add a
    5th category -- area some OTHER arm's grounding contributed -- never remove one.
    """
    w, h = img.size

    def up(m):
        if m.shape == (h, w):
            return m
        return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127

    s, g = up(src), up(tgt)
    rgba = np.zeros((h, w, 4), dtype=float)
    rgba[s & ~g] = (*C_SRC, ALPHA)
    rgba[g & ~s] = (*C_TGT, ALPHA)
    rgba[s & g] = (*C_BOTH, ALPHA)
    if fixed is not None:
        u = up(fixed)
        rgba[u & ~s & ~g] = (*C_FIXED, ALPHA)
    return rgba


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tgt_mask_dir", required=True, help="ONE arm's mask dir")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True, help="the SAME arm's render root")
    ap.add_argument("--fixed_union_dir", default=None,
                    help="optional: r28_fixed_union.py's output dir (e.g. "
                         ".../r28_tgt_masks/_fixed_union). When given, draws a 5th "
                         "purple category -- area in the FIXED union but not in this "
                         "arm's own M_src|M_tgt, i.e. contributed by another arm.")
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--frame_stride", type=int, default=8)
    ap.add_argument("--n_clips", type=int, default=6)
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("-o", "--out", default="evaluation/figures/r28_mask_sanity.pdf")
    args = ap.parse_args(argv)

    data_root = os.path.expanduser(args.data_root)
    cases = json.load(open(os.path.expanduser(args.cases), encoding="utf-8"))

    # Rank by how far M_tgt extends beyond M_src -- the quantity R28 turns on. Degenerate
    # masks are kept regardless: seeing one is worth more than a tidy spread.
    scored, degen = [], []
    for c in cases:
        st = clip_stats(c, args.tgt_mask_dir, data_root, args.frame_stride)
        if st is None:
            continue
        S, M, _ = st
        beyond = float((M & ~S).mean())
        frac = float(M.mean())
        (degen if frac in (0.0, 1.0) else scored).append((beyond, c, frac))
    if not scored and not degen:
        print("[r28_sanity] no clips with masks found", file=sys.stderr)
        return 1
    scored.sort(key=lambda x: -x[0])
    chosen = degen[: args.n_clips]
    need = args.n_clips - len(chosen)
    if need > 0 and scored:
        idx = sorted({int(round(t * (len(scored) - 1))) for t in np.linspace(0, 1, need)})
        chosen += [scored[i] for i in idx]
    chosen = chosen[: args.n_clips]
    print(f"[r28_sanity] {len(chosen)} clips, ranked by M_tgt-beyond-M_src area")

    fig, axes = plt.subplots(len(chosen), args.n_frames,
                             figsize=(3.0 * args.n_frames, 2.0 * len(chosen)), squeeze=False)
    for ri, (beyond, c, frac) in enumerate(chosen):
        t, v = int(c["edit_type"]), c["video_name"]
        S, M, _ = clip_stats(c, args.tgt_mask_dir, data_root, args.frame_stride)
        U = None
        if args.fixed_union_dir is not None:
            U = fixed_union_for_clip(c, args.fixed_union_dir, args.frame_stride)
            if U is not None:
                n = min(U.shape[0], M.shape[0])
                S, M, U = S[:n], M[:n], U[:n]
        rp = list_images(os.path.join(args.out_root, f"edit{t}", v))[:: args.frame_stride][: M.shape[0]]
        pick = [int(round(x * (len(rp) - 1))) for x in np.linspace(0, 1, args.n_frames)] if rp else []
        for ci in range(args.n_frames):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ci >= len(pick):
                ax.set_axis_off(); continue
            k = pick[ci]
            img = Image.open(rp[k]).convert("RGB")
            ax.imshow(img)
            ax.imshow(overlay(img, S[k], M[k], fixed=U[k] if U is not None else None))
            if ci == 0:
                flag = "  ⚠DEGENERATE" if frac in (0.0, 1.0) else ""
                extra = f" fixed_union={float(U.mean()):.3f}" if U is not None else ""
                ax.set_ylabel(f"edit{t} {v}\nsrc={float(S.mean()):.3f} tgt={frac:.3f}\n"
                              f"beyond={beyond:.3f}{extra}{flag}",
                              fontsize=6.5, rotation=0, ha="right", va="center")
            ax.set_title(f"frame {k * args.frame_stride}", fontsize=6.5)

    legend_handles = [
        Patch(facecolor=C_SRC, alpha=ALPHA, label="$M_{src}$ only (GT object the edit vacated)"),
        Patch(facecolor=C_TGT, alpha=ALPHA,
              label="$M_{tgt}$ only — pixels `_unedit_part` WRONGLY scores as background"),
        Patch(facecolor=C_BOTH, alpha=ALPHA, edgecolor="grey", label="both"),
    ]
    if args.fixed_union_dir is not None:
        legend_handles.append(Patch(facecolor=C_FIXED, alpha=ALPHA,
            label="fixed union only — contributed by ANOTHER arm, excluded only under `_union_fixed`"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 4), fontsize=7, frameon=False)
    fig.suptitle(f"R28 mask sanity — {os.path.basename(args.tgt_mask_dir.rstrip('/'))} "
                 "· overlay on the arm's OWN rendered frames", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[r28_sanity] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
