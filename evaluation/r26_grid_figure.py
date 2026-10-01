#!/usr/bin/env python
"""R26 figures: tau heatmaps, qualitative grids, and the mask-sanity overlay.

Three outputs, in the order they should be read:

(a) ``r26_tau_heatmap.pdf``
    3x3 heatmaps over ``(tau_bg, tau_fg)`` for ``clip_similarity_target_image`` and
    ``lpips_unedit_part``, with the ``(2,2)`` control cell marked. Both subsets written by
    ``r26_summarize.py`` are drawn (``all`` and ``excl_degenerate``), because the single
    degenerate clip is a sixth of the macro table and a 22nd of the micro one -- reading
    only one of the two hides how much it moves.

(b) ``r26_grids/{edit}_{video}.pdf``
    One page per clip: ``{source, baseline vp (2,2), extreme tau (0,50), best tau
    (overall, union), best tau (overall, part), best tau (per video, union), best tau
    (per video, part), M_f (pass 1)}`` x 5 frames (rows that dedupe against ``(2,2)`` or
    against each other are dropped).

    Row 3, ``extreme tau (0,50)``, is a FIXED arm, not a selection: ``tau_bg=0`` pins the
    background to the source for the whole rollout (the arm most prone to seam/freeze,
    see below) while ``tau_fg=50`` is the most aggressive edit-side push in the sweep. It
    shows what the two per-region extremes look like taken together, independent of
    whether any selection criterion would actually pick it.

    Rows 4-7 are two SELECTION CRITERIA, each run twice (once "overall", once "per
    video"):

    - ``union`` (rows 4, 6): min-max-normalized ``clip_similarity_target_image`` (edit
      fidelity) + min-max-normalized ``ssim_unedit_union`` (R28's per-arm union-mask
      preservation, ``1 - (M_src | M_tgt_arm)``), summed and maximised.
    - ``part`` (rows 5, 7): the SAME normalized-CLIP term + min-max-normalized
      ``ssim_unedit_part`` (today's FiVE-Bench background metric, ``1 - M_src``) instead
      -- this one lives in the R26 summary itself, so it needs no R28 input.

    For each criterion:

    - ``overall``: normalized ONCE across the twelve cells' DATASET-WIDE means (i.e. one
      argmax shared by every clip). On the current CSVs, ``union`` picks
      ``(tau_bg=0, tau_fg=6)``.
    - ``per video``: normalized SEPARATELY within EACH clip's own twelve arm values, so
      the winning cell is an INDEPENDENT argmax per clip and can (and does) vary from
      page to page -- read from R26's/R28's own per-video CSVs directly, since the
      aggregated summary has already averaged the per-clip signal away.

    The ``per video`` rows are additional, not a replacement for ``overall``: they answer
    different questions ("what single cell is best on average" vs "what would the best
    per-clip choice look like"), and showing both is what makes the gap between them
    visible. Likewise ``union`` and ``part`` are additional to each other, not
    alternatives -- they disagree in general and the disagreement is itself informative.

    ⚠️ Neither criterion is a like-for-like cross-arm ranking. ``ssim_unedit_union`` is a
    PER-ARM metric, and R28 measured ``Spearman(per-arm background area, union score) =
    +0.972`` -- i.e. the per-arm union ordering is largely explained by how much
    background each arm was scored on, not by preservation. ``ssim_unedit_part`` scores
    background as ``1 - M_src`` regardless of arm, which is R28's ORIGINAL motivating
    flaw: it penalizes a shape-changing edit on pixels it was correct to change. All four
    rows are QUALITATIVE illustrations of their selection criterion, not a claim of a
    Pareto win; that verdict is read from the heatmaps and the ``*_unedit_union_fixed``
    family instead. Both criteria also tend to select ``tau_bg=0`` (background pinned to
    source), which scores well on preservation but is the arm most prone to seam/freeze
    artefacts -- another reason to read these rows against the mask row.

    Every row is first TRIMMED to ``find_closest_num_frame``, which is what run_fivebench.py
    feeds the pipeline: the source dir holds more frames than were ever edited (55 on disk
    vs 45 rendered for 0007_guitar-violin). The VAE itself round-trips n -> n, so it is the
    TRIM, not the VAE, that makes the raw source longer than the render. Sampling normalized
    time over the untrimmed source puts its t=1.0 ten frames past the end of the render and
    the rows drift apart at the tail -- which reads as a temporal artifact of the method
    when it is only an artifact of this figure. After the trim all rows are the same length
    and normalized time is exact.

    The ``M_f`` row is the pass-1 mask made visible: the source is encoded, the latents
    outside the mask are zeroed, and the result is VAE-decoded (see ``MaskDecoder``). It
    replaces an earlier ``tau_bg=0`` arm row, which showed a barely-distinguishable render
    and told the reader far less than the mask driving the whole experiment does.

(c) ``r26_mask_sanity.pdf``
    ``M_f`` drawn over source frames. This is the only output that can tell "spatial tau
    does not help" apart from "the masks were wrong" -- a mask that is empty, inverted,
    frame-misaligned, or grounding the wrong object looks perfectly healthy in every
    number in the heatmaps.

The mask row needs the VAE and therefore a GPU. Everything else is CPU-only:
pass ``--mask_row none`` to keep the whole script runnable on a login node.

Usage
-----
    python evaluation/r26_grid_figure.py \\
        --summary evaluation/csv/r26_spatial_tau.csv \\
        --r28_summary evaluation/csv/r28_union_vs_part.csv \\
        --masks /projects/dataggen/outputs/five_bench/r26_masks \\
        --out_root /projects/dataggen/outputs/five_bench/r26_spatial_tau \\
        --cases evaluation/cases.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

CONTROL = (2, 2)
EXTREME = (0, 50)  # default 'extreme tau' row: tau_bg=0 (bg pinned to source) x tau_fg=50
_HIGHER_IS_BETTER = {"clip_similarity_target_image", "psnr_unedit_part", "ssim_unedit_part",
                     "clip_similarity_target_image_edit_part"}


# ---------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------
def load_summary(path: str) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
    """``{subset: {(tau_bg, tau_fg): {metric: value}}}`` from r26_summarize.py's CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no rows")
    skip = {"subset", "tau_bg", "tau_fg", "arm", "n_clips", "is_control"}
    out: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}
    for r in rows:
        cell = (int(r["tau_bg"]), int(r["tau_fg"]))
        vals = {}
        for k, v in r.items():
            if k in skip or k.endswith("_delta") or k.endswith("_macro") or v == "":
                continue
            vals[k] = float(v)
        out.setdefault(r["subset"], {})[cell] = vals
    return out


_ARM_CELL_RE = re.compile(r"taubg(\d+)_taufg(\d+)_vp")


def load_r28_union_metric(path: str, metric: str = "ssim_unedit_union",
                          subset: str = "all") -> Dict[Tuple[int, int], float]:
    """``{(tau_bg, tau_fg): value}`` for one R28 per-arm union metric from r28_union_vs_part.csv.

    Only the tau cells are returned; the ``baseline`` / ``r7_visual_prompting`` comparator
    rows do not match ``taubg{X}_taufg{Y}_vp`` and are skipped. ``ssim_unedit_union`` is
    R28's structural-preservation score on the per-arm union background ``1 - (M_src | M_tgt)``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"R28 summary {path!r} not found. The 'best tau' row is selected from R28's "
            f"per-arm union metric ({metric}); run evaluation/r28_summarize.py first, or "
            f"point --r28_summary at the produced CSV."
        )
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and metric not in rows[0]:
        raise KeyError(
            f"{path!r} has no column {metric!r}; columns are {sorted(rows[0])}. "
            f"The R28 union metrics must be present to select the best-tau row."
        )
    out: Dict[Tuple[int, int], float] = {}
    for r in rows:
        if r.get("subset") != subset:
            continue
        m = _ARM_CELL_RE.fullmatch(r.get("arm", ""))
        if not m:
            continue
        v = r.get(metric, "")
        if v == "":
            continue
        out[(int(m.group(1)), int(m.group(2)))] = float(v)
    if not out:
        raise ValueError(
            f"{path!r}: no tau cells found for subset={subset!r}; available subsets are "
            f"{sorted({r.get('subset') for r in rows})}."
        )
    return out


def _minmax(vals: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """Min-max normalize a {cell: value} map to [0, 1]; a flat map maps to all-zeros."""
    lo, hi = min(vals.values()), max(vals.values())
    rng = hi - lo
    return {c: ((v - lo) / rng if rng > 1e-12 else 0.0) for c, v in vals.items()}


def pick_best_tau(clip_by_cell: Dict[Tuple[int, int], float],
                  ssim_by_cell: Dict[Tuple[int, int], float]) -> Tuple[
                      Tuple[int, int], Dict[Tuple[int, int], float]]:
    """Best tau cell = argmax of (normalized CLIP + normalized union-SSIM).

    Both metrics are higher-is-better, so each is min-max normalized across the shared tau
    cells and the sums are compared. Returns ``(best_cell, score_by_cell)``. Ties (rare on
    real data) break on higher CLIP then higher union-SSIM for determinism.
    """
    cells = sorted(set(clip_by_cell) & set(ssim_by_cell))
    if not cells:
        raise ValueError(
            "No tau cells are common to the CLIP summary and the R28 union summary; "
            "the two CSVs describe different arm sets."
        )
    cn = _minmax({c: clip_by_cell[c] for c in cells})
    sn = _minmax({c: ssim_by_cell[c] for c in cells})
    score = {c: cn[c] + sn[c] for c in cells}
    best = max(cells, key=lambda c: (score[c], clip_by_cell[c], ssim_by_cell[c]))
    return best, score


_PER_VIDEO_R28 = "evaluation/csv/edit{t}_FiVE_r28_{arm}_frame_stride8.csv"
_PER_VIDEO_R26 = "evaluation/csv/edit{t}_FiVE_r26_{arm}_frame_stride8.csv"


def load_edit_id_to_name(data_root: str, edit_types: Sequence[int]) -> Dict[int, Dict[int, str]]:
    """``{edit_type: {file_id: video_name}}`` from the FiVE annotation jsons.

    ``evaluate.py``'s per-video CSV ``file_id`` is the ``enumerate()`` index over the
    FULL (unfiltered, 100-video) ``edit{T}_FiVE.json`` for that edit type -- NOT a video
    name and NOT an index into ``cases.json`` -- so joining a per-video row back to a
    case requires re-reading that same json, in the same order, rather than guessing
    from ``cases.json`` alone.
    """
    out: Dict[int, Dict[int, str]] = {}
    for t in edit_types:
        p = os.path.join(data_root, "edit_prompt", f"edit{t}_FiVE.json")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found; cannot map R28 per-video file_id -> video_name for edit{t}."
            )
        with open(p, encoding="utf-8") as fh:
            ann = json.load(fh)
        out[t] = {i: item["video_name"] for i, item in enumerate(ann)}
    return out


def load_per_video_metrics(cases, arms: Sequence[Tuple[int, int]], data_root: str,
                           csv_tmpl: str, clip_metric: str = "clip_similarity_target_image",
                           ssim_metric: str = "ssim_unedit_union"
                           ) -> Dict[Tuple[int, str], Dict[Tuple[int, int], Dict[str, float]]]:
    """``{(edit_type, video_name): {(tau_bg, tau_fg): {'clip': .., 'ssim': ..}}}``.

    Reads per-video CSVs directly rather than any aggregated summary: a PER-CLIP best-tau
    selection needs the (CLIP, SSIM) pair at per-clip granularity, which only the
    per-video files carry (the ``_avg``/summary CSVs are already averaged over clips).
    ``csv_tmpl`` selects the CRITERION by selecting which evaluator's per-video CSV is
    read: R26's own (``_PER_VIDEO_R26``) carries ``ssim_unedit_part``, R28's
    (``_PER_VIDEO_R28``) carries ``ssim_unedit_union`` -- both carry
    ``clip_similarity_target_image`` too, so one file per (arm, edit_type) is the only
    input either way.
    """
    edit_types = sorted({int(c["edit_type"]) for c in cases})
    id_to_name = load_edit_id_to_name(data_root, edit_types)
    out: Dict[Tuple[int, str], Dict[Tuple[int, int], Dict[str, float]]] = {}
    for bg, fg in arms:
        arm = arm_name(bg, fg)
        for t in edit_types:
            p = csv_tmpl.format(t=t, arm=arm)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found; per-video best-tau selection needs every tau arm's "
                    f"per-video CSV (evaluate.py output). Pass --per_video_row off to "
                    f"skip this row instead."
                )
            with open(p, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            clip_col, ssim_col = f"{arm}|{clip_metric}", f"{arm}|{ssim_metric}"
            if rows and (clip_col not in rows[0] or ssim_col not in rows[0]):
                raise KeyError(f"{p!r} is missing {clip_col!r} or {ssim_col!r}")
            for row in rows:
                name = id_to_name[t].get(int(row["file_id"]))
                if name is None:
                    continue
                cv, sv = row.get(clip_col, ""), row.get(ssim_col, "")
                if cv == "" or sv == "":
                    continue
                out.setdefault((t, name), {})[(bg, fg)] = {
                    "clip": float(cv), "ssim": float(sv)}
    return out


def pick_best_tau_per_video(
        per_video: Dict[Tuple[int, str], Dict[Tuple[int, int], Dict[str, float]]]
) -> Dict[Tuple[int, str], Tuple[int, int]]:
    """``{(edit_type, video_name): best_cell}`` -- one INDEPENDENT argmax per clip.

    Reuses ``pick_best_tau``'s normalized-CLIP + normalized-SSIM criterion (whichever SSIM
    ``load_per_video_metrics`` was pointed at), but the min-max normalization is taken
    across each clip's OWN twelve arm values, not across the dataset-wide aggregate: a
    clip that is uniformly hard (or easy) on one metric does not get penalized (or
    favoured) purely for being an outlier relative to other clips.
    """
    out: Dict[Tuple[int, str], Tuple[int, int]] = {}
    for key, by_cell in per_video.items():
        clip_by_cell = {c: v["clip"] for c, v in by_cell.items()}
        ssim_by_cell = {c: v["ssim"] for c, v in by_cell.items()}
        best, _ = pick_best_tau(clip_by_cell, ssim_by_cell)
        out[key] = best
    return out


def frame_paths(d: str, exts: Sequence[str] = (".png", ".jpg", ".jpeg")) -> List[str]:
    if not os.path.isdir(d):
        return []
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(tuple(exts)))
    return [os.path.join(d, n) for n in names]


def sample_by_time(paths: Sequence[str], n: int) -> List[str]:
    """Pick ``n`` frames spread over NORMALIZED time, so sequences of different length align."""
    if not paths:
        return []
    if len(paths) == 1:
        return list(paths) * n
    idx = [int(round(t * (len(paths) - 1))) for t in np.linspace(0.0, 1.0, n)]
    return [paths[i] for i in idx]


def load_mask(path: str) -> np.ndarray:
    """Unpack a pass-1 union npz to a ``[F_latent, frame_seq_length]`` bool array."""
    with np.load(path) as d:
        shape = d["shape"]
        return np.unpackbits(d["M"], axis=-1)[:, : int(shape[1])].astype(bool)


def latent_grid(frame_seq_length: int, width: int, height: int) -> Tuple[int, int]:
    """Latent token grid ``(h, w)`` for a frame_seq_length at a given pixel aspect ratio.

    Derived rather than hardcoded: at 832x480 with 1560 tokens this yields (30, 52), i.e.
    a /16 downsample (VAE /8 then a 2x2 patchify).

    Every exact factor pair is enumerated and the one whose aspect ratio is closest to the
    image's is returned, rather than rounding sqrt and hoping it divides. The display frame
    is not always the model's input frame -- FiVE sources are 864x480 while the pipeline
    renders 832x480 -- and a near-miss rounding can land on a wrong-but-divisible factor,
    which would transpose or shear every overlay while still "working".
    """
    if frame_seq_length <= 0:
        raise ValueError(f"frame_seq_length must be positive, got {frame_seq_length}")
    target = height / float(width)
    best, best_err = None, float("inf")
    for h in range(1, frame_seq_length + 1):
        if frame_seq_length % h:
            continue
        w = frame_seq_length // h
        err = abs(math.log((h / float(w)) / target))
        if err < best_err:
            best, best_err = (h, w), err
    assert best is not None
    return best


def pixel_to_latent(p: int, n_pixel: int, n_latent: int) -> int:
    """Map a pixel frame index to its latent frame under the VAE's 4x temporal stride."""
    if n_latent <= 1 or n_pixel <= 1:
        return 0
    lat = 0 if p == 0 else (p - 1) // 4 + 1
    return int(min(max(lat, 0), n_latent - 1))


# ---------------------------------------------------------------------------------
# (a) heatmaps
# ---------------------------------------------------------------------------------
def draw_heatmaps(summary, metrics: Sequence[str], out_path: str) -> None:
    subsets = [s for s in ("all", "excl_degenerate") if s in summary]
    fig, axes = plt.subplots(len(metrics), len(subsets),
                             figsize=(5.2 * len(subsets), 4.4 * len(metrics)), squeeze=False)
    for i, metric in enumerate(metrics):
        for j, sub in enumerate(subsets):
            ax = axes[i][j]
            grid = summary[sub]
            bgs = sorted({c[0] for c in grid})
            fgs = sorted({c[1] for c in grid})
            M = np.array([[grid[(b, f)].get(metric, np.nan) for f in fgs] for b in bgs])
            higher = metric in _HIGHER_IS_BETTER
            im = ax.imshow(M, cmap="viridis" if higher else "viridis_r", aspect="auto")
            ctl = grid.get(CONTROL, {}).get(metric)
            for bi, b in enumerate(bgs):
                for fi, f in enumerate(fgs):
                    v = M[bi, fi]
                    lbl = f"{v:.4f}"
                    if ctl is not None:
                        lbl += f"\n{v - ctl:+.4f}"
                    is_ctl = (b, f) == CONTROL
                    # Contrast from the ACTUAL mapped colour, not from the value: with a
                    # reversed colormap a low value is a LIGHT cell, so keying off
                    # `v < mean` puts white text on yellow and makes cells unreadable.
                    r, g, bl, _ = im.cmap(im.norm(v))
                    lum = 0.299 * r + 0.587 * g + 0.114 * bl
                    ax.text(fi, bi, lbl + ("\n(control)" if is_ctl else ""),
                            ha="center", va="center", fontsize=8,
                            color="white" if lum < 0.5 else "black")
                    if is_ctl:
                        ax.add_patch(plt.Rectangle((fi - 0.5, bi - 0.5), 1, 1, fill=False,
                                                   edgecolor="red", linewidth=2.5))
            ax.set_xticks(range(len(fgs)), [f"$\\tau_{{fg}}$={f}" for f in fgs])
            ax.set_yticks(range(len(bgs)), [f"$\\tau_{{bg}}$={b}" for b in bgs])
            ax.set_title(f"{metric}\n({'higher' if higher else 'lower'} is better) — subset={sub}",
                         fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("R26 spatial tau — control (2,2) boxed in red; second line is delta vs control",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[r26_figures] wrote {out_path}")


# ---------------------------------------------------------------------------------
# (b) qualitative grids
# ---------------------------------------------------------------------------------
def arm_name(bg: int, fg: int) -> str:
    return f"taubg{bg}_taufg{fg}_vp"


def _closest_num_frame(x: int, a: int = 4, b: int = 3) -> Optional[int]:
    """``find_closest_num_frame`` from inference_edit_streamedit.py, copied to avoid
    importing that module (it touches CUDA at import time via wan.modules.t5)."""
    m = (x + a - 1) // (a * b)
    while m > 0:
        y = a * b * m - a + 1
        if y <= x:
            return y
        m -= 1
    return None


class MaskDecoder:
    """Render pass-1's ``M_f`` as a video by masking the SOURCE LATENTS and VAE-decoding.

    The mask is a latent-space object: one bool per DiT token, i.e. a /16 grid (VAE /8
    then a 2x2 patchify), on the VAE's 4x-compressed time axis. Drawing it as a pixel
    overlay would show where the mask is, but not what the model actually keeps. Encoding
    the source, zeroing the latents outside ``M_f``, and decoding shows the mask at the
    resolution and alignment the blend really applies it -- including the decoder's
    spatial bleed, which a nearest-neighbour overlay cannot show.

    Masked-out latents are set to ZERO in the wrapper's NORMALIZED space, i.e. the dataset
    mean latent, which decodes to a flat neutral texture rather than to black. That is a
    deliberate choice of "absent", not an artifact -- there is no true zero here.

    The source is preprocessed exactly as run_fivebench.py does it (Resize(480, 832),
    ToTensor, Normalize([0.5],[0.5]), trimmed to ``find_closest_num_frame``); any drift
    would put the latents on a different grid from the dumped masks.
    """

    def __init__(self, mask_dir: str, src_root: str, sf_root: str, device: str = "cuda"):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.mask_dir = mask_dir
        self.src_root = src_root
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((480, 832)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        sf_root = os.path.abspath(sf_root)
        if sf_root not in sys.path:
            sys.path.insert(0, sf_root)
        from utils.wan_wrapper import WanVAEWrapper
        # The wrapper resolves its checkpoint by a RELATIVE path, so build it from inside
        # the Self-Forcing tree and restore the cwd afterwards.
        cwd = os.getcwd()
        try:
            os.chdir(sf_root)
            self.vae = WanVAEWrapper().to(device).eval()
        finally:
            os.chdir(cwd)
        print(f"[r26_figures] VAE loaded on {device} for the pass-1 mask row")

    def __call__(self, case: dict):
        torch = self.torch
        t, name = int(case["edit_type"]), case["video_name"]
        mask_path = os.path.join(self.mask_dir, f"edit{t}", f"{name}.npz")
        if not os.path.exists(mask_path):
            return None
        paths = frame_paths(os.path.join(self.src_root, "images", name))
        if not paths:
            return None
        n = _closest_num_frame(len(paths))
        if not n:
            return None

        imgs = [Image.open(p).convert("RGB") for p in paths[:n]]
        x = torch.stack([self.transform(i) for i in imgs], dim=1).unsqueeze(0)
        with torch.no_grad():
            lat = self.vae.encode_to_latent(x.to(self.device, torch.float32))
            self.vae.model.clear_cache()

        M = load_mask(mask_path)
        f_lat, h, w = lat.shape[1], lat.shape[3], lat.shape[4]
        if M.shape[0] != f_lat:
            raise ValueError(
                f"mask has {M.shape[0]} latent frames but the source encodes to {f_lat}; "
                "the mask and this render are not on the same time axis"
            )
        gh, gw = latent_grid(M.shape[1], 832, 480)
        m = torch.nn.functional.interpolate(
            torch.from_numpy(M).view(-1, gh, gw).float().unsqueeze(1),
            size=(h, w), mode="nearest").squeeze(1)

        with torch.no_grad():
            px = self.vae.decode_to_pixel(lat * m.to(self.device)[None, :, None, :, :])
            self.vae.model.clear_cache()
        v = (px.float() * 0.5 + 0.5).clamp(0, 1)[0]          # [F, 3, H, W]
        out = [Image.fromarray((v[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
               for i in range(v.shape[0])]
        del lat, px
        torch.cuda.empty_cache()
        return out


def draw_grids(cases, out_root: str, src_root: str, rows_spec, n_frames: int,
               out_dir: str, mask_decoder=None, ext: str = "pdf",
               per_video_rows: Optional[
                   List[Tuple[str, Dict[Tuple[int, str], Tuple[int, int]]]]] = None) -> int:
    """One page per clip. ``mask_decoder(case) -> [PIL.Image] | None`` adds the pass-1 mask row.

    ``per_video_rows``, if given, is a list of ``(label_prefix, best_cell_by_clip)`` pairs
    -- one entry per selection CRITERION (e.g. union-SSIM, part-SSIM) -- and each adds ONE
    MORE row per clip: that clip's own best-tau cell under that criterion (independent per
    clip, see ``pick_best_tau_per_video``), inserted between the fixed ``rows_spec`` rows
    and the mask row. The label carries the chosen cell, since it can differ from one
    clip's page to the next.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for case in cases:
        t, name = int(case["edit_type"]), case["video_name"]
        src_dir = os.path.join(src_root, "images", name)
        panels: List[Tuple[str, List]] = []
        sp = frame_paths(src_dir)
        if sp:
            # TRIM the source to what the pipeline actually rendered. run_fivebench.py cuts
            # every clip to find_closest_num_frame(len) before encoding, so the source dir
            # holds MORE frames than any arm: 0007_guitar-violin is 55 on disk but only the
            # first 45 were ever edited. Sampling normalized time over the untrimmed 55 puts
            # the source row's t=1.0 ten frames past the end of the render and the rows drift
            # apart at the tail -- visible, and easy to misread as a temporal artifact of the
            # method rather than of this figure.
            n_keep = _closest_num_frame(len(sp))
            if n_keep:
                sp = sp[:n_keep]
        if sp:
            panels.append(("source", sample_by_time(sp, n_frames)))
        for label, (bg, fg) in rows_spec:
            d = os.path.join(out_root, arm_name(bg, fg), f"edit{t}", name)
            fp = frame_paths(d)
            if fp:
                # After the trim the VAE round-trips n -> n, so every row here should be the
                # same length; anything else means the render is not this source's render.
                if sp and len(fp) != len(sp):
                    print(f"[r26_figures] WARNING edit{t}/{name}: {len(fp)} rendered frames "
                          f"vs {len(sp)} trimmed source frames — rows may not align")
                panels.append((f"{label}\n({bg},{fg})", sample_by_time(fp, n_frames)))
        for label_prefix, per_video_best in (per_video_rows or []):
            cell = per_video_best.get((t, name))
            if cell is None:
                print(f"[r26_figures] no per-video best-tau cell for edit{t}/{name} "
                      f"under {label_prefix!r}; row omitted for this clip")
                continue
            bg, fg = cell
            d = os.path.join(out_root, arm_name(bg, fg), f"edit{t}", name)
            fp = frame_paths(d)
            if fp:
                if sp and len(fp) != len(sp):
                    print(f"[r26_figures] WARNING edit{t}/{name}: {len(fp)} rendered "
                          f"frames vs {len(sp)} trimmed source frames — rows may not align")
                panels.append((f"{label_prefix}\n({bg},{fg})", sample_by_time(fp, n_frames)))
        if mask_decoder is not None:
            try:
                frames = mask_decoder(case)
            except Exception as exc:  # a decode failure must not silently drop the row
                print(f"[r26_figures] mask row FAILED for edit{t}/{name}: {exc}")
                frames = None
            if frames:
                panels.append(("$M_f$ (pass 1)\nlatent-masked\nsource decode",
                               sample_by_time(frames, n_frames)))
        if len(panels) < 2:
            print(f"[r26_figures] skip edit{t}/{name}: only {len(panels)} row(s) available")
            continue

        fig, axes = plt.subplots(len(panels), n_frames,
                                 figsize=(2.6 * n_frames, 1.7 * len(panels)), squeeze=False)
        for ri, (label, items) in enumerate(panels):
            for ci in range(n_frames):
                ax = axes[ri][ci]
                ax.set_xticks([]); ax.set_yticks([])
                if ci < len(items):
                    it = items[ci]
                    ax.imshow(it if isinstance(it, Image.Image)
                              else Image.open(it).convert("RGB"))
                if ci == 0:
                    ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
                if ri == 0:
                    ax.set_title(f"t={ci / max(n_frames - 1, 1):.2f}", fontsize=7)
        fig.suptitle(f"edit{t} — {name}   (rows sampled by normalized time)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = os.path.join(out_dir, f"edit{t}_{name}.{ext}")
        fig.savefig(p)
        plt.close(fig)
        written += 1
    print(f"[r26_figures] wrote {written} qualitative grid(s) -> {out_dir}/")
    return written


# ---------------------------------------------------------------------------------
# (c) mask-sanity overlay
# ---------------------------------------------------------------------------------
def pick_sanity_clips(cases, mask_dir: str, k: int) -> List[dict]:
    """Choose clips spanning the fg-fraction range, always keeping any degenerate one.

    A degenerate clip (all-ones / all-zeros) is the single most important thing to SEE,
    so it is never dropped in favour of a nicer spread.
    """
    scored: List[Tuple[float, dict]] = []
    degenerate: List[dict] = []
    for c in cases:
        p = os.path.join(mask_dir, f"edit{int(c['edit_type'])}", f"{c['video_name']}.npz")
        if not os.path.exists(p):
            continue
        frac = float(load_mask(p).mean())
        if frac in (0.0, 1.0):
            degenerate.append(c)
        else:
            scored.append((frac, c))
    scored.sort(key=lambda x: x[0])
    chosen = list(degenerate[:k])
    if scored and len(chosen) < k:
        need = k - len(chosen)
        idx = [int(round(t * (len(scored) - 1))) for t in np.linspace(0, 1, need)]
        seen = set()
        for i in idx:
            if i not in seen:
                seen.add(i)
                chosen.append(scored[i][1])
    return chosen[:k]


def draw_mask_sanity(clips, mask_dir: str, src_root: str, n_frames: int,
                     out_path: str) -> None:
    if not clips:
        print("[r26_figures] no clips available for the mask-sanity overlay", file=sys.stderr)
        return
    fig, axes = plt.subplots(len(clips), n_frames,
                             figsize=(2.9 * n_frames, 1.9 * len(clips)), squeeze=False)
    for ri, c in enumerate(clips):
        t, name = int(c["edit_type"]), c["video_name"]
        M = load_mask(os.path.join(mask_dir, f"edit{t}", f"{name}.npz"))
        sp = frame_paths(os.path.join(src_root, "images", name))
        if not sp:
            for ci in range(n_frames):
                axes[ri][ci].set_axis_off()
            continue
        pick = [int(round(x * (len(sp) - 1))) for x in np.linspace(0, 1, n_frames)]
        w, h = Image.open(sp[0]).size
        gh, gw = latent_grid(M.shape[1], w, h)
        frac = float(M.mean())
        for ci, p in enumerate(pick):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            img = Image.open(sp[p]).convert("RGB")
            ax.imshow(img)
            lf = pixel_to_latent(p, len(sp), M.shape[0])
            m = M[lf].reshape(gh, gw)
            big = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
                img.size, Image.NEAREST)) > 127
            # Foreground only, tinted; background fully transparent. Drawing the mask as a
            # plain imshow instead would let matplotlib AUTOSCALE it per panel, so an
            # all-ones mask (0042_gym-ball) renders in the same colour that means
            # "background" everywhere else -- the exact misreading this figure must prevent.
            overlay = np.zeros(big.shape + (4,), dtype=float)
            overlay[big] = (1.0, 0.15, 0.0, 0.45)
            ax.imshow(overlay)
            if ci == 0:
                flag = "  ⚠DEGENERATE" if frac in (0.0, 1.0) else ""
                ax.set_ylabel(f"edit{t} {name}\nfg={frac:.3f}{flag}",
                              fontsize=6.5, rotation=0, ha="right", va="center")
            ax.set_title(f"px {p} → latent {lf}", fontsize=6.5)
    fig.suptitle("R26 mask sanity — union $M_f$ over source frames "
                 "— ORANGE = foreground (tau_fg), untinted = background (tau_bg). "
                 "Checks empty / inverted / misaligned / wrong-object.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[r26_figures] wrote {out_path}")


# ---------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="evaluation/csv/r26_spatial_tau.csv")
    ap.add_argument("--r28_summary", default="evaluation/csv/r28_union_vs_part.csv",
                    help="R28 union-vs-part CSV; supplies the per-arm union SSIM used "
                         "(with CLIP) to select the qualitative-grid 'best tau' row")
    ap.add_argument("--select_ssim_metric", default="ssim_unedit_union",
                    help="R28 per-arm union metric maximized (with CLIP) for the 'union' "
                         "best-tau criterion")
    ap.add_argument("--select_ssim_metric_part", default="ssim_unedit_part",
                    help="R26/FiVE-Bench background metric maximized (with CLIP) for the "
                         "'part' best-tau criterion (1 - M_src; no R28 input needed)")
    ap.add_argument("--select_subset", default="all",
                    help="subset (present in the R26 summary and the R28 union summary) "
                         "the two 'overall' best-tau selections read")
    ap.add_argument("--per_video_row", choices=("on", "off"), default="on",
                    help="'on' adds a per-clip best-tau row for EACH criterion (union, "
                         "part) -- independent argmax per clip, read from R26/R28's own "
                         "per-video CSVs; 'off' skips both if those CSVs are unavailable")
    ap.add_argument("--extreme_cell", default="%d,%d" % EXTREME,
                    help="'tau_bg,tau_fg' for a FIXED (non-selected) row shown right "
                         "after the baseline -- the two per-region extremes together, "
                         "independent of any selection criterion; set to '' to omit")
    ap.add_argument("--masks", default="/projects/dataggen/outputs/five_bench/r26_masks")
    ap.add_argument("--out_root", default="/projects/dataggen/outputs/five_bench/r26_spatial_tau")
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--src_root", default=os.path.expanduser(
        "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"),
        help="FiVE benchmark root holding images/{video}/*.jpg")
    ap.add_argument("--out_dir", default="evaluation/figures")
    ap.add_argument("--n_frames", type=int, default=5)
    ap.add_argument("--sanity_clips", type=int, default=4)
    ap.add_argument("--skip_grids", action="store_true",
                    help="skip (b); it is the slow part (one PDF per clip)")
    ap.add_argument("--mask_row", choices=("decode", "none"), default="decode",
                    help="'decode' adds a pass-1 M_f row to each grid by masking the source "
                         "LATENTS and VAE-decoding; needs a GPU. 'none' omits the row and "
                         "keeps the script CPU-only.")
    ap.add_argument("--sf_root", default="Self-Forcing_StreamEdit",
                    help="Self-Forcing tree holding utils/wan_wrapper.py and wan_models/")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cases, encoding="utf-8") as fh:
        cases = json.load(fh)
    summary = load_summary(args.summary)
    print(f"[r26_figures] {len(cases)} cases, subsets in summary: {sorted(summary)}")

    # (a) -------------------------------------------------------------------------
    metrics = [m for m in ("clip_similarity_target_image", "lpips_unedit_part")
               if any(m in v for v in next(iter(summary.values())).values())]
    if not metrics:
        print("[r26_figures] neither verdict metric present in the summary", file=sys.stderr)
        return 1
    draw_heatmaps(summary, metrics, os.path.join(args.out_dir, "r26_tau_heatmap.pdf"))

    # choose the "best tau" comparison cells ---------------------------------------
    # TWO independent selection criteria, each producing an "overall" row (one argmax
    # shared by every clip) and a "per video" row (an independent argmax per clip):
    #   - "union": normalized clip_similarity_target_image + normalized ssim_unedit_union
    #     (R28's per-arm union-mask preservation, 1 - (M_src | M_tgt_arm))
    #   - "part":  normalized clip_similarity_target_image + normalized ssim_unedit_part
    #     (today's FiVE-Bench background metric, 1 - M_src) -- lives in the R26 summary
    #     itself and needs no R28 input at all
    # Both are min-max normalized -- over the tau cells for "overall", over each clip's
    # own 12 arm values for "per video" -- and the argmax of the sum is taken.
    #
    # CAVEATS (see the module docstring): ssim_unedit_union is a PER-ARM metric that R28
    # showed is ~0.97 rank-correlated with each arm's background AREA, so the "union" rows
    # are a qualitative illustration of the criterion, not a like-for-like cross-arm win.
    # ssim_unedit_part carries R28's ORIGINAL motivating flaw instead: it scores
    # background as 1 - M_src, so a shape-changing edit is penalised on pixels it was
    # correct to change. Neither is the Pareto verdict, which lives in the heatmaps and
    # the *_unedit_union_fixed family.
    clip_grid = summary.get(args.select_subset, summary.get("all",
                                                            next(iter(summary.values()))))
    clip_by_cell = {c: v["clip_similarity_target_image"]
                    for c, v in clip_grid.items()
                    if "clip_similarity_target_image" in v}

    criteria = [
        ("union", args.select_ssim_metric,
         load_r28_union_metric(args.r28_summary, metric=args.select_ssim_metric,
                                subset=args.select_subset),
         _PER_VIDEO_R28),
        ("part", args.select_ssim_metric_part,
         {c: v[args.select_ssim_metric_part] for c, v in clip_grid.items()
          if args.select_ssim_metric_part in v},
         _PER_VIDEO_R26),
    ]

    rows_spec = [("baseline vp\n(2,2)", CONTROL)]
    if args.extreme_cell.strip():
        eb, ef = (int(x) for x in args.extreme_cell.split(","))
        rows_spec.append((f"extreme $\\tau$\n({eb},{ef})", (eb, ef)))
    per_video_rows: List[Tuple[str, Dict[Tuple[int, str], Tuple[int, int]]]] = []
    for crit_name, ssim_metric, ssim_by_cell, per_video_tmpl in criteria:
        best_cell, score = pick_best_tau(clip_by_cell, ssim_by_cell)
        print(f"[r26_figures] best tau (overall, {crit_name}) = {best_cell} by normalized "
              f"CLIP + normalized {ssim_metric} (subset={args.select_subset}); "
              f"score={score[best_cell]:.4f}")
        if best_cell == CONTROL:
            print(f"[r26_figures] NOTE: best tau (overall, {crit_name}) IS the (2,2) "
                  f"control; that row will be deduplicated in the grid.")
        rows_spec.append((f"best $\\tau$ (overall, {crit_name})\n"
                          f"({best_cell[0]},{best_cell[1]})", best_cell))

        if args.per_video_row == "on":
            # A SEPARATE, independent selection per clip (not the same argmax repeated):
            # each clip's own 12 arm values are min-max normalized and summed, so the
            # winning cell can differ clip to clip.
            tau_cells = sorted(set(clip_by_cell) & set(ssim_by_cell))
            per_video = load_per_video_metrics(cases, tau_cells, args.src_root,
                                               per_video_tmpl, ssim_metric=ssim_metric)
            per_video_best = pick_best_tau_per_video(per_video)
            dist: Dict[Tuple[int, int], int] = {}
            for c in per_video_best.values():
                dist[c] = dist.get(c, 0) + 1
            top = sorted(dist.items(), key=lambda kv: -kv[1])
            print(f"[r26_figures] best tau (per video, {crit_name}) computed for "
                  f"{len(per_video_best)}/{len(cases)} clips; cell distribution "
                  f"(most-picked first): " + ", ".join(f"{c}={n}" for c, n in top))
            missing = [c for c in cases
                       if (int(c["edit_type"]), c["video_name"]) not in per_video_best]
            if missing:
                print(f"[r26_figures] WARNING: no per-video ({crit_name}) selection for "
                      f"{len(missing)} clip(s): " + ", ".join(
                          f"edit{int(c['edit_type'])}/{c['video_name']}" for c in missing))
            per_video_rows.append((f"best $\\tau$ (per video, {crit_name})", per_video_best))

    seen, uniq = set(), []
    for label, cell in rows_spec:
        if cell not in seen:
            seen.add(cell)
            uniq.append((label, cell))

    # (b) -------------------------------------------------------------------------
    if not args.skip_grids:
        decoder = None
        if args.mask_row == "decode":
            decoder = MaskDecoder(args.masks, args.src_root, args.sf_root, args.device)
        draw_grids(cases, args.out_root, args.src_root, uniq, args.n_frames,
                   os.path.join(args.out_dir, "r26_grids"), mask_decoder=decoder,
                   per_video_rows=per_video_rows)
    else:
        print("[r26_figures] --skip_grids: qualitative grids not drawn")

    # (c) -------------------------------------------------------------------------
    clips = pick_sanity_clips(cases, args.masks, args.sanity_clips)
    names = ", ".join(f"edit{int(c['edit_type'])}/{c['video_name']}" for c in clips)
    print(f"[r26_figures] mask-sanity clips: {names}")
    draw_mask_sanity(clips, args.masks, args.src_root, args.n_frames,
                     os.path.join(args.out_dir, "r26_mask_sanity.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
