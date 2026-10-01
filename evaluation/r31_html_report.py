#!/usr/bin/env python
"""R31 HTML report: per-video qualitative grid (source / streamedit baseline / lpips /
dino_patch / normals) directly above that SAME VIDEO's own CLIP-vs-LPIPS Pareto point,
so visual and quantitative comparison sit next to each other instead of a table averaged
over all 22 clips.

Three outputs, all local files (no GPU, no external upload):

1. Row strips ``evaluation/figures/r31_html_strips/{case_id}_{row}.png`` -- cropped
   straight from the pixels the EXISTING figures already baked in, not re-rendered:
   ``source``/``lpips``/``dino_patch``/``normals`` are pixel crops of
   ``evaluation/figures/r31_arm_grids/edit{T}_{name}_arms.png`` (``latent`` is dropped, per
   the user's row list); ``baseline`` is the ``taubg2_taufg2_vp`` (uniform b=2) row
   extracted from ``evaluation/figures/r26_grids/edit{T}_{name}.pdf`` via PyMuPDF's
   embedded-image list (that PDF's second row is exactly this control cell -- see
   ``r26_grid_figure.py``'s ``rows_spec``). Two different source figures, two different
   frame-sampling grids -- rows are NOT time-aligned frame-for-frame across the
   source/baseline split from the lpips/dino/normals split; each row is its own strip.

2. Per-video Pareto points ``evaluation/figures/r31_video_pareto/{case_id}_clip_vs_lpips.png``
   -- the SAME clip_target-vs-LPIPS axes/curves as ``r31_clip_vs_lpips.pdf``
   (``r31_score.py``), but every curve/point/frontier computed for `clips=[case_id]` alone
   instead of averaged over all 22 -- reuses ``r30_score``'s ``load_r26_metric`` /
   ``oracle_frontier`` and ``r31_score``'s ``load_r31_percli_metric`` unmodified, just
   indexed down to one clip.

3. ``evaluation/figures/r31_report.html`` -- one section per video, grid strips stacked
   above that video's Pareto point, in ``cases.json`` order (grouped by edit type).

Why crop existing figures instead of re-rendering from raw frames: this machine's
``/projects/dataggen`` mount is empty (checked at report-writing time) -- the raw stage-3
render trees only exist on nodes that had that mount, and are NOT needed here since the
qualitative grids/reference-curve PDFs were already built from them and are sitting on
disk locally.

Usage
-----
    python evaluation/r31_html_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import (  # noqa: E402
    CONST_BS,
    build_join_tables,
    load_r26_metric,
    oracle_frontier,
)
from r31_score import ARMS, load_r31_percli_metric  # noqa: E402

ROW_ORDER: Sequence[str] = ("source", "baseline", "lpips", "dino_patch", "normals")
ROW_LABELS: Dict[str, str] = {
    "source": "source",
    "baseline": "streamedit baseline\n(uniform b=2)",
    "lpips": "lpips arm",
    "dino_patch": "dino arm",
    "normals": "normals arm",
}


# --------------------------------------------------------------------------- strips ----

def _row_bands(png_path: Path, std_thresh: float = 5.0) -> List[Tuple[int, int]]:
    """Contiguous [y0, y1) bands of non-uniform rows in ``png_path``.

    ``r31_stage3_grids.py``'s arm grid is a fixed 1(title)+1(frame labels)+5(rows) layout
    at identical pixel dimensions across all 22 clips (same row labels -> same tight-bbox
    crop) -- band detection (row pixel std > threshold) recovers those bands without
    hardcoding pixel offsets that would silently break if that script's figure params
    ever change.
    """
    a = np.asarray(Image.open(png_path).convert("RGB"))
    row_std = a.reshape(a.shape[0], -1).std(axis=1)
    mask = row_std > std_thresh
    bands: List[Tuple[int, int]] = []
    in_band = False
    start = 0
    for y, v in enumerate(mask):
        if v and not in_band:
            start, in_band = y, True
        elif not v and in_band:
            bands.append((start, y))
            in_band = False
    if in_band:
        bands.append((start, len(mask)))
    return bands


def crop_arm_rows(arms_png: Path, out_dir: Path, case_id: str) -> Dict[str, Path]:
    """{'source'|'lpips'|'dino_patch'|'normals': path}, cropped from the arms PNG.

    Band order is fixed by ``r31_stage3_grids.build_arm_grid``: title, frame-index labels,
    source, lpips, dino_patch, normals, latent (dropped here -- not in ROW_ORDER).
    """
    bands = _row_bands(arms_png)
    if len(bands) != 7:
        raise ValueError(f"{arms_png}: expected 7 row bands (title, frame labels, source, "
                         f"lpips, dino_patch, normals, latent), found {len(bands)}: {bands}")
    labelled = dict(zip(("_title", "_frame_labels", "source", "lpips", "dino_patch",
                        "normals", "latent"), bands))
    im = Image.open(arms_png).convert("RGB")
    w = im.width
    out: Dict[str, Path] = {}
    for row in ("source", "lpips", "dino_patch", "normals"):
        y0, y1 = labelled[row]
        crop = im.crop((0, y0, w, y1))
        p = out_dir / f"{case_id}_{row}.png"
        crop.save(p)
        out[row] = p
    return out


def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def extract_baseline_row(r26_grid_pdf: Path, out_dir: Path, case_id: str,
                         label_w: int = 170) -> Path:
    """The ``taubg2_taufg2_vp`` (uniform b=2) row, extracted from an r26_grids PDF page.

    ``r26_grid_figure.py`` always emits ``[source, baseline vp(2,2), extreme tau, ...]`` --
    the second row (by ascending bbox y0) is always exactly this control cell, verified
    across all 22 r26_grids PDFs (7 rows x 5 cols each) before this script was written.
    Embedded raster images only (``page.get_image_info``), not the vector row-label text,
    so a matching label is drawn on a left margin with PIL to look like the other rows.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(r26_grid_pdf)
    page = doc[0]
    infos = page.get_image_info(xrefs=True)
    if not infos:
        raise ValueError(f"{r26_grid_pdf}: no embedded images")
    row_ys = sorted({round(im["bbox"][1], 1) for im in infos})
    if len(row_ys) < 2:
        raise ValueError(f"{r26_grid_pdf}: expected >=2 row bands, found {row_ys}")
    baseline_y = row_ys[1]
    row_imgs = sorted((im for im in infos if round(im["bbox"][1], 1) == baseline_y),
                      key=lambda im: im["bbox"][0])
    frames = [Image.open(__import__("io").BytesIO(doc.extract_image(im["xref"])["image"]))
             .convert("RGB") for im in row_imgs]

    h = max(f.height for f in frames)
    frames = [f.resize((round(f.width * h / f.height), h)) for f in frames]
    gap = 4
    strip_w = sum(f.width for f in frames) + gap * (len(frames) - 1)
    canvas = Image.new("RGB", (label_w + strip_w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, h // 2 - 16), ROW_LABELS["baseline"], fill="black",
                        font=_font(15), spacing=4)
    x = label_w
    for f in frames:
        canvas.paste(f, (x, 0))
        x += f.width + gap
    p = out_dir / f"{case_id}_baseline.png"
    canvas.save(p)
    return p


def label_strip(path: Path, row: str, label_w: int = 170) -> Path:
    """Prepend a left-margin text label to a bare row crop, so every strip looks uniform."""
    im = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", (label_w + im.width, im.height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, im.height // 2 - 16), ROW_LABELS[row], fill="black",
                        font=_font(15), spacing=4)
    canvas.paste(im, (label_w, 0))
    canvas.save(path)
    return path


def build_strips(case: dict, arm_grids_dir: Path, r26_grids_dir: Path,
                 out_dir: Path) -> Dict[str, Path]:
    name, t = case["video_name"], int(case["edit_type"])
    case_id = case["case_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    arms_png = arm_grids_dir / f"edit{t}_{name}_arms.png"
    r26_pdf = r26_grids_dir / f"edit{t}_{name}.pdf"
    strips: Dict[str, Path] = {}
    if arms_png.exists():
        for row, p in crop_arm_rows(arms_png, out_dir, case_id).items():
            strips[row] = label_strip(p, row)
    if r26_pdf.exists():
        strips["baseline"] = extract_baseline_row(r26_pdf, out_dir, case_id)
    return strips


# ----------------------------------------------------------------------- pareto point --

def build_video_pareto(case_id: str, clip_target: Dict[Tuple[str, int], float],
                       lpips: Dict[Tuple[str, int], float],
                       uniform_clip_target: Dict[Tuple[str, int], float],
                       uniform_lpips: Dict[Tuple[str, int], float],
                       r31_clip: Dict[str, Dict[str, float]],
                       r31_lpips: Dict[str, Dict[str, float]],
                       out_dir: Path, n_alpha: int, video_name: str, edit_type: int
                       ) -> Path:
    """This video's own clip_target-vs-LPIPS panel -- same axes/curves as
    ``r31_clip_vs_lpips.pdf``, but every curve/point restricted to ``case_id`` alone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    spatial_curve = [(clip_target[(case_id, b)], lpips[(case_id, b)], b) for b in CONST_BS]
    uniform_curve = [(uniform_clip_target[(case_id, b)], uniform_lpips[(case_id, b)], b)
                     for b in CONST_BS]
    frontier = oracle_frontier([case_id], clip_target, lpips, CONST_BS,
                               n_alpha=n_alpha, higher_is_better=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    xs = [p for p, _, _ in spatial_curve]
    ys = [c for _, c, _ in spatial_curve]
    ax.plot(xs, ys, "-o", color="black", label="constant-$b$ SPATIAL ($\\tau_{bg}$=0)",
           zorder=3)
    for p, c, b in spatial_curve:
        ax.annotate(f"$b$={b}", (p, c), textcoords="offset points", xytext=(3, 3),
                    fontsize=6.5)
    uxs = [p for p, _, _ in uniform_curve]
    uys = [c for _, c, _ in uniform_curve]
    ax.plot(uxs, uys, marker="D", color="dimgray", linestyle="--", markersize=4,
           label="uniform baseline ($b_{bg}=b_{fg}$)", zorder=2)

    cmap = plt.get_cmap("tab10")
    for i, arm in enumerate(ARMS):
        if case_id not in r31_clip.get(arm, {}) or case_id not in r31_lpips.get(arm, {}):
            continue
        ax.scatter([r31_clip[arm][case_id]], [r31_lpips[arm][case_id]], marker="*", s=220,
                  color=cmap(i % 10), edgecolor="black", linewidth=0.7, zorder=5,
                  label=f"r31_{arm} ({r31_lpips[arm][case_id]:.3f})")

    fx = [c for _, c, _ in frontier]
    fy = [y for _, _, y in frontier]
    ax.plot(fx, fy, "-", color="tab:green", linewidth=1.6, zorder=1.5,
           label="per-clip oracle ($\\alpha$ sweep)")

    ax.set_xlabel("clip_similarity_target_image (achieved edit)", fontsize=8)
    ax.set_ylabel("lpips_unedit_part (lower = better preserved)", fontsize=8)
    ax.set_title(f"edit{edit_type} {video_name} -- CLIP vs. LPIPS (this clip only)",
               fontsize=8.5)
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    out = out_dir / f"{case_id}_clip_vs_lpips.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- html ------

def write_html(cases: List[dict], strips_by_case: Dict[str, Dict[str, Path]],
               pareto_by_case: Dict[str, Path], out_path: Path, root: Path) -> None:
    def rel(p: Path) -> str:
        return str(p.relative_to(root))

    sections = []
    for c in sorted(cases, key=lambda c: (int(c["edit_type"]), c["video_name"])):
        cid, t, name = c["case_id"], int(c["edit_type"]), c["video_name"]
        strips = strips_by_case.get(cid, {})
        rows_html = "\n".join(
            f'<img class="strip" src="{rel(strips[row])}" alt="{row}">'
            for row in ROW_ORDER if row in strips)
        pareto = pareto_by_case.get(cid)
        pareto_html = (f'<img class="pareto" src="{rel(pareto)}" alt="pareto">'
                      if pareto else "<p><em>no per-clip R26/R31 data</em></p>")
        prompt = f'{c.get("src_word", "")} → {c.get("trg_word", "")}'
        sections.append(f"""
<section class="video">
  <h2>edit{t} &mdash; {name}</h2>
  <p class="prompt">{prompt}</p>
  <div class="grid">{rows_html}</div>
  <div class="pareto-wrap">{pareto_html}</div>
</section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>R31 qualitative report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #f7f7f8; color: #1a1a1a; }}
  header {{ padding: 20px 24px; background: #1a1a1a; color: #fff; }}
  header p {{ margin: 4px 0 0; color: #c9c9c9; font-size: 13px; }}
  .video {{ background: #fff; margin: 20px auto; max-width: 1100px; padding: 18px 22px;
           border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  .video h2 {{ margin: 0 0 2px; font-size: 17px; }}
  .prompt {{ margin: 0 0 12px; color: #555; font-size: 13px; font-style: italic; }}
  .grid img.strip {{ display: block; width: 100%; margin-bottom: 2px; border: 1px solid #eee; }}
  .pareto-wrap {{ margin-top: 14px; text-align: center; }}
  .pareto-wrap img.pareto {{ max-width: 560px; width: 100%; border: 1px solid #eee; }}
  nav {{ position: sticky; top: 0; background: #fff; padding: 8px 24px; border-bottom: 1px
        solid #ddd; font-size: 12px; z-index: 10; }}
  nav a {{ margin-right: 10px; color: #444; text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1 style="margin:0;font-size:19px;">R31 &mdash; per-video qualitative grid + per-video CLIP-vs-LPIPS Pareto</h1>
  <p>Each section: source / streamedit baseline (uniform b=2) / lpips / dino / normals renders,
     with THAT video's own CLIP-vs-LPIPS point (not averaged over the 22-clip set) below it.</p>
</header>
{"".join(sections)}
</body>
</html>"""
    out_path.write_text(html)


# --------------------------------------------------------------------------- main ------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--r26_csv_dir", type=Path, default=Path("evaluation/csv"))
    p.add_argument("--arm_grids_dir", type=Path,
                   default=Path("evaluation/figures/r31_arm_grids"))
    p.add_argument("--r26_grids_dir", type=Path, default=Path("evaluation/figures/r26_grids"))
    p.add_argument("--strips_out", type=Path,
                   default=Path("evaluation/figures/r31_html_strips"))
    p.add_argument("--pareto_out", type=Path,
                   default=Path("evaluation/figures/r31_video_pareto"))
    p.add_argument("--out", type=Path, default=Path("evaluation/figures/r31_report.html"))
    p.add_argument("--n_alpha", type=int, default=21)
    args = p.parse_args(argv)

    cases = json.loads(args.cases.expanduser().read_text())
    idx2vid, name2case = build_join_tables(args.data_root.expanduser(), cases)

    lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                            "lpips_unedit_part")
    clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                  "clip_similarity_target_image")
    uniform_lpips = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                    "lpips_unedit_part", method_fmt="taubg{b}_taufg{b}_vp")
    uniform_clip_target = load_r26_metric(args.r26_csv_dir, idx2vid, name2case, CONST_BS,
                                          "clip_similarity_target_image",
                                          method_fmt="taubg{b}_taufg{b}_vp")

    r31_clip: Dict[str, Dict[str, float]] = {}
    r31_lpips: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        try:
            r31_clip[arm] = load_r31_percli_metric(args.csv_dir, idx2vid, name2case, arm,
                                                    "", "clip_similarity_target_image")
            r31_lpips[arm] = load_r31_percli_metric(args.csv_dir, idx2vid, name2case, arm,
                                                     "", "lpips_unedit_part")
        except SystemExit as ex:
            print(f"[r31_html_report] SKIP r31_{arm}: {ex}")

    strips_by_case: Dict[str, Dict[str, Path]] = {}
    pareto_by_case: Dict[str, Path] = {}
    for c in cases:
        cid = c["case_id"]
        try:
            strips_by_case[cid] = build_strips(c, args.arm_grids_dir, args.r26_grids_dir,
                                               args.strips_out)
        except Exception as ex:
            print(f"[r31_html_report] STRIPS FAILED {cid}: {type(ex).__name__}: {ex}")
        try:
            pareto_by_case[cid] = build_video_pareto(
                cid, clip_target, lpips, uniform_clip_target, uniform_lpips,
                r31_clip, r31_lpips, args.pareto_out, args.n_alpha,
                c["video_name"], int(c["edit_type"]))
        except Exception as ex:
            print(f"[r31_html_report] PARETO FAILED {cid}: {type(ex).__name__}: {ex}")

    write_html(cases, strips_by_case, pareto_by_case, args.out,
              root=args.out.parent)
    print(f"[r31_html_report] wrote {args.out} "
         f"({len(strips_by_case)} grids, {len(pareto_by_case)} pareto points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
