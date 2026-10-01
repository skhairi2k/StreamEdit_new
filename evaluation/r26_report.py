#!/usr/bin/env python
"""Generate an HTML report summarising R26 (spatial tau via mask union).

Writes PNG previews alongside the HTML (default) or optionally embeds them as
base64. Each clip section shows the R26 grid plus the matching R25 IoU-adaptive
grid underneath.

Usage
-----
    python evaluation/r26_report.py \\
        --cases evaluation/cases.json \\
        --summary evaluation/csv/r26_spatial_tau.csv \\
        --out evaluation/figures/r26_report.html

    # then open via a local server (recommended — file:// is unreliable for large pages):
    cd evaluation/figures && python -m http.server 8765
    # → http://localhost:8765/r26_report.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # pymupdf


def _pdf_page_png(path: str, dpi: int = 120) -> bytes:
    doc = fitz.open(path)
    try:
        pix = doc[0].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _img_tag(png: bytes, alt: str, rel_path: Optional[str], embed: bool) -> str:
    if embed:
        enc = base64.b64encode(png).decode("ascii")
        src = f"data:image/png;base64,{enc}"
    else:
        src = rel_path or ""
    return f'<img src="{src}" alt="{alt}" loading="lazy">'


def _write_png(assets_dir: Path, name: str, png: bytes) -> str:
    assets_dir.mkdir(parents=True, exist_ok=True)
    p = assets_dir / f"{name}.png"
    p.write_bytes(png)
    return p.name


def _load_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _load_r26_summary(path: str) -> List[dict]:
    rows = [r for r in _load_csv(path) if r.get("subset") == "all"]
    ctrl = next(r for r in rows if r.get("is_control") == "1")
    out = []
    for r in rows:
        if r.get("is_control") == "1":
            continue
        out.append({
            "cell": f"({r['tau_bg']},{r['tau_fg']})",
            "clip": float(r["clip_similarity_target_image"]),
            "clip_d": float(r["clip_similarity_target_image"])
                    - float(ctrl["clip_similarity_target_image"]),
            "lpips": float(r["lpips_unedit_part"]),
            "lpips_d": float(r["lpips_unedit_part"])
                     - float(ctrl["lpips_unedit_part"]),
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--summary", default="evaluation/csv/r26_spatial_tau.csv")
    ap.add_argument("--r25_iou", default="evaluation/csv/r25_iou.csv")
    ap.add_argument("--r25_tau_map", default="evaluation/csv/r25_tau_map.csv")
    ap.add_argument("--r25_adaptive", default="evaluation/csv/r25_adaptive.csv")
    ap.add_argument("--r26_grids", default="evaluation/figures/r26_grids")
    ap.add_argument("--r25_grids", default="evaluation/figures/r25_grids")
    ap.add_argument("--heatmap", default="evaluation/figures/r26_tau_heatmap.pdf")
    ap.add_argument("--mask_sanity", default="evaluation/figures/r26_mask_sanity.pdf")
    ap.add_argument("--out", default="evaluation/figures/r26_report.html")
    ap.add_argument("--assets_dir", default="evaluation/figures/r26_report_assets",
                    help="PNG output folder (referenced by HTML unless --embed)")
    ap.add_argument("--embed", action="store_true",
                    help="inline images as base64 (creates a huge single file; not recommended)")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args(argv)

    out = Path(args.out)
    assets_dir = Path(args.assets_dir)
    assets_prefix = f"{assets_dir.name}/" if not args.embed else ""

    with open(args.cases, encoding="utf-8") as fh:
        cases = json.load(fh)

    iou_by_case = {r["case_id"]: r for r in _load_csv(args.r25_iou)}
    tau_by_vt = {(r["video_name"], int(r["edit_type"])): r
                 for r in _load_csv(args.r25_tau_map)}
    r25_adaptive_rows = [r for r in _load_csv(args.r25_adaptive)
                         if not r["video_name"].startswith("MEAN_")]
    r25_by_case: Dict[str, dict] = {}
    for c in cases:
        vn, et = c["video_name"], int(c["edit_type"])
        for r in r25_adaptive_rows:
            if r["video_name"] == vn and int(r["edit_type"]) == et:
                r25_by_case[c["case_id"]] = r
                break

    def _fig(pdf_path: str, asset_name: str, alt: str) -> str:
        if not os.path.exists(pdf_path):
            return ""
        png = _pdf_page_png(pdf_path, args.dpi)
        rel = _write_png(assets_dir, asset_name, png) if not args.embed else None
        rel_path = f"{assets_prefix}{rel}" if rel else None
        return _img_tag(png, alt, rel_path, args.embed)

    r26_cells = _load_r26_summary(args.summary)
    heatmap_img = _fig(args.heatmap, "heatmap", "R26 heatmap")
    mask_img = _fig(args.mask_sanity, "mask_sanity", "R26 mask sanity")

    video_cards: List[str] = []
    for c in cases:
        cid = c["case_id"]
        t, name = int(c["edit_type"]), c["video_name"]
        r26_pdf = os.path.join(args.r26_grids, f"edit{t}_{name}.pdf")
        r25_pdf = os.path.join(args.r25_grids, f"{cid}.pdf")
        iou_row = iou_by_case.get(cid, {})
        tau_row = tau_by_vt.get((name, t), {})
        r25 = r25_by_case.get(cid, {})

        r26_png = _fig(r26_pdf, f"r26_{cid}", f"R26 grid edit{t}/{name}")
        r25_png = _fig(r25_pdf, f"r25_{cid}", f"R25 IoU grid {cid}")

        note = iou_row.get("note", "")
        degenerate = "0042_gym-ball" in cid or "DEGENERATE" in note or "TYPE6" in note

        clip_d = r25.get("delta|clip_similarity_target_image", "")
        lpips_d = r25.get("delta|lpips_unedit_part", "")

        video_cards.append(f"""
<section class="video-card" id="{cid}">
  <header>
    <h3><code>{cid}</code> · edit type {t}</h3>
    <p class="edit-desc"><strong>{c['src_word']}</strong> → <strong>{c['trg_word']}</strong></p>
    <p class="instruction">{c['instruction']}</p>
  </header>
  <div class="metrics-row">
    <span class="pill">R25 IoU = {_f(iou_row.get('iou', tau_row.get('iou')))}</span>
    <span class="pill">R25 routed τ = {_f(tau_row.get('tau', r25.get('tau')), 2)}</span>
    <span class="pill">R25 ΔCLIP = {_f(clip_d, 2)}</span>
    <span class="pill">R25 ΔLPIPS = {_f(lpips_d, 4)}</span>
    {"<span class='pill warn'>degenerate M_f (all-ones) — not evidence for/against spatial τ</span>" if degenerate else ""}
    {f"<span class='pill note'>{note}</span>" if note else ""}
  </div>
  <figure class="grid-figure">
    <figcaption>R26 spatial τ qualitative grid</figcaption>
    {r26_png or "<p class='missing'>R26 grid PDF not found</p>"}
  </figure>
  <figure class="grid-figure r25">
    <figcaption>R25 IoU-adaptive τ (same clip) — source / Eq.4 baseline (ρ=2) / IoU mask overlay / adaptive render</figcaption>
    {r25_png or "<p class='missing'>R25 grid PDF not found</p>"}
  </figure>
</section>""")

    # quantitative table rows
    table_rows = "\n".join(
        f"<tr><td>{r['cell']}</td><td>{r['clip']:.4f}</td>"
        f"<td class='{'pos' if r['clip_d']>0 else 'neg'}'>{r['clip_d']:+.4f}</td>"
        f"<td>{r['lpips']:.4f}</td>"
        f"<td class='{'neg' if r['lpips_d']>0 else 'pos'}'>{r['lpips_d']:+.4f}</td></tr>"
        for r in sorted(r26_cells, key=lambda x: (int(x['cell'].split(",")[0][1:]),
                                                    int(x['cell'].split(",")[1][:-1])))
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R26 — Spatial Tau via Mask Union</title>
<style>
:root {{
  --bg: #0f1117; --surface: #1a1d27; --border: #2a2f3d;
  --text: #e8eaed; --muted: #9aa0a6; --accent: #7cacf8;
  --pos: #6fcf97; --neg: #ff6b6b; --warn: #f2c94c;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: "Segoe UI", system-ui, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.55; margin: 0; padding: 0;
}}
header.hero {{
  background: linear-gradient(135deg, #1a2332 0%, #0f1117 60%);
  border-bottom: 1px solid var(--border);
  padding: 2.5rem 2rem 2rem; max-width: 1100px; margin: 0 auto;
}}
header.hero h1 {{ margin: 0 0 .25rem; font-size: 1.9rem; }}
header.hero .subtitle {{ color: var(--muted); font-size: 1.05rem; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
section {{ margin-bottom: 3rem; }}
h2 {{ color: var(--accent); border-bottom: 1px solid var(--border);
     padding-bottom: .4rem; margin-top: 2.5rem; }}
h3 {{ margin: 0 0 .5rem; }}
p {{ margin: .6rem 0; }}
ul, ol {{ padding-left: 1.4rem; }}
li {{ margin: .35rem 0; }}
code {{ background: var(--surface); padding: .1em .35em; border-radius: 4px;
       font-size: .92em; }}
.figure-block {{ background: var(--surface); border: 1px solid var(--border);
                 border-radius: 10px; padding: 1rem; margin: 1rem 0; }}
.figure-block figcaption {{ color: var(--muted); font-size: .9rem; margin-bottom: .75rem; }}
.figure-block img {{ width: 100%; height: auto; border-radius: 6px; }}
table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
th, td {{ border: 1px solid var(--border); padding: .45rem .6rem; text-align: right; }}
th {{ background: var(--surface); color: var(--muted); }}
td:first-child, th:first-child {{ text-align: left; }}
.pos {{ color: var(--pos); }}
.neg {{ color: var(--neg); }}
.verdict {{
  background: #1e2433; border-left: 4px solid var(--warn);
  padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1.5rem 0;
}}
.video-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1.25rem; margin-bottom: 2rem;
}}
.video-card header {{ margin-bottom: .75rem; }}
.edit-desc {{ color: var(--muted); margin: .25rem 0; }}
.instruction {{ font-size: .92rem; color: var(--muted); font-style: italic; }}
.metrics-row {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .75rem 0 1rem; }}
.pill {{
  background: #252a38; border: 1px solid var(--border); border-radius: 999px;
  padding: .25rem .75rem; font-size: .82rem;
}}
.pill.warn {{ border-color: var(--warn); color: var(--warn); }}
.pill.note {{ border-color: #666; color: var(--muted); font-size: .78rem; }}
.grid-figure {{ margin: .75rem 0; }}
.grid-figure figcaption {{ color: var(--accent); font-weight: 600; margin-bottom: .5rem; }}
.grid-figure.r25 figcaption {{ color: #b39ddb; }}
.grid-figure img {{ width: 100%; border-radius: 6px; border: 1px solid var(--border); }}
.missing {{ color: var(--neg); font-size: .9rem; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
@media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.toc a {{ color: var(--accent); text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header class="hero">
  <h1>R26 — Spatial Tau via Mask Union</h1>
  <p class="subtitle">Oracle test · 22 FiVE-Bench clips · verdict recorded 2026-09-01</p>
</header>
<main>

<section id="goal">
<h2>Goal</h2>
<p>R26 asks whether making the blend-schedule exponent <strong>τ spatial</strong> — varying per latent token instead of using one global scalar — can beat StreamGVE's Eq.4 baseline (<code>blend_power = ρ = 2</code>, visual-prompt anchoring).</p>
<p>This is <em>not</em> a proposed method: the spatial mask <code>M_f</code> is <strong>privileged</strong> (it requires a full extra render pass to obtain). The experiment measures the <strong>ceiling</strong> a spatial τ field could reach under perfect oracle masks. A null result closes this direction at the single-global-cell level.</p>
<p>R26 is the spatial counterpart of <strong>R25</strong> (per-video τ routed from first-frame IoU). Where R25 asks “does τ vary with edit magnitude?”, R26 asks “does τ vary with <em>where</em> in the frame?”</p>
</section>

<section id="protocol">
<h2>Experiment protocol</h2>
<ol>
  <li><strong>Pass 1 — mask dump.</strong> Run StreamEdit with <code>--blend_sched zero</code> and <code>step=7</code> (<code>t_inj = 0.5</code>). At the existing in-loop union site, save per-latent-frame grounding masks and write their union<br>
      <code>M_f = M<sup>src</sup>_f ∪ M<sup>trg</sup>_f</code> to <code>r26_masks/edit{{T}}/{{video}}.npz</code>.</li>
  <li><strong>Pass 2 — spatial τ sweep.</strong> Re-render the same 22 clips with a cache-aligned two-level field<br>
      <code>τ(f,p) = τ_bg + (τ_fg − τ_bg) · M_f(p)</code><br>
      sweeping <code>τ_bg ∈ {{0,1,2}} × τ_fg ∈ {{2,6,10,50}}</code> (12 arms) under VP anchoring. Cell <code>(2,2)</code> is a degenerate control that must reproduce the global baseline bit-for-bit.</li>
  <li><strong>Evaluation.</strong> 16 FiVE-Bench metrics per arm on the 22-case subset (<code>cases.json</code>). Summarised in <code>r26_spatial_tau.csv</code> with deltas vs the <code>(2,2)</code> control.</li>
  <li><strong>Figures.</strong> Heatmaps, per-clip qualitative grids (up to 8 rows: source, baseline vp, extreme τ (0,50), best-τ rows under union/part criteria, pass-1 <code>M_f</code> decode), and a mask-sanity overlay.</li>
</ol>
<p><strong>Verdict metrics:</strong> <code>clip_similarity_target_image</code> (edit fidelity, higher better) vs <code>lpips_unedit_part</code> (background preservation, lower better). A cell wins only if it Pareto-beats <code>(2,2)</code> on both axes <em>and</em> masks are visually sound.</p>
</section>

<section id="quantitative">
<h2>Quantitative results</h2>
<div class="figure-block">
  <figcaption>τ heatmaps — CLIP (top) and LPIPS on unedited part (bottom); control <code>(2,2)</code> boxed in red</figcaption>
  {heatmap_img}
</div>

<p>Across all 12 non-control cells in subset <code>all</code>, every arm that improves CLIP (+0.9 to +1.2 vs control) worsens background LPIPS (+0.005 to +0.045). The only cells that slightly improve LPIPS — <code>(0,2)</code> and <code>(1,2)</code> — do so with slightly worse CLIP. This is a <strong>monotone trade-off curve</strong>, not a Pareto frontier with a winner.</p>

<table>
  <thead>
    <tr><th>Cell (τ_bg, τ_fg)</th><th>CLIP</th><th>ΔCLIP</th><th>LPIPS<sub>unedit</sub></th><th>ΔLPIPS</th></tr>
  </thead>
  <tbody>
    <tr><td>(2,2) control</td><td>27.0280</td><td>0</td><td>0.2086</td><td>0</td></tr>
    {table_rows}
  </tbody>
</table>

<p>Best overall cells under the qualitative selection criteria (normalized CLIP + SSIM, argmax): <strong>(0,6)</strong> for both union-SSIM and part-SSIM criteria — but this is an illustration of a selection rule, not a Pareto win (see caveats in R28 for union-SSIM confounding).</p>
</section>

<section id="qualitative">
<h2>Qualitative results</h2>
<div class="figure-block">
  <figcaption>Mask sanity — union <code>M_f</code> over source frames (orange = foreground). Catches empty/inverted/misaligned masks.</figcaption>
  {mask_img}
</div>

<p><strong>Mask check:</strong> Only <code>0042_gym-ball</code> (1/22) is degenerate (all-ones <code>M_f</code> → uniform τ_fg, no spatial signal). <code>0034_cows</code>' broad mask is expected: cow→dragon union includes wing pixels in original background (R28).</p>
<p><strong>τ_bg=0 artefact check:</strong> No seam/freeze/ghosting in reviewed clips; the <code>extreme (0,50)</code> arm shows mild background texture softening vs <code>(2,2)</code>.</p>

<p>Below: each clip's R26 grid followed by the <strong>R25 IoU-adaptive τ</strong> result on the same video (IoU measured on anchor vs source, τ routed linearly in IoU, global scalar — not spatial).</p>

<nav class="toc"><strong>Clips:</strong>
  {" · ".join(f'<a href="#{c["case_id"]}">{c["case_id"]}</a>' for c in cases)}
</nav>

{"".join(video_cards)}
</section>

<section id="conclusion">
<h2>Conclusion</h2>
<div class="verdict">
  <p><strong>Verdict (2026-09-01): direction closed at the single-global-cell level.</strong></p>
  <ul>
    <li><strong>No Pareto win:</strong> no <code>(τ_bg, τ_fg)</code> cell beats <code>(2,2)</code> on both CLIP and background LPIPS in either subset.</li>
    <li><strong>Trade-off, not routing gain:</strong> higher edit-side τ improves alignment at the cost of background fidelity — same shape as R25's constant-τ effect.</li>
    <li><strong>Masks sound</strong> except the known <code>0042_gym-ball</code> degenerate case; losses with sound masks close spatial τ as an oracle idea.</li>
    <li><strong>τ_bg=0</strong> does not produce visible seam/freeze artefacts in reviewed clips, but high τ_fg softens fine background texture.</li>
    <li><strong>Not tested here:</strong> per-video oracle pick of τ cells (R23/R25-style); the per-video “best τ” grid rows are qualitative illustrations only.</li>
  </ul>
</div>
<p>Compared to <strong>R25</strong> (scalar τ from IoU): R25 also trades CLIP for LPIPS without IoU-correlated benefit (r=+0.101, p=0.656 for ΔCLIP vs |τ−2|). R26 adds spatial splitting but does not escape the same editability/preservation frontier under oracle masks.</p>
</section>

<footer style="color:var(--muted);font-size:.85rem;margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border)">
  Generated by <code>evaluation/r26_report.py</code> · data from <code>r26_spatial_tau.csv</code>, <code>r25_iou.csv</code>, <code>r25_adaptive.csv</code>
</footer>
</main>
</body>
</html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    n_assets = len(list(assets_dir.glob("*.png"))) if assets_dir.exists() else 0
    print(f"[r26_report] wrote {out} ({len(html)/1024:.0f} KiB)")
    if not args.embed:
        print(f"[r26_report] wrote {n_assets} PNG(s) -> {assets_dir}/")
        print(f"[r26_report] open with: cd {out.parent} && python -m http.server 8765")
        print(f"[r26_report] then browse: http://localhost:8765/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
