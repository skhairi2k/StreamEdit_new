"""R38 qualitative browser: every FiVE-Bench edit, source + the four N=5 arms, every frame.

For each of the 419 (edit_type, video) pairs the page shows a 5 x F grid -- rows = source
video, DINO·unblended, DINO·τ=0.9, LPIPS·unblended, LPIPS·τ=0.9; columns = all F rendered
frames -- with a frame-number ruler on top and each arm's per-clip background LPIPS, CLIP-D
and full FiVE-Acc next to its row label.

Each grid row is ONE pre-rendered JPEG strip (F thumbnails side by side), lazy-loaded, so
the page stays light although it covers ~145k frames. Rows of one edit share a horizontal
scroll container, so they stay frame-aligned.

FRAME ALIGNMENT: rendered frame i (``{i:05d}.png``, 0-based) is source frame ``{i+1:05d}.jpg``
(1-based) -- verified on 0001_bus by mean abs difference (offset +1 is the minimum). Renders
cover the first F source frames; the source's extra trailing frames are not shown.

Output (NOT in the repo; ~200 MB):  <out>/index.html, <out>/strips/*.jpg, <out>/rulers/*.png
View: `cd <out> && python -m http.server 8000`, then open the forwarded port (VS Code
forwards it automatically), or copy the folder locally and open index.html.

    python evaluation/r38_qual_html.py [--out DIR] [--thumb_w 128] [--workers 8]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r35_score import clip_index, load_clipd, load_harness  # noqa: E402

FIVE = Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark").expanduser()
FB = Path("~/Data/dataggen/outputs/five_bench").expanduser()
CSV = Path(__file__).resolve().parent / "csv"
ARMS: Tuple[Tuple[str, str], ...] = (
    ("DINO · unblended", "dino_n5_unblended"), ("DINO · τ=0.9", "dino_n5_tgt09"),
    ("LPIPS · unblended", "lpips_n5_unblended"), ("LPIPS · τ=0.9", "lpips_n5_tgt09"))
TYPE_NAME = {1: "color", 2: "object substitution (non-rigid)", 3: "object substitution (rigid)",
             4: "material", 5: "add", 6: "remove"}


def render_dir(combo: str, t: int, v: str) -> Path:
    return FB / "r38_arms" / f"r38_{combo}" / "step14" / f"edit{t}" / v


def make_strip(job: Tuple[List[str], str, int, int]) -> str:
    """Paste F frames side by side at thumbnail size; skip if already built (resumable)."""
    paths, out, tw, th = job
    if Path(out).exists():
        return out
    strip = Image.new("RGB", (tw * len(paths), th), "black")
    for i, p in enumerate(paths):
        strip.paste(Image.open(p).convert("RGB").resize((tw, th), Image.BILINEAR), (i * tw, 0))
    strip.save(out, quality=80, optimize=True)
    return out


def make_ruler(n: int, tw: int, out: Path) -> None:
    im = Image.new("RGB", (tw * n, 16), "white")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for i in range(n):
        x = i * tw
        dr.line([(x, 10), (x, 16)], fill="#999")
        if i % 5 == 0:
            dr.text((x + 2, 0), str(i), fill="#333", font=font)
    im.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=FB / "r38_qual_html")
    ap.add_argument("--thumb_w", type=int, default=128, help="thumbnail width (height keeps 480:832)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tw = args.thumb_w
    th = round(tw * 480 / 832)
    (args.out / "strips").mkdir(parents=True, exist_ok=True)
    (args.out / "rulers").mkdir(parents=True, exist_ok=True)

    clips = clip_index(FIVE)
    met = {}
    for lab, c in ARMS:
        met[lab] = {"bgLPIPS": load_harness(CSV, f"r38_{c}", "lpips_unedit_part", clips),
                    "CLIP-D": load_clipd(CSV / "r38_clip_directional.csv", f"r38_{c}", clips),
                    "FiVE-Acc": load_harness(CSV, f"r38_fiveacc_{c}", "five_acc", clips)}
    ann = {t: {e["video_name"]: e for e in json.loads((FIVE / "edit_prompt" / f"edit{t}_FiVE.json").read_text())}
           for t in TYPE_NAME}
    order = {(t, v): int(i) for (t, i), v in clips.items()}           # benchmark order (file_id)
    keys = sorted(order, key=lambda k: (k[0], order[k]))

    jobs, edits = [], []
    for t, v in keys:
        f = len(list(render_dir(ARMS[0][1], t, v).glob("*.png")))
        for _, c in ARMS:                                               # all arms must agree on F
            fc = len(list(render_dir(c, t, v).glob("*.png")))
            if fc != f:
                raise SystemExit(f"[r38_qual_html] edit{t}/{v}: {c} has {fc} frames, {ARMS[0][1]} has {f}")
        src = [FIVE / "images" / v / f"{i + 1:05d}.jpg" for i in range(f)]
        missing = [p for p in src if not p.exists()]
        if missing:
            raise SystemExit(f"[r38_qual_html] edit{t}/{v}: missing source frames, e.g. {missing[0]}")
        rows = [("source", "source", src)]
        rows += [(lab, c, [render_dir(c, t, v) / f"{i:05d}.png" for i in range(f)]) for lab, c in ARMS]
        row_files = []
        for lab, tag, paths in rows:
            name = f"edit{t}_{v}_{tag}.jpg"
            jobs.append(([str(p) for p in paths], str(args.out / "strips" / name), tw, th))
            row_files.append((lab, name))
        spread = float(np.ptp([met[lab]["bgLPIPS"][(t, v)] for lab, _ in ARMS]))
        edits.append(dict(t=t, v=v, f=f, rows=row_files, spread=spread, e=ann[t][v]))

    for n in sorted({e["f"] for e in edits}):
        make_ruler(n, tw, args.out / "rulers" / f"ruler_{n}.png")
    print(f"[r38_qual_html] {len(edits)} edits, {len(jobs)} strips, thumb {tw}x{th}, {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(make_strip, jobs, chunksize=4), 1):
            if k % 200 == 0 or k == len(jobs):
                print(f"[r38_qual_html] strips {k}/{len(jobs)}", flush=True)

    # ------------------------------------------------------------------ HTML
    def fmt_row(lab: str, t: int, v: str) -> str:
        if lab == "source":
            return "<b>source</b>"
        m = met[lab]
        return (f"<b>{html.escape(lab)}</b><br><span class=m>bgLPIPS {m['bgLPIPS'][(t, v)]:.3f} · "
                f"CLIP-D {m['CLIP-D'][(t, v)]:+.3f} · FiVE-Acc {m['FiVE-Acc'][(t, v)]:.2f}</span>")

    sections = []
    for t in TYPE_NAME:
        blocks = []
        for e in (x for x in edits if x["t"] == t):
            a = e["e"]
            rows_html = "".join(
                f"<div class=row><div class=lab>{fmt_row(lab, t, e['v'])}</div>"
                f"<img loading=lazy width={tw * e['f']} height={th} src='strips/{name}'></div>"
                for lab, name in e["rows"])
            blocks.append(
                f"<section class=edit id='e{t}_{html.escape(e['v'])}' data-type={t} data-spread={e['spread']:.5f}>"
                f"<h3>edit{t} / {html.escape(e['v'])} — '{html.escape(str(a.get('source_object', '?')))}' → "
                f"'{html.escape(str(a.get('target_object', '?')))}' <span class=m>(F={e['f']}, "
                f"bgLPIPS spread across arms {e['spread']:.3f})</span></h3>"
                f"<p class=p><b>target prompt:</b> {html.escape(str(a.get('target_prompt', '')))}</p>"
                f"<div class=scroll><div class=row><div class=lab></div>"
                f"<img width={tw * e['f']} height=16 src='rulers/ruler_{e['f']}.png'></div>{rows_html}</div></section>")
        n_t = sum(1 for x in edits if x["t"] == t)
        sections.append(f"<h2 id=t{t}>edit{t} · {TYPE_NAME[t]} ({n_t} edits)</h2><div class=group>{''.join(blocks)}</div>")
    toc = " · ".join(f"<a href='#t{t}'>edit{t} {TYPE_NAME[t]}</a>" for t in TYPE_NAME)
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>R38 Arm Comparison</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#1f1f1e}}
header{{position:sticky;top:0;background:#fcfcfb;padding:6px 0;border-bottom:1px solid #ddd;z-index:5}}
h2{{margin-top:28px;border-bottom:2px solid #1f1f1e}} h3{{margin:18px 0 2px;font-size:15px}}
.m{{color:#6b6b67;font-weight:normal;font-size:12px}} .p{{margin:0 0 4px;color:#444;font-size:12px}}
.scroll{{overflow-x:auto;border:1px solid #e6e6e3;background:#fff}}
.row{{display:flex;align-items:center;white-space:nowrap}}
.lab{{position:sticky;left:0;z-index:2;background:#fff;min-width:210px;max-width:210px;padding:0 6px;
      font-size:12px;border-right:1px solid #e6e6e3;white-space:normal;align-self:stretch;display:flex;
      flex-direction:column;justify-content:center}}
img{{display:block;flex:none}} button{{margin-left:8px}}
</style></head><body>
<header><b>R38 · N=5 draft · source + 4 arms · all {len(edits)} FiVE-Bench edits</b> — {toc}
<button onclick="reorder('spread')">order by arm disagreement</button>
<button onclick="reorder('bench')">benchmark order</button><br>
<span class=m>Rows: source, DINO·unblended, DINO·τ=0.9, LPIPS·unblended, LPIPS·τ=0.9. Columns: every rendered frame
(ruler = frame index; rendered frame i = source frame i+1). Scroll each grid horizontally; labels stay pinned.
Per-arm numbers are that clip's background LPIPS (↓), CLIP-D (↑) and full FiVE-Acc (↑).</span></header>
{''.join(sections)}
<script>
const orig = {{}};
document.querySelectorAll('.group').forEach((g, i) => orig[i] = [...g.children]);
function reorder(mode) {{
  document.querySelectorAll('.group').forEach((g, i) => {{
    const items = [...orig[i]];
    if (mode === 'spread') items.sort((a, b) => b.dataset.spread - a.dataset.spread);
    items.forEach(el => g.appendChild(el));
  }});
}}
</script></body></html>"""
    (args.out / "index.html").write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6
    print(f"[r38_qual_html] wrote {args.out / 'index.html'}  ({size:.0f} MB total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
