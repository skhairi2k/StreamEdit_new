#!/usr/bin/env python3
"""Build the self-contained R20 artifact (all images inlined as data URIs).

Emits an Artifact-ready HTML fragment (title + style + body markup + script, no
html/head/body wrappers) to the path given by --out. Every grid is downscaled to
~1000px JPEG and base64-embedded, so the page has zero external requests and can
be published as a claude.ai Artifact. A click-to-enlarge lightbox recovers the
zoom that inlined images otherwise lose.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
GRID_W = 1000
GRID_Q = 72

VP_ORDER = [("novp", "No VP"), ("vp", "VP · §4.5"), ("pvp", "Persistent VP")]
EDIT_TITLES = {1: "Edit type 1", 2: "Edit type 2", 5: "Edit type 5", 6: "Edit type 6"}

# dataviz categorical slots (for the schedule swatches)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6b7580")


def png_data_uri(path: Path) -> str:
    b = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def jpg_data_uri(path: Path, width: int, q: int) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CSS = """
:root{
 --ground:#f7f9fb; --card:#ffffff; --ink:#0f1518; --ink2:#48535d;
 --muted:#87929c; --rule:#e5eaef; --accent:#2a6fd0; --accent-soft:#eaf2fc;
 --shadow:0 1px 2px rgba(16,32,48,.05),0 8px 24px rgba(16,32,48,.05);
 --serif:"Iowan Old Style","Charter","Georgia","Times New Roman",serif;
 --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
 :root:not([data-theme="light"]){
  --ground:#0d1013; --card:#161b1f; --ink:#eef3f6; --ink2:#aeb9c2;
  --muted:#78838d; --rule:#252c32; --accent:#5aa2f2; --accent-soft:#16212e;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
 }
}
:root[data-theme="dark"]{
 --ground:#0d1013; --card:#161b1f; --ink:#eef3f6; --ink2:#aeb9c2;
 --muted:#78838d; --rule:#252c32; --accent:#5aa2f2; --accent-soft:#16212e;
 --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
:root[data-theme="light"]{
 --ground:#f7f9fb; --card:#ffffff; --ink:#0f1518; --ink2:#48535d;
 --muted:#87929c; --rule:#e5eaef; --accent:#2a6fd0; --accent-soft:#eaf2fc;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
 font-size:16px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:56px 28px 120px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;
 text-transform:uppercase;color:var(--accent);margin:0 0 14px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5vw,46px);
 line-height:1.08;letter-spacing:-.01em;margin:0 0 12px;text-wrap:balance}
.lede{font-size:19px;line-height:1.55;color:var(--ink2);max-width:64ch;margin:0 0 24px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 8px}
.chip{font-family:var(--mono);font-size:12px;color:var(--ink2);background:var(--card);
 border:1px solid var(--rule);border-radius:999px;padding:4px 12px;box-shadow:var(--shadow)}
h2{font-family:var(--serif);font-weight:600;font-size:26px;letter-spacing:-.01em;
 margin:64px 0 6px;padding-bottom:10px;border-bottom:1px solid var(--rule);text-wrap:balance}
h2 .n{font-family:var(--mono);font-size:13px;color:var(--muted);font-weight:400;
 letter-spacing:.04em;margin-left:10px}
p{color:var(--ink2);max-width:70ch}
b,strong{color:var(--ink)}
code{font-family:var(--mono);font-size:.86em;background:var(--accent-soft);
 color:var(--ink);padding:1.5px 6px;border-radius:5px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:14px;
 box-shadow:var(--shadow)}
.pad{padding:22px 24px}
.regimes{width:100%;border-collapse:collapse;margin-top:14px;font-size:15px}
.regimes td{padding:9px 4px;border-top:1px solid var(--rule);vertical-align:top;color:var(--ink2)}
.regimes tr:first-child td{border-top:none}
.regimes td:first-child{white-space:nowrap;width:150px}
.schedwrap{display:grid;grid-template-columns:1.02fr .98fr;gap:26px;align-items:start;margin-top:16px}
@media(max-width:860px){.schedwrap{grid-template-columns:1fr}}
table.sched{border-collapse:collapse;width:100%;font-size:14.5px}
table.sched th{text-align:left;color:var(--muted);font-weight:600;font-size:12px;
 letter-spacing:.05em;text-transform:uppercase;padding:0 10px 8px}
table.sched td{padding:10px;border-top:1px solid var(--rule);vertical-align:top;color:var(--ink2)}
table.sched td.name{color:var(--ink);white-space:nowrap}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:8px;vertical-align:middle}
.formula{font-family:var(--mono);font-size:12.5px;color:var(--ink)}
figure.plot{margin:0}
figure.plot img{width:100%;border:1px solid var(--rule);border-radius:12px;background:#fcfcfb}
.note{font-size:13.5px;color:var(--muted);max-width:74ch;margin-top:16px}
.howto{margin:18px 0 6px;border-left:3px solid var(--accent);background:var(--accent-soft);
 border-radius:0 12px 12px 0;padding:16px 20px;font-size:14.5px;color:var(--ink2);max-width:78ch}
.howto b{color:var(--ink)}
.clip{margin:34px 0 6px}
.clip .name{font-family:var(--mono);font-size:15px;font-weight:600;color:var(--ink)}
.clip .prompt{color:var(--ink2);font-size:14.5px}
.clip .prompt .arw{color:var(--accent);font-weight:700;padding:0 6px}
.threeup{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:10px 0 4px}
@media(max-width:820px){.threeup{grid-template-columns:1fr}}
.threeup figure{margin:0}
.threeup figcaption{font-family:var(--mono);font-size:11.5px;letter-spacing:.03em;
 color:var(--muted);text-align:center;margin-bottom:6px;text-transform:uppercase}
.threeup img{width:100%;display:block;border:1px solid var(--rule);border-radius:9px;
 background:var(--card);cursor:zoom-in;transition:box-shadow .12s,transform .12s}
.threeup img:hover{box-shadow:0 0 0 2px var(--accent)}
.edithdr{display:flex;align-items:baseline;justify-content:space-between}
.edithdr .cnt{font-family:var(--mono);font-size:12px;color:var(--muted)}
footer{margin-top:70px;padding-top:20px;border-top:1px solid var(--rule);
 font-size:13px;color:var(--muted)}
/* lightbox */
#lb{position:fixed;inset:0;background:rgba(8,11,14,.92);display:none;
 align-items:center;justify-content:center;z-index:50;padding:24px;cursor:zoom-out}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:92vh;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,.5)}
#lb .cap{position:fixed;top:16px;left:0;right:0;text-align:center;color:#cfd6dc;
 font-family:var(--mono);font-size:13px}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""

SCHEDULES = [
    ("Eq.4 (paper)", C_GREY, "s = t<sub>i+1</sub><sup>&rho;</sup>&nbsp;&nbsp;(&rho;=2)",
     "StreamGVE default — the schedule R20 replaces. Convex, tied to the noise level; blend rate rises 0.13&nbsp;&rarr;&nbsp;1.00."),
    ("cos_full", C_BLUE, "s = &frac12;(1 + cos&nbsp;&pi;p)",
     "Cosine. Holds the source anchor across the whole block, releases to target only at the very end."),
    ("cos_half", C_ORANGE, "s = &frac12;(1 + cos&nbsp;&pi;&middot;min(1,2p))",
     "Cosine, anchor fully released by the mid-point (p&nbsp;=&nbsp;&frac12;)."),
    ("cos_third", C_AQUA, "s = &frac12;(1 + cos&nbsp;&pi;&middot;min(1,3p))",
     "Cosine, earliest release to target (p&nbsp;=&nbsp;&#8531;)."),
    ("const", C_YELLOW, "s &equiv; 1&nbsp;&nbsp;(b&equiv;0)",
     "Degenerate bracket: pure source every step — the &ldquo;no-edit&rdquo; floor."),
    ("zero", C_MAGENTA, "s &equiv; 0&nbsp;&nbsp;(b&equiv;1)",
     "Degenerate bracket: no blending, pure target — the anchor never pulls (&asymp; R10 blend_off)."),
]


def build(cases: list[dict], plot_uri: str, grid_uris: dict[str, str]) -> str:
    sched_rows = "".join(
        f"<tr><td class='name'><span class='sw' style='background:{col}'></span>"
        f"<b>{esc(name)}</b></td><td class='formula'>{formula}</td>"
        f"<td>{role}</td></tr>"
        for name, col, formula, role in SCHEDULES)

    by_edit: dict[int, list[dict]] = {}
    for c in cases:
        by_edit.setdefault(c["edit_type"], []).append(c)

    results = []
    for et in sorted(by_edit):
        clips = by_edit[et]
        results.append(
            f"<h2 id='edit{et}'>{EDIT_TITLES.get(et, f'Edit type {et}')}"
            f"<span class='n'>{len(clips)} clip{'s' if len(clips)>1 else ''}</span></h2>")
        for c in clips:
            v, src, trg = c["video_name"], c.get("src_word", "?"), c.get("trg_word", "?")
            cells = []
            for mode, label in VP_ORDER:
                key = f"{mode}/{v}_edit{et}_{mode}"
                uri = grid_uris.get(key, "")
                cap = f"{esc(v)} — {label}"
                cells.append(
                    f"<figure><figcaption>{label}</figcaption>"
                    f"<img src='{uri}' alt='{cap}' data-cap='{cap}' loading='lazy'></figure>")
            results.append(
                f"<div class='clip'><span class='name'>{esc(v)}</span>"
                f"<span class='prompt'>&nbsp;&mdash;&nbsp;{esc(src)}"
                f"<span class='arw'>&rarr;</span>{esc(trg)}</span></div>"
                f"<div class='threeup'>{''.join(cells)}</div>")

    return f"""<title>R20 — Blend-Rate Schedule Sweep</title>
<style>{CSS}</style>
<div class="wrap" id="top">
<p class="eyebrow">Streaming Video Editing · Ablation R20</p>
<h1>Blend-Rate Schedule Sweep</h1>
<p class="lede">Replacing StreamGVE's Eq.4 blender rate with source-anchor
schedules indexed by the denoising step, scored under three visual-prompting
regimes across 22 FiVE-Bench clips.</p>
<div class="meta">
<span class="chip">22 clips</span><span class="chip">6 schedules</span>
<span class="chip">3 VP regimes</span><span class="chip">66 comparison grids</span>
</div>

<h2>What R20 tests</h2>
<div class="card pad">
<p style="margin-top:0">StreamGVE blends a <b>source anchor</b> into the target
branch at every denoising step. The mix is the <b>blender rate</b> b
(<code>b = 1</code> &rArr; pure target / edit, <code>b = 0</code> &rArr; pure
source / preserve). The paper ties it to the noise level via <b>Eq.4</b>,
<code>b = 1 &minus; t<sub>i+1</sub><sup>&rho;</sup></code>, so the anchor decays
on a fixed convex profile. <b>R20 asks whether that profile is right</b>: it
swaps in schedules indexed directly by the step <code>p = i/14</code> and
measures the preservation&#8202;&harr;&#8202;editability trade-off.</p>
<p style="margin-bottom:0">Each schedule runs under <b>three visual-prompting
(VP) regimes</b> — the schedule and the anchor mechanism control the same
quantity, so a schedule that helps <i>without</i> an anchor may hurt
<i>with</i> one:</p>
<table class="regimes"><tbody>
<tr><td><b>No VP</b></td><td>no anchor — the R1 baseline path.</td></tr>
<tr><td><b>VP · §4.5</b></td><td>the paper's cached first frame; enters the
target cache as an initial latent and <i>fades</i> (R7).</td></tr>
<tr><td><b>Persistent VP</b></td><td>R10's re-roped anchor bank, re-stamped every
rollout window, so it <i>never</i> fades.</td></tr>
</tbody></table>
</div>

<h2>The six schedules<span class="n">Eq.4 is the reference they replace</span></h2>
<div class="schedwrap">
<table class="sched">
<thead><tr><th>Schedule</th><th>source weight s(p)</th><th>Role</th></tr></thead>
<tbody>{sched_rows}</tbody>
</table>
<figure class="plot">
<img src="{plot_uri}" alt="Blender rate per denoising step for each schedule">
</figure>
</div>
<p class="note">The plot shows the <b>blender rate b(p) = 1 &minus; s</b> — the
target-mixing weight, which <b>rises as the video is denoised (edited)</b>: step 0
is pure noise (t = 1000), step 14 is the clean result. <code>const</code> and
<code>zero</code> <b>bracket</b> the cosines; nothing should score outside
<code>[const, zero]</code> on preservation. Caveat: the prev-key blend accumulates
source keys <i>in place</i> across steps, so a schedule that reaches b = 1 early
does not undo what earlier steps injected — expect less separation than the curves
suggest.</p>

<h2>Qualitative results<span class="n">3 VP regimes side by side</span></h2>
<div class="howto">
<b>How to read each grid.</b> Rows are <b>Source</b> then the six schedules
(Eq.4, cos_full, cos_half, cos_third, const, zero); columns are five frames sampled
evenly across the clip (first &amp; last always shown). The three grids per clip are
the same schedules under <b>No&nbsp;VP · VP&nbsp;§4.5 · Persistent&nbsp;VP</b>.
<b>Click any grid to enlarge.</b> The source row is aligned <b>1:1</b> (the edit is
the first N source frames after prefix-truncation to a valid length); the Eq.4 row
maps to <code>baseline</code> / <code>r7_visual_prompting</code> / <code>paper_pvp</code>.
</div>
{''.join(results)}

<footer>Generated by <code>evaluation/r20_artifact.py</code>. Grids downscaled to
{GRID_W}px for embedding; full-resolution versions live in
<code>figures/r20_grids/</code>. Quantitative metrics
(<code>r20_blend_sched.csv</code>) are produced separately by
<code>r20_summarize.py</code> once the FiVE evaluation completes.</footer>
</div>

<div id="lb"><div class="cap"></div><img alt=""></div>
<script>
(function(){{
 var lb=document.getElementById('lb'),img=lb.querySelector('img'),cap=lb.querySelector('.cap');
 document.querySelectorAll('.threeup img').forEach(function(el){{
  el.addEventListener('click',function(){{
   img.src=el.src;img.alt=el.alt;cap.textContent=el.getAttribute('data-cap')||'';
   lb.classList.add('on');
  }});
 }});
 lb.addEventListener('click',function(){{lb.classList.remove('on');img.src='';}});
 document.addEventListener('keydown',function(e){{if(e.key==='Escape')lb.classList.remove('on');}});
}})();
</script>
"""


def main() -> None:
    cases = json.load(open(REPO / "evaluation/cases.json"))
    plot_uri = png_data_uri(REPO / "figures/r20_schedule_curves.png")
    grid_uris: dict[str, str] = {}
    for c in cases:
        et, v = c["edit_type"], c["video_name"]
        for mode, _ in VP_ORDER:
            key = f"{mode}/{v}_edit{et}_{mode}"
            p = REPO / f"figures/r20_grids/{key}.png"
            grid_uris[key] = jpg_data_uri(p, GRID_W, GRID_Q)
    html = build(cases, plot_uri, grid_uris)
    out = Path(REPO / "evaluation" / "_r20_artifact.html")
    if len(__import__("sys").argv) > 1:
        out = Path(__import__("sys").argv[1])
    out.write_text(html, encoding="utf-8")
    print(f"[r20_artifact] wrote {out}  ({len(html)/1e6:.1f} MB, {len(grid_uris)} grids)")


if __name__ == "__main__":
    main()
