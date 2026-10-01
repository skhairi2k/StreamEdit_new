#!/usr/bin/env python3
"""Generate the R20 HTML report + the schedule-curve figure.

Outputs
    figures/r20_schedule_curves.png   the 5 schedules (+ Eq.4 reference) as s(p)
    r20_report.html                   standalone page, opened locally in a browser

The page references the per-video grid PNGs (figures/r20_grids/{mode}/...) by
relative path, so keep it at repo root. Colours come from the dataviz reference
categorical palette (slots 1-5, documented order) on the light surface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# dataviz reference palette (light surface) ---------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# categorical slots 1..5 (fixed documented order)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

NUM_STEPS = 15  # step=15


def s_curves():
    """Source-anchoring weight s(p) per denoising step for every schedule."""
    idx = list(range(NUM_STEPS))
    p = [i / (NUM_STEPS - 1) for i in idx]
    return idx, {
        # Eq.4 (paper): blender_rate = 1 - t_{i+1}^2, t linear 1->0 over 15 steps
        # => s = t_{i+1}^2 = ((N-1-i)/N)^2, reproduces the 0.13->1.00 blender range.
        "Eq.4 (paper)": [((NUM_STEPS - 1 - i) / NUM_STEPS) ** 2 for i in idx],
        "cos_full":  [0.5 * (1 + math.cos(math.pi * pp)) for pp in p],
        "cos_half":  [0.5 * (1 + math.cos(math.pi * min(1.0, 2 * pp))) for pp in p],
        "cos_third": [0.5 * (1 + math.cos(math.pi * min(1.0, 3 * pp))) for pp in p],
        "const":     [1.0 for _ in p],
        "zero":      [0.0 for _ in p],
    }


# name -> (colour, linestyle, dash reference?)
STYLE = {
    "Eq.4 (paper)": (INK2,     (0, (1, 1)), 1.6),   # dotted grey reference
    "cos_full":     (C_BLUE,   "-",          2.2),
    "cos_half":     (C_ORANGE, "-",          2.2),
    "cos_third":    (C_AQUA,   "-",          2.2),
    "const":        (C_YELLOW, (0, (5, 2)),  2.2),   # dashed = degenerate bracket
    "zero":         (C_MAGENTA,(0, (5, 2)),  2.2),
}


def make_schedule_figure(out: Path) -> None:
    idx, curves = s_curves()
    # plot the BLENDER RATE itself, b(p) = 1 - s(p): the target-mixing weight that
    # RISES as the video is denoised (edited). Step 0 = t=1000 (pure noise),
    # step 14 = clean, so left->right is the editing direction.
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # the 4 varying schedules carry a framed legend; const/zero are flat
    # degenerate brackets, labelled inline on their own lines.
    for name, ys in curves.items():
        col, ls, lw = STYLE[name]
        b = [1.0 - y for y in ys]                 # blender rate
        lab = name if name not in ("const", "zero") else "_nolegend_"
        ax.plot(idx, b, color=col, linestyle=ls, linewidth=lw,
                marker="o", markersize=3.5, markerfacecolor=col,
                markeredgecolor=SURFACE, markeredgewidth=0.5, label=lab, zorder=3)

    # inline labels for the two bracket lines (const -> b=0, zero -> b=1)
    ax.text(4.0, -0.052, "const  ·  b ≡ 0  →  pure source every step (no-edit floor)",
            fontsize=8, color="#b07a00", va="top", ha="left", fontweight="bold")
    ax.text(4.0, 1.035, "zero  ·  b ≡ 1  →  pure target, no blending (anchor never pulls)",
            fontsize=8, color="#c04f7a", va="bottom", ha="left", fontweight="bold")

    ax.set_xlim(-0.3, NUM_STEPS - 0.5)
    ax.set_ylim(-0.10, 1.10)
    ax.set_xticks(range(0, NUM_STEPS, 2))
    ax.set_xlabel("denoising step  i   (0 = pure noise t=1000  →  14 = clean)   ·   editing direction →",
                  color=INK2, fontsize=9.5)
    ax.set_ylabel("blender rate  b(p) = 1 − s\n(1 = pure target / edit,  0 = pure source / preserve)",
                  color=INK2, fontsize=10)
    ax.set_title("R20 blend schedules — how the blend rate rises as the video is edited",
                 color=INK, fontsize=11, pad=10)
    ax.axhline(1.0, color=GRID, lw=1, zorder=0)
    ax.axhline(0.0, color=GRID, lw=1, zorder=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    leg = ax.legend(loc="upper left", frameon=True, fontsize=9, ncol=1,
                    labelcolor=INK2, bbox_to_anchor=(0.015, 0.965),
                    handlelength=2.4)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor("#d5d4cc")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.94)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


VP_ORDER = [("novp", "No VP"), ("vp", "VP (§4.5)"), ("pvp", "Persistent VP")]
EDIT_TITLES = {1: "edit 1", 2: "edit 2", 5: "edit 5", 6: "edit 6"}

CSS = """
:root{--surface:#fcfcfb;--plane:#f4f3ef;--ink:#0b0b0b;--ink2:#52514e;
 --muted:#898781;--grid:#e1e0d9;--rule:#dcdbd3;--blue:#2a78d6;--card:#ffffff;}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1200px;margin:0 auto;padding:40px 28px 96px;}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:21px;margin:44px 0 14px;padding-bottom:7px;border-bottom:2px solid var(--rule)}
h3{font-size:16px;margin:26px 0 4px}
.sub{color:var(--ink2);font-size:15px;margin:0 0 8px}
p{color:var(--ink2)}
code{background:#f0efe9;padding:1px 5px;border-radius:4px;font-size:13px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#2a2a28}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;
 padding:18px 20px;margin:14px 0}
.grid-sched{display:grid;grid-template-columns:1.05fr .95fr;gap:22px;align-items:center}
.grid-sched img{width:100%;border:1px solid var(--rule);border-radius:8px;background:#fff}
table.sched{border-collapse:collapse;width:100%;font-size:13.5px}
table.sched th,table.sched td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
table.sched th{color:var(--muted);font-weight:600}
.swatch{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:7px;vertical-align:middle}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#2a2a28}
.threeup{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:6px 0 4px}
.threeup figure{margin:0}
.threeup figcaption{font-size:12px;color:var(--muted);text-align:center;margin-bottom:4px;font-weight:600;letter-spacing:.02em}
.threeup img{width:100%;border:1px solid var(--rule);border-radius:6px;background:#fff;
 transition:box-shadow .12s;cursor:zoom-in}
.threeup img:hover{box-shadow:0 0 0 2px var(--blue)}
.vid{margin:30px 0 8px;padding-top:8px}
.vid .name{font-weight:700;font-size:16px}
.vid .prompt{color:var(--ink2);font-size:14px}
.vid .prompt .arrow{color:var(--blue);font-weight:700;padding:0 4px}
.legend{font-size:13px;color:var(--ink2);background:#fbfaf6;border:1px solid var(--rule);
 border-radius:8px;padding:12px 16px;margin:12px 0}
.note{font-size:13px;color:var(--muted)}
a.top{font-size:12px;color:var(--blue);text-decoration:none;float:right}
.tag{display:inline-block;font-size:11px;color:#fff;background:var(--muted);
 border-radius:5px;padding:1px 7px;margin-left:8px;vertical-align:middle;font-weight:600}
"""


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_html(cases: list[dict]) -> str:
    rows = "".join(
        f"<tr><td><span class='swatch' style='background:{col}'></span>"
        f"<b>{esc(name)}</b></td><td class='mono'>{formula}</td><td>{role}</td></tr>"
        for name, col, formula, role in [
            ("Eq.4 (paper)", INK2, "s = t<sub>i+1</sub><sup>ρ</sup>&nbsp;(ρ=2)",
             "StreamGVE default — the schedule R20 replaces (convex, tied to noise level)"),
            ("cos_full", C_BLUE, "s = ½(1+cos&nbsp;πp)",
             "cosine, holds the anchor across the whole block, releases only at the end"),
            ("cos_half", C_ORANGE, "s = ½(1+cos&nbsp;π·min(1,2p))",
             "cosine, anchor fully released by the mid-point (p=½)"),
            ("cos_third", C_AQUA, "s = ½(1+cos&nbsp;π·min(1,3p))",
             "cosine, earliest release (p=⅓)"),
            ("const", C_YELLOW, "s ≡ 1",
             "degenerate bracket: pure source every step — the &ldquo;no-edit&rdquo; floor"),
            ("zero", C_MAGENTA, "s ≡ 0",
             "degenerate bracket: no blending — the anchor never pulls (≈ R10 blend_off)"),
        ])

    # per-video 3-up grids, grouped by edit type
    body = []
    by_edit: dict[int, list[dict]] = {}
    for c in cases:
        by_edit.setdefault(c["edit_type"], []).append(c)

    for et in sorted(by_edit):
        body.append(f"<h2 id='edit{et}'>{EDIT_TITLES.get(et, f'edit {et}')} "
                    f"<span class='tag'>{len(by_edit[et])} clip"
                    f"{'s' if len(by_edit[et])>1 else ''}</span>"
                    f"<a class='top' href='#top'>↑ top</a></h2>")
        for c in by_edit[et]:
            v, src, trg = c["video_name"], c.get("src_word", "?"), c.get("trg_word", "?")
            body.append(
                f"<div class='vid'><span class='name'>{esc(v)}</span> "
                f"<span class='prompt'>&nbsp;—&nbsp; {esc(src)}"
                f"<span class='arrow'>→</span>{esc(trg)}</span></div>")
            cells = []
            for mode, label in VP_ORDER:
                img = f"figures/r20_grids/{mode}/{v}_edit{et}_{mode}.png"
                cells.append(
                    f"<figure><figcaption>{label}</figcaption>"
                    f"<a href='{img}' target='_blank'><img src='{img}' "
                    f"alt='{esc(v)} {label}' loading='lazy'></a></figure>")
            body.append("<div class='threeup'>" + "".join(cells) + "</div>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R20 — Blend-Rate Schedule Sweep</title>
<style>{CSS}</style></head>
<body><div class="wrap" id="top">
<h1>R20 — Blend-Rate Schedule Sweep</h1>
<p class="sub">Replacing StreamGVE's Eq.4 blender rate with source-anchor
schedules, scored under three visual-prompting regimes on 22 FiVE-Bench clips.</p>

<h2>What R20 tests</h2>
<div class="card">
<p>StreamGVE blends a <b>source anchor</b> into the target branch at every
denoising step. The mix is set by a <b>blender rate</b>
(<code>blender_rate = 1</code> ⇒ pure target/edit, <code>0</code> ⇒ pure
source/preserve). The paper ties it to the noise level via
<b>Eq.4</b>: <code>blender_rate = 1 − t<sub>i+1</sub><sup>ρ</sup></code>, so the
source anchor decays on a fixed convex profile (rising 0.13 → 1.00 across the 15
steps). <b>R20 asks whether that profile is the right one</b>: it swaps in
schedules indexed directly by the denoising step and measures the
preservation ↔ editability trade-off.</p>
<p style="margin-bottom:0">Every schedule is run under <b>three
visual-prompting (VP) regimes</b>, because the schedule and the anchor mechanism
control the same quantity — how hard the target is pulled toward a reference — so
a schedule that helps <i>without</i> an anchor may hurt <i>with</i> one:</p>
<table class="sched" style="margin-top:10px"><tbody>
<tr><td><b>No VP</b></td><td>no anchor (R1 baseline path)</td></tr>
<tr><td><b>VP (§4.5)</b></td><td>the paper's cached first frame — enters the
target cache as an initial latent and <i>fades</i> (R7)</td></tr>
<tr><td><b>Persistent VP</b></td><td>R10's re-roped anchor bank — re-stamped
every rollout window, so it <i>never</i> fades</td></tr>
</tbody></table>
</div>

<h2>The 5 schedules</h2>
<p class="sub">All five are functions of the normalised step <code>p = i/14</code>.
<b>s</b> is the source-anchoring weight; <code>blender_rate = 1 − s</code>. Eq.4
is the reference they replace.</p>
<div class="card grid-sched">
<div>
<table class="sched"><thead><tr><th>schedule</th><th>s(p)</th><th>role</th></tr></thead>
<tbody>{rows}</tbody></table>
</div>
<div><img src="figures/r20_schedule_curves.png"
 alt="source-anchoring weight per denoising step for each schedule"></div>
</div>
<p class="note">The two degenerate schedules <b>bracket</b> the cosines: nothing
should score outside <code>[const, zero]</code> on preservation. A caveat from
the implementation — the prev-key blend accumulates source keys <i>in place</i>
across steps, so a schedule that reaches <code>s=0</code> early does not undo what
earlier steps injected; expect less separation than the curves suggest.</p>

<h2>Qualitative results — 3 VP modes side by side</h2>
<div class="legend">
<b>How to read each grid.</b> Rows = <b>Source</b> then the 6 schedules
(Eq.4, cos_full, cos_half, cos_third, const, zero); columns = 5 frames sampled
evenly across the clip (first &amp; last always shown). The three grids per video
are the same schedules under <b>No VP · VP (§4.5) · Persistent VP</b>. Click any
grid to open it full-resolution. The source row is aligned <i>1:1</i> (the edit
is the first N source frames after prefix-truncation to a valid length), and the
Eq.4 row maps to
<code>baseline</code> / <code>r7_visual_prompting</code> / <code>paper_pvp</code>
respectively.
</div>
{''.join(body)}
<p class="note" style="margin-top:40px">Generated by
<code>evaluation/r20_report.py</code>. Metrics table
(<code>r20_blend_sched.csv</code>) is produced separately by
<code>r20_summarize.py</code> once the FiVE evaluation completes.</p>
</div></body></html>
"""


def main() -> None:
    cases = json.load(open(REPO / "evaluation/cases.json"))
    make_schedule_figure(REPO / "figures/r20_schedule_curves.png")
    html = build_html(cases)
    out = REPO / "r20_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"[r20_report] schedule figure -> figures/r20_schedule_curves.png")
    print(f"[r20_report] wrote {out}  ({len(cases)} videos x 3 VP modes)")


if __name__ == "__main__":
    main()
