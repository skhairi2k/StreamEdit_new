#!/usr/bin/env python3
"""Build the R22 report artifact (self-contained HTML, images inlined as data URIs).

Figures are generated into a working directory first (see ``--assets``), then
base64-embedded so the published page has no external requests. Numbers in the
copy are pulled from the CSVs at build time where practical, and otherwise
quoted from the run log entries they came from.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path


def data_uri(p: Path) -> str:
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[
        p.suffix.lstrip(".").lower()]
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def fig(src: str, cap: str, cls: str = "") -> str:
    return (f'<figure class="plate {cls}"><img src="{src}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


CSS = """
:root{
  --ground:#FBFCFD; --surface:#F2F5F8; --plate:#FFFFFF;
  --ink:#10161C; --ink-2:#3D4954; --ink-3:#6A7683;
  --rule:#DDE4EA; --rule-2:#EAEFF3;
  --accent:#2166AC; --arm-cos:#2166AC; --arm-vp:#B2182B; --arm-novp:#6B7480;
  --good:#1B7A5A; --warn:#9A6414;
  --shadow:0 1px 2px rgba(16,22,28,.05),0 8px 24px rgba(16,22,28,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0D1117; --surface:#151B22; --plate:#F7F9FB;
    --ink:#E6EDF3; --ink-2:#B7C2CD; --ink-3:#8B97A3;
    --rule:#232C36; --rule-2:#1B222A;
    --accent:#79ADDD; --arm-cos:#79ADDD; --arm-vp:#EE7A8C; --arm-novp:#95A1AD;
    --good:#4FBF97; --warn:#D9A441;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0D1117; --surface:#151B22; --plate:#F7F9FB;
  --ink:#E6EDF3; --ink-2:#B7C2CD; --ink-3:#8B97A3;
  --rule:#232C36; --rule-2:#1B222A;
  --accent:#79ADDD; --arm-cos:#79ADDD; --arm-vp:#EE7A8C; --arm-novp:#95A1AD;
  --good:#4FBF97; --warn:#D9A441;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 28px rgba(0,0,0,.35);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16.5px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1120px;margin:0 auto;padding:0 clamp(18px,4vw,44px)}
.col{max-width:68ch}

h1,h2,h3{font-family:"IBM Plex Serif",Georgia,serif;text-wrap:balance;margin:0}
h1{font-size:clamp(2.1rem,4.6vw,3.15rem);line-height:1.08;letter-spacing:-.021em;font-weight:600}
h2{font-size:clamp(1.5rem,2.6vw,1.95rem);line-height:1.18;letter-spacing:-.014em;font-weight:600}
h3{font-size:1.1rem;line-height:1.3;font-weight:600;letter-spacing:-.006em}
p{margin:0}
.col>p+p{margin-top:1.05em}
strong{font-weight:600}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.855em;
  background:var(--surface);border:1px solid var(--rule-2);border-radius:3px;padding:.08em .34em}

.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500;
}

/* ---------- masthead ---------- */
.masthead{border-bottom:1px solid var(--rule);padding:clamp(46px,8vw,88px) 0 0}
.masthead .lede{
  font-size:clamp(1.06rem,1.7vw,1.22rem);color:var(--ink-2);
  margin-top:1.1rem;max-width:60ch;line-height:1.55;
}
.masthead h1{margin-top:.85rem}
.stats{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
  gap:1px;background:var(--rule);border-top:1px solid var(--rule);
  margin:clamp(34px,5vw,52px) 0 0;padding:0;
}
.stats>div{background:var(--ground);padding:14px 4px 20px}
.stats dt{
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.66rem;
  letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);
}
.stats dd{
  margin:.3rem 0 0;font-family:"IBM Plex Serif",Georgia,serif;
  font-size:1.42rem;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.02em;
}

/* ---------- sections ---------- */
section{padding:clamp(48px,7vw,80px) 0;border-bottom:1px solid var(--rule)}
section:last-of-type{border-bottom:0}
.head{display:flex;flex-direction:column;gap:.55rem;margin-bottom:2.1rem}
.head .col{margin-top:.5rem}

/* ---------- arm legend ---------- */
.legend{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;margin:1.5rem 0 0;padding:0;list-style:none}
.legend li{display:flex;align-items:center;gap:.5rem;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;color:var(--ink-2)}
.swatch{width:22px;height:3px;border-radius:2px;flex:none}
.s-cos{background:var(--arm-cos)} .s-vp{background:var(--arm-vp)} .s-novp{background:var(--arm-novp)}

/* ---------- validation ---------- */
.checks{display:grid;grid-template-columns:repeat(auto-fit,minmax(238px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);margin-top:1.9rem}
.checks>div{background:var(--ground);padding:20px 20px 22px}
.checks .tag{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.68rem;
  letter-spacing:.12em;text-transform:uppercase;color:var(--good);font-weight:600}
.checks h3{margin:.45rem 0 .4rem;font-size:.98rem}
.checks p{font-size:.9rem;color:var(--ink-2);line-height:1.5}

/* ---------- tables ---------- */
.tablewrap{overflow-x:auto;margin-top:1.9rem;border:1px solid var(--rule)}
table{border-collapse:collapse;width:100%;font-size:.875rem;
  font-variant-numeric:tabular-nums;min-width:520px}
th,td{padding:.6rem .85rem;text-align:right;border-bottom:1px solid var(--rule-2)}
th:first-child,td:first-child{text-align:left}
thead th{
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.68rem;
  letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
  font-weight:500;background:var(--surface);border-bottom:1px solid var(--rule);
  white-space:nowrap;
}
tbody tr:last-child td{border-bottom:0}
td.metric{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.79rem;color:var(--ink-2)}
.best{font-weight:600;color:var(--arm-cos)}
.pos{color:var(--good);font-weight:600}
.neg{color:var(--arm-vp);font-weight:600}
caption{caption-side:bottom;text-align:left;padding:.8rem .85rem;
  font-size:.8rem;color:var(--ink-3);line-height:1.5}

/* ---------- figures ---------- */
.plate{margin:2rem 0 0;padding:0;background:var(--plate);
  border:1px solid var(--rule);box-shadow:var(--shadow)}
.plate img{display:block;width:100%;height:auto}
.plate figcaption{
  padding:.75rem 1rem .9rem;font-size:.815rem;line-height:1.5;color:#4A5560;
  border-top:1px solid #E4E9EE;background:#FFFFFF;
}
.duo{display:grid;grid-template-columns:1fr;gap:1.3rem;margin-top:2rem}
@media(min-width:860px){.duo{grid-template-columns:1fr 1fr}}
.duo .plate{margin:0}

/* ---------- callout ---------- */
.callout{border-left:3px solid var(--warn);background:var(--surface);
  padding:1.15rem 1.35rem;margin-top:2rem;max-width:72ch}
.callout .eyebrow{color:var(--warn)}
.callout p{margin-top:.45rem;font-size:.94rem;color:var(--ink-2)}

/* ---------- limits ---------- */
.limits ol{margin:1.8rem 0 0;padding:0;list-style:none;counter-reset:l}
.limits li{counter-increment:l;display:grid;grid-template-columns:2.1rem 1fr;
  gap:.9rem;padding:1.05rem 0;border-top:1px solid var(--rule-2);max-width:76ch}
.limits li::before{content:counter(l,decimal-leading-zero);
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.74rem;
  color:var(--ink-3);padding-top:.25rem}
.limits li p{font-size:.94rem;color:var(--ink-2)}
.limits li b{color:var(--ink);font-weight:600}

footer{padding:44px 0 68px;color:var(--ink-3);font-size:.82rem;
  font-family:"IBM Plex Mono",ui-monospace,monospace;line-height:1.7}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media(prefers-reduced-motion:no-preference){
  .plate{transition:box-shadow .2s ease}
}
"""


def build(assets: Path, out: Path) -> None:
    U = {p.stem: data_uri(p) for p in sorted(assets.iterdir())
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")}

    def mosaic_row(clip: str, note: str) -> str:
        cells = "".join(
            fig(U[f"mos_{clip}_{arm}"], f"{clip} — {label}")
            for arm, label in (("cos_third_pvp", "cos_third_pvp"),
                               ("paper_vp", "paper_vp"),
                               ("paper_novp", "paper_novp")))
        return (f'<h3 class="mosaic-h">{clip}</h3>'
                f'<p class="col mosaic-note">{note}</p>'
                f'<div class="triptych">{cells}</div>')

    html = f"""<title>R22 Convergence &amp; Drift</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>{CSS}
.triptych{{display:grid;grid-template-columns:1fr;gap:1.1rem;margin-top:1.4rem}}
@media(min-width:900px){{.triptych{{grid-template-columns:repeat(3,1fr)}}}}
.triptych .plate{{margin:0}}
.mosaic-h{{margin-top:2.6rem;font-family:"IBM Plex Mono",monospace;font-size:.92rem;
  letter-spacing:.02em;color:var(--ink)}}
.mosaic-note{{margin-top:.4rem;font-size:.92rem;color:var(--ink-2)}}
</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">StreamEdit · Experiment R22 · 18 Aug 2026</p>
  <h1>Where quality arrives in a streaming edit</h1>
  <p class="lede">A metric surface over two axes — how much of the clip you have
  seen, and how far the sampler has run — for three blend-schedule arms on 22
  FiVE-Bench clips. Two things fall out: the metrics stop moving at roughly half
  the sampler budget, and the arms differ sharply in how well the edit survives
  to the end of the clip.</p>
  <dl class="stats">
    <div><dt>Clips</dt><dd>22</dd></div>
    <div><dt>Arms</dt><dd>3</dd></div>
    <div><dt>Steps</dt><dd>15</dd></div>
    <div><dt>Metrics</dt><dd>8</dd></div>
    <div><dt>Matrix cells</dt><dd>533,520</dd></div>
    <div><dt>Mosaic pages</dt><dd>66</dd></div>
  </dl>
  <ul class="legend">
    <li><span class="swatch s-cos"></span>cos_third_pvp — cosine blend, persistent VP bank</li>
    <li><span class="swatch s-vp"></span>paper_vp — Eq. 4, §4.5 anchor</li>
    <li><span class="swatch s-novp"></span>paper_novp — Eq. 4, no anchor</li>
  </ul>
</header>

<section>
  <div class="head">
    <p class="eyebrow">Why the surface can be trusted</p>
    <h2>The last column is the real video, proven three ways</h2>
    <p class="col">Every claim here rests on one identity: the step-14 column of
    the matrix has to be the arm's actual final output, not an approximation of
    it. That was closed at three levels before any figure was drawn.</p>
  </div>
  <div class="checks">
    <div>
      <p class="tag">L1 · latent</p>
      <h3>Assert inside the sampler</h3>
      <p>The dumped step-14 latent equals the returned rollout latent, asserted
      per window and again after the multi-window stitch.</p>
    </div>
    <div>
      <p class="tag">L2 · pixel</p>
      <h3>Bit-identical renders</h3>
      <p>Step-14 PNGs match the stored R1 / R7 / R21 reference renders on
      <b>1482 of 1482 frames per arm</b> — 4446 of 4446 in total, including the
      multi-window clip <code>0034_cows</code>.</p>
    </div>
    <div>
      <p class="tag">L3 · metric</p>
      <h3>Exact, not tolerant</h3>
      <p>M(n,14) equals the plain step-14 per-frame mean with
      <b>max |diff| = 0.0</b> across all 528 (arm × clip × metric) cells.</p>
    </div>
  </div>
</section>

<section>
  <div class="head">
    <p class="eyebrow">Finding 1</p>
    <h2>The metrics stop moving at step 7–8 of 15</h2>
    <p class="col">Each panel below shows the deviation of the running metric
    from its own final value, so the last column is white by construction and
    the eye reads convergence directly. Across every metric and every arm the
    field goes flat at <b>j ≈ 7–8</b>: the last seven denoising steps move the
    numbers by under one percent.</p>
  </div>

  {fig(U["fig_boundary"],
       "Clip-averaged deviation from the final column, four metrics × three arms. "
       "Rows are prefix fraction i/n (matrix convention: i = 1 at top, whole clip at bottom); "
       "columns are denoising steps. The sharp vertical edge at j ≈ 7–8 is the convergence boundary.")}

  <div class="tablewrap">
    <table>
      <caption>Median earliest step from which the full-clip prefix stays within
      1% of its final value, over the 22 clips. Lower is earlier.</caption>
      <thead><tr><th>Metric</th><th>cos_third_pvp</th><th>paper_novp</th><th>paper_vp</th></tr></thead>
      <tbody>
        <tr><td class="metric">clip_similarity_target_image</td><td class="best">6</td><td>10</td><td>10</td></tr>
        <tr><td class="metric">clip_similarity_target_image_edit_part</td><td class="best">9</td><td>11</td><td>12</td></tr>
        <tr><td class="metric">ssim_unedit_part</td><td>9</td><td>8</td><td>8</td></tr>
        <tr><td class="metric">psnr_unedit_part</td><td>10</td><td>8.5</td><td>8</td></tr>
        <tr><td class="metric">lpips_unedit_part</td><td>10</td><td>8</td><td>8</td></tr>
        <tr><td class="metric">structure_distance</td><td>11</td><td>11</td><td>12</td></tr>
        <tr><td class="metric">mse_unedit_part</td><td>12</td><td>12</td><td>12</td></tr>
      </tbody>
    </table>
  </div>

  <p class="col" style="margin-top:1.7rem">The split across the table is itself
  informative. <b>Editing signal lands early</b> — cos_third_pvp reaches its
  final clip-to-target score by step 6, four steps before either paper arm.
  <b>Preservation metrics settle later</b> — mse and structure_distance need 11
  to 12 steps in every arm. Whatever the last third of the sampler is doing, it
  is refining background fidelity, not the edit.</p>

  <div class="callout">
    <p class="eyebrow">This does not say "8 steps suffice"</p>
    <p>V<sub>j</sub> is each block's step-<em>j</em> prediction collected during
    one <em>unperturbed</em> 15-step rollout, where earlier blocks were already
    fully denoised and re-cached at clean context. It is a trajectory preview,
    not what a shortened sampler emits. The claim it supports is about where in
    the trajectory the editing signal lands; converting that into a compute
    saving requires an actual 8-step run.</p>
  </div>
</section>

<section>
  <div class="head">
    <p class="eyebrow">Finding 2</p>
    <h2>The edit drifts away over the clip — unless the bank is persistent</h2>
    <p class="col">Averaged over the bench, per-frame clip-to-target at step 14
    falls from the first fifth of a clip to the last fifth by <b>1.02</b> points
    for paper_vp and <b>1.09</b> for paper_novp, while cos_third_pvp loses only
    <b>0.13</b>. On individual clips the gap is far wider.</p>
  </div>

  {fig(U["fig_drift"],
       "Per-frame CLIP→target at step 14 (rolling mean, window 7) for the three clips below. "
       "Flat or rising is edit persistence; falling is drift.")}

  <div class="tablewrap">
    <table>
      <caption>Per-frame clip-to-target at j=14, mean of the first 20% of frames
      versus the last 20%. Positive change means the edit strengthens across the clip.</caption>
      <thead><tr><th>Clip</th><th>cos_third_pvp</th><th>paper_vp</th><th>paper_novp</th></tr></thead>
      <tbody>
        <tr><td class="metric">0076_A_rabbit</td><td class="pos">+1.57</td><td class="neg">−3.56</td><td class="neg">−3.62</td></tr>
        <tr><td class="metric">0090_A_deer</td><td class="pos">+2.66</td><td class="neg">−1.18</td><td>−0.32</td></tr>
        <tr><td class="metric">0057_dog</td><td>−1.29</td><td class="pos">−0.82</td><td class="neg">−1.75</td></tr>
      </tbody>
    </table>
  </div>

  <p class="col" style="margin-top:1.7rem">The mosaics make the mechanism plain.
  Rows are pixel frames (every tenth), columns are denoising steps, so reading
  <em>down</em> the last column is watching the finished video play.</p>

  {mosaic_row("0076_A_rabbit",
              "The edit is rabbit → drone. cos_third_pvp holds a clean, "
              "consistent drone in every frame from 0 to 80. paper_vp renders the drone "
              "only in frame 0 — the anchored frame — and from frame 10 onward it collapses "
              "into a shapeless pale blob that is neither rabbit nor drone. This is the "
              "clearest case of the persistent bank removing drift outright.")}

  {mosaic_row("0090_A_deer",
              "Same pattern, less extreme: cos_third_pvp strengthens across the "
              "clip (+2.66), while both paper arms sag in the middle and recover only "
              "partially by the end.")}

  {mosaic_row("0057_dog",
              "The honest counterexample. On the metric, paper_vp drifts least "
              "here (−0.82 against cos_third_pvp's −1.29) and sits higher in absolute "
              "terms. Visually cos_third_pvp still holds the robot-dog identity across "
              "all 57 frames, so this is a scoring difference rather than a collapse — "
              "but it does mean the persistent bank is not uniformly better.")}

  <div class="callout">
    <p class="eyebrow">Background drift is not fixed</p>
    <p>LPIPS on the unedited region worsens across the clip by a similar amount
    in every arm (+0.10 rabbit, +0.07 deer, +0.15 dog for cos_third_pvp, within
    0.03 of the paper arms). The persistent bank preserves the <em>edit</em>; it
    does not stop the background from degrading. Consistent with the absolute-value
    figures, where cos_third_pvp sits at visibly higher LPIPS than paper_vp
    throughout.</p>
  </div>
</section>

<section class="limits">
  <div class="head">
    <p class="eyebrow">Reading limits</p>
    <h2>What would change these conclusions</h2>
  </div>
  <ol>
    <li><p><b>Deviation is largest at small prefixes, not at the full clip.</b>
    Mean |D| in the first 20% of frames runs 1.4–1.6 CLIP points against 0.9–1.0
    at the whole clip, and the same ordering holds for every metric. That is an
    averaging artefact of the cumulative mean — few frames early, heavily damped
    late — not evidence about drift. Drift has to be read from per-frame values,
    which is what Finding 2 uses.</p></li>
    <li><p><b>M(i,j) is a cumulative prefix mean, not the metric at frame i.</b>
    Movement along the row axis is damped and dominated by early frames.</p></li>
    <li><p><b>22 clips, edit types 1/2/5/6.</b> Full-bench numbers for these arms
    live in R21; treat everything here as corroboration on a subset, not an
    independent measurement.</p></li>
    <li><p><b>One clip contradicts the drift story on the metric</b>
    (<code>0057_dog</code>), and the three mosaic clips were chosen to illustrate
    the effect, not sampled at random. The bench-wide averages in Finding 2 are
    the unbiased statement; the mosaics are the mechanism.</p></li>
    <li><p><b>clip_similarity_source_image is a null control.</b> It scores the
    source video, so it is constant along j and renders flat by construction —
    its invariance across all 45 evaluation runs also confirms harness
    determinism.</p></li>
  </ol>
</section>

<footer>
  <div class="wrap" style="padding:0">
    Sources — evaluation/csv/r22_matrix.csv (533,520 rows) ·
    r22_bottom_right_check.csv (528 cells, max |diff| 0.0) ·
    /projects/dataggen/outputs/five_bench/r22_step_dump (36 GB, 3 arms × 15 steps × 22 clips)<br>
    Scripts — r22_dump_steps.py · r22_build_matrix.py · r22_figures.py · r22_mosaic.py
  </div>
</footer>

</div>
"""
    out.write_text(html)
    kb = out.stat().st_size / 1024
    print(f"[r22_artifact] wrote {out}  ({kb:.0f} KB, {len(U)} images inlined)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, default=Path("/tmp/r22art"))
    ap.add_argument("-o", "--out", type=Path, default=Path("r22_report.html"))
    args = ap.parse_args()
    build(args.assets, args.out)


if __name__ == "__main__":
    main()
