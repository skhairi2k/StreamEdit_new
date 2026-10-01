#!/usr/bin/env python
"""Directional CLIP (CLIP-D) -- an edit-achievement axis that does NOT collapse the way
``clip_similarity_target_image`` does on this benchmark.

WHY THIS EXISTS
---------------
FiVE-Bench's ``clip_similarity_target_image`` is absolute CLIPScore:
``100 * cos(E_img(render), E_txt(trg_prompt))``. On this case set it does not rank renders
by edit strength -- measured, not assumed (all per-clip, over R26's 8-b SPATIAL sweep):

  Spearman(b, clip_similarity_target_image) = +0.098 mean, positive on only 11/22 clips
  Spearman(b, niqe_target_image)            = -0.530 mean  (quality IMPROVES with b)
  Spearman(niqe, clip_target)               = -0.093 mean  (CLIP does not track quality)
  cos(E_txt(src_prompt), E_txt(trg_prompt)) = 0.846 mean, up to 0.970

The cause is the last line. src_prompt and trg_prompt differ by ONE noun out of ~35
tokens; the rest describes the scene ("grass, trees, the camera remains stationary").
That shared majority is satisfied at EVERY b -- a regenerated tree is still a tree, so
CLIP, which is source-agnostic, keeps scoring it as a tree -- and it dominates the
embedding. What is left moving the number is render-level image variation that correlates
with neither b nor quality. Note this is a RENDER-level confound, not a frame-level one:
every frame of one render shares its texture/composition statistics, so the harness's
averaging over the strided frames does not reduce it at all.

WHAT CLIP-D COMPUTES INSTEAD
----------------------------
    d_txt = E_txt(trg_prompt) - E_txt(src_prompt)      (once per clip)
    d_img = E_img(render_frame) - E_img(source_frame)  (per paired frame)
    CLIP-D = mean_frames cos(d_img, d_txt)

Both subtractions cancel what the two sides SHARE: the scene description on the text side,
the untouched background on the image side. What survives is the edit axis itself. Scale is
interpretable in a way absolute CLIPScore is not: +1 = moved exactly as the caption asked,
0 = moved somewhere unrelated, NEGATIVE = moved the wrong way. Introduced by StyleGAN-NADA
and used as the achievement axis in the image-editing literature (e.g. InstructPix2Pix's
trade-off plots) for exactly this reason.

CONVENTIONS (chosen deliberately; see the flags to override)
------------------------------------------------------------
* Embeddings are L2-normalised BEFORE the subtraction, so each difference is a direction on
  the unit sphere rather than a magnitude. ``--no_prenorm`` differences the raw projections.
* The sign is KEPT. Absolute CLIPScore clamps at zero (``max(cos, 0)``); here a negative
  value is the single most informative outcome an over-released arm can produce, so
  clamping it away would hide the finding.
* ``clip_d_word`` is emitted alongside ``clip_d_prompt``: the same quantity with
  ``src_word``/``trg_word`` ("a golden retriever" -> "A robot dog") in place of the full
  paragraphs. Free to compute and a useful cross-check -- the word pair is far more
  separable (mean cos 0.71 vs 0.85 for the paragraphs).
* Tokenisation uses ``truncation=True``, which KEEPS the EOT token. This matters: CLIP pools
  its text embedding at the EOT position, and torchmetrics 1.9.0 (``clip_score.py:145-146``,
  what the FiVE harness calls) instead raw-slices ``input_ids[:77]`` and drops EOT on any
  caption over the limit. On this case set that hits exactly one clip, 0040_tennis (85/84
  tokens), whose text embedding therefore collapses to the BOS position and stops depending
  on the prompt at all -- its src_prompt and trg_prompt return BIT-IDENTICAL scores (5.528),
  and the stored clip_similarity_* columns for that clip are measuring nothing. Fixed here.
* Frame pairing mirrors evaluate.py exactly: both sides strided by ``--frame_stride``, then
  zipped positionally after truncating the source to the render's length, and the render
  resized to the source's size.

Needs the rendered frame trees under /projects, so this runs on a compute node
(slurm_scripts/five_bench/r33_clip_directional.sh), unlike the other analysis
scripts.

Usage
-----
    python evaluation/r33_clip_directional.py --which all      # 28 arms: R26 16, R30 8, R31 4
    python evaluation/r33_clip_directional.py --which r31 --suffix _lin04
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_score import CONST_BS  # noqa: E402

ARMS: Sequence[str] = ("lpips", "dino_patch", "normals", "latent")
# R30's eight static divergence arms (r30_stage2.sh:107). Three of them -- lpips,
# dino_patch, normals -- collide by NAME with R31's, so every method label written to the
# CSV carries an r26_/r30_/r31_ prefix; without it the two tasks' rows are indistinguishable.
R30_ARMS: Sequence[str] = ("lpips", "dino_cls", "dino_patch", "clip_image",
                           "clip_prompt", "depth", "normals", "selfsim")
MODEL = "openai/clip-vit-large-patch14"


# ------------------------------------------------------------------------ clip ---------

class Clip:
    """CLIP embeddings via the projection heads directly.

    ``get_text_features``/``get_image_features`` return a bare Tensor on some transformers
    versions and a ModelOutput on others; going through ``text_model``/``vision_model`` plus
    the projection is the same computation and is stable across both.
    """

    def __init__(self, device: str = "cuda", model_name: str = MODEL):
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.tok = CLIPTokenizer.from_pretrained(model_name)
        self.max_len = self.model.config.text_config.max_position_embeddings

    @torch.no_grad()
    def text(self, txt: str) -> torch.Tensor:
        e = self.tok(txt, return_tensors="pt", padding="max_length",
                     truncation=True, max_length=self.max_len)
        out = self.model.text_model(input_ids=e["input_ids"].to(self.device),
                                    attention_mask=e["attention_mask"].to(self.device))
        return self.model.text_projection(out[1])[0]

    @torch.no_grad()
    def image(self, img: Image.Image) -> torch.Tensor:
        px = self.proc(images=[np.array(img)], return_tensors="pt")["pixel_values"]
        out = self.model.vision_model(pixel_values=px.to(self.device))
        return self.model.visual_projection(out[1])[0]


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def clip_d(src_feats: Sequence[torch.Tensor], tgt_feats: Sequence[torch.Tensor],
           d_txt: torch.Tensor) -> Tuple[float, float]:
    """(mean cos(E(render) - E(source), d_txt), mean ||E(render) - E(source)||).

    The norm is returned because the cosine is SCALE-FREE: a render that barely differs
    from the source still yields a unit direction and therefore still produces a
    confident-looking number, even though that direction was fixed by whatever tiny
    differences happened to exist rather than by the edit. The low-b end of the sweep is
    exactly that regime, so a CLIP-D value is only readable next to its ||d_img||.
    """
    cos_vals, norms = [], []
    for s, t in zip(src_feats, tgt_feats):
        d_img = t - s
        n = float(d_img.norm())
        norms.append(n)                   # kept even when degenerate -- that IS the signal
        if n < 1e-8:                      # render identical to source: no direction at all
            continue
        cos_vals.append(float((d_img / n * d_txt).sum()))
    return (float(np.mean(cos_vals)) if cos_vals else float("nan"),
            float(np.mean(norms)) if norms else float("nan"))


# ------------------------------------------------------------------------ io -----------

def list_frames(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
                  key=lambda p: p.name)


def method_specs(which: str, r31_root: Path, r26_root: Path, r30_root: Path,
                 suffix: str) -> List[Tuple[str, Path]]:
    """[(method_name, frames_root)] -- frames live at {root}/edit{T}/{video}/.

    The three tasks do NOT share a tree shape, so each root is expanded by its own rule
    rather than by a common formula: R31 nests the final render under a step dir
    (r31_{arm}/step14/), while R26 and R30 write the arm dir directly (r30_stage2.sh sets
    METHOD="$ARM", a bare arm name).
    """
    out: List[Tuple[str, Path]] = []
    if which in {"r31", "r26r31", "all"}:
        for arm in ARMS:
            name = f"r31_{arm}{suffix}"
            out.append((name, r31_root / name / "step14"))
    if which in {"r26", "r26r31", "all"}:
        for b in CONST_BS:
            out.append((f"r26_taubg0_taufg{b}_vp", r26_root / f"taubg0_taufg{b}_vp"))
        for b in CONST_BS:
            out.append((f"r26_taubg{b}_taufg{b}_vp", r26_root / f"taubg{b}_taufg{b}_vp"))
    if which in {"r30", "all"}:
        for arm in R30_ARMS:
            out.append((f"r30_{arm}", r30_root / arm))
    return out


def fullbench_cases(data_root: Path) -> List[Dict[str, object]]:
    """All 419 pairs straight from edit_prompt/edit{T}_FiVE.json, in cases.json's shape.

    Added for R35 (2026-09-24): cases.json covers 22 clips, and the full-bench case list
    (r7_anchor_manifest.json) carries no prompts. Field mapping, checked against the 22
    cases.json entries: source_prompt/target_prompt are IDENTICAL to src_prompt/
    trg_prompt on all 22, so clip_d_prompt is unaffected. source_object/target_object
    stand in for src_word/trg_word, which cases.json hand-edited on 5 of the 22 (e.g.
    0017_kid-football: 'a red cap' vs FiVE's 'a young boy') -- so clip_d_word on the full
    bench uses FiVE's raw objects, not those edits.
    """
    out: List[Dict[str, object]] = []
    for t in range(1, 7):
        for e in json.loads((data_root / "edit_prompt" / f"edit{t}_FiVE.json").read_text()):
            out.append({"case_id": f"edit{t}/{e['video_name']}", "video_name": e["video_name"],
                        "edit_type": t, "src_prompt": e["source_prompt"],
                        "trg_prompt": e["target_prompt"], "src_word": e["source_object"],
                        "trg_word": e["target_object"]})
    return out


# ------------------------------------------------------------------------ main ---------

def run(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser()
    cases = (fullbench_cases(data_root) if args.fullbench
             else json.loads(args.cases.expanduser().read_text()))
    clip = Clip(args.device)

    if args.methods:
        #✨ R35: explicit NAME=DIR pairs replace the --which rosters entirely.
        specs = [(n, Path(d).expanduser()) for n, d in
                 (m.split("=", 1) for m in args.methods)]
    else:
        specs = method_specs(args.which, args.r31_root.expanduser(),
                             args.r26_root.expanduser(), args.r30_root.expanduser(),
                             args.suffix)
    print(f"[clip_d] {len(specs)} method(s) x {len(cases)} clip(s), "
          f"stride {args.frame_stride}, prenorm={not args.no_prenorm}")

    # Source frames and their embeddings are shared by every method -- embed once per clip.
    src_cache: Dict[str, List[torch.Tensor]] = {}
    txt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    rows: List[Dict[str, object]] = []
    for c in cases:
        cid, name, T = c["case_id"], c["video_name"], int(c["edit_type"])
        src_dir = data_root / "images" / name
        src_paths = list_frames(src_dir)[::args.frame_stride]
        if args.max_frames is not None:
            src_paths = src_paths[:args.max_frames]
        if not src_paths:
            print(f"[clip_d] SKIP {cid}: no source frames under {src_dir}")
            continue
        src_imgs = [Image.open(p).convert("RGB") for p in src_paths]
        src_cache[cid] = [clip.image(i) for i in src_imgs]

        def d_of(a: str, b: str) -> torch.Tensor:
            ea, eb = clip.text(a), clip.text(b)
            if not args.no_prenorm:
                ea, eb = _unit(ea), _unit(eb)
            return _unit(eb - ea)

        txt_cache[cid] = (d_of(c["src_prompt"], c["trg_prompt"]),
                          d_of(c["src_word"], c["trg_word"]))

        for method, root in specs:
            tgt_dir = root / f"edit{T}" / name
            tgt_paths = list_frames(tgt_dir)[::args.frame_stride]
            if args.max_frames is not None:
                tgt_paths = tgt_paths[:args.max_frames]
            if not tgt_paths:
                print(f"[clip_d] MISSING {method}/{cid}: {tgt_dir}")
                continue
            n = min(len(src_paths), len(tgt_paths))
            size = src_imgs[0].size
            tgt_feats = [clip.image(Image.open(p).convert("RGB").resize(size))
                         for p in tgt_paths[:n]]
            sf = src_cache[cid][:n]
            if not args.no_prenorm:
                sf = [_unit(v) for v in sf]
                tgt_feats = [_unit(v) for v in tgt_feats]
            d_prompt, d_word = txt_cache[cid]
            cd_prompt, dnorm = clip_d(sf, tgt_feats, d_prompt)
            cd_word, _ = clip_d(sf, tgt_feats, d_word)
            rows.append({
                "method": method,
                "case_id": cid,
                "edit_type": T,
                "video_name": name,
                "n_frames": n,
                "clip_d_prompt": round(cd_prompt, 6),
                "clip_d_word": round(cd_word, 6),
                "delta_img_norm": round(dnorm, 6),
            })
            print(f"[clip_d] {method:<28} {cid:<20} n={n:<3} "
                  f"prompt={cd_prompt:+.4f} word={cd_word:+.4f} "
                  f"|d_img|={dnorm:.4f}", flush=True)

    if not rows:
        raise SystemExit("[clip_d] nothing scored -- check --r31_root/--r26_root.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[clip_d] wrote {args.out} ({len(rows)} rows)")

    print(f"\n{'method':<30} {'n':>4} {'clip_d_prompt':>14} {'clip_d_word':>12} "
          f"{'|d_img|':>9}")
    for method, _ in specs:
        vals = [r for r in rows if r["method"] == method]
        if not vals:
            continue
        mp = float(np.mean([r["clip_d_prompt"] for r in vals]))
        mw = float(np.mean([r["clip_d_word"] for r in vals]))
        dn = float(np.mean([r["delta_img_norm"] for r in vals]))
        flag = "  <-- near-degenerate" if dn < 0.05 else ""
        print(f"{method:<30} {len(vals):>4} {mp:>14.4f} {mw:>12.4f} {dn:>9.4f}{flag}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=("r31", "r26", "r30", "r26r31", "all"),
                   default="all",
                   help="r26r31 selects R26's 16 constant-b arms + R31's 4 without also "
                        "walking R30's 8 -- R34's roster.")
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--r31_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_arms"),
                   help="Point at r31_arms_lin04 with --suffix _lin04 for the "
                        "linear_threshold run.")
    p.add_argument("--r26_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_spatial_tau"),
                   help="R26's stored constant-b renders, for the reference curves.")
    p.add_argument("--r30_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r30_arms"),
                   help="R30's 8 routed arms; each clip is rendered at its OWN routed b, "
                        "so these are 8 points, not a curve.")
    p.add_argument("--suffix", type=str, default="",
                   help="Appended to 'r31_{arm}' to select the render tree/method name.")
    p.add_argument("--frame_stride", type=int, default=8,
                   help="Must match the stride the FiVE harness scored at, or CLIP-D and "
                        "the stored CSV metrics describe different frames.")
    #✨ R34: same window flag evaluate.py grew, with the same semantics (applied AFTER
    # striding, to both sides), so `--frame_stride 1 --max_frames 9` here scores the
    # identical 9 frames the chunk-1 eval array scores.
    p.add_argument("--max_frames", type=int, default=None,
                   help="Keep only the first N frames after striding (source and "
                        "target). Default: no truncation. R34's first chunk is "
                        "--frame_stride 1 --max_frames 9.")
    #✨ R35 2026-09-24: both default off -- every existing invocation is unchanged.
    p.add_argument("--methods", nargs="+", default=None, metavar="NAME=DIR",
                   help="Explicit methods to score, each NAME=DIR with frames at "
                        "DIR/edit{T}/{video}/. Overrides --which.")
    p.add_argument("--fullbench", action="store_true",
                   help="Build the case list from all six edit{T}_FiVE.json files (419 "
                        "pairs) instead of reading --cases.")
    p.add_argument("--no_prenorm", action="store_true",
                   help="Difference the raw projections instead of L2-normalising first.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.out is None:
        args.out = Path(f"evaluation/csv/r33_clip_directional{args.suffix}.csv")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
