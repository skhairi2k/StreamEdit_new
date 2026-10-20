"""R49 side experiment: the Wan-Edit draft (17-step grid) started at step 3, step 2 and step 1, with its max-divergence map.

PDF with 18 rows (the clips of r49_cases.json) x 7 columns; every cell is a first | last frame concatenation:
  source | draft, start step 3 | div map step 3 | draft, start step 2 | div map step 2 | draft, start step 1 | div map step 1
Start steps on the 17-step grid (shift 5): step 3 = sigma 0.958 (the R49 draft; = skip 9 on the 50-step grid), step 2 = 0.974
(closest to skip 5 = 0.978), step 1 = 0.988. A div map shows the FIRST and LAST latent frame of
    div = max( clip(D_dino, 0, 0.55) / 0.55 , clip(D_lpips, 0, 1.0) / 1.0 )        (colour range 0..1)
computed from the draft-vs-source distances (r31_divergence.py). A missing render shows a grey "not rendered" cell.
Reads frames and divergence npz only; runs no model.
usage: python evaluation/r49_draft_compare.py [--out FILE.pdf]     (a PNG twin is written next to it)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
D = Path.home() / "Data/dataggen/outputs/five_bench"
FIVE = Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
KAPPA = {"dino": 0.55, "lpips": 1.0}
GRID = (30, 52)
# (start step, sigma, draft frames root, divergence root holding wan_dino/ and wan_lpips/)
DRAFTS = [(3, 0.958, D / "r49_wan_draft/frames/step00", D / "r49_div"),
          (2, 0.974, D / "r49_wan_draft/frames/skip2", D / "r49_draft_skip_div/skip2"),
          (1, 0.988, D / "r49_wan_draft/frames/skip1", D / "r49_draft_skip_div/skip1")]


def frames(p: Path):
    return sorted(p.glob("*.png")) if p.is_dir() else []


def rgb(p) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def source_frame(vid: str, idx: int, size) -> np.ndarray:
    cap = cv2.VideoCapture(str(FIVE / "videos" / f"{vid}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), size) if ok else np.zeros((size[1], size[0], 3), np.uint8)


def div_map(root: Path, t: int, vid: str):
    """max-divergence [F, 30, 52] in [0, 1], or None if either distance npz is missing."""
    ps = {m: root / f"wan_{m}" / f"edit{t}" / f"{vid}.npz" for m in ("dino", "lpips")}
    if not all(p.is_file() for p in ps.values()):
        return None
    m = {k: np.clip(np.load(p)["d"].reshape(-1, *GRID), 0, KAPPA[k]) / KAPPA[k] for k, p in ps.items()}
    return np.maximum(m["dino"], m["lpips"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "r49_cases.json"))
    ap.add_argument("--out", default=str(HERE / "figures" / "r49_draft_skip" / "draft_step3_2_1.pdf"))
    a = ap.parse_args()
    cases = sorted({(int(c["edit_type"]), c["video_name"]) for c in json.loads(Path(a.cases).read_text())})
    titles = ["source"] + [x for s, sg, _, _ in DRAFTS for x in (f"draft, start step {s} (σ {sg})", f"div map, step {s}")]
    fig, axes = plt.subplots(len(cases), len(titles), figsize=(3.4 * len(titles), 1.25 * len(cases)), squeeze=False)
    n_missing = 0
    for r, (t, vid) in enumerate(cases):
        ref = frames(DRAFTS[0][2] / f"edit{t}" / vid)
        n = len(ref) if ref else 41
        idx = (0, n - 1)                                              # first | last pixel frame
        h, w = (rgb(ref[0]).shape[:2]) if ref else (480, 832)
        cells = [("img", np.hstack([source_frame(vid, i, (w, h)) for i in idx]))]
        for _s, _sg, droot, vroot in DRAFTS:
            fs = frames(droot / f"edit{t}" / vid)
            cells.append(("img", np.hstack([cv2.resize(rgb(fs[min(i, len(fs) - 1)]), (w, h)) for i in idx])) if fs else ("none", None))
            dm = div_map(vroot, t, vid)
            cells.append(("div", np.hstack([dm[0], dm[-1]])) if dm is not None else ("none", None))   # first | last latent frame
        for c, (kind, im) in enumerate(cells):
            ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
            if kind == "none":
                n_missing += 1
                ax.set_facecolor("#dddddd"); ax.text(0.5, 0.5, "not rendered", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            elif kind == "img":
                ax.imshow(im)
            else:
                ax.imshow(im, cmap="inferno", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            if c == 0:
                ax.set_ylabel(f"e{t}\n{vid}", fontsize=6, rotation=0, ha="right", va="center")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    fig.suptitle("Wan-Edit draft (17-step grid) at three start steps with its max-divergence map — each cell: first | last frame", fontsize=10)
    fig.tight_layout(pad=0.3)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=70, bbox_inches="tight")
    print(f"[r49_draft_compare] {out}  (cells not rendered: {n_missing} of {len(cases) * len(titles)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
