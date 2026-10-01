"""Anchor-referenced appearance metrics for R10.

R10 injects one image -- the Qwen-edited first frame ("the anchor") -- into a
subset of attention heads. The two questions that decide the experiment are
therefore about that image:

  ADHERENCE   did the anchor's appearance land at all?      -> anchor_sim_f0
  VANISHING   does it survive the clip, or fade?            -> anchor_sim_slope

Neither exists in the FiVE benchmark. ``calculate_clip_similarity`` is
image-to-TEXT only, and ``metrics_calculator`` has no image-to-image similarity
anywhere, so this module adds one.

WHY NOT COMPARE TO THE SOURCE INSTEAD
-------------------------------------
Every source-referenced metric (``lpips_edit_part``, ``structure_distance_edit
_part``) scores the output against the source *inside the region we deliberately
change*. A successful edit must score badly on them, so they have no readable
direction. Anchor similarity has one: high means the injected appearance is
present.

WHY DINO CLS AND NOT LPIPS
--------------------------
The anchor is a still frame; the subject moves. LPIPS is pixel-aligned and would
punish a turned head as hard as a lost texture, so ordinary motion would read as
"vanishing". The DINO CLS token is a global appearance/identity descriptor and is
far more pose-tolerant. Even so, some pose-induced decay is unavoidable -- which
is why the slope is only ever read BETWEEN ARMS on the same clip (they share the
source motion, so that component largely differences out), never as an absolute.

WHY CROP RATHER THAN ZERO-FILL
------------------------------
``calculate_structure_distance`` multiplies the image by the mask, leaving a
black surround. For a ViT reading a single global CLS token that surround
dominates the representation. This module crops to the mask bounding box
instead -- the same construction ``MotionFidelityScore`` uses for its own masked
tracking (metrics_calculator.py:336-338).

TWO DELIBERATE DEVIATIONS FROM ``calculate_structure_distance``
--------------------------------------------------------------
1. It feeds 0-255 floats into an ImageNet normaliser (which expects [0,1]).
   That is preserved there because R1/R7 were scored with it and changing it
   would break comparability. This module is a NEW metric with no history to
   protect, so it scales to [0,1] correctly.
2. It zero-fills; this crops, per above.
Both mean anchor_sim numbers are not comparable to structure_distance numbers.
They are not meant to be.

INTERPRETATION GUARD
--------------------
The slope is meaningless where ``anchor_sim_f0`` is low: an arm that never
adopted the anchor has nothing to fade from and will show a flat, healthy-looking
slope. Read the pair, never the slope alone. R10 predicts exactly this for the
``temporal`` and ``none`` arms.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))
sys.path.insert(0, str(_REPO_ROOT / "evaluation" / "fivebench"))

_IMAGE_EXT = (".png", ".jpg", ".jpeg")
# Same DINO backbone structure_distance already loads, so no second model.
_DINO_MODEL = "dino_vitb8"
_DINO_PATCH = 224
_CLS_LAYER = -1          # last block, matching calculate_crop_cls_loss
_BBOX_PAD = 8            # px of context around the edit region


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=str, required=True,
                   help="root holding {arm}/edit{T}/{video}/ frame dirs")
    p.add_argument("--arms", nargs="+",
                   default=["none", "all", "spatial", "temporal",
                            "rand_spatial", "rand_temporal"])
    p.add_argument("--r7_root", type=str, default=None,
                   help="R7 SS4.5 root, scored as arm 'r7_vp'. This is the FADING "
                        "reference the slope exists to compare against -- omitting "
                        "it makes anchor_sim_slope uninterpretable.")
    p.add_argument("--anchor_root", type=str,
                   default="/projects/dataggen/outputs/five_bench/anchors",
                   help="{root}/edit{T}/{video}.png -- the same files r10_vp_arms.py injects")
    p.add_argument("--src_image_folder", type=str,
                   default=str(Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark").expanduser()),
                   help="holds bmasks/{video}/ ; masks are 1-indexed, frames 0-indexed")
    p.add_argument("--cases_json", type=str,
                   default=str(_REPO_ROOT / "evaluation" / "cases.json"))
    p.add_argument("--cases", nargs="*", default=None,
                   help="optional case_id subset (smoke tests)")
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--out_csv", type=str, required=True,
                   help="summary CSV path; per-frame rows go to <stem>_per_frame.csv")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def list_frames(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMAGE_EXT)


def slope_per100(idx: np.ndarray, vals: np.ndarray) -> float:
    """OLS slope per 100 frames; nan with fewer than 2 finite points.

    Same definition r10_metrics.slope uses for the CLIP fading curves, so the
    two slopes are read on one scale.
    """
    ok = np.isfinite(vals)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(idx[ok], vals[ok], 1)[0] * 100.0)


class AnchorEmbedder:
    """DINO CLS embedding of a mask-cropped region."""

    def __init__(self, device: str):
        from metrics_calculator import VitExtractor          # noqa: E402
        from torchvision import transforms
        from torchvision.transforms import Resize

        self.device = device
        self.extractor = VitExtractor(model_name=_DINO_MODEL, device=device)
        # Identical to LossG.global_transform (metrics_calculator.py:540-543).
        self.transform = transforms.Compose([
            Resize(_DINO_PATCH, max_size=480),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def bbox(mask: np.ndarray, pad: int = _BBOX_PAD) -> tuple[int, int, int, int] | None:
        """(top, bottom, left, right) of the mask, padded. None if empty."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        h, w = mask.shape
        return (max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h),
                max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w))

    @torch.no_grad()
    def embed(self, img: Image.Image, box: tuple[int, int, int, int]) -> torch.Tensor:
        t, b, l, r = box
        crop = np.asarray(img.convert("RGB"), dtype=np.float32)[t:b, l:r] / 255.0
        x = torch.from_numpy(crop).permute(2, 0, 1).to(self.device)
        x = self.transform(x).unsqueeze(0)
        cls = self.extractor.get_feature_from_input(x)[_CLS_LAYER][0, 0, :]
        return F.normalize(cls.float(), dim=0)


def load_masks(src_image_folder: Path, video: str) -> list[np.ndarray]:
    """Binary masks, index-aligned to OUTPUT frames.

    bmasks are named 00001.. (1-indexed) while rendered frames are 00000..
    (0-indexed); evaluate.py pairs them positionally after sorting, so position i
    of this list is the mask for output frame i.
    """
    d = src_image_folder / "bmasks" / video
    if not d.is_dir():
        return []
    out = []
    for p in sorted(q for q in d.iterdir() if q.suffix.lower() in _IMAGE_EXT):
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        out.append(m > 0)
    return out


def main() -> None:
    args = parse_args()
    cases = json.load(open(args.cases_json))
    if args.cases:
        want = set(args.cases)
        cases = [c for c in cases
                 if c["case_id"] in want or c["video_name"] in want]
    if not cases:
        raise SystemExit("[anchor-sim] no cases selected")

    in_root = Path(args.in_root)
    anchor_root = Path(args.anchor_root).expanduser()
    src_folder = Path(args.src_image_folder).expanduser()

    # `dir` or `dir=key`: the directory is fixed by r10_vp_arms.py to equal the
    # GATE name, so the persistent bank is called `all` with blending on AND off.
    # The key is what lands in the CSV, so those two can be told apart
    # (`all=all_blendoff`, `all=all_blendon`). A dir containing a separator is a
    # path in its own right, letting arms under unrelated roots be scored together.
    arms: list[tuple[str, Path]] = []
    for spec in args.arms:
        d, _, key = spec.partition("=")
        arms.append((key or d, Path(d).expanduser() if "/" in d else in_root / d))
    if args.r7_root:
        arms.append(("r7_vp", Path(args.r7_root)))
    else:
        print("[anchor-sim] WARNING: no --r7_root. The slope has no fading "
              "reference to be read against.")

    emb = AnchorEmbedder(args.device)
    per_frame: list[dict] = []
    summary: list[dict] = []

    for case in cases:
        video, etype = case["video_name"], case["edit_type"]
        rel = Path(f"edit{etype}") / video
        anchor_path = anchor_root / f"edit{etype}" / f"{video}.png"
        if not anchor_path.exists():
            print(f"[anchor-sim] {video} e{etype}: no anchor at {anchor_path} -- skipped")
            continue
        masks = load_masks(src_folder, video)
        if not masks:
            print(f"[anchor-sim] {video}: no bmasks -- skipped")
            continue

        a_box = emb.bbox(masks[0])
        if a_box is None:
            print(f"[anchor-sim] {video}: empty first mask -- skipped")
            continue
        a_emb = emb.embed(Image.open(anchor_path), a_box)

        for arm, root in arms:
            frames = list_frames(root / rel)
            if not frames:
                print(f"[anchor-sim] {video} e{etype}: arm '{arm}' missing -- skipped")
                continue
            idx, sims = [], []
            for i in range(0, min(len(frames), len(masks)), args.frame_stride):
                box = emb.bbox(masks[i])
                if box is None:
                    continue
                s = float(torch.dot(a_emb, emb.embed(Image.open(frames[i]), box)))
                idx.append(i)
                sims.append(s)
                per_frame.append({"video_name": video, "editing_type_id": etype,
                                  "arm": arm, "frame_idx": i, "anchor_sim": f"{s:.6f}"})
            if len(sims) < 2:
                print(f"[anchor-sim] {video} e{etype} {arm}: <2 usable frames -- skipped")
                continue
            a = np.asarray(idx, dtype=float)
            v = np.asarray(sims, dtype=float)
            summary.append({
                "video_name": video, "editing_type_id": etype, "arm": arm,
                "n_frames": len(v),
                "anchor_sim_f0": f"{v[0]:.6f}",
                "anchor_sim_mean": f"{v.mean():.6f}",
                "anchor_sim_last": f"{v[-1]:.6f}",
                "anchor_sim_slope_per100f": f"{slope_per100(a, v):.6f}",
                # retention is undefined when nothing adhered; f0 guards it
                "anchor_sim_retention": f"{v[-1] / v[0]:.6f}" if abs(v[0]) > 1e-6 else "nan",
            })
            print(f"[anchor-sim] {video:22s} e{etype} {arm:14s} "
                  f"f0={v[0]:.3f} mean={v.mean():.3f} slope={slope_per100(a, v):+.2f}")

    if not summary:
        raise SystemExit("[anchor-sim] nothing scored")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    pf = out.with_name(out.stem + "_per_frame.csv")
    with pf.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_frame[0].keys()))
        w.writeheader()
        w.writerows(per_frame)
    print(f"[anchor-sim] {len(summary)} arm x clip rows -> {out}")
    print(f"[anchor-sim] {len(per_frame)} per-frame rows -> {pf}")


if __name__ == "__main__":
    main()
