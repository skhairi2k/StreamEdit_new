#!/usr/bin/env python
"""R28 step 1 -- per-arm TARGET masks for union-based background preservation.

FiVE-Bench ships a source mask per frame (``bmasks/{video}/``) but no target mask, so
``evaluate.py`` scores background preservation on ``1 - M_src`` alone (evaluate.py:462
passes ``mask, mask``). An edit that moves, grows or reshapes the object puts its new
pixels inside that region, where they are counted as failed preservation. This script
produces the missing ``M_tgt`` by grounding ``trg_word`` on the METHOD'S OWN rendered
frames, so ``evaluate.py`` can score on ``1 - (M_src | M_tgt)`` instead.

Grounding is GroundingDINO (box) refined by SAM2 (extent), imported from
``evaluation/r25_iou.py`` rather than re-implemented -- that module is already validated
at median IoU 0.974 against the FiVE GT source masks.

⚠️ GROUNDINGDINO NEVER ABSTAINS. It returns its best box unconditionally, so a phrase
naming something absent still "fires" -- R25 measured 0.94 for 'A dragon' on an unedited
cow. ``fired``/``score`` are recorded as AUDIT fields, never as evidence of presence, and
the two edit types that would depend on such evidence are handled by CONSTRUCTION:

    types 1-4 (swap / colour / material)  M_tgt = m(trg_word, tgt_frame)
    type  5   (addition, region GREW)     M_tgt = m(trg_word, tgt_frame) | M_src
    type  6   (removal, region is gone)   M_tgt = empty  => union == M_src

⚠️ FRAME ALIGNMENT. evaluate.py strides BOTH sides by ``--frame_stride`` and zips them
positionally (evaluate.py:442), so mask row ``i`` must correspond to strided target frame
``i``. The stride is written into the npz and a mismatch at read time is a hard failure --
a silent stride disagreement would misalign every mask after the first.

Usage
-----
    HF_HUB_OFFLINE=1 python evaluation/r28_target_masks.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --tgt_root /projects/dataggen/outputs/five_bench/r26_spatial_tau/taubg2_taufg2_vp \\
        --out_dir /projects/dataggen/outputs/five_bench/r28_tgt_masks/taubg2_taufg2_vp \\
        --frame_stride 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r25_iou import load_models, phrase_mask, resize_nearest, set_seed  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(directory: str) -> List[str]:
    """Same ordering evaluate.py uses (``list_images`` -> sorted glob)."""
    if not os.path.isdir(directory):
        return []
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(IMAGE_EXTS))
    return [os.path.join(directory, n) for n in names]


def load_src_masks(mask_dir: str, src_paths: Sequence[str], hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """GT source masks for the given source frames, as ``[n, H, W]`` bool.

    Read exactly the way evaluate.py:307-316 reads them: the bmask filename is the source
    frame's basename, and any non-zero pixel is foreground.
    """
    out = []
    for p in src_paths:
        mp = os.path.join(mask_dir, os.path.basename(p))
        if not os.path.exists(mp):
            return None
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        m = m > 0
        if m.shape != hw:
            m = resize_nearest(m, hw)
        out.append(m)
    return np.stack(out, axis=0)


def build_target_masks(
    edit_type: int,
    trg_word: str,
    tgt_paths: Sequence[str],
    src_masks: np.ndarray,
    hw: Tuple[int, int],
    *,
    gdino_proc,
    gdino,
    predictor,
    device: str,
    box_thr: float,
) -> Tuple[np.ndarray, List[int], List[float]]:
    """``M_tgt`` per frame plus the per-frame audit fields, under R25's per-type rules."""
    n = len(tgt_paths)
    masks = np.zeros((n,) + hw, dtype=bool)
    fired: List[int] = []
    scores: List[float] = []

    if edit_type == 6:
        # Removal: the object is gone, so there is nothing to ground on the target and the
        # union collapses to M_src -- i.e. exactly today's behaviour for this type. This is
        # an ASSUMPTION carried over from R25, not a measurement.
        return masks, [0] * n, [0.0] * n

    for i, p in enumerate(tgt_paths):
        img = Image.open(p).convert("RGB")
        if img.size != (hw[1], hw[0]):
            # evaluate.py resizes the render to the SOURCE size before scoring; ground on
            # the same pixels it will score, or the mask lands on a different grid.
            img = img.resize((hw[1], hw[0]))
        m, score, _ = phrase_mask(img, trg_word, gdino_proc, gdino, predictor, device, box_thr)
        fired.append(int(m is not None))
        scores.append(round(float(score), 4))
        if m is not None:
            masks[i] = resize_nearest(m, hw)

    if edit_type == 5:
        # Addition: the edited region is the original object PLUS what was added.
        masks |= src_masks[: len(masks)]
    return masks, fired, scores


def save_npz(path: str, masks: np.ndarray, fired, scores, *, stride: int, arm: str,
             video: str, edit_type: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        M=np.packbits(masks, axis=-1),
        shape=np.array(masks.shape, dtype=np.int64),
        stride=np.array([stride], dtype=np.int64),
        fired=np.array(fired, dtype=np.int64),
        score=np.array(scores, dtype=np.float32),
        arm=np.array([arm]),
        video=np.array([video]),
        edit_type=np.array([edit_type], dtype=np.int64),
    )


def load_npz(path: str, expect_stride: Optional[int] = None) -> np.ndarray:
    """Read a dumped target mask back as ``[n, H, W]`` bool, checking the stride."""
    with np.load(path, allow_pickle=False) as d:
        shape = tuple(int(v) for v in d["shape"])
        stride = int(d["stride"][0])
        if expect_stride is not None and stride != expect_stride:
            raise ValueError(
                f"{path}: dumped at frame_stride {stride} but read with {expect_stride}. "
                "evaluate.py strides both sides positionally, so a mismatch misaligns "
                "every mask after the first."
            )
        return np.unpackbits(d["M"], axis=-1)[:, : shape[1] * shape[2]].reshape(shape).astype(bool)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--data_root", required=True, help="FiVE root holding images/ and bmasks/")
    ap.add_argument("--tgt_root", required=True, help="one arm root, {edit_type}/{video}/ layout")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frame_stride", type=int, default=8)
    ap.add_argument("--box_thr", type=float, default=0.35)
    ap.add_argument("--gdino_id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam2_id", default="facebook/sam2-hiera-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke only: first N cases")
    ap.add_argument("--only_video", default=None, help="smoke only: restrict to one video_name")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    data_root = os.path.expanduser(args.data_root)
    with open(os.path.expanduser(args.cases), encoding="utf-8") as fh:
        cases = json.load(fh)
    if args.only_video:
        cases = [c for c in cases if c["video_name"] == args.only_video]
    if args.limit:
        cases = cases[: args.limit]
    print(f"[r28_masks] {len(cases)} case(s), stride {args.frame_stride}, arm root {args.tgt_root}")

    gdino_proc, gdino, predictor = load_models(args.gdino_id, args.sam2_id, args.device)
    print(f"[r28_masks] GroundingDINO + SAM2 loaded on {args.device}")

    failures = 0
    for c in cases:
        t, video = int(c["edit_type"]), c["video_name"]
        trg_word = c["trg_word"]
        src_paths = list_images(os.path.join(data_root, "images", video))[:: args.frame_stride]
        tgt_paths = list_images(os.path.join(args.tgt_root, f"edit{t}", video))[:: args.frame_stride]
        if not src_paths or not tgt_paths:
            print(f"[r28_masks] MISSING edit{t}/{video}: {len(src_paths)} src, {len(tgt_paths)} tgt")
            failures += 1
            continue
        # evaluate.py caps the frame loop at the shorter side; match it exactly.
        n = min(len(src_paths), len(tgt_paths))
        src_paths, tgt_paths = src_paths[:n], tgt_paths[:n]

        w, h = Image.open(src_paths[0]).size
        hw = (h, w)
        src_masks = load_src_masks(os.path.join(data_root, "bmasks", video), src_paths, hw)
        if src_masks is None:
            print(f"[r28_masks] MISSING bmasks for {video}")
            failures += 1
            continue

        masks, fired, scores = build_target_masks(
            t, trg_word, tgt_paths, src_masks, hw,
            gdino_proc=gdino_proc, gdino=gdino, predictor=predictor,
            device=args.device, box_thr=args.box_thr,
        )
        out = os.path.join(args.out_dir, f"edit{t}", f"{video}.npz")
        save_npz(out, masks, fired, scores, stride=args.frame_stride,
                 arm=os.path.basename(args.tgt_root.rstrip("/")), video=video, edit_type=t)

        frac = float(masks.mean())
        union_frac = float((masks | src_masks[:n]).mean())
        flag = ""
        if frac == 0.0 and t != 6:
            flag = "  ⚠EMPTY"
        elif frac == 1.0:
            flag = "  ⚠ALL-ONES"
        print(f"[r28_masks] edit{t}/{video:22s} n={n:3d} fired={sum(fired):3d}/{n} "
              f"tgt_fg={frac:.3f} union_fg={union_frac:.3f}{flag}")

    print(f"[r28_masks] failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
