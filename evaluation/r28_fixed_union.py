#!/usr/bin/env python
"""R28 step 2 -- reduce every arm's target mask into ONE fixed cross-arm union.

Pure reduction: no grounding, no models, no GPU. Reads the per-arm dumps written by
``evaluation/r28_target_masks.py`` and writes, per (edit_type, video):

    U = M_src | (M_tgt over EVERY arm)

Scoring on ``1 - U`` gives every arm an IDENTICAL background region, so cross-arm deltas
are a like-for-like pixel comparison. The per-arm union (``1 - (M_src | M_tgt_arm)``) is
the faithful per-method reading but is computed over a different region for each arm, so
the two answer different questions and R28 reports both.

⚠️ THE ARM SET IS PART OF THE RESULT. Adding an arm can only SHRINK the resulting
background, so fixed-union numbers are comparable only within an identical arm set. The
contributing arms are stored inside every npz and `load_fixed_union` refuses a set it was
not built from, rather than letting two differently-composed unions be compared silently.

⚠️ FAILURE MODE, the mirror of the per-arm one. A single OVER-firing arm shrinks the
background for EVERY arm, so one bad grounding contaminates all twelve numbers at once
instead of inflating one. The per-arm marginal contribution is printed here and carried
into `r28_summarize.py` so a dominant arm is attributable rather than diffuse.

Usage
-----
    python evaluation/r28_fixed_union.py \\
        --mask_root /projects/dataggen/outputs/five_bench/r28_tgt_masks \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --cases evaluation/cases.json \\
        -o /projects/dataggen/outputs/five_bench/r28_tgt_masks/_fixed_union
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r28_target_masks import list_images, load_src_masks, load_npz  # noqa: E402

OUT_DIRNAME = "_fixed_union"


def discover_arms(mask_root: str) -> List[str]:
    """Every arm directory under `mask_root`, excluding the output dir itself."""
    return sorted(
        d for d in os.listdir(mask_root)
        if os.path.isdir(os.path.join(mask_root, d)) and d != OUT_DIRNAME
    )


def load_fixed_union(path: str, expect_arms: Optional[Sequence[str]] = None,
                     expect_stride: Optional[int] = None) -> np.ndarray:
    """Read a fixed union back, refusing a different arm set or stride."""
    with np.load(path, allow_pickle=False) as d:
        shape = tuple(int(v) for v in d["shape"])
        arms = [str(a) for a in d["arms"]]
        stride = int(d["stride"][0])
        if expect_arms is not None and list(expect_arms) != arms:
            raise ValueError(
                f"{path}: built from arms {arms} but read expecting {list(expect_arms)}. "
                "Adding an arm can only shrink the background, so unions from different "
                "arm sets are not comparable."
            )
        if expect_stride is not None and stride != expect_stride:
            raise ValueError(f"{path}: stride {stride} != {expect_stride}")
        return np.unpackbits(d["M"], axis=-1)[:, : shape[1] * shape[2]].reshape(shape).astype(bool)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask_root", required=True, help="root holding one dir per arm")
    ap.add_argument("--data_root", required=True, help="FiVE root, for bmasks/ and images/")
    ap.add_argument("--cases", default="evaluation/cases.json")
    ap.add_argument("--frame_stride", type=int, default=8)
    ap.add_argument("-o", "--out", default=None, help="default: {mask_root}/_fixed_union")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="explicit arm list; default is every dir under --mask_root")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(args.mask_root, OUT_DIRNAME)
    data_root = os.path.expanduser(args.data_root)
    with open(os.path.expanduser(args.cases), encoding="utf-8") as fh:
        cases = json.load(fh)

    arms = args.arms if args.arms else discover_arms(args.mask_root)
    if not arms:
        print(f"[r28_union] no arm directories under {args.mask_root}", file=sys.stderr)
        return 1
    print(f"[r28_union] {len(arms)} arms: {', '.join(arms)}")
    print(f"[r28_union] {len(cases)} cases, stride {args.frame_stride} -> {out_dir}")

    # How often each arm is the ONLY one contributing a pixel: a high share means the
    # union is being driven by one arm, which is what an over-firing grounding looks like.
    marginal: Dict[str, float] = {a: 0.0 for a in arms}
    failures = 0
    n_done = 0

    for c in cases:
        t, video = int(c["edit_type"]), c["video_name"]
        src_paths = list_images(os.path.join(data_root, "images", video))[:: args.frame_stride]
        if not src_paths:
            print(f"[r28_union] MISSING source frames for {video}")
            failures += 1
            continue
        w, h = Image.open(src_paths[0]).size
        hw = (h, w)

        per_arm = {}
        for a in arms:
            p = os.path.join(args.mask_root, a, f"edit{t}", f"{video}.npz")
            if not os.path.exists(p):
                print(f"[r28_union] MISSING {a}/edit{t}/{video}.npz -- the fixed union "
                      "must be built from ALL arms or it is not fixed")
                failures += 1
                per_arm = {}
                break
            per_arm[a] = load_npz(p, expect_stride=args.frame_stride)
        if not per_arm:
            continue

        n = min(m.shape[0] for m in per_arm.values())
        src_masks = load_src_masks(os.path.join(data_root, "bmasks", video), src_paths[:n], hw)
        if src_masks is None:
            print(f"[r28_union] MISSING bmasks for {video}")
            failures += 1
            continue

        union = src_masks[:n].copy()
        for a in arms:
            union |= per_arm[a][:n]

        # Marginal contribution: pixels this arm adds that NO other arm and not M_src has.
        for a in arms:
            others = src_masks[:n].copy()
            for b in arms:
                if b != a:
                    others |= per_arm[b][:n]
            only_a = per_arm[a][:n] & ~others
            marginal[a] += float(only_a.mean())

        out = os.path.join(out_dir, f"edit{t}", f"{video}.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(
            out,
            M=np.packbits(union, axis=-1),
            shape=np.array(union.shape, dtype=np.int64),
            stride=np.array([args.frame_stride], dtype=np.int64),
            arms=np.array(arms),
            video=np.array([video]),
            edit_type=np.array([t], dtype=np.int64),
        )
        n_done += 1

        # Invariant: the fixed background is contained in every per-arm background, i.e.
        # the fixed union contains every per-arm union. Holds by construction; if it fails
        # the reduction ran on the wrong axis, which no count-based check would catch.
        for a in arms:
            arm_union = src_masks[:n] | per_arm[a][:n]
            if not np.all(arm_union <= union):
                raise AssertionError(
                    f"edit{t}/{video}: fixed union does not contain arm {a}'s union -- "
                    "the reduction is wrong (check the frame axis)."
                )
        print(f"[r28_union] edit{t}/{video:22s} n={n:3d} union_fg={float(union.mean()):.3f} "
              f"(src {float(src_masks[:n].mean()):.3f})")

    print(f"\n[r28_union] wrote {n_done} clips, failures: {failures}")
    if n_done:
        print("[r28_union] mean marginal contribution per arm "
              "(fraction of pixels ONLY that arm adds; a large value = that arm is "
              "driving the union, the signature of an over-firing grounding):")
        for a, v in sorted(marginal.items(), key=lambda kv: -kv[1]):
            print(f"    {a:28s} {v / max(n_done, 1):.5f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
