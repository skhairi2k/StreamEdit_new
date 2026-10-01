#!/usr/bin/env python
"""R29 step 1 -- per-case bidirectional correspondence residual. No video generation, no tau.

Measures SYNTHESIS DEMAND: how much of an edit's content has no counterpart in the source
at all. For dense DINO ViT-B/8 patch features of the source first frame and the Qwen
anchor, with S = F_edit @ F_src.T the cosine matrix between every anchor token and every
source token,

    N_fwd(p) = 1 - max_q S[p, q]     one value per ANCHOR token -- content INVENTED
    N_bwd(q) = 1 - max_p S[p, q]     one value per SOURCE token -- content ERASED

Each maximum is taken over the WHOLE other image, so a token scores high only when nothing
anywhere in the other image resembles it. Background inside the edit region therefore
scores ~0, because it matches background elsewhere.

WHY THIS EXISTS. R27 measures how far depth MOVED, which for an addition is bounded by the
added object's own extent -- a hat displaces almost nothing. Additions consequently
measured BELOW swaps (0.906 vs 1.346) while the required tau ordering puts them above, and
budget-linear interpolation is monotone so no remapping can invert an order. R27 is blocked
at `taumap` for that reason. Cosine novelty is blind to displacement by construction, which
is exactly the property the depth signal lacks.

WHY BOTH DIRECTIONS. N_fwd alone cannot see a removal: remove the gym ball and the anchor
shows background, which matches background elsewhere in the source, so N_fwd ~ 0 and only
N_bwd fires on the vacated region. With removals targeted at tau 30-50, a one-directional
measure would floor the single removal case.

    N_fwd high, N_bwd low   -> addition   (disocclusion, in optical-flow terms)
    N_fwd low,  N_bwd high  -> removal    (occlusion)
    both high               -> form swap
    both low                -> colour

N_fwd and N_bwd are computed with MAX SIMILARITY, not mutual nearest neighbour. MNN is the
stronger test in that literature, but it reports false novelty in repetitive regions -- many
grass tokens are near-identical, so the cycle p -> q* -> p* lands on a different but equally
good token. This benchmark's backgrounds are grass, sky and water, the same cases that broke
R27's sigma estimate. Max similarity is immune: a grass token has an excellent match.

GRID AND COMBINATION. Both images are resized to the ANCHOR grid 480x832 (R27's convention:
the render pipeline builds the anchor via transforms.Resize((480, 832)), an anisotropic
squeeze, so resampling to the source's 864x480 would re-introduce a 3.7% warp). At patch 8
that is 60 x 104 = 6240 tokens. NOTE N_fwd indexes the anchor grid and N_bwd the source
grid: same shape, and token p is the same spatial LOCATION in both, but not the same
content. max/sum/asym are therefore POSITIONAL combinations, reading as "at location x,
either something new appeared or something disappeared".

AGGREGATION. The PRIMARY is a fixed-count top slice over the WHOLE FRAME. Region-relative
aggregation is surface-DEPENDENT here and is emitted only to EXHIBIT that: the region
R = M_src | M_edit holds the unchanged source object as well as the changed content, so the
novel fraction of R spans 15x within the addition class alone (hat 4.3%, dog 46%, flamingo
63%) and a region mean would report the hat 15x lower for reasons having nothing to do with
synthesis demand. A whole-frame pool is a CONSTANT 6240 tokens, so the percentile is
identical for every case and any fully-novel object above n tokens fills the slice equally.

The sorted top-200 values of every quantity are written to a companion npz so that n can be
swept in r29_compare.py without recomputing features -- n enters only the aggregation.

Usage
-----
    TORCH_HOME=~/.cache/torch HF_HUB_OFFLINE=1 python evaluation/r29_novelty.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --anchor_root /projects/dataggen/outputs/five_bench/anchors \\
        --mask_dir evaluation/figures/r25_iou_masks \\
        --facet key --layer -1 \\
        -o evaluation/csv/r29_novelty.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r27_depth_delta import load_frame0_masks  # noqa: E402  (shared data reader)

PATCH: int = 8
GRID_HW: Tuple[int, int] = (480, 832)          # the anchor grid, R27's convention
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Quantities derived from the two maps. max/sum/asym are POSITIONAL combinations.
QUANTITIES: Tuple[str, ...] = ("n_fwd", "n_bwd", "nmax", "nsum", "nasym")
# Aggregations. `frame_topn` is the primary; the four region_* exist to exhibit dilution
# and r27_med_cov is emitted for comparability with R27 only.
AGGREGATIONS: Tuple[str, ...] = (
    "frame_topn", "region_mean", "region_median", "region_top20",
    "region_mean_er1", "r27_med_cov",
)
TOP_K_DUMP: int = 200      # sorted top values written to the npz, for the n sweep


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG that could touch the forward pass. Extraction is deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dino(device: str):
    """DINO ViT-B/8 from the local torch hub cache. Nothing downloads."""
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8",
                           source="github", trust_repo=True, verbose=False)
    return model.to(device).eval()


def to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Resize onto the anchor grid and ImageNet-normalise."""
    h, w = GRID_HW
    arr = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - torch.tensor(IMAGENET_MEAN)[:, None, None]) \
        / torch.tensor(IMAGENET_STD)[:, None, None]
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def extract_tokens(model, img_t: torch.Tensor, facet: str, layer: int) -> torch.Tensor:
    """Dense patch descriptors, CLS dropped. Returns (N, C) L2-normalised float32.

    The KEY facet is taken by hooking the qkv projection of the chosen block and slicing
    the middle third: qkv is (B, N, 3C), reshaped to (B, N, 3, heads, C/heads) so that
    index 1 is the keys, then heads are concatenated back to C. Keys are the standard
    choice for semantic correspondence -- the final output tokens are shaped by the CLS
    objective and match less cleanly across images.
    """
    if facet == "token":
        out = model.get_intermediate_layers(img_t, n=abs(layer))[layer]
        feats = out[:, 1:, :]
    else:
        block = model.blocks[layer]
        store: Dict[str, torch.Tensor] = {}

        def hook(_m, _i, out):
            store["qkv"] = out

        h = block.attn.qkv.register_forward_hook(hook)
        try:
            model.get_intermediate_layers(img_t, n=1)
        finally:
            h.remove()
        qkv = store["qkv"]                                   # (B, N, 3C)
        b, n, _ = qkv.shape
        heads = block.attn.num_heads
        qkv = qkv.reshape(b, n, 3, heads, -1).permute(2, 0, 3, 1, 4)
        idx = {"query": 0, "key": 1, "value": 2}[facet]
        f = qkv[idx]                                         # (B, heads, N, head_dim)
        feats = f.permute(0, 2, 1, 3).reshape(b, n, -1)      # (B, N, C)
        feats = feats[:, 1:, :]                              # drop CLS

    feats = feats[0].float()
    return F.normalize(feats, dim=-1)


def mask_to_tokens(mask: np.ndarray) -> np.ndarray:
    """Downsample a pixel mask onto the token grid by AREA MEAN, then threshold at 0.5.

    Area mean rather than nearest: a token covers 8x8 pixels, and what matters is whether
    the majority of that patch is inside the region, not what its top-left corner happens
    to be.
    """
    h, w = GRID_HW
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m, size=(h, w), mode="area")
    pooled = F.avg_pool2d(m, kernel_size=PATCH, stride=PATCH)[0, 0]
    return (pooled.numpy() > 0.5)


def erode_tokens(region: np.ndarray) -> np.ndarray:
    """Erode a token-grid region by one token (8-connectivity).

    Boundary-token control: a compact region of k tokens carries ~3.5*sqrt(k) boundary
    tokens whose patches straddle edited and unedited content, so the boundary FRACTION
    ~3.5/sqrt(k) is ~20% at 312 tokens against ~8% at 1872 -- a systematic size dependence
    the region-mean derivation does not cover. Only meaningful for the region_* columns;
    the primary does not aggregate over a region at all.
    """
    from scipy.ndimage import binary_erosion, generate_binary_structure
    return binary_erosion(region, generate_binary_structure(2, 2))


def aggregate(values: np.ndarray, region: np.ndarray, n_top: int) -> Dict[str, float]:
    """Every candidate aggregation of one novelty map, in one pass.

    `frame_topn` is the PRIMARY and is the only one that never touches the mask: the pool
    is all 6240 tokens for every case, so the percentile is identical across cases by
    construction and no region-size effect can enter.
    """
    flat = values.ravel()
    k = min(n_top, flat.size)
    out: Dict[str, float] = {
        "frame_topn": float(np.partition(flat, flat.size - k)[flat.size - k:].mean()),
    }
    reg = values[region]
    if reg.size:
        out["region_mean"] = float(reg.mean())
        out["region_median"] = float(np.median(reg))
        kk = max(1, int(round(0.20 * reg.size)))
        out["region_top20"] = float(np.sort(reg)[-kk:].mean())
        # R27's winner, for comparability only. `coverage` explicitly scales small regions
        # down, so it is anti-surface-independent and must not be adopted on that basis.
        out["r27_med_cov"] = out["region_median"] * min(1.0, reg.size / float(n_top))
    else:
        for key in ("region_mean", "region_median", "region_top20", "r27_med_cov"):
            out[key] = 0.0
    er = values[erode_tokens(region)] if region.any() else np.array([])
    out["region_mean_er1"] = float(er.mean()) if er.size else 0.0
    return out


def save_panel(out_path: Path, src_img: Image.Image, anchor_img: Image.Image,
               n_fwd: np.ndarray, n_bwd: np.ndarray, region: np.ndarray,
               row: Dict[str, object]) -> None:
    """2x3 audit panel: the images, both novelty maps, their asymmetry, and the region.

    The asymmetry panel is the one that reads the edit CLASS at a glance -- one-sided for
    an addition or a removal, two-sided for a form swap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = GRID_HW
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.0 * h / max(w, 1) * 3 + 1.5))
    axes[0][0].imshow(np.asarray(src_img.resize((w, h), Image.BILINEAR)))
    axes[0][0].set_title("source $I_0$", fontsize=11)
    axes[0][1].imshow(np.asarray(anchor_img.resize((w, h), Image.BILINEAR)))
    axes[0][1].set_title("anchor $I_{0,\\mathrm{edit}}$", fontsize=11)

    # Shared scale so the two directions are directly comparable by eye.
    vmax = max(float(n_fwd.max()), float(n_bwd.max()), 1e-6)
    im = axes[0][2].imshow(n_fwd, cmap="inferno", vmin=0, vmax=vmax)
    axes[0][2].set_title("$N_{\\mathrm{fwd}}$  (invented)", fontsize=11)
    fig.colorbar(im, ax=axes[0][2], fraction=0.03, pad=0.01).ax.tick_params(labelsize=7)
    im = axes[1][0].imshow(n_bwd, cmap="inferno", vmin=0, vmax=vmax)
    axes[1][0].set_title("$N_{\\mathrm{bwd}}$  (erased)", fontsize=11)
    fig.colorbar(im, ax=axes[1][0], fraction=0.03, pad=0.01).ax.tick_params(labelsize=7)

    d = n_fwd - n_bwd
    a = max(float(np.abs(d).max()), 1e-6)
    im = axes[1][1].imshow(d, cmap="coolwarm", vmin=-a, vmax=a)
    axes[1][1].set_title("$N_{\\mathrm{fwd}} - N_{\\mathrm{bwd}}$  "
                         "(red = invented, blue = erased)", fontsize=10)
    fig.colorbar(im, ax=axes[1][1], fraction=0.03, pad=0.01).ax.tick_params(labelsize=7)

    axes[1][2].imshow(region, cmap="gray", vmin=0, vmax=1)
    axes[1][2].set_title(f"region $R = M_{{src}} \\cup M_{{edit}}$  "
                         f"({int(region.sum())} tokens)", fontsize=10)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"{row['case_id']}  type {row['edit_type']}   "
        f"frame_topn: fwd {row['n_fwd__frame_topn']:.3f}  bwd {row['n_bwd__frame_topn']:.3f}"
        f"  max {row['nmax__frame_topn']:.3f}  asym {row['nasym__frame_topn']:.3f}",
        fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# per-case measurement
# --------------------------------------------------------------------------------------
def measure_case(case: Dict[str, object], data_root: Path, anchor_root: Path,
                 mask_dir: Path, model, device: str, facet: str, layer: int,
                 n_top: int, viz_dir: Optional[Path] = None
                 ) -> Tuple[Dict[str, object], Dict[str, np.ndarray], Optional[str]]:
    """Measure one case. Returns (csv row, npz payload, hard-failure reason or None)."""
    case_id = str(case["case_id"])
    video_name = str(case["video_name"])
    edit_type = int(case["edit_type"])

    src_path = data_root / "images" / video_name / "00001.jpg"
    anchor_path = anchor_root / f"edit{edit_type}" / f"{video_name}.png"
    mask_path = mask_dir / f"{case_id}.npz"
    for p in (src_path, anchor_path, mask_path):
        if not p.exists():
            raise FileNotFoundError(f"{case_id}: missing {p}")

    src_img = Image.open(src_path).convert("RGB")
    anchor_img = Image.open(anchor_path).convert("RGB")
    m_src, m_edit, _ = load_frame0_masks(mask_path)
    region = mask_to_tokens(m_src | m_edit)

    f_src = extract_tokens(model, to_tensor(src_img, device), facet, layer)
    f_edit = extract_tokens(model, to_tensor(anchor_img, device), facet, layer)
    gh, gw = GRID_HW[0] // PATCH, GRID_HW[1] // PATCH
    if f_src.shape[0] != gh * gw:
        raise ValueError(f"{case_id}: got {f_src.shape[0]} tokens, expected {gh * gw}")

    # Both are L2-normalised, so the matmul IS the cosine matrix. 6240x6240 float32 is
    # ~155 MB -- computed whole rather than tiled, which keeps the maxima exact.
    with torch.no_grad():
        sim = f_edit @ f_src.T
        n_fwd = (1.0 - sim.max(dim=1).values).cpu().numpy().reshape(gh, gw)
        n_bwd = (1.0 - sim.max(dim=0).values).cpu().numpy().reshape(gh, gw)
        del sim

    maps = {
        "n_fwd": n_fwd,
        "n_bwd": n_bwd,
        # POSITIONAL combinations: token p is the same LOCATION in both grids, not the
        # same content. Reads as "at x, either something appeared or something vanished".
        "nmax": np.maximum(n_fwd, n_bwd),
        "nsum": n_fwd + n_bwd,
        "nasym": np.abs(n_fwd - n_bwd),
    }

    row: Dict[str, object] = {
        "case_id": case_id, "video_name": video_name, "edit_type": edit_type,
        "n_tokens_region": int(region.sum()),
        "region_frac": round(float(region.mean()), 6),
        "n_top": n_top,
    }
    payload: Dict[str, np.ndarray] = {}
    for qname, qmap in maps.items():
        for aname, val in aggregate(qmap, region, n_top).items():
            row[f"{qname}__{aname}"] = round(val, 6)
        # Sorted descending top-K, so r29_compare.py can sweep n without recomputing.
        flat = qmap.ravel()
        k = min(TOP_K_DUMP, flat.size)
        payload[qname] = np.sort(np.partition(flat, flat.size - k)[flat.size - k:])[::-1]

    if viz_dir is not None:
        save_panel(viz_dir / f"{case_id}.png", src_img, anchor_img,
                   n_fwd, n_bwd, region, row)
    return row, payload, None


def build_fieldnames() -> List[str]:
    base = ["case_id", "video_name", "edit_type", "n_tokens_region", "region_frac", "n_top"]
    return base + [f"{q}__{a}" for q in QUANTITIES for a in AGGREGATIONS]


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path, required=True,
                   help="FiVE-Bench root holding images/")
    p.add_argument("--anchor_root", type=Path, required=True,
                   help="Anchor root; the anchor is {anchor_root}/edit{T}/{video}.png")
    p.add_argument("--mask_dir", type=Path,
                   default=Path("evaluation/figures/r25_iou_masks"),
                   help="R25's dumped frame-0 masks. Their union is the region R, used "
                        "ONLY for the region_* diagnostic columns and frac-in-region "
                        "reporting -- the primary aggregation never touches it.")
    p.add_argument("--facet", choices=("key", "query", "value", "token"), default="key",
                   help="Which ViT facet to describe tokens with. `key` is the standard "
                        "choice for semantic correspondence; the final output tokens "
                        "(`token`) are shaped by the CLS objective and match less cleanly "
                        "across images.")
    p.add_argument("--layer", type=int, default=-1, help="Block index; -1 is the last.")
    p.add_argument("--top_n", type=int, default=None,
                   help="Size of the whole-frame top slice. Default ceil(0.005 * 6240) "
                        "= 31, inherited from R27 and NOT independently justified -- the "
                        "constraint here is that n must not exceed the smallest novel "
                        "object or the slice fills with non-novel tokens: N_bwd is bound "
                        "by 0091_A_hawk (the hawk, 3122 px = 49 tokens) and N_fwd by "
                        "0007_guitar-violin (the added hat, ~4448 px = 70 tokens), so "
                        "n = 31 leaves only a 1.6x margin. r29_compare.py sweeps n over "
                        "the top-200 dump rather than trusting this default.")
    p.add_argument("--viz_dir", type=Path, default=None,
                   help="If given, write one 2x3 audit panel per case. Off by default; "
                        "the CSV is unaffected either way.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r29_novelty.csv"))
    p.add_argument("--dump_top", type=Path, default=None,
                   help="Sorted top-200 values per quantity per case, for the n sweep in "
                        "r29_compare.py. Defaults to the CSV path with a .npz suffix.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    gh, gw = GRID_HW[0] // PATCH, GRID_HW[1] // PATCH
    n_tokens = gh * gw
    n_top = args.top_n if args.top_n else int(math.ceil(0.005 * n_tokens))

    data_root = args.data_root.expanduser().resolve()
    anchor_root = args.anchor_root.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    viz_dir = args.viz_dir.expanduser() if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    dump_top = args.dump_top.expanduser() if args.dump_top \
        else args.out.with_suffix(".npz")
    cases = json.loads(args.cases.expanduser().read_text())

    print("=== R29 step 1: bidirectional correspondence residual ===", flush=True)
    print(f"cases       {args.cases} ({len(cases)} cases)", flush=True)
    print(f"data_root   {data_root}", flush=True)
    print(f"anchor_root {anchor_root}", flush=True)
    print(f"mask_dir    {mask_dir}", flush=True)
    print(f"features    DINO ViT-B/8  facet={args.facet}  layer={args.layer}", flush=True)
    print(f"grid        {GRID_HW[0]}x{GRID_HW[1]} / patch {PATCH} "
          f"= {gh}x{gw} = {n_tokens} tokens", flush=True)
    print(f"top_n       {n_top}  (whole-frame pool, constant across cases; "
          f"admissibility bound is n <= 49)", flush=True)
    print(f"viz_dir     {viz_dir if viz_dir else '(off)'}", flush=True)
    print(f"device      {args.device}   seed {args.seed}", flush=True)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    model = load_dino(args.device)

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    dump: Dict[str, np.ndarray] = {}
    for case in cases:
        row, payload, failure = measure_case(
            case, data_root, anchor_root, mask_dir, model=model, device=args.device,
            facet=args.facet, layer=args.layer, n_top=n_top, viz_dir=viz_dir)
        rows.append(row)
        for q, arr in payload.items():
            dump[f"{row['case_id']}__{q}"] = arr
        if failure is not None:
            failures.append(failure)
            print(f"[FAIL] {failure}", flush=True)
        else:
            print(f"[ok] {row['case_id']:22s} type {row['edit_type']} "
                  f"fwd {row['n_fwd__frame_topn']:.3f}  bwd {row['n_bwd__frame_topn']:.3f}"
                  f"  max {row['nmax__frame_topn']:.3f}  asym {row['nasym__frame_topn']:.3f}"
                  f"   region {row['n_tokens_region']:5d} tok", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames()
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nwrote {args.out}  ({len(rows)} rows, {len(fieldnames)} cols)", flush=True)

    np.savez_compressed(dump_top, **dump)
    print(f"wrote {dump_top}  (top-{TOP_K_DUMP} per quantity per case, for the n sweep)",
          flush=True)
    if viz_dir is not None:
        print(f"wrote audit panels to {viz_dir}", flush=True)

    for q in ("n_fwd", "n_bwd", "nmax"):
        vals = [float(r[f"{q}__frame_topn"]) for r in rows]
        print(f"{q}__frame_topn range over {len(vals)} cases: "
              f"[{min(vals):.4f}, {max(vals):.4f}]", flush=True)

    print(f"failures: {len(failures)}", flush=True)
    if failures:
        print("\nHard failures -- do NOT proceed to compare:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
