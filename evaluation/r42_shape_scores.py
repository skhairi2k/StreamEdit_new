#!/usr/bin/env python
"""R42 -- structural divergence between source frame 0 and the VP anchor, inside the object.

For every FiVE-Bench pair, build the object region ``M = M_src | M_trg`` on frame 0 and
measure two DINO divergences between the source frame and the anchor INSIDE ``M``. Each
score ranks the 419 pairs; its top-100 is one FiVE-Bench-Shape candidate.

MASKS (480x832, the pipeline's own resize)
------------------------------------------
    types 1-4  M_src = FiVE bmask frame 0 (> 0, evaluate.py's convention)
               M_trg = Grounded-SAM(target_object, anchor)
    type  5    M_src = empty      M_trg = Grounded-SAM(curated phrase, anchor)
    type  6    M_src = Grounded-SAM(curated phrase, SOURCE frame 0)      M_trg = empty
The type-6 M_src is the REMOVED object, not FiVE's subject bmask. Curated phrases live in
``r42_grounding_prompts.json``. Grounding is R25's GroundingDINO best box -> SAM2. ⚠
GroundingDINO never abstains (R25): ``gd_score`` is an audit field, not evidence of presence.

SCORES (both area-weighted over M, both with an unmasked ``*_global`` twin)
-------------------------------------------------------------------------
  patch    DINOv3 ViT-B/16, per-position 1 - cos on the 30x52 grid -- R35's DINO map,
           via ``r31_divergence.dinov3_patch_divergence`` unchanged.
  selfsim  DINO v1 ViT-B/8 keys of block 11 (FiVE's ``structure_distance`` backbone and
           facet), heads concatenated, at 480x832 = 60x104 patch tokens, CLS dropped.
           S = cosine self-similarity over all patch tokens, computed on the UNMASKED
           image; per-row error e_i = mean_j (S_src[i,j] - S_anc[i,j])^2; score = mean of
           e_i over rows in M. Two deliberate deviations from FiVE: input scaled to [0,1]
           before ImageNet normalisation (FiVE feeds 0-255), and no zero-filling of the
           image by the mask (black surround dominates a self-similarity matrix).

Weights: the float mask is area-pooled onto each token grid, so a token half inside M
counts half. A pair whose M is empty gets NaN scores and an AUDIT flag.

Usage
-----
    HF_HUB_OFFLINE=1 python evaluation/r42_shape_scores.py \\
        --manifest evaluation/r7_anchor_manifest.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --prompts evaluation/r42_grounding_prompts.json \\
        --out_dir ~/Data/dataggen/outputs/five_bench/r42_scores \\
        -o evaluation/csv/r42_shape_scores.csv --seed 0
    # --smoke: one case per edit type + identity gate, CSV written next to the npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))

from r25_iou import load_models, phrase_mask, resize_nearest, set_seed  # noqa: E402
from r30_divergence import IMAGENET_MEAN, IMAGENET_STD, resize_to  # noqa: E402
from r31_divergence import (  # noqa: E402
    DINOV3_MODEL_ID, GRID_H, GRID_W, PIXEL_H, PIXEL_W, dinov3_patch_divergence, load_dinov3,
)

HW: Tuple[int, int] = (PIXEL_H, PIXEL_W)
# DINO v1 ViT-B/8 at 480x832: 480/8 x 832/8 patch tokens.
SS_H: int = PIXEL_H // 8          # 60
SS_W: int = PIXEL_W // 8          # 104
SS_LAYER: int = 11                # FiVE LossG: get_keys_self_sim_from_input(layer_num=11)
ADD_TYPE: int = 5
REMOVE_TYPE: int = 6
AUDIT_MAX_FRAC: float = 0.5       # M covering more than half the frame is flagged
EXPECT_PER_TYPE: Dict[int, int] = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}

FIELDS: List[str] = [
    "case_id", "video_name", "edit_type", "phrase", "ground_image",
    "gd_fired", "gd_score", "gd_box", "area_M", "area_src", "area_trg",
    "patch", "patch_global", "selfsim", "selfsim_global", "audit",
]


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
def load_dino_v1(device: str):
    """DINO ViT-B/8 from the local torch hub cache (same call as r29_novelty.load_dino)."""
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8",
                           source="github", trust_repo=True, verbose=False)
    return model.to(device).eval()


def imagenet_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """(1, 3, 480, 832), [0,1] then ImageNet-normalised."""
    arr = np.asarray(resize_to(img, HW), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
    return t.unsqueeze(0).to(device)


@torch.no_grad()
def dino_v1_keys(model, img: Image.Image, device: str) -> torch.Tensor:
    """Block-11 KEY facet, heads concatenated, CLS dropped: ``[SS_H*SS_W, 768]`` float32.

    Same qkv layout as FiVE's ``VitExtractor.get_keys_from_qkv`` and r29_novelty.extract:
    qkv is (B, N, 3C) -> (B, N, 3, heads, C/heads), index 1 = keys.
    """
    block = model.blocks[SS_LAYER]
    store: Dict[str, torch.Tensor] = {}

    def hook(_m, _i, out):
        store["qkv"] = out

    h = block.attn.qkv.register_forward_hook(hook)
    try:
        model(imagenet_tensor(img, device))
    finally:
        h.remove()
    qkv = store["qkv"]
    b, n, _ = qkv.shape
    if n != 1 + SS_H * SS_W:
        raise ValueError(f"R42: DINO v1 returned {n} tokens, expected 1 + {SS_H}x{SS_W}")
    heads = block.attn.num_heads
    k = qkv.reshape(b, n, 3, heads, -1)[:, :, 1]           # (B, N, heads, head_dim)
    return k.reshape(b, n, -1)[0, 1:].float()               # drop CLS


@torch.no_grad()
def selfsim_row_error(model, src: Image.Image, anc: Image.Image, device: str) -> np.ndarray:
    """``e_i = mean_j (S_src[i,j] - S_anc[i,j])^2`` on the 60x104 grid, float64."""
    ka = F.normalize(dino_v1_keys(model, src, device), dim=-1)
    kb = F.normalize(dino_v1_keys(model, anc, device), dim=-1)
    diff = ka @ ka.T - kb @ kb.T                            # [6240, 6240]
    e = (diff * diff).mean(dim=1)
    return e.reshape(SS_H, SS_W).double().cpu().numpy()


# --------------------------------------------------------------------------------------
# masks and reductions
# --------------------------------------------------------------------------------------
def load_bmask(data_root: Path, video: str) -> np.ndarray:
    """FiVE source mask, frame 0, on the 480x832 grid. Non-zero = foreground (evaluate.py)."""
    p = data_root / "bmasks" / video / "00001.jpg"
    if not p.exists():
        raise FileNotFoundError(f"R42: no bmask frame 0 at {p}")
    m = np.array(Image.open(p))
    if m.ndim == 3:
        m = m[..., 0]
    return resize_nearest(m > 0, HW)


def area_weights(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Fraction of each token covered by ``mask`` (area pooling), float64 ``[gh, gw]``."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(t, size=(gh, gw), mode="area")[0, 0].double().numpy()


def weighted_mean(values: np.ndarray, w: np.ndarray) -> float:
    s = float(w.sum())
    return float((values * w).sum() / s) if s > 0 else math.nan


# --------------------------------------------------------------------------------------
# per case
# --------------------------------------------------------------------------------------
def target_objects(data_root: Path) -> Dict[Tuple[int, str], str]:
    """(edit_type, video_name) -> FiVE ``target_object``, from the annotation files."""
    out: Dict[Tuple[int, str], str] = {}
    for t in EXPECT_PER_TYPE:
        for x in json.load(open(data_root / "edit_prompt" / f"edit{t}_FiVE.json")):
            out[(t, x["video_name"])] = x["target_object"]
    return out


def measure(case: Dict, data_root: Path, prompts: Dict[str, Dict], tgt_obj: Dict,
            models: Dict, args: argparse.Namespace, anchor_override: Optional[Image.Image] = None
            ) -> Tuple[Dict, Dict[str, np.ndarray]]:
    T, video = int(case["edit_type"]), case["video_name"]
    src = resize_to(Image.open(data_root / case["src_first_frame"]).convert("RGB"), HW)
    anc = anchor_override if anchor_override is not None else \
        resize_to(Image.open(args.anchor_root.expanduser() / f"edit{T}" / f"{video}.png").convert("RGB"), HW)
    empty = np.zeros(HW, dtype=bool)

    if T in (ADD_TYPE, REMOVE_TYPE):
        if case["case_id"] not in prompts:
            raise KeyError(f"R42: {case['case_id']} has no entry in --prompts")
        phrase = prompts[case["case_id"]]["phrase"]
        ground_image = "anchor" if T == ADD_TYPE else "source"
    else:
        phrase, ground_image = tgt_obj[(T, video)], "anchor"

    m, score, box = phrase_mask(anc if ground_image == "anchor" else src, phrase,
                                models["gdino_proc"], models["gdino"], models["sam2"],
                                args.device, args.box_thr)
    grounded = resize_nearest(m, HW) if m is not None else empty
    if T == ADD_TYPE:
        M_src, M_trg = empty, grounded
    elif T == REMOVE_TYPE:
        M_src, M_trg = grounded, empty
    else:
        M_src, M_trg = load_bmask(data_root, video), grounded
    M = M_src | M_trg

    d_patch = dinov3_patch_divergence(models["dinov3"], models["n_prefix"], src, anc, args.device)
    e_ss = selfsim_row_error(models["dino_v1"], src, anc, args.device)

    area_M = float(M.mean())
    audit = []
    if m is None:
        audit.append("no_box")
    if area_M == 0.0:
        audit.append("empty_M")
    elif area_M > AUDIT_MAX_FRAC:
        audit.append("M_gt_half")

    row = {
        "case_id": case["case_id"], "video_name": video, "edit_type": T,
        "phrase": phrase, "ground_image": ground_image,
        "gd_fired": int(m is not None), "gd_score": round(score, 4),
        "gd_box": json.dumps(box) if box is not None else "",
        "area_M": round(area_M, 6), "area_src": round(float(M_src.mean()), 6),
        "area_trg": round(float(M_trg.mean()), 6),
        "patch": weighted_mean(d_patch, area_weights(M, GRID_H, GRID_W)),
        "patch_global": float(d_patch.mean()),
        "selfsim": weighted_mean(e_ss, area_weights(M, SS_H, SS_W)),
        "selfsim_global": float(e_ss.mean()),
        "audit": ";".join(audit),
    }
    arrays = {"M_src": M_src, "M_trg": M_trg, "d_patch": d_patch.astype(np.float32),
              "selfsim_row_err": e_ss.astype(np.float32)}
    return row, arrays


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", type=Path, default=_REPO_ROOT / "evaluation/r7_anchor_manifest.json")
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--anchor_root", type=Path, default=Path("~/Data/dataggen/outputs/five_bench/anchors"),
                   help="anchors/edit{T}/{video}.png (R40's convention; the manifest's anchor_image "
                        "points at evaluation/anchors/, which only holds the 22 R26-era anchors)")
    p.add_argument("--prompts", type=Path, default=_REPO_ROOT / "evaluation/r42_grounding_prompts.json")
    p.add_argument("--out_dir", type=Path, required=True, help="per-pair npz root")
    p.add_argument("-o", "--out", type=Path, default=_REPO_ROOT / "evaluation/csv/r42_shape_scores.csv")
    p.add_argument("--box_thr", type=float, default=0.35, help="R25's GroundingDINO threshold")
    p.add_argument("--gdino_id", type=str, default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam2_id", type=str, default="facebook/sam2-hiera-large")
    p.add_argument("--dinov3_id", type=str, default=DINOV3_MODEL_ID)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="first case of each edit type + identity gate; CSV goes to out_dir")
    return p.parse_args(argv)


def fmt(v) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    print(f"[r42] python={platform.python_version()} torch={torch.__version__} device={args.device}")
    print(f"[r42] args={vars(args)}")

    manifest = json.load(open(args.manifest))
    prompts = {x["case_id"]: x for x in json.load(open(args.prompts))}
    tgt_obj = target_objects(args.data_root)
    if args.smoke:
        seen, cases = set(), []
        for c in manifest:
            if c["edit_type"] not in seen:
                seen.add(c["edit_type"])
                cases.append(c)
        out_csv = args.out_dir / "r42_shape_scores_smoke.csv"
    else:
        cases, out_csv = manifest, args.out

    gdino_proc, gdino, sam2 = load_models(args.gdino_id, args.sam2_id, args.device)
    dinov3, n_prefix = load_dinov3(args.dinov3_id, args.device)
    models = {"gdino_proc": gdino_proc, "gdino": gdino, "sam2": sam2,
              "dinov3": dinov3, "n_prefix": n_prefix, "dino_v1": load_dino_v1(args.device)}
    print(f"[r42] models: gdino={args.gdino_id} sam2={args.sam2_id} dinov3={args.dinov3_id} "
          f"(n_prefix={n_prefix}) dino_v1=dino_vitb8 block {SS_LAYER} keys")

    rows: List[Dict] = []
    n_err = 0
    for i, case in enumerate(cases):
        try:
            row, arrays = measure(case, args.data_root, prompts, tgt_obj, models, args)
        except Exception as e:  # logged, counted, and failed on by the sbatch guard
            n_err += 1
            print(f"[ERROR] {case['case_id']}: {type(e).__name__}: {e}", flush=True)
            continue
        npz = args.out_dir / f"edit{case['edit_type']}" / f"{case['video_name']}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays, case_id=case["case_id"], phrase=row["phrase"],
                            ground_image=row["ground_image"], seed=args.seed,
                            box_thr=args.box_thr, dinov3_id=args.dinov3_id)
        rows.append(row)
        print(f"[{i + 1}/{len(cases)}] {case['case_id']:<32} patch={row['patch']:.4f} "
              f"selfsim={row['selfsim']:.5f} area_M={row['area_M']:.3f} {row['audit']}", flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) for k in FIELDS})
    print(f"[r42] wrote {len(rows)} rows -> {out_csv}  ({n_err} errors)")

    for r in rows:
        if r["audit"]:
            print(f"AUDIT {r['case_id']:<32} {r['audit']:<16} phrase={r['phrase']!r} "
                  f"gd_score={r['gd_score']} area_M={r['area_M']}")

    ok = n_err == 0
    if args.smoke:
        # identity gate: the source as its own anchor must give exactly zero on both scores
        c = cases[0]
        src = resize_to(Image.open(args.data_root / c["src_first_frame"]).convert("RGB"), HW)
        row, _ = measure(c, args.data_root, prompts, tgt_obj, models, args, anchor_override=src)
        g_id = abs(row["patch"]) < 1e-5 and row["selfsim"] < 1e-8
        print(f"GATE identity {'PASS' if g_id else 'FAIL'}: patch={row['patch']:.3g} "
              f"selfsim={row['selfsim']:.3g}")
        g_shape = len(rows) == len(EXPECT_PER_TYPE) and all(
            math.isfinite(r["patch"]) and math.isfinite(r["selfsim"]) for r in rows)
        print(f"GATE finite   {'PASS' if g_shape else 'FAIL'}: {len(rows)} cases, all scores finite")
        ok = ok and g_id and g_shape
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
