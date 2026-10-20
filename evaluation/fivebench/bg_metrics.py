"""R37: distance-weighted background-preservation metrics over the fixed region ¬S_t.

FiVE's official `*_unedit_part` metrics black out the source mask S in both frames and
score the full frame. That penalises legitimate shape changes right next to the object
(person -> lion) exactly as much as leakage far away from it. Here each background
pixel's error is weighted by its distance to S:

    dist = EDT(~S_t)                      (0 on S)
    L    = sqrt(mean_t |S_t|)             (per clip, over the scored frames)
    w    = clip(dist / (alpha * L), 0, 1) (alpha > 0);   w = 1 on ¬S for alpha = 0
    D_w  = sum(w * d) / sum(w)            (d = per-pixel MSE, 1 - SSIM, LPIPS)

The denominator sum(w) depends only on S, never on the edited frames, so a method
cannot improve its score by changing which pixels count (anti-gaming; tested).

alpha = 0 is the plain mean over ¬S. It is NOT FiVE's official number: the official
metric counts S as perfect pixels (MSE divided by H*W, not |¬S|), and uses
LPIPS-SqueezeNet and torchmetrics Gaussian SSIM. The official number is therefore
computed separately, by the unmodified FiVE code (`official_fivebench`).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity

_HERE = os.path.dirname(os.path.abspath(__file__))

# A clip whose mask covers more than this fraction of a frame is flagged (not skipped).
LARGE_MASK_FRAC = 0.9

# FiVE metric names whose unmodified implementation gives the official column.
OFFICIAL_METRICS = {
    "psnr": "psnr_unedit_part",
    "ssim": "ssim_unedit_part",
    "lpips": "lpips_unedit_part",
}


def _import_evaluate():
    """Import FiVE's evaluate.py lazily: it pulls in CLIP/Qwen/CoTracker deps, which the
    weighted metric itself does not need (the unit tests on synthetic data skip it)."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import evaluate  # noqa: E402  (FiVE harness, script-style imports)
    return evaluate


# --------------------------------------------------------------------------- loading


def load_clip(data_root: str, method_root: str, edit_type: int, video: str,
              stride: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load one (source, edit, mask) clip with the same alignment as FiVE's evaluate.py.

    Both sides are strided by `stride` and zipped positionally (truncated to the
    shorter); the edit is resized to the source size with PIL's default filter
    (bicubic), as evaluate.py does; masks are `> 0` of the L-mode JPEG.

    Returns `src [T,H,W,3] uint8`, `edit [T,H,W,3] uint8`, `masks [T,H,W] bool` and an
    info dict of edge-case facts (resize, truncation, mask resize).
    """
    ev = _import_evaluate()
    src_dir = os.path.join(data_root, "images", video)
    mask_dir = os.path.join(data_root, "bmasks", video)
    edit_dir = os.path.join(method_root, f"edit{edit_type}", video)
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"mask dir missing: {mask_dir}")
    if os.path.exists(edit_dir + ".mp4"):
        edit_dir = ev.mp4_to_frames_ffmpeg(edit_dir + ".mp4")

    src_names = ev.list_images(src_dir)[::stride]
    edit_names = ev.list_images(edit_dir)[::stride]
    if not src_names:
        raise FileNotFoundError(f"no source frames in {src_dir}")
    if not edit_names:
        raise FileNotFoundError(f"no edited frames in {edit_dir}")
    n = min(len(src_names), len(edit_names))

    info = {
        "n_src_strided": len(src_names),
        "n_edit_strided": len(edit_names),
        "n_frames": n,
        "truncated": len(src_names) != len(edit_names),
        "edit_resized_from": "",
        "mask_resized": False,
    }

    src, edit, masks = [], [], []
    for src_path, edit_path in zip(src_names[:n], edit_names[:n]):
        s_img = Image.open(src_path)
        e_img = Image.open(edit_path)
        if e_img.size != s_img.size:
            info["edit_resized_from"] = f"{e_img.size[0]}x{e_img.size[1]}"
            e_img = e_img.resize(s_img.size)  # PIL default = bicubic, as evaluate.py
        s_arr, e_arr = np.array(s_img), np.array(e_img)
        if s_arr.ndim != 3 or s_arr.shape[2] != 3 or e_arr.shape != s_arr.shape:
            raise ValueError(f"unexpected frame shapes {s_arr.shape} / {e_arr.shape} "
                             f"({src_path}, {edit_path})")

        m_img = Image.open(os.path.join(mask_dir, os.path.basename(src_path)))
        if m_img.size != s_img.size:
            info["mask_resized"] = True
            m_img = m_img.resize(s_img.size, Image.NEAREST)
        m_arr = np.array(m_img)
        if m_arr.ndim != 2:
            raise ValueError(f"mask is not single-channel: {m_arr.shape}")
        src.append(s_arr)
        edit.append(e_arr)
        masks.append(m_arr > 0)  # same rule as evaluate.py (JPEG ringing counts as S)

    return np.stack(src), np.stack(edit), np.stack(masks), info


# ------------------------------------------------------------------ weighted metric


def alpha_grid(step: float = 0.05, to: float = 1.0) -> list[float]:
    """{0, step, ..., to}, rounded so that 0.15 etc. compare exactly."""
    n = int(round(to / step))
    return [round(k * step, 10) for k in range(n + 1)]


def build_lpips(device: str | torch.device):
    """LPIPS-AlexNet with per-pixel output (`spatial=True`), in eval mode."""
    import lpips
    return lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()


def weight_map(dist: np.ndarray, S: np.ndarray, alpha: float, L: float) -> np.ndarray:
    """w = clip(dist / (alpha*L), 0, 1), 0 on S; alpha = 0 -> indicator of ¬S."""
    if alpha == 0:
        return (~S).astype(np.float64)
    w = np.clip(dist / (alpha * L), 0.0, 1.0)
    w[S] = 0.0
    return w


def _to_float(frames: np.ndarray) -> np.ndarray:
    if frames.dtype == np.uint8:
        return frames.astype(np.float32) / 255.0
    return frames.astype(np.float32)


def error_maps(src: np.ndarray, edit: np.ndarray, lpips_model) -> dict[str, np.ndarray]:
    """Per-pixel error maps `[T,H,W]` for float frames `[T,H,W,3]` in [0, 1]:
    squared error (channel mean), 1 - SSIM (skimage map, channel mean), LPIPS-alex."""
    mse = ((src - edit) ** 2).mean(axis=-1)

    dssim = np.empty(src.shape[:3], dtype=np.float32)
    for t in range(src.shape[0]):
        _, smap = structural_similarity(src[t], edit[t], channel_axis=-1,
                                        data_range=1.0, full=True)
        dssim[t] = 1.0 - smap.mean(axis=-1)

    device = next(lpips_model.parameters()).device
    x = torch.from_numpy(src).permute(0, 3, 1, 2).to(device) * 2 - 1
    y = torch.from_numpy(edit).permute(0, 3, 1, 2).to(device) * 2 - 1
    with torch.no_grad():
        lp = lpips_model(x, y)  # [T,1,H,W], upsampled to the input size
    return {"mse": mse, "dssim": dssim, "lpips": lp[:, 0].float().cpu().numpy()}


def _psnr_from_mse(d: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(1.0 / d)  # d == 0 -> inf, no warning


def distance_weighted_bg_metrics(src_frames: np.ndarray, edit_frames: np.ndarray,
                                 masks: np.ndarray, alphas: Sequence[float],
                                 lpips_model=None, alpha_max: float = 0.5) -> dict:
    """Distance-weighted PSNR/SSIM/LPIPS of one clip for every alpha.

    Args:
        src_frames, edit_frames: `[T,H,W,3]`, uint8 or float in [0, 1].
        masks: `[T,H,W]` bool source-object mask S_t.
        alphas: tolerance values; 0 means the plain mean over ¬S.
        lpips_model: from `build_lpips`; built on CPU if None.
        alpha_max: the alpha-averaged headline uses alphas <= alpha_max.

    Returns a dict with
        `per_alpha[alpha] = {psnr_w, ssim_w, lpips_w}` (video = mean over frames),
        `alpha_avg = {psnr_w, ssim_w, lpips_w}`,
        `D [A, T_scored, 3]` (D_w of mse / 1-ssim / lpips per frame),
        `w_sum [A, T_scored]` (depends on S only),
        `L`, `scored_frames`, and edge-case counters.
    PSNR of a frame with D_w[MSE] = 0 is inf (so the video mean is inf).
    """
    src = _to_float(np.asarray(src_frames))
    edit = _to_float(np.asarray(edit_frames))
    masks = np.asarray(masks, dtype=bool)
    if not (src.shape == edit.shape and src.shape[:3] == masks.shape and src.shape[-1] == 3):
        raise ValueError(f"shape mismatch: src {src.shape}, edit {edit.shape}, "
                         f"masks {masks.shape}")
    alphas = [float(a) for a in alphas]
    if any(a < 0 for a in alphas):
        raise ValueError("alphas must be >= 0")
    avg_idx = [i for i, a in enumerate(alphas) if a <= alpha_max + 1e-9]

    frac = masks.reshape(masks.shape[0], -1).mean(axis=1)
    scored = [t for t in range(masks.shape[0]) if masks[t].any()]
    nan3 = {"psnr_w": float("nan"), "ssim_w": float("nan"), "lpips_w": float("nan")}
    out = {
        "alphas": alphas,
        "scored_frames": scored,
        "n_empty_frames": masks.shape[0] - len(scored),
        "max_mask_frac": float(frac.max()) if frac.size else float("nan"),
        "flag_large_mask": bool(frac.size and frac.max() > LARGE_MASK_FRAC),
        "n_wsum0_frames": 0,
        "clip_skipped": False,
        "L": float("nan"),
        "per_alpha": {a: dict(nan3) for a in alphas},
        "alpha_avg": dict(nan3),
        "D": np.full((len(alphas), len(scored), 3), np.nan),
        "w_sum": np.zeros((len(alphas), len(scored))),
    }
    if not scored:
        out["clip_skipped"] = True  # every frame has an empty mask
        return out

    S = masks[scored]
    L = float(np.sqrt(S.reshape(S.shape[0], -1).sum(axis=1).mean()))
    out["L"] = L

    if lpips_model is None:
        lpips_model = build_lpips("cpu")
    maps = error_maps(src[scored], edit[scored], lpips_model)
    stacked = np.stack([maps["mse"], maps["dssim"], maps["lpips"]], axis=1)  # [T,3,H,W]
    dists = [distance_transform_edt(~S[t]) for t in range(len(scored))]

    D, w_sum = out["D"], out["w_sum"]
    for t in range(len(scored)):
        flat = stacked[t].reshape(3, -1).astype(np.float64)
        for ai, a in enumerate(alphas):
            w = weight_map(dists[t], S[t], a, L).reshape(-1)
            ws = w.sum()
            w_sum[ai, t] = ws
            if ws > 0:
                D[ai, t] = flat @ w / ws
    # sum(w) = 0 only when S covers the whole frame, which then holds for every alpha.
    valid = w_sum[0] > 0
    out["n_wsum0_frames"] = int((~valid).sum())
    if not valid.any():
        out["clip_skipped"] = True
        return out

    Dv = D[:, valid]  # [A, T_valid, 3]
    for ai, a in enumerate(alphas):
        out["per_alpha"][a] = {
            "psnr_w": float(_psnr_from_mse(Dv[ai, :, 0]).mean()),
            "ssim_w": float((1.0 - Dv[ai, :, 1]).mean()),
            "lpips_w": float(Dv[ai, :, 2].mean()),
        }
    if avg_idx:
        # PSNR: average D_w[MSE] over alpha first, then dB, then mean over frames.
        out["alpha_avg"] = {
            "psnr_w": float(_psnr_from_mse(Dv[avg_idx, :, 0].mean(axis=0)).mean()),
            "ssim_w": float(np.mean([out["per_alpha"][alphas[i]]["ssim_w"] for i in avg_idx])),
            "lpips_w": float(np.mean([out["per_alpha"][alphas[i]]["lpips_w"] for i in avg_idx])),
        }
    return out


# ------------------------------------------------------------------ official column


def build_official_calculator(device: str | torch.device):
    """A MetricsCalculator holding ONLY the three metric objects the official bg column
    needs, built exactly as in its __init__ (no CLIP/DINO/CoTracker/Qwen loading)."""
    ev = _import_evaluate()
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    mc = object.__new__(ev.MetricsCalculator)
    mc.device = device
    mc.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
    mc.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
    mc.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return mc


def official_fivebench(src_u8: np.ndarray, edit_u8: np.ndarray, masks: np.ndarray,
                       mc) -> dict[str, float]:
    """FiVE's official `{psnr,ssim,lpips}_unedit_part` clip scores, through the
    unmodified `evaluate.calculate_metric` / `calculate_mean` on every frame (including
    empty-mask frames, as the official harness does)."""
    ev = _import_evaluate()
    vals = {k: [] for k in OFFICIAL_METRICS}
    with torch.no_grad():  # no effect on the values; avoids building LPIPS graphs
        for t in range(src_u8.shape[0]):
            m3 = masks[t][:, :, np.newaxis].repeat(3, axis=2)
            for key, metric in OFFICIAL_METRICS.items():
                vals[key].append(ev.calculate_metric(
                    mc, metric, src_u8[t], edit_u8[t], m3, m3,
                    None, None, None, None, None, None))
    out = {}
    for key, v in vals.items():
        m = ev.calculate_mean(v)
        out[key] = float("nan") if m == "nan" else float(m)
    return out


# ----------------------------------------------------------------------------- CLI

CSV_HEADER = ["method", "sample_id", "file_id", "video_name", "category", "alpha",
              "psnr_w", "ssim_w", "lpips_w"]
EDGE_HEADER = ["method", "sample_id", "n_src_strided", "n_edit_strided", "n_frames",
               "truncated", "edit_resized_from", "mask_resized", "n_empty_frames",
               "clip_skipped", "max_mask_frac", "flag_large_mask", "n_wsum0_frames", "L",
               "gap_psnr", "gap_ssim", "gap_lpips"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R37 distance-weighted bg metrics, one method dir.")
    p.add_argument("--method_root", required=True,
                   help="dir holding edit{T}/{video}/*.png (or {video}.mp4)")
    p.add_argument("--method_name", required=True, help="stem written to the `method` column")
    p.add_argument("--data_root", required=True, help="FiVE root (images/, bmasks/, edit_prompt/)")
    p.add_argument("--edit_types", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--alpha_to", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=0.5,
                   help="alpha-avg headline over alphas <= alpha_max")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--edgecase_csv", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _log_env(args: argparse.Namespace) -> None:
    import lpips
    import scipy
    import skimage
    import PIL
    print(f"[r37] args: {json.dumps(vars(args))}")
    print(f"[r37] python {platform.python_version()} torch {torch.__version__} "
          f"numpy {np.__version__} lpips {getattr(lpips, '__version__', '?')} "
          f"skimage {skimage.__version__} scipy {scipy.__version__} PIL {PIL.__version__}")
    if str(args.device).startswith("cuda"):
        print(f"[r37] device {args.device}: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[r37] device {args.device}")


def _fmt(x: float) -> str:
    return repr(float(x))  # keeps 'inf' / 'nan' readable by pandas


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _log_env(args)

    alphas = alpha_grid(args.alpha_step, args.alpha_to)
    lpips_model = build_lpips(args.device)
    mc = build_official_calculator(args.device)

    for path in (args.out_csv, args.edgecase_csv):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n_clips = 0
    with open(args.out_csv, "w", newline="") as f_out, \
            open(args.edgecase_csv, "w", newline="") as f_edge:
        w_out, w_edge = csv.writer(f_out), csv.writer(f_edge)
        w_out.writerow(CSV_HEADER)
        w_edge.writerow(EDGE_HEADER)
        for T in args.edit_types:
            ann_path = os.path.join(args.data_root, "edit_prompt", f"edit{T}_FiVE.json")
            with open(ann_path, encoding="utf-8") as f:
                items = json.load(f)
            for file_id, item in enumerate(items):
                video = item["video_name"]
                sample_id = f"edit{T}/{video}"
                try:
                    src, edit, masks, info = load_clip(args.data_root, args.method_root,
                                                       T, video, args.stride)
                    res = distance_weighted_bg_metrics(src, edit, masks, alphas,
                                                       lpips_model, args.alpha_max)
                    off = official_fivebench(src, edit, masks, mc)
                except Exception as e:  # report the clip, then fail loudly
                    print(f"ERROR {sample_id}: {type(e).__name__}: {e}", flush=True)
                    raise

                base = [args.method_name, sample_id, file_id, video, item["editing_type_id"]]
                for a in alphas:
                    sc = res["per_alpha"][a]
                    w_out.writerow(base + [a] + [_fmt(sc[m]) for m in ("psnr_w", "ssim_w", "lpips_w")])
                sc = res["alpha_avg"]
                w_out.writerow(base + ["avg"] + [_fmt(sc[m]) for m in ("psnr_w", "ssim_w", "lpips_w")])
                w_out.writerow(base + ["official"] + [_fmt(off[k]) for k in ("psnr", "ssim", "lpips")])

                clean0 = res["per_alpha"][0.0] if 0.0 in res["per_alpha"] else None
                gaps = ([off["psnr"] - clean0["psnr_w"], off["ssim"] - clean0["ssim_w"],
                         off["lpips"] - clean0["lpips_w"]] if clean0 else [float("nan")] * 3)
                w_edge.writerow([
                    args.method_name, sample_id, info["n_src_strided"], info["n_edit_strided"],
                    info["n_frames"], int(info["truncated"]), info["edit_resized_from"],
                    int(info["mask_resized"]), res["n_empty_frames"], int(res["clip_skipped"]),
                    _fmt(res["max_mask_frac"]), int(res["flag_large_mask"]),
                    res["n_wsum0_frames"], _fmt(res["L"])] + [_fmt(g) for g in gaps])
                f_out.flush()
                f_edge.flush()
                n_clips += 1
                if res["clip_skipped"]:
                    print(f"[r37] {sample_id}: weighted metric skipped (empty masks)")
            print(f"[r37] edit{T}: {len(items)} clips done", flush=True)
    print(f"[r37] done: {n_clips} clips, {len(alphas)} alphas -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
