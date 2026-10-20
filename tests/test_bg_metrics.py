"""R37 gates for evaluation/fivebench/bg_metrics.py.

Run:  ~/anaconda3/envs/five-bench/bin/python -m pytest tests/test_bg_metrics.py -v
Tests 1-6 use small synthetic clips on CPU; test 7 is a parity smoke on real data
(skipped when the FiVE data, the R36 render or its official CSV is missing).
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation" / "fivebench"))
import bg_metrics as bm  # noqa: E402

SEED = 0
H, W = 128, 224
ALPHAS = bm.alpha_grid(0.05, 1.0)
METRICS = ("psnr_w", "ssim_w", "lpips_w")


@pytest.fixture(scope="module")
def lpips_model():
    torch.manual_seed(SEED)
    return bm.build_lpips("cpu")


def disc_mask(h: int, w: int, radius: float, t: int = 2) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    m = (yy - h / 2 + 0.5) ** 2 + (xx - w / 2 + 0.5) ** 2 <= radius ** 2
    return np.repeat(m[None], t, axis=0)


def textured_scene(t: int = 2) -> np.ndarray:
    """Smooth gradient + mild deterministic texture, [T,H,W,3] float32 in [0,1]."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
    base = np.stack([0.3 + 0.4 * xx, 0.4 + 0.3 * yy, 0.5 + 0.2 * np.sin(6 * xx)], -1)
    tex = 0.05 * rng.standard_normal((H, W, 3))
    frame = np.clip(base + tex, 0, 1).astype(np.float32)
    return np.repeat(frame[None], t, axis=0)


def worse(metric: str, a: float, b: float) -> bool:
    """True if score a is worse than score b."""
    return a > b if metric == "lpips_w" else a < b


# --------------------------------------------------------------------------- tests


def test_identical_frames_are_perfect(lpips_model):
    src = textured_scene()
    S = disc_mask(H, W, 20)
    res = bm.distance_weighted_bg_metrics(src, src.copy(), S, ALPHAS, lpips_model)
    for scores in list(res["per_alpha"].values()) + [res["alpha_avg"]]:
        assert scores["lpips_w"] < 1e-6
        assert scores["ssim_w"] > 1 - 1e-6
        assert scores["psnr_w"] == np.inf


def test_alpha0_equals_plain_outside_mask_mean(lpips_model):
    rng = np.random.default_rng(SEED)
    src = textured_scene()
    edit = np.clip(src + 0.05 * rng.standard_normal(src.shape), 0, 1).astype(np.float32)
    S = disc_mask(H, W, 20)
    res = bm.distance_weighted_bg_metrics(src, edit, S, ALPHAS, lpips_model)

    # Independent reference: boolean-indexed mean over ¬S of each map, per frame,
    # accumulated in float64 like the module (float32 means are off by ~1e-6 dB).
    maps = bm.error_maps(src, edit, lpips_model)
    ref_mse = [maps["mse"][t][~S[t]].astype(np.float64).mean() for t in range(len(S))]
    ref_dssim = [maps["dssim"][t][~S[t]].astype(np.float64).mean() for t in range(len(S))]
    ref_lpips = [maps["lpips"][t][~S[t]].astype(np.float64).mean() for t in range(len(S))]
    ref = {
        "psnr_w": np.mean([10 * np.log10(1 / d) for d in ref_mse]),
        "ssim_w": np.mean([1 - d for d in ref_dssim]),
        "lpips_w": np.mean(ref_lpips),
    }
    for m in METRICS:
        assert abs(res["per_alpha"][0.0][m] - ref[m]) <= 1e-6, m


def test_weight_sum_independent_of_edit(lpips_model):
    rng = np.random.default_rng(SEED)
    src = textured_scene()
    S = disc_mask(H, W, 20)
    edit_a = np.clip(src + 0.1 * rng.standard_normal(src.shape), 0, 1).astype(np.float32)
    edit_b = np.clip(1 - src, 0, 1).astype(np.float32)
    res_a = bm.distance_weighted_bg_metrics(src, edit_a, S, ALPHAS, lpips_model)
    res_b = bm.distance_weighted_bg_metrics(src, edit_b, S, ALPHAS, lpips_model)
    assert np.array_equal(res_a["w_sum"], res_b["w_sum"])
    assert res_a["per_alpha"][0.25]["lpips_w"] != res_b["per_alpha"][0.25]["lpips_w"]


def test_far_perturbation_hurts_more_than_near(lpips_model):
    src = textured_scene()
    S = disc_mask(H, W, 20)
    alpha = 0.5
    L = np.sqrt(S.reshape(len(S), -1).sum(1).mean())
    rho = alpha * L
    dist = distance_transform_edt(~S[0])

    near = (dist > 0) & (dist < 0.2 * rho)
    n = int(near.sum())
    far_idx = np.flatnonzero(dist.reshape(-1) > 1.5 * rho)[:n]
    assert len(far_idx) == n, "frame too small for an equal-size far region"
    far = np.zeros(H * W, dtype=bool)
    far[far_idx] = True
    far = far.reshape(H, W)

    def perturbed(region):
        e = src.copy()
        e[:, region] = np.clip(e[:, region] + 0.3, 0, 1)
        return e

    res_near = bm.distance_weighted_bg_metrics(src, perturbed(near), S, ALPHAS, lpips_model)
    res_far = bm.distance_weighted_bg_metrics(src, perturbed(far), S, ALPHAS, lpips_model)
    assert res_far["per_alpha"][alpha]["psnr_w"] < res_near["per_alpha"][alpha]["psnr_w"]
    assert res_far["alpha_avg"]["psnr_w"] < res_near["alpha_avg"]["psnr_w"]


def test_larger_perturbed_area_scores_worse(lpips_model):
    rng = np.random.default_rng(SEED)
    src = textured_scene()
    S = disc_mask(H, W, 20)
    bg = np.flatnonzero(~S[0].reshape(-1))
    order = rng.permutation(bg)
    noise = 0.15 * rng.standard_normal((H * W, 3)).astype(np.float32)

    def perturbed(fraction):
        idx = order[: int(fraction * len(bg))]
        e = src.reshape(len(S), H * W, 3).copy()
        e[:, idx] = np.clip(e[:, idx] + noise[idx], 0, 1)
        return e.reshape(src.shape)

    res_5 = bm.distance_weighted_bg_metrics(src, perturbed(0.05), S, ALPHAS, lpips_model)
    res_60 = bm.distance_weighted_bg_metrics(src, perturbed(0.60), S, ALPHAS, lpips_model)
    for m in METRICS:
        assert worse(m, res_60["alpha_avg"][m], res_5["alpha_avg"][m]), m


def test_downscale_stability(lpips_model):
    """Smooth scene + smooth off-object error: alpha-avg changes < 5% at 2x downscale."""
    t = 2
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
    src = np.stack([0.3 + 0.4 * xx, 0.4 + 0.3 * yy, 0.5 + 0.2 * np.sin(3 * xx)], -1)
    bump = 0.1 * np.exp(-(((yy - 0.25) / 0.2) ** 2 + ((xx - 0.2) / 0.15) ** 2))
    edit = np.clip(src + bump[..., None], 0, 1)
    src = np.repeat(src[None], t, 0).astype(np.float32)
    edit = np.repeat(edit[None], t, 0).astype(np.float32)
    S = disc_mask(H, W, 20, t)

    def down2(x):  # exact 2x2 area average
        return x.reshape(t, H // 2, 2, W // 2, 2, 3).mean(axis=(2, 4)).astype(np.float32)

    S_half = disc_mask(H // 2, W // 2, 10, t)
    full = bm.distance_weighted_bg_metrics(src, edit, S, ALPHAS, lpips_model)["alpha_avg"]
    half = bm.distance_weighted_bg_metrics(down2(src), down2(edit), S_half, ALPHAS,
                                           lpips_model)["alpha_avg"]
    for m in METRICS:
        rel = abs(half[m] - full[m]) / abs(full[m])
        assert rel < 0.05, f"{m}: full {full[m]:.6f} half {half[m]:.6f} rel {rel:.4f}"


# ------------------------------------------------------------- real-data smoke test

DATA_ROOT = os.path.expanduser("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
ARM_ROOT = os.path.expanduser("~/Data/dataggen/outputs/five_bench/r36_rho_sweep/r36_rho2_vp")
OFFICIAL_CSV = REPO / "evaluation" / "csv" / "edit1_FiVE_r36_rho2_vp_frame_stride8.csv"


@pytest.mark.skipif(
    not (os.path.isdir(DATA_ROOT) and os.path.isdir(ARM_ROOT) and OFFICIAL_CSV.exists()),
    reason="FiVE data, R36 rho2 render or its official CSV not available")
def test_official_parity_real_clip():
    """Official column on 0001_bus e1 (file_id 0) == R36's per-clip CSV to 1e-4: proves
    the frame/mask alignment of load_clip matches evaluate.py."""
    src, edit, masks, info = bm.load_clip(DATA_ROOT, ARM_ROOT, 1, "0001_bus", stride=8)
    assert info["n_frames"] == 9
    mc = bm.build_official_calculator("cpu")
    got = bm.official_fivebench(src, edit, masks, mc)

    with open(OFFICIAL_CSV) as f:
        row = next(r for r in csv.DictReader(f) if r["file_id"] == "0")
    for key, metric in bm.OFFICIAL_METRICS.items():
        want = float(row[f"r36_rho2_vp|{metric}"])
        assert abs(got[key] - want) <= 1e-4, f"{key}: got {got[key]} want {want}"
