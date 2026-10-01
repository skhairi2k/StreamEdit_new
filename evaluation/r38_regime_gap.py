"""R38 regime gap: how far the anchored stage-1 regime moves the draft away from UNBLENDED.

At N=5, ``tgt0.9`` anchors only the t=1.0 call -- whose current-chunk input is the same
pure noise in both branches -- so it may land close to ``unblended``. If it does, its
divergence maps add nothing and stages 2-3 on the two tgt09 combos are wasted GPU time.
This script measures that gap, and the SAME gap at N=15 (R35's ``first2`` vs
``unblended``; == ``tgt0.9`` there, R38 smoke gate G1) as the reference it is read
against at ``gate-lowN``.

Two measures per N:
  * latent gap -- ``||z_a - z_b|| / ||z_b||`` on the stage-1 ``z_trg`` (the measured final
    x0), per clip, over ALL 419 pairs. Numpy only.
  * frame gap  -- mean LPIPS (AlexNet, the stage-2 LPIPS arm's own loader) between the two
    regimes' final frames, on the 22 ``cases.json`` clips, every ``--stride``-th frame.

Writes ``evaluation/csv/r38_regime_gap.csv``, one row per N. Read-only on the stage-1
trees; hard-fails on any missing clip rather than averaging over fewer.

Run in the ``five-bench`` env (``lpips`` is not installed in ``streamgve``).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r30_divergence import load_lpips, to_unit_tensor  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIVE = Path("~/Data/dataggen/outputs/five_bench").expanduser()
EXPECT = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
PIXEL_HW = (480, 832)

# (N, anchored regime, unblended regime): frames root, frame step dir, latents root.
# N=15 is R35's tree: its first2 IS R38's tgt0.9 at 15 steps (smoke gate G1, job 1016420).
PAIRS = [
    dict(N=15, a="first2", b="unblended", frames=FIVE / "r35_stage1", step="step14",
         latents=FIVE / "r35_latents"),
    dict(N=5, a="n5_tgt09", b="n5_unblended", frames=FIVE / "r38_stage1", step="step04",
         latents=FIVE / "r38_latents"),
]


def latent_gaps(root: Path, a: str, b: str) -> List[float]:
    """Per-clip relative z_trg gap over the full bench; asserts the exact breakdown."""
    gaps: List[float] = []
    for T, n_exp in EXPECT.items():
        files_a = sorted((root / a / f"edit{T}").glob("*.npz"))
        names_b = {p.name for p in (root / b / f"edit{T}").glob("*.npz")}
        if len(files_a) != n_exp or {p.name for p in files_a} != names_b:
            raise SystemExit(f"[r38_regime_gap] {root.name}/{{{a},{b}}}/edit{T}: "
                             f"{len(files_a)} vs {len(names_b)} npz, expect {n_exp} matching")
        for fa in files_a:
            za = np.load(fa)["z_trg"].astype(np.float64)
            zb = np.load(root / b / f"edit{T}" / fa.name)["z_trg"].astype(np.float64)
            if za.shape != zb.shape:
                raise SystemExit(f"[r38_regime_gap] shape mismatch {fa}: {za.shape} vs {zb.shape}")
            gaps.append(float(np.linalg.norm(za - zb) / np.linalg.norm(zb)))
    return gaps


def frame_lpips(model, root: Path, step: str, a: str, b: str,
                cases: Sequence[Tuple[int, str]], stride: int, device: str) -> List[float]:
    """Per-clip mean LPIPS between the two regimes' final frames (every stride-th frame)."""
    out: List[float] = []
    for T, v in cases:
        da, db = root / a / step / f"edit{T}" / v, root / b / step / f"edit{T}" / v
        fa, fb = sorted(da.glob("*.png")), sorted(db.glob("*.png"))
        if not fa or [p.name for p in fa] != [p.name for p in fb]:
            raise SystemExit(f"[r38_regime_gap] frame lists differ or empty: {da} vs {db}")
        vals = []
        for pa in fa[::stride]:
            x = to_unit_tensor(Image.open(pa).convert("RGB"), PIXEL_HW, device) * 2.0 - 1.0
            y = to_unit_tensor(Image.open(db / pa.name).convert("RGB"), PIXEL_HW, device) * 2.0 - 1.0
            with torch.no_grad():
                vals.append(float(model(x, y).mean()))
        out.append(float(np.mean(vals)))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cases", type=Path, default=REPO / "evaluation" / "cases.json")
    p.add_argument("--stride", type=int, default=8, help="frame stride for the LPIPS gap")
    p.add_argument("-o", "--out", type=Path, default=REPO / "evaluation" / "csv" / "r38_regime_gap.csv")
    args = p.parse_args(argv)

    cases = [(int(c["edit_type"]), c["video_name"]) for c in json.loads(args.cases.read_text())]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_lpips(device)
    print(f"[r38_regime_gap] device={device} cases={len(cases)} stride={args.stride}", flush=True)

    rows: List[Dict[str, object]] = []
    for pr in PAIRS:
        lg = latent_gaps(pr["latents"], pr["a"], pr["b"])
        fl = frame_lpips(model, pr["frames"], pr["step"], pr["a"], pr["b"],
                         cases, args.stride, device)
        row = dict(N=pr["N"], anchored=pr["a"], unblended=pr["b"],
                   n_clips_latent=len(lg),
                   latent_gap_mean=round(float(np.mean(lg)), 5),
                   latent_gap_median=round(float(np.median(lg)), 5),
                   latent_gap_p10=round(float(np.percentile(lg, 10)), 5),
                   latent_gap_p90=round(float(np.percentile(lg, 90)), 5),
                   n_clips_lpips=len(fl),
                   lpips_mean=round(float(np.mean(fl)), 5),
                   lpips_median=round(float(np.median(fl)), 5))
        rows.append(row)
        print(f"[r38_regime_gap] N={pr['N']}: {row}", flush=True)

    ref, low = rows[0], rows[1]
    print(f"[r38_regime_gap] N=5 / N=15 ratio: latent gap {low['latent_gap_mean'] / ref['latent_gap_mean']:.3f}, "
          f"LPIPS {low['lpips_mean'] / ref['lpips_mean']:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[r38_regime_gap] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
