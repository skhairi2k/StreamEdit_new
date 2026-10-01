"""R9 recalibration: budget-normalised head scores from the dumped attention maps.

WHY
---
The R9 margin

    m = (err_spatial - err_temporal) / (err_spatial + err_temporal)

compares reconstruction errors from two key sets of very different size
(``r9_head_profiler.py:185-187``): the spatial set is the query's own latent
frame (``FRAME_TOKENS`` = 1560 keys), the temporal set is the query's own
spatial position across visible frames (``frames_vis`` keys, 3-18 here). At the
classification point that is 1560 vs 18 -- an 87x budget gap, and 520x at the
first block.

Restricted-softmax reconstruction error falls as the retained key set grows, so
a head with *no* specialisation reconstructs its output better from 1560 keys
than from 18 purely by sample size. ``m < 0`` is therefore the null expectation,
not evidence of spatial specialisation, and the sign of ``m`` cannot be read as
a head label. This script asks how much of the 281-vs-10 split survives once the
budget is normalised away.

WHAT THIS MEASURES (and what it does not)
-----------------------------------------
The profiler saves no q/k/v, only the reduced errors and ``maps`` -- the full
attention distribution over visible keys for ``MAP_ROWS`` query rows at the last
block and last profile step. So this is NOT the size-matched version of the SVG
output-reconstruction criterion; that needs the values ``v`` and a GPU re-run
(see RECOMMENDED FOLLOW-UP below).

What it does compute is where a head's attention *mass* goes, normalised by what
each key set would receive by chance:

    enrichment  e_S = mass(S) / (|S| / S_vis)          # 1.0 == chance
    score       d   = log2(e_temporal) - log2(e_spatial)

``d`` is dimensionless and budget-free: both terms are already divided by their
own set size, so the 87x gap cancels. Sign convention matches R9's margin --
``d > 0`` means the head over-attends its temporal set more than its spatial
one, relative to chance.

Mass is a proxy for the reconstruction criterion, not a restatement of it: a
head can place mass on a key set whose values are redundant, or reconstruct well
from little mass. Treat ``d`` as a calibration check on the R9 labels, not as a
replacement metric.

THE SELF KEY
------------
The two sets intersect in exactly one key -- the query's own token (its own
frame, its own position). It is 1/1560 of the spatial set but 1/|temporal| of
the temporal set, so if a head attends strongly to itself that inflates temporal
enrichment far more than spatial. The original criterion includes it in both.
This script reports both variants and leads with the exclusive one.

RECOMMENDED FOLLOW-UP (needs GPU)
---------------------------------
For the true size-matched SVG criterion, add to ``_capture`` a third error that
subsamples ``spat_mask`` down to ``|temp_mask|`` keys (averaged over several
draws) and re-run the 5 videos. That measures the same quantity R9 reported,
with the budget controlled. This script is the no-GPU screen that says whether
that re-run is warranted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

NUM_LAYERS = 30
NUM_HEADS = 12

# R9's adopted thresholds, reproduced here only to label heads for the contingency
# table -- this script does not depend on them being correct.
SPATIAL_MAX_MARGIN = -0.2
TEMPORAL_MIN_MARGIN = 0.4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--profile_root",
        type=Path,
        default=Path("/projects/dataggen/outputs/five_bench/r9_head_profile"),
        help="Root holding {case}/r9_scalars.npz from the R9 profiler.",
    )
    p.add_argument(
        "--margins_csv",
        type=Path,
        default=Path("evaluation/csv/r9_head_margins.csv"),
        help="R9 margin table, joined in for the contingency table.",
    )
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path("evaluation/csv/r9_recalibrated_heads.csv"),
        help="Per-head output: R9 margin alongside the budget-normalised scores.",
    )
    p.add_argument(
        "--include_self_key",
        action="store_true",
        help="Keep the query's own token in both sets (the original convention). "
             "Default excludes it, since it is 1/|temporal| of the temporal set "
             "but 1/1560 of the spatial one and so inflates temporal enrichment.",
    )
    return p.parse_args()


def load_margins(csv_path: Path) -> np.ndarray:
    """R9 margins as a dense [NUM_LAYERS, NUM_HEADS] array, or NaN if absent."""
    margins = np.full((NUM_LAYERS, NUM_HEADS), np.nan)
    if not csv_path.exists():
        print(f"[recal] WARNING: {csv_path} missing -- contingency table skipped")
        return margins
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            margins[int(row["layer"]), int(row["head"])] = float(row["margin"])
    return margins


def set_masks(
    row: int, frames_vis: int, q_frames: int, frame_tokens: int, s_k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial / temporal key masks for one query row.

    Mirrors ``r9_head_profiler._capture`` exactly: the query row is block-local,
    so its absolute frame is offset by the frames already in the cache.
    """
    row_frame = frames_vis - q_frames + row // frame_tokens
    row_pos = row % frame_tokens
    key_idx = np.arange(s_k)
    spat = (key_idx // frame_tokens) == row_frame
    temp = (key_idx % frame_tokens) == row_pos
    return spat, temp


def log2_enrichment(probs: np.ndarray, mask: np.ndarray, s_k: int) -> np.ndarray:
    """log2 of (observed mass / mass expected under uniform attention).

    probs: [H, S_k] for one query row. Returns [H].
    """
    observed = probs[:, mask].sum(axis=1)
    expected = mask.sum() / s_k
    # Floor at a mass far below anything meaningful rather than emitting -inf:
    # a head with literally zero mass on a set is a valid, extreme observation.
    observed = np.maximum(observed, 1e-12)
    return np.log2(observed / expected)


def score_case_from_masses(npz_path: Path) -> Dict[str, np.ndarray]:
    """Preferred path: read the mass columns the profiler now records directly.

    These cover all N_SAMPLE_ROWS random query rows (the same rows the errors use),
    rather than the handful of fixed MAP_ROWS the map dump happens to hold, so this
    is the version to trust. Uses the last block / last profile step -- the
    classification point R9 labels heads at.
    """
    data = np.load(npz_path)
    meta = json.loads(data["meta"].tobytes().decode())
    frame_tokens = int(data["frame_tokens"])

    s_k = int(data["svis"][-1])
    frames_vis = s_k // frame_tokens
    # log means are geometric averages over rows, matching the map-based path.
    log_mass_spat = data["logmass_spat"][-1, -1]      # [L, H] at last block/step
    log_mass_temp = data["logmass_temp"][-1, -1]

    return {
        "log2_e_spat": log_mass_spat - np.log2(frame_tokens / s_k),
        "log2_e_temp": log_mass_temp - np.log2(frames_vis / s_k),
        "frames_vis": np.asarray(frames_vis),
        "case": meta["case_id"],
        "source": "mass columns (all sampled rows)",
    }


def score_case(npz_path: Path, include_self: bool) -> Dict[str, np.ndarray]:
    """Per-head log2 enrichments for one video, averaged over the dumped rows."""
    data = np.load(npz_path)
    if "logmass_spat" in data.files:
        return score_case_from_masses(npz_path)
    if "maps" not in data.files:
        raise SystemExit(
            f"[recal] {npz_path} has neither the 'logmass_*' columns nor 'maps'. "
            "Re-run r9_head_profiler.py (its mass columns are always recorded)."
        )
    print(f"[recal] WARNING: {npz_path.parent.name} predates the mass columns; falling "
          "back to the legacy map dump, which covers only the fixed MAP_ROWS "
          "(2 distinct positions, both at grid column 48/52). Re-run the profiler.")

    maps = data["maps"]                       # [L, H, R, S_k] fp16, probs
    map_rows = data["map_rows"]
    frame_tokens = int(data["frame_tokens"])
    meta = json.loads(data["meta"].tobytes().decode())

    n_layers, n_heads, n_rows, s_k = maps.shape
    frames_vis = s_k // frame_tokens
    q_frames = meta["n_latent_frames"] // meta["n_blocks"]

    ls = np.zeros((n_layers, n_heads, n_rows))
    lt = np.zeros((n_layers, n_heads, n_rows))

    for r_i, row in enumerate(map_rows[:n_rows]):
        spat, temp = set_masks(int(row), frames_vis, q_frames, frame_tokens, s_k)
        if not include_self:
            # The single shared key -- own frame AND own position.
            self_key = (frames_vis - q_frames + int(row) // frame_tokens) * frame_tokens \
                + int(row) % frame_tokens
            spat[self_key] = False
            temp[self_key] = False
        for layer in range(n_layers):
            probs = maps[layer, :, r_i, :].astype(np.float64)
            ls[layer, :, r_i] = log2_enrichment(probs, spat, s_k)
            lt[layer, :, r_i] = log2_enrichment(probs, temp, s_k)

    return {
        "log2_e_spat": ls.mean(axis=2),   # geometric mean over rows
        "log2_e_temp": lt.mean(axis=2),
        "frames_vis": np.asarray(frames_vis),
        "case": meta["case_id"],
        "source": f"legacy map dump ({n_rows} fixed rows)",
    }


def r9_class(margin: float) -> str:
    if np.isnan(margin):
        return "unknown"
    if margin > TEMPORAL_MIN_MARGIN:
        return "temporal"
    if margin < SPATIAL_MAX_MARGIN:
        return "spatial"
    return "middle"


def main() -> None:
    args = parse_args()

    npz_paths = sorted(args.profile_root.glob("*/r9_scalars.npz"))
    if not npz_paths:
        raise SystemExit(f"[recal] no r9_scalars.npz under {args.profile_root}")

    per_case = [score_case(p, args.include_self_key) for p in npz_paths]
    cases = [c["case"] for c in per_case]
    ls = np.stack([c["log2_e_spat"] for c in per_case])   # [V, L, H]
    lt = np.stack([c["log2_e_temp"] for c in per_case])

    log_e_spat = ls.mean(axis=0)
    log_e_temp = lt.mean(axis=0)
    d_cal = log_e_temp - log_e_spat
    # Cross-video sign stability of the calibrated score, the analogue of R9's flip rate.
    d_per_case = lt - ls
    sign_agree = (np.sign(d_per_case) == np.sign(d_cal)[None]).mean(axis=0)

    margins = load_margins(args.margins_csv)

    variant = "including" if args.include_self_key else "excluding"
    print(f"[recal] {len(cases)} videos: {', '.join(cases)}")
    print(f"[recal] source: {per_case[0]['source']}")
    print(f"[recal] key sets {variant} the shared self key; "
          f"spatial={per_case[0]['frames_vis']} frames visible")
    print()

    # ---- the headline question: are the R9 'spatial' heads actually enriched? ----
    print("=== Budget-normalised enrichment by R9 class ===")
    print(f"{'R9 class':>10} {'n':>4} {'log2 e_spat':>12} {'log2 e_temp':>12} "
          f"{'e_spat<=1':>10} {'e_temp>1':>9}")
    for name in ("spatial", "middle", "temporal"):
        sel = np.array([[r9_class(margins[i, j]) == name for j in range(NUM_HEADS)]
                        for i in range(NUM_LAYERS)])
        if not sel.any():
            continue
        n = int(sel.sum())
        at_chance = int((log_e_spat[sel] <= 0.0).sum())
        temp_enr = int((log_e_temp[sel] > 0.0).sum())
        print(f"{name:>10} {n:4d} {log_e_spat[sel].mean():12.2f} "
              f"{log_e_temp[sel].mean():12.2f} {at_chance:10d} {temp_enr:9d}")
    print()
    print("  e_spat<=1 counts heads with NO above-chance mass on their own frame:")
    print("  under the R9 label they are 'spatial', but they are unspecialised.")
    print()

    # ---- calibrated ranking ----
    order = np.argsort(d_cal, axis=None)[::-1]
    print("=== Top 12 heads by calibrated score d = log2 e_temp - log2 e_spat ===")
    print(f"{'layer':>6}{'head':>5}{'d_cal':>8}{'log2 e_t':>10}{'log2 e_s':>10}"
          f"{'m_r9':>8}{'r9 class':>10}{'sign ok':>8}")
    for flat in order[:12]:
        i, j = int(flat) // NUM_HEADS, int(flat) % NUM_HEADS
        print(f"{i:6d}{j:5d}{d_cal[i, j]:8.2f}{log_e_temp[i, j]:10.2f}"
              f"{log_e_spat[i, j]:10.2f}{margins[i, j]:8.2f}"
              f"{r9_class(margins[i, j]):>10}{sign_agree[i, j]:8.0%}")
    print()

    # ---- agreement between the two labellings, top-10 vs top-10 ----
    cal_top10 = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=bool)
    for flat in order[:10]:
        cal_top10[int(flat) // NUM_HEADS, int(flat) % NUM_HEADS] = True
    r9_temporal = margins > TEMPORAL_MIN_MARGIN
    overlap = int((cal_top10 & r9_temporal).sum())
    print(f"=== R9 temporal core vs calibrated top-10: {overlap}/10 shared ===")
    if overlap < 10:
        lost = [(i, j) for i in range(NUM_LAYERS) for j in range(NUM_HEADS)
                if r9_temporal[i, j] and not cal_top10[i, j]]
        gained = [(i, j) for i in range(NUM_LAYERS) for j in range(NUM_HEADS)
                  if cal_top10[i, j] and not r9_temporal[i, j]]
        print(f"  only R9:         {lost}")
        print(f"  only calibrated: {gained}")
    print()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer", "head", "margin_r9", "class_r9",
            "log2_e_spat", "log2_e_temp", "d_cal",
            "sign_agree_across_videos", "cal_top10",
        ])
        for i in range(NUM_LAYERS):
            for j in range(NUM_HEADS):
                writer.writerow([
                    i, j, f"{margins[i, j]:.6f}", r9_class(margins[i, j]),
                    f"{log_e_spat[i, j]:.6f}", f"{log_e_temp[i, j]:.6f}",
                    f"{d_cal[i, j]:.6f}", f"{sign_agree[i, j]:.4f}",
                    int(cal_top10[i, j]),
                ])
    print(f"[recal] per-head table -> {args.out_csv}")


if __name__ == "__main__":
    main()
