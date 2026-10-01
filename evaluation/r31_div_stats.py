"""R31 `check-stage2`: the NUMERIC half of the GO/NO-GO.

Run::

    python evaluation/r31_div_stats.py -o evaluation/csv/r31_div_stats.csv

WHY THIS EXISTS
---------------
The plan's `check-stage2` command block prints `frac(m > 0.5)` per clip, which -- as its
own comment admits -- "cannot tell a localized field from a diffuse one, only how much of
it is high". That was tolerable while stage 1 measured a PRE-injection target. It is not
tolerable now.

Since 2026-09-15 stage 1 is a 7-step run whose measured index is POST-injection, so the
background source-KV injection has already pinned the BG to the source (measured: BG
divergence 23% lower, FG/BG contrast 3.79x -> 4.28x). That is desirable for the gate --
it is why the 7-step target was kept -- but it means the divergence field may now be
little more than a GRADED VERSION OF THE R26 GROUNDING MASK. If so, R31 reduces to a
soft-edged R26, and R26's 52-arm sweep already closed that direction.

THE DISCRIMINATOR IS WITHIN-REGION VARIANCE, NOT LOCALIZATION
------------------------------------------------------------
R26's field is CONSTANT inside the union mask and CONSTANT outside it: exactly two values.
So the question "does R31 carry structure R26 cannot express?" is answered by how much `m`
VARIES WITHIN each region, not by whether it localizes:

  * `std_fg` / `std_bg` ~ 0   => the field is two-valued => R31 IS R26 with soft edges.
  * `std_fg` / `std_bg` large => there is genuine per-token structure R26 cannot represent.

`auc` is the companion number: the probability that a randomly chosen FG token scores above
a randomly chosen BG token (Mann-Whitney U / |FG| |BG|, ties counted as half).

  * auc -> 1.00  => `m` reproduces the mask almost perfectly (R31 ~ R26).
  * auc -> 0.50  => `m` is unrelated to the mask -- interesting, but then say WHY.
  * auc <  0.50  => `m` is ANTI-correlated with the mask, which would invert the gate.

READ THEM TOGETHER. High auc with high within-region std is the good case: the field agrees
with the mask about WHERE the edit is while adding graded detail the mask cannot carry.
High auc with near-zero std is the NO-GO case.

OTHER GATES COVERED
-------------------
  * `span` = d_max - d_min per clip per arm. Per-clip min/max normalisation means a clip
    that never diverged still gets a full-range `m` -- pure noise stretched over the whole
    tau span. This is the plan's gate (c).
  * inter-arm token-level correlation. R30 found its eight per-clip measures correlated
    0.40-0.92 and concluded they were "one shared factor measured eight ways". If the same
    holds per-token, R31's four arms are also one measurement and the arm sweep is moot.

`0042_gym-ball` is EXCLUDED from aggregates by default (`--exclude`): it is the degenerate
removal case whose trg_word is a negation with no visual referent, and its union mask
covers 100% of the frame, so it has no background and no FG/BG statistic is defined. It is
still reported per-clip.

No model inference; reads only npz already on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ARMS: Tuple[str, ...] = ("lpips", "dino_patch", "normals", "latent")
DEFAULT_EXCLUDE: Tuple[str, ...] = ("0042_gym-ball",)


def load_mask(mask_root: Path, edit: str, name: str) -> Optional[np.ndarray]:
    """R26's stored union mask -> bool [F_lat, frame_seq_length]."""
    p = mask_root / edit / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return np.unpackbits(z["M"], axis=1)[:, : int(z["frame_seq_length"])].astype(bool)


def auc_mannwhitney(scores: np.ndarray, pos: np.ndarray) -> float:
    """P(score[FG] > score[BG]) with ties counted as half, via rank sums.

    Exact and O(n log n); no sampling. Returns nan when either class is empty (which is
    the `0042_gym-ball` case -- its mask covers the whole frame).
    """
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks over ties, or the AUC is biased whenever `m` saturates (it does:
    # per-clip min/max normalisation pins at least one token to 0 and one to 1).
    s = scores[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--div_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_div"))
    p.add_argument("--mask_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r26_masks"),
                   help="R26's stored grounding-mask unions, the reference R31 is asked "
                        "to beat.")
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE),
                   help="video_names kept in the per-clip table but dropped from the "
                        "aggregates. Default: the degenerate removal case.")
    p.add_argument("-o", "--out", type=Path,
                   default=Path("evaluation/csv/r31_div_stats.csv"))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cases = json.loads(args.cases.expanduser().read_text())
    rows: List[dict] = []
    fields: Dict[Tuple[str, str], np.ndarray] = {}

    for c in cases:
        T, name = int(c["edit_type"]), c["video_name"]
        edit = f"edit{T}"
        M = load_mask(args.mask_root.expanduser(), edit, name)
        for arm in args.arms:
            f = args.div_root.expanduser() / arm / edit / f"{name}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            m = z["m"].astype(np.float64)
            fields[(arm, f"{edit}/{name}")] = m.reshape(-1)
            r = dict(arm=arm, edit_type=T, video_name=name,
                     n_lat=m.shape[0], n_tok=m.shape[1],
                     d_min=float(z["d_min"]), d_max=float(z["d_max"]),
                     span=float(z["d_max"]) - float(z["d_min"]),
                     frac_gt_half=float((m > 0.5).mean()),
                     m_mean=float(m.mean()), m_std=float(m.std()))
            if M is not None and M.shape == m.shape:
                fg, bg = m[M], m[~M]
                r.update(mask_cov=float(M.mean()),
                         auc=auc_mannwhitney(m.reshape(-1), M.reshape(-1)),
                         m_fg=float(fg.mean()) if fg.size else float("nan"),
                         m_bg=float(bg.mean()) if bg.size else float("nan"),
                         std_fg=float(fg.std()) if fg.size else float("nan"),
                         std_bg=float(bg.std()) if bg.size else float("nan"))
                r["contrast"] = (r["m_fg"] / r["m_bg"]) if r.get("m_bg") else float("nan")
            else:
                r.update(mask_cov=float("nan"), auc=float("nan"), m_fg=float("nan"),
                         m_bg=float("nan"), std_fg=float("nan"), std_bg=float("nan"),
                         contrast=float("nan"))
            rows.append(r)

    if not rows:
        raise SystemExit(f"[r31_div_stats] no npz found under {args.div_root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    keep = [r for r in rows if r["video_name"] not in args.exclude]
    print(f"=== R31 check-stage2 statistics  ({len(rows)} rows, "
          f"{len(rows) - len(keep)} excluded from aggregates: "
          f"{', '.join(args.exclude) or 'none'}) ===\n")
    print(f"{'arm':<12}{'n':>4}{'auc':>8}{'m_fg':>8}{'m_bg':>8}{'contr':>8}"
          f"{'std_fg':>9}{'std_bg':>9}{'span_min':>10}{'frac>.5':>9}")
    for arm in args.arms:
        R = [r for r in keep if r["arm"] == arm]
        if not R:
            print(f"{arm:<12}{'--- no output ---':>40}")
            continue
        g = lambda k: np.nanmean([r[k] for r in R])
        print(f"{arm:<12}{len(R):>4}{g('auc'):>8.3f}{g('m_fg'):>8.3f}{g('m_bg'):>8.3f}"
              f"{g('contrast'):>8.2f}{g('std_fg'):>9.3f}{g('std_bg'):>9.3f}"
              f"{min(r['span'] for r in R):>10.4f}{g('frac_gt_half'):>9.3f}")

    # ---- the two readings that decide the GO/NO-GO -----------------------------------
    print("\n--- How to read this ---")
    print("  auc    1.00 => m reproduces the R26 mask (R31 ~ R26 with soft edges)")
    print("         0.50 => m is unrelated to the mask")
    print("  std_fg/std_bg: R26's field is CONSTANT in each region, so these ARE the")
    print("         structure R26 cannot express. Near 0 => nothing new to test.")

    # ---- gate (c): degenerate clips ---------------------------------------------------
    flat = [(r["arm"], r["video_name"], r["span"]) for r in rows if r["span"] <= 1e-9]
    print(f"\n--- gate (c) zero-span clips (noise stretched over the full tau range): "
          f"{len(flat)} ---")
    for a, v, s in flat:
        print(f"    {a:<12} {v:<26} span={s:.3g}")

    # ---- inter-arm redundancy ---------------------------------------------------------
    present = [a for a in args.arms if any(r["arm"] == a for r in keep)]
    if len(present) > 1:
        print("\n--- inter-arm token-level correlation (pooled over clips) ---")
        print("    R30 found its 8 per-clip measures correlated 0.40-0.92 and called them")
        print("    'one shared factor measured eight ways'. If that holds per-token here,")
        print("    the four arms are one measurement and the arm sweep is moot.")
        print(f"\n{'':<12}" + "".join(f"{a:>12}" for a in present))
        for a in present:
            line = f"{a:<12}"
            for b in present:
                keys = [k for k in fields if k[0] == a and
                        (b, k[1]) in fields and k[1].split('/')[-1] not in args.exclude]
                if not keys:
                    line += f"{'--':>12}"; continue
                x = np.concatenate([fields[(a, k[1])] for k in keys])
                y = np.concatenate([fields[(b, k[1])] for k in keys])
                line += f"{np.corrcoef(x, y)[0, 1]:>12.3f}"
            print(line)

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
