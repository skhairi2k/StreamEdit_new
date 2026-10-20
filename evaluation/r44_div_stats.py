"""R44 divergence stats: does dropping the visual prompt change the raw divergence the
routing is built on?

Without VP the stage-1 draft itself changes (no edited anchor frame to propagate), so its
divergence from the source changes too -- before routing does anything. R44 keeps R35's
ABSOLUTE normalisation (``m = clip(d, 0, div_max) / div_max``, ``div_max`` 0.65 DINO /
1.0 LPIPS), set on VP drafts. If text-only ``d`` is systematically lower, m shrinks
everywhere and stage 3 anchors more; if higher, more tokens saturate at m=1. Either shifts
the routed arm along the trade-off for a reason unrelated to WHERE the map points, which is
the confound to read before interpreting a null routing gain. This table gates nothing.

Per selected combo (evaluation/r44_selected_combos.txt), two rows -- text-only
(``r44_div/{combo}``) and its VP counterpart:
    n15_tgt09      -> r35_div/{metric}_first2     (tau = 0.9 IS first2 at N=15, R38 gate G1)
    n15_unblended  -> r35_div/{metric}_unblended
    n5_{regime}    -> r38_div/{combo}
pooled over every token of every clip (``d`` is [n_latent_frames, 1560] per clip, so
longer clips weigh more -- the weighting the stage-3 render sees): mean, p50, p90, p99 of
``d``, and the fraction of tokens with ``d >= div_max``. Both sides cover the same 419
clips, so their token counts must match exactly (asserted).

Writes ``evaluation/csv/r44_div_stats.csv``. Numpy only; hard-fails on any combo dir that
is not exactly 100/100/100/100/9/10 (``pooled_d`` from r38_div_stats.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r38_div_stats import DIV_MAX, FIVE, REPO, pooled_d  # noqa: E402

# Must match R44_ALL_COMBOS in slurm_scripts/five_bench/r44_combos.sh.
ALL_COMBOS = ("dino_n15_tgt09", "dino_n15_unblended", "lpips_n15_tgt09", "lpips_n15_unblended",
              "dino_n5_tgt09", "dino_n5_unblended", "lpips_n5_tgt09", "lpips_n5_unblended")


def parse_combo(combo: str) -> Tuple[str, int, str]:
    """'lpips_n5_tgt09' -> ('lpips', 5, 'tgt09')."""
    metric, ntag, regime = combo.split("_")
    return metric, int(ntag[1:]), regime


def read_selection(path: Path) -> List[str]:
    """Same rules as r44_combos.sh's r44_selected: non-empty, no blank/comment lines,
    known combos only, no duplicates. File order is kept."""
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"[r44_div_stats] selection file missing or empty: {path} "
                         f"-- run the choose-config step first")
    combos, bad = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.rstrip("\r")
        if not line.strip() or line.startswith("#"):
            bad.append(f"{path}:{i}: blank or comment line not allowed")
        elif line not in ALL_COMBOS:
            bad.append(f"{path}:{i}: unknown combo {line!r}")
        elif line in combos:
            bad.append(f"{path}:{i}: duplicate {line!r}")
        else:
            combos.append(line)
    if bad:
        raise SystemExit("[r44_div_stats] invalid selection:\n  " + "\n  ".join(bad))
    return combos


def vp_counterpart(combo: str) -> Path:
    metric, n, regime = parse_combo(combo)
    if n == 15:
        return FIVE / "r35_div" / f"{metric}_{'first2' if regime == 'tgt09' else 'unblended'}"
    return FIVE / "r38_div" / combo


def stats_row(modality: str, combo: str, combo_dir: Path) -> Dict[str, object]:
    metric, n, regime = parse_combo(combo)
    d = pooled_d(combo_dir)
    dmax = DIV_MAX[metric]
    p50, p90, p99 = np.percentile(d, [50, 90, 99])
    return dict(modality=modality, combo=combo, source_dir=f"{combo_dir.parent.name}/{combo_dir.name}",
                N=n, metric=metric, regime=regime, div_max=dmax, n_tokens=int(d.size),
                d_mean=round(float(d.mean()), 5), d_p50=round(float(p50), 5),
                d_p90=round(float(p90), 5), d_p99=round(float(p99), 5),
                sat_frac=round(float(np.mean(d >= dmax)), 5))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selection", type=Path, default=REPO / "evaluation" / "r44_selected_combos.txt")
    p.add_argument("--text_root", type=Path, default=FIVE / "r44_div",
                   help="root of the text-only divergence maps, one dir per combo")
    p.add_argument("-o", "--out", type=Path, default=REPO / "evaluation" / "csv" / "r44_div_stats.csv")
    args = p.parse_args(argv)

    rows: List[Dict[str, object]] = []
    for combo in read_selection(args.selection):
        text = stats_row("text", combo, args.text_root / combo)
        vp = stats_row("vp", combo, vp_counterpart(combo))
        if text["n_tokens"] != vp["n_tokens"]:
            raise SystemExit(f"[r44_div_stats] {combo}: {text['n_tokens']} text tokens vs "
                             f"{vp['n_tokens']} VP tokens -- the two sides cover different clips")
        rows += [text, vp]
        for r in (text, vp):
            print(f"[r44_div_stats] {r}", flush=True)
        # Text-only vs VP: the shift this table exists to show.
        print(f"[r44_div_stats] {combo}: d_mean text/VP = {text['d_mean'] / vp['d_mean']:.3f}, "
              f"d_p90 text/VP = {text['d_p90'] / vp['d_p90']:.3f}, "
              f"sat_frac VP {vp['sat_frac']:.4f} -> text {text['sat_frac']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[r44_div_stats] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
