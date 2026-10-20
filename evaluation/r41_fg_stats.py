"""R41 FG statistics -- how much of each clip a binary cut marks as foreground (rho = 50),
and what source budget that spends next to the continuous arm at the same N.

TWO MODES
---------
``--precheck`` (gate-draft; needs no rho field, runs before DRAFT_N is set)
    From the chosen N's absolute-normalised fields ``m`` (``r35_div_abs`` at N=15,
    ``r38_div_abs`` at N=5), per combo and per cut theta: the FG fraction ``mean(m >= theta)``
    as a plain per-clip mean, overall and per edit type, and the share of clips with
    < 0.5% FG -- the clips where a binary arm collapses to rho = 2 everywhere. Also the mean
    of ``m``, which is the FG fraction a binary map would need to spend the continuous
    arm's source budget (budget is linear in m by construction of budget_linear).
    Writes ``evaluation/csv/r41_precheck_n{N}.csv``.

default (after rhomap; DRAFT_N must be set and equal --draft_n)
    Per arm, from ``r41_rho/n{N}/{arm}``: FG fraction ``mean(rho == tau_max)`` and the mean
    source budget ``mean_tokens A_disc(rho)`` on the 15-step stage-3 grid, next to the SAME
    combo's continuous budget from the stored continuous rho at the same N. A binary point
    that moves on the Pareto plane may simply spend a different budget; this table is what
    tells that apart from a better allocation.
    Writes ``evaluation/csv/r41_n{N}_fg_stats.csv``.

Plain per-clip means (each clip weight 1), never a mean of per-type means.

Examples
--------
    python evaluation/r41_fg_stats.py --precheck --draft_n 15
    python evaluation/r41_fg_stats.py --precheck --draft_n 5 --thetas 0.3 0.5 0.7 0.8
    python evaluation/r41_fg_stats.py --draft_n 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r41_config  # noqa: E402
from r30_b_map import t_next_schedule  # noqa: E402
from r31_rho_map import a_disc_vec  # noqa: E402

LOW_FG: float = 0.005                 # "< 0.5% of tokens" = the binary arm is ~all rho=2
STAGE3_STEP: int = 15                 # budget is spent on stage 3's grid, whatever N is
FLOW_SHIFT: float = 1.0
TAU_MAX: float = 50.0


def clip_files(root: Path) -> Dict[Tuple[int, str], Path]:
    """{(edit_type, video): npz} under root/edit{T}/; hard-fails unless exactly 419."""
    out = {(int(p.parent.name[4:]), p.stem): p for p in sorted(root.glob("edit*/*.npz"))}
    per_type = {t: sum(1 for (tt, _) in out if tt == t) for t in r41_config.EXPECT}
    if per_type != r41_config.EXPECT:
        raise SystemExit(f"[r41_fg_stats] {root}: per-type counts {per_type}, expected "
                         f"{r41_config.EXPECT}")
    return out


def summarise(per_clip: Dict[Tuple[int, str], Dict[str, float]],
              keys: Sequence[str]) -> List[Dict[str, object]]:
    """One row for all clips, then one per edit type: plain mean of each key over clips."""
    rows = []
    for scope, sel in [("all", lambda t: True)] + [(f"edit{t}", (lambda t0: lambda t: t == t0)(t))
                                                  for t in r41_config.EXPECT]:
        vals = [v for (t, _), v in per_clip.items() if sel(t)]
        row: Dict[str, object] = {"scope": scope, "n_clips": len(vals)}
        for k in keys:
            row[k] = float(np.mean([v[k] for v in vals]))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def precheck(cfg: dict, thetas: Sequence[float]) -> List[Dict[str, object]]:
    rows = []
    for combo in cfg["combos"]:
        src = cfg["src_name"][combo]
        files = clip_files(cfg["div_abs_root"] / src)
        m_by_clip = {k: np.load(f)["m"].astype(np.float64) for k, f in files.items()}
        for theta in thetas:
            per_clip = {}
            for k, m in m_by_clip.items():
                fg = float((m >= theta).mean())
                per_clip[k] = {"fg_frac": fg, "low_fg_clip": float(fg < LOW_FG),
                               "mean_m": float(m.mean())}
            for r in summarise(per_clip, ("fg_frac", "low_fg_clip", "mean_m")):
                rows.append({"draft_n": cfg["draft_n"], "combo": combo, "src_name": src,
                             "theta": theta, **r})
    return rows


def arm_stats(cfg: dict) -> List[Dict[str, object]]:
    t_next = t_next_schedule(STAGE3_STEP, FLOW_SHIFT)
    budget = lambda rho: float(a_disc_vec(rho.astype(np.float64), t_next).mean())  # noqa: E731

    cont_budget: Dict[str, Dict[Tuple[int, str], float]] = {}
    for combo in cfg["combos"]:
        files = clip_files(cfg["cont_rho_root"] / cfg["src_name"][combo])
        cont_budget[combo] = {k: budget(np.load(f)["rho"]) for k, f in files.items()}
        print(f"[r41_fg_stats] continuous {combo}: mean budget "
              f"{np.mean(list(cont_budget[combo].values())):.5f}", flush=True)

    rows = []
    for a in cfg["arms"]:
        arm, combo, theta = a["arm"], a["combo"], a["theta"]
        files = clip_files(cfg["rho_root"] / arm)
        per_clip = {}
        for k, f in files.items():
            rho = np.load(f)["rho"]
            fg = float((rho == TAU_MAX).mean())
            b = budget(rho)
            per_clip[k] = {"fg_frac": fg, "low_fg_clip": float(fg < LOW_FG),
                           "budget_binary": b, "budget_continuous": cont_budget[combo][k]}
        for r in summarise(per_clip, ("fg_frac", "low_fg_clip",
                                      "budget_binary", "budget_continuous")):
            r["budget_ratio"] = r["budget_binary"] / r["budget_continuous"]
            rows.append({"draft_n": cfg["draft_n"], "arm": arm, "combo": combo,
                         "theta": theta, **r})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft_n", type=int, required=True, choices=(5, 15))
    ap.add_argument("--precheck", action="store_true",
                    help="gate-draft mode: FG fractions from m, no rho needed, DRAFT_N may "
                         "be unset")
    ap.add_argument("--thetas", type=float, nargs="+",
                    help="--precheck only: cuts to report (default: r41_config.sh THETAS)")
    ap.add_argument("--csv_dir", type=Path, default=r41_config.REPO / "evaluation" / "csv")
    args = ap.parse_args(argv)

    cfg = r41_config.load(args.draft_n, allow_unset=args.precheck)
    if args.precheck:
        thetas = args.thetas or cfg["thetas"]
        rows = precheck(cfg, thetas)
        out = args.csv_dir / f"r41_precheck_n{args.draft_n}.csv"
        print(f"\n{'combo':<16} {'theta':>5} {'FG frac':>8} {'clips<0.5%FG':>13} {'mean m':>7}")
        for r in rows:
            if r["scope"] == "all":
                print(f"{r['combo']:<16} {r['theta']:>5} {r['fg_frac']:>8.4f} "
                      f"{r['low_fg_clip']:>13.2f} {r['mean_m']:>7.4f}")
    else:
        if args.thetas:
            ap.error("--thetas is a --precheck option; the arms' cuts come from r41_config.sh")
        rows = arm_stats(cfg)
        out = args.csv_dir / f"r41_n{args.draft_n}_fg_stats.csv"
        print(f"\n{'arm':<22} {'FG frac':>8} {'clips<0.5%FG':>13} {'budget bin':>11} "
              f"{'budget cont':>12} {'ratio':>6}")
        for r in rows:
            if r["scope"] == "all":
                print(f"{r['arm']:<22} {r['fg_frac']:>8.4f} {r['low_fg_clip']:>13.2f} "
                      f"{r['budget_binary']:>11.5f} {r['budget_continuous']:>12.5f} "
                      f"{r['budget_ratio']:>6.2f}")
    write_csv(out, rows)
    print(f"\n[r41_fg_stats] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
