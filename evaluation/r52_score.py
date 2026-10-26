"""R52 scoring: B1 (rho = 2 + 48 q, linear in rho) against A0 (R49's exposure inverse, = B0) and the official baseline.

Same q = S * max(d_DINO, d_LPIPS) and same everything else; only the q -> rho map differs. Plain per-clip means (never an ``_avg.csv``), on all
419 pairs and on the 401 pairs outside the 18-clip set of r49_cases.json (scope column). Figures use the 419 pairs: x = alignment (FiVE-Acc,
CLIP-D, CLIP-T), y = background LPIPS, with the equal-NFE R39 N=20 curve and the native R36 N=15 curve. No CIs, no per-type table, no grids.
StreamEdit only (lab L40S): on Wan-Edit the exposure inverse already is linear in the native step knob, so there is no Wan-Edit arm.

Reads R52's B1 CSVs (r52_se_max_linrho_full, r52_fiveacc_..., r52_clip_directional_...), R49's se_max CSVs and the R36 / R39 curves, and the
motion CSVs: B1 from R52 (r52_motion_se_max_linrho_full), A0 and the official rho=2 render from R51's motion units (r51_motion_se_max_full,
r51_motion_se_r36_rho2_vp_full) -- R51's motion jobs must have finished.

Outputs evaluation/csv/r52_points_se.csv, evaluation/csv/r52_motion.csv, evaluation/figures/r52_se_{five_acc,clip_d,clip_tgt}.{pdf,png}
usage: python evaluation/r52_score.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_score as R  # noqa: E402
import r49_score as S49  # noqa: E402
import r51_score as S51  # noqa: E402

B1_BASE = "se_max_linrho_full"
B1_LABEL = "B1 linear rho = 2 + 48q"
A0_LABEL = "A0 = B0 (R49 inverse, se_max)"
MOTION_ITEMS = [(B1_BASE, "B1 linear rho", f"r52_motion_{B1_BASE}"),
                ("se_max_full", "A0 = B0 (R49 inverse)", "r51_motion_se_max_full"),
                ("se_r36_rho2_vp_full", "official rho=2, N=15", "r51_motion_se_r36_rho2_vp_full")]


def require_motion_csvs(csv_dir: Path, scopes) -> None:
    """Fail with an R52-specific message when a motion CSV is missing (A0 and official come from R51's motion jobs)."""
    types = sorted({t for _tag, scope in scopes for t, _ in scope})
    missing = [f"edit{t}_FiVE_{stem}_frame_stride8.csv" for _b, _l, stem in MOTION_ITEMS for t in types
               if not (csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv").is_file()]
    if missing:
        raise SystemExit(f"[r52_score] {len(missing)} motion CSV(s) missing in {csv_dir}, e.g. {missing[0]} -- B1's come from R52's motion job "
                         f"(r51_motion.sh r52_motion), A0 and the official render's from R51's (r51_motion.sh array 4 and 5)")


def points(csv_dir: Path, scope, names):
    a0, equal_nfe, native, official, curves = S51.se_references(csv_dir, scope, names)
    a0 = dict(a0, label=A0_LABEL)
    b1 = S49.point(B1_LABEL, "se", "r52", R.unit_values(csv_dir, f"r52_{B1_BASE}", f"r52_fiveacc_{B1_BASE}",
                                                       csv_dir / f"r52_clip_directional_{B1_BASE}.csv", f"r52_{B1_BASE}", scope, names))
    return [a0, b1], curves, official, [*equal_nfe, *native, official]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    a = ap.parse_args(argv)
    csv_dir, fig_dir = Path(a.csv_dir), Path(a.fig_dir)
    plat = S51.PLATFORM["se"]
    scopes, names = S51.scopes_and_names()
    print(f"[r52_score] StreamEdit ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)
    require_motion_csvs(csv_dir, scopes)

    pts_rows, figs = [], None
    for tag, scope in scopes:
        arms, curves, official, others = points(csv_dir, scope, names)
        for p in [*arms, *others]:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})
        print(f"\n=== R52, se ({plat}), scope {tag} ({len(scope)} pairs) ===")
        for p in [*arms, official]:
            print(f"[se] {p['label']:32s} bg LPIPS {p[S51.Y]:.4f}  bg PSNR {p[S51.Y2]:.2f}  FiVE-Acc {p['five_acc']:.3f}  "
                  f"CLIP-D {p['clip_d_prompt']:.4f}  CLIP-T {p[S51.CLIP_T]:.2f}")
        if tag == "419":
            figs = (curves, official, arms)
    cols = ["host", "platform", "scope", "role", "label", "n_clips", S51.Y, S51.Y2, S51.CLIP_T, "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / "r52_points_se.csv"
    S51.write_csv(out_p, pts_rows, cols)
    print(f"\n[r52_score] wrote {out_p} ({len(pts_rows)} points)")

    curves, official, arms = figs
    for xkey in S51.FIG_X:
        S51.plot("se", xkey, curves, official, [], arms, fig_dir / f"r52_se_{S51.FIG_NAME[xkey]}.pdf",
                 f"R52 (419 pairs, {plat}): StreamEdit mapping ablation — plain per-clip mean")
    print(f"[r52_score] figures -> {fig_dir}/r52_se_*.pdf")

    mrows = S51.motion_table(csv_dir, scopes, names, MOTION_ITEMS)
    out_m = csv_dir / "r52_motion.csv"
    S51.write_csv(out_m, mrows)
    print(f"[r52_score] wrote {out_m} ({len(mrows)} rows)")
    for r in mrows:
        if r["scope"] == "419" and r["edit_type"] == "all":
            print(f"   {r['label']:24s} MFS {r[S51.MOTION[0]]:.4f} (NaN {r[S51.MOTION[0] + '_n_nan']})  "
                  f"MFS-edit {r[S51.MOTION[1]]:.4f} (NaN {r[S51.MOTION[1] + '_n_nan']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
