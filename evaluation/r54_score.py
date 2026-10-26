"""R54 scoring: the draft-free arm (R40's flow-warped anchor maps, max of the two capped maps, R49's exposure inverse) against its draft-based
counterpart A4 and the rest of the StreamEdit references. Supplementary: StreamEdit + VP only (a text-only run has no anchor to warp, and Wan-Edit
drafts the whole clip at once).

Plain per-clip means (never an ``_avg.csv``), on all 419 pairs and on the 401 pairs outside the 18-clip set of r49_cases.json (scope column).
Figures use the 419 pairs: x = alignment (FiVE-Acc, CLIP-D, CLIP-T), y = background LPIPS, with the equal-NFE R39 N=20 curve and the native R36 N=15
curve. No paired CIs, no disocclusion split (user choice).

  R54  warp_max   q = max(capped warped DINO, capped warped LPIPS), no support, NO draft: NFE 15 -> read it on the R36 N=15 curve
  A4   nosupp     (R51) the same q measured per frame on the N=5 draft, no support: NFE 20 -> the pair R54 vs A4 isolates "per-frame draft vs
                  flow-warped anchor"
  A0   se_max     (R49) the full method (support S * max), NFE 20; official = stock rho=2, N=15
  motion          motion_fidelity_score and _edit_part of R54 (r54_motion units) and, from R51's motion units, A4, A0 and the official render

Inputs: R54's CSVs (r54_se_warp_max_full, r54_fiveacc_..., r54_clip_directional_..., r54_motion_...), R51's A4 and motion CSVs, R49's se_max CSVs and
the R36 / R39 curves. R51's A4 evaluation and motion jobs must have finished.
Outputs evaluation/csv/r54_points_se.csv, evaluation/csv/r54_motion.csv, evaluation/figures/r54_se_{five_acc,clip_d,clip_tgt}.{pdf,png}
usage: python evaluation/r54_score.py
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

BASE = "se_warp_max_full"
LABEL = "R54 draft-free warp_max (NFE 15)"
A4_LABEL = "A4 no support, per-frame draft (R51)"
A0_LABEL = "A0 full (R49 se_max)"
MOTION_ITEMS = [(BASE, "R54 warp_max", f"r54_motion_{BASE}"),
                ("se_nosupp_full", "A4 no support", "r51_motion_se_nosupp_full"),
                ("se_max_full", "A0 full (R49 se_max)", "r51_motion_se_max_full"),
                ("se_r36_rho2_vp_full", "official rho=2, N=15", "r51_motion_se_r36_rho2_vp_full")]
A4_FILES = ("r51_se_nosupp_full", "r51_fiveacc_se_nosupp_full", "r51_clip_directional_se_nosupp_full.csv")


def require_inputs(csv_dir: Path, scopes) -> None:
    """Fail with an R54-specific message when R51's A4 or any motion CSV is missing."""
    types = sorted({t for _tag, scope in scopes for t, _ in scope})
    missing = [f"edit{t}_FiVE_{stem}_frame_stride8.csv" for _b, _l, stem in MOTION_ITEMS for t in types
               if not (csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv").is_file()]
    missing += [f"edit{t}_FiVE_{stem}_frame_stride8.csv" for stem in A4_FILES[:2] for t in types
                if not (csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv").is_file()]
    missing += [] if (csv_dir / A4_FILES[2]).is_file() else [A4_FILES[2]]
    if missing:
        raise SystemExit(f"[r54_score] {len(missing)} input CSV(s) missing in {csv_dir}, e.g. {missing[0]} -- R54's come from r54_se_chain.sh "
                         f"(eval + motion), A4's and the A0 / official motion ones from R51's chain (r51_eval.sh se arm 3, r51_motion.sh arrays 3-5)")


def points(csv_dir: Path, scope, names):
    a0, equal_nfe, native, official, curves = S51.se_references(csv_dir, scope, names)
    a0 = dict(a0, label=A0_LABEL)
    a4 = S49.point(A4_LABEL, "se", "r51", S51.r51_values(csv_dir, "se_nosupp_full", scope, names))
    r54 = S49.point(LABEL, "se", "r54", R.unit_values(csv_dir, f"r54_{BASE}", f"r54_fiveacc_{BASE}", csv_dir / f"r54_clip_directional_{BASE}.csv",
                                                      f"r54_{BASE}", scope, names))
    return [r54, a4, a0], curves, official, [*equal_nfe, *native, official]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    a = ap.parse_args(argv)
    csv_dir, fig_dir = Path(a.csv_dir), Path(a.fig_dir)
    plat = S51.PLATFORM["se"]
    scopes, names = S51.scopes_and_names()
    print(f"[r54_score] StreamEdit + VP ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)
    require_inputs(csv_dir, scopes)

    pts_rows, figs = [], None
    for tag, scope in scopes:
        arms, curves, official, others = points(csv_dir, scope, names)
        for p in [*arms, *others]:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})
        print(f"\n=== R54, se ({plat}), scope {tag} ({len(scope)} pairs) ===")
        for p in [*arms, official]:
            print(f"[se] {p['label']:40s} bg LPIPS {p[S51.Y]:.4f}  bg PSNR {p[S51.Y2]:.2f}  FiVE-Acc {p['five_acc']:.3f}  "
                  f"CLIP-D {p['clip_d_prompt']:.4f}  CLIP-T {p[S51.CLIP_T]:.2f}")
        if tag == "419":
            figs = (curves, official, arms)
    cols = ["host", "platform", "scope", "role", "label", "n_clips", S51.Y, S51.Y2, S51.CLIP_T, "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / "r54_points_se.csv"
    S51.write_csv(out_p, pts_rows, cols)
    print(f"\n[r54_score] wrote {out_p} ({len(pts_rows)} points)")

    curves, official, arms = figs
    for xkey in S51.FIG_X:
        S51.plot("se", xkey, curves, official, [], arms, fig_dir / f"r54_se_{S51.FIG_NAME[xkey]}.pdf",
                 f"R54 (419 pairs, {plat}): draft-free flow-warped anchor vs per-frame draft — plain per-clip mean")
    print(f"[r54_score] figures -> {fig_dir}/r54_se_*.pdf")

    mrows = S51.motion_table(csv_dir, scopes, names, MOTION_ITEMS)
    out_m = csv_dir / "r54_motion.csv"
    S51.write_csv(out_m, mrows)
    print(f"[r54_score] wrote {out_m} ({len(mrows)} rows)")
    for r in mrows:
        if r["scope"] == "419" and r["edit_type"] == "all":
            print(f"   {r['label']:24s} MFS {r[S51.MOTION[0]]:.4f} (NaN {r[S51.MOTION[0] + '_n_nan']})  "
                  f"MFS-edit {r[S51.MOTION[1]]:.4f} (NaN {r[S51.MOTION[1] + '_n_nan']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
