"""R56 scoring: the draft-budget ablation -- se_max with a 15-step draft (sigma_d = 0.96, release from t = 0.933; 15 + 15 = 30 NFE) next to se_max
with the 5-step draft (R49, release from t = 0.8; 5 + 15 = 20 NFE), each read on the uniform-rho curve of ITS OWN compute (R39 N = 30 / N = 20).

Plain per-clip means (never an ``_avg.csv``) on all 419 pairs, on the 401 pairs outside the 18-clip set of r49_cases.json, and per edit type
(scope column: 419, 401, type1 .. type6; types 3-5 -- colour / material / add -- are the main readout). Figures use the 419 pairs: x = alignment
(FiVE-Acc, CLIP-D, CLIP-T), y = background LPIPS, with both matched-compute curves. No curve-gap interpolation per type, no CIs (user choice).
StreamEdit only (lab L40S), VP.

  arms      se_max N15 (r56_se_max_n15_full)          se_max N5 = R49 se_max full (r49_se_max_full)
  curves    R39 uniform rho, N = 30 (r39_n30_rho{R}_vp)   R39 uniform rho, N = 20 (r39_n20_rho{R}_vp); R in 2, 3, 4, 6, 8, 10, 20, 50
  official  stock StreamEdit: rho = 2, N = 15 (r36_rho2_vp)
  motion    motion_fidelity_score and _edit_part: N15 from R56 (r56_motion_se_max_n15_full), N5 and the official render from R51's motion units

Outputs evaluation/csv/r56_points_se.csv, evaluation/csv/r56_motion.csv, evaluation/figures/r56_se_{five_acc,clip_d,clip_tgt}.{pdf,png}
usage: python evaluation/r56_score.py
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

N15_BASE = "se_max_n15_full"
N15_LABEL = "se_max, 15-step draft (30 NFE)"
N5_LABEL = "se_max, 5-step draft (R49, 20 NFE)"
MOTION_ITEMS = [(N15_BASE, "se_max 15-step draft", f"r56_motion_{N15_BASE}"),
                ("se_max_full", "se_max 5-step draft (R49)", "r51_motion_se_max_full"),
                ("se_r36_rho2_vp_full", "official rho=2, N=15", "r51_motion_se_r36_rho2_vp_full")]


def require_inputs(csv_dir: Path, scopes) -> None:
    """One message naming what is missing: R56's own CSVs, or R51's motion units (N5 and official)."""
    types = sorted({t for _tag, scope in scopes for t, _ in scope})
    missing = [f"edit{t}_FiVE_{stem}_frame_stride8.csv" for _b, _l, stem in MOTION_ITEMS for t in types
               if not (csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv").is_file()]
    missing += [f"edit{t}_FiVE_r56_{p}{N15_BASE}_frame_stride8.csv" for p in ("", "fiveacc_") for t in types
                if not (csv_dir / f"edit{t}_FiVE_r56_{p}{N15_BASE}_frame_stride8.csv").is_file()]
    missing += [] if (csv_dir / f"r56_clip_directional_{N15_BASE}.csv").is_file() else [f"r56_clip_directional_{N15_BASE}.csv"]
    if missing:
        raise SystemExit(f"[r56_score] {len(missing)} input CSV(s) missing in {csv_dir}, e.g. {missing[0]} -- R56's come from r56_se_chain.sh "
                         f"(eval + motion); the 5-step arm's and the official render's motion from R51's chain (r51_motion.sh arrays 4 and 5)")


def all_scopes():
    """(419, 401) from r51_score, plus one scope per edit type (subsets of the 419 pairs)."""
    scopes, names = S51.scopes_and_names()
    full = dict(scopes)["419"]
    return scopes + [(f"type{t}", {k for k in full if k[0] == t}) for t in range(1, 7)], scopes, names


def points(csv_dir: Path, scope, names):
    def uv(stem, fstem, clipd, method):
        return R.unit_values(csv_dir, stem, fstem, csv_dir / clipd, method, scope, names)

    def sweep(n: int, role: str):
        return [S49.point(f"rho={rho}", "se", role, uv(f"r39_n{n}_rho{rho}_vp", f"r39_fiveacc_n{n}_rho{rho}_vp", "r39_clip_directional.csv",
                                                        f"r39_n{n}_rho{rho}_vp")) for rho in R.RHOS]

    n5 = S49.point(N5_LABEL, "se", "r49", uv("r49_se_max_full", "r49_fiveacc_se_max_full", S49.clipd_file("se", "se_max_full"), "r49_se_max_full"))
    n15 = S49.point(N15_LABEL, "se", "r56", uv(f"r56_{N15_BASE}", f"r56_fiveacc_{N15_BASE}", f"r56_clip_directional_{N15_BASE}.csv", f"r56_{N15_BASE}"))
    n20, n30 = sweep(20, "curve"), sweep(30, "curve30")
    official = S49.point("official: StreamEdit rho=2, N=15", "se", "official",
                         uv("r36_rho2_vp", "r36_fiveacc_rho2_vp", "r36_clip_directional.csv", "r36_rho2_vp"))
    curves = [("R39 uniform rho, N=20 (matches the 5-step draft)", n20, S49.COL["curve"]),
              ("R39 uniform rho, N=30 (matches the 15-step draft)", n30, S49.COL["m59"])]
    return [n5, n15], curves, official, [*n20, *n30, official]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    a = ap.parse_args(argv)
    csv_dir, fig_dir = Path(a.csv_dir), Path(a.fig_dir)
    plat = S51.PLATFORM["se"]
    scopes, base_scopes, names = all_scopes()
    print(f"[r56_score] StreamEdit + VP ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)
    require_inputs(csv_dir, base_scopes)

    pts_rows, figs = [], None
    for tag, scope in scopes:
        arms, curves, official, others = points(csv_dir, scope, names)
        for p in [*arms, *others]:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})
        if tag in ("419", "401") or tag in ("type3", "type4", "type5"):
            print(f"\n=== R56, se ({plat}), scope {tag} ({len(scope)} pairs) ===")
            for p in [*arms, official]:
                print(f"[se] {p['label']:36s} bg LPIPS {p[S51.Y]:.4f}  bg PSNR {p[S51.Y2]:.2f}  FiVE-Acc {p['five_acc']:.3f}  "
                      f"CLIP-D {p['clip_d_prompt']:.4f}  CLIP-T {p[S51.CLIP_T]:.2f}")
        if tag == "419":
            figs = (curves, official, arms)
    cols = ["host", "platform", "scope", "role", "label", "n_clips", S51.Y, S51.Y2, S51.CLIP_T, "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / "r56_points_se.csv"
    S51.write_csv(out_p, pts_rows, cols)
    print(f"\n[r56_score] wrote {out_p} ({len(pts_rows)} points: scopes 419, 401, type1..type6)")

    curves, official, arms = figs
    for xkey in S51.FIG_X:
        S51.plot("se", xkey, curves, official, [], arms, fig_dir / f"r56_se_{S51.FIG_NAME[xkey]}.pdf",
                 f"R56 (419 pairs, {plat}): draft budget at matched NFE — plain per-clip mean")
    print(f"[r56_score] figures -> {fig_dir}/r56_se_*.pdf")

    mrows = S51.motion_table(csv_dir, base_scopes, names, MOTION_ITEMS)
    out_m = csv_dir / "r56_motion.csv"
    S51.write_csv(out_m, mrows)
    print(f"[r56_score] wrote {out_m} ({len(mrows)} rows)")
    for r in mrows:
        if r["scope"] == "419" and r["edit_type"] == "all":
            print(f"   {r['label']:28s} MFS {r[S51.MOTION[0]]:.4f} (NaN {r[S51.MOTION[0] + '_n_nan']})  "
                  f"MFS-edit {r[S51.MOTION[1]]:.4f} (NaN {r[S51.MOTION[1] + '_n_nan']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
