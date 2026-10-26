"""R53 scoring: se_max WITHOUT visual prompting (text only) on StreamEdit, against the text-only uniform-rho curves.

Plain per-clip means (never an ``_avg.csv``), on all 419 pairs and on the 401 pairs outside the 18-clip set of r49_cases.json (scope column).
Figures use the 419 pairs: x = alignment (FiVE-Acc, CLIP-D, CLIP-T), y = background LPIPS. No paired CIs, no R44 routed arm, no VP-vs-text
demand statistics, no grids. StreamEdit only (lab L40S).

  arm        se_max_novp  (r53_se_max_novp_full: R49's procedure on a text-only draft; NFE 5 + 15 = 20)
  official   text-only stock StreamEdit: rho = 2, N = 15 (R44 r44_n15_rho2_novp)
  equal NFE  text-only uniform rho at N = 20: R44's rho 2, 4, 10, 50 (r44_n20_rho{R}_novp) + R53's rho 3, 6, 8, 20 (r53_n20_rho{R}_novp)
  native     text-only uniform rho at N = 15: R44's 8 rho (r44_n15_rho{R}_novp)
  motion     motion_fidelity_score and _edit_part of se_max_novp and the official text-only render (r53_motion units), overall and per
             edit type; NaN cells (empty edit mask) are skipped in the means and counted.

Inputs: R44's text-only CSVs (harness r44_n{N}_rho{R}_novp, FiVE-Acc r44_fiveacc_..., CLIP-D r44_clip_directional_sweep_n{N}.csv) and R53's
(r53_se_*_full, r53_fiveacc_se_*_full, r53_clip_directional_se_*_full.csv, r53_motion_se_*_full).
Outputs evaluation/csv/r53_points_se.csv, evaluation/csv/r53_motion.csv, evaluation/figures/r53_se_{five_acc,clip_d,clip_tgt}.{pdf,png}
usage: python evaluation/r53_score.py
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

ARM_BASE = "se_max_novp_full"
ARM_LABEL = "se_max, text only (R53)"
OFFICIAL_LABEL = "official text-only: rho=2, N=15"
R44_N20_RHOS = (2, 4, 10, 50)                     # R44's equal-NFE text-only sweep
R53_N20_RHOS = (3, 6, 8, 20)                      # R53's additions at N = 20 (r53_sweep group)
MOTION_ITEMS = [(ARM_BASE, "se_max text-only", f"r53_motion_{ARM_BASE}"),
                ("se_n15_rho2_novp_full", "official text-only rho=2, N=15", "r53_motion_se_n15_rho2_novp_full")]


def require_motion_csvs(csv_dir: Path, scopes) -> None:
    types = sorted({t for _tag, scope in scopes for t, _ in scope})
    missing = [f"edit{t}_FiVE_{stem}_frame_stride8.csv" for _b, _l, stem in MOTION_ITEMS for t in types
               if not (csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv").is_file()]
    if missing:
        raise SystemExit(f"[r53_score] {len(missing)} motion CSV(s) missing in {csv_dir}, e.g. {missing[0]} -- they come from "
                         f"r51_motion.sh r53_motion (array 0 = se_max_novp, array 1 = the official render on node39)")


def points(csv_dir: Path, scope, names):
    def uv(stem, fstem, clipd, method):
        return R.unit_values(csv_dir, stem, fstem, csv_dir / clipd, method, scope, names)

    def r44(n: int, rho: int, role: str):
        return S49.point(f"rho={rho}", "se", role, uv(f"r44_n{n}_rho{rho}_novp", f"r44_fiveacc_n{n}_rho{rho}_novp",
                                                       f"r44_clip_directional_sweep_n{n}.csv", f"r44_n{n}_rho{rho}_novp"))

    def r53(rho: int):
        base = f"se_n20_rho{rho}_novp_full"
        return S49.point(f"rho={rho}", "se", "curve", uv(f"r53_{base}", f"r53_fiveacc_{base}", f"r53_clip_directional_{base}.csv", f"r53_{base}"))

    arm = S49.point(ARM_LABEL, "se", "r53", uv(f"r53_{ARM_BASE}", f"r53_fiveacc_{ARM_BASE}", f"r53_clip_directional_{ARM_BASE}.csv", f"r53_{ARM_BASE}"))
    equal_nfe = sorted([*(r44(20, rho, "curve") for rho in R44_N20_RHOS), *(r53(rho) for rho in R53_N20_RHOS)],
                       key=lambda p: int(str(p["label"]).split("=")[1]))
    native = [r44(15, rho, "native") for rho in R.RHOS]
    official = dict(next(p for p in native if p["label"] == "rho=2"), label=OFFICIAL_LABEL, role="official")
    curves = [("native sweep: text-only rho, N=15", native, S49.COL["native"]), ("equal NFE: text-only rho, N=20", equal_nfe, S49.COL["curve"])]
    return [arm], curves, official, [*equal_nfe, *native]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    a = ap.parse_args(argv)
    csv_dir, fig_dir = Path(a.csv_dir), Path(a.fig_dir)
    plat = S51.PLATFORM["se"]
    scopes, names = S51.scopes_and_names()
    print(f"[r53_score] StreamEdit text-only ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)
    require_motion_csvs(csv_dir, scopes)

    pts_rows, figs = [], None
    for tag, scope in scopes:
        arms, curves, official, others = points(csv_dir, scope, names)
        for p in [*arms, *others]:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})
        print(f"\n=== R53, se text-only ({plat}), scope {tag} ({len(scope)} pairs) ===")
        for p in [*arms, official]:
            print(f"[se] {p['label']:34s} bg LPIPS {p[S51.Y]:.4f}  bg PSNR {p[S51.Y2]:.2f}  FiVE-Acc {p['five_acc']:.3f}  "
                  f"CLIP-D {p['clip_d_prompt']:.4f}  CLIP-T {p[S51.CLIP_T]:.2f}")
        if tag == "419":
            figs = (curves, official, arms)
    cols = ["host", "platform", "scope", "role", "label", "n_clips", S51.Y, S51.Y2, S51.CLIP_T, "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / "r53_points_se.csv"
    S51.write_csv(out_p, pts_rows, cols)
    print(f"\n[r53_score] wrote {out_p} ({len(pts_rows)} points)")

    curves, official, arms = figs
    for xkey in S51.FIG_X:
        S51.plot("se", xkey, curves, official, [], arms, fig_dir / f"r53_se_{S51.FIG_NAME[xkey]}.pdf",
                 f"R53 (419 pairs, {plat}): StreamEdit TEXT ONLY — plain per-clip mean")
    print(f"[r53_score] figures -> {fig_dir}/r53_se_*.pdf")

    mrows = S51.motion_table(csv_dir, scopes, names, MOTION_ITEMS)
    out_m = csv_dir / "r53_motion.csv"
    S51.write_csv(out_m, mrows)
    print(f"[r53_score] wrote {out_m} ({len(mrows)} rows)")
    for r in mrows:
        if r["scope"] == "419" and r["edit_type"] == "all":
            print(f"   {r['label']:32s} MFS {r[S51.MOTION[0]]:.4f} (NaN {r[S51.MOTION[0] + '_n_nan']})  "
                  f"MFS-edit {r[S51.MOTION[1]]:.4f} (NaN {r[S51.MOTION[1] + '_n_nan']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
