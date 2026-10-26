"""R55 scoring: R49's StreamEdit procedure (se_max) on the LongLive backbone, with VP, against LongLive's own uniform-rho curves from R45.

Plain per-clip means (never an ``_avg.csv``), on all 419 pairs and on the 401 pairs outside the 18-clip set of r49_cases.json (scope column).
Figures use the 419 pairs: x = alignment (FiVE-Acc, CLIP-D, CLIP-T), y = background LPIPS. No paired CIs, no grids. StreamEdit family, lab L40S.
Optional (P2) robustness check, not a new host: LongLive is also distilled from Wan2.1-T2V-1.3B.

  arm        se_max_ll   (r55_se_max_ll_full: R49's procedure on a LongLive draft, VP, fg_boost 2; NFE 5 + 15 = 20)
  official   stock LongLive: rho = 2, N = 15 (R45 r45_ll_rho2_n15_vp)
  equal NFE  LongLive uniform rho at N = 20 (R45's sweepM, matched to 5 + 15): r45_ll_rho{R}_n20_vp, R in 2, 3, 4, 6, 8, 10, 20, 50
  native     LongLive uniform rho at N = 15: r45_ll_rho{R}_n15_vp
  motion     motion_fidelity_score and _edit_part of se_max_ll and the official LongLive render (r55_motion units), overall and per edit type;
             NaN cells (empty edit mask) are skipped in the means and counted.

Inputs: R45's LongLive sweep CSVs -- they must have been scored under R45 first (r45_eval.sh, r45_fiveacc.sh, r45_clipd.sh): harness
r45_ll_rho{R}_n{N}_vp, FiVE-Acc r45_fiveacc_ll_rho{R}_n{N}_vp, CLIP-D r45_clip_directional.csv -- and R55's (r55_se_max_ll_full, r55_fiveacc_...,
r55_clip_directional_se_max_ll_full.csv, r55_motion_se_*).
Outputs evaluation/csv/r55_points_ll.csv, evaluation/csv/r55_motion.csv, evaluation/figures/r55_ll_{five_acc,clip_d,clip_tgt}.{pdf,png}
usage: python evaluation/r55_score.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_score as R  # noqa: E402
import r49_score as S49  # noqa: E402
import r51_score as S51  # noqa: E402

ARM_BASE = "se_max_ll_full"
ARM_LABEL = "se_max on LongLive (R55)"
OFFICIAL_LABEL = "official LongLive: rho=2, N=15"
R45_RHOS = (2, 3, 4, 6, 8, 10, 20, 50)
R45_CLIPD = "r45_clip_directional.csv"
MOTION_ITEMS = [(ARM_BASE, "se_max LongLive", f"r55_motion_{ARM_BASE}"),
                ("se_ll_rho2_n15_vp_full", "official LongLive rho=2, N=15", "r55_motion_se_ll_rho2_n15_vp_full")]


def r45_stem(n: int, rho: int) -> str:
    return f"ll_rho{rho}_n{n}_vp"


def require_inputs(csv_dir: Path, scopes) -> None:
    """Fail with one R45/R55-specific message listing what is missing (R45's sweeps must be scored first)."""
    types = sorted({t for _tag, scope in scopes for t, _ in scope})
    need_r45 = [f"edit{t}_FiVE_{pre}_{r45_stem(n, rho)}_frame_stride8.csv" for n in (15, 20) for rho in R45_RHOS for t in types
                for pre in ("r45", "r45_fiveacc")]
    miss_r45 = [f for f in need_r45 if not (csv_dir / f).is_file()]
    clipd = csv_dir / R45_CLIPD
    methods = set()
    if clipd.is_file():
        with clipd.open() as fh:
            methods = {r["method"] for r in csv.DictReader(fh)}
    miss_clipd = [f"r45_{r45_stem(n, rho)}" for n in (15, 20) for rho in R45_RHOS if f"r45_{r45_stem(n, rho)}" not in methods]
    need_r55 = [f"edit{t}_FiVE_{stem}_frame_stride8.csv" for _b, _l, stem in MOTION_ITEMS for t in types] + \
               [f"edit{t}_FiVE_r55_{p}{ARM_BASE}_frame_stride8.csv" for t in types for p in ("", "fiveacc_")]
    miss_r55 = [f for f in need_r55 if not (csv_dir / f).is_file()]
    if not (csv_dir / f"r55_clip_directional_{ARM_BASE}.csv").is_file():
        miss_r55.append(f"r55_clip_directional_{ARM_BASE}.csv")
    msgs = []
    if miss_r45 or miss_clipd:
        msgs.append(f"R45's LongLive sweeps are not fully scored ({len(miss_r45)} harness / FiVE-Acc CSV(s) and {len(miss_clipd)} CLIP-D method(s) missing, "
                    f"e.g. {(miss_r45 or miss_clipd)[0]}) -- run R45's eval, fiveacc and clipd steps first")
    if miss_r55:
        msgs.append(f"{len(miss_r55)} R55 CSV(s) missing, e.g. {miss_r55[0]} -- from r55_chain.sh (eval + motion jobs)")
    if msgs:
        raise SystemExit("[r55_score] " + " | ".join(msgs))


def points(csv_dir: Path, scope, names):
    def sweep(n: int, role: str):
        return [S49.point(f"rho={rho}", "se", role,
                          R.unit_values(csv_dir, f"r45_{r45_stem(n, rho)}", f"r45_fiveacc_{r45_stem(n, rho)}", csv_dir / R45_CLIPD,
                                        f"r45_{r45_stem(n, rho)}", scope, names)) for rho in R45_RHOS]

    equal_nfe, native = sweep(20, "curve"), sweep(15, "native")
    arm = S49.point(ARM_LABEL, "se", "r55", R.unit_values(csv_dir, f"r55_{ARM_BASE}", f"r55_fiveacc_{ARM_BASE}",
                                                          csv_dir / f"r55_clip_directional_{ARM_BASE}.csv", f"r55_{ARM_BASE}", scope, names))
    official = dict(next(p for p in native if p["label"] == "rho=2"), label=OFFICIAL_LABEL, role="official")
    curves = [("native sweep: LongLive rho, N=15", native, S49.COL["native"]), ("equal NFE: LongLive rho, N=20", equal_nfe, S49.COL["curve"])]
    return [arm], curves, official, [*equal_nfe, *native]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    a = ap.parse_args(argv)
    csv_dir, fig_dir = Path(a.csv_dir), Path(a.fig_dir)
    plat = S51.PLATFORM["se"]
    scopes, names = S51.scopes_and_names()
    print(f"[r55_score] StreamEdit on LongLive + VP ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)
    require_inputs(csv_dir, scopes)

    pts_rows, figs = [], None
    for tag, scope in scopes:
        arms, curves, official, others = points(csv_dir, scope, names)
        for p in [*arms, *others]:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})
        print(f"\n=== R55, LongLive ({plat}), scope {tag} ({len(scope)} pairs) ===")
        for p in [*arms, official]:
            print(f"[ll] {p['label']:34s} bg LPIPS {p[S51.Y]:.4f}  bg PSNR {p[S51.Y2]:.2f}  FiVE-Acc {p['five_acc']:.3f}  "
                  f"CLIP-D {p['clip_d_prompt']:.4f}  CLIP-T {p[S51.CLIP_T]:.2f}")
        if tag == "419":
            figs = (curves, official, arms)
    cols = ["host", "platform", "scope", "role", "label", "n_clips", S51.Y, S51.Y2, S51.CLIP_T, "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / "r55_points_ll.csv"
    S51.write_csv(out_p, pts_rows, cols)
    print(f"\n[r55_score] wrote {out_p} ({len(pts_rows)} points)")

    curves, official, arms = figs
    for xkey in S51.FIG_X:
        S51.plot("se", xkey, curves, official, [], arms, fig_dir / f"r55_ll_{S51.FIG_NAME[xkey]}.pdf",
                 f"R55 (419 pairs, {plat}): se_max on LongLive + VP — plain per-clip mean")
    print(f"[r55_score] figures -> {fig_dir}/r55_ll_*.pdf")

    mrows = S51.motion_table(csv_dir, scopes, names, MOTION_ITEMS)
    out_m = csv_dir / "r55_motion.csv"
    S51.write_csv(out_m, mrows)
    print(f"[r55_score] wrote {out_m} ({len(mrows)} rows)")
    for r in mrows:
        if r["scope"] == "419" and r["edit_type"] == "all":
            print(f"   {r['label']:32s} MFS {r[S51.MOTION[0]]:.4f} (NaN {r[S51.MOTION[0] + '_n_nan']})  "
                  f"MFS-edit {r[S51.MOTION[1]]:.4f} (NaN {r[S51.MOTION[1] + '_n_nan']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
