"""R51 scoring: the demand-component ablation arms placed on R49's trade-off curves, plus FiVE motion fidelity.

Plain per-clip means (never an ``_avg.csv``), reported on all 419 pairs and on the 401 pairs outside the 18-clip set of
r49_cases.json (scope column). Figures use the 419 pairs. x = alignment (FiVE-Acc, CLIP-D, CLIP-T), y = background LPIPS.
No paired CIs (R51 reads positions on the curves, not deltas). One host per run: each track is scored on its own platform and a
table or figure never mixes the two.

--host se  (StreamEdit, lab L40S)
    arms     A0 R49 se_max (r49_se_max_full) and A1 nodiv, A2 dino, A3 lpips, A4 nosupp (r51_se_{arm}_full)
    curves   equal NFE: R39 rho sweep N=20;  native: R36 rho sweep N=15 (its rho=2 point is the official baseline)
    motion   motion_fidelity_score and motion_fidelity_score_edit_part for A0..A4 and the official rho=2 render, on 419 / 401 pairs,
             overall and per edit type. NaN cells (empty edit mask, ~8 over the bench) are skipped in the means and counted.
--host wan (Wan-Edit, Jean Zay H100; needs the fetched R49 full-stage CSVs of the base and m59 groups and of wan_max_s5, and R51's
            wan_nosupp_s5)
    arms     A0 R49 wan_max_s5 and A4 wan_nosupp_s5
    curves   equal NFE: matched-59 (4 points);  native: N=50 global start (6 points + native_s0.978); official = native_s0.92

Outputs evaluation/csv/r51_points_{host}.csv, evaluation/csv/r51_motion.csv (se),
        evaluation/figures/r51_pareto/{host}_{five_acc,clip_d,clip_tgt}.{pdf,png}
usage: python evaluation/r51_score.py --host se
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_score as R  # noqa: E402
import r49_score as S49  # noqa: E402
import r49_units as U49  # noqa: E402
import r51_units as U51  # noqa: E402

Y, Y2 = R.Y, R.Y2
CLIP_T = S49.CLIP_T
FIG_X = (*R.FIG_X, CLIP_T)                                   # FiVE-Acc, CLIP-D, CLIP-T
FIG_NAME = {"five_acc": "five_acc", "clip_d_prompt": "clip_d", CLIP_T: "clip_tgt"}
MOTION = ("motion_fidelity_score", "motion_fidelity_score_edit_part")
PLATFORM = S49.PLATFORM
SE_ARMS = {"nodiv": "A1 no divergence (q=S)", "dino": "A2 DINO only", "lpips": "A3 LPIPS only", "nosupp": "A4 no support"}
MOTION_LABEL = {"se_nodiv_full": "A1 no divergence", "se_dino_full": "A2 DINO only", "se_lpips_full": "A3 LPIPS only",
                "se_nosupp_full": "A4 no support", "se_max_full": "A0 full (R49 se_max)", "se_r36_rho2_vp_full": "official rho=2, N=15"}
ARM_COLORS = ["#2a78d6", "#eb6834", "#b7418d", "#c9a227", "#17a2b8"]    # A0, A1.. in order


def scopes_and_names():
    manifest = {(int(c["edit_type"]), c["video_name"]) for c in json.loads((HERE / "r7_anchor_manifest.json").read_text())}
    scope18 = {(int(c["edit_type"]), c["video_name"]) for c in json.loads((HERE / "r49_cases.json").read_text())}
    if len(manifest) != 419 or not scope18 <= manifest:
        raise SystemExit(f"[r51_score] r7_anchor_manifest.json has {len(manifest)} pairs (expect 419) / the 18 clips are not all in it")
    return [("419", manifest), ("401", manifest - scope18)], R.clip_names()


def plot(host: str, xkey: str, curves, official, refs, arms, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, pts, col in curves:
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: float(p[Y]))
        ax.plot([p[xkey] for p in pts], [p[Y] for p in pts], "-o", color=col, ms=4, lw=1.4, label=name, alpha=0.9)
        for p in pts:
            ax.annotate(p["label"], (p[xkey], p[Y]), fontsize=6, color=col, xytext=(3, 3), textcoords="offset points")
    ax.plot(official[xkey], official[Y], "*", color=S49.COL["official"], ms=7, label=official["label"])
    for p in refs:
        ax.plot(p[xkey], p[Y], "s", color=S49.COL["ref"], mfc="none", ms=5, mew=1.0, label=p["label"])
    for i, p in enumerate(arms):
        ax.plot(p[xkey], p[Y], "o" if i == 0 else "D", color=ARM_COLORS[i % len(ARM_COLORS)], ms=7, mew=0.6, mec="white", label=p["label"])
    ax.set_xlabel(f"{R.X[xkey]}  → more edit")
    ax.set_ylabel("background LPIPS (unedited part)  ↓ better")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=130)
    plt.close(fig)


def r51_values(csv_dir: Path, base: str, scope, names):
    return R.unit_values(csv_dir, f"r51_{base}", f"r51_fiveacc_{base}", csv_dir / f"r51_clip_directional_{base}.csv", f"r51_{base}",
                         scope, names)


def se_references(csv_dir: Path, scope, names):
    """StreamEdit reference points shared with r52_score.py: A0 (R49 se_max), the equal-NFE R39 N=20 curve, the native R36 N=15 curve
    and the official rho=2 point, as (a0, equal_nfe, native, official, curves)."""
    def uv(stem, fstem, clipd, method):
        return R.unit_values(csv_dir, stem, fstem, csv_dir / clipd, method, scope, names)

    a0 = S49.point("A0 full (R49 se_max)", "se", "r49",
                   uv("r49_se_max_full", "r49_fiveacc_se_max_full", S49.clipd_file("se", "se_max_full"), "r49_se_max_full"))
    equal_nfe = [S49.point(f"rho={rho}", "se", "curve", uv(f"r39_n20_rho{rho}_vp", f"r39_fiveacc_n20_rho{rho}_vp",
                                                          "r39_clip_directional.csv", f"r39_n20_rho{rho}_vp")) for rho in R.RHOS]
    native = [S49.point(f"rho={rho}", "se", "native", uv(f"r36_rho{rho}_vp", f"r36_fiveacc_rho{rho}_vp",
                                                        "r36_clip_directional.csv", f"r36_rho{rho}_vp")) for rho in R.RHOS]
    official = S49.point("official: StreamEdit rho=2, N=15", "se", "official",
                         uv("r36_rho2_vp", "r36_fiveacc_rho2_vp", "r36_clip_directional.csv", "r36_rho2_vp"))
    curves = [("native sweep: R36 rho, N=15", native, S49.COL["native"]), ("equal NFE: R39 rho, N=20", equal_nfe, S49.COL["curve"])]
    return a0, equal_nfe, native, official, curves


def points_se(csv_dir: Path, scope, names):
    a0, equal_nfe, native, official, curves = se_references(csv_dir, scope, names)
    arms = [a0] + [S49.point(label, "se", "r51", r51_values(csv_dir, f"se_{arm}_full", scope, names)) for arm, label in SE_ARMS.items()]
    return arms, curves, official, [*equal_nfe, *native, official]


def points_wan(csv_dir: Path, scope, names):
    def uv(stem, fstem, clipd, method):
        return R.unit_values(csv_dir, stem, fstem, csv_dir / clipd, method, scope, names)

    def group_points(group: str, role: str) -> Dict[str, Dict[str, object]]:
        return {cfg: S49.point(cfg, h, role, uv(f"r49_{base}", f"r49_fiveacc_{base}", S49.clipd_file(group, base), f"r49_{base}"))
                for base, _d, h, cfg in U49.units(group, "full") if h == "wan"}

    base_pts, m59_pts = group_points("base", "baseline"), group_points("m59", "baseline")
    a0 = S49.point("A0 full (R49 max, s_min 5)", "wan", "r49",
                   uv("r49_wan_max_s5_full", "r49_fiveacc_wan_max_s5_full", S49.clipd_file("wan", "wan_max_s5_full"), "r49_wan_max_s5_full"))
    a4 = S49.point("A4 no support (s_min 5)", "wan", "r51", r51_values(csv_dir, "wan_nosupp_s5_full", scope, names))
    native = sorted(base_pts.values(), key=lambda p: -float(str(p["label"]).split("_s")[1]))
    matched59 = sorted(m59_pts.values(), key=lambda p: -float(str(p["label"]).split("_s")[1]))
    official = dict(base_pts["native_s0.92"], label="official: Wan-Edit N=50, skip 15", role="official")
    curves = [("native sweep: N=50, global start", native, S49.COL["native"]), ("equal NFE: matched-59", matched59, S49.COL["m59"])]
    return [a0, a4], curves, official, [*native, *matched59]


def motion_items() -> List[Tuple[str, str, str]]:
    """(unit, label, CSV stem) of R51's motion units."""
    return [(base, MOTION_LABEL[base], f"r51_motion_{base}") for base, _d, _h, _cfg in U51.units("motion")]


def motion_table(csv_dir: Path, scopes, names, items: Optional[List[Tuple[str, str, str]]] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for base, label, stem in (items or motion_items()):
        for tag, scope in scopes:
            v = R.load_harness(csv_dir, stem, MOTION, scope, names)
            for m in MOTION:
                if set(v[m]) != scope:
                    raise SystemExit(f"[r51_score] {stem}: '{m}' covers {len(v[m])} of the {len(scope)} clips in scope {tag}")
            for etype in ["all", *range(1, 7)]:
                keys = sorted(k for k in scope if etype == "all" or k[0] == etype)      # sorted: a set iterates in hash order, which changed the last float digit between runs
                if not keys:
                    continue
                row = {"unit": base, "label": label, "scope": tag, "edit_type": etype, "n_clips": len(keys)}
                for m in MOTION:
                    vals = np.array([v[m][k] for k in keys], dtype=float)
                    ok = np.isfinite(vals)
                    row[m] = float(vals[ok].mean()) if ok.any() else math.nan
                    row[f"{m}_n_nan"] = int((~ok).sum())
                rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], cols: Optional[List[str]] = None) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, choices=("se", "wan"))
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures" / "r51_pareto"))
    a = ap.parse_args(argv)
    csv_dir, fig_dir, host = Path(a.csv_dir), Path(a.fig_dir), a.host
    plat = PLATFORM[host]
    scopes, names = scopes_and_names()
    print(f"[r51_score] host {host} ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)

    pts_rows: List[Dict[str, object]] = []
    figs = None
    for tag, scope in scopes:
        arms, curves, official, others = (points_se if host == "se" else points_wan)(csv_dir, scope, names)
        for p in [*arms, *others]:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})
        print(f"\n=== R51, {host} ({plat}), scope {tag} ({len(scope)} pairs) ===")
        for p in [*arms, official]:
            print(f"[{host}] {p['label']:34s} bg LPIPS {p[Y]:.4f}  bg PSNR {p[Y2]:.2f}  FiVE-Acc {p['five_acc']:.3f}  "
                  f"CLIP-D {p['clip_d_prompt']:.4f}  CLIP-T {p[CLIP_T]:.2f}")
        if tag == "419":
            figs = (curves, official, arms)
    cols = ["host", "platform", "scope", "role", "label", "n_clips", Y, Y2, CLIP_T, "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / f"r51_points_{host}.csv"
    write_csv(out_p, pts_rows, cols)
    print(f"\n[r51_score] wrote {out_p} ({len(pts_rows)} points)")

    curves, official, arms = figs
    for xkey in FIG_X:
        plot(host, xkey, curves, official, [], arms, fig_dir / f"{host}_{FIG_NAME[xkey]}.pdf",
             f"R51 (419 pairs, {plat}): {'StreamEdit' if host == 'se' else 'Wan-Edit'} demand ablation — plain per-clip mean")
    print(f"[r51_score] figures -> {fig_dir}")

    if host == "se":
        mrows = motion_table(csv_dir, scopes, names)
        out_m = csv_dir / "r51_motion.csv"
        write_csv(out_m, mrows)
        print(f"[r51_score] wrote {out_m} ({len(mrows)} rows)")
        for r in mrows:
            if r["scope"] == "419" and r["edit_type"] == "all":
                print(f"   {r['label']:26s} MFS {r[MOTION[0]]:.4f} (NaN {r[MOTION[0] + '_n_nan']})  "
                      f"MFS-edit {r[MOTION[1]]:.4f} (NaN {r[MOTION[1] + '_n_nan']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
