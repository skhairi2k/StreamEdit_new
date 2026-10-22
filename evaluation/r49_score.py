"""R49 scoring: per-host pareto figures, deltas against the equal-compute curves, paired CIs against the official baseline
and R48's dino_SD arm. Plain per-clip means over the 18 clips of r49_cases.json (never an ``_avg.csv``).

StreamEdit  curve      R39 global-rho sweep at N = 20 (= the adaptive 5 + 15 budget)
            official   stock StreamEdit: rho = 2, N = 15, VP (R36 r36_rho2_vp)
            references R38 dino_n5_tgt09, R48 dino_SD (base group)
            arm        R49 se_max                                   -> deltas vs the R39 curve
Wan-Edit    curves     native N = 50 (6 points + native_s0.978, base group), matched-55 (R48 baselines: N 69/79/104),
                       matched-59 (m59 group: N 75/85/111, only if its CSVs exist)
            official   stock Wan-Edit: N = 50, skip 15 (native_s0.92)
            references R48 dino_SD (base group)
            arms       R49 wan_max_s9 -> deltas vs matched-55;  wan_max_s5 -> vs matched-59 (vs the native curve, flagged
                       unmatched_nfe, until matched-59 exists)
Deltas: the arm's FiVE-Acc / CLIP-D minus the curve linearly interpolated at the arm's own bg LPIPS (never extrapolated:
``outside_range`` / ``nonmonotone`` / ``no_curve`` flags as R39/R48). Paired 95% bootstrap CIs (2000 resamples) over clips for
arm - official and arm - R48 dino_SD on FiVE-Acc, CLIP-D, bg LPIPS and bg PSNR.

Outputs evaluation/csv/r49_points.csv, evaluation/csv/r49_deltas.csv, evaluation/figures/r49_pareto/{se,wan}_{five_acc,clip_d_prompt}.{pdf,png}
usage: python evaluation/r49_score.py

--mode full --host se|wan: the R49 FULL STAGE, 419 pairs, ONE host per run (each track runs on its own platform: StreamEdit on the lab
L40S, Wan-Edit on Jean Zay H100; a table or figure never mixes the two). Reported on all 419 pairs and on the 401 pairs outside the 18-clip
set (scope column). Same maths as above (plain per-clip means, interpolated deltas at the arm's own bg LPIPS, paired 95% bootstrap CIs).
  se : equal-NFE curve R39 rho sweep, N=20; NATIVE sweep R36 rho sweep, N=15 (stock budget; its rho=2 point is the official baseline);
       reference R38 dino_n5_tgt09; arm se_max (r49_se_max_full) -> deltas vs R39 (matched) and vs the R36 sweep (context); pairs vs official, R38
  wan: NATIVE sweep N=50 (6 points + native_s0.978, official native_s0.92) and equal-NFE curve matched-59 (4 points); arm wan_max_s5
       -> deltas vs matched-59 (equal NFE) and vs the native sweep (context); pairs vs official and vs native_s0.978 (same floor, uniform)
  Three figures per host: FiVE-Acc, CLIP-D and CLIP-T (target image, file name *_clip_tgt) against background LPIPS, each with the native
  sweep AND the equal-NFE curve; the same three metrics get curve deltas and paired CIs.
  CLIP-D per unit: r49_clip_directional_{base,m59}_full_{unit}.csv (array eval tasks), r49_clip_directional_{wan,se}_full.csv.
  R48 reference arms are skipped. Outputs evaluation/csv/r49_full_{host}_points.csv, r49_full_{host}_deltas.csv (platform + scope columns),
  evaluation/figures/r49_full_pareto/{host}_{five_acc,clip_d_prompt}.{pdf,png} (419 pairs; the platform in the title).
usage: python evaluation/r49_score.py --mode full --host wan
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_score as R  # noqa: E402
import r49_units as U49  # noqa: E402

Y, Y2 = R.Y, R.Y2
PAIR_METRICS = ("five_acc", "clip_d_prompt", Y, Y2)
COL = {"curve": "#1f9d6b", "native": "#8c8c87", "m59": "#7a5195", "official": "#111111", "ref": "#8c8c87", "arm": "#2a78d6", "arm2": "#eb6834"}


def point(label: str, host: str, role: str, v) -> Dict[str, object]:
    p = R.mean_point(label, host, role, v)
    p["vals"] = v
    return p


def plot(host: str, xkey: str, curves, official, refs, arms, out: Path, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, pts, col in curves:
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: float(p[Y]))
        ax.plot([p[xkey] for p in pts], [p[Y] for p in pts], "-o", color=col, ms=4, lw=1.4, label=name, alpha=0.9)
        for p in pts:
            ax.annotate(p["label"], (p[xkey], p[Y]), fontsize=6, color=col, xytext=(3, 3), textcoords="offset points")
    # small markers: the exact position of the official point and of the arm must be readable against the curves
    ax.plot(official[xkey], official[Y], "*", color=COL["official"], ms=6, label=official["label"])
    for p in refs:
        ax.plot(p[xkey], p[Y], "s", color=COL["ref"], mfc="none", ms=5, mew=1.0, label=p["label"])
    for i, p in enumerate(arms):
        ax.plot(p[xkey], p[Y], "o" if i == 0 else "D", color=COL["arm"] if i == 0 else COL["arm2"], ms=7, mew=0.6, mec="white",
                label=p["label"])
    ax.set_xlabel(f"{R.X[xkey]}  → more edit")
    ax.set_ylabel("background LPIPS (unedited part)  ↓ better")
    ax.set_title(title or f"R49 subset (18 clips): {'StreamEdit' if host == 'se' else 'Wan-Edit'} — plain per-clip mean", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=130)
    plt.close(fig)


PLATFORM = {"se": "L40S", "wan": "H100"}
CLIP_T = "clip_similarity_target_image"
FIG_X_FULL = (*R.FIG_X, CLIP_T)                       # FiVE-Acc, CLIP-D, CLIP-T (target image): the three figures
PAIR_FULL = (*PAIR_METRICS, CLIP_T)
FIG_NAME = {"five_acc": "five_acc", "clip_d_prompt": "clip_d_prompt", CLIP_T: "clip_tgt"}


def clipd_file(group: str, base: str) -> str:
    """Full-mode CLIP-D file of a unit: per-unit for the array-evaluated groups, per-group otherwise (see r49_eval_all.sh)."""
    if group in ("base", "m59"):
        return f"r49_clip_directional_{group}_full_{base[:-len('_full')]}.csv"
    return f"r49_clip_directional_{group}_full.csv"


def main_full(a) -> int:
    """R49 full stage: one host, 419 pairs and the 401 outside the 18-clip set."""
    host = a.host
    csv_dir = Path(a.csv_dir)
    fig_dir = Path(a.fig_dir) if a.fig_dir else HERE / "figures" / "r49_full_pareto"
    plat = PLATFORM[host]
    scope_all = {(int(c["edit_type"]), c["video_name"]) for c in json.loads((HERE / "r7_anchor_manifest.json").read_text())}
    scope18 = {(int(c["edit_type"]), c["video_name"]) for c in json.loads((HERE / "r49_cases.json").read_text())}
    if len(scope_all) != 419 or not scope18 <= scope_all:
        raise SystemExit(f"[r49_score] r7_anchor_manifest.json has {len(scope_all)} pairs (expect 419) / the 18 clips are not all in it")
    scopes = [("419", scope_all), ("401", scope_all - scope18)]
    names = R.clip_names()
    print(f"[r49_score] FULL stage, host {host} ({plat}): scopes " + ", ".join(f"{k}={len(v)}" for k, v in scopes) + " (plain per-clip means)", flush=True)

    pts_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []
    figs = {}
    for tag, scope in scopes:
        def uv(stem, fstem, clipd, method):
            return R.unit_values(csv_dir, stem, fstem, csv_dir / clipd, method, scope, names)

        def group_points(group: str, role: str) -> Dict[str, Dict[str, object]]:
            return {cfg: point(cfg, h, role, uv(f"r49_{base}", f"r49_fiveacc_{base}", clipd_file(group, base), f"r49_{base}"))
                    for base, _d, h, cfg in U49.units(group, "full") if h == host}

        if host == "se":
            arm = dict(point("R49 max (inverse response)", "se", "r49", uv("r49_se_max_full", "r49_fiveacc_se_max_full",
                                                                           clipd_file("se", "se_max_full"), "r49_se_max_full")))
            curve = [point(f"rho={rho}", "se", "curve", uv(f"r39_n20_rho{rho}_vp", f"r39_fiveacc_n20_rho{rho}_vp",
                                                           "r39_clip_directional.csv", f"r39_n20_rho{rho}_vp")) for rho in R.RHOS]
            native_sweep = [point(f"rho={rho}", "se", "native", uv(f"r36_rho{rho}_vp", f"r36_fiveacc_rho{rho}_vp",
                                                                   "r36_clip_directional.csv", f"r36_rho{rho}_vp")) for rho in R.RHOS]
            official = point("official: StreamEdit rho=2, N=15", "se", "official",
                             uv("r36_rho2_vp", "r36_fiveacc_rho2_vp", "r36_clip_directional.csv", "r36_rho2_vp"))
            r38 = point("R38 dino τ0.9 (budget-linear, no S)", "se", "ref",
                        uv("r38_dino_n5_tgt09", "r38_fiveacc_dino_n5_tgt09", "r38_clip_directional.csv", "r38_dino_n5_tgt09"))
            refs = [("R39 N=20 (equal NFE)", curve, "matched"), ("R36 N=15 (native sweep)", native_sweep, "unmatched_nfe")]
            others = [official, r38]
            fig_curves = [("native sweep: R36 rho, N=15", native_sweep, COL["native"]), ("equal NFE: R39 rho, N=20", curve, COL["curve"])]
            fig_refs = [r38]
            allpts = [arm, *curve, *native_sweep, official, r38]
        else:
            base_pts = group_points("base", "baseline")
            m59_pts = group_points("m59", "baseline")
            arm = point("R49 max, s_min 5 (59 steps)", "wan", "r49", uv("r49_wan_max_s5_full", "r49_fiveacc_wan_max_s5_full",
                                                                       clipd_file("wan", "wan_max_s5_full"), "r49_wan_max_s5_full"))
            native = sorted(base_pts.values(), key=lambda p: -float(str(p["label"]).split("_s")[1]))   # start sigma high .. low
            matched59 = sorted(m59_pts.values(), key=lambda p: -float(str(p["label"]).split("_s")[1]))
            official = dict(base_pts["native_s0.92"], label="official: Wan-Edit N=50, skip 15", role="official")
            n978 = dict(base_pts["native_s0.978"], label="native_s0.978 (uniform start at the floor)")
            refs = [("matched-59 (equal NFE)", matched59, "matched"), ("native N=50 (native sweep)", native, "unmatched_nfe")]
            others = [official, n978]
            fig_curves = [("native sweep: N=50, global start", native, COL["native"]), ("equal NFE: matched-59", matched59, COL["m59"])]
            fig_refs = []
            allpts = [arm, *native, *matched59]
        for p in allpts:
            pts_rows.append({**{k: v for k, v in p.items() if k != "vals"}, "scope": tag, "platform": plat})

        print(f"\n=== R49 full, {host} ({plat}), scope {tag} ({len(scope)} pairs) ===")
        print(f"[{host}] {arm['label']}: bg LPIPS {arm[Y]:.4f}, bg PSNR {arm[Y2]:.2f}, FiVE-Acc {arm['five_acc']:.3f}, CLIP-D {arm['clip_d_prompt']:.4f}")
        for refname, curve_pts, kind in refs:
            for x in FIG_X_FULL:
                r, flag = R.interp_at(curve_pts, float(arm[Y]), x)
                d = arm[x] - r if not math.isnan(r) else math.nan
                delta_rows.append({"host": host, "platform": plat, "scope": tag, "arm": arm["label"], "metric": x, "reference": refname,
                                   "kind": kind, "value": arm[x], "ref_value": r, "delta": d, "ci_lo": math.nan, "ci_hi": math.nan, "flag": flag})
                print(f"   {x:14s} vs {refname} at own LPIPS: {'n/a' if math.isnan(d) else format(d, '+.4f')} ({flag}, {kind})")
        rng = np.random.default_rng(0)
        for other in others:
            for m in PAIR_FULL:
                dm, lo, hi = R.paired_ci(arm["vals"][m], other["vals"][m], rng)
                delta_rows.append({"host": host, "platform": plat, "scope": tag, "arm": arm["label"], "metric": m, "reference": other["label"],
                                   "kind": "paired", "value": arm[m], "ref_value": other[m], "delta": dm, "ci_lo": lo, "ci_hi": hi, "flag": "ok"})
                print(f"   {m:22s} - {other['label']}: {dm:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        if tag == "419":
            figs = {"curves": fig_curves, "official": official, "refs": fig_refs, "arm": arm}

    cols = ["host", "platform", "scope", "role", "label", "n_clips", Y, Y2, "clip_similarity_target_image", "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_p = csv_dir / f"r49_full_{host}_points.csv"
    with out_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(pts_rows)
    out_d = csv_dir / f"r49_full_{host}_deltas.csv"
    with out_d.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(delta_rows[0]))
        w.writeheader()
        w.writerows(delta_rows)
    print(f"\n[r49_score] wrote {out_p} ({len(pts_rows)} points), {out_d} ({len(delta_rows)} rows)")
    for xkey in FIG_X_FULL:
        plot(host, xkey, figs["curves"], figs["official"], figs["refs"], [figs["arm"]], fig_dir / f"{host}_{FIG_NAME[xkey]}.pdf",
             title=f"R49 full (419 pairs, {plat}): {'StreamEdit' if host == 'se' else 'Wan-Edit'} — plain per-clip mean")
    print(f"[r49_score] figures -> {fig_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=None, help="default: figures/r49_pareto (subset) or figures/r49_full_pareto (full)")
    ap.add_argument("--cases", default=str(HERE / "r49_cases.json"))
    ap.add_argument("--mode", choices=("subset", "full"), default="subset")
    ap.add_argument("--host", choices=("se", "wan"), default=None, help="full mode only: one host per run (one platform each)")
    a = ap.parse_args(argv)
    if a.mode == "full":
        if a.host is None:
            ap.error("--mode full needs --host se|wan (each track is scored on its own platform)")
        return main_full(a)
    if a.fig_dir is None:
        a.fig_dir = str(HERE / "figures" / "r49_pareto")
    csv_dir, fig_dir = Path(a.csv_dir), Path(a.fig_dir)
    scope = {(int(c["edit_type"]), c["video_name"]) for c in json.loads(Path(a.cases).read_text())}
    names = R.clip_names()
    print(f"[r49_score] {len(scope)} clips (plain per-clip means)", flush=True)

    def uv(h, f, clipd_file, method):
        return R.unit_values(csv_dir, h, f, csv_dir / clipd_file, method, scope, names)

    def group_points(group: str, role: str) -> Dict[tuple, Dict[str, object]]:
        clipd = f"r49_clip_directional_{group}.csv"
        return {(host, cfg): point(cfg, host, role, uv(f"r49_{base}", f"r49_fiveacc_{base}", clipd, f"r49_{base}"))
                for base, _d, host, cfg in U49.units(group)}

    main_pts = group_points("main", "r49")                  # required
    base_pts = group_points("base", "baseline")             # 9 Wan baselines, native_s0.978, R48 dino_SD on both hosts
    m59_pts: Dict[tuple, Dict[str, object]] = {}
    if (csv_dir / "r49_clip_directional_m59.csv").is_file():
        m59_pts = group_points("m59", "baseline")
    else:
        print("[r49_score] matched-59 CSVs not there yet -- wan_max_s5 is compared to the native curve (unmatched_nfe)")

    # ---- StreamEdit references
    se_curve = [point(f"rho={rho}", "se", "curve", uv(f"r39_n20_rho{rho}_vp", f"r39_fiveacc_n20_rho{rho}_vp",
                                                      "r39_clip_directional.csv", f"r39_n20_rho{rho}_vp")) for rho in R.RHOS]
    se_official = point("official: StreamEdit rho=2, N=15", "se", "official",
                        uv("r36_rho2_vp", "r36_fiveacc_rho2_vp", "r36_clip_directional.csv", "r36_rho2_vp"))
    se_r38 = point("R38 dino τ0.9 (budget-linear, no S)", "se", "ref",
                   uv("r38_dino_n5_tgt09", "r38_fiveacc_dino_n5_tgt09", "r38_clip_directional.csv", "r38_dino_n5_tgt09"))
    se_r48 = dict(base_pts[("se", "r48_dino_SD")], label="R48 dino_SD (linear rho)", role="ref")

    # ---- Wan-Edit curves and references
    wan_base = {cfg: p for (host, cfg), p in base_pts.items() if host == "wan" and cfg != "r48_dino_SD"}
    wan_r48 = dict(base_pts[("wan", "r48_dino_SD")], label="R48 dino_SD (linear sigma)", role="ref")
    wan_official = dict(wan_base["native_s0.92"], label="official: Wan-Edit N=50, skip 15", role="official")
    native = [p for c, p in wan_base.items() if c.startswith("native")]
    native.sort(key=lambda p: -float(str(p["label"]).split("_s")[1]))   # natural order = start sigma high .. low (interp_at's check)
    matched55 = [p for c, p in wan_base.items() if c.startswith("matched")]
    matched59 = [p for (_h, c), p in m59_pts.items()]

    arms = {"se": [dict(main_pts[("se", "max")], label="R49 max (inverse response)")],
            "wan": [dict(main_pts[("wan", "max_s9")], label="R49 max, s_min 9 (55 steps)"),
                    dict(main_pts[("wan", "max_s5")], label="R49 max, s_min 5 (59 steps)")]}
    eq_curve = {("se", 0): ("R39 N=20 (equal NFE)", se_curve, "matched"),
                ("wan", 0): ("matched-55", matched55, "matched"),
                ("wan", 1): ("matched-59", matched59, "matched") if matched59 else ("native N=50", native, "unmatched_nfe")}
    official = {"se": se_official, "wan": wan_official}
    r48ref = {"se": se_r48, "wan": wan_r48}

    # ---- points table
    cols = ["host", "role", "label", "n_clips", Y, Y2, "clip_similarity_target_image", "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    allpts = [*main_pts.values(), *se_curve, se_official, se_r38, se_r48, *wan_base.values(), *m59_pts.values(), wan_r48]
    out_p = csv_dir / "r49_points.csv"
    with out_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allpts)
    print(f"[r49_score] wrote {out_p} ({len(allpts)} points)")

    # ---- deltas vs curves + paired CIs vs official and R48 dino_SD
    rng = np.random.default_rng(0)
    rows: List[Dict[str, object]] = []
    print(f"\n=== R49 arms ({len(scope)} clips) ===")
    for host in ("se", "wan"):
        for i, p in enumerate(arms[host]):
            print(f"\n[{host}] {p['label']}: bg LPIPS {p[Y]:.4f}, bg PSNR {p[Y2]:.2f}, FiVE-Acc {p['five_acc']:.3f}, CLIP-D {p['clip_d_prompt']:.4f}")
            refs = [eq_curve[(host, i)]]
            if host == "wan" and refs[0][2] != "unmatched_nfe":     # native curve as context (once)
                refs.append(("native N=50", native, "unmatched_nfe"))
            for refname, curve, kind in refs:
                for x in R.FIG_X:
                    r, flag = R.interp_at(curve, float(p[Y]), x)
                    d = p[x] - r if not math.isnan(r) else math.nan
                    rows.append({"host": host, "arm": p["label"], "metric": x, "reference": refname, "kind": kind, "value": p[x],
                                 "ref_value": r, "delta": d, "ci_lo": math.nan, "ci_hi": math.nan, "flag": flag})
                    print(f"   {x:14s} vs {refname} at own LPIPS: {'n/a' if math.isnan(d) else format(d, '+.4f')} ({flag}, {kind})")
            for other in (official[host], r48ref[host]):
                for m in PAIR_METRICS:
                    dm, lo, hi = R.paired_ci(p["vals"][m], other["vals"][m], rng)
                    rows.append({"host": host, "arm": p["label"], "metric": m, "reference": other["label"], "kind": "paired",
                                 "value": p[m], "ref_value": other[m], "delta": dm, "ci_lo": lo, "ci_hi": hi, "flag": "ok"})
                    print(f"   {m:22s} - {other['label']}: {dm:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    out_d = csv_dir / "r49_deltas.csv"
    with out_d.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[r49_score] wrote {out_d} ({len(rows)} rows)")

    # ---- figures
    for xkey in R.FIG_X:
        plot("se", xkey, [("R39 N=20 (equal NFE)", se_curve, COL["curve"])], se_official, [se_r38, se_r48], arms["se"],
             fig_dir / f"se_{xkey}.pdf")
        plot("wan", xkey, [("native N=50", native, COL["native"]), ("matched-55", matched55, COL["curve"]),
                           ("matched-59", matched59, COL["m59"])], wan_official, [wan_r48], arms["wan"], fig_dir / f"wan_{xkey}.pdf")
    print(f"[r49_score] figures -> {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
