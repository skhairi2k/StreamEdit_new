"""R48 scoring: per-host pareto figures, matched deltas against the host's equal-compute curve, and the gate table.

Everything is a PLAIN MEAN OVER THE CLIPS of the chosen scope (never an ``_avg.csv``): ``--mode subset`` scopes to
evaluation/r48_cases.json (18 clips), ``--mode full`` to the 419 benchmark pairs.

POINTS (one per unit of r48_units.py; stems: harness r48_{base}, FiVE-Acc r48_fiveacc_{base}, CLIP-D method r48_{base}):
  StreamEdit  arms dino_SD / lpips_SD (C4), dino_D (C2, no support), S (C3, no divergence)
  Wan-Edit    the same four arms, plus the global baselines: native curve (6 points, N=50) and matched curve (3 points,
              55 active steps)
CURVES / REFERENCES
  StreamEdit  R39's uniform-rho sweep at 20 NFE (r39_n20_rho{2..50}_vp, = the adaptive run's 5 + 15), filtered to the scope --
              existing per-clip CSVs, no re-render. R38 dino_n5_tgt09 (budget-linear rho, no support, DINO 0.65) is plotted as a
              reference point.
  Wan-Edit    the matched curve is the equal-compute reference; the native curve (cheaper, NFE varies) is drawn for context.

MATCHED DELTAS (r48_{mode}_deltas.csv): for every arm and x metric, the arm's value minus the host's matched curve linearly
interpolated at the arm's OWN bg LPIPS (Y_METRIC); never extrapolated: ``outside_range`` / ``nonmonotone`` / ``no_curve`` flags as
R39. Where the matched curve does not span the arm's LPIPS (likely for Wan-Edit: its 3 points are all early starts, the adaptive
arm preserves more), the same delta against the NATIVE curve is also written, labelled ``unmatched_nfe`` -- context, not the gate.

GATE (pre-registered, Decisions "Gate"), per host and per C4 arm (dino_SD, lpips_SD):
  G_curve  delta > 0 on FiVE-Acc AND on CLIP-D against the matched curve (not outside its range)
  G_S      C4 >= C2 (dino_D) on FiVE-Acc with bg LPIPS <= C2's          (S helps; C2 is DINO-only, so the check is DINO vs DINO)
  G_D      C4 >= C3 (S) on FiVE-Acc with bg LPIPS <= C3's               (D helps)
plus paired bootstrap 95% CIs over clips (2000 resamples) for the C4 - C2 and C4 - C3 differences, printed for the borderline
calls the final plan asks for. The script PRINTS the verdicts; the user writes the passing (host, arm) pairs to
evaluation/r48_selected_arms.txt ("se dino_SD" per line).

usage: python evaluation/r48_score.py --mode subset
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
import r48_units as U  # noqa: E402
import r48_wan_sched as SCH  # noqa: E402

FIVE = Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark"
Y = "lpips_unedit_part"                                   # preservation (lower = better)
Y2 = "psnr_unedit_part"
X = {"five_acc": "FiVE-Acc (mean of yes/no, MC, union, inter)", "clip_d_prompt": "CLIP-D (directional, prompt)",
     "five_acc_yes_no": "FiVE-Acc (yes/no only)", "clip_similarity_target_image": "CLIP-T (target image)"}
FIG_X = ("five_acc", "clip_d_prompt")
GATE_X = ("five_acc", "clip_d_prompt")
HARNESS = (Y, Y2, "clip_similarity_target_image")
FIVEACC = ("five_acc", "five_acc_yes_no")
RHOS = (2, 3, 4, 6, 8, 10, 20, 50)
N_BOOT = 2000
C4 = ("dino_SD", "lpips_SD")
COLORS = {"dino": "#2a78d6", "lpips": "#eb6834", "ref": "#8c8c87", "match": "#1f9d6b"}


# ------------------------------------------------------------------------------------------------- loading
def clip_names() -> Dict[Tuple[int, str], str]:
    out = {}
    for t in range(1, 7):
        for i, e in enumerate(json.loads((FIVE / "edit_prompt" / f"edit{t}_FiVE.json").read_text())):
            out[(t, str(i))] = e["video_name"]
    return out


def load_harness(csv_dir: Path, stem: str, metrics: Sequence[str], scope: set, names) -> Dict[str, Dict[Tuple[int, str], float]]:
    out: Dict[str, Dict[Tuple[int, str], float]] = {m: {} for m in metrics}
    for t in sorted({t for t, _ in scope}):
        path = csv_dir / f"edit{t}_FiVE_{stem}_frame_stride8.csv"
        if not path.is_file():
            raise SystemExit(f"[r48_score] missing {path} -- has its scoring step run?")
        rows = list(csv.DictReader(path.open()))
        for m in metrics:
            col = next((k for k in rows[0] if k == m or k.endswith(f"|{m}")), None)
            if col is None:
                raise SystemExit(f"[r48_score] {path.name}: no '{m}' column; have {list(rows[0])}")
            for r in rows:
                key = (t, names[(t, r["file_id"])])
                if key in scope:
                    out[m][key] = float(r[col])
    return out


def load_clipd(path: Path, method: str, scope: set) -> Dict[Tuple[int, str], float]:
    if not path.is_file():
        raise SystemExit(f"[r48_score] missing {path} -- has the clipd step run?")
    return {(int(r["edit_type"]), r["video_name"]): float(r["clip_d_prompt"])
            for r in csv.DictReader(path.open()) if r["method"] == method and (int(r["edit_type"]), r["video_name"]) in scope}


def unit_values(csv_dir: Path, harness_stem: str, fiveacc_stem: str, clipd_path: Path, clipd_method: str,
                scope: set, names) -> Dict[str, Dict[Tuple[int, str], float]]:
    v = load_harness(csv_dir, harness_stem, HARNESS, scope, names)
    v.update(load_harness(csv_dir, fiveacc_stem, FIVEACC, scope, names))
    v["clip_d_prompt"] = load_clipd(clipd_path, clipd_method, scope)
    for m, d in v.items():
        if set(d) != scope:
            raise SystemExit(f"[r48_score] {harness_stem}: '{m}' covers {len(d)} of the {len(scope)} clips in scope "
                             f"(missing {len(scope - set(d))})")
    return v


def mean_point(label: str, host: str, role: str, v) -> Dict[str, object]:
    return {"label": label, "host": host, "role": role, "n_clips": len(next(iter(v.values()))),
            **{m: float(np.mean(list(d.values()))) for m, d in v.items()}}


# --------------------------------------------------------------------------------------------- reference maths
def interp_at(curve: List[Dict[str, object]], lpips: float, x: str) -> Tuple[float, str]:
    """Metric x along a curve (given in its natural order), linearly interpolated at a given LPIPS; never extrapolated."""
    if len(curve) < 2:
        return math.nan, "no_curve"
    order = [float(p[Y]) for p in curve]
    flag = "ok" if all(a < b for a, b in zip(order, order[1:])) or all(a > b for a, b in zip(order, order[1:])) else "nonmonotone"
    pts = sorted((float(p[Y]), float(p[x])) for p in curve)
    ls, xs = zip(*pts)
    if not ls[0] <= lpips <= ls[-1]:
        return math.nan, "outside_range"
    return float(np.interp(lpips, ls, xs)), flag


def paired_ci(a: Dict, b: Dict, rng) -> Tuple[float, float, float]:
    """Mean paired difference a - b over clips with a 95% bootstrap CI."""
    keys = sorted(a)
    d = np.array([a[k] - b[k] for k in keys])
    boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ----------------------------------------------------------------------------------------------------- figures
def plot(host: str, xkey: str, arms, curves, refs, out: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, pts, col, ls in curves:
        if not pts:
            continue
        ax.plot([p[xkey] for p in pts], [p[Y] for p in pts], "-o", color=col, ms=4, lw=1.4, label=name, alpha=0.9)
        for p in pts:
            ax.annotate(p["label"], (p[xkey], p[Y]), fontsize=6, color=col, xytext=(3, 3), textcoords="offset points")
    for p in refs:
        ax.plot(p[xkey], p[Y], "s", color=COLORS["ref"], mfc="none", ms=7, label=p["label"])
    for p in arms:
        arm = p["label"]
        col = COLORS["lpips"] if arm.startswith("lpips") else COLORS["dino"]
        marker = "D" if arm == "S" else ("s" if arm.startswith("lpips") else "o")
        fill = {"dino_D": "full", "S": "none"}.get(arm, "left")
        ax.plot(p[xkey], p[Y], marker, color=col, fillstyle=fill, ms=9, mew=1.5, label=f"{arm} (C{'4' if arm in C4 else '2' if arm == 'dino_D' else '3'})")
    ax.set_xlabel(f"{X[xkey]}  → more edit")
    ax.set_ylabel("background LPIPS (unedited part)  ↓ better")
    ax.set_title(f"R48 {mode}: {'StreamEdit' if host == 'se' else 'Wan-Edit'} — plain per-clip mean", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=130)      # PNG twin for quick viewing
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("subset", "full"), required=True)
    ap.add_argument("--csv_dir", default=str(HERE / "csv"))
    ap.add_argument("--fig_dir", default=None)
    a = ap.parse_args(argv)
    mode, csv_dir = a.mode, Path(a.csv_dir)
    cases = HERE / ("r48_cases.json" if mode == "subset" else "r7_anchor_manifest.json")
    scope = {(int(c["edit_type"]), c["video_name"]) for c in json.loads(cases.read_text())}
    names = clip_names()
    fig_dir = Path(a.fig_dir) if a.fig_dir else HERE / "figures" / f"r48_{mode}_pareto"
    clipd = csv_dir / f"r48_clip_directional_{mode}.csv"
    print(f"[r48_score] mode={mode}: {len(scope)} clips (plain per-clip means)", flush=True)

    # ---- points
    pts: Dict[str, Dict[str, object]] = {}
    vals: Dict[str, Dict] = {}
    for base, _d, host, cfg in U.units(mode):
        v = unit_values(csv_dir, f"r48_{base}", f"r48_fiveacc_{base}", clipd, f"r48_{base}", scope, names)
        role = "baseline" if cfg.startswith(("native", "matched")) else "arm"
        label = cfg if role == "baseline" else cfg.replace("adapt_", "")
        pts[base] = mean_point(label, host, role, v)
        pts[base]["base"] = base
        vals[base] = v

    # ---- StreamEdit's existing references (R39 N=20 curve, R38 reference point), filtered to the scope
    r39 = []
    r39_clipd = csv_dir / "r39_clip_directional.csv"
    for rho in RHOS:
        v = unit_values(csv_dir, f"r39_n20_rho{rho}_vp", f"r39_fiveacc_n20_rho{rho}_vp", r39_clipd, f"r39_n20_rho{rho}_vp", scope, names)
        r39.append(mean_point(f"rho={rho}", "se", "curve", v))
    v38 = unit_values(csv_dir, "r38_dino_n5_tgt09", "r38_fiveacc_dino_n5_tgt09", csv_dir / "r38_clip_directional.csv",
                      "r38_dino_n5_tgt09", scope, names)
    r38 = mean_point("R38 dino τ0.9 (no S)", "se", "reference", v38)

    arms = {h: {p["label"]: p for p in pts.values() if p["host"] == h and p["role"] == "arm"} for h in ("se", "wan")}
    wan_native = [p for p in pts.values() if p["host"] == "wan" and p["label"].startswith("native")]
    wan_match = [p for p in pts.values() if p["host"] == "wan" and p["label"].startswith("matched")]
    ref = {"se": ("R39 N=20 (matched)", r39), "wan": ("matched (55 active steps)", wan_match)}
    ctx = {"se": [], "wan": [("native N=50 (unmatched NFE)", wan_native)]}

    # ---- table of every point
    cols = ["host", "role", "label", "n_clips", Y, Y2, "clip_similarity_target_image", "five_acc", "five_acc_yes_no", "clip_d_prompt"]
    out_csv = csv_dir / f"r48_{mode}.csv"
    allpts = [*pts.values(), *r39, r38]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(allpts)
    print(f"[r48_score] wrote {out_csv} ({len(allpts)} points)")

    # ---- matched deltas
    drows = []
    for host in ("se", "wan"):
        for arm, p in arms[host].items():
            for x in X:
                for refname, curve, kind in ([(ref[host][0], ref[host][1], "matched")] +
                                             [(n, c, "unmatched_nfe") for n, c in ctx[host]]):
                    r, flag = interp_at(curve, float(p[Y]), x)
                    drows.append({"host": host, "arm": arm, "x_metric": x, "reference": refname, "kind": kind,
                                  Y: p[Y], "value": p[x], "ref_at_lpips": r, "delta": p[x] - r if not math.isnan(r) else math.nan,
                                  "flag": flag})
    out_d = csv_dir / f"r48_{mode}_deltas.csv"
    with out_d.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(drows[0]))
        w.writeheader()
        w.writerows(drows)
    print(f"[r48_score] wrote {out_d} ({len(drows)} rows)")

    # ---- figures
    for host in ("se", "wan"):
        curves = [(ref[host][0], ref[host][1], COLORS["match"], "-")] + [(n, c, COLORS["ref"], "-") for n, c in ctx[host]]
        for xkey in FIG_X:
            plot(host, xkey, list(arms[host].values()), curves, [r38] if host == "se" else [], fig_dir / f"{host}_{xkey}.pdf", mode)
    print(f"[r48_score] figures -> {fig_dir}")

    # ---- gate
    rng = np.random.default_rng(0)
    base_of = {(p["host"], p["label"]): b for b, p in pts.items() if p["role"] == "arm"}
    print("\n=== GATE (verdicts are for the user; write passing pairs to evaluation/r48_selected_arms.txt) ===")
    grows = []
    for host in ("se", "wan"):
        for arm in C4:
            p = arms[host][arm]
            D = {(r["x_metric"], r["kind"]): r for r in drows if r["host"] == host and r["arm"] == arm}
            g_curve_vals = {x: D[(x, "matched")] for x in GATE_X}
            usable = all(r["flag"] in ("ok", "nonmonotone") for r in g_curve_vals.values())
            g_curve = usable and all(r["delta"] > 0 for r in g_curve_vals.values())
            c2, c3 = arms[host]["dino_D"], arms[host]["S"]
            g_s = (p["five_acc"] >= c2["five_acc"]) and (p[Y] <= c2[Y]) if arm == "dino_SD" else None
            g_d = (p["five_acc"] >= c3["five_acc"]) and (p[Y] <= c3[Y])
            vb = vals[base_of[(host, arm)]]
            ci = {}
            for other, key in (("dino_D", "C2"), ("S", "C3")):
                vo = vals[base_of[(host, other)]]
                for m in ("five_acc", Y):
                    ci[(key, m)] = paired_ci(vb[m], vo[m], rng)
            print(f"\n[{host}] {arm}: bg LPIPS {p[Y]:.4f}, FiVE-Acc {p['five_acc']:.3f}, CLIP-D {p['clip_d_prompt']:.4f}")
            for x in GATE_X:
                r = g_curve_vals[x]
                extra = ""
                if r["flag"] == "outside_range":
                    nat = D.get((x, "unmatched_nfe"))
                    if nat is not None and not math.isnan(nat["delta"]):
                        extra = f"  [vs native curve (unmatched NFE): {nat['delta']:+.4f}]"
                dtxt = "n/a" if math.isnan(r["delta"]) else format(r["delta"], "+.4f")
                print(f"   delta {x:14s} vs {r['reference']}: {dtxt} ({r['flag']}){extra}")
            print(f"   G_curve (delta > 0 on both, matched ref usable): {'PASS' if g_curve else 'FAIL' if usable else 'NO-REF (arm outside the matched curve range)'}")
            if g_s is not None:
                print(f"   G_S (>= dino_D on FiVE-Acc, LPIPS <=): {'PASS' if g_s else 'FAIL'}   "
                      f"FiVE-Acc diff {ci[('C2', 'five_acc')][0]:+.3f} [{ci[('C2', 'five_acc')][1]:+.3f}, {ci[('C2', 'five_acc')][2]:+.3f}], "
                      f"LPIPS diff {ci[('C2', Y)][0]:+.4f} [{ci[('C2', Y)][1]:+.4f}, {ci[('C2', Y)][2]:+.4f}]")
            print(f"   G_D (>= S on FiVE-Acc, LPIPS <=): {'PASS' if g_d else 'FAIL'}   "
                  f"FiVE-Acc diff {ci[('C3', 'five_acc')][0]:+.3f} [{ci[('C3', 'five_acc')][1]:+.3f}, {ci[('C3', 'five_acc')][2]:+.3f}], "
                  f"LPIPS diff {ci[('C3', Y)][0]:+.4f} [{ci[('C3', Y)][1]:+.4f}, {ci[('C3', Y)][2]:+.4f}]")
            grows.append({"host": host, "arm": arm, "G_curve": "PASS" if g_curve else "FAIL" if usable else "NO-REF",
                          "G_S": "n/a" if g_s is None else "PASS" if g_s else "FAIL", "G_D": "PASS" if g_d else "FAIL"})
    out_g = csv_dir / f"r48_{mode}_gate.csv"
    with out_g.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grows[0]))
        w.writeheader()
        w.writerows(grows)
    print(f"\n[r48_score] wrote {out_g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
