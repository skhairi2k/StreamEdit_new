"""R50 jz-compare: does Jean Zay (H100) reproduce the lab's (L40S) Wan-Edit result on the 18 clips?

H100 will not give the lab's pixels (different kernels, bf16 accumulation order, torch build; a token near a rounding boundary can get
another start step, the VLM's yes/no can flip), so raw numbers are NOT compared. What the paper claims is a PAIRED DIFFERENCE, so that is
what is tested, for the arm of the full stage against the official baseline:

    delta(metric) = mean over the 18 clips of  [ wan_max_s5(clip) - wan_native_s0.92(clip) ]      (adaptive minus official Wan-Edit)

  PASS  <=>  for FiVE-Acc, CLIP-D, background LPIPS and background PSNR, the Jean Zay delta lies inside the 95% bootstrap CI of the
             lab delta (same units, same clips, same bootstrap as r49_score.py: r48_score.unit_values / paired_ci, 2000 resamples, seed 0).

Also reported (information, not part of the verdict): the platform shift of each delta (Jean Zay minus lab, paired over clips, with a CI),
the platform shift of each arm, the 3 clips that moved most, and a warning for every metric whose lab effect is not distinguishable
from 0 (inside-the-CI is then a weak check). Recomputed lab deltas are cross-checked against the published r49_deltas.csv.

Inputs : --jz  DIR  Jean Zay CSVs  (default evaluation/csv): edit{T}_FiVE_r49_{wan_max_s5,wan_native_s0p92}_frame_stride8.csv, the r49_fiveacc_*
                    twins and r49_clip_directional_wan_pair.csv (r49_eval_all.sh wan_pair)
         --ref  DIR  the lab's CSVs, read-only (default $JZ_WORK/r50_ref/evaluation_csv): the same stems; CLIP-D in r49_clip_directional_{main,base}.csv
Output : evaluation/csv/r50_compare.csv (long format, column `kind` = delta | arm_shift | worst_clip) and the table on stdout
usage  : python evaluation/r50_compare.py [--jz DIR] [--ref DIR] [--cases evaluation/r49_cases.json] [--out FILE]
exit   : 0 PASS, 1 FAIL, 2 inputs missing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_score as R  # noqa: E402  (the lab's own loaders and bootstrap: the CI method cannot drift)

# arm -> (harness stem, fiveacc stem, lab CLIP-D file, Jean Zay CLIP-D file, CLIP-D method name)
ARMS: Dict[str, Tuple[str, str, str, str, str]] = {
    "wan_max_s5":       ("r49_wan_max_s5", "r49_fiveacc_wan_max_s5", "r49_clip_directional_main.csv", "r49_clip_directional_wan_pair.csv", "r49_wan_max_s5"),
    "wan_native_s0p92": ("r49_wan_native_s0p92", "r49_fiveacc_wan_native_s0p92", "r49_clip_directional_base.csv", "r49_clip_directional_wan_pair.csv", "r49_wan_native_s0p92"),
}
ADAPT, OFFICIAL = "wan_max_s5", "wan_native_s0p92"
# metric key, display name, sign of "better" (+1 higher is better)
METRICS: Tuple[Tuple[str, str, int], ...] = (("five_acc", "FiVE-Acc", +1), ("clip_d_prompt", "CLIP-D", +1),
                                             ("lpips_unedit_part", "bg LPIPS", -1), ("psnr_unedit_part", "bg PSNR", +1))
PUBLISHED_ARM, PUBLISHED_REF = "R49 max, s_min 5", "official: Wan-Edit"
Vals = Dict[str, Dict[Tuple[int, str], float]]


def default_ref() -> Optional[Path]:
    work = os.environ.get("JZ_WORK")
    if not work:
        cfg = Path(os.environ.get("R50_ENV", HERE.parent / "jz" / "r50_jz.env"))
        if cfg.is_file():
            m = re.search(r'^JZ_WORK="?([^"\n]+)"?', cfg.read_text(), re.M)
            work = m.group(1) if m else None
    return Path(work) / "r50_ref" / "evaluation_csv" if work else None


def load_arm(csv_dir: Path, arm: str, jz: bool, scope: set, names) -> Vals:
    hstem, fstem, clipd_lab, clipd_jz, method = ARMS[arm]
    return R.unit_values(csv_dir, hstem, fstem, csv_dir / (clipd_jz if jz else clipd_lab), method, scope, names)


def diff(a: Dict, b: Dict) -> Dict:
    return {k: a[k] - b[k] for k in a}


def fmt(x: float, m: str) -> str:
    return f"{x:+.4f}" if m != "psnr_unedit_part" else f"{x:+.3f}"


def published(ref: Path) -> Dict[str, Tuple[float, float, float]]:
    """metric -> (delta, ci_lo, ci_hi) of `R49 max, s_min 5` against the official baseline in the lab's r49_deltas.csv."""
    p = ref / "r49_deltas.csv"
    if not p.is_file():
        return {}
    return {r["metric"]: (float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in csv.DictReader(p.open())
            if r["host"] == "wan" and r["arm"].startswith(PUBLISHED_ARM) and r["reference"].startswith(PUBLISHED_REF) and r["kind"] == "paired"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jz", type=Path, default=HERE / "csv")
    ap.add_argument("--ref", type=Path, default=default_ref())
    ap.add_argument("--cases", type=Path, default=HERE / "r49_cases.json")
    ap.add_argument("--out", type=Path, default=HERE / "csv" / "r50_compare.csv")
    a = ap.parse_args(argv)
    if a.ref is None or not a.ref.is_dir():
        print(f"[r50_compare] FATAL: lab reference CSV folder not found ({a.ref}); pass --ref DIR or run jz/r50_setup_dirs.sh")
        return 2
    scope = {(int(c["edit_type"]), c["video_name"]) for c in json.loads(a.cases.read_text())}
    names = R.clip_names()
    try:
        lab = {arm: load_arm(a.ref, arm, False, scope, names) for arm in ARMS}
        jz = {arm: load_arm(a.jz, arm, True, scope, names) for arm in ARMS}
    except SystemExit as e:                                     # the lab's loaders exit with a message that names the missing file / clips
        print(f"[r50_compare] FATAL: {e}")
        return 2
    n = len(scope)
    rng = np.random.default_rng(0)
    print(f"[r50_compare] {n} clips | Jean Zay {a.jz} | lab reference {a.ref}")
    print(f"[r50_compare] paired delta  {ADAPT} - {OFFICIAL} (adaptive minus official Wan-Edit), mean over clips [95% bootstrap CI, {R.N_BOOT} resamples]\n")

    rows: List[Dict[str, object]] = []
    head = f"{'metric':<10} {'lab (L40S)':>30}   {'Jean Zay (H100)':>16}  {'inside':>6}  {'shift of the delta [95% CI]':>34}"
    print(head)
    print("-" * len(head))
    pub = published(a.ref)
    all_inside, notes, sanity = True, [], []
    for m, label, sgn in METRICS:
        dl = diff(lab[ADAPT][m], lab[OFFICIAL][m])
        dj = diff(jz[ADAPT][m], jz[OFFICIAL][m])
        mean_l, lo_l, hi_l = R.paired_ci(lab[ADAPT][m], lab[OFFICIAL][m], rng)
        mean_j, lo_j, hi_j = R.paired_ci(jz[ADAPT][m], jz[OFFICIAL][m], rng)
        sh, sh_lo, sh_hi = R.paired_ci(dj, dl, rng)
        inside = lo_l <= mean_j <= hi_l
        all_inside &= inside
        weak = lo_l <= 0.0 <= hi_l
        if weak:
            notes.append(f"{label}: the lab effect is not distinguishable from 0 (CI [{fmt(lo_l, m)}, {fmt(hi_l, m)}]), so 'inside' is a weak check here")
        if m in pub:      # (metric, |published - recomputed| for the mean, same for the CI bounds as a fraction of the published CI width)
            sanity.append((label, abs(pub[m][0] - mean_l), max(abs(pub[m][1] - lo_l), abs(pub[m][2] - hi_l)) / max(pub[m][2] - pub[m][1], 1e-12)))
        print(f"{label:<10} {fmt(mean_l, m):>9} [{fmt(lo_l, m)}, {fmt(hi_l, m)}] {fmt(mean_j, m):>16}  {'yes' if inside else 'NO':>6}  "
              f"{fmt(sh, m):>9} [{fmt(sh_lo, m)}, {fmt(sh_hi, m)}]{'  (CI excludes 0)' if not sh_lo <= 0 <= sh_hi else ''}")
        rows.append(dict(kind="delta", metric=m, arm=f"{ADAPT}-{OFFICIAL}", lab=mean_l, lab_lo=lo_l, lab_hi=hi_l, jz=mean_j, jz_lo=lo_j, jz_hi=hi_j,
                         shift=sh, shift_lo=sh_lo, shift_hi=sh_hi, verdict="inside" if inside else "OUTSIDE",
                         note="lab effect not distinguishable from 0" if weak else ""))

    print("\n[r50_compare] platform shift of each arm (Jean Zay minus lab, mean over the same clips) and the 3 clips that moved most")
    for arm in ARMS:
        for m, label, sgn in METRICS:
            vl, vj = lab[arm][m], jz[arm][m]
            mean_sh, lo, hi = R.paired_ci(vj, vl, rng)
            worst = sorted(vl, key=lambda k: -abs(vj[k] - vl[k]))[:3]
            wtxt = ", ".join(f"{k[0]}/{k[1]} {fmt(vj[k] - vl[k], m)}" for k in worst)
            print(f"  {arm:<17} {label:<9} lab {np.mean(list(vl.values())):.4f}  JZ {np.mean(list(vj.values())):.4f}  shift {fmt(mean_sh, m)} [{fmt(lo, m)}, {fmt(hi, m)}]   worst: {wtxt}")
            rows.append(dict(kind="arm_shift", metric=m, arm=arm, lab=float(np.mean(list(vl.values()))), jz=float(np.mean(list(vj.values()))),
                             shift=mean_sh, shift_lo=lo, shift_hi=hi, verdict="", note=""))
            for k in worst:
                rows.append(dict(kind="worst_clip", metric=m, arm=arm, lab=vl[k], jz=vj[k], shift=vj[k] - vl[k], verdict="", note=f"{k[0]}/{k[1]}"))

    if sanity:
        d_mean = max(s[1] for s in sanity)
        d_ci = max(s[2] for s in sanity)
        print(f"\n[r50_compare] cross-check against the published {a.ref.name}/r49_deltas.csv: recomputed lab means differ by at most {d_mean:.2e} "
              f"({'identical' if d_mean < 1e-9 else 'DIFFERENT -- the reference CSVs are not the ones behind the paper numbers'}); "
              f"CI bounds by at most {100 * d_ci:.1f}% of the CI width (bootstrap noise: r49_score.py shares one random stream across all its CIs)")
    else:
        print("\n[r50_compare] (no published r49_deltas.csv rows found in the reference folder: cross-check skipped)")
    for t in notes:
        print(f"[r50_compare] NOTE {t}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["kind", "metric", "arm", "lab", "lab_lo", "lab_hi", "jz", "jz_lo", "jz_hi", "shift", "shift_lo", "shift_hi", "verdict", "note"]
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"[r50_compare] wrote {a.out} ({len(rows)} rows)")
    print(f"[r50_compare] {'PASS' if all_inside else 'FAIL'}: Jean Zay delta inside the lab 95% CI for "
          f"{sum(1 for r in rows if r['kind'] == 'delta' and r['verdict'] == 'inside')} of {len(METRICS)} metrics")
    return 0 if all_inside else 1


if __name__ == "__main__":
    sys.exit(main())
